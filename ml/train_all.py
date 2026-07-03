#!/usr/bin/env python3
"""
Umami ML Model Training Scheduler
Periodically retrains all ML models from fresh session data.
Runs daily via cron to keep recommendations up to date.

Run: python3 ml/train_all.py [--website <id>] [--days 30]
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from ml.config import CONFIG
from ml.data.session_sequence import SessionDataExtractor
from ml.models.next_page_predictor import NextPagePredictor
from ml.models.funnel_predictor import FunnelPredictor
from ml.models.session_intent import SessionIntentClassifier
from ml.models.recommender import Recommender
from ml.data.age_graph import AgeGraphClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("umami-train")


def get_websites(db_url: str) -> list[str]:
    import psycopg2
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute("SELECT website_id FROM website WHERE deleted_at IS NULL")
    sites = [r[0] for r in cur.fetchall()]
    cur.close(); conn.close()
    return sites


def train_for_website(website_id: str, days: int):
    """Train all models for a single website."""
    end = datetime.utcnow()
    start = end - timedelta(days=days)
    
    logger.info(f"Training for website {website_id} ({days} days)...")
    
    with SessionDataExtractor(CONFIG.db.url) as extractor:
        sequences = extractor.get_session_sequences(website_id, start, end)
        page_sequences = [s.pages for s in sequences if len(s.pages) >= 2]
        
        if not page_sequences:
            logger.warning(f"No sessions for {website_id}, skipping")
            return
        
        logger.info(f"  Sessions: {len(page_sequences)}")
        
        # Train NextPagePredictor (uses Markov chain + GPU Transformer)
        np = NextPagePredictor()
        np.train(page_sequences, use_transformer=True, epochs=20)
        np.save()
        logger.info(f"  NextPagePredictor: {np.vocab_size} pages")
        
        # Train Recommender (uses GRU + pgvector)
        long_seqs = [s.pages for s in sequences if len(s.pages) >= 3]
        if len(long_seqs) >= 5:
            rec = Recommender()
            rec.train_session_encoder(long_seqs, epochs=15)
            rec._build_page_embeddings(website_id)  # Sync to pgvector too
            rec.save()
            logger.info(f"  Recommender: {len(rec.page_embeddings)} embeddings")
        
        # Sync to AGE
        try:
            age = AgeGraphClient(CONFIG.db.url)
            if age.check_available():
                age.sync_to_age(website_id, start, end)
                age.close()
        except Exception as e:
            logger.warning(f"  AGE sync skipped: {e}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train all ML models")
    parser.add_argument("--website", type=str, default=None)
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    
    logger.info("=== ML Model Training ===")
    
    websites = [args.website] if args.website else get_websites(CONFIG.db.url)
    logger.info(f"Training for {len(websites)} website(s)")
    
    for wid in websites:
        try:
            train_for_website(wid, args.days)
        except Exception as e:
            logger.error(f"Failed on {wid}: {e}")
    
    logger.info("=== Training Complete ===")


if __name__ == "__main__":
    main()
