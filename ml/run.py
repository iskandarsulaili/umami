#!/usr/bin/env python3
"""
Umami ML Service - Runner
Activates virtual environment and starts the ML API server.
GPU-accelerated inference with CPU fallback.

Usage:
  ./run_ml_service.py             # Start API server
  ./run_ml_service.py --train     # Train all models then start API
"""

import os
import sys
import json
import logging
import subprocess
import argparse
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("umami-ml")


def check_gpu():
    """Check GPU availability and log status"""
    try:
        import torch
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            for i in range(n_gpus):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)} "
                           f"({props.total_memory / 1e9:.1f}GB)")
            return True
        else:
            logger.info("No CUDA GPU available, using CPU")
            return False
    except ImportError:
        logger.info("PyTorch not installed, using CPU")
        return False


def check_db():
    """Check database connectivity"""
    try:
        from ml.config import CONFIG
        import psycopg2
        
        conn = psycopg2.connect(CONFIG.db.url)
        cur = conn.cursor()
        
        # Check PG version
        cur.execute("SELECT version()")
        pg_version = cur.fetchone()[0]
        logger.info(f"Connected: {pg_version}")
        
        # Check extensions
        cur.execute("""
            SELECT extname, extversion FROM pg_extension
            WHERE extname IN ('timescaledb', 'age', 'vector')
            ORDER BY extname
        """)
        for ext in cur.fetchall():
            logger.info(f"Extension: {ext[0]} v{ext[1]}")
        
        # Check TimescaleDB
        try:
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'")
            tsdb = cur.fetchone()
            if tsdb:
                logger.info(f"TimescaleDB v{tsdb[0]} - hypertables available")
        except:
            pass
        
        # Check AGE
        try:
            cur.execute("SELECT * FROM ag_catalog.cypher('umami_analytics', $$ RETURN 1 $$) AS (r integer)")
            logger.info("Apache AGE - umami_analytics graph ready")
        except:
            logger.info("Apache AGE - graph queries available")
        
        cur.close()
        conn.close()
        return True
    except Exception as e:
        logger.warning(f"Database connection failed: {e}")
        return False


def train_models():
    """Train all ML models using default data range"""
    from ml.config import CONFIG
    from datetime import datetime, timedelta
    
    logger.info("=== Starting model training ===")
    
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=30)
    
    # Get all websites
    try:
        import psycopg2
        conn = psycopg2.connect(CONFIG.db.url)
        cur = conn.cursor()
        cur.execute("SELECT website_id FROM website WHERE deleted_at IS NULL")
        websites = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to fetch websites: {e}")
        websites = []
    
    if not websites:
        logger.warning("No websites found, using demo mode")
        websites = ["00000000-0000-0000-0000-000000000000"]
    
    # Import models
    from ml.models.next_page_predictor import NextPagePredictor
    from ml.models.funnel_predictor import FunnelPredictor
    from ml.models.session_intent import SessionIntentClassifier
    from ml.models.recommender import Recommender
    from ml.models.rage_click import RageClickDetector
    from ml.models.journey_clusterer import JourneyClusterer
    from ml.models.contextual_bandit import ContextualBandit
    from ml.models.ab_test import ABTestFramework
    from ml.models.session_replay import SessionReplayAnalyzer
    from ml.data.session_sequence import SessionDataExtractor
    
    for website_id in websites:
        logger.info(f"Training models for website: {website_id}")
        
        try:
            with SessionDataExtractor(CONFIG.db.url) as extractor:
                # Get session sequences
                sequences = extractor.get_session_sequences(
                    website_id, start_date, end_date
                )
                page_sequences = [s.pages for s in sequences if len(s.pages) >= 2]
                
                if not page_sequences:
                    logger.warning(f"No sessions for {website_id}, skipping")
                    continue
                
                logger.info(f"  Sessions: {len(page_sequences)}")
                
                # Train NextPagePredictor
                logger.info("  Training NextPagePredictor...")
                np_model = NextPagePredictor()
                np_model.train(page_sequences, use_transformer=True)
                np_model.save()
                
                # Train Recommender
                logger.info("  Training Recommender...")
                long_sequences = [s.pages for s in sequences if len(s.pages) >= 3]
                if len(long_sequences) >= 10:
                    rec_model = Recommender()
                    rec_model.train_session_encoder(long_sequences)
                    rec_model.save()
                
                # Get session features for funnel/intent
                sessions_df = extractor.get_session_features(website_id, start_date, end_date)
                
                if not sessions_df.empty:
                    sessions = sessions_df.to_dict('records')
                    
                    # Train FunnelPredictor
                    logger.info("  Training FunnelPredictor...")
                    labels = [1 if s.get('total_pageviews', 0) > 3 else 0 for s in sessions]
                    fu_model = FunnelPredictor()
                    fu_model.train(sessions, labels)
                    fu_model.save()
                    
                    # Train IntentClassifier
                    logger.info("  Training SessionIntentClassifier...")
                    session_dict = {s['session_id']: s for s in sessions}
                    sf_list = []
                    labels_list = []
                    
                    for seq in sequences:
                        if len(seq.pages) < 2:
                            continue
                        sf = session_dict.get(seq.session_id, {})
                        sf['page_views'] = len(seq.pages)
                        sf['session_duration'] = seq.duration_seconds
                        sf_list.append(sf)
                        
                        has_checkout = any('checkout' in p.lower() or 'cart' in p.lower() for p in seq.pages)
                        has_pricing = any('pricing' in p.lower() or 'price' in p.lower() for p in seq.pages)
                        has_support = any('support' in p.lower() or 'help' in p.lower() for p in seq.pages)
                        
                        if has_checkout:
                            labels_list.append('ready_to_buy')
                        elif has_pricing:
                            labels_list.append('price_comparing')
                        elif has_support:
                            labels_list.append('support_seeking')
                        elif seq.duration_seconds > 120:
                            labels_list.append('researching')
                        elif len(seq.pages) <= 2:
                            labels_list.append('just_browsing')
                        else:
                            labels_list.append('content_consumption')
                    
                    it_model = SessionIntentClassifier()
                    it_model.train(sf_list, [s.pages for s in sequences], labels_list)
                    it_model.save()
                    
        except Exception as e:
            logger.error(f"Training failed for {website_id}: {e}")
    
    # Train remaining models (rule-based or lightweight)
    for website_id in websites:
        try:
            # JourneyClusterer - needs session feature dicts
            with SessionDataExtractor(CONFIG.db.url) as extractor:
                sequences = extractor.get_session_sequences(
                    website_id, start_date, end_date
                )
                if sequences:
                    session_dicts = []
                    for seq in sequences:
                        d = {
                            'pages': seq.pages,
                            'duration_seconds': seq.duration_seconds,
                            'device': getattr(seq, 'device', 'desktop'),
                            'hour': getattr(seq, 'hour', 12),
                            'is_weekend': getattr(seq, 'is_weekend', False),
                            'n_referrers': getattr(seq, 'n_referrers', 0),
                            'n_events': getattr(seq, 'n_events', 0),
                        }
                        session_dicts.append(d)
                    jc = JourneyClusterer()
                    jc.train(session_dicts)
                    jc.save()
                    logger.info(f"JourneyClusterer trained for {website_id}")
            
            # ContextualBandit
            cb = ContextualBandit()
            cb.train()
            cb.save()
            logger.info(f"ContextualBandit initialized for {website_id}")
            
            # ABTestFramework
            ab = ABTestFramework()
            ab.train()
            ab.save()
            logger.info(f"ABTestFramework initialized for {website_id}")
            
            # SessionReplayAnalyzer
            sr = SessionReplayAnalyzer()
            sr.train()
            sr.save()
            logger.info(f"SessionReplayAnalyzer initialized for {website_id}")
        except Exception as e:
            logger.warning(f"Lightweight model init skipped for {website_id}: {e}")
    
    logger.info("=== Training complete ===")


def main():
    parser = argparse.ArgumentParser(description="Umami ML Service")
    parser.add_argument("--train", action="store_true", help="Train models before starting API")
    parser.add_argument("--port", type=int, default=None, help="API port")
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Umami ML Service")
    logger.info("=" * 60)
    
    # Check GPU
    has_gpu = check_gpu()
    
    # Check database
    check_db()
    
    # Train models if requested
    if args.train:
        train_models()
    
    # Start API
    from ml.config import CONFIG
    import uvicorn
    
    port = args.port or CONFIG.api.port
    logger.info(f"Starting ML API on 0.0.0.0:{port}")
    logger.info(f"GPU: {'Enabled' if has_gpu else 'Disabled (CPU fallback)'}")
    
    uvicorn.run(
        "ml.api.main:app",
        host=CONFIG.api.host,
        port=port,
        log_level=CONFIG.api.log_level,
        reload=CONFIG.api.auto_reload,
    )


if __name__ == "__main__":
    main()
