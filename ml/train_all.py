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
from ml.models.rage_click import RageClickDetector
from ml.models.journey_clusterer import JourneyClusterer
from ml.models.contextual_bandit import ContextualBandit
from ml.models.ab_test import ABTestFramework
from ml.models.session_replay import SessionReplayAnalyzer
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
    """Train all models for a single website with rich features."""
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
        
        # Train FunnelPredictor (XGBoost with GPU)
        try:
            with SessionDataExtractor(CONFIG.db.url) as ex:
                sessions_df = ex.get_session_features(website_id, start, end)
                if not sessions_df.empty:
                    sessions = sessions_df.to_dict('records')
                    # Use pageview count + performance issues as label
                    labels = [1 if s.get('total_pageviews', 0) > 3 else 0 for s in sessions]
                    fu = FunnelPredictor()
                    fu.train(sessions, labels)
                    fu.save()
                    logger.info(f"  FunnelPredictor: {len(sessions)} sessions, "
                               f"acc={fu.metadata.get('train_accuracy', 0):.3f}")
        except Exception as e:
            logger.warning(f"  FunnelPredictor skipped: {e}")
        
        # Train SessionIntentClassifier (with richer features)
        try:
            with SessionDataExtractor(CONFIG.db.url) as ex:
                sessions_df = ex.get_session_features(website_id, start, end)
                event_types = ex.get_event_types_summary(website_id, start, end)
                session_dict = {s['session_id']: s for s in sessions_df.to_dict('records')}
                sf_list, pl_list, labels_list = [], [], []
                for seq in sequences:
                    if len(seq.pages) < 2:
                        continue
                    sf = session_dict.get(seq.session_id, {})
                    sf['page_views'] = len(seq.pages)
                    sf['session_duration'] = seq.duration_seconds
                    sf['avg_time_on_page'] = seq.duration_seconds / max(1, len(seq.pages))
                    sf['avg_lcp'] = seq.avg_lcp
                    sf['avg_cls'] = seq.avg_cls
                    sf['avg_inp'] = seq.avg_inp
                    sf['unique_pages'] = seq.unique_pages
                    sf['page_depth'] = seq.page_depth
                    
                    # Add custom event counts from event_types summary
                    visit_events = event_types.get(seq.visit_id, {})
                    for event_name, count in visit_events.items():
                        sf[f'event_{event_name}'] = count
                    
                    sf_list.append(sf)
                    pl_list.append(seq.pages)
                    
                    # Improved label assignment using page patterns + events
                    has_checkout = seq.has_checkout
                    has_pricing = seq.has_pricing
                    has_support = seq.has_support
                    has_search = seq.has_search
                    has_custom_event = 'purchase' in visit_events or 'signup' in visit_events or 'order' in visit_events
                    
                    if has_custom_event or has_checkout:
                        labels_list.append('ready_to_buy')
                    elif has_pricing:
                        labels_list.append('price_comparing')
                    elif has_support:
                        labels_list.append('support_seeking')
                    elif seq.duration_seconds > 180:
                        labels_list.append('researching')
                    elif has_search:
                        labels_list.append('researching')
                    elif seq.length <= 2 and seq.duration_seconds < 30:
                        labels_list.append('just_browsing')
                    elif any('/blog' in p.lower() or '/article' in p.lower() for p in seq.pages):
                        labels_list.append('content_consumption')
                    else:
                        labels_list.append('content_consumption')
                
                unique_labels = set(labels_list)
                if len(unique_labels) >= 2:
                    it = SessionIntentClassifier()
                    it.train(sf_list, pl_list, labels_list)
                    it.save()
                    logger.info(f"  SessionIntentClassifier: {len(sf_list)} sessions, "
                               f"acc={it.metadata.get('train_accuracy', 0):.3f}, "
                               f"labels={len(unique_labels)}")
        except Exception as e:
            logger.warning(f"  SessionIntentClassifier skipped: {e}")
        
        # Train JourneyClusterer (with richer session features)
        try:
            session_dicts = []
            for seq in sequences:
                d = seq.to_feature_dict()
                d['device'] = 'desktop'
                d['hour'] = 12
                d['is_weekend'] = False
                # Extract hour from first timestamp if available
                if seq.timestamps:
                    d['hour'] = seq.timestamps[0].hour
                    d['is_weekend'] = seq.timestamps[0].weekday() >= 5
                session_dicts.append(d)
            jc = JourneyClusterer()
            jc.train(session_dicts)
            jc.save()
            logger.info(f"  JourneyClusterer: {jc.metadata.get('n_clusters', '?')} clusters "
                       f"from {len(session_dicts)} sessions")
        except Exception as e:
            logger.warning(f"  JourneyClusterer skipped: {e}")
        
        # Train RageClickDetector
        try:
            import psycopg2
            import psycopg2.extras
            conn = psycopg2.connect(CONFIG.db.url)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            cur.execute("""
                SELECT session_id, visit_id,
                       array_agg(ROW(x, y, page_x, page_y, scroll_pct,
                               viewport_w, viewport_h, page_h, created_at)
                               ORDER BY created_at) AS clicks
                FROM heatmap_event
                WHERE website_id = %s
                  AND created_at BETWEEN %s AND %s
                GROUP BY session_id, visit_id
                HAVING COUNT(*) >= 2
                LIMIT 5000
            """, (website_id, start, end))
            from ml.models.rage_click import RageClickDetector
            rc = RageClickDetector()
            click_sessions = []
            for row in cur.fetchall():
                clicks_raw = row['clicks']
                clicks = [{'x': c[0], 'y': c[1], 'page_x': c[2], 'page_y': c[3],
                           'scroll_pct': c[4], 'viewport_w': c[5], 'viewport_h': c[6],
                           'page_h': c[7], 'created_at': c[8]} for c in clicks_raw]
                features = rc.extract_click_features(clicks)
                if features:
                    click_sessions.append(features)
            cur.close()
            conn.close()
            rc.train(click_sessions)
            rc.save()
            logger.info(f"  RageClickDetector: {len(click_sessions)} sessions")
        except Exception as e:
            logger.warning(f"  RageClickDetector skipped: {e}")
        
        # Train ContextualBandit (LinUCB)
        try:
            cb = ContextualBandit()
            cb.train()
            cb.save()
            logger.info("  ContextualBandit: initialized")
        except Exception as e:
            logger.warning(f"  ContextualBandit skipped: {e}")
        
        # Train ABTestFramework (rule-based, no training needed)
        try:
            ab = ABTestFramework()
            ab.train()
            ab.save()
            logger.info("  ABTestFramework: initialized")
        except Exception as e:
            logger.warning(f"  ABTestFramework skipped: {e}")
        
        # Train SessionReplayAnalyzer (rule-based, no training needed)
        try:
            sr = SessionReplayAnalyzer()
            sr.train()
            sr.save()
            logger.info("  SessionReplayAnalyzer: initialized")
        except Exception as e:
            logger.warning(f"  SessionReplayAnalyzer skipped: {e}")


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
