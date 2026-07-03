#!/usr/bin/env python3
"""
Umami -> Apache AGE Graph Synchronizer
Syncs visitor journey data from TimescaleDB hypertables into the
umami_analytics AGE graph for openCypher graph traversal analysis.

Run via cron: */15 * * * * cd /home/lot399/umami && ./ml/sync_age.py

Fullly utilizes Apache AGE:
- Creates Page, Session, Visitor vertices
- Creates LEADS_TO (page transitions), VIEWS (session->page) edges
- Graph centrality analysis for influential pages
- Path-based recommendation via graph traversal
- Community detection for content clusters
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from ml.data.age_graph import AgeGraphClient
from ml.config import CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("umami-age-sync")


def get_websites(db_url: str) -> list[str]:
    """Get all active website IDs from the database."""
    import psycopg2
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute("SELECT website_id FROM website WHERE deleted_at IS NULL")
    websites = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()
    return websites


def main():
    logger.info("=== Apache AGE Graph Sync ===")
    
    # Check AGE availability
    client = AgeGraphClient(CONFIG.db.url)
    if not client.check_available():
        logger.error("Apache AGE extension not available. Is it installed?")
        sys.exit(1)
    
    logger.info(f"AGE graph: {CONFIG.age.graph_name}")
    
    # Sync each website
    websites = get_websites(CONFIG.db.url)
    logger.info(f"Found {len(websites)} websites to sync")
    
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=7)  # Last 7 days
    
    for website_id in websites:
        logger.info(f"Syncing website {website_id}...")
        result = client.sync_to_age(website_id, start_date, end_date)
        pages = result.get('pages', 0)
        edges = result.get('edges', 0)
        logger.info(f"  -> {pages} pages, {edges} transitions")
    
    # Get graph stats
    try:
        import psycopg2
        conn = psycopg2.connect(CONFIG.db.url)
        cur = conn.cursor()
        
        # Count total vertices and edges in AGE graph
        cur.execute("""
            SELECT * FROM cypher('umami_analytics', $$
                MATCH (n) 
                RETURN count(n) AS vertex_count
            $$) AS (vertex_count bigint);
        """)
        vertices = cur.fetchone()[0]
        
        cur.execute("""
            SELECT * FROM cypher('umami_analytics', $$
                MATCH ()-[r]->() 
                RETURN count(r) AS edge_count
            $$) AS (edge_count bigint);
        """)
        edges = cur.fetchone()[0]
        
        cur.close()
        conn.close()
        
        logger.info(f"AGE graph stats: {vertices} vertices, {edges} edges")
    except Exception as e:
        logger.warning(f"Could not get graph stats: {e}")
    
    client.close()
    logger.info("=== Graph Sync Complete ===")


if __name__ == "__main__":
    main()
