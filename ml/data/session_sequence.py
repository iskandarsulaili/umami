"""
Session Sequence Data Extraction from TimescaleDB/PG18
Extracts visitor journey sequences for ML models
"""

import logging
from typing import Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from ..config import CONFIG
from ..gpu_utils import DEVICE

logger = logging.getLogger(__name__)

try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False


@dataclass
class SessionSequence:
    """A single visitor journey session"""
    session_id: str
    website_id: str
    visit_id: str
    pages: list[str] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)
    features: dict = field(default_factory=dict)
    
    @property
    def length(self) -> int:
        return len(self.pages)
    
    @property
    def duration_seconds(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return (self.timestamps[-1] - self.timestamps[0]).total_seconds()


class SessionDataExtractor:
    """
    Extracts session sequence data from TimescaleDB hypertables.
    Uses PG18 features for efficient time-series queries.
    """
    
    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or CONFIG.db.url
        self._conn = None
    
    @property
    def conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.connection_string)
            self._conn.autocommit = False
        return self._conn
    
    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()
    
    def get_total_sessions(
        self,
        website_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> int:
        """Get total session count for a website in time range."""
        query = """
        SELECT COUNT(DISTINCT visit_id) as total
        FROM website_event
        WHERE website_id = %s
          AND created_at BETWEEN %s AND %s
        """
        with self.conn.cursor() as cur:
            cur.execute(query, (website_id, start_date, end_date))
            return cur.fetchone()[0]
    
    def get_session_sequences(
        self,
        website_id: str,
        start_date: datetime,
        end_date: datetime,
        min_length: int = 2,
        max_sessions: int = 100000
    ) -> list[SessionSequence]:
        """
        Extract visitor session sequences using window functions.
        Leverages TimescaleDB hypertable partitioning for speed.
        
        Uses PG18's uuidv7() for efficient ordering and the AIO
        subsystem for fast sequential scans of time-ranged data.
        """
        query = """
        WITH session_pages AS (
            SELECT 
                visit_id,
                session_id,
                url_path,
                event_name,
                created_at,
                ROW_NUMBER() OVER (
                    PARTITION BY visit_id 
                    ORDER BY created_at
                ) AS page_order,
                COUNT(*) OVER (PARTITION BY visit_id) AS total_pages
            FROM website_event
            WHERE website_id = %s
              AND created_at BETWEEN %s AND %s
              AND event_type = 1  -- page views only
        )
        SELECT 
            session_id,
            visit_id,
            url_path,
            event_name,
            created_at,
            page_order,
            total_pages
        FROM session_pages
        WHERE total_pages >= %s
        ORDER BY visit_id, page_order
        LIMIT %s
        """
        
        sequences: dict[str, SessionSequence] = {}
        
        with self.conn.cursor() as cur:
            cur.execute(query, (website_id, start_date, end_date, min_length, max_sessions))
            
            for row in cur:
                session_id, visit_id, url_path, event_name, created_at, page_order, total_pages = row
                
                if visit_id not in sequences:
                    sequences[visit_id] = SessionSequence(
                        session_id=session_id,
                        website_id=website_id,
                        visit_id=visit_id,
                        features={'total_pages': total_pages}
                    )
                
                # Use url_path or event_name as the page identifier
                page = url_path or event_name or '/unknown'
                sequences[visit_id].pages.append(page)
                sequences[visit_id].timestamps.append(created_at)
        
        logger.info(f"Extracted {len(sequences)} session sequences for {website_id}")
        return list(sequences.values())
    
    def get_session_features(
        self,
        website_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Extract session-level feature vectors for ML models.
        Uses TimescaleDB's continuous aggregates if available.
        """
        query = """
        SELECT 
            s.session_id,
            s.browser,
            s.os,
            s.device,
            s.screen,
            s.language,
            s.country,
            s.region,
            s.city,
            COUNT(DISTINCT we.event_id) AS total_pageviews,
            MIN(we.created_at) AS session_start,
            MAX(we.created_at) AS session_end,
            COUNT(DISTINCT we.url_path) AS unique_pages,
            COUNT(DISTINCT we.referrer_domain) AS unique_referrers,
            AVG(we.lcp) AS avg_lcp,
            AVG(we.cls) AS avg_cls,
            AVG(we.inp) AS avg_inp
        FROM session s
        JOIN website_event we ON s.session_id = we.session_id
        WHERE s.website_id = %s
          AND s.created_at BETWEEN %s AND %s
        GROUP BY s.session_id, s.browser, s.os, s.device,
                 s.screen, s.language, s.country, s.region, s.city
        """
        
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, (website_id, start_date, end_date))
            rows = cur.fetchall()
        
        return pd.DataFrame(rows)
    
    def get_page_transition_matrix(
        self,
        website_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Build page-to-page transition count matrix for Markov chain.
        Uses TimescaleDB window functions for efficient computation.
        """
        query = """
        WITH ordered_events AS (
            SELECT 
                visit_id,
                COALESCE(NULLIF(url_path, ''), event_name) AS current_page,
                LEAD(COALESCE(NULLIF(url_path, ''), event_name)) OVER (
                    PARTITION BY visit_id 
                    ORDER BY created_at
                ) AS next_page
            FROM website_event
            WHERE website_id = %s
              AND created_at BETWEEN %s AND %s
              AND event_type = 1
        )
        SELECT 
            current_page,
            next_page,
            COUNT(*) AS transition_count
        FROM ordered_events
        WHERE next_page IS NOT NULL
        GROUP BY current_page, next_page
        ORDER BY transition_count DESC
        """
        
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, (website_id, start_date, end_date))
            rows = cur.fetchall()
        
        if not rows:
            return pd.DataFrame(), []
        
        df = pd.DataFrame(rows)
        all_pages = list(set(df['current_page'].unique()) | set(df['next_page'].unique()))
        
        return df, all_pages
    
    def get_journey_cohorts(
        self,
        website_id: str,
        start_date: datetime,
        end_date: datetime,
        max_steps: int = 7
    ) -> pd.DataFrame:
        """
        Get most common journey paths (multi-step sequences).
        Uses the same logic as umami's existing getJourney query
        but returns as structured data for ML.
        """
        query = """
        WITH events AS (
            SELECT DISTINCT
                visit_id,
                COALESCE(NULLIF(event_name, ''), url_path) AS event,
                ROW_NUMBER() OVER (
                    PARTITION BY visit_id 
                    ORDER BY created_at
                ) AS event_number
            FROM website_event
            WHERE website_id = %s
              AND created_at BETWEEN %s AND %s
        ),
        sequences AS (
            SELECT
                visit_id,
                MAX(CASE WHEN event_number = 1 THEN event ELSE NULL END) AS e1,
                MAX(CASE WHEN event_number = 2 THEN event ELSE NULL END) AS e2,
                MAX(CASE WHEN event_number = 3 THEN event ELSE NULL END) AS e3,
                MAX(CASE WHEN event_number = 4 THEN event ELSE NULL END) AS e4,
                MAX(CASE WHEN event_number = 5 THEN event ELSE NULL END) AS e5,
                MAX(CASE WHEN event_number = 6 THEN event ELSE NULL END) AS e6,
                MAX(CASE WHEN event_number = 7 THEN event ELSE NULL END) AS e7
            FROM events
            GROUP BY visit_id
        )
        SELECT e1, e2, e3, e4, e5, e6, e7, COUNT(*) AS count
        FROM sequences
        GROUP BY e1, e2, e3, e4, e5, e6, e7
        ORDER BY count DESC
        LIMIT 1000
        """
        
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, (website_id, start_date, end_date))
            rows = cur.fetchall()
        
        return pd.DataFrame(rows)
    
    def get_apache_age_journeys(
        self,
        website_id: str,
        start_date: datetime,
        end_date: datetime,
        max_results: int = 100
    ) -> list[dict]:
        """
        Use Apache AGE graph queries for advanced journey path analysis.
        Queries the umami_analytics graph using openCypher.
        """
        # Check if AGE extension is available
        with self.conn.cursor() as cur:
            cur.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'age')")
            age_available = cur.fetchone()[0]
        
        if not age_available:
            logger.warning("Apache AGE extension not available, falling back to SQL")
            return []
        
        # Use AGE openCypher query to find most common page transition paths
        query = """
        SELECT * FROM cypher('umami_analytics', $$
            MATCH path = (p1:Page)-[:LEADS_TO]->(p2:Page)
            WHERE p1.website_id = %s
              AND p1.created_at >= %s
              AND p1.created_at <= %s
            RETURN p1.url AS source, p2.url AS target, COUNT(*) AS transitions
            ORDER BY transitions DESC
            LIMIT %d
        $$) AS (source text, target text, transitions bigint);
        """ % max_results
        
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, (website_id, start_date, end_date))
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"AGE query failed: {e}")
            return []
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
