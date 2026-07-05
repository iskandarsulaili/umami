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
    """A single visitor journey session with rich features"""
    session_id: str
    website_id: str
    visit_id: str
    pages: list[str] = field(default_factory=list)
    timestamps: list[datetime] = field(default_factory=list)
    event_names: list[str] = field(default_factory=list)
    referrers: list[str] = field(default_factory=list)
    lcp_values: list[float] = field(default_factory=list)
    cls_values: list[float] = field(default_factory=list)
    inp_values: list[float] = field(default_factory=list)
    features: dict = field(default_factory=dict)
    
    @property
    def length(self) -> int:
        return len(self.pages)
    
    @property
    def duration_seconds(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return (self.timestamps[-1] - self.timestamps[0]).total_seconds()
    
    @property
    def unique_pages(self) -> int:
        return len(set(self.pages))
    
    @property
    def avg_lcp(self) -> float:
        vals = [v for v in self.lcp_values if v and v > 0]
        return float(np.mean(vals)) if vals else 0.0
    
    @property
    def avg_cls(self) -> float:
        vals = [v for v in self.cls_values if v is not None]
        return float(np.mean(vals)) if vals else 0.0
    
    @property
    def avg_inp(self) -> float:
        vals = [v for v in self.inp_values if v and v > 0]
        return float(np.mean(vals)) if vals else 0.0
    
    @property
    def unique_referrers(self) -> int:
        return len(set(r for r in self.referrers if r))
    
    @property
    def has_checkout(self) -> bool:
        return any('checkout' in p.lower() or '/cart' in p.lower() for p in self.pages)
    
    @property
    def has_pricing(self) -> bool:
        return any('pricing' in p.lower() or '/price' in p.lower() for p in self.pages)
    
    @property
    def has_blog(self) -> bool:
        return any('/blog' in p.lower() or '/article' in p.lower() for p in self.pages)
    
    @property
    def has_support(self) -> bool:
        return any(p.lower() in ('/support', '/faq', '/contact', '/help') for p in self.pages)
    
    @property
    def has_search(self) -> bool:
        return any('/search' in p.lower() for p in self.pages)
    
    @property
    def page_depth(self) -> int:
        max_depth = 0
        for p in self.pages:
            depth = len(p.rstrip('/').split('/'))
            if depth > max_depth:
                max_depth = depth
        return max_depth
    
    def to_feature_dict(self) -> dict:
        """Convert to flat feature dict for ML models"""
        return {
            'n_pages': self.length,
            'unique_pages': self.unique_pages,
            'session_duration_seconds': self.duration_seconds,
            'avg_time_on_page': self.duration_seconds / max(1, self.length),
            'avg_lcp': self.avg_lcp,
            'avg_cls': self.avg_cls,
            'avg_inp': self.avg_inp,
            'unique_referrers': self.unique_referrers,
            'has_checkout': 1 if self.has_checkout else 0,
            'has_pricing': 1 if self.has_pricing else 0,
            'has_blog': 1 if self.has_blog else 0,
            'has_support': 1 if self.has_support else 0,
            'has_search': 1 if self.has_search else 0,
            'page_depth': self.page_depth,
            'bounce': 1 if self.length <= 1 else 0,
            **self.features,
        }


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
        Extract visitor session sequences with rich features.
        Leverages TimescaleDB hypertable partitioning for speed.
        Includes: page paths, event names, web vitals, referrers
        """
        query = """
        WITH session_pages AS (
            SELECT 
                visit_id,
                session_id,
                url_path,
                event_name,
                event_type,
                referrer_domain,
                created_at,
                lcp,
                cls,
                inp,
                ROW_NUMBER() OVER (
                    PARTITION BY visit_id 
                    ORDER BY created_at
                ) AS page_order,
                COUNT(*) OVER (PARTITION BY visit_id) AS total_pages
            FROM website_event
            WHERE website_id = %s
              AND created_at BETWEEN %s AND %s
        )
        SELECT 
            session_id,
            visit_id,
            url_path,
            event_name,
            event_type,
            referrer_domain,
            created_at,
            lcp,
            cls,
            inp,
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
                session_id, visit_id, url_path, event_name, event_type, \
                    referrer_domain, created_at, lcp, cls, inp, \
                    page_order, total_pages = row
                
                if visit_id not in sequences:
                    sequences[visit_id] = SessionSequence(
                        session_id=session_id,
                        website_id=website_id,
                        visit_id=visit_id,
                        features={'total_pages': total_pages}
                    )
                
                seq = sequences[visit_id]
                
                # Use url_path or event_name as the page identifier
                page = url_path or event_name or '/unknown'
                seq.pages.append(page)
                seq.timestamps.append(created_at)
                seq.event_names.append(event_name or '')
                seq.referrers.append(referrer_domain or '')
                
                if lcp is not None:
                    seq.lcp_values.append(float(lcp))
                if cls is not None:
                    seq.cls_values.append(float(cls))
                if inp is not None:
                    seq.inp_values.append(float(inp))
        
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
        Includes session metadata, web vitals, UTM params, referrers.
        """
        query = """
        SELECT 
            s.session_id,
            we.visit_id,
            s.browser,
            s.os,
            s.device,
            s.screen,
            s.language,
            s.country,
            s.region,
            s.city,
            MAX(we.referrer_domain) AS referrer_domain,
            MAX(we.utm_source) AS utm_source,
            MAX(we.utm_medium) AS utm_medium,
            MAX(we.utm_campaign) AS utm_campaign,
            MAX(we.utm_term) AS utm_term,
            MAX(we.utm_content) AS utm_content,
            EXTRACT(HOUR FROM s.created_at) AS hour_of_day,
            EXTRACT(DOW FROM s.created_at) AS day_of_week,
            COUNT(DISTINCT we.event_id) AS total_pageviews,
            MIN(we.created_at) AS session_start,
            MAX(we.created_at) AS session_end,
            COUNT(DISTINCT we.url_path) AS unique_pages,
            COUNT(DISTINCT we.referrer_domain) AS unique_referrers,
            AVG(we.lcp) AS avg_lcp,
            AVG(we.cls) AS avg_cls,
            AVG(we.inp) AS avg_inp,
            BOOL_OR(we.lcp > 4000 OR we.cls > 0.25 OR we.inp > 500) AS has_performance_issues
        FROM session s
        JOIN website_event we ON s.session_id = we.session_id
        WHERE s.website_id = %s
          AND s.created_at BETWEEN %s AND %s
        GROUP BY s.session_id, we.visit_id, s.browser, s.os, s.device,
                 s.screen, s.language, s.country, s.region, s.city, s.created_at
        """
        
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, (website_id, start_date, end_date))
            rows = cur.fetchall()
        
        return pd.DataFrame(rows)

    def get_event_data_features(
        self,
        website_id: str,
        start_date: datetime,
        end_date: datetime,
        limit: int = 10000
    ) -> pd.DataFrame:
        """
        Extract custom event data (event_data table) for rich behavioral features.
        Each row is a custom event with its properties.
        """
        query = """
        SELECT 
            ed.website_event_id AS event_id,
            we.visit_id,
            we.session_id,
            we.url_path,
            we.event_name,
            ed.data_key AS event_key,
            ed.string_value,
            ed.number_value,
            ed.date_value,
            we.created_at
        FROM event_data ed
        JOIN website_event we ON ed.website_event_id = we.event_id
        WHERE we.website_id = %s
          AND we.created_at BETWEEN %s AND %s
          AND we.event_type = 2  -- custom events only
        ORDER BY we.created_at DESC
        LIMIT %s
        """
        
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, (website_id, start_date, end_date, limit))
            rows = cur.fetchall()
        
        return pd.DataFrame(rows)

    def get_event_types_summary(
        self,
        website_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> dict:
        """
        Get event type distribution per session (signup, purchase, download, etc.)
        """
        query = """
        SELECT 
            visit_id,
            event_name,
            COUNT(*) AS event_count
        FROM website_event
        WHERE website_id = %s
          AND created_at BETWEEN %s AND %s
          AND event_type = 2  -- custom events
          AND event_name IS NOT NULL
          AND event_name != ''
        GROUP BY visit_id, event_name
        ORDER BY visit_id, event_count DESC
        """
        
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, (website_id, start_date, end_date))
            rows = cur.fetchall()
        
        # Pivot to per-session event type counts
        session_events = {}
        for row in rows:
            vid = row['visit_id']
            if vid not in session_events:
                session_events[vid] = {}
            session_events[vid][row['event_name']] = row['event_count']
        
        return session_events

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

    def get_age_recommendations(
        self,
        website_id: str,
        current_page: str,
        max_results: int = 20
    ) -> list[dict]:
        """
        Use Apache AGE graph to recommend next pages based on graph paths.
        Queries the umami_analytics graph using openCypher.
        Returns pages most commonly visited after the current page.
        """
        with self.conn.cursor() as cur:
            cur.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'age')")
            age_available = cur.fetchone()[0]
        
        if not age_available:
            return []
        
        query = """
        SELECT * FROM cypher('umami_analytics', $$
            MATCH (p1:Page {url: %s, website_id: %s})-[r:LEADS_TO]->(p2:Page)
            RETURN p2.url AS target, r.weight AS weight
            ORDER BY weight DESC
            LIMIT %d
        $$) AS (target text, weight float);
        """ % max_results
        
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, (current_page, website_id))
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.debug(f"AGE query failed: {e}")
            return []

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
            RETURN p1.url AS source, p2.url AS target, p1.weight AS weight
            ORDER BY weight DESC
            LIMIT %d
        $$) AS (source text, target text, weight float);
        """ % max_results
        
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, (website_id,))
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"AGE query failed: {e}")
            return []
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
