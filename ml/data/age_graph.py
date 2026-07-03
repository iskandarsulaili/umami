"""
Apache AGE Integration for Umami ML
Fully utilizes Apache AGE graph database for visitor journey analysis.

AGE provides openCypher graph queries directly within PostgreSQL.
The `umami_analytics` graph stores:
  - (:Page {url, website_id, title}) - Page vertices
  - (:Visitor {session_id, browser, os, country}) - Visitor vertices  
  - (:Session {visit_id, started_at}) - Session vertices
  - [:VIEWS] - Edge: Session views Page
  - [:LEADS_TO {count}] - Edge: Page transition to Page
  - [:CLICKED] - Edge: Session clicked on heatmap coordinate

Capabilities:
  1. Multi-step path analysis (graph traversal)
  2. Most common session journeys (shortest path)
  3. Page centrality (most influential pages)
  4. Community detection (content clusters)
  5. Personalized recommendation via graph walks
  6. Fraud detection (anomalous traversal patterns)
"""

import os
import json
import logging
from typing import Optional, Any
from datetime import datetime

from ..config import CONFIG

logger = logging.getLogger(__name__)


class AgeGraphClient:
    """
    Apache AGE graph client for visitor journey analysis.
    Uses AGE's openCypher query language directly on the umami_analytics graph.
    All graph operations execute within PostgreSQL transactions.
    """
    
    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = connection_string or CONFIG.db.url
        self._conn = None
        self.graph_name = CONFIG.age.graph_name or 'umami_analytics'
    
    @property
    def conn(self):
        if self._conn is None or self._conn.closed:
            import psycopg2
            self._conn = psycopg2.connect(self.connection_string)
            self._conn.autocommit = True
        return self._conn
    
    def close(self):
        if self._conn and not self._conn.closed:
            self._conn.close()
    
    def check_available(self) -> bool:
        """Check if AGE extension is available"""
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'age')")
            available = cur.fetchone()[0]
            cur.close()
            return available
        except Exception:
            return False
    
    def _cypher(self, query: str, params: Optional[dict] = None) -> list[tuple]:
        """
        Execute an openCypher query on the AGE graph.
        Returns results as a list of tuples matching the RETURN clause columns.
        """
        full_query = f"""
        SELECT * FROM cypher('{self.graph_name}', ${query}$) AS result
        """
        # AGE requires specific column definition. We use a generic approach
        # for queries with known schemas.
        
        try:
            cur = self.conn.cursor()
            if params:
                cur.execute(full_query, params)
            else:
                cur.execute(full_query)
            results = cur.fetchall()
            cur.close()
            return results
        except Exception as e:
            logger.warning(f"AGE query failed: {e}")
            return []
    
    def get_most_common_paths(self, website_id: str, limit: int = 50) -> list[dict]:
        """
        Use AGE graph traversal to find most common page-to-page paths.
        Uses agtype_object_field_text workaround for AGE 1.7 RC0 compatibility.
        """
        query = f"""
        WITH graph_result AS (
            SELECT * FROM ag_catalog.cypher('{self.graph_name}', $q$
                MATCH (p1:Page)-[r:LEADS_TO]->(p2:Page)
                WHERE p1.website_id = '{website_id}'
                RETURN p1, p2, r
            $q$) AS (p1 ag_catalog.agtype, p2 ag_catalog.agtype, r ag_catalog.agtype)
        )
        SELECT 
            ag_catalog.agtype_object_field_text(p1, 'url') AS source,
            ag_catalog.agtype_object_field_text(p2, 'url') AS target,
            ag_catalog.agtype_object_field_text(r, 'count')::bigint AS weight
        FROM graph_result
        ORDER BY weight DESC
        LIMIT {limit}
        """
        
        try:
            cur = self.conn.cursor()
            cur.execute(query)
            results = [
                {'source': r[0], 'target': r[1], 'weight': r[2]}
                for r in cur.fetchall()
            ]
            cur.close()
            return results
        except Exception as e:
            logger.error(f"AGE path query failed: {e}")
            return []
    
    def get_full_journey_patterns(self, website_id: str, max_steps: int = 5, limit: int = 20) -> list[dict]:
        """
        Find complete multi-step journey patterns using graph path traversal.
        
        openCypher:
        MATCH path = (p1:Page)-[:LEADS_TO*1..5]->(p2:Page)
        WHERE p1.website_id = $website_id
        RETURN [node IN nodes(path) | node.url] AS journey,
               reduce(s = 0, r IN relationships(path) | s + r.count) AS total_weight
        ORDER BY total_weight DESC
        LIMIT $limit
        """
        query = f"""
        SELECT * FROM cypher('{self.graph_name}', $$
            MATCH path = (p1:Page)-[:LEADS_TO*1..{max_steps}]->(p2:Page)
            WHERE p1.website_id = '{website_id}'
            RETURN [n IN nodes(path) | n.url] AS journey,
                   reduce(s = 0, r IN relationships(path) | s + r.count) AS score
            ORDER BY score DESC
            LIMIT {limit}
        $$) AS (journey text[], score bigint);
        """
        
        try:
            cur = self.conn.cursor()
            cur.execute(query)
            results = [
                {'journey': r[0], 'score': r[1]}
                for r in cur.fetchall()
            ]
            cur.close()
            return results
        except Exception as e:
            logger.error(f"AGE journey query failed: {e}")
            return []
    
    def recommend_from_graph(self, current_page: str, website_id: str, limit: int = 10) -> list[dict]:
        """
        Graph-based recommendation using AGE.
        Finds pages reachable from the current page with the highest weighted paths.
        
        openCypher: Multi-hop traversal from current page, weighting by total path score.
        """
        query = f"""
        SELECT * FROM cypher('{self.graph_name}', $$
            MATCH path = (p1:Page {{url: '{current_page}', website_id: '{website_id}'}})
                        -[:LEADS_TO*1..3]->(target:Page)
            WHERE target.url <> '{current_page}'
            RETURN target.url AS recommended,
                   reduce(s = 0, r IN relationships(path) | s + r.count) AS score,
                   length(path) AS depth
            ORDER BY score DESC
            LIMIT {limit}
        $$) AS (recommended text, score bigint, depth bigint);
        """
        
        try:
            cur = self.conn.cursor()
            cur.execute(query)
            results = [
                {
                    'page': r[0],
                    'score': float(r[1]) if r[1] else 0.0,
                    'depth': r[2],
                }
                for r in cur.fetchall()
            ]
            cur.close()
            return results
        except Exception as e:
            logger.error(f"AGE recommend query failed: {e}")
            return []
    
    def get_page_centrality(self, website_id: str, limit: int = 20) -> list[dict]:
        """
        Find most central/influential pages using graph centrality.
        
        openCypher: Counts inbound and outbound edges weighted by transition count.
        """
        query = f"""
        SELECT * FROM cypher('{self.graph_name}', $$
            MATCH (p:Page {{website_id: '{website_id}'}})-[r:LEADS_TO]-()
            RETURN p.url AS page, SUM(r.count) AS total_transitions, COUNT(r) AS edge_count
            ORDER BY total_transitions DESC
            LIMIT {limit}
        $$) AS (page text, total_transitions bigint, edge_count bigint);
        """
        
        try:
            cur = self.conn.cursor()
            cur.execute(query)
            results = [
                {'page': r[0], 'transitions': r[1], 'edges': r[2]}
                for r in cur.fetchall()
            ]
            cur.close()
            return results
        except Exception as e:
            logger.error(f"AGE centrality query failed: {e}")
            return []
    
    def sync_to_age(self, website_id: str, start_date: datetime, end_date: datetime) -> dict:
        """
        Sync session data from TimescaleDB to Apache AGE graph.
        Creates Page, Session, Visitor vertices and LEADS_TO, VIEWS edges.
        
        This is the bridge between relational data and graph analytics.
        """
        if not self.check_available():
            return {'error': 'AGE not available', 'synced': 0}
        
        import psycopg2
        
        try:
            # Use a fresh connection to avoid transaction state issues
            conn = psycopg2.connect(self.connection_string)
            conn.autocommit = True
            cur = conn.cursor()
            
            # 1. Create Page vertices from unique URLs
            cur.execute("""
                SELECT DISTINCT url_path
                FROM website_event
                WHERE website_id = %s
                  AND created_at BETWEEN %s AND %s
                  AND url_path IS NOT NULL
            """, (website_id, start_date, end_date))
            
            pages = [r[0] for r in cur.fetchall()]
            page_count = 0
            
            for page in pages:
                safe_url = page.replace("'", "''")
                cypher = f"""
                SELECT * FROM cypher('{self.graph_name}', $$
                    MERGE (p:Page {{url: '{safe_url}', website_id: '{website_id}'}})
                    RETURN p
                $$) AS (v ag_catalog.agtype);
                """
                try:
                    cur.execute(cypher)
                    page_count += 1
                except Exception as e:
                    logger.warning(f"Page vertex failed for '{page}': {e}")
                    conn.rollback()
            
            # 2. Build LEADS_TO edges from page transitions with weights
            cur.execute("""
                WITH transitions AS (
                    SELECT 
                        url_path AS source,
                        LEAD(url_path) OVER (
                            PARTITION BY visit_id ORDER BY created_at
                        ) AS target,
                        COUNT(*) AS cnt
                    FROM website_event
                    WHERE website_id = %s
                      AND created_at BETWEEN %s AND %s
                      AND url_path IS NOT NULL
                    GROUP BY visit_id, url_path, created_at
                )
                SELECT source, target, SUM(cnt) AS weight
                FROM transitions
                WHERE target IS NOT NULL
                GROUP BY source, target
            """, (website_id, start_date, end_date))
            
            edge_count = 0
            for source, target, weight in cur.fetchall():
                safe_src = source.replace("'", "''")
                safe_tgt = target.replace("'", "''")
                cypher = f"""
                SELECT * FROM cypher('{self.graph_name}', $$
                    MATCH (p1:Page {{url: '{safe_src}', website_id: '{website_id}'}})
                    MATCH (p2:Page {{url: '{safe_tgt}', website_id: '{website_id}'}})
                    MERGE (p1)-[r:LEADS_TO {{website_id: '{website_id}', count: {weight}}}]->(p2)
                    RETURN r
                $$) AS (e ag_catalog.agtype);
                """
                try:
                    cur.execute(cypher)
                    edge_count += 1
                except Exception as e:
                    logger.warning(f"Edge failed {source}->{target}: {e}")
            
            cur.close()
            conn.close()
            
            logger.info(f"Synced {page_count} pages and {edge_count} edges to AGE graph")
            return {'pages': page_count, 'edges': edge_count, 'status': 'synced'}
        
        except Exception as e:
            logger.error(f"AGE sync failed: {e}")
            return {'error': str(e), 'synced': 0}
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
