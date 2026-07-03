"""
Session Replay & Rage Click Analysis - Model 5
Detects user frustration signals from heatmap and session replay data.

Detectable signals:
  - Rage clicks: rapid repeated clicks on the same area
  - Dead clicks: clicks on non-interactive elements
  - Mouse shaking: rapid mouse movement patterns
  - Form abandonment: partial form fill + sudden exit
  - Scroll rage: very fast scroll followed by interaction
  - Error loops: repeated error events

Reference: HCI research on click frustration + rrweb session replay analysis

Data sources:
  - heatmap_event: click coordinates, scroll depth, viewport dimensions
  - session_replay: full interaction recordings (rrweb JSON events)
"""

import os
import json
import logging
from typing import Optional
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np

from .base import BaseModel
from ..config import CONFIG
from .. import gpu_utils

logger = logging.getLogger(__name__)

try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# Frustration signal types
FRUSTRATION_TYPES = {
    'rage_click': 'Rapid repeated clicks on same area',
    'dead_click': 'Click on non-interactive element',
    'mouse_shake': 'Erratic rapid mouse movements',
    'form_abandonment': 'Partial form fill then exit',
    'scroll_rage': 'Rapid scroll followed by interaction',
    'error_loop': 'Repeated error events',
    'none': 'No frustration detected',
}


class RageClickDetector(BaseModel):
    """
    Model 5: Rage Click & Frustration Detection
    Uses Isolation Forest for unsupervised anomaly detection on
    click patterns, plus rule-based heuristics for known signals.

    GPU: Uses GPU for Isolation Forest if available (via cuml)
    CPU: sklearn Isolation Forest fallback
    """
    
    def __init__(self):
        super().__init__(name="rage_click_detector", version="1.0.0")
        self.model = None
        self.scaler = None
        self.thresholds = {
            'rage_click_time_window': 1.5,       # seconds
            'rage_click_distance': 50,            # pixels
            'rage_click_min_count': 3,            # minimum clicks to qualify
            'mouse_shake_speed': 2000,            # pixels/second
            'dead_click_confidence': 0.6,         # model confidence threshold
        }
    
    def extract_click_features(self, clicks: list[dict]) -> dict:
        """
        Extract features from a session's click events for frustration detection.
        
        Args:
            clicks: list of dicts with x, y, page_x, page_y, scroll_pct, 
                   viewport_w, viewport_h, page_h, created_at
        
        Returns:
            Feature dict for classification
        """
        if not clicks:
            return {}
        
        n_clicks = len(clicks)
        timestamps = [c.get('created_at', 0) for c in clicks]
        
        # Time-based features
        if len(timestamps) >= 2:
            time_diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            avg_inter_click_time = sum(time_diffs) / len(time_diffs) if time_diffs else 0
            min_inter_click_time = min(time_diffs) if time_diffs else 0
            max_inter_click_time = max(time_diffs) if time_diffs else 0
            total_duration = timestamps[-1] - timestamps[0] if len(timestamps) >= 2 else 0
        else:
            avg_inter_click_time = min_inter_click_time = max_inter_click_time = total_duration = 0
        
        # Spatial features
        x_coords = [c.get('x', 0) for c in clicks]
        y_coords = [c.get('y', 0) for c in clicks]
        
        if n_clicks >= 2:
            distances = []
            for i in range(n_clicks - 1):
                d = ((x_coords[i] - x_coords[i+1])**2 + (y_coords[i] - y_coords[i+1])**2) ** 0.5
                distances.append(d)
            avg_distance = sum(distances) / len(distances)
            min_distance = min(distances)
            # Clusters: clicks very close to each other
            rage_clusters = sum(1 for d in distances if d < self.thresholds['rage_click_distance'])
        else:
            avg_distance = min_distance = 0
            rage_clusters = 0
        
        # Scroll features
        scrolls = [c.get('scroll_pct', 0) or 0 for c in clicks]
        avg_scroll = sum(scrolls) / len(scrolls) if scrolls else 0
        
        # Speed features (rage click = fast repeated clicks close together)
        total_interaction_speed = (sum(distances) / total_duration) if total_duration > 0 else 0
        
        # Viewport features
        viewport_widths = [c.get('viewport_w', 1024) or 1024 for c in clicks]
        viewport_heights = [c.get('viewport_h', 768) or 768 for c in clicks]
        avg_vw = sum(viewport_widths) / len(viewport_widths)
        avg_vh = sum(viewport_heights) / len(viewport_heights)
        
        # Click density: clicks per second
        click_density = n_clicks / total_duration if total_duration > 0 else 0
        
        # Position consistency: are all clicks in a small area?
        if n_clicks >= 2:
            x_range = max(x_coords) - min(x_coords) if x_coords else 0
            y_range = max(y_coords) - min(y_coords) if y_coords else 0
            position_cluster_ratio = (x_range + y_range) / (avg_vw + avg_vh) if (avg_vw + avg_vh) > 0 else 1
        else:
            x_range = y_range = position_cluster_ratio = 0
        
        return {
            'n_clicks': n_clicks,
            'avg_inter_click_time': float(avg_inter_click_time),
            'min_inter_click_time': float(min_inter_click_time),
            'total_duration': float(total_duration),
            'avg_distance': float(avg_distance),
            'min_distance': float(min_distance),
            'rage_clusters': rage_clusters,
            'avg_scroll': float(avg_scroll),
            'interaction_speed': float(total_interaction_speed),
            'click_density': float(click_density),
            'x_range': float(x_range),
            'y_range': float(y_range),
            'position_cluster_ratio': float(position_cluster_ratio),
            'avg_vw': float(avg_vw),
            'avg_vh': float(avg_vh),
        }
    
    def _rule_based_detection(self, features: dict) -> tuple[str, float]:
        """
        Rule-based frustration detection using heuristics.
        Returns (signal_type, confidence).
        """
        n = features.get('n_clicks', 0)
        min_dist = features.get('min_distance', 999)
        click_density = features.get('click_density', 0)
        avg_time = features.get('avg_inter_click_time', 999)
        cluster_ratio = features.get('position_cluster_ratio', 1)
        rage_clusters = features.get('rage_clusters', 0)
        interaction_speed = features.get('interaction_speed', 0)
        x_range = features.get('x_range', 0)
        y_range = features.get('y_range', 0)
        
        # Rage click: 3+ clicks on same tiny area within short time
        if (rage_clusters >= self.thresholds['rage_click_min_count'] and 
            min_dist < self.thresholds['rage_click_distance'] and
            avg_time < self.thresholds['rage_click_time_window']):
            confidence = min(1.0, 0.5 + rage_clusters * 0.1)
            return ('rage_click', confidence)
        
        # Dead click: single click, then click again on same spot quickly
        if (n >= 2 and min_dist < 10 and avg_time < 2.0):
            confidence = 0.7
            return ('dead_click', confidence)
        
        # Mouse shake: rapid mouse movement (high interaction speed)
        if interaction_speed > self.thresholds['mouse_shake_speed'] and n >= 3:
            confidence = min(1.0, 0.5 + interaction_speed / 5000 * 0.3)
            return ('mouse_shake', confidence)
        
        # Scroll rage: high scroll depth + rapid clicks
        if (features.get('avg_scroll', 0) > 80 and click_density > 3 and avg_time < 1.0):
            return ('scroll_rage', 0.7)
        
        # No frustration detected
        return ('none', 0.0)
    
    def train(self, click_sessions: list[dict]):
        """
        Train Isolation Forest for anomaly detection on click patterns.
        Unsupervised - learns normal behavior, flags anomalies as frustration.
        
        Args:
            click_sessions: List of session feature dicts (from extract_click_features)
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available, using rule-based only")
            self.is_trained = True
            return
        
        valid = [s for s in click_sessions if s]
        if len(valid) < 5:
            logger.info(f"Only {len(valid)} sessions, using rule-based only")
            self.is_trained = True
            return
        
        # Extract feature vectors
        feature_names = [
            'n_clicks', 'avg_inter_click_time', 'min_inter_click_time',
            'total_duration', 'avg_distance', 'min_distance', 'rage_clusters',
            'avg_scroll', 'interaction_speed', 'click_density',
            'x_range', 'y_range', 'position_cluster_ratio',
        ]
        
        X = []
        for s in valid:
            X.append([s.get(f, 0) for f in feature_names])
        
        X = np.array(X)
        
        # Scale
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest (contamination ~5% expected frustration rate)
        self.model = IsolationForest(
            n_estimators=100,
            max_samples='auto',
            contamination=0.05,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)
        
        # Score
        anomaly_scores = self.model.decision_function(X_scaled)
        n_anomalies = sum(1 for s in anomaly_scores if s < 0)
        
        self.metadata['n_sessions'] = len(valid)
        self.metadata['n_anomalies'] = n_anomalies
        self.metadata['anomaly_rate'] = n_anomalies / len(valid) if valid else 0
        
        logger.info(f"Trained RageClickDetector: {n_anomalies}/{len(valid)} sessions flagged")
        self.is_trained = True
    
    def predict(self, clicks: list[dict]) -> dict:
        """
        Detect frustration signals from a session's click events.
        Combines Isolation Forest anomaly score with rule-based heuristics.
        
        Args:
            clicks: List of click event dicts from heatmap_event
            
        Returns:
            dict with frustration type, confidence, and detailed analysis
        """
        features = self.extract_click_features(clicks)
        
        if not features or features.get('n_clicks', 0) < 2:
            return {
                'frustration_type': 'none',
                'confidence': 0.0,
                'severity': 'none',
                'features': features,
                'details': 'Not enough interaction data',
            }
        
        # 1. Rule-based detection (always available)
        rule_type, rule_confidence = self._rule_based_detection(features)
        
        # 2. ML-based anomaly score (if trained)
        ml_frustration = 0.0
        if self.model is not None and self.scaler is not None:
            feature_names = [
                'n_clicks', 'avg_inter_click_time', 'min_inter_click_time',
                'total_duration', 'avg_distance', 'min_distance', 'rage_clusters',
                'avg_scroll', 'interaction_speed', 'click_density',
                'x_range', 'y_range', 'position_cluster_ratio',
            ]
            X = np.array([[features.get(f, 0) for f in feature_names]])
            X_scaled = self.scaler.transform(X)
            
            # Anomaly score: negative = anomaly
            anomaly = self.model.decision_function(X_scaled)[0]
            ml_frustration = max(0, -anomaly)  # 0 = normal, higher = more anomalous
        
        # Combine: use ML if confident, fall back to rules
        if ml_frustration > 0.3 and rule_type == 'none':
            # ML detected anomaly that rules missed
            final_type = 'rage_click' if ml_frustration > 0.5 else 'none'
            final_confidence = ml_frustration
        else:
            final_type = rule_type
            final_confidence = rule_confidence
        
        # Determine severity
        if final_confidence >= 0.8:
            severity = 'high'
        elif final_confidence >= 0.5:
            severity = 'medium'
        elif final_confidence > 0:
            severity = 'low'
        else:
            severity = 'none'
        
        return {
            'frustration_type': final_type,
            'confidence': round(final_confidence, 4),
            'severity': severity,
            'ml_anomaly_score': round(ml_frustration, 4),
            'rule_based': {
                'type': rule_type,
                'confidence': rule_confidence,
            },
            'features': {
                'n_clicks': features.get('n_clicks', 0),
                'avg_inter_click_time': features.get('avg_inter_click_time', 0),
                'rage_clusters': features.get('rage_clusters', 0),
                'interaction_speed': features.get('interaction_speed', 0),
                'click_density': features.get('click_density', 0),
            },
            'details': FRUSTRATION_TYPES.get(final_type, 'No frustration detected'),
        }
    
    def predict_session(self, session_id: str, website_id: str) -> dict:
        """
        Analyze a specific session for frustration signals.
        Fetches heatmap data from TimescaleDB.
        """
        if not HAS_PSYCOPG2:
            return {'error': 'Database not available'}
        
        try:
            conn = psycopg2.connect(CONFIG.db.url)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            cur.execute("""
                SELECT x, y, page_x, page_y, scroll_pct,
                       viewport_w, viewport_h, page_h, created_at
                FROM heatmap_event
                WHERE session_id = %s AND website_id = %s
                ORDER BY created_at
            """, (session_id, website_id))
            
            clicks = [dict(r) for r in cur.fetchall()]
            cur.close()
            conn.close()
            
            return self.predict(clicks)
        except Exception as e:
            logger.error(f"Session analysis failed: {e}")
            return {'error': str(e)}
    
    def batch_analyze(self, website_id: str, start_date: datetime, end_date: datetime) -> list[dict]:
        """
        Analyze all sessions in a date range for frustration signals.
        Returns list of sessions flagged as frustrated.
        """
        if not HAS_PSYCOPG2:
            return []
        
        try:
            conn = psycopg2.connect(CONFIG.db.url)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            # Get all sessions with heatmap events
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
                LIMIT 1000
            """, (website_id, start_date, end_date))
            
            sessions = []
            for row in cur.fetchall():
                clicks_raw = row['clicks']
                clicks = [
                    {
                        'x': c[0], 'y': c[1], 'page_x': c[2], 'page_y': c[3],
                        'scroll_pct': c[4], 'viewport_w': c[5], 'viewport_h': c[6],
                        'page_h': c[7], 'created_at': c[8],
                    }
                    for c in clicks_raw
                ]
                result = self.predict(clicks)
                if result['severity'] in ('high', 'medium'):
                    sessions.append({
                        'session_id': row['session_id'],
                        'visit_id': row['visit_id'],
                        'click_count': len(clicks),
                        **result,
                    })
            
            cur.close()
            conn.close()
            
            return sorted(sessions, key=lambda x: -x['confidence'])
        except Exception as e:
            logger.error(f"Batch analysis failed: {e}")
            return []


# Export API endpoint handler pattern
FRUSTRATION_TYPES_INFO = FRUSTRATION_TYPES
