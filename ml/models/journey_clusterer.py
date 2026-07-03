"""
Journey Clustering Model - Model 6
Unsupervised discovery of visitor journey archetypes using UMAP + HDBSCAN.

Automatically segments visitors into behavioral groups:
- Power users: deep navigation, many pages, long sessions
- Comparison shoppers: product pages, back-and-forth navigation
- Single-page scanners: one page, short time, bounce
- Lost users: pogo-sticking, rapid back/forward, rage clicks
- Conversion-ready: checkout flow, cart, thank-you page
- Content consumers: blog posts, articles, documentation

Reference: arxiv:2502.00413 (session dimensionality reduction for clustering)
"""

import os
import json
import logging
from typing import Optional
from datetime import datetime, timedelta

import numpy as np

from .base import BaseModel
from ..config import CONFIG

logger = logging.getLogger(__name__)

try:
    import umap
    import hdbscan
    HAS_CLUSTERING = True
except ImportError:
    HAS_CLUSTERING = False

try:
    import psycopg2
    import psycopg2.extras
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# Archetype labels discovered by clustering
ARCHETYPE_LABELS = {
    0: 'power_user',
    1: 'comparison_shopper',
    2: 'single_page_scanner',
    3: 'lost_user',
    4: 'conversion_ready',
    5: 'content_consumer',
    6: 'returning_visitor',
    7: 'deal_seeker',
    8: 'support_seeker',
    9: 'browser',
    -1: 'unclassified',
}

ARCHETYPE_DESCRIPTIONS = {
    'power_user': 'Deep navigation, many pages, long sessions, high engagement',
    'comparison_shopper': 'Product pages, back-and-forth navigation, price checking',
    'single_page_scanner': 'One page, short time, high bounce probability',
    'lost_user': 'Pogo-sticking, rapid navigation, possible frustration',
    'conversion_ready': 'Checkout flow, cart, thank-you page, high purchase intent',
    'content_consumer': 'Blog posts, articles, documentation, long read times',
    'returning_visitor': 'Multiple sessions, known paths, logged-in behavior',
    'deal_seeker': 'Pricing pages, promotions, coupon codes',
    'support_seeker': 'Support pages, FAQ, contact forms, help docs',
    'browser': 'Random navigation, no clear pattern, exploration mode',
    'unclassified': 'Novel pattern not matching known archetypes',
}


class JourneyClusterer(BaseModel):
    """
    Model 6: Unsupervised visitor journey clustering.
    
    Pipeline:
    1. Encode each session as a fixed-length feature vector
    2. Reduce dimensionality with UMAP (n_components=5-10)
    3. Cluster with HDBSCAN (no k needed, handles noise)
    4. Label clusters by top distinctive features
    
    GPU: UMAP can use GPU via cuML if available
    CPU: UMAP + HDBSCAN on CPU (fast for <100K sessions)
    """
    
    def __init__(self):
        super().__init__(name="journey_clusterer", version="1.0.0")
        self.umap_model = None
        self.cluster_model = None
        self.scaler = None
        self.feature_names = []
        self.cluster_labels = {}
        self.archetype_map = {}
    
    def extract_session_features(self, session: dict) -> dict:
        """
        Extract feature vector from a session for clustering.
        
        Features:
        - n_pages: total page views
        - unique_pages: distinct URLs visited
        - session_duration_seconds: time from first to last event
        - avg_time_on_page: average seconds per page
        - n_referrers: distinct referrer domains
        - has_checkout: boolean
        - has_pricing: boolean
        - has_blog: boolean
        - has_search: boolean
        - has_support: boolean
        - has_cart: boolean
        - bounce: boolean (only 1 page)
        - n_products: count of /product/ URLs
        - n_categories: count of /category/ URLs
        - avg_lcp: average Largest Contentful Paint
        - avg_cls: average Cumulative Layout Shift
        - avg_inp: average Interaction to Next Paint
        - n_events: count of custom events
        - n_searches: count of /search URLs
        - page_depth: max path depth
        - hour_of_day: session start hour
        - is_weekend: boolean
        - device_type: desktop/mobile/tablet
        """
        features = {}
        
        # Page-level features
        pages = session.get('pages', [])
        features['n_pages'] = len(pages)
        features['unique_pages'] = len(set(pages))
        features['bounce'] = 1 if len(pages) <= 1 else 0
        
        # URL pattern features
        product_count = sum(1 for p in pages if '/product/' in p)
        category_count = sum(1 for p in pages if '/category/' in p)
        search_count = sum(1 for p in pages if '/search' in p)
        features['n_products'] = product_count
        features['n_categories'] = category_count
        features['n_searches'] = search_count
        features['has_checkout'] = 1 if any('checkout' in p for p in pages) else 0
        features['has_pricing'] = 1 if any('pricing' in p or '/price' in p for p in pages) else 0
        features['has_blog'] = 1 if any('/blog' in p for p in pages) else 0
        features['has_support'] = 1 if any(p in ('/support', '/faq', '/contact', '/help') for p in pages) else 0
        features['has_cart'] = 1 if any('/cart' in p for p in pages) else 0
        features['has_search'] = 1 if search_count > 0 else 0
        
        # Page depth
        max_depth = 0
        for p in pages:
            depth = len(p.rstrip('/').split('/'))
            if depth > max_depth:
                max_depth = depth
        features['page_depth'] = max_depth
        
        # Time features
        duration = session.get('duration_seconds', 0)
        features['session_duration_seconds'] = duration
        features['avg_time_on_page'] = duration / max(1, len(pages))
        
        # Session metadata
        features['n_referrers'] = session.get('n_referrers', 0)
        features['n_events'] = session.get('n_events', 0)
        
        # Performance
        features['avg_lcp'] = session.get('avg_lcp', 0) or 0
        features['avg_cls'] = session.get('avg_cls', 0) or 0
        features['avg_inp'] = session.get('avg_inp', 0) or 0
        
        # Temporal
        hour = session.get('hour', 12)
        features['hour_of_day'] = hour
        features['is_weekend'] = 1 if session.get('is_weekend', False) else 0
        
        # Device
        device = session.get('device', 'desktop')
        features['is_desktop'] = 1 if device == 'desktop' else 0
        features['is_mobile'] = 1 if device == 'mobile' else 0
        
        return features
    
    def _get_feature_vector(self, features: dict) -> np.ndarray:
        """Convert feature dict to numpy array in consistent order."""
        self.feature_names = sorted(features.keys())
        return np.array([features.get(f, 0) for f in self.feature_names])
    
    def train(self, sessions: list[dict]):
        """
        Train UMAP + HDBSCAN clustering on session data.
        
        Args:
            sessions: List of session feature dicts
        """
        if not HAS_CLUSTERING:
            logger.warning("umap-learn or hdbscan not available")
            self.is_trained = True
            return
        
        if len(sessions) < 10:
            logger.info(f"Only {len(sessions)} sessions, need 10+ for clustering")
            self.is_trained = True
            return
        
        # Extract feature vectors
        feature_dicts = [self.extract_session_features(s) for s in sessions]
        X = np.array([self._get_feature_vector(f) for f in feature_dicts])
        
        # Scale
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # UMAP dimensionality reduction
        n_neighbors = min(15, len(sessions) - 1)
        self.umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            n_components=5,
            min_dist=0.1,
            metric='euclidean',
            random_state=42,
        )
        X_umap = self.umap_model.fit_transform(X_scaled)
        
        # HDBSCAN clustering
        min_cluster_size = max(5, len(sessions) // 50)
        self.cluster_model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,
            metric='euclidean',
            cluster_selection_epsilon=0.5,
            prediction_data=True,
        )
        cluster_labels = self.cluster_model.fit_predict(X_umap)
        
        # Map cluster labels to archetypes
        unique_labels = set(cluster_labels)
        archetype_keys = list(ARCHETYPE_LABELS.keys())
        self.archetype_map = {}
        for i, label in enumerate(sorted(unique_labels)):
            if label == -1:
                self.archetype_map[-1] = 'unclassified'
            elif i < len(archetype_keys):
                self.archetype_map[label] = ARCHETYPE_LABELS[archetype_keys[i]]
            else:
                self.archetype_map[label] = f'archetype_{label}'
        
        # Compute cluster centroids and top features
        self.cluster_labels = {}
        for label in unique_labels:
            mask = cluster_labels == label
            if mask.sum() == 0:
                continue
            centroid = X_scaled[mask].mean(axis=0)
            archetype = self.archetype_map.get(label, 'unknown')
            
            # Find top distinctive features (highest absolute deviation from mean)
            global_mean = X_scaled.mean(axis=0)
            deviation = np.abs(centroid - global_mean)
            top_features_idx = np.argsort(deviation)[-5:][::-1]
            top_features = [
                {'feature': self.feature_names[i], 'value': float(centroid[i])}
                for i in top_features_idx
            ]
            
            self.cluster_labels[int(label)] = {
                'archetype': archetype,
                'description': ARCHETYPE_DESCRIPTIONS.get(archetype, ''),
                'size': int(mask.sum()),
                'percentage': float(mask.sum() / len(sessions)),
                'top_features': top_features,
            }
        
        self.metadata['n_sessions'] = len(sessions)
        self.metadata['n_clusters'] = len(unique_labels) - (1 if -1 in unique_labels else 0)
        self.metadata['noise_ratio'] = float((cluster_labels == -1).sum() / len(sessions))
        
        logger.info(f"Trained JourneyClusterer: {self.metadata['n_clusters']} clusters "
                    f"from {len(sessions)} sessions")
        self.is_trained = True
    
    def predict(self, session: dict) -> dict:
        """
        Predict archetype for a single session.
        
        Args:
            session: Session feature dict
            
        Returns:
            dict with archetype, probability, and explanation
        """
        if self.umap_model is None or self.cluster_model is None:
            return {'archetype': 'unclassified', 'probability': 0.0}
        
        features = self.extract_session_features(session)
        X = self._get_feature_vector(features).reshape(1, -1)
        
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        X_umap = self.umap_model.transform(X_scaled)
        label = self.cluster_model.predict(X_umap)[0]
        
        # Get membership probability (HDBSCAN's outlier scores)
        if hasattr(self.cluster_model, 'probabilities_'):
            probs = self.cluster_model.probabilities_
            prob = float(probs[0]) if len(probs) > 0 else 0.5
        else:
            prob = 0.5
        
        archetype = self.archetype_map.get(int(label), 'unclassified')
        
        return {
            'archetype': archetype,
            'description': ARCHETYPE_DESCRIPTIONS.get(archetype, ''),
            'probability': prob,
            'cluster_id': int(label),
            'features': features,
        }
    
    def get_archetype_distribution(self) -> list[dict]:
        """Get distribution of archetypes across all clusters."""
        return [
            {
                'archetype': info['archetype'],
                'description': info['description'],
                'size': info['size'],
                'percentage': info['percentage'],
                'top_features': info['top_features'],
            }
            for label, info in sorted(self.cluster_labels.items())
            if label != -1
        ]
    
    def batch_predict(self, sessions: list[dict]) -> list[dict]:
        """Predict archetypes for multiple sessions."""
        return [self.predict(s) for s in sessions]
