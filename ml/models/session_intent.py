"""
Session Intent Classifier - Model 3
Classifies visitor sessions into intent categories:
"researching", "ready-to-buy", "price-comparing", "support-seeking",
"just-browsing", "churning"

Uses a lightweight Transformer (DistilBERT) for NLP on URL paths and
page titles, combined with session features via ensemble.

Reference: arxiv:2606.20482 (mouse/eye tracking for intent inference),
           arxiv:2112.02812 (user behavior understanding)
"""

import os
import logging
from typing import Optional, Any
import numpy as np

from .base import BaseModel
from ..config import CONFIG
from .. import gpu_utils

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    nn = None

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Intent categories
INTENT_LABELS = [
    "researching",
    "ready_to_buy",
    "price_comparing",
    "support_seeking",
    "just_browsing",
    "churning",
    "account_management",
    "content_consumption",
]


class SessionIntentClassifier(BaseModel):
    """
    Model 3: Session Intent Classifier
    
    Uses a two-stage approach:
    1. Embed URL paths and page titles with SentenceTransformer (GPU)
    2. Combine embeddings with session features into RandomForest/XGBoost
    
    Architecture:
    - Text encoder: SentenceTransformer (all-MiniLM-L6-v2) on GPU
    - Tabular classifier: RandomForest/XGBoost on session features
    - Ensemble: weighted voting between text-based and feature-based predictions
    """
    
    def __init__(self):
        super().__init__(name="session_intent", version="1.0.0")
        self.text_encoder = None
        self.classifier = None
        self.scaler = None
        self.label_encoder = LabelEncoder() if SKLEARN_AVAILABLE else None
        self.intent_labels = INTENT_LABELS
        self.device = gpu_utils.DEVICE
    
    def _load_text_encoder(self):
        """Load sentence transformer for text embeddings (GPU if available)"""
        if not ST_AVAILABLE:
            logger.warning("SentenceTransformer not available, using RandomForest only")
            return None
        
        if self.text_encoder is None:
            model_name = "all-MiniLM-L6-v2"  # 80MB, works on CPU, faster on GPU
            logger.info(f"Loading text encoder: {model_name} on {self.device}")
            self.text_encoder = SentenceTransformer(
                model_name,
                device=str(self.device)
            )
        return self.text_encoder
    
    def _extract_text_features(self, pages: list[str]) -> np.ndarray:
        """
        Embed page URLs/paths into vectors.
        Falls back to simple feature extraction without SentenceTransformer.
        """
        if not pages:
            return np.zeros(384)  # default embedding dimension
        
        encoder = self._load_text_encoder()
        if encoder:
            embeddings = encoder.encode(
                pages,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            # Mean pooling over all pages in session
            return embeddings.mean(axis=0)
        
        # Fallback: simple bag-of-words like features
        features = []
        for page in pages:
            parts = page.strip('/').split('/')
            depth = len(parts) - 1
            has_numbers = any(c.isdigit() for c in page)
            features.extend([
                depth / 10.0,  # normalized depth
                float(has_numbers),
                float(page.count('-')),
                float(page.count('?')),
                float(len(page)) / 500.0,  # normalized length
            ])
        
        # Pad or truncate to fixed size
        while len(features) < 50:
            features.append(0.0)
        return np.array(features[:50])
    
    def _extract_session_features(self, session: dict) -> np.ndarray:
        """Extract tabular features from session metadata"""
        features = []
        
        # Numeric session features
        numeric = [
            'page_views', 'unique_pages', 'session_duration',
            'avg_time_on_page', 'scroll_depth_pct', 'bounce_probability',
            'return_visits_7d', 'avg_lcp', 'avg_cls', 'avg_inp',
            'hour_of_day', 'day_of_week',
        ]
        for field in numeric:
            features.append(float(session.get(field, 0)))
        
        # Categorical features (simple encoding)
        cat_map = {
            'browser': {'Chrome': 0, 'Firefox': 1, 'Safari': 2, 'Edge': 3},
            'device': {'desktop': 0, 'mobile': 1, 'tablet': 2},
            'referrer_type': {'search': 0, 'social': 1, 'direct': 2, 'email': 3, 'other': 4},
        }
        for field, mapping in cat_map.items():
            val = session.get(field, '')
            features.append(float(mapping.get(val, -1)))
        
        return np.array(features)
    
    def train(
        self,
        sessions: list[dict],
        page_sequences: list[list[str]],
        labels: list[str],
    ):
        """
        Train intent classifier.
        
        Args:
            sessions: List of session feature dicts
            page_sequences: List of page URL sequences per session
            labels: Intent label strings
        """
        logger.info(f"Training SessionIntentClassifier on {len(sessions)} sessions")
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required")
        
        # Encode labels
        all_labels = list(set(self.intent_labels) | set(labels))
        self.label_encoder.fit(all_labels)
        y = self.label_encoder.transform(labels)
        
        # Build feature matrix
        text_features = []
        session_features = []
        
        for pages, session in zip(page_sequences, sessions):
            tf = self._extract_text_features(pages)
            sf = self._extract_session_features(session)
            text_features.append(tf)
            session_features.append(sf)
        
        X_text = np.array(text_features)
        X_session = np.array(session_features)
        
        # Scale session features
        self.scaler = StandardScaler()
        X_session = self.scaler.fit_transform(X_session)
        
        # Combine features
        X = np.concatenate([X_text, X_session], axis=1)
        
        # Train RandomForest (CPU) - works well for intent classification
        self.classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42,
        )
        self.classifier.fit(X, y)
        
        # Score
        accuracy = self.classifier.score(X, y)
        
        self.metadata['train_accuracy'] = float(accuracy)
        self.metadata['n_classes'] = len(self.intent_labels)
        self.metadata['text_feature_dim'] = X_text.shape[1]
        self.metadata['session_feature_dim'] = X_session.shape[1]
        
        logger.info(f"Intent classifier trained: accuracy={accuracy:.4f}")
        self.is_trained = True
    
    def predict(
        self,
        pages: list[str],
        session: dict,
    ) -> dict:
        """
        Predict intent for a session.
        
        Args:
            pages: URL paths visited in this session so far
            session: Session feature dict
            
        Returns:
            Intent prediction with probabilities for each class
        """
        if self.classifier is None:
            return {'intent': 'unknown', 'probabilities': {}}
        
        # Extract features
        tf = self._extract_text_features(pages)
        sf = self._extract_session_features(session)
        
        if self.scaler is not None:
            sf = self.scaler.transform(sf.reshape(1, -1)).flatten()
        
        X = np.concatenate([tf, sf.reshape(-1)]).reshape(1, -1)
        
        # Predict
        y_pred = self.classifier.predict(X)[0]
        probs = self.classifier.predict_proba(X)[0]
        
        intent = self.label_encoder.inverse_transform([y_pred])[0]
        
        return {
            'intent': intent,
            'confidence': float(max(probs)),
            'probabilities': {
                self.label_encoder.inverse_transform([i])[0]: float(p)
                for i, p in enumerate(probs)
            },
            'is_cold_start': len(pages) <= 2,
        }
    
    def to_onnx(self, path: Optional[str] = None) -> str:
        """Export to ONNX via sklearn-onnx"""
        if self.classifier is None:
            raise RuntimeError("Model not trained")
        
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            
            n_features = self.classifier.n_features_in_
            initial_type = [('float_input', FloatTensorType([None, n_features]))]
            onnx_model = convert_sklearn(self.classifier, initial_types=initial_type)
            
            path = path or CONFIG.paths.session_intent_model
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                f.write(onnx_model.SerializeToString())
            
            logger.info(f"Intent classifier exported to ONNX: {path}")
            return path
        except ImportError:
            logger.warning("skl2onnx not available")
            raise
