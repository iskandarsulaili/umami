"""
Funnel Drop-off Predictor - Model 2
Predicts probability that a visitor will proceed through each funnel step.
Uses Gradient Boosting (XGBoost) with GPU support.

Reference: arxiv:2502.00413 (CAWAL framework - 92% accuracy predicting
           user behavior from session/event data)
"""

import logging
from typing import Optional
from datetime import datetime, timedelta
import numpy as np

from .base import BaseModel
from ..config import CONFIG
from .. import gpu_utils

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class FunnelPredictor(BaseModel):
    """
    Model 2: Funnel Drop-off Prediction
    Predicts which visitors will drop off at each step of a funnel.
    
    Features:
    - Session metadata (browser, OS, device, country)
    - Performance metrics (LCP, CLS, INP at each step)
    - Previous step completion time
    - UTM parameters
    - Referrer type
    
    GPU: Uses XGBoost GPU accelerated training (hist tree method)
    CPU: Falls back to XGBoost CPU training
    """
    
    def __init__(self):
        super().__init__(name="funnel_predictor", version="1.0.0")
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = []
    
    def _extract_features(self, session_data: list[dict]) -> np.ndarray:
        """
        Extract feature vectors from session data.
        
        Expected session_data format: list of dicts with keys:
        - browser, os, device, screen, language, country
        - avg_lcp, avg_cls, avg_inp (performance)
        - step_duration, referrer_domain, utm_source, utm_medium
        - session_duration, pages_per_session, unique_pages
        - hour_of_day, day_of_week
        - is_returning_visitor
        """
        # Collect all unique categorical values first
        cat_fields = ['browser', 'os', 'device', 'screen', 'language',
                     'country', 'region', 'utm_source', 'utm_medium']
        if not self.label_encoders and SKLEARN_AVAILABLE:
            for field in cat_fields:
                all_vals = set()
                for session in session_data:
                    all_vals.add(str(session.get(field, 'unknown')))
                all_vals.add('unknown')
                self.label_encoders[field] = LabelEncoder()
                self.label_encoders[field].fit(sorted(all_vals))
        
        features = []
        
        for session in session_data:
            row = []
            
            # Numeric features (preserve as-is)
            numeric_fields = [
                'avg_lcp', 'avg_cls', 'avg_inp',
                'step_duration', 'session_duration',
                'pages_per_session', 'unique_pages',
                'hour_of_day', 'day_of_week',
                'window_completion_pct',
            ]
            for field in numeric_fields:
                val = session.get(field, 0)
                row.append(float(val if val is not None else 0))
            
            # Categorical features (label encode)
            for field in cat_fields:
                val = str(session.get(field, 'unknown'))
                if field in self.label_encoders:
                    try:
                        encoded = self.label_encoders[field].transform([val])[0]
                    except ValueError:
                        encoded = 0
                else:
                    encoded = 0
                row.append(float(encoded))
            
            # Boolean features
            bool_fields = ['is_returning_visitor', 'has_performance_issues']
            for field in bool_fields:
                val = session.get(field, 0)
                row.append(float(val if val is not None else 0))
            
            features.append(row)
        
        return np.array(features)
    
    def train(
        self,
        sessions: list[dict],
        labels: list[int],
        use_gpu: bool = True,
    ):
        """
        Train XGBoost funnel predictor.
        
        Args:
            sessions: List of session feature dicts
            labels: Binary labels (0 = dropped off, 1 = continued)
            use_gpu: Use GPU acceleration
        """
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost is required for FunnelPredictor. "
                            "Install with: pip install xgboost")
        
        logger.info(f"Training FunnelPredictor on {len(sessions)} sessions")
        
        # Extract features
        X = self._extract_features(sessions)
        y = np.array(labels)
        
        # Scale numeric features
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        
        # Determine device for XGBoost
        device = 'cuda' if (use_gpu and gpu_utils.GPU_COUNT > 0) else 'cpu'
        if use_gpu and gpu_utils.GPU_COUNT > 0:
            # Use Tesla P40 or which ever GPU is free
            gpu_id = gpu_utils.BEST_GPU or 0
            logger.info(f"Training XGBoost on GPU {gpu_id} ({device})")
        
        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist' if use_gpu and gpu_utils.GPU_COUNT > 0 else 'auto',
            device=device,
            eval_metric=['logloss', 'auc'],
            early_stopping_rounds=20,
            random_state=42,
        )
        
        # Split for validation (handle small datasets)
        if len(X) >= 10:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, 
                stratify=y if len(set(y)) > 1 else None
            )
        else:
            X_train, X_val, y_train, y_val = X, X, y, y
        
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        if len(X) >= 10:
            val_score = self.model.score(X_val, y_val)
        else:
            val_score = train_score
        
        self.metadata['train_accuracy'] = float(train_score)
        self.metadata['n_estimators'] = self.model.n_estimators
        self.metadata['device'] = device
        
        logger.info(f"FunnelPredictor trained: train_acc={train_score:.4f}, "
                   f"val_acc={val_score:.4f}")
        
        self.is_trained = True
    
    def predict(
        self,
        session: dict,
        threshold: float = 0.5
    ) -> dict:
        """
        Predict drop-off probability for a single session at a funnel step.
        
        Returns:
            dict with 'will_continue' (bool), 'probability' (float),
            and 'confidence' (float)
        """
        if self.model is None:
            return {'will_continue': True, 'probability': 0.5, 
                   'confidence': 0.0}
        
        X = self._extract_features([session])
        if self.scaler:
            X = self.scaler.transform(X)
        
        proba = self.model.predict_proba(X)[0]
        continue_prob = float(proba[1])
        
        return {
            'will_continue': continue_prob >= threshold,
            'probability': continue_prob,
            'confidence': abs(continue_prob - 0.5) * 2,
            'threshold': threshold,
        }
    
    def explain(self, session: dict) -> list[dict]:
        """
        Get feature importance for a prediction.
        Shows which factors most influenced the drop-off risk.
        """
        if self.model is None:
            return []
        
        try:
            import shap
            X = self._extract_features([session])
            if self.scaler:
                X = self.scaler.transform(X)
            
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            
            feature_importance = []
            for i, val in enumerate(shap_values[0]):
                feature_importance.append({
                    'feature': self.feature_names[i] if i < len(self.feature_names) else f'f_{i}',
                    'impact': float(val),
                    'abs_impact': abs(float(val)),
                })
            
            return sorted(feature_importance, key=lambda x: -x['abs_impact'])[:10]
        except ImportError:
            # SHAP not available, use built-in feature importance
            importance = self.model.feature_importances_
            return [
                {'feature': f'f_{i}', 'importance': float(v)}
                for i, v in enumerate(importance)
                if v > 0.01
            ][:10]
    
    def to_onnx(self, path: Optional[str] = None) -> str:
        """Export XGBoost model to ONNX"""
        if self.model is None:
            raise RuntimeError("Model not trained")
        
        try:
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            
            n_features = len(self.feature_names) if self.feature_names else 10
            initial_type = [('float_input', FloatTensorType([None, n_features]))]
            onnx_model = convert_sklearn(self.model, initial_types=initial_type)
            
            path = path or CONFIG.paths.funnel_model.replace('.json', '.onnx')
            with open(path, 'wb') as f:
                f.write(onnx_model.SerializeToString())
            
            logger.info(f"Funnel predictor exported to ONNX: {path}")
            return path
        except ImportError:
            logger.warning("skl2onnx not available, saving as XGBoost JSON")
            path = path or CONFIG.paths.funnel_model
            self.model.save_model(path)
            return path
