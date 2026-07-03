"""
Contextual Bandit for Real-Time Content Recommendations - Model 7
Online learning: adapts recommendations based on session context and user feedback.

Uses LinUCB (linear upper confidence bound) algorithm:
- Context vector: current page embedding, device, browser, country, time, session depth
- Arms: top-K recommendable pages
- Reward: click (positive) or scroll-away / exit (negative)

Reference: arxiv:2402.17129 (side-info-driven SBR), arxiv:2205.04181
"""

import os
import json
import logging
import pickle
import hashlib
from typing import Optional
from datetime import datetime
from collections import defaultdict

import numpy as np

from .base import BaseModel
from ..config import CONFIG

logger = logging.getLogger(__name__)

try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class LinUCB:
    """
    Linear Upper Confidence Bound bandit algorithm.
    
    For each arm (recommendable page), maintains:
    - A: design matrix (d x d)
    - b: response vector (d x 1)
    - theta: estimated coefficients
    
    At each step, selects arm with highest upper confidence bound:
    arm = argmax(theta^T * x + alpha * sqrt(x^T * A^{-1} * x))
    """
    
    def __init__(self, n_features: int, alpha: float = 0.5):
        self.n_features = n_features
        self.alpha = alpha
        self.arms: dict[str, dict] = {}
    
    def _init_arm(self, arm_id: str):
        if arm_id not in self.arms:
            self.arms[arm_id] = {
                'A': np.eye(self.n_features),
                'b': np.zeros(self.n_features),
                'theta': np.zeros(self.n_features),
                'n_pulls': 0,
                'n_rewards': 0,
            }
    
    def select_arm(self, context: np.ndarray, available_arms: list[str]) -> tuple[str, float]:
        """Select arm with highest UCB score."""
        best_arm = None
        best_score = -float('inf')
        scores = {}
        
        for arm_id in available_arms:
            self._init_arm(arm_id)
            arm = self.arms[arm_id]
            
            A_inv = np.linalg.inv(arm['A'])
            theta = A_inv @ arm['b']
            
            # UCB score: exploitation + exploration
            exploitation = theta @ context
            exploration = self.alpha * np.sqrt(context @ A_inv @ context)
            score = exploitation + exploration
            
            scores[arm_id] = float(score)
            
            if score > best_score:
                best_score = score
                best_arm = arm_id
        
        return best_arm, best_score
    
    def update(self, arm_id: str, context: np.ndarray, reward: float):
        """Update arm parameters with observed reward."""
        self._init_arm(arm_id)
        arm = self.arms[arm_id]
        arm['A'] += np.outer(context, context)
        arm['b'] += reward * context
        arm['n_pulls'] += 1
        arm['n_rewards'] += reward
        arm['theta'] = np.linalg.inv(arm['A']) @ arm['b']
    
    def get_arm_stats(self, arm_id: str) -> dict:
        if arm_id not in self.arms:
            return {'n_pulls': 0, 'n_rewards': 0, 'ctr': 0.0}
        arm = self.arms[arm_id]
        return {
            'n_pulls': arm['n_pulls'],
            'n_rewards': arm['n_rewards'],
            'ctr': arm['n_rewards'] / max(1, arm['n_pulls']),
        }


class ContextualBandit(BaseModel):
    """
    Model 7: Contextual Bandit for Real-Time Recommendations.
    
    Online learning: updates after every user interaction.
    No batch training needed — learns continuously.
    """
    
    def __init__(self):
        super().__init__(name="contextual_bandit", version="1.0.0")
        self.bandit = None
        self.scaler = None
        self.n_features = 0
        self.feature_names = []
    
    def _build_context(self, session: dict) -> np.ndarray:
        """Build context vector from session state."""
        features = []
        
        # Current page embedding (simplified: deterministic hash)
        current_page = session.get('current_page', '/')
        features.append(abs(hashlib.md5(current_page.encode()).digest()[0]) % 1000 / 1000.0)
        
        # Session depth
        features.append(min(session.get('session_depth', 0) / 50, 1.0))
        
        # Time features
        hour = session.get('hour', 12)
        features.append(hour / 24.0)
        features.append(1 if session.get('is_weekend', False) else 0)
        
        # Device
        features.append(1 if session.get('device') == 'desktop' else 0)
        features.append(1 if session.get('device') == 'mobile' else 0)
        
        # Browser
        browser = session.get('browser', '')
        features.append(1 if browser == 'Chrome' else 0)
        features.append(1 if browser == 'Firefox' else 0)
        features.append(1 if browser == 'Safari' else 0)
        
        # Country (simplified, deterministic)
        country = session.get('country', '')
        features.append(abs(hashlib.md5(country.encode()).digest()[0]) % 100 / 100.0)
        
        # Session engagement
        features.append(min(session.get('pages_seen', 1) / 20, 1.0))
        features.append(min(session.get('time_on_site', 0) / 600, 1.0))
        
        # Has cart/checkout
        features.append(1 if session.get('has_cart', False) else 0)
        features.append(1 if session.get('has_checkout', False) else 0)
        
        return np.array(features)
    
    def train(self, sessions: list[dict] = None):
        """Initialize the bandit. No batch training needed."""
        self.n_features = 16  # matches _build_context output
        self.bandit = LinUCB(n_features=self.n_features, alpha=0.5)
        self.is_trained = True
        logger.info(f"ContextualBandit initialized with {self.n_features} features")
    
    def recommend(self, session: dict, available_pages: list[str], top_k: int = 5) -> list[dict]:
        """
        Recommend pages using LinUCB bandit.
        
        Args:
            session: Current session context dict
            available_pages: List of recommendable page URLs
            top_k: Number of recommendations to return
            
        Returns:
            List of {page, score, arm_stats}
        """
        if self.bandit is None:
            return [{'page': p, 'score': 0.5, 'arm_stats': {}} for p in available_pages[:top_k]]
        
        context = self._build_context(session)
        
        results = []
        for page in available_pages:
            _, score = self.bandit.select_arm(context, [page])
            stats = self.bandit.get_arm_stats(page)
            results.append({
                'page': page,
                'score': float(score),
                'arm_stats': stats,
            })
        
        results.sort(key=lambda x: -x['score'])
        return results[:top_k]
    
    def record_reward(self, page: str, session: dict, reward: float):
        """
        Record user feedback for a recommendation.
        
        Args:
            page: The recommended page URL
            session: Session context at time of recommendation
            reward: 1.0 for click, 0.5 for scroll, 0.0 for exit/bounce
        """
        if self.bandit is None:
            return
        
        context = self._build_context(session)
        self.bandit.update(page, context, reward)
        logger.debug(f"Bandit updated: {page}, reward={reward}")
    
    def get_arm_stats(self, page: str = None) -> dict:
        """Get statistics for all arms or a specific page."""
        if self.bandit is None:
            return {}
        
        if page:
            return self.bandit.get_arm_stats(page)
        
        return {
            arm_id: self.bandit.get_arm_stats(arm_id)
            for arm_id in self.bandit.arms
        }
    
    def predict(self, data: dict = None) -> dict:
        """Return bandit stats as prediction."""
        return self.get_arm_stats()
