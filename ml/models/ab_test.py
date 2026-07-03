"""
A/B Testing Framework for ML Recommendations - Model 8
Statistical experiment framework for comparing recommendation strategies.

Supports:
- A/B/n tests: compare multiple recommendation strategies simultaneously
- Statistical significance via chi-squared test
- Automatic experiment assignment (consistent per visitor/session)
- Real-time dashboard of experiment results

Reference: Standard A/B testing methodology
"""

import os
import json
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Optional
from collections import defaultdict

import numpy as np

from .base import BaseModel
from ..config import CONFIG

logger = logging.getLogger(__name__)

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class ABTestExperiment:
    """Single A/B test experiment definition."""
    
    def __init__(
        self,
        experiment_id: str,
        name: str,
        variants: list[dict],
        traffic_fraction: float = 1.0,
        min_sample_size: int = 100,
    ):
        self.experiment_id = experiment_id
        self.name = name
        self.variants = variants  # [{'id': 'control', 'config': {...}}, {'id': 'treatment_a', 'config': {...}}]
        self.traffic_fraction = traffic_fraction
        self.min_sample_size = min_sample_size
        self.results: dict[str, dict] = {}
        self.is_active = True
        self.created_at = datetime.utcnow()
    
    def assign_variant(self, visitor_id: str) -> str:
        """Deterministically assign a visitor to a variant."""
        if not self.is_active:
            return 'control'
        
        # Hash visitor_id to get consistent assignment
        h = hashlib.md5(f"{self.experiment_id}:{visitor_id}".encode()).hexdigest()
        bucket = int(h[:8], 16) % 100000
        
        # Check if visitor is in the experiment
        if bucket / 100000 > self.traffic_fraction:
            return 'control'
        
        # Assign to variant based on bucket
        n_variants = len(self.variants)
        variant_idx = bucket % n_variants
        return self.variants[variant_idx]['id']
    
    def record_result(self, variant_id: str, converted: bool):
        """Record a conversion or non-conversion for a variant."""
        if variant_id not in self.results:
            self.results[variant_id] = {'n': 0, 'conversions': 0, 'conversion_rate': 0.0}
        
        self.results[variant_id]['n'] += 1
        if converted:
            self.results[variant_id]['conversions'] += 1
        self.results[variant_id]['conversion_rate'] = (
            self.results[variant_id]['conversions'] / self.results[variant_id]['n']
        )
    
    def get_significance(self) -> dict:
        """Calculate statistical significance using chi-squared test."""
        if not HAS_SCIPY:
            return {'significant': False, 'p_value': 1.0, 'error': 'scipy not available'}
        
        if len(self.results) < 2:
            return {'significant': False, 'p_value': 1.0, 'error': 'not enough data'}
        
        # Build contingency table
        variants = list(self.results.keys())
        control = self.results.get('control', {'n': 0, 'conversions': 0})
        
        results = []
        for vid in variants:
            r = self.results[vid]
            if r['n'] < self.min_sample_size:
                continue
            
            non_conversions = r['n'] - r['conversions']
            control_non = control['n'] - control['conversions']
            
            # Chi-squared test
            obs = np.array([[r['conversions'], non_conversions],
                           [control['conversions'], control_non]])
            
            try:
                chi2, p_value, dof, expected = stats.chi2_contingency(obs)
                lift = (r['conversion_rate'] / control['conversion_rate'] - 1) * 100 if control['conversion_rate'] > 0 else 0
                
                results.append({
                    'variant_id': vid,
                    'n': r['n'],
                    'conversions': r['conversions'],
                    'conversion_rate': r['conversion_rate'],
                    'lift_pct': lift,
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                })
            except Exception as e:
                logger.warning(f"Chi-squared test failed: {e}")
        
        return {
            'experiment_id': self.experiment_id,
            'name': self.name,
            'is_active': self.is_active,
            'total_observations': sum(r['n'] for r in self.results.values()),
            'variants': results,
            'winner': max(results, key=lambda r: r['conversion_rate']) if results else None,
        }


class ABTestFramework(BaseModel):
    """
    Model 8: A/B Testing Framework for Recommendations.
    Manages experiments, assigns visitors, tracks conversions.
    """
    
    def __init__(self):
        super().__init__(name="ab_test_framework", version="1.0.0")
        self.experiments: dict[str, ABTestExperiment] = {}
    
    def train(self, data: list = None):
        """No training needed — framework is rule-based."""
        self.is_trained = True
        logger.info("ABTestFramework initialized")
    
    def create_experiment(
        self,
        experiment_id: str,
        name: str,
        variants: list[dict],
        traffic_fraction: float = 1.0,
        min_sample_size: int = 100,
    ) -> dict:
        """Create a new A/B test experiment."""
        exp = ABTestExperiment(
            experiment_id=experiment_id,
            name=name,
            variants=variants,
            traffic_fraction=traffic_fraction,
            min_sample_size=min_sample_size,
        )
        self.experiments[experiment_id] = exp
        logger.info(f"Created experiment: {name} ({len(variants)} variants)")
        return {
            'experiment_id': experiment_id,
            'name': name,
            'variants': [v['id'] for v in variants],
            'traffic_fraction': traffic_fraction,
        }
    
    def assign(self, experiment_id: str, visitor_id: str) -> str:
        """Assign a visitor to a variant."""
        exp = self.experiments.get(experiment_id)
        if not exp:
            return 'control'
        return exp.assign_variant(visitor_id)
    
    def record(self, experiment_id: str, variant_id: str, converted: bool):
        """Record a conversion event."""
        exp = self.experiments.get(experiment_id)
        if exp:
            exp.record_result(variant_id, converted)
    
    def get_results(self, experiment_id: str = None) -> dict:
        """Get experiment results with statistical significance."""
        if experiment_id:
            exp = self.experiments.get(experiment_id)
            if not exp:
                return {'error': 'experiment not found'}
            return exp.get_significance()
        
        return {
            eid: exp.get_significance()
            for eid, exp in self.experiments.items()
        }
    
    def stop_experiment(self, experiment_id: str):
        """Stop an experiment."""
        exp = self.experiments.get(experiment_id)
        if exp:
            exp.is_active = False