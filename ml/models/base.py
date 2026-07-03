"""
Base ML Model Class for Umami
Common interface for all models with GPU/CPU support
"""

import os
import json
import pickle
import logging
from abc import ABC, abstractmethod
from typing import Optional, Any
from pathlib import Path

from ..config import CONFIG
from .. import gpu_utils

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all ML models"""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.device = gpu_utils.DEVICE
        self.is_trained = False
        self.metadata: dict = {}
    
    @abstractmethod
    def train(self, *args, **kwargs) -> Any:
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, *args, **kwargs) -> Any:
        """Run inference"""
        pass
    
    def save(self, path: Optional[str] = None) -> str:
        """Save model to disk"""
        path = path or str(Path(CONFIG.paths.base_dir) / f"{self.name}.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'model': self,
            'metadata': {
                'name': self.name,
                'version': self.version,
                'device': str(self.device),
                'is_trained': self.is_trained,
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model {self.name} saved to {path}")
        return path
    
    def load(self, path: Optional[str] = None) -> 'BaseModel':
        """Load model from disk"""
        path = path or str(Path(CONFIG.paths.base_dir) / f"{self.name}.pkl")
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.__dict__.update(model_data['model'].__dict__)
        self.metadata = model_data['metadata']
        self.is_trained = True
        
        logger.info(f"Model {self.name} loaded from {path}")
        return self
    
    @property
    def info(self) -> dict:
        return {
            'name': self.name,
            'version': self.version,
            'device': str(self.device),
            'is_trained': self.is_trained,
            'gpu_available': gpu_utils.GPU_COUNT > 0,
            'gpu_count': gpu_utils.GPU_COUNT,
            'onnx_providers': gpu_utils.ONNX_PROVIDERS,
            **self.metadata,
        }
    
    def to_onnx(self, path: Optional[str] = None) -> str:
        """Export model to ONNX format for production inference.
        Subclasses should override if they support ONNX export."""
        raise NotImplementedError(f"{self.name} does not support ONNX export")
