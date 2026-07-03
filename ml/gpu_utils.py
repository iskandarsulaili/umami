"""
GPU Utilities for Umami ML Service
- Auto-detects best available GPU with fallback to CPU
- Manages GPU memory allocation across models
- Supports both CUDA and ONNX runtime providers
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


def detect_best_gpu() -> Optional[int]:
    """
    Detect the best available GPU for ML inference.
    Prefers Tesla P40 (24GB) over RTX 3060 (12GB) for model inference
    since P40 has more memory.
    
    Returns:
        Optional[int]: Best GPU index, or None if no GPU available
    """
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available, falling back to CPU")
        return None
    
    if not torch.cuda.is_available():
        logger.info("No CUDA GPU detected, using CPU")
        return None
    
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        return None
    
    # Find the GPU with most free memory
    best_gpu = 0
    best_memory = 0
    
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory
        # Prefer Tesla P40 (around 24GB) over RTX 3060 (12GB) for inference
        # RTX 3060 might be busy with other workloads
        memory_gb = total_memory / 1e9
        logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)} - {memory_gb:.1f}GB")
        
        if memory_gb > best_memory:
            best_memory = memory_gb
            best_gpu = i
    
    logger.info(f"Selected GPU {best_gpu}: {torch.cuda.get_device_name(best_gpu)}")
    return best_gpu


def get_device(gpu_id: Optional[int] = None) -> torch.device:
    """
    Get the best available torch device.
    
    Args:
        gpu_id: Specific GPU to use, or None for auto-detect
        
    Returns:
        torch.device: Best available device
    """
    if not TORCH_AVAILABLE:
        return torch.device('cpu')
    
    if gpu_id is None:
        gpu_id = detect_best_gpu()
    
    if gpu_id is not None and torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    
    return torch.device('cpu')


def get_onnx_providers() -> list:
    """
    Get ONNX Runtime providers ordered by preference.
    
    Returns:
        list: Provider names for ORT session
    """
    if not ONNX_AVAILABLE:
        return ['CPUExecutionProvider']
    
    available = ort.get_available_providers()
    preferred = []
    
    # GPU providers in order of preference
    gpu_providers = [
        'CUDAExecutionProvider',
        'TensorrtExecutionProvider',
        'AzureExecutionProvider',
    ]
    
    for prov in gpu_providers:
        if prov in available:
            preferred.append(prov)
    
    preferred.append('CPUExecutionProvider')
    
    return preferred


def get_gpu_memory_info(gpu_id: int = 0) -> dict:
    """
    Get GPU memory usage information.
    
    Args:
        gpu_id: GPU index
        
    Returns:
        dict: Memory info with free, used, total in bytes
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return {'free': 0, 'used': 0, 'total': 0, 'device': 'cpu'}
    
    try:
        props = torch.cuda.get_device_properties(gpu_id)
        allocated = torch.cuda.memory_allocated(gpu_id)
        reserved = torch.cuda.memory_reserved(gpu_id)
        
        return {
            'free': props.total_memory - reserved,
            'used': allocated,
            'reserved': reserved,
            'total': props.total_memory,
            'device': torch.cuda.get_device_name(gpu_id),
        }
    except Exception as e:
        logger.error(f"Error getting GPU memory: {e}")
        return {'error': str(e)}


def get_gpu_count() -> int:
    """Get number of available CUDA GPUs."""
    if not TORCH_AVAILABLE:
        return 0
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


# Initialize on import
BEST_GPU = detect_best_gpu()
DEVICE = get_device(BEST_GPU)
ONNX_PROVIDERS = get_onnx_providers()
GPU_COUNT = get_gpu_count()

logger.info(f"ML Device: {DEVICE}")
logger.info(f"ONNX Providers: {ONNX_PROVIDERS}")
logger.info(f"GPU Count: {GPU_COUNT}")
