try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from .optimizer import TorchOptimizer
    from .amp import AMPManager

__all__ = ["TorchOptimizer", "AMPManager"]