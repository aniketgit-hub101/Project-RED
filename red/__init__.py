try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from red.engines import *

__version__ = "0.1.0"
__all__ = ["TORCH_AVAILABLE", "__version__"]