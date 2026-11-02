import sys
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# Only import engine modules if torch is present
if TORCH_AVAILABLE:
    from red.engines import *

__version__ = "0.1.0"