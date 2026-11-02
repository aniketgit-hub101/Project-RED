from importlib import import_module

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from .optimizer import TorchOptimizer
    from .amp import AMPManager
    from .dbc import DynamicBatchController

try:
    _stubs = import_module("..stubs", package=__name__)
    EarlyStopperMock = _stubs.EarlyStopperMock
    MemoryProfilerStub = _stubs.MemoryProfilerStub
    MagnitudePrunerStub = _stubs.MagnitudePrunerStub
except (ImportError, AttributeError):
    EarlyStopperMock = MemoryProfilerStub = MagnitudePrunerStub = None

__all__ = [
    "TorchOptimizer", 
    "AMPManager", 
    "DynamicBatchController",
    "EarlyStopperMock",
    "MemoryProfilerStub",
    "MagnitudePrunerStub"
]