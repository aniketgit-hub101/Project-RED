"""
red.engines
Runtime optimization, mixed-precision, and dynamic batching engines.
"""

from .stubs import (
    EarlyStopperMock,
    MOCK_CONTRACT_VERSION,
    MagnitudePrunerStub,
    MemoryProfilerStub,
    STUB_CONTRACT_VERSION,
)

try:
    import torch as _torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from . import amp as _amp
    AMPManager = _amp.AMPManager
    RedCheckpointError = getattr(_amp, "RedCheckpointError", None)
    from .dbc import DynamicBatchController
    from .optimizer import TorchOptimizer
else:
    AMPManager = None  # type: ignore[assignment,misc]
    RedCheckpointError = None  # type: ignore[assignment,misc]
    DynamicBatchController = None  # type: ignore[assignment,misc]
    TorchOptimizer = None  # type: ignore[assignment,misc]

__all__ = [
    "AMPManager",
    "RedCheckpointError",
    "DynamicBatchController",
    "TorchOptimizer",
    "EarlyStopperMock",
    "MemoryProfilerStub",
    "MagnitudePrunerStub",
    "MOCK_CONTRACT_VERSION",
    "STUB_CONTRACT_VERSION",
]