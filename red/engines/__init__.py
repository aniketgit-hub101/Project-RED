"""
red.engines
Engine subsystem package exposing Phase 0 contract stubs/mocks and Stage S3
PyTorch runtime engines (`AMPManager`, `TorchOptimizer`, `DynamicBatchController`).
"""

from .amp import AMPManager
from .dbc import DynamicBatchController
from .optimizer import TorchOptimizer
from .stubs import (
    EarlyStopperMock,
    MOCK_CONTRACT_VERSION,
    MagnitudePrunerStub,
    MemoryProfilerStub,
    STUB_CONTRACT_VERSION,
)

__all__ = [
    "STUB_CONTRACT_VERSION",
    "MOCK_CONTRACT_VERSION",
    "MemoryProfilerStub",
    "MagnitudePrunerStub",
    "EarlyStopperMock",
    "AMPManager",
    "TorchOptimizer",
    "DynamicBatchController",
]
