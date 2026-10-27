"""
red.engines
Engine subsystem package exposing Phase 0 contract stubs and mocks.
"""

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
]
