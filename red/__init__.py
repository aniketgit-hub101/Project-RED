"""
red
Project RED runtime resilience, mixed-precision, and dynamic batching package.
"""

from .config import RedConfig
from . import exceptions as _exceptions

# Import from the exceptions module while preserving backwards-compatible symbols.
# Some static analyzers cannot resolve names created dynamically or re-exported in
# this package, so keep the public API explicit here.
from .exceptions import (
    RED_COMPILE_WARNING,
    VRAM_PRESSURE_CRITICAL,
    RedConfigError,
    RedHookConflictError,
    RedStateError,
)

RedException = getattr(_exceptions, "RedException", Exception)
RedSecurityError = getattr(
    _exceptions,
    "RedSecurityError",
    type("RedSecurityError", (RedException,), {}),
)

RedContractError = getattr(
    _exceptions,
    "RedContractError",
    type("RedContractError", (RedException,), {}),
)
from .state import EVENT_SCHEMA_VERSION, RedSharedState, RedState, RedStateMachine

try:
    import torch as _torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .engines import (
    AMPManager,
    DynamicBatchController,
    EarlyStopperMock,
    MOCK_CONTRACT_VERSION,
    MagnitudePrunerStub,
    MemoryProfilerStub,
    STUB_CONTRACT_VERSION,
    TorchOptimizer,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "TORCH_AVAILABLE",
    "RedConfig",
    "RedException",
    "RedConfigError",
    "RedStateError",
    "RedHookConflictError",
    "RedContractError",
    "RedSecurityError",
    "RED_COMPILE_WARNING",
    "VRAM_PRESSURE_CRITICAL",
    "EVENT_SCHEMA_VERSION",
    "RedState",
    "RedStateMachine",
    "RedSharedState",
    "AMPManager",
    "DynamicBatchController",
    "TorchOptimizer",
    "EarlyStopperMock",
    "MemoryProfilerStub",
    "MagnitudePrunerStub",
    "MOCK_CONTRACT_VERSION",
    "STUB_CONTRACT_VERSION",
]