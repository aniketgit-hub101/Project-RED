"""
red
Public package exports per Master §6.8, AR-02.6, and Commandment 13.
"""

from .config import RedConfig
from .exceptions import (
    PREPRUNING_CHECKPOINT_MISSING,
    RED_COMPILE_WARNING,
    VRAM_PRESSURE_CRITICAL,
    RedCheckpointError,
    RedCompatibilityError,
    RedConfigError,
    RedHookConflictError,
    RedStateError,
)
from .state import RedSharedState, RedState, RedStateMachine, status

__all__ = [
    "RedCompatibilityError",
    "RedStateError",
    "RedHookConflictError",
    "RedConfigError",
    "RedCheckpointError",
    "RED_COMPILE_WARNING",
    "VRAM_PRESSURE_CRITICAL",
    "PREPRUNING_CHECKPOINT_MISSING",
    "RedConfig",
    "RedState",
    "RedStateMachine",
    "RedSharedState",
    "status",
]
