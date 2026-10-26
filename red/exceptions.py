"""
red.exceptions
Public exception hierarchy and log markers per Master §6.11.
"""

__all__ = [
    "RedCompatibilityError",
    "RedStateError",
    "RedHookConflictError",
    "RedConfigError",
    "RedCheckpointError",
    "RED_COMPILE_WARNING",
    "VRAM_PRESSURE_CRITICAL",
    "PREPRUNING_CHECKPOINT_MISSING",
]


class RedCompatibilityError(Exception):
    """Raised when hardware, backend, or model compatibility checks fail."""


class RedStateError(Exception):
    """Raised on invalid RedState transitions or repeated public sanitize() calls."""


class RedHookConflictError(Exception):
    """Raised when competing runtime hooks or instrumentation conflict."""


class RedConfigError(Exception):
    """Raised when RedConfig parameters or environment overrides are invalid."""


class RedCheckpointError(Exception):
    """Raised when checkpoint saving, loading, or verification fails."""


# §6.11 Log Markers (logged, not raised)
RED_COMPILE_WARNING: str = "RED_COMPILE_WARNING"
VRAM_PRESSURE_CRITICAL: str = "VRAM_PRESSURE_CRITICAL"
PREPRUNING_CHECKPOINT_MISSING: str = "PREPRUNING_CHECKPOINT_MISSING"
