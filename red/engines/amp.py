"""
red.engines.amp
AMPManager engine for mixed-precision gradient inspection, 3-strike consecutive
overflow tracking, and permanent amp_disabled latching on RedSharedState.
"""

import logging
from typing import Any, Optional
import torch

from ..config import RedConfig
from ..exceptions import RED_COMPILE_WARNING
from ..state import RedSharedState

__all__ = ["AMPManager"]

logger = logging.getLogger("red.engines.amp")


class AMPManager:
    """
    Monitors gradients for NaN/Inf overflows during mixed-precision training.
    After `max_strikes` (default 3) consecutive overflow steps, permanently
    disables AMP and publishes `amp_disabled` to RedSharedState (`event-v1.0`).
    """

    def __init__(
        self,
        config: RedConfig,
        shared_state: RedSharedState,
        *,
        enabled: bool = True,
        max_strikes: int = 3,
    ) -> None:
        if isinstance(max_strikes, bool) or not isinstance(max_strikes, int) or max_strikes < 1:
            raise ValueError("max_strikes must be an int >= 1.")
        self._config = config
        self._shared_state = shared_state
        self._enabled: bool = bool(enabled)
        self._max_strikes: int = max_strikes
        self._consecutive_strikes: int = 0
        self._total_overflows: int = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def consecutive_strikes(self) -> int:
        return self._consecutive_strikes

    @property
    def total_overflows(self) -> int:
        return self._total_overflows

    @staticmethod
    def has_non_finite_gradients(optimizer: torch.optim.Optimizer) -> bool:
        """Inspect all parameter gradients in the optimizer for NaN or Inf."""
        for group in optimizer.param_groups:
            for param in group.get("params", []):
                grad = getattr(param, "grad", None)
                if grad is not None and not bool(torch.isfinite(grad).all().item()):
                    return True
        return False

    def inspect_and_record(
        self,
        optimizer: torch.optim.Optimizer,
        global_step: int,
        *,
        force_overflow: bool = False,
    ) -> bool:
        """
        Inspect optimizer gradients (requires RedState.RUNNING).
        Returns True if gradients are finite and optimizer.step() may proceed.
        Returns False if overflow is detected (step must be skipped).
        """
        self._shared_state.state_machine.require_running("AMPManager.inspect_and_record")

        overflow = force_overflow or self.has_non_finite_gradients(optimizer)
        if not overflow:
            self._consecutive_strikes = 0
            return True

        self._consecutive_strikes += 1
        self._total_overflows += 1

        if self._enabled and self._consecutive_strikes >= self._max_strikes:
            self._enabled = False
            logger.warning(
                "%s: AMP permanently disabled at step %d after %d consecutive overflow strikes.",
                RED_COMPILE_WARNING,
                global_step,
                self._consecutive_strikes,
            )
            self._shared_state.publish(
                "amp_disabled",
                {
                    "strikes": int(self._consecutive_strikes),
                    "step": int(global_step),
                    "reason": "max_consecutive_overflow_strikes",
                },
            )

        return False

    def state_dict(self) -> dict[str, Any]:
        return {
            "enabled": self._enabled,
            "max_strikes": self._max_strikes,
            "consecutive_strikes": self._consecutive_strikes,
            "total_overflows": self._total_overflows,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._enabled = bool(state.get("enabled", self._enabled))
        self._consecutive_strikes = int(state.get("consecutive_strikes", 0))
        self._total_overflows = int(state.get("total_overflows", 0))
