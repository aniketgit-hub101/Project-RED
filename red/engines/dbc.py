"""
red.engines.dbc
DynamicBatchController engine: handles runtime OOM recovery by halving micro-batch
size, scaling gradient accumulation to preserve effective batch size, publishing
`batch_event` and `vram_critical` onto RedSharedState, and stepping batch size
back up after `config.dbc_reset_interval` stable steps.
"""

from typing import Any, Optional
import torch

from ..config import RedConfig
from ..state import RedSharedState

__all__ = ["DynamicBatchController"]


class DynamicBatchController:
    """
    Coordinates dynamic micro-batch sizing and gradient accumulation steps.
    Publishes `batch_event` on every batch adjustment so EarlyStopper suppresses
    false-positive stopping during batch-size transitions.
    """

    def __init__(
        self,
        config: RedConfig,
        shared_state: RedSharedState,
        *,
        initial_batch_size: int = 32,
    ) -> None:
        if isinstance(initial_batch_size, bool) or not isinstance(initial_batch_size, int) or initial_batch_size < config.min_batch_size:
            raise ValueError(f"initial_batch_size must be an int >= {config.min_batch_size}.")
        self._config = config
        self._shared_state = shared_state
        self._initial_batch_size: int = initial_batch_size
        self._current_batch_size: int = initial_batch_size
        self._steps_since_last_change: int = 0
        self._oom_recoveries: int = 0

    @property
    def initial_batch_size(self) -> int:
        return self._initial_batch_size

    @property
    def current_batch_size(self) -> int:
        return self._current_batch_size

    @property
    def accumulation_steps(self) -> int:
        return max(1, self._initial_batch_size // self._current_batch_size)

    @property
    def oom_recoveries(self) -> int:
        return self._oom_recoveries

    def handle_oom(
        self,
        global_step: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> int:
        """
        Recover from a runtime CUDA/memory OOM at `global_step`.
        Halves `current_batch_size` down to `config.min_batch_size`, clears
        gradients and CUDA cache, and publishes `batch_event` (or `vram_critical`
        if already at `min_batch_size`).
        """
        self._shared_state.state_machine.require_running("DynamicBatchController.handle_oom")

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        min_bs = self._config.min_batch_size
        if self._current_batch_size <= min_bs:
            self._shared_state.publish(
                "vram_critical",
                {
                    "active": True,
                    "accumulation_steps": self.accumulation_steps,
                    "step": int(global_step),
                },
            )
            raise RuntimeError(
                f"VRAM_PRESSURE_CRITICAL: OOM at min_batch_size={min_bs} (step {global_step})."
            )

        new_bs = max(min_bs, self._current_batch_size // 2)
        self._current_batch_size = new_bs
        self._steps_since_last_change = 0
        self._oom_recoveries += 1

        self._shared_state.publish(
            "batch_event",
            {
                "batch_size": int(self._current_batch_size),
                "accumulation_steps": int(self.accumulation_steps),
                "step": int(global_step),
                "scaled_direction": "down",
            },
        )
        return self._current_batch_size

    def record_successful_step(self, global_step: int) -> int:
        """
        Record a finite, OOM-free step. If `current_batch_size < initial_batch_size`
        and `dbc_reset_interval` stable steps have elapsed, step batch size up by 2x.
        """
        self._shared_state.state_machine.require_running(
            "DynamicBatchController.record_successful_step"
        )
        self._steps_since_last_change += 1

        if (
            self._current_batch_size < self._initial_batch_size
            and self._steps_since_last_change >= self._config.dbc_reset_interval
        ):
            self._current_batch_size = min(
                self._initial_batch_size, self._current_batch_size * 2
            )
            self._steps_since_last_change = 0
            self._shared_state.publish(
                "batch_event",
                {
                    "batch_size": int(self._current_batch_size),
                    "accumulation_steps": int(self.accumulation_steps),
                    "step": int(global_step),
                    "scaled_direction": "up",
                },
            )
            if self._current_batch_size > self._config.min_batch_size:
                self._shared_state.publish(
                    "vram_critical",
                    {
                        "active": False,
                        "accumulation_steps": int(self.accumulation_steps),
                        "step": int(global_step),
                    },
                )
        return self._current_batch_size

    def state_dict(self) -> dict[str, Any]:
        return {
            "initial_batch_size": self._initial_batch_size,
            "current_batch_size": self._current_batch_size,
            "steps_since_last_change": self._steps_since_last_change,
            "oom_recoveries": self._oom_recoveries,
        }
