"""
red.engines.optimizer
TorchOptimizer runtime engine wrapping native PyTorch optimizers with
RedState execution gating, hook conflict detection, and AMP strike protection.
"""

import time
from typing import Any, Callable, Optional
import torch

from ..config import RedConfig
from ..exceptions import RedHookConflictError
from ..state import RedSharedState, RedState
from .amp import AMPManager

__all__ = ["TorchOptimizer"]

_HOOK_ATTR = "_red_active_optimizer_hook"


class TorchOptimizer:
    """
    Core Project RED optimizer wrapper enforcing RedState.RUNNING before
    any parameter update and coordinating AMP overflow strikes.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: Optional[RedConfig] = None,
        shared_state: Optional[RedSharedState] = None,
        *,
        amp_manager: Optional[AMPManager] = None,
    ) -> None:
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                f"optimizer must be a torch.optim.Optimizer instance, got {type(optimizer).__name__}."
            )
        if getattr(optimizer, _HOOK_ATTR, False):
            raise RedHookConflictError(
                "Optimizer already has an active TorchOptimizer hook attached."
            )

        self._optimizer = optimizer
        self._config = config if config is not None else RedConfig()
        self._shared_state = shared_state if shared_state is not None else RedSharedState()
        self._amp_manager = (
            amp_manager
            if amp_manager is not None
            else AMPManager(self._config, self._shared_state)
        )
        self._global_step: int = 0
        self._skipped_steps: int = 0
        self._last_step_duration_s: float = 0.0
        self._attached: bool = True
        setattr(self._optimizer, _HOOK_ATTR, True)

        if self._shared_state.state_machine.state is RedState.UNCONFIGURED:
            self._shared_state.state_machine.transition_to(RedState.CONFIGURED)

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def config(self) -> RedConfig:
        return self._config

    @property
    def shared_state(self) -> RedSharedState:
        return self._shared_state

    @property
    def amp_manager(self) -> AMPManager:
        return self._amp_manager

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def skipped_steps(self) -> int:
        return self._skipped_steps

    @property
    def last_step_duration_s(self) -> float:
        return self._last_step_duration_s

    @property
    def is_attached(self) -> bool:
        return self._attached

    def start(self) -> RedState:
        """Transition state machine from CONFIGURED or PAUSED to RUNNING."""
        return self._shared_state.state_machine.transition_to(RedState.RUNNING)

    def pause(self) -> RedState:
        """Transition state machine from RUNNING to PAUSED."""
        return self._shared_state.state_machine.transition_to(RedState.PAUSED)

    def detach(self) -> None:
        """Idempotently remove the hook marker from the underlying optimizer."""
        if self._attached:
            setattr(self._optimizer, _HOOK_ATTR, False)
            self._attached = False

    def sanitize(self, *, internal: bool = False) -> bool:
        """Detach optimizer hooks and transition state machine to SANITIZED."""
        self.detach()
        return self._shared_state.state_machine.sanitize(internal=internal)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._optimizer.zero_grad(set_to_none=set_to_none)

    def step(
        self,
        closure: Optional[Callable[[], float]] = None,
        *,
        force_overflow: bool = False,
    ) -> bool:
        """
        Execute a guarded optimization step.
        Requires RedState.RUNNING. Returns True if weights were updated,
        or False if the step was skipped due to gradient overflow.
        """
        self._shared_state.state_machine.require_running("TorchOptimizer.step")

        t0 = time.perf_counter()
        next_step = self._global_step + 1

        is_finite = self._amp_manager.inspect_and_record(
            self._optimizer,
            next_step,
            force_overflow=force_overflow,
        )
        if not is_finite:
            self._skipped_steps += 1
            self._optimizer.zero_grad(set_to_none=True)
            self._last_step_duration_s = time.perf_counter() - t0
            return False

        if closure is not None:
            self._optimizer.step(closure)
        else:
            self._optimizer.step()

        self._global_step = next_step
        self._last_step_duration_s = time.perf_counter() - t0
        return True

    def state_dict(self) -> dict[str, Any]:
        return {
            "optimizer": self._optimizer.state_dict(),
            "amp_manager": self._amp_manager.state_dict(),
            "global_step": self._global_step,
            "skipped_steps": self._skipped_steps,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if "optimizer" in state_dict:
            self._optimizer.load_state_dict(state_dict["optimizer"])
        if "amp_manager" in state_dict:
            self._amp_manager.load_state_dict(state_dict["amp_manager"])
        self._global_step = int(state_dict.get("global_step", self._global_step))
        self._skipped_steps = int(state_dict.get("skipped_steps", self._skipped_steps))
