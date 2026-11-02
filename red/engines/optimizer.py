import time
from typing import Any, Callable, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from ..config import RedConfig
from ..exceptions import RedHookConflictError
from ..state import RedSharedState, RedState
from .amp import AMPManager

__all__ = ["TorchOptimizer"]

HOOK_ATTR = "_red_active_optimizer_hook"

class TorchOptimizer:
    """
    Core Project RED optimizer wrapper enforcing RedState.RUNNING before
    any parameter update and coordinating AMP overflow strikes.
    """

    def __init__(
        self,
        optimizer,
        config: Optional['RedConfig'] = None,
        shared_state: Optional['RedSharedState'] = None,
        *,
        amp_manager: Optional['AMPManager'] = None,
    ) -> None:
        if TORCH_AVAILABLE and torch is not None and not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                f"Optimizer must be a torch.optim.Optimizer instance, got {type(optimizer).__name__}."
            )
        
        if getattr(optimizer, HOOK_ATTR, False):
            raise RedHookConflictError(
                "Optimizer already has an active TorchOptimizer hook attached."
            )
        
        self.optimizer = optimizer
        setattr(self.optimizer, HOOK_ATTR, True)
        
        # Register PyTorch step pre-hook
        if hasattr(self.optimizer, "register_step_pre_hook"):
            self._pre_hook_handle = self.optimizer.register_step_pre_hook(self._pre_step_hook)
        else:
            self._pre_hook_handle = None

        self.config = config if config is not None else RedConfig()
        self.shared_state = shared_state if shared_state is not None else RedSharedState()
        self.amp_manager = amp_manager

    def _pre_step_hook(self, optimizer, args, kwargs):
        """Detach or sanitize gradients safely prior to step execution."""
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    p.grad = p.grad.detach()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict:
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.optimizer.load_state_dict(state_dict)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """
        Performs a single optimization step.
        """
        if self.shared_state:
            self.shared_state.state_machine.require_running("TorchOptimizer.step")

        # Gradient sanitization is handled natively by _pre_step_hook.
        # Execute the underlying optimizer step cleanly.
        return self.optimizer.step(closure=closure)