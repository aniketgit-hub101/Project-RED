import logging
from typing import Optional, Callable, Any

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from ..config import RedConfig
from ..state import RedSharedState
from .amp import AMPManager

__all__ = ["TorchOptimizer"]
logger = logging.getLogger("red.optimizer")

class TorchOptimizer:
    def __init__(
        self,
        optimizer: Any,
        config: Optional['RedConfig'] = None,
        shared_state: Optional['RedSharedState'] = None,
        *,
        amp_manager: Optional['AMPManager'] = None,
    ) -> None:
        if TORCH_AVAILABLE:
            assert torch is not None
            if not isinstance(optimizer, torch.optim.Optimizer):
                raise TypeError("Optimizer must be a torch.optim.Optimizer instance.")
        
        self.optimizer = optimizer
        self.config = config if config is not None else RedConfig()
        self.shared_state = shared_state if shared_state is not None else RedSharedState()
        self.amp_manager = amp_manager

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
        if self.shared_state:
            self.shared_state.state_machine.require_running("TorchOptimizer.step")

        # AMP NaN-skip logic
        if self.amp_manager and getattr(self.amp_manager, 'enabled', False):
            is_finite = True
            if TORCH_AVAILABLE:
                assert torch is not None
                for group in self.optimizer.param_groups:
                    for p in group["params"]:
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            is_finite = False
                            break
                    if not is_finite:
                        break
            
            if not is_finite:
                setattr(
                    self.amp_manager,
                    "consecutive_strikes",
                    getattr(self.amp_manager, "consecutive_strikes", 0) + 1,
                )
                setattr(
                    self.amp_manager,
                    "total_overflows",
                    getattr(self.amp_manager, "total_overflows", 0) + 1,
                )
                logger.warning("Corrupted gradients detected (NaN/Inf). Skipping optimizer step.")
                return None
            else:
                setattr(self.amp_manager, "consecutive_strikes", 0)

        return self.optimizer.step(closure=closure)