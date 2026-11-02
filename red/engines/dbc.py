import logging
from typing import Optional, Any

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

__all__ = ["DynamicBatchController", "RedVRAMExhaustedError"]
logger = logging.getLogger("red.dbc")

class RedVRAMExhaustedError(RuntimeError):
    pass

class DynamicBatchController:
    def __init__(self, config: Any, shared_state: Any):
        self._config = config
        self.shared_state = shared_state
        self._current_batch_size = getattr(config, 'min_batch_size', 16) * 2
        self.accumulation_steps = 1
        self._oom_recoveries = 0

    def handle_oom(self, optimizer: Optional[Any] = None, global_step: int = 0):
        self.shared_state.state_machine.require_running("DynamicBatchController.handle_oom")
        
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            
        if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        min_bs = getattr(self._config, 'min_batch_size', 1)
        
        if self._current_batch_size <= min_bs:
            self.shared_state.publish(
                "vram_critical",
                {"active": True, "accumulation_steps": self.accumulation_steps, "step": int(global_step)}
            )
            logger.warning("VRAM_PRESSURE_CRITICAL: memory threshold breached.")
            self.shared_state.state_machine.transition_to("FAILED")
            raise RedVRAMExhaustedError("Minimum batch size reached. Cannot scale down further.")

        new_bs = max(min_bs, self._current_batch_size // 2)
        self._current_batch_size = new_bs
        self._oom_recoveries += 1
        
        self.shared_state.publish(
            "batch_event",
            {"batch_size": int(self._current_batch_size), "accumulation_steps": int(self.accumulation_steps), "scaled_direction": "down", "step": int(global_step)}
        )
        return self._current_batch_size