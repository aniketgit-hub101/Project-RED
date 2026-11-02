from typing import Any, Dict

__all__ = ["AMPManager", "RedCheckpointError"]

class RedCheckpointError(ValueError):
    pass

class AMPManager:
    def __init__(self, enabled: bool = False, opt_level: str = "O1"):
        self._enabled = enabled
        self._opt_level = opt_level
        self.consecutive_strikes = 0
        self.total_overflows = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def opt_level(self) -> str:
        return self._opt_level

    def state_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self._enabled,
            "opt_level": self._opt_level,
            "consecutive_strikes": self.consecutive_strikes,
            "total_overflows": self.total_overflows
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if not isinstance(state_dict, dict):
            raise RedCheckpointError("Checkpoint state must be a dictionary")

        valid_keys = {"enabled", "opt_level", "consecutive_strikes", "total_overflows"}
        extra_keys = set(state_dict.keys()) - valid_keys
        if extra_keys:
            raise RedCheckpointError(f"Unexpected keys in checkpoint state: {extra_keys}")

        if "enabled" not in state_dict:
            raise RedCheckpointError("Missing required key 'enabled'")

        enabled_val = state_dict["enabled"]
        if not isinstance(enabled_val, bool):
            raise RedCheckpointError(f"'enabled' must be an explicit boolean.")
        
        opt_val = state_dict.get("opt_level", "O1")
        if opt_val not in {"O0", "O1", "O2", "O3"}:
            raise RedCheckpointError(f"Invalid opt_level: {opt_val}")

        self._enabled = enabled_val
        self._opt_level = opt_val
        self.consecutive_strikes = state_dict.get("consecutive_strikes", 0)
        self.total_overflows = state_dict.get("total_overflows", 0)