from typing import Any, Dict, Optional

class RedCheckpointError(ValueError):
    """Raised when checkpoint state dictionary fails strict schema validation."""
    pass

class AMPManager:
    """
    Project RED Automatic Mixed Precision (AMP) state coordinator.
    """
    def __init__(self, enabled: bool = False, opt_level: str = "O1"):
        self._enabled = enabled
        self._opt_level = opt_level

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def opt_level(self) -> str:
        return self._opt_level

    def state_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self._enabled,
            "opt_level": self._opt_level
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if not isinstance(state_dict, dict):
            raise RedCheckpointError("Checkpoint state must be a dictionary")

        valid_keys = {"enabled", "opt_level"}
        extra_keys = set(state_dict.keys()) - valid_keys
        if extra_keys:
            raise RedCheckpointError(f"Unexpected keys in checkpoint state: {extra_keys}")

        if "enabled" not in state_dict:
            raise RedCheckpointError("Missing required key 'enabled'")

        enabled_val = state_dict["enabled"]
        # Explicit type check: forbids string booleans like "False"
        if not isinstance(enabled_val, bool):
            raise RedCheckpointError(f"'enabled' must be an explicit boolean, got {type(enabled_val).__name__}")

        self._enabled = enabled_val
        if "opt_level" in state_dict:
            self._opt_level = state_dict["opt_level"]