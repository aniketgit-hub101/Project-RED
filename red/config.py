"""
red.config
RedConfig dataclass and environment variable overrides per Master §6.12.
"""

import math
import os
from dataclasses import dataclass
from typing import Any, Optional
from .exceptions import RedConfigError

__all__ = ["RedConfig"]

_MIN_LLM_GATE_SAMPLES = 512


class _DerivedInt(int):
    """Marker subclass of int used for §6.12 derived default fields."""


def _extract_sample_count(loader: Any) -> int:
    if hasattr(loader, "dataset") and hasattr(loader.dataset, "__len__"):
        try:
            return int(len(loader.dataset))
        except TypeError as exc:
            raise RedConfigError("llm_gate_val_loader.dataset must have a valid length.") from exc
    if hasattr(loader, "__len__"):
        try:
            return int(len(loader))
        except TypeError as exc:
            raise RedConfigError("llm_gate_val_loader must expose a valid sample count.") from exc
    raise RedConfigError("llm_gate_val_loader must define __len__ or a dataset with __len__.")


@dataclass(frozen=True)
class RedConfig:
    dbc_reset_interval: int = 1000
    min_batch_size: int = 1
    patience: int = 5
    warmup_steps: int = 100
    min_delta: float = 1e-4
    suppression_steps: Optional[int] = None
    pruning_start_step: Optional[int] = None
    pruning_end_step: Optional[int] = None
    pruning_stages: int = 3
    pruning_target: float = 0.30
    total_training_steps: Optional[int] = None
    output_dir: Optional[str] = None
    llm_gate_val_loader: Optional[Any] = None

    def __post_init__(self) -> None:
        if isinstance(self.dbc_reset_interval, bool) or not isinstance(self.dbc_reset_interval, int) or self.dbc_reset_interval <= 0:
            raise RedConfigError("dbc_reset_interval must be an int > 0.")
        if isinstance(self.min_batch_size, bool) or not isinstance(self.min_batch_size, int) or self.min_batch_size < 1:
            raise RedConfigError("min_batch_size must be an int >= 1.")
        if isinstance(self.patience, bool) or not isinstance(self.patience, int) or self.patience < 1:
            raise RedConfigError("patience must be an int >= 1.")
        if isinstance(self.warmup_steps, bool) or not isinstance(self.warmup_steps, int) or self.warmup_steps < 0:
            raise RedConfigError("warmup_steps must be an int >= 0.")
        if (
            isinstance(self.min_delta, bool)
            or not isinstance(self.min_delta, (int, float))
            or not math.isfinite(self.min_delta)
            or self.min_delta < 0.0
        ):
            raise RedConfigError("min_delta must be a finite float >= 0.0.")

        if self.suppression_steps is None or isinstance(self.suppression_steps, _DerivedInt):
            object.__setattr__(self, "suppression_steps", _DerivedInt(self.dbc_reset_interval))
        elif isinstance(self.suppression_steps, bool) or not isinstance(self.suppression_steps, int) or self.suppression_steps < 0:
            raise RedConfigError("suppression_steps must be an int >= 0.")

        if self.total_training_steps is not None:
            if isinstance(self.total_training_steps, bool) or not isinstance(self.total_training_steps, int) or self.total_training_steps <= 0:
                raise RedConfigError("total_training_steps must be an int > 0 when set.")

        explicit_start = self.pruning_start_step is not None and not isinstance(self.pruning_start_step, _DerivedInt)
        explicit_end = self.pruning_end_step is not None and not isinstance(self.pruning_end_step, _DerivedInt)

        if explicit_start:
            if isinstance(self.pruning_start_step, bool) or not isinstance(self.pruning_start_step, int) or self.pruning_start_step < 0:
                raise RedConfigError("pruning_start_step must be an int >= 0.")
        if explicit_end:
            if isinstance(self.pruning_end_step, bool) or not isinstance(self.pruning_end_step, int) or self.pruning_end_step <= 0:
                raise RedConfigError("pruning_end_step must be an int > 0.")

        if isinstance(self.pruning_stages, bool) or not isinstance(self.pruning_stages, int) or self.pruning_stages < 0:
            raise RedConfigError("pruning_stages must be an int >= 0.")
        if (
            isinstance(self.pruning_target, bool)
            or not isinstance(self.pruning_target, (int, float))
            or not math.isfinite(self.pruning_target)
            or not (0.0 <= self.pruning_target < 1.0)
        ):
            raise RedConfigError("pruning_target must be a finite float in [0.0, 1.0).")

        if self.output_dir is not None and not str(self.output_dir).strip():
            raise RedConfigError("output_dir cannot be an empty or whitespace string.")

        if self.llm_gate_val_loader is not None:
            count = _extract_sample_count(self.llm_gate_val_loader)
            if count < _MIN_LLM_GATE_SAMPLES:
                raise RedConfigError(
                    f"llm_gate_val_loader must contain at least {_MIN_LLM_GATE_SAMPLES} samples (got {count})."
                )

        if explicit_start or explicit_end or self.llm_gate_val_loader is not None:
            self.validate_pruner(is_transformer=self.llm_gate_val_loader is not None)

        if self.total_training_steps is not None:
            if not explicit_start:
                object.__setattr__(self, "pruning_start_step", _DerivedInt(int(0.10 * self.total_training_steps)))
            if not explicit_end:
                object.__setattr__(self, "pruning_end_step", _DerivedInt(int(0.80 * self.total_training_steps)))
        else:
            if not explicit_start:
                object.__setattr__(self, "pruning_start_step", None)
            if not explicit_end:
                object.__setattr__(self, "pruning_end_step", None)

        if self.pruning_start_step is not None and self.pruning_end_step is not None:
            if self.pruning_end_step <= self.pruning_start_step:
                raise RedConfigError("pruning_end_step must be > pruning_start_step.")
            if self.total_training_steps is not None and self.pruning_end_step > self.total_training_steps:
                raise RedConfigError("pruning_end_step cannot exceed total_training_steps.")

    def validate_pruner(self, is_transformer: bool = False) -> None:
        if self.pruning_stages <= 0 or self.pruning_target <= 0.0:
            raise RedConfigError("pruning_stages > 0 and pruning_target > 0.0 are required when pruner is active.")
        if self.total_training_steps is None:
            raise RedConfigError("total_training_steps must be set when pruner is active.")
        if self.output_dir is None or not str(self.output_dir).strip():
            raise RedConfigError("output_dir must be set and non-empty when pruner is active.")
        if is_transformer:
            if self.llm_gate_val_loader is None:
                raise RedConfigError("llm_gate_val_loader is required when pruner is active on transformer models.")
            count = _extract_sample_count(self.llm_gate_val_loader)
            if count < _MIN_LLM_GATE_SAMPLES:
                raise RedConfigError(
                    f"llm_gate_val_loader must contain at least {_MIN_LLM_GATE_SAMPLES} samples (got {count})."
                )

    @classmethod
    def from_env(cls, llm_gate_val_loader: Optional[Any] = None) -> "RedConfig":
        def _get_opt_int(env_key: str, default: Optional[int]) -> Optional[int]:
            raw = os.environ.get(env_key)
            if raw is None:
                return default
            try:
                return int(raw)
            except ValueError as exc:
                raise RedConfigError(f"Invalid integer for {env_key}: {raw!r}") from exc

        def _get_int(env_key: str, default: int) -> int:
            val = _get_opt_int(env_key, default)
            assert val is not None
            return val

        def _get_float(env_key: str, default: float) -> float:
            raw = os.environ.get(env_key)
            if raw is None:
                return default
            try:
                val = float(raw)
            except ValueError as exc:
                raise RedConfigError(f"Invalid float for {env_key}: {raw!r}") from exc
            if not math.isfinite(val):
                raise RedConfigError(f"Non-finite float for {env_key}: {raw!r}")
            return val

        defaults = cls()
        out_dir = os.environ.get("RED_OUTPUT_DIR", defaults.output_dir)

        return cls(
            dbc_reset_interval=_get_int("RED_DBC_RESET_INTERVAL", defaults.dbc_reset_interval),
            min_batch_size=_get_int("RED_MIN_BATCH_SIZE", defaults.min_batch_size),
            patience=_get_int("RED_PATIENCE", defaults.patience),
            warmup_steps=_get_int("RED_WARMUP_STEPS", defaults.warmup_steps),
            min_delta=_get_float("RED_MIN_DELTA", defaults.min_delta),
            suppression_steps=_get_opt_int("RED_SUPPRESSION_STEPS", None),
            pruning_start_step=_get_opt_int("RED_PRUNING_START_STEP", None),
            pruning_end_step=_get_opt_int("RED_PRUNING_END_STEP", None),
            pruning_stages=_get_int("RED_PRUNING_STAGES", defaults.pruning_stages),
            pruning_target=_get_float("RED_PRUNING_TARGET", defaults.pruning_target),
            total_training_steps=_get_opt_int("RED_TOTAL_TRAINING_STEPS", defaults.total_training_steps),
            output_dir=out_dir,
            llm_gate_val_loader=llm_gate_val_loader,
        )
