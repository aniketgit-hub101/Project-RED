"""
red.engines.stubs
Phase 0 contract stubs (`stub-v0.1.0` and `mock-v0.1.0`) for parallel engine
development under DG-08 (Sequencing Exception E1) and §2.10 Contract Registry.
"""

from typing import Any, Final, Optional

from ..config import RedConfig
from ..exceptions import RedStateError
from ..state import RedSharedState

__all__ = [
    "STUB_CONTRACT_VERSION",
    "MOCK_CONTRACT_VERSION",
    "MemoryProfilerStub",
    "MagnitudePrunerStub",
    "EarlyStopperMock",
]

STUB_CONTRACT_VERSION: Final[str] = "stub-v0.1.0"
MOCK_CONTRACT_VERSION: Final[str] = "mock-v0.1.0"


class MemoryProfilerStub:
    """
    Phase 0 contract stub (`stub-v0.1.0`) for Adeel Ahmad's MemoryProfiler.
    Enforces RUNNING state guard and publishes `fragmentation_warning` over `event-v1.0`.
    """

    contract_version: str = STUB_CONTRACT_VERSION

    def __init__(
        self,
        config: RedConfig,
        shared_state: RedSharedState,
        *,
        warning_threshold: float = 0.35,
    ) -> None:
        self._config = config
        self._shared_state = shared_state
        self._warning_threshold = float(warning_threshold)

    def poll_telemetry(
        self,
        inactive_split_bytes: Optional[int] = None,
        reserved_bytes: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Compute fragmentation_ratio = inactive_split_bytes / reserved_bytes.
        Returns 'UNAVAILABLE' when counters are None or <= 0.
        """
        self._shared_state.state_machine.require_running("MemoryProfiler.poll_telemetry")

        if (
            inactive_split_bytes is None
            or reserved_bytes is None
            or reserved_bytes <= 0
            or inactive_split_bytes < 0
        ):
            return {
                "contract_version": self.contract_version,
                "status": "UNAVAILABLE",
                "fragmentation_ratio": None,
            }

        ratio = float(inactive_split_bytes) / float(reserved_bytes)
        is_warn = ratio >= self._warning_threshold
        self._shared_state.publish(
            "fragmentation_warning",
            {"active": is_warn, "fragmentation_ratio": ratio},
        )
        return {
            "contract_version": self.contract_version,
            "status": "OK",
            "fragmentation_ratio": ratio,
        }


class MagnitudePrunerStub:
    """
    Phase 0 contract stub (`stub-v0.1.0`) for Adeel Ahmad's MagnitudePruner.
    Enforces C-CFG-PRUNER-01 (`config.validate_pruner`) in `__init__` and
    `require_running` on `step()`.
    """

    contract_version: str = STUB_CONTRACT_VERSION

    def __init__(
        self,
        config: RedConfig,
        shared_state: RedSharedState,
        *,
        is_transformer: bool = False,
    ) -> None:
        # C-CFG-PRUNER-01: Must validate pruner preconditions before accessing schedule/output_dir
        config.validate_pruner(is_transformer=is_transformer)
        self._config = config
        self._shared_state = shared_state
        self._is_transformer = is_transformer
        self._completed_stages: int = 0

    @property
    def completed_stages(self) -> int:
        return self._completed_stages

    def step(self, global_step: int) -> dict[str, Any]:
        """Execute a pruning check step (requires RedState.RUNNING)."""
        self._shared_state.state_machine.require_running("MagnitudePruner.step")
        start = self._config.pruning_start_step
        end = self._config.pruning_end_step
        assert start is not None and end is not None

        in_window = start <= global_step <= end
        return {
            "contract_version": self.contract_version,
            "step": int(global_step),
            "in_pruning_window": in_window,
            "completed_stages": self._completed_stages,
        }


class EarlyStopperMock:
    """
    Phase 0 contract mock (`mock-v0.1.0`) for Abhay Shukla's EarlyStopper.
    Subscribes to `batch_event` on `RedSharedState` and suppresses stopping
    for `config.suppression_steps` steps after any batch size change.
    """

    contract_version: str = MOCK_CONTRACT_VERSION

    def __init__(self, config: RedConfig, shared_state: RedSharedState) -> None:
        self._config = config
        self._shared_state = shared_state
        self._suppressed_until_step: int = -1
        self._bad_epochs: int = 0
        self._best_metric: Optional[float] = None
        self._shared_state.subscribe("batch_event", self._on_batch_event)

    @property
    def suppressed_until_step(self) -> int:
        return self._suppressed_until_step

    def _on_batch_event(self, event: dict[str, Any]) -> None:
        payload = event.get("payload", {})
        step = int(payload.get("step", 0))
        suppression = self._config.suppression_steps
        assert suppression is not None
        self._suppressed_until_step = step + int(suppression)

    def check_metric(self, metric: float, global_step: int) -> dict[str, Any]:
        """Evaluate validation metric under warmup and batch_event suppression rules."""
        self._shared_state.state_machine.require_running("EarlyStopper.check_metric")

        if global_step < self._config.warmup_steps:
            return {"should_stop": False, "reason": "warmup"}

        if global_step < self._suppressed_until_step:
            return {"should_stop": False, "reason": "suppressed_by_batch_event"}

        if self._best_metric is None or (self._best_metric - metric) > self._config.min_delta:
            self._best_metric = float(metric)
            self._bad_epochs = 0
            return {"should_stop": False, "reason": "improved"}

        self._bad_epochs += 1
        should_stop = self._bad_epochs >= self._config.patience
        return {"should_stop": should_stop, "reason": "patience_exhausted" if should_stop else "no_improvement"}
