"""
tests.test_g2_contracts
Unit tests verifying Phase 0 engine contracts (`stub-v0.1.0`, `mock-v0.1.0`,
and `C-CFG-PRUNER-01`) over `RedSharedState` (`event-v1.0`).
"""

import dataclasses
import pytest

from red.config import RedConfig
from red.engines import (
    EarlyStopperMock,
    MOCK_CONTRACT_VERSION,
    MagnitudePrunerStub,
    MemoryProfilerStub,
    STUB_CONTRACT_VERSION,
)
from red.exceptions import RedConfigError, RedStateError
from red.state import RedSharedState, RedState, RedStateMachine, status


def test_memory_profiler_stub_contract() -> None:
    cfg = RedConfig()
    bus = RedSharedState(RedStateMachine(initial_state=RedState.CONFIGURED))
    profiler = MemoryProfilerStub(cfg, bus, warning_threshold=0.30)
    assert profiler.contract_version == STUB_CONTRACT_VERSION

    # Calling outside RUNNING raises RedStateError
    with pytest.raises(RedStateError, match="requires RedState.RUNNING"):
        profiler.poll_telemetry(100, 1000)

    bus.state_machine.transition_to(RedState.RUNNING)

    # Zero or missing counters return UNAVAILABLE safely
    unavail = profiler.poll_telemetry(0, 0)
    assert unavail["status"] == "UNAVAILABLE"

    # High fragmentation triggers fragmentation_warning on event-v1.0 bus
    res = profiler.poll_telemetry(inactive_split_bytes=400, reserved_bytes=1000)
    assert res["status"] == "OK"
    assert res["fragmentation_ratio"] == pytest.approx(0.40)
    assert status(bus)["fragmentation_warning"] is True


def test_magnitude_pruner_stub_enforces_c_cfg_pruner_01() -> None:
    bus = RedSharedState(RedStateMachine(initial_state=RedState.RUNNING))

    # Bare RedConfig() fails C-CFG-PRUNER-01 when instantiating MagnitudePrunerStub
    with pytest.raises(RedConfigError, match="total_training_steps must be set when pruner is active"):
        MagnitudePrunerStub(RedConfig(), bus)

    # Transformer model requires llm_gate_val_loader >= 512 samples
    cfg_no_loader = RedConfig(total_training_steps=2000, output_dir="/tmp/checkpoints")
    with pytest.raises(RedConfigError, match="llm_gate_val_loader is required"):
        MagnitudePrunerStub(cfg_no_loader, bus, is_transformer=True)

    pruner = MagnitudePrunerStub(cfg_no_loader, bus, is_transformer=False)
    step_info = pruner.step(global_step=500)
    assert step_info["in_pruning_window"] is True


def test_early_stopper_mock_batch_event_suppression() -> None:
    cfg = RedConfig(dbc_reset_interval=500, warmup_steps=100, patience=2)
    bus = RedSharedState(RedStateMachine(initial_state=RedState.RUNNING))
    stopper = EarlyStopperMock(cfg, bus)
    assert stopper.contract_version == MOCK_CONTRACT_VERSION

    # Warmup guard
    assert stopper.check_metric(1.0, global_step=50)["reason"] == "warmup"

    # Publish batch_event at step 200 -> suppresses EarlyStopper until step 200 + 500 = 700
    bus.publish("batch_event", {"batch_size": 64, "step": 200})
    assert stopper.suppressed_until_step == 700
    assert stopper.check_metric(1.0, global_step=450)["reason"] == "suppressed_by_batch_event"

    # If dbc_reset_interval is updated via dataclasses.replace, derived suppression_steps updates too
    cfg_fast = dataclasses.replace(cfg, dbc_reset_interval=250)
    stopper_fast = EarlyStopperMock(cfg_fast, bus)
    bus.publish("batch_event", {"batch_size": 32, "step": 1000})
    assert stopper_fast.suppressed_until_step == 1250
