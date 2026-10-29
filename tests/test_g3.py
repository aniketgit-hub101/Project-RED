"""
tests.test_g3
Gate G3 verification suite (Stage S3 PyTorch Foundations):
1. RedState.RUNNING execution guard on TorchOptimizer.step().
2. Real PyTorch forward/backward parameter updates and state_dict round-trip.
3. AMPManager 3-strike consecutive overflow latching and event-v1.0 broadcast.
4. RedHookConflictError protection and clean hook detachment on sanitize().
5. DynamicBatchController OOM halving, accumulation scaling, and step-up recovery.
6. End-to-end integration: DBC batch_event automatically suppresses EarlyStopperMock.
"""

import pytest
import torch
from torch import nn

from red.config import RedConfig
from red.engines import (
    AMPManager,
    DynamicBatchController,
    EarlyStopperMock,
    TorchOptimizer,
)
from red.exceptions import RedHookConflictError, RedStateError
from red.state import RedSharedState, RedState, RedStateMachine, status


def test_torch_optimizer_requires_running_and_lifecycle_transitions() -> None:
    model = nn.Linear(4, 2)
    raw_opt = torch.optim.SGD(model.parameters(), lr=0.1)
    bus = RedSharedState(RedStateMachine(initial_state=RedState.UNCONFIGURED))

    red_opt = TorchOptimizer(raw_opt, RedConfig(), bus)
    assert bus.state_machine.state is RedState.CONFIGURED

    with pytest.raises(RedStateError, match="requires RedState.RUNNING"):
        red_opt.step()

    red_opt.start()
    assert bus.state_machine.state is RedState.RUNNING

    red_opt.pause()
    assert bus.state_machine.state is RedState.PAUSED
    with pytest.raises(RedStateError, match="requires RedState.RUNNING"):
        red_opt.step()

    red_opt.start()
    assert red_opt.sanitize(internal=False) is True
    assert bus.state_machine.state is RedState.SANITIZED
    assert red_opt.is_attached is False


def test_torch_optimizer_updates_parameters_on_valid_step() -> None:
    torch.manual_seed(42)
    model = nn.Linear(4, 1, bias=False)
    initial_weights = model.weight.detach().clone()

    raw_opt = torch.optim.SGD(model.parameters(), lr=0.5)
    bus = RedSharedState(RedStateMachine(initial_state=RedState.UNCONFIGURED))
    red_opt = TorchOptimizer(raw_opt, RedConfig(), bus)
    red_opt.start()

    x = torch.ones((2, 4))
    target = torch.zeros((2, 1))

    red_opt.zero_grad()
    pred = model(x)
    loss = ((pred - target) ** 2).mean()
    loss.backward()

    updated = red_opt.step()
    assert updated is True
    assert red_opt.global_step == 1
    assert red_opt.skipped_steps == 0
    assert red_opt.last_step_duration_s >= 0.0
    assert not torch.allclose(model.weight, initial_weights)

    saved = red_opt.state_dict()
    assert saved["global_step"] == 1


def test_amp_manager_three_strike_latch_and_event_bus() -> None:
    model = nn.Linear(4, 1, bias=False)
    raw_opt = torch.optim.SGD(model.parameters(), lr=0.1)
    bus = RedSharedState(RedStateMachine(initial_state=RedState.UNCONFIGURED))
    amp = AMPManager(RedConfig(), bus, enabled=True, max_strikes=3)
    red_opt = TorchOptimizer(raw_opt, RedConfig(), bus, amp_manager=amp)
    red_opt.start()

    weights_before = model.weight.detach().clone()

    for _ in range(2):
        model.weight.grad = torch.full_like(model.weight, float("nan"))
        assert red_opt.step() is False
    assert amp.consecutive_strikes == 2
    assert amp.enabled is True
    assert status(bus)["amp_disabled"] is False
    assert torch.allclose(model.weight, weights_before)

    model.weight.grad = torch.ones_like(model.weight)
    assert red_opt.step() is True
    assert amp.consecutive_strikes == 0
    assert red_opt.global_step == 1

    for _ in range(3):
        model.weight.grad = torch.full_like(model.weight, float("inf"))
        assert red_opt.step() is False

    assert amp.consecutive_strikes == 3
    assert amp.total_overflows == 5
    assert amp.enabled is False
    assert status(bus)["amp_disabled"] is True
    assert red_opt.skipped_steps == 5


def test_torch_optimizer_hook_conflict_and_sanitize_detach() -> None:
    model = nn.Linear(2, 2)
    raw_opt = torch.optim.SGD(model.parameters(), lr=0.01)
    bus = RedSharedState(RedStateMachine(initial_state=RedState.RUNNING))

    opt1 = TorchOptimizer(raw_opt, RedConfig(), bus)
    with pytest.raises(RedHookConflictError, match="already has an active TorchOptimizer hook"):
        TorchOptimizer(raw_opt, RedConfig(), bus)

    opt1.sanitize(internal=False)
    assert bus.state_machine.state is RedState.SANITIZED
    bus.state_machine.transition_to(RedState.CONFIGURED)

    opt2 = TorchOptimizer(raw_opt, RedConfig(), bus)
    assert opt2.is_attached is True
    opt2.detach()


def test_dynamic_batch_controller_oom_halving_and_step_up() -> None:
    cfg = RedConfig(dbc_reset_interval=3, min_batch_size=4)
    bus = RedSharedState(RedStateMachine(initial_state=RedState.RUNNING))
    dbc = DynamicBatchController(cfg, bus, initial_batch_size=16)

    assert dbc.current_batch_size == 16
    assert dbc.accumulation_steps == 1

    new_bs = dbc.handle_oom(global_step=5)
    assert new_bs == 8
    assert dbc.accumulation_steps == 2
    assert status(bus)["last_batch_event"]["batch_size"] == 8
    assert status(bus)["last_batch_event"]["scaled_direction"] == "down"

    dbc.record_successful_step(6)
    dbc.record_successful_step(7)
    recovered_bs = dbc.record_successful_step(8)
    assert recovered_bs == 16
    assert dbc.accumulation_steps == 1
    assert status(bus)["last_batch_event"]["scaled_direction"] == "up"


def test_dbc_and_early_stopper_coordinated_suppression() -> None:
    cfg = RedConfig(dbc_reset_interval=50, warmup_steps=10, patience=2)
    bus = RedSharedState(RedStateMachine(initial_state=RedState.RUNNING))
    dbc = DynamicBatchController(cfg, bus, initial_batch_size=32)
    stopper = EarlyStopperMock(cfg, bus)

    dbc.handle_oom(global_step=20)
    assert stopper.suppressed_until_step == 70
    res = stopper.check_metric(metric=2.5, global_step=45)
    assert res["should_stop"] is False
    assert res["reason"] == "suppressed_by_batch_event"
