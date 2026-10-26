"""
tests.test_g2_state
Gate G2 verification suite.
"""

import itertools
import threading
import pytest

from red.exceptions import RedStateError
from red.state import (
    EVENT_SCHEMA_VERSION,
    LEGAL_STATE_TRANSITIONS,
    RedSharedState,
    RedState,
    RedStateMachine,
    status,
)


def test_exhaustive_36_state_transition_pairs() -> None:
    all_states = list(RedState)
    assert len(all_states) == 6

    pairs = list(itertools.product(all_states, all_states))
    assert len(pairs) == 36

    expected_legal: set[tuple[RedState, RedState]] = {
        (RedState.UNCONFIGURED, RedState.CONFIGURED),
        (RedState.UNCONFIGURED, RedState.FAILED),
        (RedState.CONFIGURED, RedState.RUNNING),
        (RedState.CONFIGURED, RedState.SANITIZED),
        (RedState.CONFIGURED, RedState.FAILED),
        (RedState.RUNNING, RedState.PAUSED),
        (RedState.RUNNING, RedState.SANITIZED),
        (RedState.RUNNING, RedState.FAILED),
        (RedState.PAUSED, RedState.RUNNING),
        (RedState.PAUSED, RedState.SANITIZED),
        (RedState.PAUSED, RedState.FAILED),
        (RedState.SANITIZED, RedState.CONFIGURED),
        (RedState.SANITIZED, RedState.FAILED),
    }

    for src, dst in pairs:
        machine = RedStateMachine(initial_state=src)
        if (src, dst) in expected_legal:
            assert machine.can_transition(dst) is True
            result = machine.transition_to(dst, reason="test_transition")
            assert result is dst
            assert machine.state is dst
        else:
            assert machine.can_transition(dst) is False
            with pytest.raises(RedStateError, match="Illegal RedState transition"):
                machine.transition_to(dst)
            assert machine.state is src


def test_require_running_rejects_all_non_running_states() -> None:
    for st in RedState:
        machine = RedStateMachine(initial_state=st)
        if st is RedState.RUNNING:
            machine.require_running("dbc.step")
        else:
            with pytest.raises(RedStateError, match="requires RedState.RUNNING"):
                machine.require_running("dbc.step")


def test_dg63_public_vs_internal_sanitize_and_terminal_failed() -> None:
    machine = RedStateMachine(initial_state=RedState.RUNNING)
    assert machine.sanitize(internal=False) is True
    assert machine.state is RedState.SANITIZED

    with pytest.raises(RedStateError, match="Repeated public sanitize"):
        machine.sanitize(internal=False)

    assert machine.sanitize(internal=True) is False
    assert machine.state is RedState.SANITIZED

    machine.enter_failed("cuda_oom_unrecoverable")
    assert machine.state is RedState.FAILED
    assert machine.failure_reason == "cuda_oom_unrecoverable"

    with pytest.raises(RedStateError, match="terminal FAILED state"):
        machine.sanitize(internal=False)
    assert machine.sanitize(internal=True) is False


def test_shared_state_topics_tensor_ban_and_subscriber_isolation() -> None:
    bus = RedSharedState(RedStateMachine(initial_state=RedState.RUNNING))
    assert bus.schema_version == EVENT_SCHEMA_VERSION

    received: list[str] = []

    def _bad_subscriber(event: dict[str, object]) -> None:
        raise RuntimeError("Simulated broken subscriber")

    def _good_subscriber(event: dict[str, object]) -> None:
        received.append(str(event["topic"]))

    bus.subscribe("batch_event", _bad_subscriber)
    bus.subscribe("batch_event", _good_subscriber)

    env = bus.publish("batch_event", {"batch_size": 32, "step": 10})
    assert env["schema_version"] == "event-v1.0"
    assert received == ["batch_event"]
    assert bus.subscriber_error_count == 1

    bus.publish("amp_disabled", {"strikes": 3})
    bus.publish("fragmentation_warning", {"active": True, "ratio": 0.42})
    bus.publish("vram_critical", {"active": True})
    bus.publish("engine_disabled", {"engine": "MagnitudePruner"})

    snap = status(bus)
    assert snap["state"] == "RUNNING"
    assert snap["failed"] is False
    assert snap["amp_disabled"] is True
    assert snap["fragmentation_warning"] is True
    assert snap["vram_critical"] is True
    assert snap["engine_disabled"]["MagnitudePruner"] is True
    assert bus.is_engine_disabled("MagnitudePruner") is True

    with pytest.raises(TypeError):
        snap["vram_critical"] = False

    class FakeTensor:
        grad_fn = object()
        def data_ptr(self) -> int:
            return 12345
        def numel(self) -> int:
            return 64

    with pytest.raises(TypeError, match="Commandment 3/4 violation"):
        bus.publish("batch_event", {"raw_tensor": FakeTensor()})


def test_shared_state_100_run_concurrency_stress() -> None:
    for _ in range(100):
        bus = RedSharedState(RedStateMachine(initial_state=RedState.RUNNING))
        counter_lock = threading.Lock()
        observed_count = 0

        def _on_event(evt: dict[str, object]) -> None:
            nonlocal observed_count
            with counter_lock:
                observed_count += 1

        bus.subscribe("batch_event", _on_event)

        def _publisher(tid: int) -> None:
            for step in range(5):
                bus.publish("batch_event", {"worker": tid, "step": step})
                _ = status(bus)

        threads = [threading.Thread(target=_publisher, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
            assert not t.is_alive(), "Deadlock detected in RedSharedState stress run"

        snap = bus.snapshot()
        assert snap["event_seq"] == 20
        assert observed_count == 20
