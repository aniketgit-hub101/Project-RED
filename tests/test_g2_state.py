"""
tests.test_g2_state
Gate G2 verification suite.
"""
import itertools
import threading
import time
import pytest

from red.exceptions import RedStateError
from red.state import EVENT_SCHEMA_VERSION, LEGAL_STATE_TRANSITIONS, RedSharedState, RedState, RedStateMachine, status


def test_exhaustive_12_legal_state_pairs() -> None:
    all_states = list(RedState)
    pairs = list(itertools.product(all_states, all_states))
    expected_legal = {
        (RedState.UNCONFIGURED, RedState.CONFIGURED), (RedState.UNCONFIGURED, RedState.FAILED),
        (RedState.CONFIGURED, RedState.RUNNING), (RedState.CONFIGURED, RedState.FAILED),
        (RedState.RUNNING, RedState.PAUSED), (RedState.RUNNING, RedState.SANITIZED), (RedState.RUNNING, RedState.FAILED),
        (RedState.PAUSED, RedState.RUNNING), (RedState.PAUSED, RedState.SANITIZED), (RedState.PAUSED, RedState.FAILED),
        (RedState.SANITIZED, RedState.CONFIGURED), (RedState.FAILED, RedState.SANITIZED),
    }
    assert len(expected_legal) == 12

    for src, dst in pairs:
        machine = RedStateMachine(initial_state=src)
        if (src, dst) in expected_legal:
            assert machine.can_transition(dst) is True
            res = machine.transition_to(dst, reason="legal_test")
            assert res is dst
            assert machine.state is dst
        else:
            assert machine.can_transition(dst) is False
            with pytest.raises(RedStateError, match="Illegal RedState transition"):
                machine.transition_to(dst)


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

    machine.transition_to(RedState.CONFIGURED)
    with pytest.raises(RedStateError, match="Cannot call public sanitize"):
        machine.sanitize(internal=False)
    assert machine.sanitize(internal=True) is False

    machine.transition_to(RedState.FAILED, reason="test_failure")
    with pytest.raises(RedStateError, match="Cannot call public sanitize"):
        machine.sanitize(internal=False)
    assert machine.sanitize(internal=True) is False

    assert machine.emergency_sanitize() is True
    assert machine.state is RedState.SANITIZED

    with pytest.raises(RedStateError, match="emergency_sanitize"):
        machine.emergency_sanitize()


def test_shared_state_topics_tensor_ban_and_subscriber_isolation() -> None:
    bus = RedSharedState(RedStateMachine(initial_state=RedState.RUNNING))

    received = []
    def _bad_subscriber(event): raise RuntimeError("Simulated broken subscriber")
    def _good_subscriber(event): received.append(event["payload"])

    bus.subscribe("batch_event", _bad_subscriber)
    bus.subscribe("batch_event", _good_subscriber)

    bus.publish("vram_critical", {"active": "false"})
    assert status(bus)["vram_critical"] is False

    payload = {"step": 10}
    env = bus.publish("batch_event", payload)

    received[0]["step"] = 999
    assert env["payload"]["step"] == 10

    class FakeTensor:
        grad_fn = object()

    with pytest.raises(TypeError, match="Commandments 3/4"):
        bus.publish("batch_event", {"raw_tensor": FakeTensor()})

    initial_seq = bus.snapshot()["event_seq"]
    with pytest.raises(ValueError, match="engine_disabled event requires non-empty string 'engine'"):
        bus.publish("engine_disabled", {"engine": ""})
    assert bus.snapshot()["event_seq"] == initial_seq

    bus.publish("engine_disabled", {"engine": "MagnitudePruner"})
    assert status(bus)["engine_disabled"]["MagnitudePruner"] is True


def test_shared_state_highly_concurrent_stress() -> None:
    bus = RedSharedState(RedStateMachine(initial_state=RedState.RUNNING))
    barrier = threading.Barrier(9)

    def _publisher(tid: int):
        barrier.wait()
        for step in range(10): bus.publish("batch_event", {"worker": tid, "step": step})

    def _subscriber(tid: int):
        barrier.wait()
        def _cb(evt): pass
        for _ in range(10):
            bus.subscribe("batch_event", _cb)
            status(bus)
            bus.unsubscribe("batch_event", _cb)

    def _transition():
        barrier.wait()
        time.sleep(0.01)
        bus.enter_failed("Stress Test Fault")
        status(bus)

    threads = [threading.Thread(target=_publisher, args=(i,)) for i in range(4)]
    threads += [threading.Thread(target=_subscriber, args=(i,)) for i in range(4)]
    threads.append(threading.Thread(target=_transition))

    for t in threads: t.start()
    for t in threads: t.join(timeout=5.0)

    snap = bus.snapshot()
    assert snap["failed"] is True
    assert snap["event_seq"] == 40
