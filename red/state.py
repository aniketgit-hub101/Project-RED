"""
red.state
Core state machine (RedState, RedStateMachine) and thread-safe event bus
(RedSharedState, event-v1.0) per Master DG-01, DG-02, DG-33, DG-63, and AR-02.3/02.4.
"""
import copy
import logging
import threading
import time
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Final, Mapping, Optional, Tuple

from ._time import wall_clock_utc
from .exceptions import RedStateError

__all__ = [
    "EVENT_SCHEMA_VERSION",
    "VALID_EVENT_TOPICS",
    "LEGAL_STATE_TRANSITIONS",
    "RedState",
    "RedStateMachine",
    "RedSharedState",
    "get_default_runtime_state",
    "status",
]

logger = logging.getLogger("red.state")
EVENT_SCHEMA_VERSION: Final[str] = "event-v1.0"

VALID_EVENT_TOPICS: Final[frozenset[str]] = frozenset(
    {
        "batch_event",
        "amp_disabled",
        "fragmentation_warning",
        "engine_disabled",
        "vram_critical",
    }
)


class RedState(str, Enum):
    UNCONFIGURED = "UNCONFIGURED"
    CONFIGURED = "CONFIGURED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    SANITIZED = "SANITIZED"
    FAILED = "FAILED"


# DG-02: Exact 12 legal state transitions mapping current state -> allowed target states
LEGAL_STATE_TRANSITIONS: Final[Mapping[RedState, frozenset[RedState]]] = MappingProxyType(
    {
        RedState.UNCONFIGURED: frozenset({RedState.CONFIGURED, RedState.FAILED}),
        RedState.CONFIGURED: frozenset({RedState.RUNNING, RedState.FAILED}),
        RedState.RUNNING: frozenset({RedState.PAUSED, RedState.SANITIZED, RedState.FAILED}),
        RedState.PAUSED: frozenset({RedState.RUNNING, RedState.SANITIZED, RedState.FAILED}),
        RedState.SANITIZED: frozenset({RedState.CONFIGURED}),
        RedState.FAILED: frozenset({RedState.SANITIZED}),
    }
)


def _is_pure_scalar(obj: Any, _depth: int = 0) -> bool:
    if _depth > 8: return False
    if obj is None or type(obj) in (bool, int, float, str):
        return True
    if type(obj) in (list, tuple, set, frozenset):
        return all(_is_pure_scalar(x, _depth + 1) for x in obj)
    if type(obj) is dict:
        return all(type(k) is str and _is_pure_scalar(v, _depth + 1) for k, v in obj.items())
    return False


def _strict_bool(val: Any) -> bool:
    if type(val) is str and val.lower() == "false": return False
    return bool(val)


class RedStateMachine:
    def __init__(self, initial_state: RedState = RedState.UNCONFIGURED) -> None:
        if not isinstance(initial_state, RedState):
            raise RedStateError(f"Invalid initial state: {initial_state!r}")
        self._lock = threading.RLock()
        self._state: RedState = initial_state
        self._last_transition_perf: float = time.perf_counter()
        self._last_transition_utc: float = wall_clock_utc()
        self._failure_reason: Optional[str] = None

    @property
    def state(self) -> RedState:
        with self._lock: return self._state

    @property
    def failure_reason(self) -> Optional[str]:
        with self._lock: return self._failure_reason

    def get_state_and_reason(self) -> Tuple[RedState, Optional[str]]:
        with self._lock:
            return self._state, self._failure_reason

    @property
    def last_transition_utc(self) -> float:
        with self._lock: return self._last_transition_utc

    @property
    def time_in_current_state(self) -> float:
        with self._lock: return time.perf_counter() - self._last_transition_perf

    def can_transition(self, target: RedState) -> bool:
        if not isinstance(target, RedState): return False
        with self._lock:
            return target in LEGAL_STATE_TRANSITIONS[self._state]

    def transition_to(self, target: RedState, *, reason: Optional[str] = None) -> RedState:
        if not isinstance(target, RedState):
            raise RedStateError(f"Target state must be a RedState enum member, got {target!r}.")
        with self._lock:
            current = self._state
            if target not in LEGAL_STATE_TRANSITIONS[current]:
                raise RedStateError(f"Illegal RedState transition: {current.value} -> {target.value}.")
            self._state = target
            self._last_transition_perf = time.perf_counter()
            self._last_transition_utc = wall_clock_utc()
            if target is RedState.FAILED and reason is not None:
                self._failure_reason = str(reason)
            return self._state

    def require_running(self, operation: str = "engine_step") -> None:
        with self._lock:
            if self._state is not RedState.RUNNING:
                raise RedStateError(f"Operation '{operation}' requires RedState.RUNNING, but current state is {self._state.value}.")

    def sanitize(self, *, internal: bool = False) -> bool:
        with self._lock:
            if internal:
                if self._state in (RedState.SANITIZED, RedState.UNCONFIGURED, RedState.CONFIGURED, RedState.FAILED):
                    return False
                self.transition_to(RedState.SANITIZED)
                return True

            if self._state is RedState.SANITIZED:
                raise RedStateError("Repeated public sanitize() call is forbidden (DG-02/DG-63).")
            if self._state in (RedState.FAILED, RedState.UNCONFIGURED, RedState.CONFIGURED):
                raise RedStateError(f"Cannot call public sanitize() from {self._state.value} state.")

            self.transition_to(RedState.SANITIZED)
            return True

    def emergency_sanitize(self) -> bool:
        """DG-02: Transitions FAILED -> SANITIZED. Raises RedStateError from any other state."""
        with self._lock:
            if self._state is not RedState.FAILED:
                raise RedStateError(
                    f"emergency_sanitize() is only valid from FAILED state, but current state is {self._state.value}."
                )
            self.transition_to(RedState.SANITIZED)
            return True

    def enter_failed(self, reason: str = "unhandled_engine_fault") -> RedState:
        with self._lock:
            if self._state is RedState.FAILED:
                return self._state
            return self.transition_to(RedState.FAILED, reason=reason)


SubscriberCallback = Callable[[dict[str, Any]], None]


class RedSharedState:
    def __init__(
        self,
        state_machine: Optional[RedStateMachine] = None,
        schema_version: str = EVENT_SCHEMA_VERSION,
    ) -> None:
        if schema_version != EVENT_SCHEMA_VERSION:
            raise RedStateError(f"Unsupported RedSharedState schema {schema_version!r}.")
        self._lock = threading.RLock()
        self._schema_version: str = schema_version
        self._machine: RedStateMachine = state_machine if state_machine is not None else RedStateMachine()
        self._subscribers: dict[str, list[SubscriberCallback]] = {topic: [] for topic in VALID_EVENT_TOPICS}
        self._engine_disabled: dict[str, bool] = {}
        self._flags: dict[str, Any] = {
            "amp_disabled": False,
            "fragmentation_warning": False,
            "vram_critical": False,
            "last_batch_event": None,
        }
        self._event_seq: int = 0
        self._subscriber_error_count: int = 0

    @property
    def schema_version(self) -> str: return self._schema_version

    @property
    def state_machine(self) -> RedStateMachine: return self._machine

    @property
    def subscriber_error_count(self) -> int:
        with self._lock: return self._subscriber_error_count

    def subscribe(self, topic: str, callback: SubscriberCallback) -> None:
        if topic not in VALID_EVENT_TOPICS: raise ValueError(f"Unknown event-v1.0 topic {topic!r}.")
        if not callable(callback): raise TypeError("Subscriber callback must be callable.")
        with self._lock:
            if callback not in self._subscribers[topic]:
                self._subscribers[topic].append(callback)

    def unsubscribe(self, topic: str, callback: SubscriberCallback) -> None:
        if topic not in VALID_EVENT_TOPICS: raise ValueError(f"Unknown event-v1.0 topic {topic!r}.")
        with self._lock:
            subs = self._subscribers[topic]
            if callback in subs: subs.remove(callback)

    def clear_subscribers(self) -> None:
        with self._lock:
            for topic in self._subscribers: self._subscribers[topic].clear()

    def publish(self, topic: str, payload: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        if topic not in VALID_EVENT_TOPICS: raise ValueError(f"Unknown event-v1.0 topic {topic!r}.")
        if payload is not None and not _is_pure_scalar(payload):
            raise TypeError("Commandments 3/4: Tensors/complex objects are strictly forbidden in RedSharedState.")

        clean_payload: dict[str, Any] = copy.deepcopy(dict(payload)) if payload is not None else {}
        with self._lock:
            if topic == "engine_disabled":
                engine_name = clean_payload.get("engine")
                if type(engine_name) is not str or not engine_name.strip():
                    raise ValueError("engine_disabled event requires non-empty string 'engine' in payload.")
                self._engine_disabled[engine_name] = True

            self._event_seq += 1
            seq = self._event_seq
            if topic == "amp_disabled":
                self._flags["amp_disabled"] = True
            elif topic == "fragmentation_warning":
                self._flags["fragmentation_warning"] = _strict_bool(clean_payload.get("active", True))
            elif topic == "vram_critical":
                self._flags["vram_critical"] = _strict_bool(clean_payload.get("active", True))
            elif topic == "batch_event":
                self._flags["last_batch_event"] = clean_payload

            event_envelope: dict[str, Any] = {
                "schema_version": self._schema_version,
                "seq": seq,
                "topic": topic,
                "timestamp_utc": wall_clock_utc(),
                "monotonic_s": time.perf_counter(),
                "state": self._machine.state.value,
                "payload": clean_payload,
            }
            callbacks_snapshot = list(self._subscribers[topic])

        for cb in callbacks_snapshot:
            try: cb(copy.deepcopy(event_envelope))
            except Exception as exc:
                with self._lock: self._subscriber_error_count += 1
                logger.warning(f"Commandment 9: callback {cb!r} raised {exc!r} on topic {topic!r}")

        return event_envelope

    def enter_failed(self, reason: str = "unhandled_engine_fault", engine_source: Optional[str] = None) -> RedState:
        with self._lock:
            current_state = self._machine.state
            if current_state is RedState.FAILED: return current_state
            st = self._machine.enter_failed(reason=reason)
            if engine_source: self._engine_disabled[engine_source] = True
        return st

    def is_engine_disabled(self, engine_name: str) -> bool:
        with self._lock: return self._engine_disabled.get(engine_name, False)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            st_val, fault_str = self._machine.get_state_and_reason()
            return {
                "schema_version": self._schema_version,
                "state": st_val.value,
                "failed": st_val is RedState.FAILED,
                "failure_reason": fault_str,
                "amp_disabled": self._flags["amp_disabled"],
                "fragmentation_warning": self._flags["fragmentation_warning"],
                "vram_critical": self._flags["vram_critical"],
                "engine_disabled": copy.deepcopy(self._engine_disabled),
                "last_batch_event": copy.deepcopy(self._flags["last_batch_event"]),
                "event_seq": self._event_seq,
                "subscriber_error_count": self._subscriber_error_count,
            }


_DEFAULT_RUNTIME_STATE = RedSharedState()


def get_default_runtime_state() -> RedSharedState: return _DEFAULT_RUNTIME_STATE


def status(shared_state: Optional[RedSharedState] = None) -> Mapping[str, Any]:
    target = shared_state if shared_state is not None else _DEFAULT_RUNTIME_STATE
    snap = target.snapshot()
    snap["engine_disabled"] = MappingProxyType(snap["engine_disabled"])
    if type(snap["last_batch_event"]) is dict:
        snap["last_batch_event"] = MappingProxyType(snap["last_batch_event"])
    return MappingProxyType(snap)
