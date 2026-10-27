# Project RED — Phase 0 Contract Specification (`event-v1.0`)

**Status:** Frozen for Phase 0 Handoff (`DG-01`, `DG-08` Sequencing Exception `E1`)  
**Sole Merge Governor & Contract Owner:** Aniket Gautam  
**Subsystem Consumers:** Adeel Ahmad (`MemoryProfiler`, `MagnitudePruner`), Abhay Shukla (`EarlyStopper`)

---

## 1. Event Bus Schema (`event-v1.0`)

Every event published via `RedSharedState.publish(topic, payload)` produces an immutable-Semantics envelope dictionary with the following keys:

| Key | Type | Description |
| :--- | :--- | :--- |
| `schema_version` | `str` | Always `"event-v1.0"` |
| `seq` | `int` | Strictly monotonic sequence number (`>= 1`) |
| `topic` | `str` | One of the 5 canonical topics listed in Section 2 |
| `timestamp_utc` | `float` | UTC epoch seconds from `red._time.wall_clock_utc()` (`C-TIME-01`) |
| `monotonic_s` | `float` | Monotonic timer value from `time.perf_counter()` (`DG-33`) |
| `state` | `str` | Active `RedState` value at publish time |
| `payload` | `dict[str, Any]` | Scalar-only dictionary (raw PyTorch tensors are strictly forbidden per Commandments 3 & 4) |

---

## 2. Canonical Event Topics (5 Topics)

| Topic | Publisher | Primary Subscribers | Required / Optional Payload Keys | Shared State Side Effect |
| :--- | :--- | :--- | :--- | :--- |
| `batch_event` | `DynamicBatchController` | `EarlyStopper`, `TorchOptimizer` | `{"batch_size": int, "step": int, "scaled_direction": str}` | Updates `last_batch_event` snapshot |
| `amp_disabled` | `AMPManager` | `DynamicBatchController`, `TorchOptimizer` | `{"strikes": int, "step": int, "reason": str}` | Latches `amp_disabled = True` |
| `fragmentation_warning` | `MemoryProfiler` | `DynamicBatchController` | `{"active": bool, "fragmentation_ratio": float}` | Sets `fragmentation_warning = bool(active)` |
| `vram_critical` | `DynamicBatchController` / `MemoryProfiler` | `TorchOptimizer`, `EarlyStopper` | `{"active": bool, "accumulation_steps": int}` | Sets `vram_critical = bool(active)` |
| `engine_disabled` | `TorchOptimizer` / Fault Handler | All active engines | `{"engine": str, "reason": str}` (`engine` required non-empty `str`) | Sets `engine_disabled[engine] = True` |

---

## 3. Cross-Engine Contract Registry (§2.10)

### 3.1 `C-TIME-01` — Wall-Clock Isolation (`red._time.wall_clock_utc`)
* **Owner:** Aniket Gautam (`red/_time.py`)
* **Signature:** `wall_clock_utc() -> float`
* **Rule (`DG-33` / Commandment 2):** Returns current UTC timestamp in seconds since the Unix epoch via `time.time()`. Calling `time.time()`, `time.time_ns()`, `datetime.now()`, or `datetime.utcnow()` anywhere outside `red/_time.py` is blocked by `scripts/hygiene_check.py`.

### 3.2 `C-CFG-PRUNER-01` — Pruner Precondition Validation
* **Owner:** Aniket Gautam (`RedConfig.validate_pruner`)
* **Consumer:** Adeel Ahmad (`MagnitudePruner.__init__`)
* **Rule (§6.12):** `MagnitudePruner.__init__(self, config: RedConfig, shared_state: RedSharedState, *, is_transformer: bool = False)` **MUST** call `config.validate_pruner(is_transformer=is_transformer)` as its first statement before reading `config.output_dir`, `pruning_start_step`, or `pruning_end_step`.

### 3.3 `stub-v0.1.0` — Adeel Ahmad Engine Contracts (`MemoryProfiler` & `MagnitudePruner`)
* **Version Constant:** `STUB_CONTRACT_VERSION = "stub-v0.1.0"`
* **`MemoryProfiler.poll_telemetry()`:** Must execute only when `RedStateMachine.state == RedState.RUNNING` (via `require_running()`). Returns scalar dict containing `fragmentation_ratio = inactive_split_bytes / reserved_bytes` or `"UNAVAILABLE"` when counters are zero/missing.
* **`MagnitudePruner.step(global_step: int)`:** Must call `require_running()`, enforce `C-CFG-PRUNER-01` on init, and compute iterative stage sparsity `[0.10, 0.1111, 0.125]` toward the 30% cumulative target.

### 3.4 `mock-v0.1.0` — Abhay Shukla Engine Contract (`EarlyStopper`)
* **Version Constant:** `MOCK_CONTRACT_VERSION = "mock-v0.1.0"`
* **`EarlyStopper`:** Subscribes to `"batch_event"` on `RedSharedState`. Upon receiving a `batch_event`, resets/extends its suppression window by `config.suppression_steps` (which defaults dynamically to `config.dbc_reset_interval`). Evaluates validation metrics only after `config.warmup_steps` and outside active suppression windows.
