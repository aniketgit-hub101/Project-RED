# PROJECT RED — ARCHITECTURAL DECISION REGISTER (`DG-` LEDGER)

**Sole Merge Governor:** Aniket Gautam  
**Last Updated:** 2026-10-27

| Decision ID | Status | Summary & Binding Rule | Code / Test Enforcement |
| :--- | :--- | :--- | :--- |
| **`DG-01`** | **LOCKED** | Aniket Gautam owns Phase 0 `RedConfig`, `RedState`, `RedSharedState` (`event-v1.0`), and atomic checkpoint utilities. | `red/config.py`, `red/state.py`, `red/utils/atomic.py` |
| **`DG-02`** | **LOCKED** | 6-state lifecycle (`UNCONFIGURED`, `CONFIGURED`, `RUNNING`, `PAUSED`, `SANITIZED`, `FAILED`). Legal transition table stored as immutable data (`LEGAL_STATE_TRANSITIONS`). `FAILED` is reachable from any non-terminal state and is strictly terminal (0 outgoing transitions). | `red/state.py`, `tests/test_g2_state.py::test_exhaustive_36_state_transition_pairs` |
| **`DG-08` (`E1`)** | **APPROVED** | Sequencing Exception `E1` approved. With `event-v1.0`, `stub-v0.1.0`, and `mock-v0.1.0` published in Phase 0, Adeel Ahmad (`MemoryProfiler`, `MagnitudePruner`) and Abhay Shukla (`EarlyStopper`) are unblocked for parallel engine development. | `docs/contracts/event-v1.0.md`, `red/engines/stubs.py`, `tests/test_g2_contracts.py` |
| **`DG-15`** | **LOCKED** | Windows directory `fsync` is deferred per Master line 424; POSIX directory `fsync` runs under `os.name == "posix"` and raises `OSError` (`POST_COMMIT_DIR_FSYNC_FAILED`) if post-commit durability fails. | `red/utils/atomic.py`, `tests/test_g1.py` |
| **`DG-33`** | **LOCKED** | Wall-clock UTC calls (`time.time()`) are isolated exclusively inside `red/_time.py` (`wall_clock_utc() -> float`, Contract `C-TIME-01`). All internal durations use `time.perf_counter()`. Static AST checker enforces zero violations across `red/` and `tests/`. | `red/_time.py`, `scripts/hygiene_check.py`, `tests/test_g1.py` |
| **`DG-63`** | **LOCKED** | First public `sanitize()` call from `CONFIGURED`, `RUNNING`, or `PAUSED` transitions to `SANITIZED`. A repeated public `sanitize()` call (or call from `FAILED`/`UNCONFIGURED`) raises `RedStateError`. Internal cleanup hooks (`internal=True` from `atexit`, `__del__`, `__exit__`) check state first and return `False` safely when already `SANITIZED` or `FAILED`. | `red/state.py`, `tests/test_g2_state.py::test_dg63_public_vs_internal_sanitize_and_terminal_failed` |
| **`ERR-G2-03`** | **RECORDED** | Roadmap Gate G2 Criterion 3 text "12 of 12 rows" is logged as a specification erratum for the 13 parameters in Master §6.12 (including `llm_gate_val_loader`). All 13 parameters are implemented and tested. | `red/config.py`, `tests/test_g1.py` |
