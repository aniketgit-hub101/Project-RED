# PROJECT RED — REQUIREMENT TRACEABILITY MATRIX (GATES G1 & G2)

| Spec Reference | Requirement Summary | Implementation Module | Verification Test | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Master §6.8 / Cmd 13** | Public package structure & `__all__` stability guarantee | `red/__init__.py`, `pyproject.toml` | `tests/test_g1.py::test_public_exports_and_exceptions_match_spec` | **PASS** |
| **Master §6.11** | 5 Exception classes & 3 Log Markers | `red/exceptions.py` | `tests/test_g1.py::test_public_exports_and_exceptions_match_spec` | **PASS** |
| **Master §6.12** | All 13 `RedConfig` parameters, defaults, env overrides, `validate_pruner`, `>= 512` transformer loader check | `red/config.py` | `tests/test_g1.py::test_config_defaults_explicit_overrides_and_replace_reactivity`, `test_config_pruner_and_loader_validation`, `test_config_env_overrides_and_nan_rejection` | **PASS** |
| **Commandment 2 / DG-33** | Wall-clock isolation in `red/_time.py` (`C-TIME-01`) & monotonic `perf_counter()` | `red/_time.py`, `scripts/hygiene_check.py` | `tests/test_g1.py::test_wall_clock_utc`, `test_hygiene_checker_catches_violations` | **PASS** |
| **Commandment 8** | Crash-safe `.tmp` + `fsync` + `os.replace` atomic writer | `red/utils/atomic.py` | `tests/test_g1.py::test_atomic_write_success_permissions_and_post_commit_oserror` | **PASS** |
| **DG-02 / AR-02.3** | 6-state `RedState` machine, immutable transition table, 36 ordered state pairs, `require_running` guard | `red/state.py` | `tests/test_g2_state.py::test_exhaustive_36_state_transition_pairs`, `test_require_running_rejects_all_non_running_states` | **PASS** |
| **DG-63** | Public vs. internal `sanitize()` idempotency & terminal `FAILED` state | `red/state.py` | `tests/test_g2_state.py::test_dg63_public_vs_internal_sanitize_and_terminal_failed` | **PASS** |
| **AR-02.4 / Cmd 3, 4, 9** | `RedSharedState` (`event-v1.0`) 5 topics, raw tensor ban, subscriber isolation, 100-run stress test | `red/state.py` | `tests/test_g2_state.py::test_shared_state_topics_tensor_ban_and_subscriber_isolation`, `test_shared_state_100_run_concurrency_stress` | **PASS** |
| **AR-02.6** | Read-only `red.status()` snapshot | `red/state.py` | `tests/test_g2_state.py::test_shared_state_topics_tensor_ban_and_subscriber_isolation` | **PASS** |
| **DG-08 (`E1`) / §2.10** | Phase 0 `event-v1.0` contract doc, `stub-v0.1.0`, `mock-v0.1.0`, `C-CFG-PRUNER-01` | `docs/contracts/event-v1.0.md`, `red/engines/stubs.py` | `tests/test_g2_contracts.py` (3 contract tests) | **PASS** |
