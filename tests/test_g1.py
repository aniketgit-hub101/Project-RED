"""
tests.test_g1
Unit tests covering Gate G1 criteria.
"""
import dataclasses
import os
import stat
from pathlib import Path
import pytest

import red
from red._time import wall_clock_utc
from red.config import RedConfig
from red.exceptions import (
    PREPRUNING_CHECKPOINT_MISSING, RED_COMPILE_WARNING, VRAM_PRESSURE_CRITICAL,
    RedCheckpointError, RedCompatibilityError, RedConfigError, RedHookConflictError, RedStateError
)
import red.utils.atomic as atomic_mod
from red.utils.atomic import atomic_write
from scripts.hygiene_check import check_file


def test_public_exports_and_exceptions_match_spec() -> None:
    expected = {
        "RedCompatibilityError", "RedStateError", "RedHookConflictError",
        "RedConfigError", "RedCheckpointError", "RED_COMPILE_WARNING",
        "VRAM_PRESSURE_CRITICAL", "PREPRUNING_CHECKPOINT_MISSING"
    }
    assert set(red.exceptions.__all__) == expected
    for exc in (RedCompatibilityError, RedStateError, RedHookConflictError, RedConfigError, RedCheckpointError):
        assert issubclass(exc, Exception)


def test_wall_clock_utc() -> None:
    ts = wall_clock_utc()
    assert isinstance(ts, float) and ts > 1_700_000_000.0


def test_config_defaults_explicit_overrides_and_replace_reactivity() -> None:
    cfg = RedConfig()
    assert cfg.dbc_reset_interval == 1000
    assert cfg.suppression_steps == 1000
    assert cfg.pruning_start_step is None

    cfg_fast = dataclasses.replace(cfg, dbc_reset_interval=500)
    assert cfg_fast.dbc_reset_interval == 500
    assert cfg_fast.suppression_steps == 500

    scheduled = RedConfig(total_training_steps=2000)
    assert scheduled.pruning_start_step == 200
    assert scheduled.pruning_end_step == 1600

    rescheduled = dataclasses.replace(scheduled, total_training_steps=10000)
    assert rescheduled.pruning_start_step == 1000
    assert rescheduled.pruning_end_step == 8000

    explicit_sched = RedConfig(
        total_training_steps=2000,
        output_dir="/tmp/out",
        pruning_start_step=100,
        pruning_end_step=1500,
    )
    assert explicit_sched.pruning_start_step == 100
    assert explicit_sched.pruning_end_step == 1500


def test_config_pruner_and_loader_validation() -> None:
    cfg = RedConfig()
    with pytest.raises(RedConfigError, match="total_training_steps must be set when pruner is active"):
        cfg.validate_pruner(is_transformer=False)

    with pytest.raises(RedConfigError, match="total_training_steps must be set when pruner is active"):
        RedConfig(pruning_start_step=100)

    with pytest.raises(RedConfigError, match="total_training_steps must be set when pruner is active"):
        RedConfig(pruning_start_step=100, pruning_end_step=800)

    with pytest.raises(RedConfigError, match="output_dir must be set and non-empty when pruner is active"):
        RedConfig(total_training_steps=2000, pruning_start_step=100, pruning_end_step=800)

    with pytest.raises(RedConfigError, match="total_training_steps must be set when pruner is active"):
        RedConfig(llm_gate_val_loader=list(range(512)))

    non_llm_cfg = RedConfig(total_training_steps=1000, output_dir="/tmp/out")
    non_llm_cfg.validate_pruner(is_transformer=False)

    with pytest.raises(RedConfigError, match="llm_gate_val_loader is required"):
        non_llm_cfg.validate_pruner(is_transformer=True)

    with pytest.raises(RedConfigError, match="at least 512 samples"):
        RedConfig(total_training_steps=1000, output_dir="/tmp/out", llm_gate_val_loader=list(range(5)))

    valid_llm_cfg = RedConfig(
        total_training_steps=1000,
        output_dir="/tmp/out",
        llm_gate_val_loader=list(range(512)),
    )
    valid_llm_cfg.validate_pruner(is_transformer=True)


def test_config_env_overrides_and_nan_rejection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RED_MIN_DELTA", "nan")
    with pytest.raises(RedConfigError, match="Non-finite float"):
        RedConfig.from_env()


def test_atomic_write_success_permissions_and_post_commit_oserror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "checkpoint.json"
    atomic_write(target, '{"step": 1}', mode="w")
    monkeypatch.setattr(atomic_mod, "_PROC_STATUS", tmp_path / "nonexistent")
    atomic_write(tmp_path / "restricted.json", '{"step": 1}', mode="w")
    if os.name == "posix":
        assert stat.S_IMODE((tmp_path / "restricted.json").stat().st_mode) == 0o600

        real_os_open = os.open
        def _failing_dir_open(path: Path, flags: int, *args: object) -> int:
            if Path(path) == tmp_path: raise OSError(5, "Simulated directory open failure")
            return real_os_open(path, flags, *args)
        monkeypatch.setattr(os, "open", _failing_dir_open)
        with pytest.raises(OSError, match="POST_COMMIT_DIR_FSYNC_FAILED"):
            atomic_write(target, '{"step": 2}', mode="w")


def test_hygiene_checker_catches_violations(tmp_path: Path) -> None:
    fake_root = tmp_path / "repo"
    (fake_root / "red").mkdir(parents=True)
    bad_file = fake_root / "red" / "bad.py"
    bad_code = "\n".join([
        "import time as t",
        "from datetime import datetime",
        "def foo(x=[]):",
        "    return datetime.today()",
    ]) + "\n"
    bad_file.write_text(bad_code, encoding="utf-8")
    issues = check_file(bad_file, fake_root)
    assert any("Missing explicit __all__" in i for i in issues)
    assert any("mutable literal" in i for i in issues)
    assert any("DG-33 Violation" in i for i in issues)
