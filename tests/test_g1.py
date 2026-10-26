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
    PREPRUNING_CHECKPOINT_MISSING,
    RED_COMPILE_WARNING,
    VRAM_PRESSURE_CRITICAL,
    RedCheckpointError,
    RedCompatibilityError,
    RedConfigError,
    RedHookConflictError,
    RedStateError,
)
import red.utils.atomic as atomic_mod
from red.utils.atomic import atomic_write
from scripts.hygiene_check import check_file


def test_public_exports_and_exceptions_match_spec() -> None:
    expected_exc = {
        "RedCompatibilityError",
        "RedStateError",
        "RedHookConflictError",
        "RedConfigError",
        "RedCheckpointError",
        "RED_COMPILE_WARNING",
        "VRAM_PRESSURE_CRITICAL",
        "PREPRUNING_CHECKPOINT_MISSING",
    }
    assert set(red.exceptions.__all__) == expected_exc
    assert expected_exc.issubset(set(red.__all__))
    assert "RedConfig" in red.__all__
    assert "atomic_write" not in red.__all__
    for exc_cls in (
        RedCompatibilityError,
        RedStateError,
        RedHookConflictError,
        RedConfigError,
        RedCheckpointError,
    ):
        assert issubclass(exc_cls, Exception)
    assert RED_COMPILE_WARNING == "RED_COMPILE_WARNING"
    assert VRAM_PRESSURE_CRITICAL == "VRAM_PRESSURE_CRITICAL"
    assert PREPRUNING_CHECKPOINT_MISSING == "PREPRUNING_CHECKPOINT_MISSING"


def test_wall_clock_utc() -> None:
    ts = wall_clock_utc()
    assert isinstance(ts, float)
    assert ts > 1_700_000_000.0


def test_config_defaults_explicit_overrides_and_replace_reactivity() -> None:
    cfg = RedConfig()
    assert cfg.dbc_reset_interval == 1000
    assert cfg.min_batch_size == 1
    assert cfg.patience == 5
    assert cfg.warmup_steps == 100
    assert cfg.min_delta == 1e-4
    assert cfg.suppression_steps == 1000
    assert cfg.pruning_start_step is None
    assert cfg.pruning_end_step is None
    assert cfg.pruning_stages == 3
    assert cfg.pruning_target == 0.30
    assert cfg.total_training_steps is None
    assert cfg.output_dir is None
    assert cfg.llm_gate_val_loader is None

    explicit_sup = RedConfig(suppression_steps=50)
    assert explicit_sup.suppression_steps == 50
    assert dataclasses.replace(explicit_sup, dbc_reset_interval=500).suppression_steps == 50

    cfg_500 = dataclasses.replace(cfg, dbc_reset_interval=500)
    assert cfg_500.suppression_steps == 500

    scheduled = RedConfig(total_training_steps=2000)
    assert scheduled.pruning_start_step == 200
    assert scheduled.pruning_end_step == 1600
    assert dataclasses.replace(scheduled, dbc_reset_interval=500).suppression_steps == 500

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
    monkeypatch.setenv("RED_MIN_BATCH_SIZE", "8")
    monkeypatch.setenv("RED_PATIENCE", "10")
    cfg = RedConfig.from_env()
    assert cfg.min_batch_size == 8
    assert cfg.patience == 10

    monkeypatch.setenv("RED_MIN_DELTA", "nan")
    with pytest.raises(RedConfigError, match="Non-finite float"):
        RedConfig.from_env()

    monkeypatch.setenv("RED_MIN_DELTA", "inf")
    with pytest.raises(RedConfigError, match="Non-finite float"):
        RedConfig.from_env()


def test_atomic_write_success_permissions_and_post_commit_oserror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "checkpoint.json"
    atomic_write(target, '{"step": 1}', mode="w")
    assert target.read_text(encoding="utf-8") == '{"step": 1}'
    assert list(tmp_path.glob("*.tmp")) == []

    mode_bits = stat.S_IMODE(target.stat().st_mode)
    assert mode_bits & stat.S_IRUSR

    # Fallback when /proc/self/status is unavailable: 0o600 on Linux, 0o666 on Windows NTFS
    monkeypatch.setattr(atomic_mod, "_PROC_STATUS", tmp_path / "nonexistent_proc_status")
    restricted_target = tmp_path / "restricted.json"
    atomic_write(restricted_target, '{"step": 1}', mode="w")

    if os.name == "posix":
        assert stat.S_IMODE(restricted_target.stat().st_mode) == 0o600
    else:
        assert stat.S_IMODE(restricted_target.stat().st_mode) in (0o600, 0o666)

    with pytest.raises(ValueError, match="Unsupported mode"):
        atomic_write(target, "append", mode="a")
    assert list(tmp_path.glob("*.tmp")) == []

    # On POSIX, verify simulated post-commit directory fsync failure
    if os.name == "posix":
        real_os_open = os.open

        def _failing_dir_open(path: Path, flags: int, *args: object) -> int:
            if Path(path) == tmp_path:
                raise OSError(5, "Simulated directory open failure")
            return real_os_open(path, flags, *args)

        monkeypatch.setattr(os, "open", _failing_dir_open)
        with pytest.raises(OSError, match="POST_COMMIT_DIR_FSYNC_FAILED"):
            atomic_write(target, '{"step": 2}', mode="w")
        assert target.read_text(encoding="utf-8") == '{"step": 2}'


def test_hygiene_checker_catches_violations(tmp_path: Path) -> None:
    fake_root = tmp_path / "repo"
    red_dir = fake_root / "red"
    red_dir.mkdir(parents=True)

    bad_file = red_dir / "bad.py"
    bad_file.write_text(
        "import time as t\n"
        "from collections import OrderedDict\n"
        "async def foo(x=[], y=OrderedDict(), z=MyMutable()):\n"
        "    return t.time()\n",
        encoding="utf-8",
    )
    issues = check_file(bad_file, fake_root)
    assert any("Missing explicit __all__" in i for i in issues)
    assert any("mutable literal" in i for i in issues)
    assert any("OrderedDict()" in i for i in issues)
    assert any("MyMutable()" in i for i in issues)
    assert any("DG-33 Violation" in i for i in issues)