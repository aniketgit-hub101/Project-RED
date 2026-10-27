# PROJECT RED — GATE G1 & G2 SEEDED BUG & PROBE ROOT-CAUSE REPORT

**Author:** Aniket Gautam  
**Date:** 2026-10-27  
**Gate Requirement:** Gate G1 Criterion 2 (`>= 8/10` seeded bugs root-caused and fixed with regression tests).  
**Score:** 10 / 10 Root-Caused, Fixed, and Verified.

---

### Bug 1: File Descriptor Leak on `os.fdopen()` Failure in `atomic_write`
* **Symptom:** When `os.fdopen(fd, mode)` failed (e.g., invalid mode), the raw file descriptor returned by `tempfile.mkstemp()` remained open, leaking OS descriptors.
* **Root Cause:** `mkstemp()` opens an OS-level fd before `os.fdopen()` wraps it in a Python file object. If `os.fdopen()` raises, the `with` context manager is never entered.
* **Fix:** Validate `mode in {"w", "wb"}` before calling `mkstemp()`, track `fd_owned = True` until `os.fdopen()` succeeds, and close `fd` in the `except BaseException:` block if `fd_owned` is still `True`.

### Bug 2: Orphaned `.tmp` File on `KeyboardInterrupt` (`BaseException`)
* **Symptom:** Sending `SIGINT` (`KeyboardInterrupt`) during `os.fsync()` left stale `.tmp` files in the checkpoint directory.
* **Root Cause:** Catching `except Exception:` missed `KeyboardInterrupt` and `SystemExit`, which inherit directly from `BaseException`.
* **Fix:** Changed cleanup handler in `red/utils/atomic.py` to `except BaseException:`, unlinking `temp_path` whenever `not committed` before re-raising.

### Bug 3: Process-Wide `os.umask(0)` Race Window
* **Symptom:** Calling `old = os.umask(0); os.umask(old)` to read the current process umask created a multi-threaded race window where concurrent threads created world-writable (`0666`) files.
* **Root Cause:** POSIX `umask(2)` mutates process-global state.
* **Fix:** Read `Umask:` non-destructively from `/proc/self/status` on Linux, falling back to restrictive `0o600` (never widening permissions beyond the user's umask).

### Bug 4: Swallowed Post-Commit Directory `fsync` Errors vs. Overloaded `RedCheckpointError`
* **Symptom:** Wrapping directory `fsync` in `except OSError: pass` hid durability failures, while raising `RedCheckpointError` violated Master §6.11 (which reserves `RedCheckpointError` for post-load allowlist key validation) and bypassed `except OSError:` handlers.
* **Root Cause:** Post-commit durability failure is an OS I/O error occurring after `os.replace` has already committed the file.
* **Fix:** Track `committed = True` after `os.replace`, do not delete `temp_path` post-commit, and raise `OSError(exc.errno, "POST_COMMIT_DIR_FSYNC_FAILED: ...") from exc`.

### Bug 5: `dataclasses.replace()` Staleness on Derived `RedConfig` Fields
* **Symptom:** Replacing `dbc_reset_interval=500` on a default `RedConfig()` left `suppression_steps` stale at `1000`; replacing `total_training_steps=10000` left `pruning_start_step` and `pruning_end_step` at their old values.
* **Root Cause:** `dataclasses.replace(obj, **changes)` copies unchanged fields via `getattr(obj, f.name)`. Overwriting `None` with a plain `int` in `__post_init__` made derived values indistinguishable from explicit user values.
* **Fix:** Introduced `_DerivedInt(int)` marker subclass so `__post_init__` recomputes derived defaults across `dataclasses.replace()` while preserving explicit caller-supplied `int` overrides.

### Bug 6: `setattr` Descriptor Binding After `@dataclass` Class Creation
* **Symptom:** Attaching custom descriptors via `setattr(RedConfig, ...)` after class creation left descriptor `_name == ""` and caused all three derived fields to collide in `instance.__dict__[""]`.
* **Root Cause:** Python's type machinery only invokes `__set_name__(owner, name)` on descriptors defined inside the class body at class creation time, not on attributes attached later via `setattr`.
* **Fix:** Removed post-hoc descriptors in favor of `_DerivedInt(int)` inside `__post_init__`.

### Bug 7: `nan` and `inf` Acceptance in `RedConfig` Float Fields
* **Symptom:** Setting `RED_MIN_DELTA=nan` or `RED_MIN_DELTA=inf` bypassed `self.min_delta < 0.0` because IEEE 754 `NaN < 0.0` evaluates to `False`.
* **Root Cause:** Range comparisons (`<`, `>=`) silently return `False` on `NaN` and allow `+inf` unless finiteness is checked first.
* **Fix:** Added `math.isfinite()` validation to `min_delta`, `pruning_target`, and `RedConfig.from_env()`.

### Bug 8: AST Hygiene Bypass via Aliased Imports and Call-Form Mutable Defaults
* **Symptom:** `import time as t; t.time()`, `from time import time`, `datetime.now()`, and `def f(x=OrderedDict(), y=MyMutable())` bypassed naive AST checks.
* **Root Cause:** Checking only `node.func.value.id == "time"` or a small blacklist of mutable constructor names (`list`, `dict`, `set`) missed import aliases, `AsyncFunctionDef`, and custom mutable types.
* **Fix:** Tracked `ast.Import` and `ast.ImportFrom` aliases per file, inspected both `FunctionDef` and `AsyncFunctionDef`, and enforced an explicit `_IMMUTABLE_CALL_ALLOWLIST` on call-form defaults.

### Bug 9: Path Exemption Bug When Repo Lives Inside a Parent Folder Named `tests`
* **Symptom:** Checking `"tests" in filepath.parts` exempted every production module in `red/` from requiring `__all__` if the repository was cloned inside a parent directory named `/tmp/tests/repo`.
* **Root Cause:** Inspecting absolute path components instead of repository-relative paths leaked parent directory names into file classification.
* **Fix:** Computed `rel_path = filepath.resolve().relative_to(repo_root.resolve()).as_posix()` and checked `rel_path == "red/_time.py"` and `rel_path.startswith("tests/")`.

### Bug 10: Subscriber Exception Blocking Downstream Event Bus Listeners (Commandment 9)
* **Symptom:** When the first subscriber callback on `RedSharedState.publish("batch_event", ...)` raised `RuntimeError`, subsequent subscribers never received the event and the publisher crashed.
* **Root Cause:** Iterating over subscriber callbacks without per-callback exception isolation allowed a single faulty listener to abort event dispatch.
* **Fix:** Copied the subscriber list under lock, released the lock, and wrapped each callback invocation in `try ... except Exception:` logging the error and incrementing `subscriber_error_count`.
