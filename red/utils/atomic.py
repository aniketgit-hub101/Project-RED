"""
red.utils.atomic
Commandment 8: Crash-safe atomic file writing utility (.tmp + fsync + os.replace).
Note: Per Master line 424, Windows support is deferred; directory fsync runs on POSIX.
"""

import os
import tempfile
from pathlib import Path
from typing import AnyStr, Union

__all__ = ["atomic_write"]

_ALLOWED_MODES = {"w", "wb"}
_PROC_STATUS = Path("/proc/self/status")


def _default_file_mode() -> int:
    try:
        for line in _PROC_STATUS.read_text(encoding="utf-8").splitlines():
            if line.startswith("Umask:"):
                umask_val = int(line.split()[1], 8)
                return 0o666 & ~umask_val
    except OSError:
        pass
    return 0o600


def atomic_write(filepath: Union[str, Path], content: AnyStr, mode: str = "w") -> None:
    if mode not in _ALLOWED_MODES:
        raise ValueError(f"Unsupported mode {mode!r}; only 'w' and 'wb' are permitted.")

    target = Path(filepath)
    target.parent.mkdir(parents=True, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    fd_owned = True
    committed = False

    try:
        file_obj = os.fdopen(fd, mode)
        fd_owned = False
        with file_obj as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())

        os.chmod(temp_path, _default_file_mode())
        os.replace(temp_path, target)
        committed = True

        if os.name == "posix" and hasattr(os, "O_RDONLY"):
            try:
                dir_fd = os.open(target.parent, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError as exc:
                err = OSError(
                    exc.errno,
                    f"POST_COMMIT_DIR_FSYNC_FAILED: Atomic replacement committed at '{target}', "
                    f"but post-commit directory fsync failed: {exc}",
                )
                raise err from exc
    except BaseException:
        if fd_owned:
            try:
                os.close(fd)
            except OSError:
                pass
        if not committed:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass
        raise
