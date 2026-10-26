"""
red._time
Wall-clock UTC isolation module per DG-33 and Commandment 2.
"""

import time

__all__ = ["wall_clock_utc"]


def wall_clock_utc() -> float:
    """Return current UTC timestamp in seconds since the Unix epoch as a float."""
    return time.time()
