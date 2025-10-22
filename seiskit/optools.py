"""OpenSees-related utilities (pure-Python helpers).

This module intentionally avoids importing the OpenSees binary and instead
provides helpers for file parsing, result post-processing, and command
generation that are testable without OpenSees installed.
"""

from typing import List


def read_time_series(path: str) -> List[float]:
    """Read a whitespace-separated column of floats from a file.

    Returns the list of floats. Ignores blank lines and comments starting with '#'.
    """
    vals: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            try:
                v = float(parts[0])
            except Exception:
                continue
            vals.append(v)
    return vals
