"""3×3 |TF| figure: all seeds × all nodes (flattened pool)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _common import run_3x3  # noqa: E402

MODE = "all_seeds_all_nodes"

if __name__ == "__main__":
    run_3x3(MODE)
