"""3×3 |TF| figure: one seed (seed 0) × all nodes."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _common import run_3x3  # noqa: E402

MODE = "one_seed_all_nodes"

if __name__ == "__main__":
    run_3x3(MODE)
