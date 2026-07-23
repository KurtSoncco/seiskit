"""Cross-layout |TF| figure: center node × all seeds."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _common import run_cross  # noqa: E402

MODE = "center_node_all_seeds"

if __name__ == "__main__":
    run_cross(MODE)
