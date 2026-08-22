"""CLI for qualitative |TF| 3×3 sensitivity figures.

Usage
-----
python run_tf.py --mode center_node_one_seed
python run_tf.py --mode center_node_all_seeds
python run_tf.py --mode all  # all four modes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import Mode, run_3x3  # noqa: E402

MODES: tuple[Mode, ...] = (
    "center_node_one_seed",
    "center_node_all_seeds",
    "one_seed_all_nodes",
    "all_seeds_all_nodes",
)


def main() -> None:
    p = argparse.ArgumentParser(description="Qualitative |TF| 3×3 figures (cross layout removed).")
    p.add_argument(
        "--mode",
        choices=[*MODES, "all"],
        default="all",
        help="Sampling mode (default: all)",
    )
    args = p.parse_args()
    modes = MODES if args.mode == "all" else (args.mode,)
    for mode in modes:
        run_3x3(mode)


if __name__ == "__main__":
    main()
