"""Visual QA: plot one Vs realization per topology point before running OpenSees.

Usage: python plot_profiles.py [--manifest-path PATH]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from manifest import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    ensure_manifest,
    load_manifest_csv,
)

from seiskit.gaussian_field import create_three_layer_vs_realization  # noqa: E402
from seiskit.plot_results import plot_realization  # noqa: E402

LX_VARIABILITY = 500.0
BC_WIDTH = 500.0
LX = LX_VARIABILITY + 2 * BC_WIDTH
DX = 1.0
DZ = 1.0

PLOTS_DIR = THIS_DIR / "plots"


def _select_one_per_topology(entries):
    seen: set[int] = set()
    selected = []
    for entry in entries:
        if entry.topology_id in seen:
            continue
        seen.add(entry.topology_id)
        selected.append(entry)
    return selected


def _parse_args():
    parser = argparse.ArgumentParser(description="Plot three-layer Vs realizations for visual QA.")
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--overwrite-manifest", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    manifest_path = args.manifest_path
    if args.overwrite_manifest or not manifest_path.exists():
        manifest_entries = ensure_manifest(path=manifest_path, overwrite=True)
    else:
        manifest_entries = load_manifest_csv(manifest_path)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for entry in _select_one_per_topology(manifest_entries):
        Vs_extended, _x_total, _z, (h1, h2), bedrock_mask = create_three_layer_vs_realization(
            Vs1=entry.Vs1,
            Vs_mid=entry.Vs_mid,
            Vs_bedrock=entry.Vs_bedrock,
            H1=entry.H1_discretized,
            H2=entry.H2_discretized,
            Lx=LX,
            Lx_variability=LX_VARIABILITY,
            Lz=entry.Lz_discretized,
            dx=DX,
            dz=DZ,
            rH1=entry.rH1,
            aHV1=entry.aHV1,
            CV1=entry.CoV1,
            rH2=entry.rH2,
            aHV2=entry.aHV2,
            CV2=entry.CoV2,
            seed1=entry.seed1,
            seed2=entry.seed2,
        )
        save_path = PLOTS_DIR / f"case_{entry.index}_topo{entry.topology_id}_{entry.split}.png"
        plot_realization(
            Vs_1D_profile=np.array([entry.Vs1, entry.Vs_bedrock]),
            Vs_realization=Vs_extended,
            Lx=LX,
            Lz=entry.Lz_discretized,
            dx=DX,
            dz=DZ,
            save_path=save_path,
            title=(
                f"3-layer case {entry.index} [{entry.split}] topo={entry.topology_id}: "
                f"Vs1={entry.Vs1:.0f} H1={entry.H1_discretized:.0f}m, "
                f"Vs_mid={entry.Vs_mid:.0f} H2={entry.H2_discretized:.0f}m, "
                f"contrast={entry.Vs_contrast:.2f}"
            ),
            bedrock_mask=bedrock_mask,
        )
        print(f"Wrote {save_path}")
