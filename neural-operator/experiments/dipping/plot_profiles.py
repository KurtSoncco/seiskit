"""Visual QA: plot a stratified subset of dipping Vs realizations before OpenSees.

Plots one case per (split, signed angle) combination present in the manifest.

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
    MIN_BEDROCK_BELOW_INTERFACE_M,
    SHALLOW_RECORDER_DEPTH_M,
    ensure_manifest,
    load_manifest_csv,
    min_bedrock_column_below_interface,
)

from seiskit.gaussian_field import create_dipping_vs_realization  # noqa: E402
from seiskit.plot_results import plot_realization  # noqa: E402

LX_VARIABILITY = 500.0
BC_WIDTH = 500.0
LX = LX_VARIABILITY + 2 * BC_WIDTH
DX = 1.0
DZ = 1.0

PLOTS_DIR = THIS_DIR / "plots"


def _select_stratified(entries):
    """Pick the first manifest row for each (split, dip_angle_deg) pair."""
    seen: set[tuple[str, float]] = set()
    selected = []
    for entry in entries:
        key = (entry.split, entry.dip_angle_deg)
        if key in seen:
            continue
        seen.add(key)
        selected.append(entry)
    return selected


def _parse_args():
    parser = argparse.ArgumentParser(description="Plot dipping Vs realizations for visual QA.")
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

    for entry in _select_stratified(manifest_entries):
        Vs_profile_1D = np.concatenate(
            [
                np.full(entry.soil_layer_count, entry.Vs1, dtype=float),
                np.full(entry.bedrock_layer_count, entry.Vs2, dtype=float),
            ]
        )
        Vs_extended, _x_total, _z, _h, bedrock_mask = create_dipping_vs_realization(
            Vs_profile=Vs_profile_1D,
            Lx=LX,
            Lx_variability=LX_VARIABILITY,
            Lz=entry.Lz_discretized,
            dx=DX,
            dz=DZ,
            rH=entry.rH,
            aHV=entry.aHV,
            CV=entry.CoV,
            dip_angle_deg=entry.dip_angle_deg,
            dip_span=entry.dip_span,
            seed=entry.rf_seed,
            dz_1D=1.0,
        )
        bedrock_below = min_bedrock_column_below_interface(
            entry.H_discretized, entry.dip_angle_deg, entry.Lz_discretized
        )
        if bedrock_below + 1e-6 < MIN_BEDROCK_BELOW_INTERFACE_M:
            print(
                f"WARNING case {entry.index}: bedrock below deepest interface is "
                f"{bedrock_below:.1f}m < {MIN_BEDROCK_BELOW_INTERFACE_M:.0f}m"
            )
        save_path = PLOTS_DIR / (
            f"case_{entry.index}_{entry.split}_{entry.dip_angle_deg:+.0f}deg_{entry.dip_direction}.png"
        )
        plot_realization(
            Vs_1D_profile=Vs_profile_1D,
            Vs_realization=Vs_extended,
            Lx=LX,
            Lz=entry.Lz_discretized,
            dx=DX,
            dz=DZ,
            save_path=save_path,
            title=(
                f"Dipping case {entry.index} [{entry.split}]: dip={entry.dip_angle_deg:+.0f} deg "
                f"over {entry.dip_span:.0f}m ({entry.dip_direction}), "
                f"Vs1={entry.Vs1:.0f}, Vs2={entry.Vs2:.0f}, H={entry.H_discretized:.0f}m, "
                f"Lz={entry.Lz_discretized:.0f}m, bedrock>={bedrock_below:.0f}m, "
                f"recorder@y={SHALLOW_RECORDER_DEPTH_M:.0f}m,Lz"
            ),
            bedrock_mask=bedrock_mask,
        )
        print(f"Wrote {save_path}")
