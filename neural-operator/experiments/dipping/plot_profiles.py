"""Visual QA: plot each manifest case's Vs realization before running OpenSees.

Usage: python plot_profiles.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from manifest import DEFAULT_MANIFEST_PATH, ensure_manifest, write_manifest_csv  # noqa: E402

from seiskit.gaussian_field import create_dipping_vs_realization  # noqa: E402
from seiskit.plot_results import plot_realization  # noqa: E402

LX_VARIABILITY = 500.0
BC_WIDTH = 500.0
LX = LX_VARIABILITY + 2 * BC_WIDTH
DX = 1.0
DZ = 1.0

PLOTS_DIR = THIS_DIR / "plots"

if __name__ == "__main__":
    manifest_entries = ensure_manifest(path=DEFAULT_MANIFEST_PATH)
    write_manifest_csv(DEFAULT_MANIFEST_PATH, manifest_entries)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for entry in manifest_entries:
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
            dip_span=LX_VARIABILITY,
            seed=entry.seed,
            dz_1D=1.0,
        )
        save_path = PLOTS_DIR / f"case_{entry.index}_{entry.dip_direction}.png"
        plot_realization(
            Vs_1D_profile=Vs_profile_1D,
            Vs_realization=Vs_extended,
            Lx=LX,
            Lz=entry.Lz_discretized,
            dx=DX,
            dz=DZ,
            save_path=save_path,
            title=(
                f"Dipping case {entry.index}: dip={entry.dip_angle_deg:+.0f} deg over "
                f"{entry.dip_span:.0f}m span ({entry.dip_direction}), Vs1={entry.Vs1:.0f}, "
                f"Vs2={entry.Vs2:.0f}, H={entry.H_discretized:.0f}m"
            ),
            bedrock_mask=bedrock_mask,
        )
        print(f"Wrote {save_path}")
