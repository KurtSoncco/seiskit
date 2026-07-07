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

from seiskit.gaussian_field import create_three_layer_vs_realization  # noqa: E402
from seiskit.plot_results import plot_realization  # noqa: E402

LX_VARIABILITY = 500.0
BC_WIDTH = 500.0
LX = LX_VARIABILITY + 2 * BC_WIDTH
DX = 1.0
DZ = 1.0

PLOTS_DIR = THIS_DIR / "plots"

if __name__ == "__main__":
    manifest_entries = ensure_manifest(path=DEFAULT_MANIFEST_PATH, overwrite=True)
    write_manifest_csv(DEFAULT_MANIFEST_PATH, manifest_entries)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for entry in manifest_entries:
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
        save_path = PLOTS_DIR / f"case_{entry.index}.png"
        plot_realization(
            Vs_1D_profile=np.array([entry.Vs1, entry.Vs_bedrock]),
            Vs_realization=Vs_extended,
            Lx=LX,
            Lz=entry.Lz_discretized,
            dx=DX,
            dz=DZ,
            save_path=save_path,
            title=(
                f"3-layer case {entry.index}: Vs1={entry.Vs1:.0f} H1={entry.H1_discretized:.0f}m, "
                f"Vs_mid={entry.Vs_mid:.0f} H2={entry.H2_discretized:.0f}m, "
                f"Vs_bedrock={entry.Vs_bedrock:.0f}"
            ),
            bedrock_mask=bedrock_mask,
        )
        print(f"Wrote {save_path}")
