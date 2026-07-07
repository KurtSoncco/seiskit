"""Manifest for the dipping-interface capability-check experiment.

Exactly two cases at a fixed background 2-layer profile (the same
single-layer distributions used for the model's main 2-layer training data,
pinned at representative/median values), differing only in the direction of
a straight 5-degree dipping interface. See
``neural-operator/experiments/README.md``.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent.parent / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from sobol import (  # noqa: E402
    DEFAULT_BEDROCK_THICKNESS,
    DEFAULT_DZ_1D,
    DEFAULT_MOTION_FREQ,
    _discretize_length,
    bounds_CoV,
    bounds_rH,
    scale_aHV,
    scale_Vs1,
    scale_Vs2,
)

# Two cases only: dip left-to-right (+2 deg) and right-to-left (-2 deg).
# A steeper dip (e.g. 5 deg) applied across the full variability width swings
# the interface far deeper than the soil layer is thick, which isn't
# physically plausible for a soil deposit; at 2 deg the full-width swing
# (~8.7m over the 500m variability width) stays comfortably within H=15m.
DIP_ANGLES_DEG = (2.0, -2.0)
DIP_SPAN = 500.0  # Horizontal extent [m] that dips == full Lx_variability.
H_FIXED = 15.0
COV_FIXED = sum(bounds_CoV) / 2.0
RH_FIXED = sum(bounds_rH) / 2.0
AHV_FIXED = scale_aHV
SEED_FIXED = 42

MANIFEST_COLUMNS = (
    "index",
    "dip_angle_deg",
    "dip_span",
    "dip_direction",
    "Vs1",
    "Vs2",
    "H_discretized",
    "soil_layer_count",
    "CoV",
    "rH",
    "aHV",
    "bedrock_thickness_discretized",
    "bedrock_layer_count",
    "Lz_discretized",
    "seed",
    "motion_freq",
    "f0_effective",
    "duration",
    "damping_freq_first",
)


@dataclass(frozen=True)
class DippingManifestEntry:
    index: int
    dip_angle_deg: float
    dip_span: float
    dip_direction: str
    Vs1: float
    Vs2: float
    H_discretized: float
    soil_layer_count: int
    CoV: float
    rH: float
    aHV: float
    bedrock_thickness_discretized: float
    bedrock_layer_count: int
    Lz_discretized: float
    seed: int
    motion_freq: float
    f0_effective: float
    duration: float
    damping_freq_first: float

    def to_row(self) -> dict[str, int | float | str]:
        return asdict(self)


def build_manifest() -> list[DippingManifestEntry]:
    """Build the 2-row dipping-interface manifest (one row per dip direction)."""
    soil_layer_count, H_discretized = _discretize_length(H_FIXED, DEFAULT_DZ_1D)
    bedrock_layer_count, bedrock_thickness_discretized = _discretize_length(
        DEFAULT_BEDROCK_THICKNESS, DEFAULT_DZ_1D
    )
    Lz_discretized = (soil_layer_count + bedrock_layer_count) * DEFAULT_DZ_1D
    f0_effective = scale_Vs1 / (4 * H_discretized)
    duration = 50.0 if f0_effective < 1.0 else 30.0
    damping_freq_first = min(f0_effective, DEFAULT_MOTION_FREQ)

    manifest: list[DippingManifestEntry] = []
    for index, dip_angle_deg in enumerate(DIP_ANGLES_DEG):
        direction = "left_to_right" if dip_angle_deg > 0 else "right_to_left"
        manifest.append(
            DippingManifestEntry(
                index=index,
                dip_angle_deg=dip_angle_deg,
                dip_span=DIP_SPAN,
                dip_direction=direction,
                Vs1=scale_Vs1,
                Vs2=scale_Vs2,
                H_discretized=H_discretized,
                soil_layer_count=soil_layer_count,
                CoV=COV_FIXED,
                rH=RH_FIXED,
                aHV=AHV_FIXED,
                bedrock_thickness_discretized=bedrock_thickness_discretized,
                bedrock_layer_count=bedrock_layer_count,
                Lz_discretized=Lz_discretized,
                seed=SEED_FIXED,
                motion_freq=DEFAULT_MOTION_FREQ,
                f0_effective=f0_effective,
                duration=duration,
                damping_freq_first=damping_freq_first,
            )
        )
    return manifest


def write_manifest_csv(path: str | Path, manifest: list[DippingManifestEntry]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for entry in manifest:
            writer.writerow(entry.to_row())
    return output_path.resolve()


DEFAULT_MANIFEST_PATH = THIS_DIR / "manifest.csv"


def ensure_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> list[DippingManifestEntry]:
    return build_manifest()
