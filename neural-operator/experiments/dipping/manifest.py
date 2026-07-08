"""Manifest for the dipping-interface generalization experiment.

Samples a signed dip-angle grid with independent Sobol backgrounds over
(Vs1, Vs2, H) from the main training marginals, 15 RF seeds per angle, and
train/interpolation/extrapolation split tags. See
``neural-operator/experiments/README.md``.
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.stats import lognorm, qmc

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent.parent / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from sobol import (  # noqa: E402
    DEFAULT_DZ_1D,
    DEFAULT_MOTION_FREQ,
    DEFAULT_SAMPLER_SEED,
    ExecutionParameters,
    _discretize_length,
    bounds_CoV,
    bounds_H,
    bounds_rH,
    generate_rf_seed_matrix,
    scale_aHV,
    scale_Vs1,
    scale_Vs2,
    sigma_Vs1,
    sigma_Vs2,
)

# Signed dip-angle grid (degrees).
DIP_ANGLES_DEG = (-5.0, -4.0, -3.0, -2.0, -1.0, 1.0, 2.0, 3.0, 4.0, 5.0)
DIP_SPAN = 500.0  # Horizontal extent [m] == full Lx_variability.
DIP_HALF_SPAN = DIP_SPAN / 2.0
MIN_BEDROCK_BELOW_INTERFACE_M = 20.0
SHALLOW_RECORDER_DEPTH_M = 2.0

DEFAULT_SAMPLES_PER_ANGLE = 15
DEFAULT_SEEDS_PER_ANGLE = 15
DEFAULT_RF_SEED_SEED = 42

COV_FIXED = sum(bounds_CoV) / 2.0
RH_FIXED = sum(bounds_rH) / 2.0
AHV_FIXED = scale_aHV

BACKGROUND_COLUMNS = ("Vs1", "H", "Vs2")

MANIFEST_COLUMNS = (
    "index",
    "angle_id",
    "sobol_id",
    "replicate_id",
    "rf_seed",
    "split",
    "dip_angle_deg",
    "dip_span",
    "dip_direction",
    "Vs1",
    "Vs2",
    "H_requested",
    "H_discretized",
    "soil_layer_count",
    "CoV",
    "rH",
    "aHV",
    "bedrock_thickness_discretized",
    "bedrock_layer_count",
    "Lz_discretized",
    "motion_freq",
    "f0_effective",
    "duration",
    "damping_freq_first",
)


@dataclass(frozen=True)
class DippingManifestEntry:
    index: int
    angle_id: int
    sobol_id: int
    replicate_id: int
    rf_seed: int
    split: str
    dip_angle_deg: float
    dip_span: float
    dip_direction: str
    Vs1: float
    Vs2: float
    H_requested: float
    H_discretized: float
    soil_layer_count: int
    CoV: float
    rH: float
    aHV: float
    bedrock_thickness_discretized: float
    bedrock_layer_count: int
    Lz_discretized: float
    motion_freq: float
    f0_effective: float
    duration: float
    damping_freq_first: float

    def to_row(self) -> dict[str, int | float | str]:
        return asdict(self)

    @property
    def seed(self) -> int:
        """Backward-compatible alias used by older runners."""
        return self.rf_seed


def split_for_angle(dip_angle_deg: float) -> str:
    """Return the train/interp/extrap split tag for a signed dip angle."""
    magnitude = abs(dip_angle_deg)
    if math.isclose(magnitude, 2.0):
        return "interp_test"
    if math.isclose(magnitude, 5.0):
        return "extrap_test"
    return "train"


def dip_direction_for_angle(dip_angle_deg: float) -> str:
    return "left_to_right" if dip_angle_deg > 0 else "right_to_left"


def max_interface_depth(H_discretized: float, dip_angle_deg: float) -> float:
    """Deepest soil/bedrock interface over the dipping span (pivot at domain center)."""
    return H_discretized + DIP_HALF_SPAN * math.tan(math.radians(abs(dip_angle_deg)))


def min_bedrock_column_below_interface(
    H_discretized: float, dip_angle_deg: float, Lz_discretized: float
) -> float:
    """Minimum bedrock thickness [m] below the deepest interface point."""
    return Lz_discretized - max_interface_depth(H_discretized, dip_angle_deg)


def derive_dipping_execution_parameters(
    Vs1: float,
    H: float,
    dip_angle_deg: float,
    *,
    dz_1D: float = DEFAULT_DZ_1D,
    min_bedrock_below_interface: float = MIN_BEDROCK_BELOW_INTERFACE_M,
    motion_freq: float = DEFAULT_MOTION_FREQ,
) -> ExecutionParameters:
    """Derive mesh depth so every column has ``min_bedrock_below_interface`` bedrock."""
    soil_layer_count, H_discretized = _discretize_length(H, dz_1D)
    deepest_interface = max_interface_depth(H_discretized, dip_angle_deg)
    total_layer_count, Lz_discretized = _discretize_length(
        deepest_interface + min_bedrock_below_interface,
        dz_1D,
    )
    bedrock_layer_count = max(1, total_layer_count - soil_layer_count)
    Lz_discretized = (soil_layer_count + bedrock_layer_count) * dz_1D
    while (
        min_bedrock_column_below_interface(H_discretized, dip_angle_deg, Lz_discretized) + 1e-6
        < min_bedrock_below_interface
    ):
        bedrock_layer_count += 1
        Lz_discretized = (soil_layer_count + bedrock_layer_count) * dz_1D
    bedrock_thickness_discretized = bedrock_layer_count * dz_1D

    f0_effective = Vs1 / (4 * H_discretized)
    duration = 50.0 if f0_effective < 1.0 else 30.0
    damping_freq_first = min(f0_effective, motion_freq)
    return ExecutionParameters(
        H_requested=H,
        H_discretized=H_discretized,
        soil_layer_count=soil_layer_count,
        dz_1D=dz_1D,
        bedrock_thickness=bedrock_thickness_discretized,
        bedrock_thickness_discretized=bedrock_thickness_discretized,
        bedrock_layer_count=bedrock_layer_count,
        Lz_discretized=Lz_discretized,
        motion_freq=motion_freq,
        f0_effective=f0_effective,
        duration=duration,
        damping_freq_first=damping_freq_first,
    )


def _unit_to_background(unit_samples: np.ndarray) -> np.ndarray:
    """Map unit-cube samples with shape ``(n, 3)`` to (Vs1, H, Vs2)."""
    raw = np.asarray(unit_samples, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != len(BACKGROUND_COLUMNS):
        raise ValueError(
            f"Expected samples with shape (n, {len(BACKGROUND_COLUMNS)}), got {raw.shape}."
        )
    phys = np.zeros_like(raw)
    phys[:, 0] = lognorm.ppf(raw[:, 0], s=sigma_Vs1, scale=scale_Vs1)
    phys[:, 1] = bounds_H[0] + raw[:, 1] * (bounds_H[1] - bounds_H[0])
    phys[:, 2] = lognorm.ppf(raw[:, 2], s=sigma_Vs2, scale=scale_Vs2)
    return phys


def _background_bounds_mask(physical: np.ndarray) -> np.ndarray:
    return (
        (physical[:, 0] >= 100.0)
        & (physical[:, 0] <= 360.0)
        & (physical[:, 1] >= bounds_H[0])
        & (physical[:, 1] <= bounds_H[1])
        & (physical[:, 2] >= 760.0)
        & (physical[:, 2] <= 1500.0)
    )


def generate_background_samples(
    target_count: int,
    dip_angle_deg: float,
    sampler_seed: int | None = DEFAULT_SAMPLER_SEED,
    angle_id: int = 0,
) -> np.ndarray:
    """Generate ``target_count`` valid (Vs1, H, Vs2) backgrounds for one dip angle."""
    if target_count <= 0:
        raise ValueError("target_count must be positive.")

    seed = None if sampler_seed is None else int(sampler_seed) + angle_id * 10_000
    sampler = qmc.Sobol(d=len(BACKGROUND_COLUMNS), scramble=True, seed=seed)

    valid: list[np.ndarray] = []
    while len(valid) < target_count:
        remaining = target_count - len(valid)
        batch_size = max(2 ** math.ceil(math.log2(remaining + 32)), 64)
        raw = sampler.random(n=batch_size)
        physical = _unit_to_background(raw)
        valid.extend(physical[_background_bounds_mask(physical)])

    return np.asarray(valid[:target_count], dtype=float)


def build_manifest(
    samples_per_angle: int = DEFAULT_SAMPLES_PER_ANGLE,
    seeds_per_angle: int = DEFAULT_SEEDS_PER_ANGLE,
    sampler_seed: int | None = DEFAULT_SAMPLER_SEED,
    rf_seed_seed: int = DEFAULT_RF_SEED_SEED,
    dz_1D: float = DEFAULT_DZ_1D,
    motion_freq: float = DEFAULT_MOTION_FREQ,
) -> list[DippingManifestEntry]:
    """Build the flattened dipping manifest (one row per angle × background × seed)."""
    if samples_per_angle != seeds_per_angle:
        raise ValueError(
            "samples_per_angle and seeds_per_angle must match for the 1:1 "
            f"(sobol_id, rf_seed) pairing; got {samples_per_angle} and {seeds_per_angle}."
        )

    manifest: list[DippingManifestEntry] = []
    global_index = 0

    for angle_id, dip_angle_deg in enumerate(DIP_ANGLES_DEG):
        backgrounds = generate_background_samples(
            target_count=samples_per_angle,
            dip_angle_deg=dip_angle_deg,
            sampler_seed=sampler_seed,
            angle_id=angle_id,
        )
        rf_seeds = generate_rf_seed_matrix(
            sample_count=samples_per_angle,
            seeds_per_sample=1,
            seed=rf_seed_seed + angle_id,
        )[:, 0]

        split = split_for_angle(dip_angle_deg)
        direction = dip_direction_for_angle(dip_angle_deg)

        for sobol_id in range(samples_per_angle):
            Vs1, H_requested, Vs2 = (float(v) for v in backgrounds[sobol_id])
            execution = derive_dipping_execution_parameters(
                Vs1=Vs1,
                H=H_requested,
                dip_angle_deg=dip_angle_deg,
                dz_1D=dz_1D,
                motion_freq=motion_freq,
            )
            rf_seed = int(rf_seeds[sobol_id])

            manifest.append(
                DippingManifestEntry(
                    index=global_index,
                    angle_id=angle_id,
                    sobol_id=sobol_id,
                    replicate_id=sobol_id,
                    rf_seed=rf_seed,
                    split=split,
                    dip_angle_deg=dip_angle_deg,
                    dip_span=DIP_SPAN,
                    dip_direction=direction,
                    Vs1=Vs1,
                    Vs2=Vs2,
                    H_requested=execution.H_requested,
                    H_discretized=execution.H_discretized,
                    soil_layer_count=execution.soil_layer_count,
                    CoV=COV_FIXED,
                    rH=RH_FIXED,
                    aHV=AHV_FIXED,
                    bedrock_thickness_discretized=execution.bedrock_thickness_discretized,
                    bedrock_layer_count=execution.bedrock_layer_count,
                    Lz_discretized=execution.Lz_discretized,
                    motion_freq=execution.motion_freq,
                    f0_effective=execution.f0_effective,
                    duration=execution.duration,
                    damping_freq_first=execution.damping_freq_first,
                )
            )
            global_index += 1

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


def load_manifest_csv(path: str | Path) -> list[DippingManifestEntry]:
    manifest_path = Path(path)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        entries: list[DippingManifestEntry] = []
        for row in reader:
            entries.append(
                DippingManifestEntry(
                    index=int(row["index"]),
                    angle_id=int(row["angle_id"]),
                    sobol_id=int(row["sobol_id"]),
                    replicate_id=int(row["replicate_id"]),
                    rf_seed=int(row["rf_seed"]),
                    split=str(row["split"]),
                    dip_angle_deg=float(row["dip_angle_deg"]),
                    dip_span=float(row["dip_span"]),
                    dip_direction=str(row["dip_direction"]),
                    Vs1=float(row["Vs1"]),
                    Vs2=float(row["Vs2"]),
                    H_requested=float(row["H_requested"]),
                    H_discretized=float(row["H_discretized"]),
                    soil_layer_count=int(row["soil_layer_count"]),
                    CoV=float(row["CoV"]),
                    rH=float(row["rH"]),
                    aHV=float(row["aHV"]),
                    bedrock_thickness_discretized=float(row["bedrock_thickness_discretized"]),
                    bedrock_layer_count=int(row["bedrock_layer_count"]),
                    Lz_discretized=float(row["Lz_discretized"]),
                    motion_freq=float(row["motion_freq"]),
                    f0_effective=float(row["f0_effective"]),
                    duration=float(row["duration"]),
                    damping_freq_first=float(row["damping_freq_first"]),
                )
            )
    return entries


DEFAULT_MANIFEST_PATH = THIS_DIR / "manifest.csv"


def ensure_manifest(
    path: str | Path = DEFAULT_MANIFEST_PATH,
    samples_per_angle: int = DEFAULT_SAMPLES_PER_ANGLE,
    seeds_per_angle: int = DEFAULT_SEEDS_PER_ANGLE,
    sampler_seed: int | None = DEFAULT_SAMPLER_SEED,
    rf_seed_seed: int = DEFAULT_RF_SEED_SEED,
    overwrite: bool = False,
) -> list[DippingManifestEntry]:
    manifest_path = Path(path)
    if not overwrite and manifest_path.exists():
        return load_manifest_csv(manifest_path)

    manifest = build_manifest(
        samples_per_angle=samples_per_angle,
        seeds_per_angle=seeds_per_angle,
        sampler_seed=sampler_seed,
        rf_seed_seed=rf_seed_seed,
    )
    write_manifest_csv(manifest_path, manifest)
    return manifest
