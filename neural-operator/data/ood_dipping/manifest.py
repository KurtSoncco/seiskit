"""Manifest for Tier B ood_dipping: 7D Sobol × 30 Sobol seed levels.

Axes: Vs1, H, rH, aHV, dip_angle_deg, CoV, Vs2.
Soil–bedrock dip only; H in [25, 60] m so the center stays in soil under |dip|≤3°.
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
DATA_DIR = THIS_DIR.parent
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from sobol import (  # noqa: E402
    DEFAULT_DZ_1D,
    DEFAULT_MOTION_FREQ,
    DEFAULT_SAMPLER_SEED,
    ExecutionParameters,
    _discretize_length,
    bounds_CoV,
    bounds_rH,
    scale_Vs1,
    scale_Vs2,
    scale_aHV,
    sigma_Vs1,
    sigma_Vs2,
    sigma_aHV,
)

DEFAULT_PHYSICS_COUNT = 32
DEFAULT_SEED_LEVELS = 30
DEFAULT_RF_SEED_SEED = DEFAULT_SAMPLER_SEED + 1
RF_SEED_MIN = 1
RF_SEED_MAX = 9_999_999

DIP_SPAN = 500.0
DIP_HALF_SPAN = DIP_SPAN / 2.0
MIN_BEDROCK_BELOW_INTERFACE_M = 20.0
SHALLOW_RECORDER_DEPTH_M = 2.0

bounds_H = (25.0, 60.0)
bounds_dip = (-3.0, 3.0)

PHYSICS_COLUMNS = ("Vs1", "H", "rH", "aHV", "dip_angle_deg", "CoV", "Vs2")

MANIFEST_COLUMNS = (
    "index",
    "sobol_id",
    "replicate_id",
    "rf_seed",
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
class OodDippingEntry:
    index: int
    sobol_id: int
    replicate_id: int
    rf_seed: int
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


def dip_direction_for_angle(dip_angle_deg: float) -> str:
    if math.isclose(dip_angle_deg, 0.0, abs_tol=1e-12):
        return "flat"
    return "left_to_right" if dip_angle_deg > 0 else "right_to_left"


def max_interface_depth(H_discretized: float, dip_angle_deg: float) -> float:
    return H_discretized + DIP_HALF_SPAN * math.tan(math.radians(abs(dip_angle_deg)))


def min_bedrock_column_below_interface(
    H_discretized: float, dip_angle_deg: float, Lz_discretized: float
) -> float:
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


def unit_to_physical(unit: np.ndarray) -> np.ndarray:
    raw = np.asarray(unit, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != len(PHYSICS_COLUMNS):
        raise ValueError(f"Expected shape (n, {len(PHYSICS_COLUMNS)}), got {raw.shape}")
    phys = np.zeros_like(raw)
    phys[:, 0] = lognorm.ppf(raw[:, 0], s=sigma_Vs1, scale=scale_Vs1)
    phys[:, 1] = bounds_H[0] + raw[:, 1] * (bounds_H[1] - bounds_H[0])
    phys[:, 2] = bounds_rH[0] + raw[:, 2] * (bounds_rH[1] - bounds_rH[0])
    phys[:, 3] = lognorm.ppf(raw[:, 3], s=sigma_aHV, scale=scale_aHV)
    phys[:, 4] = bounds_dip[0] + raw[:, 4] * (bounds_dip[1] - bounds_dip[0])
    phys[:, 5] = bounds_CoV[0] + raw[:, 5] * (bounds_CoV[1] - bounds_CoV[0])
    phys[:, 6] = lognorm.ppf(raw[:, 6], s=sigma_Vs2, scale=scale_Vs2)
    return phys


def _valid_mask(phys: np.ndarray) -> np.ndarray:
    return (
        (phys[:, 0] >= 100.0)
        & (phys[:, 0] <= 360.0)
        & (phys[:, 1] >= bounds_H[0])
        & (phys[:, 1] <= bounds_H[1])
        & (phys[:, 2] >= bounds_rH[0])
        & (phys[:, 2] <= bounds_rH[1])
        & (phys[:, 3] >= 10.0)
        & (phys[:, 3] <= 50.0)
        & (phys[:, 4] >= bounds_dip[0])
        & (phys[:, 4] <= bounds_dip[1])
        & (phys[:, 5] >= bounds_CoV[0])
        & (phys[:, 5] <= bounds_CoV[1])
        & (phys[:, 6] >= 760.0)
        & (phys[:, 6] <= 1500.0)
    )


def generate_physics_samples(
    n: int = DEFAULT_PHYSICS_COUNT,
    sampler_seed: int | None = DEFAULT_SAMPLER_SEED,
) -> np.ndarray:
    sampler = qmc.Sobol(d=len(PHYSICS_COLUMNS), scramble=True, seed=sampler_seed)
    valid: list[np.ndarray] = []
    while len(valid) < n:
        remaining = n - len(valid)
        batch = max(2 ** math.ceil(math.log2(remaining + 32)), 64)
        phys = unit_to_physical(sampler.random(n=batch))
        valid.extend(phys[_valid_mask(phys)])
    return np.asarray(valid[:n], dtype=float)


def generate_seed_levels(
    n: int = DEFAULT_SEED_LEVELS,
    seed: int = DEFAULT_RF_SEED_SEED,
) -> np.ndarray:
    n_pow2 = 1 << math.ceil(math.log2(max(n, 1)))
    unit = qmc.Sobol(d=1, scramble=True, seed=seed).random(n=n_pow2)[:n, 0]
    return (RF_SEED_MIN + unit * (RF_SEED_MAX - RF_SEED_MIN)).astype(int)


def build_manifest(
    physics_count: int = DEFAULT_PHYSICS_COUNT,
    seed_levels: int = DEFAULT_SEED_LEVELS,
    sampler_seed: int | None = DEFAULT_SAMPLER_SEED,
    rf_seed_seed: int = DEFAULT_RF_SEED_SEED,
    dz_1D: float = DEFAULT_DZ_1D,
    motion_freq: float = DEFAULT_MOTION_FREQ,
) -> list[OodDippingEntry]:
    physical = generate_physics_samples(n=physics_count, sampler_seed=sampler_seed)
    seeds = generate_seed_levels(n=seed_levels, seed=rf_seed_seed)

    manifest: list[OodDippingEntry] = []
    global_index = 0
    for sobol_id, sample in enumerate(physical):
        Vs1, H, rH, aHV, dip_angle_deg, CoV, Vs2 = (float(v) for v in sample)
        execution = derive_dipping_execution_parameters(
            Vs1=Vs1,
            H=H,
            dip_angle_deg=dip_angle_deg,
            dz_1D=dz_1D,
            motion_freq=motion_freq,
        )
        direction = dip_direction_for_angle(dip_angle_deg)
        for replicate_id in range(seed_levels):
            manifest.append(
                OodDippingEntry(
                    index=global_index,
                    sobol_id=sobol_id,
                    replicate_id=replicate_id,
                    rf_seed=int(seeds[replicate_id]),
                    dip_angle_deg=dip_angle_deg,
                    dip_span=DIP_SPAN,
                    dip_direction=direction,
                    Vs1=Vs1,
                    Vs2=Vs2,
                    H_requested=execution.H_requested,
                    H_discretized=execution.H_discretized,
                    soil_layer_count=execution.soil_layer_count,
                    CoV=CoV,
                    rH=rH,
                    aHV=aHV,
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


def write_manifest_csv(path: str | Path, manifest: list[OodDippingEntry]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for entry in manifest:
            writer.writerow(entry.to_row())
    return output_path.resolve()


def load_manifest_csv(path: str | Path) -> list[OodDippingEntry]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            OodDippingEntry(
                index=int(row["index"]),
                sobol_id=int(row["sobol_id"]),
                replicate_id=int(row["replicate_id"]),
                rf_seed=int(row["rf_seed"]),
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
            for row in reader
        ]


DEFAULT_MANIFEST_PATH = THIS_DIR / "manifest.csv"


def ensure_manifest(
    path: str | Path = DEFAULT_MANIFEST_PATH,
    physics_count: int = DEFAULT_PHYSICS_COUNT,
    seed_levels: int = DEFAULT_SEED_LEVELS,
    sampler_seed: int | None = DEFAULT_SAMPLER_SEED,
    rf_seed_seed: int = DEFAULT_RF_SEED_SEED,
    overwrite: bool = False,
) -> list[OodDippingEntry]:
    manifest_path = Path(path)
    if not overwrite and manifest_path.exists():
        return load_manifest_csv(manifest_path)
    manifest = build_manifest(
        physics_count=physics_count,
        seed_levels=seed_levels,
        sampler_seed=sampler_seed,
        rf_seed_seed=rf_seed_seed,
    )
    write_manifest_csv(manifest_path, manifest)
    return manifest
