"""Manifest for Tier B ood_three_layer: 8D Sobol × 30 Sobol seed levels.

Axes: Vs_mid, H1, H2, rH, aHV, Vs_contrast, CoV, Vs_bedrock.
Shared (rH, aHV, CoV) across both soil layers. Flat interfaces, no wave.
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
    DEFAULT_BEDROCK_THICKNESS,
    DEFAULT_DZ_1D,
    DEFAULT_MOTION_FREQ,
    DEFAULT_SAMPLER_SEED,
    _discretize_length,
    bounds_CoV,
    bounds_rH,
    lognormal_parameter,
    scale_aHV,
    scale_Vs2,
    sigma_aHV,
    sigma_Vs2,
)

DEFAULT_PHYSICS_COUNT = 32
DEFAULT_SEED_LEVELS = 30
DEFAULT_RF_SEED_SEED = DEFAULT_SAMPLER_SEED + 1
RF_SEED_MIN = 1
RF_SEED_MAX = 9_999_999

scale_Vs_mid, sigma_Vs_mid = lognormal_parameter(450.0, 560.0)
bounds_H1 = (5.0, 12.0)
bounds_H2 = (5.0, 12.0)
bounds_Vs_contrast = (0.8, 1.6)
MAX_H_SUM = 24.0

PHYSICS_COLUMNS = (
    "Vs_mid",
    "H1",
    "H2",
    "rH",
    "aHV",
    "Vs_contrast",
    "CoV",
    "Vs_bedrock",
)

MANIFEST_COLUMNS = (
    "index",
    "sobol_id",
    "replicate_id",
    "Vs1",
    "H1_requested",
    "H1_discretized",
    "layer1_count",
    "CoV1",
    "rH1",
    "aHV1",
    "Vs_mid",
    "H2_requested",
    "H2_discretized",
    "layer2_count",
    "CoV2",
    "rH2",
    "aHV2",
    "Vs_contrast",
    "Vs_bedrock",
    "bedrock_thickness_discretized",
    "bedrock_layer_count",
    "Lz_discretized",
    "seed1",
    "seed2",
    "motion_freq",
    "f0_effective",
    "duration",
    "damping_freq_first",
)


@dataclass(frozen=True)
class OodThreeLayerEntry:
    index: int
    sobol_id: int
    replicate_id: int
    Vs1: float
    H1_requested: float
    H1_discretized: float
    layer1_count: int
    CoV1: float
    rH1: float
    aHV1: float
    Vs_mid: float
    H2_requested: float
    H2_discretized: float
    layer2_count: int
    CoV2: float
    rH2: float
    aHV2: float
    Vs_contrast: float
    Vs_bedrock: float
    bedrock_thickness_discretized: float
    bedrock_layer_count: int
    Lz_discretized: float
    seed1: int
    seed2: int
    motion_freq: float
    f0_effective: float
    duration: float
    damping_freq_first: float

    def to_row(self) -> dict[str, int | float]:
        return asdict(self)


def unit_to_physical(unit: np.ndarray) -> np.ndarray:
    raw = np.asarray(unit, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != len(PHYSICS_COLUMNS):
        raise ValueError(f"Expected shape (n, {len(PHYSICS_COLUMNS)}), got {raw.shape}")
    phys = np.zeros_like(raw)
    phys[:, 0] = lognorm.ppf(raw[:, 0], s=sigma_Vs_mid, scale=scale_Vs_mid)
    phys[:, 1] = bounds_H1[0] + raw[:, 1] * (bounds_H1[1] - bounds_H1[0])
    phys[:, 2] = bounds_H2[0] + raw[:, 2] * (bounds_H2[1] - bounds_H2[0])
    phys[:, 3] = bounds_rH[0] + raw[:, 3] * (bounds_rH[1] - bounds_rH[0])
    phys[:, 4] = lognorm.ppf(raw[:, 4], s=sigma_aHV, scale=scale_aHV)
    phys[:, 5] = bounds_Vs_contrast[0] + raw[:, 5] * (
        bounds_Vs_contrast[1] - bounds_Vs_contrast[0]
    )
    phys[:, 6] = bounds_CoV[0] + raw[:, 6] * (bounds_CoV[1] - bounds_CoV[0])
    phys[:, 7] = lognorm.ppf(raw[:, 7], s=sigma_Vs2, scale=scale_Vs2)
    return phys


def _valid_mask(phys: np.ndarray) -> np.ndarray:
    return (
        (phys[:, 0] >= 450.0)
        & (phys[:, 0] <= 560.0)
        & (phys[:, 1] >= bounds_H1[0])
        & (phys[:, 1] <= bounds_H1[1])
        & (phys[:, 2] >= bounds_H2[0])
        & (phys[:, 2] <= bounds_H2[1])
        & (phys[:, 1] + phys[:, 2] <= MAX_H_SUM)
        & (phys[:, 3] >= bounds_rH[0])
        & (phys[:, 3] <= bounds_rH[1])
        & (phys[:, 4] >= 10.0)
        & (phys[:, 4] <= 50.0)
        & (phys[:, 5] >= bounds_Vs_contrast[0])
        & (phys[:, 5] <= bounds_Vs_contrast[1])
        & (phys[:, 6] >= bounds_CoV[0])
        & (phys[:, 6] <= bounds_CoV[1])
        & (phys[:, 7] >= 760.0)
        & (phys[:, 7] <= 1500.0)
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
    """Sobol-mapped RF seeds (power-of-two draw truncated to n)."""
    n_pow2 = 1 << math.ceil(math.log2(max(n, 1)))
    unit = qmc.Sobol(d=1, scramble=True, seed=seed).random(n=n_pow2)[:n, 0]
    return (RF_SEED_MIN + unit * (RF_SEED_MAX - RF_SEED_MIN)).astype(int)


def generate_seed2_levels(
    n: int = DEFAULT_SEED_LEVELS,
    seed: int = DEFAULT_RF_SEED_SEED + 1,
) -> np.ndarray:
    return generate_seed_levels(n=n, seed=seed)


def build_manifest(
    physics_count: int = DEFAULT_PHYSICS_COUNT,
    seed_levels: int = DEFAULT_SEED_LEVELS,
    sampler_seed: int | None = DEFAULT_SAMPLER_SEED,
    rf_seed_seed: int = DEFAULT_RF_SEED_SEED,
    dz_1D: float = DEFAULT_DZ_1D,
    bedrock_thickness: float = DEFAULT_BEDROCK_THICKNESS,
    motion_freq: float = DEFAULT_MOTION_FREQ,
) -> list[OodThreeLayerEntry]:
    physical = generate_physics_samples(n=physics_count, sampler_seed=sampler_seed)
    seeds1 = generate_seed_levels(n=seed_levels, seed=rf_seed_seed)
    seeds2 = generate_seed2_levels(n=seed_levels, seed=rf_seed_seed + 1)

    bedrock_layer_count, bedrock_thickness_discretized = _discretize_length(
        bedrock_thickness, dz_1D
    )

    manifest: list[OodThreeLayerEntry] = []
    global_index = 0
    for sobol_id, sample in enumerate(physical):
        Vs_mid, H1, H2, rH, aHV, Vs_contrast, CoV, Vs_bedrock = (float(v) for v in sample)
        Vs1 = Vs_mid / np.exp(Vs_contrast)

        layer1_count, H1_discretized = _discretize_length(H1, dz_1D)
        layer2_count, H2_discretized = _discretize_length(H2, dz_1D)
        Lz_discretized = (layer1_count + layer2_count + bedrock_layer_count) * dz_1D

        travel_time = H1_discretized / Vs1 + H2_discretized / Vs_mid
        f0_effective = 1.0 / (4.0 * travel_time)
        duration = 50.0 if f0_effective < 1.0 else 30.0
        damping_freq_first = min(f0_effective, motion_freq)

        for replicate_id in range(seed_levels):
            manifest.append(
                OodThreeLayerEntry(
                    index=global_index,
                    sobol_id=sobol_id,
                    replicate_id=replicate_id,
                    Vs1=float(Vs1),
                    H1_requested=H1,
                    H1_discretized=H1_discretized,
                    layer1_count=layer1_count,
                    CoV1=CoV,
                    rH1=rH,
                    aHV1=aHV,
                    Vs_mid=Vs_mid,
                    H2_requested=H2,
                    H2_discretized=H2_discretized,
                    layer2_count=layer2_count,
                    CoV2=CoV,
                    rH2=rH,
                    aHV2=aHV,
                    Vs_contrast=Vs_contrast,
                    Vs_bedrock=Vs_bedrock,
                    bedrock_thickness_discretized=bedrock_thickness_discretized,
                    bedrock_layer_count=bedrock_layer_count,
                    Lz_discretized=Lz_discretized,
                    seed1=int(seeds1[replicate_id]),
                    seed2=int(seeds2[replicate_id]),
                    motion_freq=motion_freq,
                    f0_effective=f0_effective,
                    duration=duration,
                    damping_freq_first=damping_freq_first,
                )
            )
            global_index += 1
    return manifest


def write_manifest_csv(path: str | Path, manifest: list[OodThreeLayerEntry]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for entry in manifest:
            writer.writerow(entry.to_row())
    return output_path.resolve()


def load_manifest_csv(path: str | Path) -> list[OodThreeLayerEntry]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            OodThreeLayerEntry(
                index=int(row["index"]),
                sobol_id=int(row["sobol_id"]),
                replicate_id=int(row["replicate_id"]),
                Vs1=float(row["Vs1"]),
                H1_requested=float(row["H1_requested"]),
                H1_discretized=float(row["H1_discretized"]),
                layer1_count=int(row["layer1_count"]),
                CoV1=float(row["CoV1"]),
                rH1=float(row["rH1"]),
                aHV1=float(row["aHV1"]),
                Vs_mid=float(row["Vs_mid"]),
                H2_requested=float(row["H2_requested"]),
                H2_discretized=float(row["H2_discretized"]),
                layer2_count=int(row["layer2_count"]),
                CoV2=float(row["CoV2"]),
                rH2=float(row["rH2"]),
                aHV2=float(row["aHV2"]),
                Vs_contrast=float(row["Vs_contrast"]),
                Vs_bedrock=float(row["Vs_bedrock"]),
                bedrock_thickness_discretized=float(row["bedrock_thickness_discretized"]),
                bedrock_layer_count=int(row["bedrock_layer_count"]),
                Lz_discretized=float(row["Lz_discretized"]),
                seed1=int(row["seed1"]),
                seed2=int(row["seed2"]),
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
) -> list[OodThreeLayerEntry]:
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
