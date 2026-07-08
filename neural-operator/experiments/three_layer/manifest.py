"""Sobol-driven manifest for the 3-Vs-layer generalization experiment.

Two independently variable soil layers over fixed bedrock, with a 4D topology
Sobol design over (Vs_mid, H1, H2, Vs_contrast) and 25 RF seed replicates per
topology point. Corner holdouts in (Vs_mid, thickness, contrast) space are
tagged ``extrap_test``. See ``neural-operator/experiments/README.md``.
"""

from __future__ import annotations

import csv
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
    DEFAULT_BEDROCK_THICKNESS,
    DEFAULT_DZ_1D,
    DEFAULT_MOTION_FREQ,
    DEFAULT_SAMPLER_SEED,
    _discretize_length,
    bounds_CoV,
    bounds_rH,
    generate_rf_seed_matrix,
    lognormal_parameter,
    scale_aHV,
)

DEFAULT_TOPOLOGY_COUNT = 10
DEFAULT_RF_SEEDS_PER_TOPOLOGY = 25
DEFAULT_RF_SEED_SEED = 42
BEDROCK_VS = 1500.0

scale_Vs_mid, sigma_Vs_mid = lognormal_parameter(450.0, 560.0)
bounds_H1 = (5.0, 15.0)
bounds_H2 = (5.0, 15.0)
bounds_thickness_sum = (bounds_H1[0] + bounds_H2[0], bounds_H1[1] + bounds_H2[1])
# log(Vs_mid / Vs1); derived from the README Vs1/Vs_mid ranges.
bounds_Vs_contrast = (0.8, 1.6)

COV_FIXED = sum(bounds_CoV) / 2.0
RH_FIXED = sum(bounds_rH) / 2.0
AHV_FIXED = scale_aHV

TOPOLOGY_COLUMNS = ("Vs_mid", "H1", "H2", "Vs_contrast")

MANIFEST_COLUMNS = (
    "index",
    "topology_id",
    "replicate_id",
    "split",
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
class ThreeLayerManifestEntry:
    index: int
    topology_id: int
    replicate_id: int
    split: str
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

    def to_row(self) -> dict[str, int | float | str]:
        return asdict(self)


def unit_to_topology(unit_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map unit-cube samples ``(n, 4)`` to physical topology rows and unit rows."""
    raw = np.asarray(unit_samples, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != len(TOPOLOGY_COLUMNS):
        raise ValueError(
            f"Expected samples with shape (n, {len(TOPOLOGY_COLUMNS)}), got {raw.shape}."
        )

    phys = np.zeros_like(raw)
    phys[:, 0] = lognorm.ppf(raw[:, 0], s=sigma_Vs_mid, scale=scale_Vs_mid)
    phys[:, 1] = bounds_H1[0] + raw[:, 1] * (bounds_H1[1] - bounds_H1[0])
    phys[:, 2] = bounds_H2[0] + raw[:, 2] * (bounds_H2[1] - bounds_H2[0])
    phys[:, 3] = bounds_Vs_contrast[0] + raw[:, 3] * (bounds_Vs_contrast[1] - bounds_Vs_contrast[0])
    return phys, raw


def is_corner_holdout(unit_row: np.ndarray, h1: float, h2: float) -> bool:
    """Tag extrapolation corner: high Vs_mid, thin total thickness, high contrast."""
    thickness_sum = h1 + h2
    thin_threshold = bounds_thickness_sum[0] + 0.25 * (
        bounds_thickness_sum[1] - bounds_thickness_sum[0]
    )
    return bool(unit_row[0] > 0.75 and unit_row[3] > 0.75 and thickness_sum <= thin_threshold)


def generate_topology_samples(
    sample_count: int = DEFAULT_TOPOLOGY_COUNT,
    sampler_seed: int | None = DEFAULT_SAMPLER_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ``sample_count`` topology draws via Sobol quasi-random design."""
    sampler = qmc.Sobol(d=len(TOPOLOGY_COLUMNS), scramble=True, seed=sampler_seed)
    unit = sampler.random(n=sample_count)
    physical, unit = unit_to_topology(unit)
    return physical, unit


def build_manifest(
    topology_count: int = DEFAULT_TOPOLOGY_COUNT,
    rf_seeds_per_topology: int = DEFAULT_RF_SEEDS_PER_TOPOLOGY,
    sampler_seed: int | None = DEFAULT_SAMPLER_SEED,
    rf_seed_seed: int = DEFAULT_RF_SEED_SEED,
    dz_1D: float = DEFAULT_DZ_1D,
    bedrock_thickness: float = DEFAULT_BEDROCK_THICKNESS,
    motion_freq: float = DEFAULT_MOTION_FREQ,
) -> list[ThreeLayerManifestEntry]:
    """Build the flattened 3-layer manifest: one row per topology × RF seed pair."""
    physical_samples, unit_samples = generate_topology_samples(
        sample_count=topology_count, sampler_seed=sampler_seed
    )

    seed1_matrix = generate_rf_seed_matrix(
        sample_count=topology_count,
        seeds_per_sample=rf_seeds_per_topology,
        seed=rf_seed_seed,
    )
    seed2_matrix = generate_rf_seed_matrix(
        sample_count=topology_count,
        seeds_per_sample=rf_seeds_per_topology,
        seed=rf_seed_seed + 1,
    )

    bedrock_layer_count, bedrock_thickness_discretized = _discretize_length(
        bedrock_thickness, dz_1D
    )

    manifest: list[ThreeLayerManifestEntry] = []
    global_index = 0

    for topology_id, sample in enumerate(physical_samples):
        Vs_mid, H1, H2, Vs_contrast = (float(v) for v in sample)
        Vs1 = Vs_mid / np.exp(Vs_contrast)
        split = "extrap_test" if is_corner_holdout(unit_samples[topology_id], H1, H2) else "train"

        layer1_count, H1_discretized = _discretize_length(H1, dz_1D)
        layer2_count, H2_discretized = _discretize_length(H2, dz_1D)
        Lz_discretized = (layer1_count + layer2_count + bedrock_layer_count) * dz_1D

        travel_time = H1_discretized / Vs1 + H2_discretized / Vs_mid
        f0_effective = 1.0 / (4.0 * travel_time)
        duration = 50.0 if f0_effective < 1.0 else 30.0
        damping_freq_first = min(f0_effective, motion_freq)

        for replicate_id in range(rf_seeds_per_topology):
            seed1 = int(seed1_matrix[topology_id, replicate_id])
            seed2 = int(seed2_matrix[topology_id, replicate_id])

            manifest.append(
                ThreeLayerManifestEntry(
                    index=global_index,
                    topology_id=topology_id,
                    replicate_id=replicate_id,
                    split=split,
                    Vs1=Vs1,
                    H1_requested=H1,
                    H1_discretized=H1_discretized,
                    layer1_count=layer1_count,
                    CoV1=COV_FIXED,
                    rH1=RH_FIXED,
                    aHV1=AHV_FIXED,
                    Vs_mid=Vs_mid,
                    H2_requested=H2,
                    H2_discretized=H2_discretized,
                    layer2_count=layer2_count,
                    CoV2=COV_FIXED,
                    rH2=RH_FIXED,
                    aHV2=AHV_FIXED,
                    Vs_contrast=Vs_contrast,
                    Vs_bedrock=BEDROCK_VS,
                    bedrock_thickness_discretized=bedrock_thickness_discretized,
                    bedrock_layer_count=bedrock_layer_count,
                    Lz_discretized=Lz_discretized,
                    seed1=seed1,
                    seed2=seed2,
                    motion_freq=motion_freq,
                    f0_effective=f0_effective,
                    duration=duration,
                    damping_freq_first=damping_freq_first,
                )
            )
            global_index += 1

    return manifest


def write_manifest_csv(path: str | Path, manifest: list[ThreeLayerManifestEntry]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for entry in manifest:
            writer.writerow(entry.to_row())
    return output_path.resolve()


def load_manifest_csv(path: str | Path) -> list[ThreeLayerManifestEntry]:
    manifest_path = Path(path)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        entries: list[ThreeLayerManifestEntry] = []
        for row in reader:
            entries.append(
                ThreeLayerManifestEntry(
                    index=int(row["index"]),
                    topology_id=int(row["topology_id"]),
                    replicate_id=int(row["replicate_id"]),
                    split=str(row["split"]),
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
            )
    return entries


DEFAULT_MANIFEST_PATH = THIS_DIR / "manifest.csv"


def ensure_manifest(
    path: str | Path = DEFAULT_MANIFEST_PATH,
    topology_count: int = DEFAULT_TOPOLOGY_COUNT,
    rf_seeds_per_topology: int = DEFAULT_RF_SEEDS_PER_TOPOLOGY,
    sampler_seed: int | None = DEFAULT_SAMPLER_SEED,
    rf_seed_seed: int = DEFAULT_RF_SEED_SEED,
    overwrite: bool = False,
) -> list[ThreeLayerManifestEntry]:
    manifest_path = Path(path)
    if not overwrite and manifest_path.exists():
        return load_manifest_csv(manifest_path)

    manifest = build_manifest(
        topology_count=topology_count,
        rf_seeds_per_topology=rf_seeds_per_topology,
        sampler_seed=sampler_seed,
        rf_seed_seed=rf_seed_seed,
    )
    write_manifest_csv(manifest_path, manifest)
    return manifest
