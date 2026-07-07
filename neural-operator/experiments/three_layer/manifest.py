"""Sobol-driven manifest for the 3-Vs-layer capability-check experiment.

Two independently variable soil layers (a shallow layer around Vs1 and a
middle layer around Vs_mid), each with its own random-field intralayer
variability, over a fixed-velocity bedrock. See
``neural-operator/experiments/README.md``.

CoV/rH/aHV bounds and the general lognormal/uniform sampling pattern are
reused from ``neural-operator/data/sobol.py``; only the Vs1/Vs_mid bounds and
layer-thickness distributions are new (there was previously only one soil
layer, so only one thickness bound existed).
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
    RF_SEED_MAX,
    RF_SEED_MIN,
    _discretize_length,
    bounds_CoV,
    bounds_rH,
    lognormal_parameter,
)

DEFAULT_SAMPLE_COUNT = 3
DEFAULT_SAMPLER_SEED = 42
DEFAULT_RF_SEED_SEED = 42
BEDROCK_VS = 1500.0  # Fixed per README; not sampled.

# README literal bounds for this experiment (distinct from the single-layer
# Vs1/Vs2 bounds in sobol.py, which represent a different, wider design space).
scale_Vs1, sigma_Vs1 = lognormal_parameter(100.0, 230.0)
scale_Vs_mid, sigma_Vs_mid = lognormal_parameter(450.0, 560.0)
scale_aHV, sigma_aHV = lognormal_parameter(10.0, 50.0)

# Each layer's thickness is capped so H1 + H2 never exceeds 30 m total: large
# multi-layer domains make the OpenSees runs prohibitively slow (a single
# ~170m-deep case took an estimated 1-2 days), and 30m keeps these exploratory
# capability-check runs on the order of the single-layer training cases.
bounds_H1 = (5.0, 15.0)
bounds_H2 = (5.0, 15.0)

PHYSICAL_COLUMNS = (
    "Vs1",
    "H1",
    "CoV1",
    "rH1",
    "aHV1",
    "Vs_mid",
    "H2",
    "CoV2",
    "rH2",
    "aHV2",
)
MANIFEST_COLUMNS = (
    "index",
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


def unit_to_physical(unit_samples: np.ndarray) -> np.ndarray:
    """Map unit-cube samples with shape ``(n, 10)`` into physical parameters."""
    raw = np.asarray(unit_samples, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != len(PHYSICAL_COLUMNS):
        raise ValueError(
            f"Expected samples with shape (n, {len(PHYSICAL_COLUMNS)}), got {raw.shape}."
        )

    phys = np.zeros_like(raw)
    phys[:, 0] = lognorm.ppf(raw[:, 0], s=sigma_Vs1, scale=scale_Vs1)  # Vs1
    phys[:, 1] = bounds_H1[0] + raw[:, 1] * (bounds_H1[1] - bounds_H1[0])  # H1
    phys[:, 2] = bounds_CoV[0] + raw[:, 2] * (bounds_CoV[1] - bounds_CoV[0])  # CoV1
    phys[:, 3] = bounds_rH[0] + raw[:, 3] * (bounds_rH[1] - bounds_rH[0])  # rH1
    phys[:, 4] = lognorm.ppf(raw[:, 4], s=sigma_aHV, scale=scale_aHV)  # aHV1
    phys[:, 5] = lognorm.ppf(raw[:, 5], s=sigma_Vs_mid, scale=scale_Vs_mid)  # Vs_mid
    phys[:, 6] = bounds_H2[0] + raw[:, 6] * (bounds_H2[1] - bounds_H2[0])  # H2
    phys[:, 7] = bounds_CoV[0] + raw[:, 7] * (bounds_CoV[1] - bounds_CoV[0])  # CoV2
    phys[:, 8] = bounds_rH[0] + raw[:, 8] * (bounds_rH[1] - bounds_rH[0])  # rH2
    phys[:, 9] = lognorm.ppf(raw[:, 9], s=sigma_aHV, scale=scale_aHV)  # aHV2
    return phys


def generate_physical_samples(
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    sampler_seed: int | None = DEFAULT_SAMPLER_SEED,
) -> np.ndarray:
    """Generate ``sample_count`` physical samples via Sobol quasi-random design."""
    sampler = qmc.Sobol(d=len(PHYSICAL_COLUMNS), scramble=True, seed=sampler_seed)
    raw = sampler.random(n=sample_count)
    return unit_to_physical(raw)


def build_manifest(
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    sampler_seed: int | None = DEFAULT_SAMPLER_SEED,
    rf_seed_seed: int = DEFAULT_RF_SEED_SEED,
    dz_1D: float = DEFAULT_DZ_1D,
    bedrock_thickness: float = DEFAULT_BEDROCK_THICKNESS,
    motion_freq: float = DEFAULT_MOTION_FREQ,
) -> list[ThreeLayerManifestEntry]:
    """Build the flattened 3-layer manifest: one row per Sobol sample."""
    physical_samples = generate_physical_samples(sample_count=sample_count, sampler_seed=sampler_seed)

    rng = np.random.default_rng(rf_seed_seed)
    seed_matrix = rng.integers(RF_SEED_MIN, RF_SEED_MAX + 1, size=(sample_count, 2), endpoint=False)

    bedrock_layer_count, bedrock_thickness_discretized = _discretize_length(bedrock_thickness, dz_1D)

    manifest: list[ThreeLayerManifestEntry] = []
    for index, sample in enumerate(physical_samples):
        Vs1, H1, CoV1, rH1, aHV1, Vs_mid, H2, CoV2, rH2, aHV2 = (float(v) for v in sample)
        layer1_count, H1_discretized = _discretize_length(H1, dz_1D)
        layer2_count, H2_discretized = _discretize_length(H2, dz_1D)
        Lz_discretized = (layer1_count + layer2_count + bedrock_layer_count) * dz_1D

        # Layered quarter-wavelength travel-time formula: generalizes the
        # single-layer f0 = Vs1 / (4 H) estimate to a two-layer travel time.
        travel_time = H1_discretized / Vs1 + H2_discretized / Vs_mid
        f0_effective = 1.0 / (4.0 * travel_time)
        duration = 50.0 if f0_effective < 1.0 else 30.0
        damping_freq_first = min(f0_effective, motion_freq)

        seed1, seed2 = (int(s) for s in seed_matrix[index])

        manifest.append(
            ThreeLayerManifestEntry(
                index=index,
                Vs1=Vs1,
                H1_requested=H1,
                H1_discretized=H1_discretized,
                layer1_count=layer1_count,
                CoV1=CoV1,
                rH1=rH1,
                aHV1=aHV1,
                Vs_mid=Vs_mid,
                H2_requested=H2,
                H2_discretized=H2_discretized,
                layer2_count=layer2_count,
                CoV2=CoV2,
                rH2=rH2,
                aHV2=aHV2,
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


DEFAULT_MANIFEST_PATH = THIS_DIR / "manifest.csv"


def ensure_manifest(
    path: str | Path = DEFAULT_MANIFEST_PATH,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    sampler_seed: int | None = DEFAULT_SAMPLER_SEED,
    rf_seed_seed: int = DEFAULT_RF_SEED_SEED,
    overwrite: bool = False,
) -> list[ThreeLayerManifestEntry]:
    manifest_path = Path(path)
    manifest = build_manifest(
        sample_count=sample_count, sampler_seed=sampler_seed, rf_seed_seed=rf_seed_seed
    )
    if overwrite or not manifest_path.exists():
        write_manifest_csv(manifest_path, manifest)
    return manifest
