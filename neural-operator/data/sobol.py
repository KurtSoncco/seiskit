"""Sobol sampling helpers for neural-operator data generation.

This module keeps the original 6D design space, but exposes it as reusable,
import-safe functions so experiment runners can build a deterministic manifest
without executing exploratory analysis at import time.
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.stats import lognorm, norm, qmc

DEFAULT_CONFIDENCE = 0.99
DEFAULT_SAMPLER_SEED = 42
DEFAULT_RF_SEED_GENERATOR = 42
DEFAULT_SOBOL_SAMPLE_COUNT = 2**8
DEFAULT_RF_SEEDS_PER_SAMPLE = 30
DEFAULT_BEDROCK_THICKNESS = 10.0
DEFAULT_DZ_1D = 1.0
DEFAULT_MOTION_FREQ = 3.0
RF_SEED_MIN = 1
RF_SEED_MAX = 9_999_999

PHYSICAL_COLUMNS = ("Vs1", "H", "CoV", "rH", "aHV", "Vs2")
MANIFEST_COLUMNS = (
    "index",
    "sample_id",
    "replicate_id",
    "rf_seed",
    "Vs1",
    "H_requested",
    "H_discretized",
    "soil_layer_count",
    "CoV",
    "rH",
    "aHV",
    "Vs2",
    "dz_1D",
    "bedrock_thickness",
    "bedrock_thickness_discretized",
    "bedrock_layer_count",
    "Lz_discretized",
    "motion_freq",
    "f0_effective",
    "duration",
    "damping_freq_first",
)


@dataclass(frozen=True)
class ExecutionParameters:
    H_requested: float
    H_discretized: float
    soil_layer_count: int
    dz_1D: float
    bedrock_thickness: float
    bedrock_thickness_discretized: float
    bedrock_layer_count: int
    Lz_discretized: float
    motion_freq: float
    f0_effective: float
    duration: float
    damping_freq_first: float


@dataclass(frozen=True)
class ManifestEntry:
    index: int
    sample_id: int
    replicate_id: int
    rf_seed: int
    Vs1: float
    H_requested: float
    H_discretized: float
    soil_layer_count: int
    CoV: float
    rH: float
    aHV: float
    Vs2: float
    dz_1D: float
    bedrock_thickness: float
    bedrock_thickness_discretized: float
    bedrock_layer_count: int
    Lz_discretized: float
    motion_freq: float
    f0_effective: float
    duration: float
    damping_freq_first: float

    def to_row(self) -> dict[str, int | float]:
        return asdict(self)

    @property
    def H(self) -> float:
        """Canonical height used by the discretized simulation."""
        return self.H_discretized


def _discretize_length(length: float, spacing: float) -> tuple[int, float]:
    """Round a physical length to a mesh-consistent cell count."""
    count = max(1, int(round(length / spacing)))
    return count, count * spacing


def derive_execution_parameters(
    Vs1: float,
    H: float,
    *,
    dz_1D: float = DEFAULT_DZ_1D,
    bedrock_thickness: float = DEFAULT_BEDROCK_THICKNESS,
    motion_freq: float = DEFAULT_MOTION_FREQ,
) -> ExecutionParameters:
    """Derive the mesh-consistent execution values used by the runner."""
    soil_layer_count, H_discretized = _discretize_length(H, dz_1D)
    bedrock_layer_count, bedrock_thickness_discretized = _discretize_length(bedrock_thickness, dz_1D)
    Lz_discretized = (soil_layer_count + bedrock_layer_count) * dz_1D
    f0_effective = Vs1 / (4 * H_discretized)
    duration = 50.0 if f0_effective < 1.0 else 30.0
    damping_freq_first = min(f0_effective, motion_freq)
    return ExecutionParameters(
        H_requested=H,
        H_discretized=H_discretized,
        soil_layer_count=soil_layer_count,
        dz_1D=dz_1D,
        bedrock_thickness=bedrock_thickness,
        bedrock_thickness_discretized=bedrock_thickness_discretized,
        bedrock_layer_count=bedrock_layer_count,
        Lz_discretized=Lz_discretized,
        motion_freq=motion_freq,
        f0_effective=f0_effective,
        duration=duration,
        damping_freq_first=damping_freq_first,
    )


def lognormal_parameter(lower: float, upper: float, confidence: float = DEFAULT_CONFIDENCE) -> tuple[float, float]:
    """Return ``(scale, sigma)`` for a lognormal distribution fit to bounds."""
    ln_lower, ln_upper = np.log(lower), np.log(upper)
    z_score = norm.ppf(1 - (1 - confidence) / 2)
    mu = (ln_lower + ln_upper) / 2
    sigma = (ln_upper - ln_lower) / (2 * z_score)
    return float(np.exp(mu)), float(sigma)


scale_Vs1, sigma_Vs1 = lognormal_parameter(100.0, 360.0)
scale_Vs2, sigma_Vs2 = lognormal_parameter(760.0, 1500.0)
bounds_H = (15.0, 100.0)
bounds_CoV = (0.1, 0.3)
bounds_rH = (10.0, 100.0)
bounds_aHV = (10.0, 50.0)


def unit_to_physical(unit_samples: np.ndarray) -> np.ndarray:
    """Map unit-cube samples with shape ``(n, 6)`` into physical parameters."""
    raw = np.asarray(unit_samples, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != len(PHYSICAL_COLUMNS):
        raise ValueError(f"Expected samples with shape (n, {len(PHYSICAL_COLUMNS)}), got {raw.shape}.")

    phys = np.zeros_like(raw)
    phys[:, 0] = lognorm.ppf(raw[:, 0], s=sigma_Vs1, scale=scale_Vs1)
    phys[:, 1] = bounds_H[0] + raw[:, 1] * (bounds_H[1] - bounds_H[0])
    phys[:, 2] = bounds_CoV[0] + raw[:, 2] * (bounds_CoV[1] - bounds_CoV[0])
    phys[:, 3] = bounds_rH[0] + raw[:, 3] * (bounds_rH[1] - bounds_rH[0])
    phys[:, 4] = bounds_aHV[0] + raw[:, 4] * (bounds_aHV[1] - bounds_aHV[0])
    phys[:, 5] = lognorm.ppf(raw[:, 5], s=sigma_Vs2, scale=scale_Vs2)
    return phys


def physical_to_unit(physical_samples: np.ndarray) -> np.ndarray:
    """Map physical samples with shape ``(n, 6)`` into the unit cube."""
    physical = np.asarray(physical_samples, dtype=float)
    if physical.ndim != 2 or physical.shape[1] != len(PHYSICAL_COLUMNS):
        raise ValueError(f"Expected samples with shape (n, {len(PHYSICAL_COLUMNS)}), got {physical.shape}.")

    unit = np.zeros_like(physical)
    unit[:, 0] = lognorm.cdf(physical[:, 0], s=sigma_Vs1, scale=scale_Vs1)
    unit[:, 1] = (physical[:, 1] - bounds_H[0]) / (bounds_H[1] - bounds_H[0])
    unit[:, 2] = (physical[:, 2] - bounds_CoV[0]) / (bounds_CoV[1] - bounds_CoV[0])
    unit[:, 3] = (physical[:, 3] - bounds_rH[0]) / (bounds_rH[1] - bounds_rH[0])
    unit[:, 4] = (physical[:, 4] - bounds_aHV[0]) / (bounds_aHV[1] - bounds_aHV[0])
    unit[:, 5] = lognorm.cdf(physical[:, 5], s=sigma_Vs2, scale=scale_Vs2)
    return unit


def to_unit_cube_6d(physical_samples: np.ndarray) -> np.ndarray:
    """Backward-compatible alias for the original helper name."""
    return physical_to_unit(physical_samples)


def _bounds_mask(physical_samples: np.ndarray) -> np.ndarray:
    """Return a boolean mask for samples inside the configured physical bounds."""
    physical = np.asarray(physical_samples, dtype=float)
    return (
        (physical[:, 0] >= 100.0)
        & (physical[:, 0] <= 360.0)
        & (physical[:, 1] >= bounds_H[0])
        & (physical[:, 1] <= bounds_H[1])
        & (physical[:, 2] >= bounds_CoV[0])
        & (physical[:, 2] <= bounds_CoV[1])
        & (physical[:, 3] >= bounds_rH[0])
        & (physical[:, 3] <= bounds_rH[1])
        & (physical[:, 4] >= bounds_aHV[0])
        & (physical[:, 4] <= bounds_aHV[1])
        & (physical[:, 5] >= 760.0)
        & (physical[:, 5] <= 1500.0)
    )


def _build_sampler(method: str, sampler_seed: int | None) -> qmc.QMCEngine:
    method_normalized = method.lower()
    if method_normalized == "sobol":
        return qmc.Sobol(d=len(PHYSICAL_COLUMNS), scramble=True, seed=sampler_seed)
    if method_normalized in {"lhs", "latin", "latin_hypercube"}:
        return qmc.LatinHypercube(d=len(PHYSICAL_COLUMNS), seed=sampler_seed)
    raise ValueError("method must be 'sobol' or 'lhs'.")


def generate_and_filter(
    target_count: int,
    method: str = "sobol",
    sampler_seed: int | None = DEFAULT_SAMPLER_SEED,
) -> np.ndarray:
    """Generate ``target_count`` valid physical samples using Sobol or LHS."""
    if target_count <= 0:
        raise ValueError("target_count must be positive.")

    valid_samples: list[np.ndarray] = []
    sampler = _build_sampler(method, sampler_seed)
    method_normalized = method.lower()

    while len(valid_samples) < target_count:
        remaining = target_count - len(valid_samples)
        if method_normalized == "sobol":
            batch_size = max(2 ** math.ceil(math.log2(remaining + 100)), 512)
        else:
            batch_size = max(remaining * 3, 64)

        raw = sampler.random(n=batch_size)
        physical = unit_to_physical(raw)
        valid_samples.extend(physical[_bounds_mask(physical)])

    return np.asarray(valid_samples[:target_count], dtype=float)


def generate_physical_samples(
    target_count: int,
    method: str = "sobol",
    sampler_seed: int | None = DEFAULT_SAMPLER_SEED,
) -> np.ndarray:
    """Preferred alias for ``generate_and_filter``."""
    return generate_and_filter(target_count=target_count, method=method, sampler_seed=sampler_seed)


def generate_rf_seed_matrix(
    sample_count: int,
    seeds_per_sample: int = DEFAULT_RF_SEEDS_PER_SAMPLE,
    seed: int = DEFAULT_RF_SEED_GENERATOR,
) -> np.ndarray:
    """Return a deterministic ``(sample_count, seeds_per_sample)`` seed matrix."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    if seeds_per_sample <= 0:
        raise ValueError("seeds_per_sample must be positive.")

    rng = np.random.default_rng(seed)
    return rng.integers(
        RF_SEED_MIN,
        RF_SEED_MAX + 1,
        size=(sample_count, seeds_per_sample),
        endpoint=False,
    )


def build_manifest(
    sample_count: int = DEFAULT_SOBOL_SAMPLE_COUNT,
    seeds_per_sample: int = DEFAULT_RF_SEEDS_PER_SAMPLE,
    method: str = "sobol",
    sampler_seed: int | None = DEFAULT_SAMPLER_SEED,
    rf_seed_seed: int = DEFAULT_RF_SEED_GENERATOR,
) -> list[ManifestEntry]:
    """Build a flattened manifest with one row per Sobol sample and RF seed."""
    physical_samples = generate_physical_samples(
        target_count=sample_count,
        method=method,
        sampler_seed=sampler_seed,
    )
    rf_seed_matrix = generate_rf_seed_matrix(
        sample_count=sample_count,
        seeds_per_sample=seeds_per_sample,
        seed=rf_seed_seed,
    )

    manifest: list[ManifestEntry] = []
    global_index = 0
    for sample_id, sample in enumerate(physical_samples):
        execution = derive_execution_parameters(
            Vs1=float(sample[0]),
            H=float(sample[1]),
        )
        for replicate_id, rf_seed in enumerate(rf_seed_matrix[sample_id]):
            manifest.append(
                ManifestEntry(
                    index=global_index,
                    sample_id=sample_id,
                    replicate_id=replicate_id,
                    rf_seed=int(rf_seed),
                    Vs1=float(sample[0]),
                    H_requested=execution.H_requested,
                    H_discretized=execution.H_discretized,
                    soil_layer_count=execution.soil_layer_count,
                    CoV=float(sample[2]),
                    rH=float(sample[3]),
                    aHV=float(sample[4]),
                    Vs2=float(sample[5]),
                    dz_1D=execution.dz_1D,
                    bedrock_thickness=execution.bedrock_thickness,
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


def write_manifest_csv(path: str | Path, manifest: list[ManifestEntry]) -> Path:
    """Write manifest rows to CSV and return the resolved output path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for entry in manifest:
            writer.writerow(entry.to_row())

    return output_path.resolve()


def load_manifest_csv(path: str | Path) -> list[ManifestEntry]:
    """Load manifest rows from CSV."""
    input_path = Path(path)
    with input_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        manifest: list[ManifestEntry] = []
        for row in reader:
            Vs1 = float(row["Vs1"])
            H_requested = float(row.get("H_requested", row.get("H", "")))
            execution = derive_execution_parameters(
                Vs1=Vs1,
                H=H_requested,
                dz_1D=float(row.get("dz_1D", DEFAULT_DZ_1D)),
                bedrock_thickness=float(row.get("bedrock_thickness", DEFAULT_BEDROCK_THICKNESS)),
                motion_freq=float(row.get("motion_freq", DEFAULT_MOTION_FREQ)),
            )
            manifest.append(
                ManifestEntry(
                    index=int(row["index"]),
                    sample_id=int(row["sample_id"]),
                    replicate_id=int(row["replicate_id"]),
                    rf_seed=int(row["rf_seed"]),
                    Vs1=Vs1,
                    H_requested=H_requested,
                    H_discretized=float(row.get("H_discretized", execution.H_discretized)),
                    soil_layer_count=int(row.get("soil_layer_count", execution.soil_layer_count)),
                    CoV=float(row["CoV"]),
                    rH=float(row["rH"]),
                    aHV=float(row["aHV"]),
                    Vs2=float(row["Vs2"]),
                    dz_1D=float(row.get("dz_1D", execution.dz_1D)),
                    bedrock_thickness=float(row.get("bedrock_thickness", execution.bedrock_thickness)),
                    bedrock_thickness_discretized=float(
                        row.get("bedrock_thickness_discretized", execution.bedrock_thickness_discretized)
                    ),
                    bedrock_layer_count=int(row.get("bedrock_layer_count", execution.bedrock_layer_count)),
                    Lz_discretized=float(row.get("Lz_discretized", execution.Lz_discretized)),
                    motion_freq=float(row.get("motion_freq", execution.motion_freq)),
                    f0_effective=float(row.get("f0_effective", execution.f0_effective)),
                    duration=float(row.get("duration", execution.duration)),
                    damping_freq_first=float(row.get("damping_freq_first", execution.damping_freq_first)),
                )
            )
    return manifest


def _manifest_schema_matches(path: str | Path) -> bool:
    """Return True when the CSV header matches the current manifest schema."""
    input_path = Path(path)
    with input_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
    return header == list(MANIFEST_COLUMNS)


def ensure_manifest(
    path: str | Path,
    sample_count: int = DEFAULT_SOBOL_SAMPLE_COUNT,
    seeds_per_sample: int = DEFAULT_RF_SEEDS_PER_SAMPLE,
    method: str = "sobol",
    sampler_seed: int | None = DEFAULT_SAMPLER_SEED,
    rf_seed_seed: int = DEFAULT_RF_SEED_GENERATOR,
    overwrite: bool = False,
) -> list[ManifestEntry]:
    """Create the manifest if needed, then return the loaded rows."""
    manifest_path = Path(path)
    if overwrite or not manifest_path.exists():
        manifest = build_manifest(
            sample_count=sample_count,
            seeds_per_sample=seeds_per_sample,
            method=method,
            sampler_seed=sampler_seed,
            rf_seed_seed=rf_seed_seed,
        )
        write_manifest_csv(manifest_path, manifest)
        return manifest
    manifest = load_manifest_csv(manifest_path)
    if not _manifest_schema_matches(manifest_path):
        write_manifest_csv(manifest_path, manifest)
    return manifest


__all__ = [
    "DEFAULT_RF_SEEDS_PER_SAMPLE",
    "DEFAULT_RF_SEED_GENERATOR",
    "DEFAULT_SAMPLER_SEED",
    "DEFAULT_SOBOL_SAMPLE_COUNT",
    "DEFAULT_BEDROCK_THICKNESS",
    "DEFAULT_DZ_1D",
    "DEFAULT_MOTION_FREQ",
    "ExecutionParameters",
    "MANIFEST_COLUMNS",
    "ManifestEntry",
    "PHYSICAL_COLUMNS",
    "build_manifest",
    "derive_execution_parameters",
    "ensure_manifest",
    "generate_and_filter",
    "generate_physical_samples",
    "generate_rf_seed_matrix",
    "load_manifest_csv",
    "lognormal_parameter",
    "physical_to_unit",
    "to_unit_cube_6d",
    "unit_to_physical",
    "write_manifest_csv",
]