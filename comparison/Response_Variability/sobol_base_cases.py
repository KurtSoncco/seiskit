"""4D Sobol base cases for Response_Variability (Vs1, H, CoV, Vs2).

Fixed geostatistics: rH=10 m, aHV=50, bedrock thickness=10 m.
Reuses marginal bounds from neural-operator/data/sobol.py.
"""

from __future__ import annotations

import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.stats import lognorm, qmc

_NEURO_DATA = Path(__file__).resolve().parents[2] / "neural-operator" / "data"
if str(_NEURO_DATA) not in sys.path:
    sys.path.insert(0, str(_NEURO_DATA))

from sobol import (  # noqa: E402
    bounds_CoV,
    bounds_H,
    scale_Vs1,
    scale_Vs2,
    sigma_Vs1,
    sigma_Vs2,
)

DEFAULT_SOBOL_COUNT_FULL = 64
DEFAULT_SOBOL_COUNT_SMOKE = 4
DEFAULT_SAMPLER_SEED = 42

RH_FIXED = 10.0
AHV_FIXED = 50.0
BEDROCK_DEPTH = 10.0

CSV_COLUMNS = ("sobol_id", "Vs1", "H", "CoV", "Vs2", "rH", "aHV", "bedrock_thickness")


@dataclass(frozen=True)
class SobolBaseCase:
    sobol_id: int
    vs1: float
    H: float
    cov: float
    vs2: float
    rH: float = RH_FIXED
    aHV: float = AHV_FIXED
    bedrock_thickness: float = BEDROCK_DEPTH

    @property
    def rV(self) -> float:
        return self.rH / self.aHV

    def to_row(self) -> dict[str, float | int]:
        return {
            "sobol_id": self.sobol_id,
            "Vs1": self.vs1,
            "H": self.H,
            "CoV": self.cov,
            "Vs2": self.vs2,
            "rH": self.rH,
            "aHV": self.aHV,
            "bedrock_thickness": self.bedrock_thickness,
        }


def unit_to_physical_4d(unit_samples: np.ndarray) -> np.ndarray:
    """Map unit-cube samples (n, 4) to (Vs1, H, CoV, Vs2)."""
    raw = np.asarray(unit_samples, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != 4:
        raise ValueError(f"Expected shape (n, 4), got {raw.shape}")
    phys = np.zeros_like(raw)
    phys[:, 0] = lognorm.ppf(raw[:, 0], s=sigma_Vs1, scale=scale_Vs1)
    phys[:, 1] = bounds_H[0] + raw[:, 1] * (bounds_H[1] - bounds_H[0])
    phys[:, 2] = bounds_CoV[0] + raw[:, 2] * (bounds_CoV[1] - bounds_CoV[0])
    phys[:, 3] = lognorm.ppf(raw[:, 3], s=sigma_Vs2, scale=scale_Vs2)
    return phys


def _bounds_mask_4d(physical: np.ndarray) -> np.ndarray:
    return (
        (physical[:, 0] >= 100.0)
        & (physical[:, 0] <= 360.0)
        & (physical[:, 1] >= bounds_H[0])
        & (physical[:, 1] <= bounds_H[1])
        & (physical[:, 2] >= bounds_CoV[0])
        & (physical[:, 2] <= bounds_CoV[1])
        & (physical[:, 3] >= 760.0)
        & (physical[:, 3] <= 1500.0)
    )


def generate_base_cases(
    target_count: int,
    *,
    sampler_seed: int = DEFAULT_SAMPLER_SEED,
) -> list[SobolBaseCase]:
    if target_count <= 0:
        return []

    sampler = qmc.Sobol(d=4, scramble=True, seed=sampler_seed)
    batch = max(8, 1 << int(np.ceil(np.log2(max(target_count, 1)))))
    collected: list[SobolBaseCase] = []
    sobol_id = 0

    while len(collected) < target_count:
        unit = sampler.random(batch)
        physical = unit_to_physical_4d(unit)
        mask = _bounds_mask_4d(physical)
        for row in physical[mask]:
            if len(collected) >= target_count:
                break
            collected.append(
                SobolBaseCase(
                    sobol_id=sobol_id,
                    vs1=float(row[0]),
                    H=float(row[1]),
                    cov=float(row[2]),
                    vs2=float(row[3]),
                )
            )
            sobol_id += 1

    return collected


def default_csv_path() -> Path:
    return Path(__file__).resolve().parent / "rv_sobol_base_cases.csv"


def save_base_cases_csv(cases: list[SobolBaseCase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for case in cases:
            writer.writerow(case.to_row())


def load_base_cases_csv(path: Path) -> list[SobolBaseCase]:
    cases: list[SobolBaseCase] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            cases.append(
                SobolBaseCase(
                    sobol_id=int(row["sobol_id"]),
                    vs1=float(row["Vs1"]),
                    H=float(row["H"]),
                    cov=float(row["CoV"]),
                    vs2=float(row["Vs2"]),
                    rH=float(row.get("rH", RH_FIXED)),
                    aHV=float(row.get("aHV", AHV_FIXED)),
                    bedrock_thickness=float(row.get("bedrock_thickness", BEDROCK_DEPTH)),
                )
            )
    return cases


def ensure_base_cases(
    path: Path | None = None,
    *,
    count: int,
    overwrite: bool = False,
    sampler_seed: int = DEFAULT_SAMPLER_SEED,
) -> list[SobolBaseCase]:
    path = path or default_csv_path()
    if path.exists() and not overwrite:
        loaded = load_base_cases_csv(path)
        if len(loaded) >= count:
            return loaded[:count]
    cases = generate_base_cases(count, sampler_seed=sampler_seed)
    save_base_cases_csv(cases, path)
    return cases
