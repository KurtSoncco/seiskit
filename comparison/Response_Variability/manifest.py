"""Sobol-driven manifest for Response_Variability comparison campaign."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from sobol_base_cases import (
    AHV_FIXED,
    BEDROCK_DEPTH,
    DEFAULT_SOBOL_COUNT_FULL,
    DEFAULT_SOBOL_COUNT_SMOKE,
    RH_FIXED,
    SobolBaseCase,
    ensure_base_cases,
)

MethodId = Literal[
    "grf_2d",
    "delatorre",
    "hallal_vs",
    "hallal_tts",
    "hallal_dmin",
]

HALLAL_METHODS: list[MethodId] = ["hallal_vs", "hallal_tts", "hallal_dmin"]
RF_METHODS: list[MethodId] = ["grf_2d", "delatorre"]
METHODS: list[MethodId] = [*HALLAL_METHODS, *RF_METHODS]

MOTION_IDS = ["M1"]
MOTION_FREQS = {"M1": 3.0}

DX = 0.5
DZ = 0.5
LX_VAR = 200.0
BC_WIDTH = 100.0

HALLAL_SEEDS_FULL = list(range(1, 201))
HALLAL_SEEDS_SMOKE = list(range(1, 11))
RF_SEEDS_FULL = list(range(1, 31))
RF_SEEDS_SMOKE = list(range(1, 6))

RH = RH_FIXED
AHV = AHV_FIXED
RV = RH_FIXED / AHV_FIXED


def _smoke_mode() -> bool:
    return os.getenv("RV_SMOKE", "0") == "1"


def active_sobol_count() -> int:
    return DEFAULT_SOBOL_COUNT_SMOKE if _smoke_mode() else DEFAULT_SOBOL_COUNT_FULL


def active_hallal_seeds() -> list[int]:
    return HALLAL_SEEDS_SMOKE if _smoke_mode() else HALLAL_SEEDS_FULL


def active_rf_seeds() -> list[int]:
    return RF_SEEDS_SMOKE if _smoke_mode() else RF_SEEDS_FULL


def active_lx_var() -> float:
    return 100.0 if _smoke_mode() else LX_VAR


def active_bc_width() -> float:
    return 50.0 if _smoke_mode() else BC_WIDTH


def active_lx_total() -> float:
    return active_lx_var() + 2 * active_bc_width()


def active_dx() -> float:
    return 1.0 if _smoke_mode() else DX


def active_dz() -> float:
    return 1.0 if _smoke_mode() else DZ


def active_motion_ids() -> list[str]:
    return list(MOTION_IDS)


def active_base_cases() -> list[SobolBaseCase]:
    overwrite = os.getenv("RV_REGEN_SOBOL", "0") == "1"
    return ensure_base_cases(count=active_sobol_count(), overwrite=overwrite)


def _hallal_block_size(n_sobol: int) -> int:
    return n_sobol * len(HALLAL_METHODS) * len(active_hallal_seeds())


def _rf_block_size(n_sobol: int) -> int:
    return n_sobol * len(RF_METHODS) * len(active_rf_seeds())


def total_combinations() -> int:
    n = active_sobol_count()
    return _hallal_block_size(n) + _rf_block_size(n)


def hallal_block_size() -> int:
    return _hallal_block_size(active_sobol_count())


def rf_block_size() -> int:
    return _rf_block_size(active_sobol_count())


def hallal_index_end() -> int:
    """Exclusive upper bound for Hallal (1D) indices."""
    return hallal_block_size()


def rf_index_range() -> tuple[int, int]:
    """Inclusive start, exclusive end for grf_2d / delatorre indices."""
    start = hallal_block_size()
    return start, start + rf_block_size()


def phase1_array_tasks(*, chunk: int = 24, index_offset: int = 0, index_end: int | None = None) -> int:
    """Number of Slurm array tasks to cover [index_offset, index_end)."""
    end = index_end if index_end is not None else total_combinations()
    count = max(0, end - index_offset)
    return max(1, (count + chunk - 1) // chunk) if count else 0


@dataclass(frozen=True)
class CaseParams:
    index: int
    sobol_id: int
    vs1: float
    H: float
    cov: float
    vs2: float
    method: MethodId
    motion_id: str
    seed: int
    seed_kind: Literal["realization", "rf"]
    rH: float = RH_FIXED
    aHV: float = AHV_FIXED
    bedrock_thickness: float = BEDROCK_DEPTH

    @property
    def rV(self) -> float:
        return self.rH / self.aHV


def index_to_params(index: int) -> CaseParams:
    n_total = total_combinations()
    if index < 0 or index >= n_total:
        raise IndexError(f"Index {index} out of range 0..{n_total - 1}")

    cases = active_base_cases()
    hallal_block = _hallal_block_size(len(cases))
    hallal_seeds = active_hallal_seeds()
    rf_seeds = active_rf_seeds()

    if index < hallal_block:
        per_sobol = len(HALLAL_METHODS) * len(hallal_seeds)
        sobol_idx = index // per_sobol
        r = index % per_sobol
        method_idx = r // len(hallal_seeds)
        seed_idx = r % len(hallal_seeds)
        base = cases[sobol_idx]
        return CaseParams(
            index=index,
            sobol_id=base.sobol_id,
            vs1=base.vs1,
            H=base.H,
            cov=base.cov,
            vs2=base.vs2,
            method=HALLAL_METHODS[method_idx],
            motion_id=MOTION_IDS[0],
            seed=hallal_seeds[seed_idx],
            seed_kind="realization",
        )

    r = index - hallal_block
    per_sobol_rf = len(RF_METHODS) * len(rf_seeds)
    sobol_idx = r // per_sobol_rf
    r2 = r % per_sobol_rf
    method_idx = r2 // len(rf_seeds)
    seed_idx = r2 % len(rf_seeds)
    base = cases[sobol_idx]
    return CaseParams(
        index=index,
        sobol_id=base.sobol_id,
        vs1=base.vs1,
        H=base.H,
        cov=base.cov,
        vs2=base.vs2,
        method=RF_METHODS[method_idx],
        motion_id=MOTION_IDS[0],
        seed=rf_seeds[seed_idx],
        seed_kind="rf",
    )


def active_duration(f0: float) -> float:
    if _smoke_mode():
        return 15.0
    return 50.0 if f0 < 1.0 else 30.0


def motion_frequency(vs1: float, motion_id: str, *, H: float) -> float:
    if motion_id == "M3":
        return vs1 / (4.0 * H)
    freq = MOTION_FREQS.get(motion_id)
    if freq is None:
        raise ValueError(f"No fixed frequency for motion {motion_id}")
    return float(freq)


def case_tag(p: CaseParams) -> str:
    return (
        f"s{p.sobol_id:02d}_{p.method}_Vs1{p.vs1:.0f}_H{p.H:.0f}_"
        f"CoV{p.cov:.2f}_Vs2{p.vs2:.0f}_{p.motion_id}_{p.seed_kind}{p.seed}"
    )


def damping_method_for(p: CaseParams) -> str:
    if p.method == "hallal_dmin":
        return "elemental_varying"
    return "global_avg"
