"""Sobol-driven manifest for Response_Variability comparison campaign."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
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
    "pretell",
    "hallal_vs",
    "hallal_tts",
    "hallal_dmin",
]

HALLAL_METHODS: list[MethodId] = ["hallal_vs", "hallal_tts", "hallal_dmin"]
RF_METHODS: list[MethodId] = ["grf_2d", "pretell"]
METHODS: list[MethodId] = [*HALLAL_METHODS, *RF_METHODS]

# ---------------------------------------------------------------------------
# GIFNO / neural-operator 2D training grid (Pretell & grf_2d use this domain)
# ---------------------------------------------------------------------------
# Lateral: 1500 m total = 500 m BC + 500 m variability + 500 m BC  (dx = 1 m)
# Depth:   nz ≤ 128 rows at dz = 1 m (padded in surrogate input)
# TF:      21 lateral recorders, 1000 log-spaced freqs 0.1–10 Hz
NO_DX = 1.0
NO_DZ = 1.0
NO_LX_VAR = 500.0
NO_BC_WIDTH = 500.0
NO_LX_TOTAL = NO_BC_WIDTH + NO_LX_VAR + NO_BC_WIDTH  # 1500 m
NO_NX_FULL = int(NO_LX_TOTAL / NO_DX)  # 1500 columns
NO_NZ_MAX = 128

# Pretell et al. (2022): 1D profiles from central 100 m of the 2D field.
# Minimum 10 samples; 50 sufficient for production (Table recommendations).
PRETELL_CENTRAL_WIDTH_M = 100.0
PRETELL_SAMPLES_SMOKE = 10
PRETELL_SAMPLES_FULL = 50

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

# Hallal Approach 5: Dmin multiplier sweep (10 values, linspace 3–6).
DMIN_MULTIPLIERS: tuple[float, ...] = tuple(float(x) for x in np.linspace(3.0, 6.0, 10))

RH = RH_FIXED
AHV = AHV_FIXED
RV = RH_FIXED / AHV_FIXED


def _smoke_mode() -> bool:
    return os.getenv("RV_SMOKE", "0") == "1"


def smoke_includes_2d() -> bool:
    """In smoke mode, 2D arms (grf_2d, pretell) are opt-in via RV_SMOKE_2D=1."""
    if not _smoke_mode():
        return True
    return os.getenv("RV_SMOKE_2D", "0") == "1"


def _include_rf_block() -> bool:
    return smoke_includes_2d() if _smoke_mode() else True


def active_sobol_count() -> int:
    return DEFAULT_SOBOL_COUNT_SMOKE if _smoke_mode() else DEFAULT_SOBOL_COUNT_FULL


def active_hallal_seeds() -> list[int]:
    base = HALLAL_SEEDS_SMOKE if _smoke_mode() else HALLAL_SEEDS_FULL
    raw = os.getenv("RV_HALLAL_N_SEEDS")
    if raw:
        n = max(1, int(raw))
        # Allow counts beyond the smoke default list (e.g. RV_HALLAL_N_SEEDS=50).
        pool = HALLAL_SEEDS_FULL if n > len(base) else base
        return pool[: min(n, len(pool))]
    return base


def active_rf_seeds() -> list[int]:
    base = RF_SEEDS_SMOKE if _smoke_mode() else RF_SEEDS_FULL
    raw = os.getenv("RV_RF_N_SEEDS")
    if raw:
        n = max(1, int(raw))
        # Allow RV_RF_N_SEEDS beyond the smoke default list (e.g. --n-seeds 10).
        pool = RF_SEEDS_FULL if n > len(base) else base
        return pool[: min(n, len(pool))]
    return base


def active_dmin_multipliers() -> tuple[float, ...]:
    return DMIN_MULTIPLIERS


def _hallal_entries_per_sobol() -> int:
    return 2 * len(active_hallal_seeds()) + len(active_dmin_multipliers())


def active_lx_var() -> float:
    return 100.0 if _smoke_mode() else LX_VAR


def active_bc_width() -> float:
    return 50.0 if _smoke_mode() else BC_WIDTH


def active_lx_total() -> float:
    return active_lx_var() + 2 * active_bc_width()


def active_dx() -> float:
    return DX


def active_dz() -> float:
    return DZ


def active_rf_dx() -> float:
    return NO_DX


def active_rf_dz() -> float:
    return NO_DZ


def active_rf_lx_var() -> float:
    return NO_LX_VAR


def active_rf_bc_width() -> float:
    return NO_BC_WIDTH


def active_rf_lx_total() -> float:
    return NO_LX_TOTAL


def active_pretell_n_samples() -> int:
    raw = os.getenv("RV_PRETELL_N_SAMPLES")
    if raw:
        return max(1, int(raw))
    return PRETELL_SAMPLES_SMOKE if _smoke_mode() else PRETELL_SAMPLES_FULL


def pretell_column_indices(n_samples: int | None = None) -> np.ndarray:
    """
    Column indices (full 1500 m grid) for evenly spaced 1D extractions.

    Samples lie in the central ``PRETELL_CENTRAL_WIDTH_M`` of the 500 m
    variability strip (Pretell et al. 2022).
    """
    n = n_samples if n_samples is not None else active_pretell_n_samples()
    dx = active_rf_dx()
    bc_cols = int(round(active_rf_bc_width() / dx))
    strip_cols = int(round(active_rf_lx_var() / dx))
    central_cols = int(round(PRETELL_CENTRAL_WIDTH_M / dx))
    i0_strip = (strip_cols - central_cols) // 2
    i1_strip = i0_strip + central_cols - 1
    cols_strip = np.linspace(i0_strip, i1_strip, n, dtype=int)
    return bc_cols + cols_strip


def active_motion_ids() -> list[str]:
    return list(MOTION_IDS)


def active_base_cases() -> list[SobolBaseCase]:
    overwrite = os.getenv("RV_REGEN_SOBOL", "0") == "1"
    return ensure_base_cases(count=active_sobol_count(), overwrite=overwrite)


def _hallal_block_size(n_sobol: int) -> int:
    return n_sobol * _hallal_entries_per_sobol()


def _rf_block_size(n_sobol: int) -> int:
    return n_sobol * len(RF_METHODS) * len(active_rf_seeds())


def total_combinations() -> int:
    n = active_sobol_count()
    total = _hallal_block_size(n)
    if _include_rf_block():
        total += _rf_block_size(n)
    return total


def hallal_block_size() -> int:
    return _hallal_block_size(active_sobol_count())


def rf_block_size() -> int:
    if not _include_rf_block():
        return 0
    return _rf_block_size(active_sobol_count())


def hallal_index_end() -> int:
    """Exclusive upper bound for Hallal (1D) indices."""
    return hallal_block_size()


def rf_index_range() -> tuple[int, int]:
    """Inclusive start, exclusive end for grf_2d / pretell indices."""
    start = hallal_block_size()
    return start, start + rf_block_size()


def phase1_array_tasks(
    *, chunk: int = 24, index_offset: int = 0, index_end: int | None = None
) -> int:
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
    seed_kind: Literal["realization", "rf", "dmin_mult"]
    rH: float = RH_FIXED
    aHV: float = AHV_FIXED
    bedrock_thickness: float = BEDROCK_DEPTH
    dmin_multiplier: float = 1.0

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
        per_sobol = _hallal_entries_per_sobol()
        sobol_idx = index // per_sobol
        r = index % per_sobol
        base = cases[sobol_idx]
        n_vs = len(hallal_seeds)
        n_tts = len(hallal_seeds)
        if r < n_vs:
            return CaseParams(
                index=index,
                sobol_id=base.sobol_id,
                vs1=base.vs1,
                H=base.H,
                cov=base.cov,
                vs2=base.vs2,
                method="hallal_vs",
                motion_id=MOTION_IDS[0],
                seed=hallal_seeds[r],
                seed_kind="realization",
            )
        if r < n_vs + n_tts:
            return CaseParams(
                index=index,
                sobol_id=base.sobol_id,
                vs1=base.vs1,
                H=base.H,
                cov=base.cov,
                vs2=base.vs2,
                method="hallal_tts",
                motion_id=MOTION_IDS[0],
                seed=hallal_seeds[r - n_vs],
                seed_kind="realization",
            )
        dmin_idx = r - n_vs - n_tts
        mults = active_dmin_multipliers()
        return CaseParams(
            index=index,
            sobol_id=base.sobol_id,
            vs1=base.vs1,
            H=base.H,
            cov=base.cov,
            vs2=base.vs2,
            method="hallal_dmin",
            motion_id=MOTION_IDS[0],
            seed=dmin_idx + 1,
            seed_kind="dmin_mult",
            dmin_multiplier=mults[dmin_idx],
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
    tag = (
        f"s{p.sobol_id:02d}_{p.method}_Vs1{p.vs1:.0f}_H{p.H:.0f}_"
        f"CoV{p.cov:.2f}_Vs2{p.vs2:.0f}_{p.motion_id}_{p.seed_kind}{p.seed}"
    )
    if p.method == "hallal_dmin":
        tag += f"_dmin{p.dmin_multiplier:.2f}"
    return tag


def damping_method_for(p: CaseParams) -> str:
    if p.method == "hallal_dmin":
        return "elemental_varying"
    return "global_avg"


def dmin_multiplier_for(p: CaseParams) -> float:
    """Hallal Approach 5: Dmin multiplier from the 3–6 linspace sweep."""
    if p.method != "hallal_dmin":
        return 1.0
    return float(p.dmin_multiplier)
