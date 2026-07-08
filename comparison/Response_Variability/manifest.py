"""Factorial manifest for Response_Variability comparison campaign."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

MethodId = Literal[
    "grf_2d",
    "delatorre_2d",
    "hallal_vs",
    "hallal_tts",
    "hallal_dmin",
]

METHODS: list[MethodId] = [
    "grf_2d",
    "delatorre_2d",
    "hallal_vs",
    "hallal_tts",
    "hallal_dmin",
]

VS1_LIST = [100.0, 230.0, 360.0]
MOTION_IDS = ["M0", "M1", "M2", "M3", "M4"]
MOTION_FREQS = {
    "M0": 0.5,
    "M1": 3.0,
    "M2": 8.0,
    "M3": None,  # f0 per site
    "M4": 3.0,  # placeholder; broadband handled in runner if needed
}

# Primary geostatistics (empirical shallow-soil band)
THICKNESS = 15.0
VS2 = 1500.0
BEDROCK_DEPTH = 10.0
COV = 0.2
RH = 30.0
RV = 0.6  # m; aHV = RH / RV
AHV = RH / RV  # 50.0

DX = 0.5
DZ = 0.5
LX_VAR = 200.0
BC_WIDTH = 100.0
LX_TOTAL = LX_VAR + 2 * BC_WIDTH

DAMPING_ZETA_BASE = 0.025
DAMPING_ZETA_DMIN_MULT = 2.0

SEEDS_FULL = list(range(1, 201))
SEEDS_SMOKE = list(range(1, 11))


def _smoke_mode() -> bool:
    return os.getenv("RV_SMOKE", "0") == "1"


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


def active_duration(f0: float) -> float:
    if _smoke_mode():
        return 15.0
    return 50.0 if f0 < 1.0 else 30.0


def active_vs1_list() -> list[float]:
    if _smoke_mode():
        return [230.0]
    return VS1_LIST


def active_motion_ids() -> list[str]:
    if _smoke_mode():
        return ["M1"]
    return MOTION_IDS


def active_seeds() -> list[int]:
    return SEEDS_SMOKE if _smoke_mode() else SEEDS_FULL


def total_combinations() -> int:
    return len(active_vs1_list()) * len(METHODS) * len(active_motion_ids()) * len(active_seeds())


@dataclass(frozen=True)
class CaseParams:
    index: int
    vs1: float
    method: MethodId
    motion_id: str
    seed: int
    thickness: float = THICKNESS
    cov: float = COV
    rH: float = RH
    rV: float = RV
    aHV: float = AHV


def index_to_params(index: int) -> CaseParams:
    n = total_combinations()
    if index < 0 or index >= n:
        raise IndexError(f"Index {index} out of range 0..{n - 1}")

    seeds = active_seeds()
    motions = active_motion_ids()
    vs1s = active_vs1_list()

    per_vs1 = len(METHODS) * len(motions) * len(seeds)
    vs1_idx = index // per_vs1
    r = index % per_vs1

    per_method = len(motions) * len(seeds)
    method_idx = r // per_method
    r = r % per_method

    motion_idx = r // len(seeds)
    seed_idx = r % len(seeds)

    return CaseParams(
        index=index,
        vs1=vs1s[vs1_idx],
        method=METHODS[method_idx],
        motion_id=motions[motion_idx],
        seed=seeds[seed_idx],
    )


def motion_frequency(vs1: float, motion_id: str) -> float:
    if motion_id == "M3":
        return vs1 / (4.0 * THICKNESS)
    freq = MOTION_FREQS[motion_id]
    if freq is None:
        raise ValueError(f"No fixed frequency for motion {motion_id}")
    return float(freq)


def case_tag(p: CaseParams) -> str:
    return f"{p.method}_Vs1{p.vs1:.0f}_{p.motion_id}_s{p.seed}_rH{p.rH:.0f}_rV{p.rV:.2f}"
