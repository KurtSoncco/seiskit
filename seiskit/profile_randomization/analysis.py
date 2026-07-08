from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import minimize_scalar

from .api import generate_tts_randomized_profile, generate_vs_randomized_profile
from .common import _nominal_soil_samples
from .models import ProfileRandomizationConfig


def toro_rho(depth_m: float, layer_sep_m: float, rho200: float = 0.43, d0: float = 1.0) -> float:
    if depth_m > 200.0:
        rho_d = rho200
    else:
        rho_d = rho200 * depth_m / (depth_m + d0)
    rho_t = 0.55 * np.exp(-layer_sep_m / 16.3)
    return float(np.clip(rho_d * rho_t, 0.0, 0.99))


def vertical_acf_ln_vs(
    vs_column: np.ndarray,
    dz: float,
    max_lag_m: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    vs = np.asarray(vs_column, dtype=float).ravel()
    ln_vs = np.log(np.clip(vs, 1e-6, None))
    ln_vs = ln_vs - ln_vs.mean()
    var = float(np.var(ln_vs))
    if var < 1e-12:
        return np.array([0.0]), np.array([1.0])
    max_lag_idx = min(len(vs) - 1, int(round(max_lag_m / dz)))
    lags = np.arange(max_lag_idx + 1) * dz
    acf = np.empty(max_lag_idx + 1)
    for k in range(max_lag_idx + 1):
        acf[k] = float(np.mean(ln_vs[: len(ln_vs) - k] * ln_vs[k:])) / var
    return lags, acf


def vertical_acf_from_2d_field(
    vs_field: np.ndarray,
    dz: float,
    *,
    column: int | str = "center",
    soil_mask: np.ndarray | None = None,
    max_lag_m: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    field = np.asarray(vs_field, dtype=float)
    if field.ndim != 2:
        raise ValueError("vs_field must be 2D (nz, nx)")
    if soil_mask is not None:
        soil_rows = np.any(~np.asarray(soil_mask, dtype=bool), axis=1)
    else:
        soil_rows = np.ones(field.shape[0], dtype=bool)
    col_idx = field.shape[1] // 2 if column == "center" else int(column)
    return vertical_acf_ln_vs(field[soil_rows, col_idx], dz, max_lag_m=max_lag_m)


def acf_rmse(
    lags_a: np.ndarray,
    acf_a: np.ndarray,
    lags_b: np.ndarray,
    acf_b: np.ndarray,
    *,
    lag_max: float = 2.0,
) -> float:
    grid = np.linspace(0.0, lag_max, 41)

    def _interp(lags: np.ndarray, acf: np.ndarray) -> np.ndarray:
        return np.interp(grid, lags, acf, left=acf[0], right=acf[-1])

    return float(np.sqrt(np.mean((_interp(lags_a, acf_a) - _interp(lags_b, acf_b)) ** 2)))


def calibrate_rho_to_target_acf(
    config: ProfileRandomizationConfig,
    target_lags: np.ndarray,
    target_acf: np.ndarray,
    rng: np.random.Generator,
    *,
    n_profiles: int = 30,
    method: str = "vs",
) -> float:
    cfg = replace(config, use_full_model=False)

    def objective(rho: float) -> float:
        rho_c = float(np.clip(rho, 0.0, 0.98))
        acfs = []
        for seed in range(n_profiles):
            sub = np.random.default_rng(int(rng.integers(0, 2**31 - 1)) + seed)
            if method == "tts":
                prof = generate_tts_randomized_profile(cfg, sub, rho=rho_c)
            else:
                prof = generate_vs_randomized_profile(cfg, sub, rho=rho_c)
            _, acf = vertical_acf_ln_vs(prof[: _nominal_soil_samples(cfg)], cfg.dz, max_lag_m=2.0)
            acfs.append(acf)
        mean_acf = np.mean(np.vstack([a[: len(target_acf)] for a in acfs]), axis=0)
        return acf_rmse(
            target_lags[: len(mean_acf)], mean_acf, target_lags, target_acf, lag_max=2.0
        )

    res = minimize_scalar(objective, bounds=(0.0, 0.98), method="bounded")
    return float(res.x)
