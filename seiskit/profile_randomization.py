"""1D Vs profile randomization (Hallal / Passeri / Toro-style) and vertical ACF utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class ProfileRandomizationConfig:
    """Configuration for 1D profile randomization."""

    vs_mean: float
    thickness: float
    dz: float
    cov: float = 0.2
    sigma_ln_vs: float = 0.25
    layer_thickness: float = 0.5
    rho_interlayer: float = 0.7
    sigma_ln_tts: float = 0.02
    clip_std: float = 2.0


def toro_rho(depth_m: float, layer_sep_m: float, rho200: float = 0.43, d0: float = 1.0) -> float:
    """Toro (1995) interlayer correlation at midpoint depth *d* and separation *t*."""
    if depth_m > 200.0:
        rho_d = rho200
    else:
        rho_d = rho200 * depth_m / (depth_m + d0)
    rho_t = 0.55 * np.exp(-layer_sep_m / 16.3)
    return float(np.clip(rho_d * rho_t, 0.0, 0.99))


def _lognormal_params(mean: float, cov: float) -> tuple[float, float]:
    sigma2 = np.log(1.0 + cov**2)
    log_mean = np.log(mean) - 0.5 * sigma2
    log_std = np.sqrt(sigma2)
    return log_mean, log_std


def _layer_midpoints(n_layers: int, dz: float) -> np.ndarray:
    return (np.arange(n_layers) + 0.5) * dz


def generate_vs_randomized_profile(
    config: ProfileRandomizationConfig,
    rng: np.random.Generator,
    *,
    rho: Optional[float] = None,
) -> np.ndarray:
    """Generate a 1D lognormal Vs profile with AR(1) correlation between layers."""
    n_layers = max(1, int(round(config.thickness / config.layer_thickness)))
    dz = config.thickness / n_layers
    mid = _layer_midpoints(n_layers, dz)
    log_mean, log_std = _lognormal_params(config.vs_mean, config.cov)
    sigma_ln = config.sigma_ln_vs if config.sigma_ln_vs > 0 else log_std

    z = np.zeros(n_layers)
    eps = rng.standard_normal(n_layers)
    for i in range(n_layers):
        if i == 0:
            z[i] = eps[i]
        else:
            sep = mid[i] - mid[i - 1]
            rho_i = rho if rho is not None else toro_rho(0.5 * (mid[i] + mid[i - 1]), sep)
            z[i] = rho_i * z[i - 1] + np.sqrt(max(1e-12, 1.0 - rho_i**2)) * eps[i]

    vs = np.exp(log_mean + sigma_ln * z)
    vmin, vmax = (
        np.exp(log_mean - config.clip_std * sigma_ln),
        np.exp(log_mean + config.clip_std * sigma_ln),
    )
    return np.clip(vs, vmin, vmax)


def generate_tts_randomized_profile(
    config: ProfileRandomizationConfig,
    rng: np.random.Generator,
    *,
    rho: Optional[float] = None,
) -> np.ndarray:
    """Randomize cumulative travel time per layer, then convert to layer Vs."""
    n_layers = max(1, int(round(config.thickness / config.layer_thickness)))
    dz = config.thickness / n_layers
    mid = _layer_midpoints(n_layers, dz)

    tts_median = np.cumsum(np.full(n_layers, dz / config.vs_mean))
    log_tts = np.log(tts_median)

    z = np.zeros(n_layers)
    eps = rng.standard_normal(n_layers)
    for i in range(n_layers):
        if i == 0:
            z[i] = eps[i]
        else:
            sep = mid[i] - mid[i - 1]
            rho_i = rho if rho is not None else toro_rho(0.5 * (mid[i] + mid[i - 1]), sep)
            z[i] = rho_i * z[i - 1] + np.sqrt(max(1e-12, 1.0 - rho_i**2)) * eps[i]

    tts_sim = np.exp(log_tts + config.sigma_ln_tts * z)
    tts_sim = np.maximum.accumulate(tts_sim)
    layer_tt = np.diff(np.concatenate([[0.0], tts_sim]))
    layer_tt = np.clip(layer_tt, dz / (config.vs_mean * 3.0), dz / (config.vs_mean / 3.0))
    vs = dz / layer_tt

    log_mean, log_std = _lognormal_params(config.vs_mean, config.cov)
    vmin, vmax = (
        np.exp(log_mean - config.clip_std * log_std),
        np.exp(log_mean + config.clip_std * log_std),
    )
    return np.clip(vs, vmin, vmax)


def build_layered_profile(
    soil_vs: np.ndarray,
    vs_bedrock: float,
    bedrock_depth: float,
    dz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack soil layers and bedrock into 2D (nz, 1) arrays for OpenSees."""
    n_bed = max(1, int(round(bedrock_depth / dz)))
    soil = np.asarray(soil_vs, dtype=float).ravel()
    n_soil = len(soil)
    vs_col = np.concatenate([soil, np.full(n_bed, vs_bedrock)])
    mask = np.concatenate([np.zeros(n_soil, dtype=bool), np.ones(n_bed, dtype=bool)])
    return vs_col.reshape(-1, 1), mask.reshape(-1, 1)


def vertical_acf_ln_vs(
    vs_column: np.ndarray,
    dz: float,
    max_lag_m: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical ACF of ln(Vs) for a vertical column at lags up to *max_lag_m*."""
    vs = np.asarray(vs_column, dtype=float).ravel()
    ln_vs = np.log(np.clip(vs, 1e-6, None))
    ln_vs = ln_vs - ln_vs.mean()
    var = float(np.var(ln_vs))
    if var < 1e-12:
        lags = np.array([0.0])
        return lags, np.array([1.0])

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
    soil_mask: Optional[np.ndarray] = None,
    max_lag_m: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract vertical ACF from a 2D Vs field (soil rows only)."""
    field = np.asarray(vs_field, dtype=float)
    if field.ndim != 2:
        raise ValueError("vs_field must be 2D (nz, nx)")

    if soil_mask is not None:
        soil_rows = np.any(~np.asarray(soil_mask, dtype=bool), axis=1)
    else:
        soil_rows = np.ones(field.shape[0], dtype=bool)

    col_idx: int
    if column == "center":
        col_idx = field.shape[1] // 2
    else:
        col_idx = int(column)

    col = field[soil_rows, col_idx]
    return vertical_acf_ln_vs(col, dz, max_lag_m=max_lag_m)


def acf_rmse(
    lags_a: np.ndarray,
    acf_a: np.ndarray,
    lags_b: np.ndarray,
    acf_b: np.ndarray,
    *,
    lag_max: float = 2.0,
) -> float:
    """RMSE between two ACF curves on a common lag grid up to *lag_max*."""
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
    """Find constant *rho* minimizing ACF mismatch to a 2D reference."""

    def objective(rho: float) -> float:
        rho_c = float(np.clip(rho, 0.0, 0.98))
        acfs = []
        for s in range(n_profiles):
            sub = np.random.default_rng(rng.integers(0, 2**31 - 1) + s)
            if method == "tts":
                prof = generate_tts_randomized_profile(config, sub, rho=rho_c)
            else:
                prof = generate_vs_randomized_profile(config, sub, rho=rho_c)
            _, acf = vertical_acf_ln_vs(prof, config.layer_thickness, max_lag_m=2.0)
            acfs.append(acf)
        mean_acf = np.mean(np.vstack([a[: len(target_acf)] for a in acfs]), axis=0)
        lags = target_lags[: len(mean_acf)]
        return acf_rmse(lags, mean_acf, target_lags, target_acf, lag_max=2.0)

    res = minimize_scalar(objective, bounds=(0.0, 0.98), method="bounded")
    return float(res.x)


def profile_cov(vs: np.ndarray) -> float:
    """Coefficient of variation of a Vs profile."""
    vs = np.asarray(vs, dtype=float).ravel()
    m = float(np.mean(vs))
    if m <= 0:
        return 0.0
    return float(np.std(vs) / m)
