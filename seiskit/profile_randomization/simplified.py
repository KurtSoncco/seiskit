from __future__ import annotations

from typing import Optional

import numpy as np

from .common import _nominal_soil_samples, build_base_case_profile, geological_layer_starts
from .models import ProfileRandomizationConfig, RandomizedProfile, _GeoLayer, _LayerGrid
from .passeri import _ar1_standard_scores, _passeri_tts_layer_vs
from .toro import toro_adjacent_correlation


def _build_geological_layer_grid(
    config: ProfileRandomizationConfig,
    vs_depth: np.ndarray,
) -> _LayerGrid:
    dz = config.dz
    starts = geological_layer_starts(vs_depth, rel_tol=config.layer_jump_rel)
    n_geo = len(starts)
    thickness = np.empty(n_geo, dtype=float)
    depth_mid = np.empty(n_geo, dtype=float)
    depth_bottom = np.empty(n_geo, dtype=float)
    vs_layer = np.empty(n_geo, dtype=float)
    for k, i0 in enumerate(starts):
        i1 = int(starts[k + 1]) if k + 1 < n_geo else len(vs_depth)
        segment = vs_depth[i0:i1]
        thickness[k] = len(segment) * dz
        depth_bottom[k] = i1 * dz
        depth_mid[k] = (i0 * dz + i1 * dz) / 2.0
        vs_layer[k] = float(np.median(segment))
    layer_tt = thickness / np.clip(vs_layer, 1e-6, None)
    tts_bottom = np.cumsum(layer_tt)
    return _LayerGrid(thickness, depth_mid, depth_bottom, vs_layer, tts_bottom, starts)


def _expand_geological_to_depth(vs_geo: np.ndarray, starts: np.ndarray, n_depth: int) -> np.ndarray:
    out = np.empty(n_depth, dtype=float)
    n_geo = len(starts)
    for k, i0 in enumerate(starts):
        i1 = int(starts[k + 1]) if k + 1 < n_geo else n_depth
        out[i0:i1] = vs_geo[k]
    return out


def generate_vs_simplified(
    config: ProfileRandomizationConfig,
    rng: np.random.Generator,
    *,
    vs_base: np.ndarray | None,
    rho: Optional[float],
) -> RandomizedProfile:
    base = (
        build_base_case_profile(config) if vs_base is None else np.asarray(vs_base, float).ravel()
    )
    n_soil = _nominal_soil_samples(config)
    grid = _build_geological_layer_grid(config, base)
    soil_idx = [k for k, s in enumerate(grid.layer_starts) if int(s) < n_soil]
    vs_geo = grid.vs_base.copy()
    if soil_idx:
        ln_median = np.log(np.clip(grid.vs_base[soil_idx], 1e-6, None))
        mids = grid.depth_mid[soil_idx]
        thick = grid.thickness[soil_idx]
        rho_s = (
            np.full(max(1, len(soil_idx) - 1), float(np.clip(rho, 0, 0.99)))
            if rho is not None
            else toro_adjacent_correlation(
                mids,
                rho_0=config.toro_rho_0,
                delta=config.toro_delta,
                rho_200=config.toro_rho_200,
                b=config.toro_b,
            )
        )
        z = _ar1_standard_scores(len(soil_idx), rho_s, rng)
        ln_val = ln_median + config.sigma_ln_vs * z
        lo = ln_median - config.clip_std * config.sigma_ln_vs
        hi = ln_median + config.clip_std * config.sigma_ln_vs
        draw = np.exp(np.clip(ln_val, lo, hi))
        for i, k in enumerate(soil_idx):
            vs_geo[k] = draw[i]
    vs_depth = _expand_geological_to_depth(vs_geo, grid.layer_starts, len(base))
    return RandomizedProfile(vs_depth, n_soil, float(config.thickness))


def generate_tts_simplified(
    config: ProfileRandomizationConfig,
    rng: np.random.Generator,
    *,
    vs_base: np.ndarray | None,
    rho: Optional[float],
) -> RandomizedProfile:
    base = (
        build_base_case_profile(config) if vs_base is None else np.asarray(vs_base, float).ravel()
    )
    n_soil = _nominal_soil_samples(config)
    grid = _build_geological_layer_grid(config, base)
    soil_idx = [k for k, s in enumerate(grid.layer_starts) if int(s) < n_soil]
    vs_geo = grid.vs_base.copy()
    if soil_idx:
        layers = [
            _GeoLayer(
                grid.thickness[k],
                grid.depth_mid[k],
                grid.depth_bottom[k],
                grid.vs_base[k],
            )
            for k in soil_idx
        ]
        vs_soil = _passeri_tts_layer_vs(layers, config, rng)
        for i, k in enumerate(soil_idx):
            vs_geo[k] = vs_soil[i]
    vs_depth = _expand_geological_to_depth(vs_geo, grid.layer_starts, len(base))
    return RandomizedProfile(vs_depth, n_soil, float(config.thickness))
