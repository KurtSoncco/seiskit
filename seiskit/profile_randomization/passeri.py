from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "seiskit.profile_randomization"

from dataclasses import replace

import numpy as np

from .common import _total_column_depth
from .models import ProfileRandomizationConfig, RandomizedProfile, _GeoLayer
from .nhpp import (
    _append_bedrock_layer,
    _build_soil_layers_nhpp,
    _sample_bedrock_vs,
    _sample_interface_depth,
)
from .toro import _finalize_profile, toro_adjacent_correlation


def _passeri_joint_bedrock_draw(
    config: ProfileRandomizationConfig,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Joint lognormal draw for bedrock interface depth and Vs (Passeri rho=0.508)."""
    z_total = _total_column_depth(config)
    min_t = max(config.min_layer_thickness, config.dz)
    nominal_depth = float(config.thickness)
    nominal_vs = float(config.vs_bedrock)

    if not config.randomize_bedrock_depth and not config.vary_bedrock_vs:
        return nominal_depth, nominal_vs
    if config.randomize_bedrock_depth and not config.vary_bedrock_vs:
        return _sample_interface_depth(config, rng), nominal_vs
    if not config.randomize_bedrock_depth and config.vary_bedrock_vs:
        return nominal_depth, _sample_bedrock_vs(config, rng)

    mu_depth = np.log(max(config.thickness, min_t))
    mu_vs = np.log(max(config.vs_bedrock, 1e-6))
    sigma_d = config.sigma_ln_interface_depth
    sigma_v = config.sigma_ln_bedrock_vs
    rho = config.passeri_bedrock_depth_vs_rho
    cov = np.array(
        [
            [sigma_d**2, rho * sigma_d * sigma_v],
            [rho * sigma_d * sigma_v, sigma_v**2],
        ]
    )
    ln_d, ln_v = rng.multivariate_normal([mu_depth, mu_vs], cov)
    lo_d = mu_depth - config.clip_std * sigma_d
    hi_d = mu_depth + config.clip_std * sigma_d
    lo_v = mu_vs - config.clip_std * sigma_v
    hi_v = mu_vs + config.clip_std * sigma_v
    interface = float(np.clip(np.exp(np.clip(ln_d, lo_d, hi_d)), min_t, z_total - min_t))
    bed_vs = float(np.exp(np.clip(ln_v, lo_v, hi_v)))
    return interface, bed_vs


def _ar1_standard_scores(n: int, rho_adj: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    z = np.zeros(n, dtype=float)
    eps = rng.standard_normal(n)
    for i in range(n):
        if i == 0:
            z[i] = eps[i]
        else:
            rho_i = float(rho_adj[i - 1])
            z[i] = rho_i * z[i - 1] + np.sqrt(max(1e-12, 1.0 - rho_i**2)) * eps[i]
    return z


def _passeri_tts_layer_vs(
    soil_layers: list[_GeoLayer],
    config: ProfileRandomizationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(soil_layers)
    if n == 0:
        return np.array([], dtype=float)
    thick = np.array([layer.thickness for layer in soil_layers])
    vs_med = np.array([layer.vs_median for layer in soil_layers])
    layer_tt = thick / np.clip(vs_med, 1e-6, None)
    tts_bottom = np.cumsum(layer_tt)
    ln_median = np.log(np.clip(tts_bottom, 1e-12, None))
    mids = np.array([layer.depth_mid for layer in soil_layers])
    rho = toro_adjacent_correlation(
        mids,
        rho_0=config.toro_rho_0,
        delta=config.toro_delta,
        rho_200=config.toro_rho_200,
        b=config.toro_b,
    )
    rho = np.clip(rho + config.tts_rho_boost, 0.0, 0.99)
    z = _ar1_standard_scores(n, rho, rng)
    ln_tts = ln_median + config.sigma_ln_tts * z
    lo = ln_median - config.clip_std * config.sigma_ln_tts
    hi = ln_median + config.clip_std * config.sigma_ln_tts
    tts_rand = np.exp(np.clip(ln_tts, lo, hi))
    tts_top = np.concatenate([[0.0], tts_rand[:-1]])
    layer_tt_r = tts_rand - tts_top
    tt_min = thick / (vs_med * 3.0)
    tt_max = thick / (vs_med / 3.0)
    layer_tt_r = np.clip(layer_tt_r, tt_min, tt_max)
    vs = thick / layer_tt_r
    ln_base = np.log(np.clip(vs_med, 1e-6, None))
    sigma = max(config.sigma_ln_vs, float(np.sqrt(np.log(1.0 + config.cov**2))))
    return np.exp(
        np.clip(
            np.log(np.clip(vs, 1e-6, None)),
            ln_base - config.clip_std * sigma,
            ln_base + config.clip_std * sigma,
        )
    )


def generate_passeri_profile(
    config: ProfileRandomizationConfig,
    rng: np.random.Generator,
) -> RandomizedProfile:
    """Full Passeri: NHPP + tts (soil) -> joint bedrock depth/Vs -> merge."""
    cfg = replace(config, vary_bedrock_vs=True) if not config.vary_bedrock_vs else config
    interface, bed_vs = _passeri_joint_bedrock_draw(cfg, rng)
    soil_layers = _build_soil_layers_nhpp(cfg, interface, rng)
    soil_vs = _passeri_tts_layer_vs(soil_layers, cfg, rng)
    layers = [
        _GeoLayer(layer.thickness, layer.depth_mid, layer.depth_bottom, float(vs), is_bedrock=False)
        for layer, vs in zip(soil_layers, soil_vs)
    ]
    layers = _append_bedrock_layer(layers, interface, bed_vs, cfg)
    layer_vs = np.concatenate([soil_vs, np.array([bed_vs])])
    return _finalize_profile(layers, layer_vs, interface, cfg)


if __name__ == "__main__":
    from seiskit.profile_randomization.common import build_base_case_profile

    config = ProfileRandomizationConfig(
        vs_mean=230.0,
        thickness=15.0,
        bedrock_thickness=10.0,
        dz=0.5,
        vs_bedrock=1500.0,
        use_full_model=True,
        randomize_bedrock_depth=False,
        randomize_layer_thickness=False,
        vary_bedrock_vs=True,
    )

    base_Vs = build_base_case_profile(config)
    print("Base Vs:")
    print(base_Vs)
    n_samples = 100
    rand_vs = np.zeros((n_samples, len(base_Vs)), dtype=float)
    soil_vs = np.zeros(n_samples, dtype=float)
    bed_vs = np.zeros(n_samples, dtype=float)
    for seed in range(n_samples):
        rng = np.random.default_rng(seed)
        profile = generate_passeri_profile(config, rng)
        rand_vs[seed] = profile.vs_depth
        soil_vs[seed] = profile.vs_depth[0]
        bed_vs[seed] = profile.vs_depth[-1]

    # Let's check if it follows a log normal distribution

    # We specify the std ln of tts, compare the std ln of the soil Vs with the specify std ln of tts
    std_ln_tts = config.sigma_ln_tts
    std_ln_soil = np.std(np.log(soil_vs), ddof=1)
    print(f"Std ln of tts: {std_ln_tts:.4f}")
    print(f"Std ln of soil Vs: {std_ln_soil:.4f}")
    print(f"Ratio: {std_ln_soil / std_ln_tts:.4f}")

    # Let's check if the bedrock Vs follows a log normal distribution
    std_ln_bed = np.std(np.log(bed_vs), ddof=1)
    print(f"Std ln of bedrock Vs: {std_ln_bed:.4f}")
    print(f"Ratio: {std_ln_bed / config.sigma_ln_bedrock_vs:.4f}")
