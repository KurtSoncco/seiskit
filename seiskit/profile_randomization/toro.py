from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "seiskit.profile_randomization"

import numpy as np

from .common import (
    _resample_layers_to_dz,
    _soil_sample_index,
    _total_column_depth,
)
from .models import ProfileRandomizationConfig, RandomizedProfile, _GeoLayer
from .nhpp import _append_bedrock_layer, _build_soil_layers_nhpp, _sample_interface_depth


def toro_adjacent_correlation(
    depth_mid: np.ndarray,
    *,
    rho_0: float,
    delta: float,
    rho_200: float,
    b: float,
    bedrock_interface: bool = False,
    bedrock_interface_rho: float = 1.0,
) -> np.ndarray:
    """Adjacent-layer correlation (Toro 1995 eq. 2-4).

    Args:
        depth_mid (np.ndarray): The mid-depths of the layers.
        rho_0 (float): The rho_0 parameter.
        delta (float): The delta parameter.
        rho_200 (float): The rho_200 parameter.
        b (float): The b parameter.
        bedrock_interface (bool): Whether the bedrock interface is present.
        bedrock_interface_rho (float): The rho of the bedrock interface.

    Returns:
        np.ndarray: The adjacency correlation matrix.
    """
    mid = np.asarray(depth_mid, dtype=float)
    n = len(mid)

    if n < 2:
        return np.array([1.0])

    t = np.diff(mid)
    d = 0.5 * (mid[:-1] + mid[1:])

    corr_depth = rho_200 * np.power((d + rho_0) / (200.0 + rho_0), b)
    corr_depth = np.where(d > 200.0, rho_200, corr_depth)
    corr_thick = rho_0 * np.exp(-t / delta)
    corr = np.clip((1.0 - corr_depth) * corr_thick + corr_depth, 0.0, 0.99)

    if bedrock_interface:
        corr[-1] = float(np.clip(bedrock_interface_rho, 0.0, 0.99))
    return corr


def _toro_covariance_matrix(sigma_ln: np.ndarray, corr_adj: np.ndarray) -> np.ndarray:
    """Toro covariance matrix (Toro 1995 eq. 2-5).

    Args:
        sigma_ln (np.ndarray): The log standard deviations of the layers.
        corr_adj (np.ndarray): The adjacency correlation matrix.

    Returns:
        np.ndarray: The covariance matrix.
    """
    # Compute the covariance matrix
    n = len(sigma_ln)
    std = np.asarray(sigma_ln, dtype=float)
    var = std**2
    mat = np.diag(var)
    n_pair = min(len(corr_adj), n - 1)
    for i in range(n_pair):
        rho = float(np.clip(corr_adj[i], -0.99, 0.99))
        cov = rho * std[i] * std[i + 1]
        mat[i, i + 1] = cov
        mat[i + 1, i] = cov
    eigval, eigvec = np.linalg.eigh(mat)
    eigval = np.clip(eigval, 1e-12, None)
    return eigvec @ np.diag(eigval) @ eigvec.T


def _toro_draw_layer_vs(
    layers: list[_GeoLayer],
    config: ProfileRandomizationConfig,
    rng: np.random.Generator,
    *,
    randomize_bedrock: bool,
) -> np.ndarray:
    ln_median = np.log(np.clip([layer.vs_median for layer in layers], 1e-6, None))
    sigma = np.array(
        [
            config.sigma_ln_bedrock_vs if layer.is_bedrock else config.sigma_ln_vs
            for layer in layers
        ],
        dtype=float,
    )
    mids = np.array([layer.depth_mid for layer in layers])
    has_bedrock = any(layer.is_bedrock for layer in layers)
    corr = toro_adjacent_correlation(
        mids,
        rho_0=config.toro_rho_0,
        delta=config.toro_delta,
        rho_200=config.toro_rho_200,
        b=config.toro_b,
        bedrock_interface=has_bedrock and not randomize_bedrock,
        bedrock_interface_rho=config.toro_bedrock_interface_rho,
    )
    cov = _toro_covariance_matrix(sigma, corr)
    ln_draw = rng.multivariate_normal(ln_median, cov)
    lo = ln_median - config.clip_std * sigma
    hi = ln_median + config.clip_std * sigma
    vs = np.exp(np.clip(ln_draw, lo, hi))
    if has_bedrock and not randomize_bedrock:
        for i, layer in enumerate(layers):
            if layer.is_bedrock:
                vs[i] = layer.vs_median
    return vs


def _finalize_profile(
    layers: list[_GeoLayer],
    layer_vs: np.ndarray,
    interface_depth: float,
    config: ProfileRandomizationConfig,
) -> RandomizedProfile:
    thicknesses = [layer.thickness for layer in layers]
    vs_depth = _resample_layers_to_dz(
        thicknesses,
        layer_vs,
        config.dz,
        _total_column_depth(config),
    )
    n_soil = _soil_sample_index(vs_depth, interface_depth, config.dz)
    return RandomizedProfile(
        vs_depth=vs_depth, n_soil_samples=n_soil, interface_depth=interface_depth
    )


def generate_toro_profile(
    config: ProfileRandomizationConfig,
    rng: np.random.Generator,
) -> RandomizedProfile:
    """Full Toro: NHPP thicknesses -> bedrock depth -> correlated Vs.

    Args:
        config (ProfileRandomizationConfig): The configuration for the profile randomization.
        rng (np.random.Generator): The random number generator.

    Returns:
        RandomizedProfile: The randomized profile.
    """
    interface = _sample_interface_depth(config, rng)
    soil_layers = _build_soil_layers_nhpp(config, interface, rng)
    bed_vs = float(config.vs_bedrock)
    layers = _append_bedrock_layer(soil_layers, interface, bed_vs, config)
    layer_vs = _toro_draw_layer_vs(layers, config, rng, randomize_bedrock=config.vary_bedrock_vs)
    return _finalize_profile(layers, layer_vs, interface, config)


if __name__ == "__main__":
    from scipy import stats

    from seiskit.profile_randomization.common import build_base_case_profile

    config = ProfileRandomizationConfig(
        vs_mean=230.0,
        thickness=15.0,
        bedrock_thickness=10.0,
        dz=0.5,
        vs_bedrock=1500.0,
        randomize_bedrock_depth=False,
        randomize_layer_thickness=False,
        vary_bedrock_vs=True,
    )

    n_seeds = 100
    base_vs = build_base_case_profile(config)
    rand_vs = np.zeros((n_seeds, len(base_vs)), dtype=float)
    soil_vs = np.zeros(n_seeds, dtype=float)
    bed_vs = np.zeros(n_seeds, dtype=float)

    print("Toro Vs-only mode (fixed soil thickness + fixed bedrock depth)")
    print(f"  randomize_layer_thickness={config.randomize_layer_thickness}")
    print(f"  randomize_bedrock_depth={config.randomize_bedrock_depth}")
    print(f"  vary_bedrock_vs={config.vary_bedrock_vs}")
    print()

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        prof = generate_toro_profile(config, rng)
        rand_vs[seed] = prof.vs_depth
        soil_vs[seed] = prof.vs_depth[0]
        bed_vs[seed] = prof.vs_depth[-1]
        print(
            f"Seed {seed}: interface={prof.interface_depth:.3f} m, "
            f"soil Vs={soil_vs[seed]:.2f} m/s, bedrock Vs={bed_vs[seed]:.2f} m/s"
        )

    # One geological soil layer -> one scalar Vs per realization; check lognormality.
    print()
    print("Soil Vs ensemble (one value per realization):")
    print(np.round(soil_vs, 2))
    ln_soil = np.log(soil_vs)
    mu_hat = float(np.mean(ln_soil))
    sigma_hat = float(np.std(ln_soil, ddof=1))
    CoV_hat = sigma_hat / mu_hat
    print(
        f"  ln(Vs) mean: {mu_hat:.4f} (target ln({config.vs_mean}) = {np.log(config.vs_mean):.4f})"
    )
    print(f"  ln(Vs) std:  {sigma_hat:.4f} (target sigma_ln_vs = {config.sigma_ln_vs})")
    print(f"  ln(Vs) CoV:  {CoV_hat:.4f} (target CoV = {config.sigma_ln_vs / config.vs_mean})")
    _, p_shapiro = stats.shapiro(ln_soil)
    print(f"  Shapiro-Wilk on ln(Vs), n={n_seeds}: p={p_shapiro:.4f} (small n; indicative only)")
    if config.vary_bedrock_vs:
        print()
        print("Bedrock Vs ensemble (one value per realization):")
        print(np.round(bed_vs, 2))
        ln_bed = np.log(bed_vs)
        print(
            f"  ln(Vs) mean: {float(np.mean(ln_bed)):.4f} "
            f"(target ln({config.vs_bedrock}) = {np.log(config.vs_bedrock):.4f})"
        )
        print(
            f"  ln(Vs) std:  {float(np.std(ln_bed, ddof=1)):.4f} "
            f"(target sigma_ln_bedrock_vs = {config.sigma_ln_bedrock_vs})"
        )

    assert np.allclose(
        [
            prof.interface_depth
            for prof in [generate_toro_profile(config, np.random.default_rng(i)) for i in range(3)]
        ],
        config.thickness,
    )
    if config.vary_bedrock_vs:
        assert np.std(bed_vs) > 1.0
        print()
        print("OK: interface fixed; soil and bedrock Vs vary by seed.")
    else:
        assert np.all(rand_vs[:, 30:] == config.vs_bedrock)
        print()
        print("OK: interface fixed at nominal thickness; bedrock Vs fixed; soil Vs varies by seed.")
