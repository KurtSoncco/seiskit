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
    h0: float = 0.0,
    bedrock_interface: bool = False,
    bedrock_interface_rho: float = 1.0,
) -> np.ndarray:
    """Adjacent-layer correlation (Toro 1995 eq. 2-4 / Toro 2022 eqs. 6–8).

    ``rho_0`` and ``delta`` enter the thickness term; ``h0`` is the depth offset
    in metres (Table 1 generic value is 0), not ``rho_0``.
    """
    mid = np.asarray(depth_mid, dtype=float)
    n = len(mid)

    if n < 2:
        return np.array([1.0])

    t = np.diff(mid)
    d = 0.5 * (mid[:-1] + mid[1:])
    h0 = float(h0)

    corr_depth = rho_200 * np.power((d + h0) / (200.0 + h0), b)
    corr_depth = np.where(d > 200.0, rho_200, corr_depth)
    corr_thick = rho_0 * np.exp(-t / delta)
    corr = np.clip((1.0 - corr_depth) * corr_thick + corr_depth, 0.0, 0.99)

    if bedrock_interface:
        corr[-1] = float(np.clip(bedrock_interface_rho, 0.0, 0.99))
    return corr


def _ar1_standard_scores(n: int, rho_adj: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Toro 2022 eq. 5: Z_i = rho Z_{i-1} + sqrt(1-rho^2) epsilon."""
    z = np.zeros(int(n), dtype=float)
    if n <= 0:
        return z
    eps = rng.standard_normal(int(n))
    rho_adj = np.asarray(rho_adj, dtype=float).ravel()
    z[0] = eps[0]
    for i in range(1, int(n)):
        rho_i = float(rho_adj[min(i - 1, max(0, len(rho_adj) - 1))])
        rho_i = float(np.clip(rho_i, -0.99, 0.99))
        z[i] = rho_i * z[i - 1] + np.sqrt(max(1e-12, 1.0 - rho_i**2)) * eps[i]
    return z


def toro_sigma_ln_vs(depth_m: np.ndarray | float, config: ProfileRandomizationConfig) -> np.ndarray:
    """SPID-style σ_ln V(z): surface value tapering to ``sigma_ln_vs`` by ``sigma_ln_vs_depth_m``."""
    z = np.asarray(depth_m, dtype=float)
    z1 = max(float(config.sigma_ln_vs_depth_m), 1e-12)
    s0 = float(config.sigma_ln_vs_surface)
    s1 = float(config.sigma_ln_vs)
    t = np.clip(z / z1, 0.0, 1.0)
    out = s0 + (s1 - s0) * t
    return np.asarray(out, dtype=float)


def _toro_draw_layer_vs(
    layers: list[_GeoLayer],
    config: ProfileRandomizationConfig,
    rng: np.random.Generator,
    *,
    randomize_bedrock: bool,
    reject_profile: bool = False,
    rho: float | None = None,
    max_reject: int = 2000,
) -> np.ndarray:
    """AR(1) lognormal Vs (Toro 2022 eq. 4–5) with 1.16 σ inflation and |Z|≤2.

    Grid / frozen-H draws truncate each Z_i. Coarse NHPP draws (``reject_profile``)
    redraw the whole profile if any |Z_i| exceeds ``clip_std``.
    """
    n = len(layers)
    ln_median = np.log(np.clip([layer.vs_median for layer in layers], 1e-6, None))
    mids = np.array([layer.depth_mid for layer in layers], dtype=float)
    sigma = np.empty(n, dtype=float)
    for i, layer in enumerate(layers):
        if layer.is_bedrock:
            sigma[i] = float(config.sigma_ln_bedrock_vs) if randomize_bedrock else 0.0
        else:
            sigma[i] = float(toro_sigma_ln_vs(layer.depth_mid, config).reshape(()))
    has_bedrock = any(layer.is_bedrock for layer in layers)
    if rho is not None:
        corr = np.full(max(1, n - 1), float(np.clip(rho, 0.0, 0.99)))
    else:
        corr = toro_adjacent_correlation(
            mids,
            rho_0=config.toro_rho_0,
            delta=config.toro_delta,
            rho_200=config.toro_rho_200,
            b=config.toro_b,
            h0=config.toro_h0,
            bedrock_interface=has_bedrock and not randomize_bedrock,
            bedrock_interface_rho=config.toro_bedrock_interface_rho,
        )
    clip = float(config.clip_std)
    inflate = float(config.toro_sigma_inflate)

    def _draw_z(rng_i: np.random.Generator) -> np.ndarray:
        return _ar1_standard_scores(n, corr, rng_i)

    if reject_profile and n > 0:
        z = _draw_z(rng)
        tries = 1
        while np.any(np.abs(z) >= clip) and tries < int(max_reject):
            z = _draw_z(rng)
            tries += 1
        if np.any(np.abs(z) >= clip):
            z = np.clip(z, -clip, clip)
    else:
        z = np.clip(_draw_z(rng), -clip, clip)

    vs = np.exp(ln_median + inflate * sigma * z)
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
    """Full Toro: optional NHPP thicknesses -> bedrock depth -> AR(1) Vs.

    With ``randomize_layer_thickness=False`` (Toro 2022 Sec. 4 site-specific),
    soil is one layer per ``dz`` sample and Z is truncated per component.
    """
    interface = _sample_interface_depth(config, rng)
    soil_layers = _build_soil_layers_nhpp(config, interface, rng)
    bed_vs = float(config.vs_bedrock)
    layers = _append_bedrock_layer(soil_layers, interface, bed_vs, config)
    layer_vs = _toro_draw_layer_vs(
        layers,
        config,
        rng,
        randomize_bedrock=config.vary_bedrock_vs,
        reject_profile=bool(config.randomize_layer_thickness),
    )
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
