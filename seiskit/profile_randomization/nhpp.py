from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "seiskit.profile_randomization"

import numpy as np

from .common import _total_column_depth
from .models import ProfileRandomizationConfig, _GeoLayer


def _nhpp_depth_from_cumulative(cum_rate: float, c1: float, c2: float, c3: float) -> float:
    return float(
        np.power((c2 * cum_rate) / c3 + cum_rate / c3 + np.power(c1, c2 + 1), 1.0 / (c2 + 1.0)) - c1
    )


def generate_nhpp_layer_thicknesses(
    depth_total: float,
    rng: np.random.Generator,
    *,
    c1: float = 10.86,
    c2: float = -0.89,
    c3: float = 1.98,
    min_thickness: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Non-homogeneous Poisson layer thicknesses (Toro 1995 thickness model)."""
    if depth_total <= 0:
        return np.array([]), np.array([]), np.array([])

    cum_rate = 0.0
    depth_prev = 0.0
    thicknesses: list[float] = []
    depth_mids: list[float] = []
    depth_bottoms: list[float] = []

    while depth_prev < depth_total - 1e-9:
        cum_rate += float(rng.exponential(1.0))
        depth = _nhpp_depth_from_cumulative(cum_rate, c1, c2, c3)
        thick = depth - depth_prev
        if thick < min_thickness and thicknesses:
            continue
        if depth > depth_total:
            thick = depth_total - depth_prev
            depth = depth_total
        if thick <= 1e-9:
            break
        depth_mid = (depth_prev + depth) / 2.0
        thicknesses.append(thick)
        depth_mids.append(depth_mid)
        depth_bottoms.append(depth)
        depth_prev = depth
        if depth >= depth_total - 1e-9:
            break

    if not thicknesses:
        thicknesses = [depth_total]
        depth_mids = [depth_total / 2.0]
        depth_bottoms = [depth_total]

    return (
        np.asarray(thicknesses, dtype=float),
        np.asarray(depth_mids, dtype=float),
        np.asarray(depth_bottoms, dtype=float),
    )


def _build_soil_layers_nhpp(
    config: ProfileRandomizationConfig,
    interface_depth: float,
    rng: np.random.Generator,
) -> list[_GeoLayer]:
    if not config.randomize_layer_thickness:
        return [
            _GeoLayer(
                thickness=interface_depth,
                depth_mid=interface_depth / 2.0,
                depth_bottom=interface_depth,
                vs_median=config.vs_mean,
                is_bedrock=False,
            )
        ]

    min_t = max(config.min_layer_thickness, config.dz)
    thick, mids, bots = generate_nhpp_layer_thicknesses(
        interface_depth,
        rng,
        c1=config.nhpp_c1,
        c2=config.nhpp_c2,
        c3=config.nhpp_c3,
        min_thickness=min_t,
    )
    return [
        _GeoLayer(
            thickness=float(t),
            depth_mid=float(m),
            depth_bottom=float(b),
            vs_median=config.vs_mean,
            is_bedrock=False,
        )
        for t, m, b in zip(thick, mids, bots)
    ]


def _sample_interface_depth(config: ProfileRandomizationConfig, rng: np.random.Generator) -> float:
    z_total = _total_column_depth(config)
    min_t = max(config.min_layer_thickness, config.dz)
    if not config.randomize_bedrock_depth:
        return float(config.thickness)
    mu = np.log(max(config.thickness, min_t))
    iface = float(np.exp(mu + config.sigma_ln_interface_depth * rng.standard_normal()))
    return float(np.clip(iface, min_t, z_total - min_t))


def _sample_bedrock_vs(config: ProfileRandomizationConfig, rng: np.random.Generator) -> float:
    if not config.vary_bedrock_vs:
        return float(config.vs_bedrock)
    ln_med = np.log(max(config.vs_bedrock, 1e-6))
    ln_val = ln_med + config.sigma_ln_bedrock_vs * rng.standard_normal()
    lo = ln_med - config.clip_std * config.sigma_ln_bedrock_vs
    hi = ln_med + config.clip_std * config.sigma_ln_bedrock_vs
    return float(np.exp(np.clip(ln_val, lo, hi)))


def _append_bedrock_layer(
    soil_layers: list[_GeoLayer],
    interface_depth: float,
    bedrock_vs: float,
    config: ProfileRandomizationConfig,
) -> list[_GeoLayer]:
    z_total = _total_column_depth(config)
    bed_thick = max(config.dz, z_total - interface_depth)
    return soil_layers + [
        _GeoLayer(
            thickness=bed_thick,
            depth_mid=interface_depth + bed_thick / 2.0,
            depth_bottom=z_total,
            vs_median=bedrock_vs,
            is_bedrock=True,
        )
    ]


if __name__ == "__main__":
    config = ProfileRandomizationConfig(
        vs_mean=230.0,
        thickness=15.0,
        bedrock_thickness=10.0,
        dz=0.5,
        vs_bedrock=1500.0,
    )
    rng = np.random.default_rng(0)
    interface = _sample_interface_depth(config, rng)
    thick, mids, bots = generate_nhpp_layer_thicknesses(interface, rng, min_thickness=1.0)
    soil_layers = _build_soil_layers_nhpp(config, interface, rng)
    print("Interface depth (m):", round(interface, 2))
    print("NHPP layer count:", len(thick))
    print("NHPP thicknesses (m):", np.round(thick, 2))
    print("Soil geo-layer count:", len(soil_layers))
    print("Bedrock Vs sample:", _sample_bedrock_vs(config, rng))
