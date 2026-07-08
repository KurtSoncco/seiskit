from __future__ import annotations

import sys
from pathlib import Path

# Allow `python seiskit/profile_randomization/common.py` for quick local checks.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "seiskit.profile_randomization"

import numpy as np

from .models import ProfileRandomizationConfig


def build_base_case_profile(config: ProfileRandomizationConfig) -> np.ndarray:
    """Nominal soil + bedrock column on the FE ``dz`` grid."""
    return _resample_layers_to_dz(
        [config.thickness, config.bedrock_thickness],
        [config.vs_mean, config.vs_bedrock],
        config.dz,
        config.thickness + config.bedrock_thickness,
    )


def _total_column_depth(config: ProfileRandomizationConfig) -> float:
    """Total depth of the column."""
    return config.thickness + config.bedrock_thickness


def _nominal_soil_samples(config: ProfileRandomizationConfig) -> int:
    """Number of soil samples in the nominal profile."""
    return max(1, int(round(config.thickness / config.dz)))


def geological_layer_starts(vs_depth: np.ndarray, *, rel_tol: float = 5.0) -> np.ndarray:
    """Indices where a new geological layer begins based on relative Vs jumps.
    Args:
        vs_depth (np.ndarray): The velocity depth profile.
        rel_tol (float): The relative tolerance for Vs jumps.

    Returns:
        np.ndarray: The indices of the layer starts.
    """
    vs = np.asarray(vs_depth, dtype=float).ravel()
    if len(vs) <= 1:
        return np.array([0], dtype=int)
    denom = np.clip(vs[:-1], 1e-6, None)
    contrasts = np.abs(np.diff(vs)) / denom
    starts = np.where(contrasts > rel_tol)[0] + 1
    return np.concatenate(([0], starts)).astype(int)


def _resample_layers_to_dz(
    thicknesses: list[float] | np.ndarray,
    vs_layers: list[float] | np.ndarray,
    dz: float,
    z_total: float,
) -> np.ndarray:
    """Piecewise-constant Vs on ``(0, z_total]`` with ``dz`` spacing."""
    thicknesses = np.asarray(thicknesses, dtype=float)
    vs_layers = np.asarray(vs_layers, dtype=float)
    nz = max(1, int(round(z_total / dz)))
    z_centers = (np.arange(nz) + 0.5) * dz
    edges = np.concatenate([[0.0], np.cumsum(thicknesses)])
    k = np.searchsorted(edges, z_centers, side="right") - 1
    k = np.clip(k, 0, len(vs_layers) - 1)
    return vs_layers[k]


def _soil_sample_index(vs_depth: np.ndarray, interface_depth: float, dz: float) -> int:
    """Soil sample count from resampled profile (jump at interface) or nominal depth."""
    nz = len(vs_depth)
    jumps = np.abs(np.diff(vs_depth)) / np.clip(vs_depth[:-1], 1e-6, None)
    if jumps.size and float(jumps.max()) > 0.1:
        return int(np.clip(int(np.argmax(jumps)) + 1, 1, nz - 1))
    return int(np.clip(int(round(interface_depth / dz)), 1, nz - 1))


def profile_to_opensees_column(
    vs_profile: np.ndarray,
    n_soil_samples: int | None = None,
    *,
    vs_bedrock: float | None = None,
    dz: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a depth profile into (nz, 1) Vs and bedrock mask."""
    col = np.asarray(vs_profile, dtype=float).ravel().reshape(-1, 1)
    if n_soil_samples is None:
        if vs_bedrock is None or dz is None:
            raise ValueError("Provide n_soil_samples or (vs_bedrock, dz) for auto-detect")
        jumps = np.abs(np.diff(col.ravel())) / np.clip(col.ravel()[:-1], 1.0, None)
        if jumps.size and float(jumps.max()) > 0.15:
            n_soil_samples = int(np.argmax(jumps)) + 1
        else:
            n_soil_samples = int(np.sum(col.ravel() < vs_bedrock * 0.9))
        n_soil_samples = int(np.clip(n_soil_samples, 1, len(col) - 1))
    mask = np.zeros_like(col, dtype=bool)
    mask[int(n_soil_samples) :] = True
    return col, mask


def build_layered_profile(
    soil_vs: np.ndarray,
    vs_bedrock: float,
    bedrock_depth: float,
    dz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy soil-only + bedrock stack."""
    n_bed = max(1, int(round(bedrock_depth / dz)))
    soil = np.asarray(soil_vs, dtype=float).ravel()
    vs_col = np.concatenate([soil, np.full(n_bed, vs_bedrock)])
    mask = np.concatenate([np.zeros(len(soil), dtype=bool), np.ones(n_bed, dtype=bool)])
    return vs_col.reshape(-1, 1), mask.reshape(-1, 1)


def profile_cov(vs: np.ndarray) -> float:
    vs = np.asarray(vs, dtype=float).ravel()
    mean = float(np.mean(vs))
    if mean <= 0:
        return 0.0
    return float(np.std(vs) / mean)
