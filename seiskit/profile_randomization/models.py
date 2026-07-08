from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProfileRandomizationConfig:
    """Configuration for 1D profile randomization."""

    # Profile parameters
    vs_mean: float
    thickness: float
    dz: float
    cov: float = 0.2
    vs_bedrock: float = 1500.0
    bedrock_thickness: float = 10.0
    # Randomization parameters
    sigma_ln_vs: float = 0.15
    sigma_ln_tts: float = 0.02
    clip_std: float = 2.0
    # Randomization options
    randomize_layer_thickness: bool = True
    randomize_bedrock_depth: bool = True
    # NHPP parameters
    sigma_ln_interface_depth: float = 0.05
    sigma_ln_bedrock_vs: float = 0.10
    nhpp_c1: float = 10.86
    nhpp_c2: float = -0.89
    nhpp_c3: float = 1.98
    min_layer_thickness: float = 1.0
    toro_rho_0: float = 0.99
    toro_delta: float = 3.9
    toro_rho_200: float = 0.98
    toro_b: float = 0.344
    toro_bedrock_interface_rho: float = 1.0
    vary_bedrock_vs: bool = False
    passeri_bedrock_depth_vs_rho: float = 0.508
    tts_rho_boost: float = 0.25
    # Full model parameters
    use_full_model: bool = True
    layer_jump_rel: float = 0.05
    layer_thickness: float = 0.5

    def __post_init__(self) -> None:
        if self.dz <= 0:
            raise ValueError("Grid spacing `dz` must be strictly positive.")
        if self.thickness <= 0 or self.bedrock_thickness <= 0:
            raise ValueError("Thickness parameters must be strictly positive.")


@dataclass(frozen=True)
class RandomizedProfile:
    """Full column on the FE depth grid."""

    vs_depth: np.ndarray
    n_soil_samples: int
    interface_depth: float


@dataclass(frozen=True)
class _GeoLayer:
    thickness: float
    depth_mid: float
    depth_bottom: float
    vs_median: float
    is_bedrock: bool = False


@dataclass(frozen=True)
class _LayerGrid:
    thickness: np.ndarray
    depth_mid: np.ndarray
    depth_bottom: np.ndarray
    vs_base: np.ndarray
    tts_bottom_base: np.ndarray
    layer_starts: np.ndarray
