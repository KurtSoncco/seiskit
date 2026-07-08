from __future__ import annotations

from typing import Optional

import numpy as np

from .models import ProfileRandomizationConfig, RandomizedProfile
from .passeri import generate_passeri_profile
from .simplified import generate_tts_simplified, generate_vs_simplified
from .toro import generate_toro_profile


def generate_vs_randomized_profile(
    config: ProfileRandomizationConfig,
    rng: np.random.Generator,
    *,
    vs_base: np.ndarray | None = None,
    rho: Optional[float] = None,
) -> np.ndarray:
    """Toro Vs randomization (full or simplified). Returns Vs on the ``dz`` grid."""
    if config.use_full_model and vs_base is None and rho is None:
        return generate_toro_profile(config, rng).vs_depth
    return generate_vs_simplified(config, rng, vs_base=vs_base, rho=rho).vs_depth


def generate_tts_randomized_profile(
    config: ProfileRandomizationConfig,
    rng: np.random.Generator,
    *,
    vs_base: np.ndarray | None = None,
    rho: Optional[float] = None,
) -> np.ndarray:
    """Passeri tts randomization (full or simplified). Returns Vs on the ``dz`` grid."""
    if config.use_full_model and vs_base is None and rho is None:
        return generate_passeri_profile(config, rng).vs_depth
    return generate_tts_simplified(config, rng, vs_base=vs_base, rho=rho).vs_depth


def generate_vs_randomized_profile_full(
    config: ProfileRandomizationConfig,
    rng: np.random.Generator,
) -> RandomizedProfile:
    return generate_toro_profile(config, rng)


def generate_tts_randomized_profile_full(
    config: ProfileRandomizationConfig,
    rng: np.random.Generator,
) -> RandomizedProfile:
    return generate_passeri_profile(config, rng)
