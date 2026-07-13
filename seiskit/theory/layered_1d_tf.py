"""Closed-form 1D linear viscoelastic site-response transfer functions.

Implements the multilayer SH (vertically propagating shear-wave) solution via
layer transfer matrices (Thomson–Haskell / Kramer Ch. 7). Layers are ordered
from the free surface downward to the elastic halfspace bedrock.

Definitions
-----------
- ``AF_within(f)``  = |u_surface / u_within| at the soil–bedrock interface
- ``AF_outcrop(f)`` = |u_surface / u_rock_outcrop|

where rock outcrop is the free-surface motion of the bedrock halfspace alone
(``u_outcrop = 2 u_incident``).

Complex wave speed uses the hysteretic model ``Vs* = Vs * sqrt(1 + 2 i ξ)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class Layer:
    """One soil layer above bedrock (surface → deeper)."""

    H: float  # thickness [m]
    Vs: float  # shear-wave velocity [m/s]
    rho: float  # mass density [kg/m³]
    xi: float = 0.0  # hysteretic damping ratio [-]


@dataclass(frozen=True)
class RockHalfspace:
    """Elastic (viscoelastic) bedrock halfspace beneath the soil stack."""

    Vs: float
    rho: float
    xi: float = 0.0


def _complex_vs(Vs: float, xi: float) -> complex:
    """Complex shear-wave velocity for constant hysteretic damping."""
    return Vs * np.sqrt(1.0 + 2.0j * xi)


def amplification_single_layer_elastic(
    freq: np.ndarray | float,
    H: float,
    Vs: float,
    rho: float,
    Vs_rock: float,
    rho_rock: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Undamped single-layer closed form (Kramer).

    Returns
    -------
    af_within, af_outcrop
        Magnitudes for the same frequency array/scalar as ``freq``.
    """
    f = np.asarray(freq, dtype=float)
    omega = 2.0 * np.pi * f
    alpha = (rho * Vs) / (rho_rock * Vs_rock)

    # Static limit AF_outcrop = AF_within = 1
    af_within = np.ones_like(f, dtype=float)
    af_outcrop = np.ones_like(f, dtype=float)
    mask = omega > 0.0
    kH = omega[mask] * H / Vs
    # AF_outcrop = 1 / (cos(kH) + i α sin(kH))
    denom = np.cos(kH) + 1j * alpha * np.sin(kH)
    af_outcrop[mask] = np.abs(1.0 / denom)
    # AF_within = 1 / cos(kH) for τ_surface = 0 (independent of α)
    af_within[mask] = np.abs(1.0 / np.cos(kH))
    return af_within, af_outcrop


def _propagate_surface_to_base(
    omega: float,
    layers: Sequence[Layer],
) -> tuple[complex, complex]:
    """Propagate (u=1, τ=0) at the free surface down to the soil–rock interface.

    Returns complex (u_base, tau_base) for unit surface displacement.

    Transfer matrix (z downward, e^{iωt} convention)::

        u_b   =  u_t cos(kH) + τ_t sin(kH) / (i ω ρ Vs*)
        τ_b   = -u_t (i ω ρ Vs*) sin(kH) + τ_t cos(kH)

    The minus sign on the τ←u coupling is required so two stacked half-layers
    reproduce a single full layer (cos(2θ) = cos²θ − sin²θ).
    """
    u: complex = 1.0 + 0.0j
    tau: complex = 0.0 + 0.0j
    if omega == 0.0:
        return u, tau

    for layer in layers:
        Vs_c = _complex_vs(layer.Vs, layer.xi)
        k = omega / Vs_c
        kh = k * layer.H
        c = np.cos(kh)
        s = np.sin(kh)
        gk = 1j * omega * layer.rho * Vs_c
        u_new = u * c + tau * (s / gk)
        tau_new = -u * gk * s + tau * c
        u, tau = u_new, tau_new
    return u, tau


def layered_transfer_function(
    freq: np.ndarray | float,
    layers: Sequence[Layer],
    rock: RockHalfspace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Multilayer viscoelastic transfer-function magnitudes.

    Parameters
    ----------
    freq
        Frequency [Hz] (scalar or array). Values <= 0 use the static limit.
    layers
        Soil layers from free surface downward (not including bedrock).
    rock
        Bedrock halfspace beneath the deepest soil layer.

    Returns
    -------
    freq_out, af_within, af_outcrop
        Frequency array and amplification magnitudes.
    """
    if len(layers) == 0:
        raise ValueError("At least one soil layer is required")

    f = np.atleast_1d(np.asarray(freq, dtype=float))
    af_within = np.empty(f.shape, dtype=float)
    af_outcrop = np.empty(f.shape, dtype=float)

    Vs_r = _complex_vs(rock.Vs, rock.xi)

    for i, fi in enumerate(f):
        omega = 2.0 * np.pi * float(fi)
        if omega == 0.0:
            af_within[i] = 1.0
            af_outcrop[i] = 1.0
            continue

        u_base, tau_base = _propagate_surface_to_base(omega, layers)

        af_within[i] = float(np.abs(1.0 / u_base))

        # Rock outcrop: with τ_base = −i ω ρ Vs sin(kH) for a single undamped
        # layer, u_outcrop = u_base + τ_base / (ω ρ_r Vs_r*) = cos − i α sin,
        # recovering Kramer |1/(cos + i α sin)| = |1/(cos − i α sin)|.
        Z_r = omega * rock.rho * Vs_r
        u_outcrop = u_base + tau_base / Z_r
        af_outcrop[i] = float(np.abs(1.0 / u_outcrop))

    return f, af_within, af_outcrop


def resonance_frequencies_quarter_wave(
    Vs: float,
    H: float,
    n_modes: int = 5,
) -> np.ndarray:
    """Rigid-bedrock quarter-wave resonances f_n = (2n-1) Vs / (4 H)."""
    n = np.arange(1, n_modes + 1)
    return (2 * n - 1) * Vs / (4.0 * H)
