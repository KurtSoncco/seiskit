"""Seismic intensity measures for site-response comparison."""

from __future__ import annotations

import numpy as np


def pga(acceleration: np.ndarray) -> float:
    """Peak ground acceleration [same units as input]."""
    return float(np.max(np.abs(np.asarray(acceleration, dtype=float))))


def arias_intensity(acceleration: np.ndarray, dt: float) -> float:
    """Arias intensity Ia = (pi / 2g) * integral a^2 dt with g=9.81 m/s^2."""
    a = np.asarray(acceleration, dtype=float)
    g = 9.81
    return float((np.pi / (2.0 * g)) * np.trapezoid(a**2, dx=dt))


def compute_sa(
    acceleration: np.ndarray,
    dt: float,
    periods: np.ndarray,
    damping: float = 0.05,
) -> np.ndarray:
    """5%-damped pseudo-spectral acceleration Sa(T) via frequency-domain SDOF."""
    a = np.asarray(acceleration, dtype=float).ravel()
    periods = np.asarray(periods, dtype=float).ravel()
    n = len(a)
    nfft = int(2 ** np.ceil(np.log2(n)))
    freqs = np.fft.rfftfreq(nfft, dt)
    omega = 2.0 * np.pi * freqs
    a_fft = np.fft.rfft(a, n=nfft) / nfft

    sa = np.empty_like(periods)
    zeta = damping
    for i, T in enumerate(periods):
        wn = 2.0 * np.pi / max(T, 1e-6)
        denom = wn**2 - omega**2 + 2.0j * zeta * wn * omega
        denom[0] = wn**2
        u_fft = -a_fft / denom
        u = np.fft.irfft(u_fft, n=nfft)[:n]
        sa[i] = wn**2 * np.max(np.abs(u))
    return sa


def default_periods(vs1: float, thickness: float) -> np.ndarray:
    """Standard period grid including quarter-wave T0."""
    t0 = 4.0 * thickness / max(vs1, 1.0)
    base = np.array([0.1, 0.2, 0.5, 1.0, 2.0], dtype=float)
    return np.unique(np.clip(np.concatenate([base, [t0, 2.0 * t0]]), 0.05, None))


def sigma_ln(values: np.ndarray) -> float:
    """Standard deviation of natural log of positive values."""
    v = np.asarray(values, dtype=float).ravel()
    v = v[v > 0]
    if v.size < 2:
        return 0.0
    return float(np.std(np.log(v), ddof=1))


def ensemble_sa_stats(
    accelerations: list[np.ndarray],
    dt: float,
    periods: np.ndarray,
    damping: float = 0.05,
) -> dict[str, np.ndarray | float]:
    """Median Sa and sigma_ln(Sa) across realizations."""
    sa_matrix = np.vstack([compute_sa(a, dt, periods, damping=damping) for a in accelerations])
    return {
        "periods": periods,
        "median_sa": np.median(sa_matrix, axis=0),
        "sigma_ln_sa": np.array([sigma_ln(sa_matrix[:, j]) for j in range(sa_matrix.shape[1])]),
        "pga_median": float(np.median([pga(a) for a in accelerations])),
        "pga_sigma_ln": sigma_ln(np.array([pga(a) for a in accelerations])),
        "sa_matrix": sa_matrix,
    }
