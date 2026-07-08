"""Goodness-of-fit metrics for seismic response comparison (Anderson-style)."""

from __future__ import annotations

import numpy as np


def _integrate_abs(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapezoid(np.abs(y), x=x))


def anderson_time_domain(
    reference: np.ndarray,
    candidate: np.ndarray,
    dt: float,
) -> dict[str, float]:
    """Anderson (2004) time-domain GOF for acceleration, velocity, displacement."""
    ref = np.asarray(reference, dtype=float).ravel()
    cand = np.asarray(candidate, dtype=float).ravel()
    n = min(len(ref), len(cand))
    ref, cand = ref[:n], cand[:n]
    t = np.arange(n) * dt

    v_ref = np.cumsum(ref) * dt
    v_cand = np.cumsum(cand) * dt
    d_ref = np.cumsum(v_ref) * dt
    d_cand = np.cumsum(v_cand) * dt

    gof_a = _integrate_abs(t, ref - cand) / max(_integrate_abs(t, ref), 1e-12)
    gof_v = _integrate_abs(t, v_ref - v_cand) / max(_integrate_abs(t, v_ref), 1e-12)
    gof_d = _integrate_abs(t, d_ref - d_cand) / max(_integrate_abs(t, d_ref), 1e-12)
    return {"GOF_A": gof_a, "GOF_V": gof_v, "GOF_D": gof_d}


def anderson_frequency_domain(
    freq: np.ndarray,
    ref_af: np.ndarray,
    cand_af: np.ndarray,
    *,
    f_weight_center: float | None = None,
    f_weight_width: float = 1.0,
) -> float:
    """Weighted L1 norm of ln(AF) residuals."""
    f = np.asarray(freq, dtype=float).ravel()
    r = np.log(np.clip(np.asarray(ref_af, float), 1e-12, None))
    c = np.log(np.clip(np.asarray(cand_af, float), 1e-12, None))
    n = min(len(f), len(r), len(c))
    f, r, c = f[:n], r[:n], c[:n]

    if f_weight_center is not None:
        w = np.exp(-0.5 * ((f - f_weight_center) / max(f_weight_width, 1e-6)) ** 2)
    else:
        w = np.ones_like(f)
    w = w / max(np.sum(w), 1e-12)
    return float(np.sum(w * np.abs(r - c)))


def log_residual_bias(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Mean ln(candidate/reference) for positive arrays."""
    r = np.asarray(reference, dtype=float).ravel()
    c = np.asarray(candidate, dtype=float).ravel()
    mask = (r > 0) & (c > 0)
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.log(c[mask] / r[mask])))
