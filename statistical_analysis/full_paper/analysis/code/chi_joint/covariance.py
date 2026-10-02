"""Parsimonious joint correlation layer for standardized NGBoost residuals.

For one design cell the standardized residual profile z ∈ R^p (p = 101 nodes)
is modelled as z ~ N_p(0, R) with

    R = w_b 11ᵀ + w_s K + w_n I,     w = softmax(a_b, a_s, 0),

so diag(R) = 1 and NGBoost's pointwise σ is preserved. Models are nested:

    indep ⊂ shared ⊂ matern32 ⊂ wm ⊂ coswm

- ``shared``   : w_s = 0 (whole-profile offset + nugget)
- ``matern32`` : K = Whittle–Matérn with ν = 3/2, scale s
- ``wm``       : K = Whittle–Matérn with free ν
- ``coswm``    : K = Whittle–Matérn(ν, s) · cos(h ω/100); ω = 100/b in rad per
  100 m, so ω = 0 (no hole effect) is an interior-bounded point, not b → ∞.

Kernels reuse ``rho_wm`` / ``rho_coswm`` from ``chi_spatial/spatial_acf.py``.
Likelihoods work on sufficient statistics S = Σ z zᵀ, n = #profiles.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve, toeplitz
from scipy.optimize import minimize

_CODE = Path(__file__).resolve().parent.parent
for _p in (_CODE, _CODE / "chi_spatial"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _shared import DX_M, N_NODES  # noqa: E402
from spatial_acf import rho_coswm, rho_wm  # noqa: E402

JITTER = 1e-6
LOG2PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class ParamSpec:
    name: str
    lo: float
    hi: float
    init: float


_A_B = ParamSpec("a_b", -12.0, 12.0, 0.0)
_A_S = ParamSpec("a_s", -12.0, 12.0, 1.0)
_LOG_S = ParamSpec("log_s", float(np.log(1.0)), float(np.log(5000.0)), float(np.log(50.0)))
_LOG_NU = ParamSpec("log_nu", float(np.log(0.05)), float(np.log(5.0)), float(np.log(0.5)))
_OMEGA = ParamSpec("omega100", 0.0, 20.0, 1.0)  # b >= 5 m

MODEL_SPECS: dict[str, tuple[ParamSpec, ...]] = {
    "indep": (),
    "shared": (_A_B,),
    "matern32": (_A_B, _A_S, _LOG_S),
    "wm": (_A_B, _A_S, _LOG_S, _LOG_NU),
    "coswm": (_A_B, _A_S, _LOG_S, _LOG_NU, _OMEGA),
}
KERNEL_MODELS = ("matern32", "wm", "coswm")
# Extra multi-starts (only the listed parameters are overridden).
_EXTRA_STARTS: dict[str, list[dict[str, float]]] = {
    "matern32": [{"log_s": float(np.log(10.0))}, {"log_s": float(np.log(300.0))}],
    "wm": [{"log_nu": float(np.log(1.5))}, {"log_s": float(np.log(300.0))}],
    "coswm": [
        {"omega100": 0.1},
        {"omega100": 3.0},
        {"omega100": 8.0, "log_s": float(np.log(300.0))},
    ],
}


def lag_vector_m(n_nodes: int = N_NODES, dx: float = DX_M) -> np.ndarray:
    return dx * np.arange(n_nodes, dtype=float)


def param_names(model: str) -> list[str]:
    return [p.name for p in MODEL_SPECS[model]]


def bounds(model: str) -> list[tuple[float, float]]:
    return [(p.lo, p.hi) for p in MODEL_SPECS[model]]


def default_init(model: str) -> np.ndarray:
    return np.array([p.init for p in MODEL_SPECS[model]], dtype=float)


def unpack(model: str, theta: np.ndarray) -> dict[str, float]:
    return dict(zip(param_names(model), np.asarray(theta, dtype=float), strict=True))


def weights(model: str, theta: np.ndarray) -> tuple[float, float, float]:
    """(w_b, w_s, w_n) — softmax with the nugget logit fixed at 0."""
    if model == "indep":
        return 0.0, 0.0, 1.0
    p = unpack(model, theta)
    a_b = p["a_b"]
    a_s = p.get("a_s", -np.inf)
    logits = np.array([a_b, a_s, 0.0])
    logits = logits - np.max(logits)
    e = np.exp(logits)
    e = e / e.sum()
    return float(e[0]), float(e[1]), float(e[2])


def kernel_acf(model: str, theta: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Spatial kernel K(h) (ρ(0) = 1); zeros for models without a kernel."""
    h = np.asarray(h, dtype=float)
    if model not in KERNEL_MODELS:
        return np.where(h == 0, 1.0, 0.0)
    p = unpack(model, theta)
    s = float(np.exp(p["log_s"]))
    nu = 1.5 if model == "matern32" else float(np.exp(p["log_nu"]))
    if model == "coswm":
        omega = float(p["omega100"])
        b = 100.0 / omega if omega > 0 else np.inf
        return rho_coswm(h, 0.0, nu, s, b)
    return rho_wm(h, nu, s)


def corr_acf(model: str, theta: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Full correlation ρ_R(h) = w_b + w_s K(h) + w_n 1[h = 0]."""
    h = np.asarray(h, dtype=float)
    w_b, w_s, w_n = weights(model, theta)
    return w_b + w_s * kernel_acf(model, theta, h) + w_n * (h == 0)


def build_R(model: str, theta: np.ndarray, n_nodes: int = N_NODES, dx: float = DX_M) -> np.ndarray:
    return toeplitz(corr_acf(model, theta, lag_vector_m(n_nodes, dx)))


def _chol(R: np.ndarray):
    return cho_factor(R + JITTER * np.eye(R.shape[0]), lower=True, check_finite=False)


def loglik_suffstat(R: np.ndarray, S: np.ndarray, n: float) -> float:
    """Σ log N(z | 0, R) over n profiles with S = Σ z zᵀ."""
    p = R.shape[0]
    try:
        c = _chol(R)
    except np.linalg.LinAlgError:
        return -np.inf
    logdet = 2.0 * float(np.sum(np.log(np.diag(c[0]))))
    quad = float(np.trace(cho_solve(c, S, check_finite=False)))
    return -0.5 * (n * p * LOG2PI + n * logdet + quad)


def loglik_profiles(R: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Per-profile log N(z | 0, R); Z shape (n_profiles, p)."""
    Z = np.atleast_2d(Z)
    p = R.shape[0]
    c = _chol(R)
    logdet = 2.0 * float(np.sum(np.log(np.diag(c[0]))))
    sol = cho_solve(c, Z.T, check_finite=False)
    quad = np.sum(Z.T * sol, axis=0)
    return -0.5 * (p * LOG2PI + logdet + quad)


def suffstat(Z: np.ndarray) -> tuple[np.ndarray, float]:
    """S = Σ z zᵀ and n for Z (n_profiles, p) without NaNs."""
    Z = np.asarray(Z, dtype=float)
    return Z.T @ Z, float(Z.shape[0])


def _starts(model: str, extra: list[np.ndarray] | None) -> list[np.ndarray]:
    base = default_init(model)
    out = [base]
    names = param_names(model)
    for over in _EXTRA_STARTS.get(model, []):
        t = base.copy()
        for k, v in over.items():
            t[names.index(k)] = v
        out.append(t)
    if extra:
        out.extend(np.asarray(e, dtype=float) for e in extra)
    lo = np.array([b[0] for b in bounds(model)])
    hi = np.array([b[1] for b in bounds(model)])
    return [np.clip(t, lo, hi) for t in out]


def fit_model(
    model: str,
    S: np.ndarray,
    n: float,
    *,
    prior_mean: np.ndarray | None = None,
    lam: float = 0.0,
    extra_starts: list[np.ndarray] | None = None,
    default_starts: bool = True,
) -> tuple[np.ndarray, float]:
    """MLE (λ = 0) or ridge-MAP toward *prior_mean* for one sufficient stat.

    With ``default_starts=False`` only *prior_mean* and *extra_starts* are
    used as L-BFGS-B starts (warm-started per-cell fits).

    Returns (θ̂, log-likelihood at θ̂ without the penalty).
    """
    if model == "indep":
        return np.zeros(0), loglik_suffstat(np.eye(S.shape[0]), S, n)
    p = S.shape[0]
    scale = n * p  # normalize objective for L-BFGS-B tolerances

    def obj(theta: np.ndarray) -> float:
        ll = loglik_suffstat(build_R(model, theta, p), S, n)
        if not np.isfinite(ll):
            return 1e12
        pen = 0.0
        if lam > 0 and prior_mean is not None:
            pen = 0.5 * lam * float(np.sum((theta - prior_mean) ** 2))
        return (-ll + pen) / scale

    if default_starts:
        starts = _starts(model, extra_starts)
    else:
        starts = [np.asarray(e, dtype=float) for e in (extra_starts or [])]
    if prior_mean is not None:
        starts = [np.asarray(prior_mean, dtype=float), *starts]
    best_theta, best_f = None, np.inf
    for t0 in starts:
        res = minimize(obj, t0, method="L-BFGS-B", bounds=bounds(model))
        if res.fun < best_f:
            best_f, best_theta = float(res.fun), np.asarray(res.x, dtype=float)
    ll = loglik_suffstat(build_R(model, best_theta, p), S, n)
    return best_theta, ll


def n_params(model: str) -> int:
    return len(MODEL_SPECS[model])


def h_below(
    model: str, theta: np.ndarray, *, rho: float = 0.05, search_max: float = 2000.0
) -> float:
    """Smallest h > 0 with kernel K(h) ≤ rho (kernel part only)."""
    if model not in KERNEL_MODELS:
        return float("nan")
    hs = np.linspace(0.0, search_max, 8001)
    k = kernel_acf(model, theta, hs)
    idx = np.flatnonzero((hs > 0) & (k <= rho))
    return float(hs[idx[0]]) if idx.size else float(search_max)


# ---------------------------------------------------------------------------
# Profile diagnostics (vectorized, no-NaN versions of chi_spatial estimators)
# ---------------------------------------------------------------------------
def simulate(R: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw n profiles z ~ N(0, R); shape (n, p)."""
    L = _chol(R)[0]
    L = np.tril(L)
    return rng.standard_normal((n, R.shape[0])) @ L.T


def demeaned_acf(Z: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Pooled within-profile ACF of demeaned profiles (= spatial_acf.empirical_acf).

    Z shape (..., n_profiles, p); returns (..., len(lags)).
    """
    Z = np.asarray(Z, dtype=float)
    Y = Z - Z.mean(axis=-1, keepdims=True)
    p = Y.shape[-1]
    var = np.mean(Y**2, axis=(-1, -2))
    out = np.empty((*Y.shape[:-2], len(lags)))
    for i, k in enumerate(lags):
        cov = np.mean(Y[..., :-k] * Y[..., k:], axis=(-1, -2)) if k < p else np.nan
        out[..., i] = cov / var
    return out


def across_seed_corr(Z: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Mean between-seed Pearson ρ_ik at node lag k (= spatial_coherence estimator).

    Z shape (..., n_profiles, p); returns (..., len(lags)).
    """
    Z = np.asarray(Z, dtype=float)
    Zc = Z - Z.mean(axis=-2, keepdims=True)
    sd = np.sqrt(np.mean(Zc**2, axis=-2))
    out = np.empty((*Z.shape[:-2], len(lags)))
    for i, k in enumerate(lags):
        c = np.mean(Zc[..., :, :-k] * Zc[..., :, k:], axis=-2)
        out[..., i] = np.mean(c / (sd[..., :-k] * sd[..., k:]), axis=-1)
    return out


def profile_stats(Z: np.ndarray, lags: np.ndarray) -> dict[str, np.ndarray]:
    """Scalar profile summaries on the standardized scale.

    Returns arrays over leading dims (..., ) for:
    var_mean (variance across profiles of each profile's spatial mean),
    within_var (mean within-profile variance),
    incr_var_<k> (mean squared increment at node lag k).
    """
    Z = np.asarray(Z, dtype=float)
    m = Z.mean(axis=-1)
    out = {
        "var_profile_mean": np.var(m, axis=-1),
        "within_profile_var": np.mean(np.var(Z, axis=-1), axis=-1),
    }
    for k in lags:
        d = Z[..., k:] - Z[..., :-k]
        out[f"incr_var_{int(k * DX_M)}m"] = np.mean(d**2, axis=(-1, -2))
    return out
