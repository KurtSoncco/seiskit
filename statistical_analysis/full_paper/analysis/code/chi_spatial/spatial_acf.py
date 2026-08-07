"""Empirical lag-ACF and Exp / Gaussian / CosWM fits for χ ratios.

Per design cell × metric on ln(χ) along the 101-node array (dx = 2 m):

1. Empirical Pearson autocorrelation vs lag
2. WLS fits of Exponential, Gaussian, and Cosine Whittle–Matérn (CosWM) ACFs
3. AIC / R² / h95 (ρ ≤ 0.05), with half-aperture censoring

Writes CSV + markdown only (no plots) under
``figure_dir("chi_spatial", "spatial_acf")``.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import gamma, kv

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import BOX_ROOT, figure_dir

DATA_PATH = BOX_ROOT / "peak_analysis" / "join_master.h5"
BOX_SEMIVAR = (
    BOX_ROOT / "stats_IM" / "spatial_semivar" / "tables" / "semivariogram_master_metrics.csv"
)
METRICS = ("f_ratio", "abs_TF_ratio", "PGA_ratio", "PSA_ratio", "Ia_ratio")
FACTORS = ["Vs1", "Height", "CoV", "rH", "aHV"]
N_NODES = 101
N_SEEDS = 100
N_CELLS = 243
DX_M = 2.0
H_MAX_M = 200.0
H_FIT_MAX_M = 100.0
N_LAGS = 100  # 2, 4, ..., 200 m
RHO_H95 = 0.05
H95_SEARCH_MAX_M = 2000.0


def load_ratios(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load joined ratio table; rename channel → node."""
    cols = [
        "Vs1",
        "Height",
        "CoV",
        "rH",
        "aHV",
        "channel",
        "seed",
        *METRICS,
    ]
    with h5py.File(path, "r") as f:
        g = f["master"]
        df = pd.DataFrame({c: g[c][:] for c in cols})
    return df.rename(columns={"channel": "node"})


def lag_grid_m() -> np.ndarray:
    return DX_M * np.arange(1, N_LAGS + 1, dtype=float)


def empirical_acf(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Pooled empirical ACF of demeaned node series.

    Parameters
    ----------
    Y : (n_nodes, n_seeds) array on working scale (ln χ); NaNs allowed.

    Returns
    -------
    h_m, rho, n_pairs, n_seeds_used
    """
    h_m = lag_grid_m()
    sum_prod = np.zeros(N_LAGS, dtype=float)
    n_pairs = np.zeros(N_LAGS, dtype=float)
    sum_sq = 0.0
    n_obs = 0
    n_seeds_used = 0
    n_nodes = Y.shape[0]

    for j in range(Y.shape[1]):
        y = Y[:, j]
        finite = np.isfinite(y)
        n_fin = int(finite.sum())
        if n_fin < 8:
            continue
        y = y.copy()
        mu = float(np.nanmean(y))
        y = y - mu
        n_seeds_used += 1

        if bool(np.all(finite)):
            sum_sq += float(np.dot(y, y))
            n_obs += n_nodes
            for k in range(1, N_LAGS + 1):
                sum_prod[k - 1] += float(np.dot(y[:-k], y[k:]))
                n_pairs[k - 1] += float(n_nodes - k)
        else:
            y0 = y.copy()
            y0[~finite] = 0.0
            sum_sq += float(np.sum(y0[finite] ** 2))
            n_obs += n_fin
            for k in range(1, N_LAGS + 1):
                both = finite[:-k] & finite[k:]
                if not np.any(both):
                    continue
                sum_prod[k - 1] += float(np.sum(y0[:-k][both] * y0[k:][both]))
                n_pairs[k - 1] += float(both.sum())

    rho = np.full(N_LAGS, np.nan)
    if n_obs > 0 and sum_sq > 0:
        var = sum_sq / n_obs
        with np.errstate(invalid="ignore", divide="ignore"):
            cov = np.where(n_pairs > 0, sum_prod / n_pairs, np.nan)
            rho = cov / var
    return h_m, rho, n_pairs, n_seeds_used


# ---------------------------------------------------------------------------
# Model ACFs
# ---------------------------------------------------------------------------
def rho_exp(h: np.ndarray, c0: float, a: float) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    return (1.0 - c0) * np.exp(-h / np.maximum(a, 1e-9))


def rho_gauss(h: np.ndarray, c0: float, a: float) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    return (1.0 - c0) * np.exp(-((h / np.maximum(a, 1e-9)) ** 2))


def rho_wm(h: np.ndarray, nu: float, s: float) -> np.ndarray:
    """Whittle–Matérn correlation; ρ(0) = 1."""
    h = np.asarray(h, dtype=float)
    out = np.ones_like(h, dtype=float)
    s = max(float(s), 1e-9)
    nu = float(nu)
    mask = h > 0
    if not np.any(mask):
        return out
    r = np.sqrt(8.0 * nu) * h[mask] / s
    # Avoid overflow at tiny r
    r = np.maximum(r, 1e-12)
    coef = (2.0 ** (1.0 - nu)) / gamma(nu)
    out[mask] = coef * (r**nu) * kv(nu, r)
    # Numerical cleanup
    out = np.where(np.isfinite(out), out, 0.0)
    return out


def rho_coswm(h: np.ndarray, c0: float, nu: float, s: float, b: float) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    return (1.0 - c0) * rho_wm(h, nu, s) * np.cos(h / np.maximum(b, 1e-9))


def _wls_r2(y: np.ndarray, yhat: np.ndarray, w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    resid = y - yhat
    ss_res = float(np.sum(w * resid**2))
    ybar = float(np.sum(w * y) / np.sum(w))
    ss_tot = float(np.sum(w * (y - ybar) ** 2))
    if ss_tot <= 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def _aic_wls(y: np.ndarray, yhat: np.ndarray, w: np.ndarray, n_params: int) -> float:
    """AIC from weighted residual sum of squares (Gaussian WLS)."""
    n = int(y.size)
    if n <= n_params + 1:
        return np.nan
    w = np.asarray(w, dtype=float)
    rss = float(np.sum(w * (y - yhat) ** 2))
    # Effective: treat as sigma^2 = rss/n under unit-rescaled weights
    if rss <= 0:
        return -np.inf
    return float(n * np.log(rss / n) + 2 * n_params)


def _h95(model_fn, params: tuple, search_max: float = H95_SEARCH_MAX_M) -> tuple[float, bool]:
    """Smallest h where model ρ(h) ≤ RHO_H95; censored if > H_FIT_MAX_M."""
    hs = np.linspace(0.0, search_max, 4001)
    try:
        rh = np.asarray(model_fn(hs, *params), dtype=float)
    except Exception:
        return np.nan, True
    rh = np.where(np.isfinite(rh), rh, np.nan)
    below = np.where(rh <= RHO_H95)[0]
    if below.size == 0:
        return search_max, True
    h = float(hs[below[0]])
    return h, bool(h > H_FIT_MAX_M)


def fit_acf_models(
    h_m: np.ndarray,
    rho: np.ndarray,
    n_pairs: np.ndarray,
) -> dict:
    """WLS fit Exp / Gauss / CosWM on 0 < h ≤ 100 m."""
    mask = (
        (h_m > 0) & (h_m <= H_FIT_MAX_M) & np.isfinite(rho) & np.isfinite(n_pairs) & (n_pairs > 0)
    )
    h = h_m[mask]
    y = rho[mask]
    n_p = n_pairs[mask]
    out: dict = {
        "n_bins_fitted": int(mask.sum()),
        "n_bins_total": int(np.sum((h_m > 0) & np.isfinite(rho))),
        "n_bins_censored": int(np.sum((h_m > H_FIT_MAX_M) & np.isfinite(rho))),
    }

    for prefix in ("exp", "gauss", "coswm"):
        if prefix == "coswm":
            out.update(
                {
                    f"c0_{prefix}": np.nan,
                    f"nu_{prefix}": np.nan,
                    f"scale_s_m_{prefix}": np.nan,
                    f"period_b_m_{prefix}": np.nan,
                    f"aic_{prefix}": np.nan,
                    f"r2_wls_{prefix}": np.nan,
                    f"h95_m_{prefix}": np.nan,
                    f"h95_censored_{prefix}": True,
                    f"fit_ok_{prefix}": False,
                }
            )
        else:
            out.update(
                {
                    f"c0_{prefix}": np.nan,
                    f"a_m_{prefix}": np.nan,
                    f"aic_{prefix}": np.nan,
                    f"r2_wls_{prefix}": np.nan,
                    f"h95_m_{prefix}": np.nan,
                    f"h95_censored_{prefix}": True,
                    f"fit_ok_{prefix}": False,
                }
            )

    if h.size < 5:
        out["best_model"] = "none"
        out["best_aic"] = np.nan
        return out

    w = n_p.astype(float)
    sigma = 1.0 / np.sqrt(np.maximum(w, 1.0))

    # Exponential
    try:
        popt, _ = curve_fit(
            rho_exp,
            h,
            y,
            p0=(0.01, 30.0),
            bounds=([0.0, 0.5], [0.49, 5000.0]),
            sigma=sigma,
            absolute_sigma=False,
            maxfev=8000,
        )
        yhat = rho_exp(h, *popt)
        h95, cens = _h95(rho_exp, tuple(popt))
        out.update(
            {
                "c0_exp": float(popt[0]),
                "a_m_exp": float(popt[1]),
                "aic_exp": _aic_wls(y, yhat, w, 2),
                "r2_wls_exp": _wls_r2(y, yhat, w),
                "h95_m_exp": h95,
                "h95_censored_exp": cens,
                "fit_ok_exp": True,
            }
        )
    except Exception:
        pass

    # Gaussian
    try:
        popt, _ = curve_fit(
            rho_gauss,
            h,
            y,
            p0=(0.01, 20.0),
            bounds=([0.0, 0.5], [0.49, 5000.0]),
            sigma=sigma,
            absolute_sigma=False,
            maxfev=8000,
        )
        yhat = rho_gauss(h, *popt)
        h95, cens = _h95(rho_gauss, tuple(popt))
        out.update(
            {
                "c0_gauss": float(popt[0]),
                "a_m_gauss": float(popt[1]),
                "aic_gauss": _aic_wls(y, yhat, w, 2),
                "r2_wls_gauss": _wls_r2(y, yhat, w),
                "h95_m_gauss": h95,
                "h95_censored_gauss": cens,
                "fit_ok_gauss": True,
            }
        )
    except Exception:
        pass

    # CosWM — multiple starts
    coswm_starts = [
        (0.01, 0.5, 50.0, 40.0),
        (0.01, 0.2, 100.0, 80.0),
        (0.05, 1.0, 80.0, 60.0),
        (0.0, 0.5, 30.0, 100.0),
    ]
    best_cos = None
    best_aic = np.inf
    for p0 in coswm_starts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    rho_coswm,
                    h,
                    y,
                    p0=p0,
                    bounds=([0.0, 0.05, 1.0, 1.0], [0.49, 5.0, 5000.0, 5000.0]),
                    sigma=sigma,
                    absolute_sigma=False,
                    maxfev=12000,
                )
            yhat = rho_coswm(h, *popt)
            aic = _aic_wls(y, yhat, w, 4)
            if np.isfinite(aic) and aic < best_aic:
                best_aic = aic
                best_cos = (popt, yhat, aic)
        except Exception:
            continue
    if best_cos is not None:
        popt, yhat, aic = best_cos
        h95, cens = _h95(rho_coswm, tuple(popt))
        out.update(
            {
                "c0_coswm": float(popt[0]),
                "nu_coswm": float(popt[1]),
                "scale_s_m_coswm": float(popt[2]),
                "period_b_m_coswm": float(popt[3]),
                "aic_coswm": float(aic),
                "r2_wls_coswm": _wls_r2(y, yhat, w),
                "h95_m_coswm": h95,
                "h95_censored_coswm": cens,
                "fit_ok_coswm": True,
            }
        )

    aics = {
        "exp": out.get("aic_exp", np.nan),
        "gauss": out.get("aic_gauss", np.nan),
        "coswm": out.get("aic_coswm", np.nan),
    }
    finite = {k: v for k, v in aics.items() if np.isfinite(v)}
    if finite:
        best = min(finite, key=finite.get)
        out["best_model"] = best
        out["best_aic"] = float(finite[best])
    else:
        out["best_model"] = "none"
        out["best_aic"] = np.nan
    return out


def assess_cell_metric(
    df_cell: pd.DataFrame,
    metric: str,
    cell_keys: dict,
) -> tuple[list[dict], dict]:
    """Return (empirical rows, fit-params row) for one cell × metric."""
    pivot = df_cell.pivot(index="node", columns="seed", values=metric)
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    arr = pivot.to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        Y = np.where(np.isfinite(arr) & (arr > 0), np.log(arr), np.nan)

    h_m, rho, n_pairs, n_seeds_used = empirical_acf(Y)
    emp_rows = []
    for i in range(len(h_m)):
        emp_rows.append(
            {
                **cell_keys,
                "metric": metric,
                "h_m": float(h_m[i]),
                "rho": float(rho[i]) if np.isfinite(rho[i]) else np.nan,
                "n_pairs": float(n_pairs[i]),
                "n_seeds_used": int(n_seeds_used),
                "dx_m": DX_M,
            }
        )

    fit = fit_acf_models(h_m, rho, n_pairs)
    rho2 = float(rho[0]) if np.isfinite(rho[0]) else np.nan  # lag 2 m
    fit_row = {
        **cell_keys,
        "metric": metric,
        "n_nodes": int(pivot.shape[0]),
        "n_seeds": int(pivot.shape[1]),
        "n_seeds_used": int(n_seeds_used),
        "dx_m": DX_M,
        "h_max_empirical_m": H_MAX_M,
        "h_fit_max_m": H_FIT_MAX_M,
        "rho_lag2_m": rho2,
        "n_pairs_total": float(np.nansum(n_pairs)),
        **fit,
    }
    return emp_rows, fit_row


def build_summary_md(emp: pd.DataFrame, fits: pd.DataFrame) -> str:
    lines = [
        "# Spatial ACF summary",
        "",
        "Empirical lag correlations and Exponential / Gaussian / CosWM ACF fits "
        f"for 1D-normalized ratios (`{'`, `'.join(METRICS)}`) from `join_master.h5`.",
        "",
        f"- Nodes: {N_NODES} at spacing **{DX_M} m** (aperture **{H_MAX_M:.0f} m**)",
        "- Working scale: **ln(χ)**; empirical ρ̂(h) pooled over seeds within each design cell",
        f"- Model fits: WLS on **0 < h ≤ {H_FIT_MAX_M:.0f} m** (half-aperture); "
        f"h₉₅ = smallest h with fitted ρ(h) ≤ {RHO_H95}",
        "",
        "## Short-lag correlation (empirical)",
        "",
        "| Metric | median ρ̂(2 m) | median ρ̂(10 m) | median ρ̂(50 m) | median ρ̂(100 m) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric in METRICS:
        m = emp[emp["metric"] == metric]

        def med_at(h: float) -> float:
            s = m.loc[np.isclose(m["h_m"], h), "rho"]
            return float(s.median()) if len(s) else float("nan")

        lines.append(
            f"| {metric} | {med_at(2):.3f} | {med_at(10):.3f} | "
            f"{med_at(50):.3f} | {med_at(100):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Best ACF model by AIC (cell counts)",
            "",
            "| Metric | exp | gauss | coswm | none |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for metric in METRICS:
        m = fits[fits["metric"] == metric]
        vc = m["best_model"].value_counts()
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                metric,
                int(vc.get("exp", 0)),
                int(vc.get("gauss", 0)),
                int(vc.get("coswm", 0)),
                int(vc.get("none", 0)),
            )
        )

    lines.extend(
        [
            "",
            "## Median fit diagnostics by metric",
            "",
            "| Metric | med R² CosWM | med R² Gauss | med R² Exp | "
            "med h₉₅ CosWM (m) | frac CosWM h₉₅ censored |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for metric in METRICS:
        m = fits[fits["metric"] == metric]
        lines.append(
            "| {} | {:.3f} | {:.3f} | {:.3f} | {:.1f} | {:.3f} |".format(
                metric,
                float(m["r2_wls_coswm"].median()),
                float(m["r2_wls_gauss"].median()),
                float(m["r2_wls_exp"].median()),
                float(m["h95_m_coswm"].median()),
                float(m["h95_censored_coswm"].mean()),
            )
        )

    # Box comparison if available
    lines.extend(["", "## Comparison to Box semivariogram fits (γ framing)", ""])
    if BOX_SEMIVAR.is_file():
        box = pd.read_csv(BOX_SEMIVAR)
        lines.append(
            "Box `semivariogram_master_metrics.csv` fitted Exp/Gauss/CosWM to "
            "**semivariograms** (not ACFs). Best-model counts by min AIC:"
        )
        lines.append("")
        lines.append("| Metric | Box exp | Box gauss | Box coswm | This ACF coswm |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for metric in METRICS:
            b = box[box["metric"] == metric]
            wins = {"exp": 0, "gauss": 0, "coswm": 0}
            for _, r in b.iterrows():
                aics = {
                    "exp": r["aic_exp"],
                    "gauss": r["aic_gauss"],
                    "coswm": r["aic_coswm"],
                }
                finite = {k: v for k, v in aics.items() if np.isfinite(v)}
                if finite:
                    wins[min(finite, key=finite.get)] += 1
            ours = int((fits[fits["metric"] == metric]["best_model"] == "coswm").sum())
            lines.append(
                f"| {metric} | {wins['exp']} | {wins['gauss']} | {wins['coswm']} | {ours} |"
            )
    else:
        lines.append("Box semivariogram table not found; skipped.")

    med_rho2 = float(emp.loc[np.isclose(emp["h_m"], 2.0), "rho"].median())
    frac_coswm = float((fits["best_model"] == "coswm").mean())

    lines.extend(
        [
            "",
            "## Conclusions",
            "",
            f"1. **Spatial correlation is extremely strong.** Median empirical "
            f"ρ̂(2 m) across cells/metrics is **{med_rho2:.3f}**. Short-lag "
            f"dependence remains high at tens of metres (see table above). "
            f"Naive use of nodal samples as independent therefore greatly inflates "
            f"apparent degrees of freedom (consistent with `node_seed_iid`).",
            "",
            f"2. **CosWM usually provides the best descriptive ACF** among the three "
            f"candidates (wins in **{100 * frac_coswm:.1f}%** of cell×metric cases), "
            f"capturing possible non-monotonic / hole-effect structure that Exp/Gauss miss.",
            "",
            "3. **Aperture caveat.** The receiver span is only **200 m**, shorter than "
            "record lengths typically required to identify scales of fluctuation and "
            "hole-effect features from field data. Fitted ACFs here are **descriptive "
            "summaries of the model response**, not calibrated estimates of underlying "
            "soil correlation distances. Their role is qualitative discussion of how "
            "monotonic vs hole-effect ACF shapes influence spatial patterns in χ, "
            "rather than inference of site-specific correlation lengths. Values of h₉₅ "
            f"beyond the **{H_FIT_MAX_M:.0f} m** fit window are flagged as censored.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    print(f"Loading {DATA_PATH} …")
    df = load_ratios()
    print(f"Loaded {len(df):,} rows")

    df["cell"] = df.groupby(FACTORS, sort=False).ngroup()
    n_cells = int(df["cell"].nunique())
    print(f"Design cells: {n_cells} (expected {N_CELLS})")

    _CODE = Path(__file__).resolve().parent.parent
    if str(_CODE) not in sys.path:
        sys.path.insert(0, str(_CODE))
    from _shared import parallel_map  # noqa: E402

    grouped = list(df.groupby("cell", sort=True))

    def _one(item):
        _, g = item
        cell_keys = {f: g[f].iloc[0] for f in FACTORS}
        cell_keys["cell"] = int(g["cell"].iloc[0])
        emp_rows: list[dict] = []
        fit_rows: list[dict] = []
        for metric in METRICS:
            e, frow = assess_cell_metric(g, metric, cell_keys)
            emp_rows.extend(e)
            fit_rows.append(frow)
        return emp_rows, fit_rows

    results = parallel_map(_one, grouped, desc="spatial_acf cells")
    emp_records: list[dict] = []
    fit_records: list[dict] = []
    for emp_rows, fit_rows in results:
        emp_records.extend(emp_rows)
        fit_records.extend(fit_rows)

    emp = pd.DataFrame(emp_records)
    fits = pd.DataFrame(fit_records)

    out_dir = figure_dir("chi_spatial", "spatial_acf")
    emp_path = out_dir / "acf_empirical.csv"
    fit_path = out_dir / "acf_fit_params.csv"
    md_path = out_dir / "spatial_acf_summary.md"

    emp.to_csv(emp_path, index=False)
    fits.to_csv(fit_path, index=False)
    summary = build_summary_md(emp, fits)
    md_path.write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print(f"Wrote {emp_path}")
    print(f"Wrote {fit_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
