"""Fit the pooled / partially pooled joint correlation layer on OOF residuals.

Models (nested): indep ⊂ shared ⊂ matern32 ⊂ wm ⊂ coswm  (see ``covariance.py``).

1. Global MLE of θ per model on all training-seed OOF profiles (243 cells × 75 seeds).
2. Partial pooling: per-cell ridge-MAP  −ℓ_k(θ) + ½λ‖θ − θ_global‖²,
   λ chosen by seed-fold CV (the OOF folds) on held-out profile log-likelihood.
3. Per-cell bootstrap SE of the partially pooled θ_k (resampling seed profiles,
   same resample for all cells; θ_global held fixed).

Writes CSV + ``summary.md`` under ``figure_dir("chi_joint", "fit_joint_cov")``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import covariance as cv  # noqa: E402
from common import (  # noqa: E402
    H_FIT_MAX_M,
    METRIC,
    add_design_columns,
    cell_design_table,
    cov_cell_path,
    cov_global_path,
    load_ratios,
    out_dir,
    residuals_path,
)

MODELS = ("indep", "shared", "matern32", "wm", "coswm")
POOLED_MODELS = ("shared", "matern32", "wm", "coswm")
LAMBDA_GRID = (1e7, 1e6, 1e5, 1e4, 1e3, 1e2, 1e1)  # descending → warm starts
N_BOOT = 100
BOOT_MODELS = ("matern32", "wm", "coswm")  # SE needed for the kernel-parameter NGBoost
BOOT_SEED = 21
N_JOBS = -1


def cell_suffstats(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """z (cells, seeds, nodes) → S (cells, p, p), n (cells,)."""
    S = np.einsum("csi,csj->cij", z, z)
    n = np.full(z.shape[0], float(z.shape[1]))
    return S, n


def fit_global(S_tot: np.ndarray, n_tot: float) -> dict[str, tuple[np.ndarray, float]]:
    """Global MLE per model, nested warm starts (each model starts from the smaller one)."""
    out: dict[str, tuple[np.ndarray, float]] = {}
    for model in MODELS:
        extra = []
        if model == "matern32" and "shared" in out:
            extra.append(np.r_[out["shared"][0], 1.0, np.log(50.0)])
        if model == "wm" and "matern32" in out:
            extra.append(np.r_[out["matern32"][0], np.log(1.5)])
        if model == "coswm" and "wm" in out:
            t = out["wm"][0]
            extra += [np.r_[t, 0.0], np.r_[t, 1.0], np.r_[t, 4.0]]
        out[model] = cv.fit_model(model, S_tot, n_tot, extra_starts=extra)
    return out


def _cell_path(model, S, n, prior, lams):
    """Warm-started ridge-MAP path over descending λ; returns list of θ."""
    thetas = []
    prev = prior
    for lam in lams:
        extra = [prev] if prev is not None else None
        theta, _ = cv.fit_model(
            model, S, n, prior_mean=prior, lam=lam, extra_starts=extra, default_starts=False
        )
        thetas.append(theta)
        prev = theta
    return thetas


def _cv_fold_cell(model, S_fit, n_fit, S_hold, n_hold, prior, lams):
    thetas = _cell_path(model, S_fit, n_fit, prior, lams)
    p = S_fit.shape[0]
    ll = [cv.loglik_suffstat(cv.build_R(model, t, p), S_hold, n_hold) for t in thetas]
    ll_glob = cv.loglik_suffstat(cv.build_R(model, prior, p), S_hold, n_hold)
    return ll, ll_glob


def _loglik_hold(model, theta, S_hold, n_hold):
    p = S_hold.shape[0]
    return cv.loglik_suffstat(cv.build_R(model, theta, p), S_hold, n_hold)


def select_lambda(z: np.ndarray, fold: np.ndarray) -> pd.DataFrame:
    """Seed-fold CV of held-out profile log-likelihood vs λ (and global, λ = ∞)."""
    rows = []
    for f in np.unique(fold):
        hold, fit = fold == f, fold != f
        S_fit, n_fit = cell_suffstats(z[:, fit, :])
        S_hold, n_hold = cell_suffstats(z[:, hold, :])
        glob = fit_global(S_fit.sum(0), float(n_fit.sum()))
        for model in POOLED_MODELS:
            prior = glob[model][0]
            res = Parallel(n_jobs=N_JOBS)(
                delayed(_cv_fold_cell)(
                    model, S_fit[c], n_fit[c], S_hold[c], n_hold[c], prior, LAMBDA_GRID
                )
                for c in range(z.shape[0])
            )
            ll = np.array([r[0] for r in res])  # (cells, λ)
            ll_g = np.array([r[1] for r in res])
            n_prof = float(n_hold.sum())
            for j, lam in enumerate(LAMBDA_GRID):
                rows.append(
                    {
                        "fold": int(f),
                        "model": model,
                        "lambda": lam,
                        "ll_per_profile": ll[:, j].sum() / n_prof,
                    }
                )
            rows.append(
                {
                    "fold": int(f),
                    "model": model,
                    "lambda": np.inf,
                    "ll_per_profile": ll_g.sum() / n_prof,
                }
            )
        # Reference: independent model
        ll_ind = sum(
            _loglik_hold("indep", np.zeros(0), S_hold[c], n_hold[c]) for c in range(z.shape[0])
        )
        rows.append(
            {
                "fold": int(f),
                "model": "indep",
                "lambda": np.inf,
                "ll_per_profile": ll_ind / float(n_hold.sum()),
            }
        )
        print(f"  λ-CV fold {f} done")
    return pd.DataFrame(rows)


def _fit_cell_final(model, S, n, prior, lam):
    lams = [x for x in LAMBDA_GRID if x >= lam] if np.isfinite(lam) else []
    if not lams:
        return prior.copy()
    return _cell_path(model, S, n, prior, lams)[-1]


def _boot_cell(model, z_cell, idx_boot, theta0, prior, lam):
    out = []
    for idx in idx_boot:
        S, n = cell_suffstats(z_cell[None, idx, :])
        theta, _ = cv.fit_model(
            model,
            S[0],
            n[0],
            prior_mean=prior,
            lam=lam,
            extra_starts=[theta0],
            default_starts=False,
        )
        out.append(theta)
    return np.array(out)


def describe_theta(model: str, theta: np.ndarray) -> dict[str, float]:
    """Readable parameters + derived distances."""
    w_b, w_s, w_n = cv.weights(model, theta)
    d = {"w_b": w_b, "w_s": w_s, "w_n": w_n}
    p = cv.unpack(model, theta)
    d.update({f"theta_{k}": v for k, v in p.items()})
    if model in cv.KERNEL_MODELS:
        d["s_m"] = float(np.exp(p["log_s"]))
        d["nu"] = 1.5 if model == "matern32" else float(np.exp(p["log_nu"]))
        om = p.get("omega100", 0.0)
        d["b_m"] = 100.0 / om if om > 0 else np.inf
        h95 = cv.h_below(model, theta)
        d["h95_kernel_m"] = h95
        d["h95_censored"] = bool(h95 > H_FIT_MAX_M)
    return d


def main(metric: str = METRIC) -> None:
    t_start = time.perf_counter()
    out = out_dir("fit_joint_cov")
    cubes = np.load(residuals_path(metric))
    z = cubes["z_train"]
    fold = cubes["fold_of_train_seed"]
    if not np.all(np.isfinite(z)):
        raise ValueError("NaNs in OOF residuals; drop-NaN profiles not implemented for this metric")
    n_cells, n_seeds, p = z.shape
    design = cell_design_table(add_design_columns(load_ratios()))

    # 1. Global fits
    S, n = cell_suffstats(z)
    glob = fit_global(S.sum(0), float(n.sum()))
    g_rows = []
    for model, (theta, ll) in glob.items():
        k = cv.n_params(model)
        N = float(n.sum())
        g_rows.append(
            {
                "model": model,
                "n_params": k,
                "ll_train": ll,
                "ll_per_profile": ll / N,
                "aic": -2 * ll + 2 * k,
                "bic": -2 * ll + k * np.log(N),
                **describe_theta(model, theta),
            }
        )
        print(f"  global {model}: ll/profile={ll / N:.3f}")
    g_tab = pd.DataFrame(g_rows)
    g_tab.to_csv(cov_global_path(metric), index=False)
    np.savez(out / f"theta_global_{metric}.npz", **{m: glob[m][0] for m in MODELS})

    # 2. λ selection
    print("λ selection by seed-fold CV …")
    lam_cv = select_lambda(z, fold)
    lam_cv.to_csv(out / f"lambda_cv_{metric}.csv", index=False)
    lam_mean = lam_cv.groupby(["model", "lambda"], as_index=False)["ll_per_profile"].mean()
    best_lam = {
        m: float(lam_mean[lam_mean["model"] == m].sort_values("ll_per_profile").iloc[-1]["lambda"])
        for m in POOLED_MODELS
    }
    print(f"  best λ: {best_lam}")

    # 3. Final partially pooled per-cell fits + bootstrap SE
    rng = np.random.default_rng(BOOT_SEED)
    idx_boot = [rng.integers(0, n_seeds, n_seeds) for _ in range(N_BOOT)]
    cell_rows = []
    theta_cell: dict[str, np.ndarray] = {}
    theta_se: dict[str, np.ndarray] = {}
    for model in POOLED_MODELS:
        prior, lam = glob[model][0], best_lam[model]
        thetas = Parallel(n_jobs=N_JOBS)(
            delayed(_fit_cell_final)(model, S[c], n[c], prior, lam) for c in range(n_cells)
        )
        thetas = np.array(thetas)
        lam_boot = lam if np.isfinite(lam) else 0.0
        boots = Parallel(n_jobs=N_JOBS)(
            delayed(_boot_cell)(model, z[c], idx_boot, thetas[c], prior, lam_boot)
            if np.isfinite(lam) and model in BOOT_MODELS
            else delayed(np.tile)(thetas[c], (N_BOOT, 1))
            for c in range(n_cells)
        )
        boots = np.array(boots)  # (cells, B, k)
        theta_cell[model] = thetas
        theta_se[model] = boots.std(axis=1, ddof=1)
        names = cv.param_names(model)
        for c in range(n_cells):
            ll = cv.loglik_suffstat(cv.build_R(model, thetas[c], p), S[c], n[c])
            row = {
                "cell": c,
                "model": model,
                "lambda": lam,
                "ll_train": ll,
                **describe_theta(model, thetas[c]),
            }
            row.update({f"se_theta_{k}": float(theta_se[model][c, j]) for j, k in enumerate(names)})
            if model in BOOT_MODELS:
                h95_b = [cv.h_below(model, t) for t in boots[c][:25]]
                row["se_h95_kernel_m"] = float(np.std(h95_b, ddof=1))
            cell_rows.append(row)
        print(f"  per-cell {model} done (λ={lam:g})")

    cell_tab = pd.DataFrame(cell_rows).merge(design, on="cell", how="left")
    cell_tab.to_csv(cov_cell_path(metric), index=False)
    np.savez(
        out / f"theta_cell_{metric}.npz",
        **{f"{m}_theta": theta_cell[m] for m in POOLED_MODELS},
        **{f"{m}_se": theta_se[m] for m in POOLED_MODELS},
    )
    meta = {
        "metric": metric,
        "lambda_grid": list(LAMBDA_GRID),
        "best_lambda": best_lam,
        "n_boot": N_BOOT,
        "n_cells": n_cells,
        "n_train_seeds": n_seeds,
        "runtime_s": time.perf_counter() - t_start,
    }
    (out / f"meta_{metric}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Summary
    lam_tab = lam_mean.pivot(index="model", columns="lambda", values="ll_per_profile")
    kern = cell_tab[cell_tab["model"].isin(cv.KERNEL_MODELS)]
    q = (
        kern.groupby("model")[["w_b", "w_s", "w_n", "s_m", "nu", "h95_kernel_m"]]
        .quantile([0.1, 0.5, 0.9])
        .unstack()
    )
    cos = cell_tab[cell_tab["model"] == "coswm"]
    show_cols = [
        "model",
        "n_params",
        "ll_per_profile",
        "aic",
        "w_b",
        "w_s",
        "w_n",
        "s_m",
        "nu",
        "b_m",
        "h95_kernel_m",
    ]
    lines = [
        f"# Joint correlation layer fits ({metric})",
        "",
        "## Definitions",
        "",
        r"- Profile model: \(\mathbf z_{kj}\sim\mathcal N_{101}(\mathbf 0,R_k)\), \(R_k=w_b\mathbf 1\mathbf 1^\top+w_sK+w_nI\) on OOF-standardized NGBoost residuals.",
        r"- Kernels: Whittle–Matérn (ν = 3/2 fixed for `matern32`; free ν for `wm`) and CosWM \(=\) WM\((\nu,s)\cdot\cos(h/b)\). Nested: indep ⊂ shared ⊂ matern32 ⊂ wm ⊂ coswm.",
        r"- Partial pooling: per-cell ridge-MAP toward the global θ; λ chosen by seed-fold CV of held-out profile log-likelihood (λ = ∞ ≡ global).",
        f"- `h95_kernel_m`: smallest h with K(h) ≤ 0.05 (kernel part only); censored beyond {H_FIT_MAX_M:.0f} m.",
        f"- SE: {N_BOOT} seed-profile bootstrap resamples, θ_global fixed.",
        "",
        "## Global fits (training OOF profiles)",
        "",
        g_tab[[c for c in show_cols if c in g_tab.columns]].to_markdown(
            index=False, floatfmt=".4g"
        ),
        "",
        "## λ cross-validation (mean held-out log-lik per profile)",
        "",
        lam_tab.to_markdown(floatfmt=".3f"),
        "",
        f"Chosen λ: {json.dumps(best_lam)}",
        "",
        "## Per-cell partially pooled parameters (10/50/90% quantiles over cells)",
        "",
        q.T.to_markdown(floatfmt=".3g"),
        "",
        f"- CosWM cells with ω = 0 (no hole effect): {int((cos['theta_omega100'] <= 1e-6).sum())} / {len(cos)}",
        f"- CosWM h95 censored (> {H_FIT_MAX_M:.0f} m): {int(cos['h95_censored'].sum())} / {len(cos)}",
        "",
        "Model comparison on held-out test seeds is in `../evaluate_joint/summary.md`.",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}  ({time.perf_counter() - t_start:.0f}s)")


if __name__ == "__main__":
    main(*sys.argv[1:2])
