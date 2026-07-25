"""CosWM-whitened / feasible GLS mean baseline for full-array χ ratios.

Within each (cell, seed) node block, uses the cell CosWM (fallback Gauss/Exp)
correlation from ``chi_spatial/spatial_acf`` to form block-diagonal GLS.

Writes CSV + summary.md under ``figure_dir("chi_qbm", "spatial_ols")``.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ACF_FIT_PATH,
    CHI_OLS_STAGE1,
    DX_M,
    FACTORS,
    LAG1_FAIL_THRESH,
    METRICS,
    N_CELLS,
    N_NODES,
    N_SEEDS,
    ZCOLS,
    add_design_columns,
    fmt,
    lag1_pearson,
    load_ratios,
    log_response,
    out_dir,
    r2_score,
    rmse,
    save_split,
    seed_grouped_split_indices,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chi_spatial"))
from spatial_acf import rho_coswm, rho_exp, rho_gauss  # noqa: E402

warnings.filterwarnings("ignore")

JITTER = 1e-8
COND_MAX = 1e12


def _rho_vec(row: pd.Series, h: np.ndarray) -> tuple[np.ndarray | None, str]:
    if bool(row.get("fit_ok_coswm", False)):
        return (
            rho_coswm(
                h,
                float(row["c0_coswm"]),
                float(row["nu_coswm"]),
                float(row["scale_s_m_coswm"]),
                float(row["period_b_m_coswm"]),
            ),
            "coswm",
        )
    if bool(row.get("fit_ok_gauss", False)):
        return rho_gauss(h, float(row["c0_gauss"]), float(row["a_m_gauss"])), "gauss"
    if bool(row.get("fit_ok_exp", False)):
        return rho_exp(h, float(row["c0_exp"]), float(row["a_m_exp"])), "exp"
    return None, "none"


def correlation_matrix(rho_lags: np.ndarray, n: int = N_NODES) -> np.ndarray:
    """Build Toeplitz correlation from lag-1..n-1 values; ρ(0)=1."""
    R = np.eye(n, dtype=float)
    for k in range(1, n):
        r = float(rho_lags[k - 1]) if np.isfinite(rho_lags[k - 1]) else 0.0
        r = float(np.clip(r, -0.999, 0.999))
        np.fill_diagonal(R[k:], r)
        np.fill_diagonal(R[:, k:], r)
    R.flat[:: n + 1] = 1.0
    return R


def inv_factors(R: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """Return (Rinv, chol L with R=LL', ok). Skip block if unstable."""
    Rw = R + JITTER * np.eye(R.shape[0])
    try:
        cond = float(np.linalg.cond(Rw))
        if not np.isfinite(cond) or cond > COND_MAX:
            return np.empty(0), np.empty(0), False
        L = np.linalg.cholesky(Rw)
        Rinv = np.linalg.inv(Rw)
        return Rinv, L, True
    except LinAlgError:
        return np.empty(0), np.empty(0), False


def load_acf_lookup(path: Path = ACF_FIT_PATH) -> dict[tuple, pd.Series]:
    fits = pd.read_csv(path)
    lookup: dict[tuple, pd.Series] = {}
    for _, row in fits.iterrows():
        key = (
            float(row["Vs1"]),
            float(row["Height"]),
            float(row["CoV"]),
            float(row["rH"]),
            float(row["aHV"]),
            str(row["metric"]),
        )
        lookup[key] = row
    return lookup


def cell_key(row: pd.Series, metric: str) -> tuple:
    return (
        float(row["Vs1"]),
        float(row["Height"]),
        float(row["CoV"]),
        float(row["rH"]),
        float(row["aHV"]),
        metric,
    )


def design_row(df_row: pd.Series) -> np.ndarray:
    return np.array([1.0] + [float(df_row[z]) for z in ZCOLS], dtype=float)


def accumulate_gls(
    df: pd.DataFrame,
    y: np.ndarray,
    metric: str,
    acf_lookup: dict[tuple, pd.Series],
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Accumulate XtWX, XtWy over (cell, seed) blocks; return QA stats."""
    p = 1 + len(ZCOLS)
    XtWX = np.zeros((p, p), dtype=float)
    XtWy = np.zeros(p, dtype=float)

    work = df.copy()
    work["_y"] = y
    work["_ok"] = np.isfinite(y)
    if mask is not None:
        work = work.loc[mask].copy()

    h = DX_M * np.arange(1, N_NODES, dtype=float)
    # Cache Rinv per cell
    rinv_cache: dict[int, tuple[np.ndarray, np.ndarray, str, bool]] = {}

    n_blocks = 0
    n_ok = 0
    n_skip_acf = 0
    n_skip_chol = 0
    n_skip_nodes = 0
    lag1_pre: list[float] = []
    lag1_post: list[float] = []

    for cell_id, gcell in work.groupby("cell", sort=False):
        # one design row per cell
        xk = design_row(gcell.iloc[0])
        key = cell_key(gcell.iloc[0], metric)
        row = acf_lookup.get(key)
        if row is None:
            n_skip_acf += int(gcell["seed"].nunique())
            continue
        rho, model = _rho_vec(row, h)
        if rho is None:
            n_skip_acf += int(gcell["seed"].nunique())
            continue
        R = correlation_matrix(rho, N_NODES)
        Rinv, L, ok = inv_factors(R)
        if not ok:
            n_skip_chol += int(gcell["seed"].nunique())
            continue
        rinv_cache[int(cell_id)] = (Rinv, L, model, True)
        ones = np.ones(N_NODES, dtype=float)
        w_sum = float(ones @ Rinv @ ones)

        for seed, gseed in gcell.groupby("seed", sort=False):
            n_blocks += 1
            gseed = gseed.sort_values("node")
            if len(gseed) != N_NODES or not bool(gseed["_ok"].all()):
                n_skip_nodes += 1
                continue
            yb = gseed["_y"].to_numpy(dtype=float)
            # pre-whiten lag-1 on demeaned block
            lag1_pre.append(abs(lag1_pearson(yb - yb.mean())))
            # whitened residual relative to block mean (for QA)
            try:
                yw = np.linalg.solve(L, yb - yb.mean())
                lag1_post.append(abs(lag1_pearson(yw)))
            except LinAlgError:
                n_skip_chol += 1
                continue

            yt_rinv_1 = float(ones @ Rinv @ yb)
            XtWX += w_sum * np.outer(xk, xk)
            XtWy += yt_rinv_1 * xk
            n_ok += 1

    qa = dict(
        metric=metric,
        n_blocks=n_blocks,
        n_blocks_ok=n_ok,
        n_skip_acf=n_skip_acf,
        n_skip_chol=n_skip_chol,
        n_skip_nodes=n_skip_nodes,
        frac_ok=n_ok / n_blocks if n_blocks else np.nan,
        median_abs_lag1_pre=float(np.nanmedian(lag1_pre)) if lag1_pre else np.nan,
        median_abs_lag1_post=float(np.nanmedian(lag1_post)) if lag1_post else np.nan,
        frac_lag1_post_gt_thresh=(
            float(np.mean(np.asarray(lag1_post) > LAG1_FAIL_THRESH)) if lag1_post else np.nan
        ),
        lag1_fail_thresh=LAG1_FAIL_THRESH,
        spatial_ok=(
            bool(lag1_post)
            and float(np.mean(np.asarray(lag1_post) > LAG1_FAIL_THRESH)) <= 0.25
        ),
    )
    return XtWX, XtWy, qa


def solve_beta(XtWX: np.ndarray, XtWy: np.ndarray) -> np.ndarray:
    return np.linalg.solve(XtWX + JITTER * np.eye(XtWX.shape[0]), XtWy)


def predict_design(df: pd.DataFrame, beta: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(len(df)), df[ZCOLS].to_numpy(dtype=float)])
    return X @ beta


def fit_ols_beta(df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Naive OLS β on finite rows (for holdout comparison of same design)."""
    m = np.isfinite(y)
    X = np.column_stack([np.ones(m.sum()), df.loc[m, ZCOLS].to_numpy(dtype=float)])
    yt = y[m]
    beta, *_ = np.linalg.lstsq(X, yt, rcond=None)
    return beta


def build_summary_md(
    effects: pd.DataFrame,
    fit: pd.DataFrame,
    qa: pd.DataFrame,
) -> str:
    lines = [
        "# Spatial OLS (CosWM feasible GLS)",
        "",
        "Full-array mean model for \(Y = \\ln\\chi\) with block-diagonal "
        "CosWM correlation within each `(cell, seed)` node strip.",
        "",
        f"- Design cells: **{N_CELLS}**; \(N_x = {N_NODES}\); \(N_s = {N_SEEDS}\)",
        "- Predictors: intercept + z-scored main effects "
        f"(`{'`, `'.join(FACTORS)}`) — same mean as naive OLS",
        "- Correlation \(R_k\) from `chi_spatial/spatial_acf` "
        "(CosWM preferred; Gauss/Exp fallback)",
        "- Unstable Cholesky / incomplete node blocks are **skipped** "
        "(never replaced by identity \(R\))",
        "",
        "## Output files",
        "",
        "| File | Contents |",
        "|------|----------|",
        "| `spatial_mean_effects.csv` | GLS \(\\hat\\beta\) by metric / term |",
        "| `spatial_fit_metrics.csv` | Train / seed-holdout \(R^2\), RMSE; "
        "naive OLS holdout for reference |",
        "| `whitening_qa.csv` | Pre/post lag-1, skip counts, `spatial_ok` |",
        "| `seed_split.json` | Shared seed holdout used here and by QBM |",
        "| `summary.md` | this file |",
        "",
        "## Notation",
        "",
        "See `NOTATION.md` (`chi_qbm`). Feasible GLS with cell-constant "
        "design rows:",
        "",
        r"$$",
        r"\hat{\boldsymbol{\beta}}_{\mathrm{GLS}}"
        r" = \Bigl(\sum_{k,j}(\mathbf{1}^\top R_k^{-1}\mathbf{1})"
        r"\,\mathbf{x}_k\mathbf{x}_k^\top\Bigr)^{-1}"
        r"\sum_{k,j}(\mathbf{1}^\top R_k^{-1}\mathbf{y}_{kj})\,\mathbf{x}_k.",
        r"$$",
        "",
        "## Whitening QA",
        "",
        "| Metric | Blocks OK | Median \|lag1\| pre | post | "
        f"Frac post>\({LAG1_FAIL_THRESH}\) | spatial_ok |",
        "|--------|----------:|--------------------:|-----:|"
        "---------------------------:|:----------:|",
    ]
    for _, r in qa.iterrows():
        lines.append(
            f"| {r['metric']} | {int(r['n_blocks_ok']):,} / {int(r['n_blocks']):,} | "
            f"{fmt(r['median_abs_lag1_pre'])} | {fmt(r['median_abs_lag1_post'])} | "
            f"{fmt(r['frac_lag1_post_gt_thresh'])} | "
            f"{'yes' if r['spatial_ok'] else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Holdout fit (seed-grouped)",
            "",
            "| Metric | GLS \(R^2\) | GLS RMSE | Naive OLS \(R^2\) | "
            "Δ\(R^2\) (GLS−OLS) |",
            "|--------|------------:|---------:|------------------:|------------------:|",
        ]
    )
    for _, r in fit.iterrows():
        lines.append(
            f"| {r['metric']} | {fmt(r['r2_holdout_gls'])} | "
            f"{fmt(r['rmse_holdout_gls'])} | {fmt(r['r2_holdout_ols'])} | "
            f"{fmt(r['delta_r2_gls_minus_ols'])} |"
        )

    lines.extend(["", "## Conclusions", ""])
    n_ok = int(qa["spatial_ok"].sum())
    lines.append(
        f"- Whitening gate `spatial_ok` passes for **{n_ok}/{len(qa)}** metrics "
        f"(post lag-1 fail fraction ≤ 25%). CosWM GLS remains a **spatial "
        "correction of dependence for estimation**, not a guarantee of white "
        "residuals."
    )
    for _, r in fit.iterrows():
        d = float(r["delta_r2_gls_minus_ols"])
        verb = "improves" if d > 0.005 else ("worsens" if d < -0.005 else "matches")
        lines.append(
            f"- **{r['metric']}**: GLS holdout \(R^2 = {fmt(r['r2_holdout_gls'])}\) "
            f"{verb} naive OLS ({fmt(r['r2_holdout_ols'])}); "
            f"Δ = {fmt(d)}."
        )
    lines.extend(
        [
            "",
            "Point-prediction \(R^2\) need not rise under GLS when the mean "
            "is already nearly BLUE under OLS; the scientific gain is "
            "correcting for lateral correlation when interpreting "
            "\(\boldsymbol{\beta}\) and residual structure. Compare both "
            "baselines to QBM in `compare_models`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    if not ACF_FIT_PATH.is_file():
        raise FileNotFoundError(f"Missing ACF fits: {ACF_FIT_PATH}")

    print("Loading join_master.h5 …")
    df = add_design_columns(load_ratios())
    print(f"  rows={len(df):,}")
    acf_lookup = load_acf_lookup()
    print(f"  ACF cells×metrics={len(acf_lookup):,}")

    tr, te = seed_grouped_split_indices(df)
    dest = out_dir("spatial_ols")
    save_split(dest / "seed_split.json", tr, te, df)
    print(f"  train={len(tr):,} test={len(te):,}")

    effect_rows: list[dict] = []
    fit_rows: list[dict] = []
    qa_rows: list[dict] = []
    terms = ["Intercept"] + ZCOLS

    train_mask = np.zeros(len(df), dtype=bool)
    train_mask[tr] = True

    for metric in METRICS:
        print(f"GLS {metric} …")
        y = log_response(df, metric)
        XtWX, XtWy, qa = accumulate_gls(df, y, metric, acf_lookup, mask=train_mask)
        qa_rows.append(qa)
        if qa["n_blocks_ok"] < 10:
            print(f"  WARNING: only {qa['n_blocks_ok']} OK blocks")
            beta = np.full(1 + len(ZCOLS), np.nan)
        else:
            beta = solve_beta(XtWX, XtWy)

        for t, b in zip(terms, beta):
            effect_rows.append(
                dict(
                    metric=metric,
                    term=t,
                    factor=t.replace("_z", "") if t.endswith("_z") else t,
                    coef_gls=float(b) if np.isfinite(b) else np.nan,
                )
            )

        # Holdout predictions
        y_te = y[te]
        m_te = np.isfinite(y_te)
        df_te = df.iloc[te]
        yhat_gls = predict_design(df_te, beta)
        beta_ols = fit_ols_beta(df.iloc[tr], y[tr])
        yhat_ols = predict_design(df_te, beta_ols)

        r2_gls = r2_score(y_te[m_te], yhat_gls[m_te])
        r2_ols = r2_score(y_te[m_te], yhat_ols[m_te])
        # Train GLS R² on train rows (prediction uses design only)
        y_tr = y[tr]
        m_tr = np.isfinite(y_tr)
        yhat_tr = predict_design(df.iloc[tr], beta)
        fit_rows.append(
            dict(
                metric=metric,
                n_train=int(m_tr.sum()),
                n_test=int(m_te.sum()),
                r2_train_gls=r2_score(y_tr[m_tr], yhat_tr[m_tr]),
                rmse_train_gls=rmse(y_tr[m_tr], yhat_tr[m_tr]),
                r2_holdout_gls=r2_gls,
                rmse_holdout_gls=rmse(y_te[m_te], yhat_gls[m_te]),
                r2_holdout_ols=r2_ols,
                rmse_holdout_ols=rmse(y_te[m_te], yhat_ols[m_te]),
                delta_r2_gls_minus_ols=r2_gls - r2_ols,
                spatial_ok=qa["spatial_ok"],
                median_abs_lag1_post=qa["median_abs_lag1_post"],
            )
        )
        print(
            f"  holdout R² GLS={r2_gls:.4f} OLS={r2_ols:.4f} "
            f"lag1_post={qa['median_abs_lag1_post']:.3f} "
            f"spatial_ok={qa['spatial_ok']}"
        )

    effects = pd.DataFrame(effect_rows)
    fit = pd.DataFrame(fit_rows)
    qa_df = pd.DataFrame(qa_rows)

    # Optional: attach naive OLS coefs from chi_ols for side-by-side
    naive_path = CHI_OLS_STAGE1 / "mean_effects.csv"
    if naive_path.is_file():
        naive = pd.read_csv(naive_path)
        naive = naive.rename(columns={"coef": "coef_ols_fullsample"})
        effects = effects.merge(
            naive[["metric", "term", "coef_ols_fullsample"]],
            on=["metric", "term"],
            how="left",
        )

    effects.to_csv(dest / "spatial_mean_effects.csv", index=False)
    fit.to_csv(dest / "spatial_fit_metrics.csv", index=False)
    qa_df.to_csv(dest / "whitening_qa.csv", index=False)
    (dest / "summary.md").write_text(
        build_summary_md(effects, fit, qa_df), encoding="utf-8"
    )
    print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
