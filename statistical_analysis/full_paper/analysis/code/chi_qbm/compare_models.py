"""Three-way comparison: naive OLS, spatial GLS OLS, LightGBM QBM.

Same seed holdout. Writes CSV + summary.md under
``figure_dir("chi_qbm", "compare_models")``.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CHI_OLS_CEILING,
    CHI_OLS_STAGE1,
    FACTORS,
    LAG1_FAIL_THRESH,
    METRICS,
    QBM_FEATURES,
    TAUS,
    ZCOLS,
    add_design_columns,
    fmt,
    koenker_pseudo_r2,
    lag1_pearson,
    load_ratios,
    log_response,
    models_dir,
    out_dir,
    pinball_loss,
    r2_score,
    rmse,
    seed_grouped_split_indices,
)

warnings.filterwarnings("ignore")


def load_ols_beta_fullsample() -> dict[str, np.ndarray]:
    """Map metric → β from chi_ols full-sample Stage-1 (Intercept + ZCOLS)."""
    path = CHI_OLS_STAGE1 / "mean_effects.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing naive OLS effects: {path}")
    eff = pd.read_csv(path)
    terms = ["Intercept"] + ZCOLS
    out: dict[str, np.ndarray] = {}
    for metric in METRICS:
        sub = eff[eff["metric"] == metric].set_index("term")
        out[metric] = np.array([float(sub.loc[t, "coef"]) for t in terms], dtype=float)
    return out


def load_gls_beta() -> dict[str, np.ndarray]:
    path = out_dir("spatial_ols") / "spatial_mean_effects.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing spatial OLS effects: {path}. Run spatial_ols.py first.")
    eff = pd.read_csv(path)
    terms = ["Intercept"] + ZCOLS
    out: dict[str, np.ndarray] = {}
    for metric in METRICS:
        sub = eff[eff["metric"] == metric].set_index("term")
        out[metric] = np.array([float(sub.loc[t, "coef_gls"]) for t in terms], dtype=float)
    return out


def predict_linear(df: pd.DataFrame, beta: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(len(df)), df[ZCOLS].to_numpy(dtype=float)])
    return X @ beta


def sigma_from_train(y_tr: np.ndarray, yhat_tr: np.ndarray) -> float:
    m = np.isfinite(y_tr) & np.isfinite(yhat_tr)
    return float(np.std(y_tr[m] - yhat_tr[m], ddof=1))


def ols_quantile_band(yhat: np.ndarray, sigma: float, tau: float) -> np.ndarray:
    z = float(stats.norm.ppf(tau))
    return yhat + z * sigma


def load_qbm(metric: str) -> tuple[lgb.Booster, dict[float, lgb.Booster]]:
    mdir = models_dir()
    mean = joblib.load(mdir / f"lgbm_mean_{metric}_seed.pkl")
    qmods = {}
    for tau in TAUS:
        kind = f"q{int(round(tau * 100)):02d}"
        qmods[tau] = joblib.load(mdir / f"lgbm_{kind}_{metric}_seed.pkl")
    return mean, qmods


def predict_sorted_q(qmods: dict[float, lgb.Booster], X) -> np.ndarray:
    raw = np.column_stack([qmods[t].predict(X) for t in TAUS])
    raw.sort(axis=1)
    return raw


def residual_lag1_summary(
    df_te: pd.DataFrame,
    resid: np.ndarray,
) -> dict[str, float]:
    """Median |lag-1| of residuals within (cell, seed) blocks on holdout."""
    work = df_te.copy()
    work["_r"] = resid
    vals: list[float] = []
    for _, g in work.groupby(["cell", "seed"], sort=False):
        g = g.sort_values("node")
        if len(g) < 8:
            continue
        vals.append(abs(lag1_pearson(g["_r"].to_numpy(dtype=float))))
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return dict(
            median_abs_lag1=np.nan,
            frac_lag1_gt_thresh=np.nan,
            n_blocks=0,
        )
    return dict(
        median_abs_lag1=float(np.nanmedian(arr)),
        frac_lag1_gt_thresh=float(np.mean(arr > LAG1_FAIL_THRESH)),
        n_blocks=int(arr.size),
    )


def load_ceiling() -> dict[str, float]:
    if not CHI_OLS_CEILING.is_file():
        return {}
    c = pd.read_csv(CHI_OLS_CEILING)
    c = c[c["scope"] == "full"]
    return {str(r["metric"]): float(r["reliability_ceiling"]) for _, r in c.iterrows()}


def build_summary_md(
    cmp: pd.DataFrame,
    pi: pd.DataFrame,
    spat: pd.DataFrame,
) -> str:
    lines = [
        "# Naive OLS vs spatial GLS vs QBM",
        "",
        "Full-array three-way comparison on the shared **seed-grouped** holdout "
        f"(\(Y = \\ln\\chi\); factors `{'`, `'.join(FACTORS)}`).",
        "",
        "| Model | Role |",
        "|-------|------|",
        "| Naive OLS | Main-effects linear mean; homoscedastic Gaussian bands |",
        "| Spatial OLS | Same mean, CosWM feasible GLS |",
        "| Mean GBM / QBM | Boosted mean + conditional quantiles "
        "(interactions + hetero); `node_z` spatial feature |",
        "",
        "## Output files",
        "",
        "| File | Contents |",
        "|------|----------|",
        "| `comparison_metrics.csv` | \(R^2\), RMSE, efficiency, pinball, "
        "pseudo-\(R^2\) |",
        "| `pi_hetero.csv` | 90% PI coverage / width / max÷min by model |",
        "| `residual_spatial_acf.csv` | Holdout residual \|lag-1\| by model |",
        "| `summary.md` | this file |",
        "",
        "## Notation",
        "",
        "Pinball loss \(\\rho_\\tau\); Koenker \(R^1(\\tau) = "
        "1 - \\rho_\\tau(\\mathrm{model}) / \\rho_\\tau(\\mathrm{null})\). "
        "Naive OLS intervals use \(\\hat y + z_\\tau \\hat\\sigma\) with "
        "constant \(\\hat\\sigma\) from train residuals. QBM intervals use "
        "\(q_{0.05}, q_{0.95}\) (sorted across τ to fix crossings).",
        "",
        "## Point prediction (\(R^2\))",
        "",
        "| Metric | Naive OLS | Spatial GLS | Mean GBM | Ceiling | "
        "Eff. GBM |",
        "|--------|----------:|------------:|---------:|--------:|--------:|",
    ]
    for metric in METRICS:
        rows = {r["model"]: r for _, r in cmp[cmp["metric"] == metric].iterrows()}
        gbm = rows.get("mean_gbm", {})
        lines.append(
            f"| {metric} | {fmt(rows.get('naive_ols', {}).get('r2', np.nan))} | "
            f"{fmt(rows.get('spatial_ols', {}).get('r2', np.nan))} | "
            f"{fmt(gbm.get('r2', np.nan))} | "
            f"{fmt(gbm.get('r2_ceiling', np.nan))} | "
            f"{fmt(gbm.get('efficiency', np.nan))} |"
        )

    lines.extend(
        [
            "",
            "## Distributional (median pinball / 90% PI)",
            "",
            "| Metric | OLS pinball τ=0.5 | QBM pinball τ=0.5 | "
            "OLS 90% cov | QBM 90% cov | QBM width ratio |",
            "|--------|------------------:|------------------:|"
            "-------------:|-------------:|----------------:|",
        ]
    )
    for metric in METRICS:
        p = {r["model"]: r for _, r in pi[pi["metric"] == metric].iterrows()}
        c = cmp[(cmp["metric"] == metric) & (cmp["model"].isin(["naive_ols", "qbm"]))]
        pin_ols = c.loc[c["model"] == "naive_ols", "pinball_q50"]
        pin_q = c.loc[c["model"] == "qbm", "pinball_q50"]
        lines.append(
            f"| {metric} | {fmt(float(pin_ols.iloc[0]) if len(pin_ols) else np.nan)} | "
            f"{fmt(float(pin_q.iloc[0]) if len(pin_q) else np.nan)} | "
            f"{fmt(p.get('naive_ols', {}).get('cov90', np.nan))} | "
            f"{fmt(p.get('qbm', {}).get('cov90', np.nan))} | "
            f"{fmt(p.get('qbm', {}).get('width_ratio', np.nan))} |"
        )

    lines.extend(
        [
            "",
            "## Residual spatial lag-1 (holdout)",
            "",
            "| Metric | Naive OLS | Spatial GLS | QBM median |",
            "|--------|----------:|------------:|-----------:|",
        ]
    )
    for metric in METRICS:
        s = {r["model"]: r for _, r in spat[spat["metric"] == metric].iterrows()}
        lines.append(
            f"| {metric} | {fmt(s.get('naive_ols', {}).get('median_abs_lag1', np.nan))} | "
            f"{fmt(s.get('spatial_ols', {}).get('median_abs_lag1', np.nan))} | "
            f"{fmt(s.get('qbm', {}).get('median_abs_lag1', np.nan))} |"
        )

    lines.extend(["", "## Conclusions", ""])
    for metric in METRICS:
        rows = {r["model"]: r for _, r in cmp[cmp["metric"] == metric].iterrows()}
        p = {r["model"]: r for _, r in pi[pi["metric"] == metric].iterrows()}
        s = {r["model"]: r for _, r in spat[spat["metric"] == metric].iterrows()}
        r_ols = rows.get("naive_ols", {}).get("r2", np.nan)
        r_gls = rows.get("spatial_ols", {}).get("r2", np.nan)
        r_gbm = rows.get("mean_gbm", {}).get("r2", np.nan)
        wr = p.get("qbm", {}).get("width_ratio", np.nan)
        lines.append(
            f"- **{metric}**: mean \(R^2\) OLS {fmt(r_ols)} / GLS {fmt(r_gls)} / "
            f"GBM {fmt(r_gbm)}; QBM 90% PI width ratio {fmt(wr)} "
            f"(hetero signal); residual \|lag-1\| "
            f"OLS {fmt(s.get('naive_ols', {}).get('median_abs_lag1', np.nan))} → "
            f"QBM {fmt(s.get('qbm', {}).get('median_abs_lag1', np.nan))}."
        )
    lines.extend(
        [
            "",
            "**Takeaways.** QBM improves distributional calibration and "
            "factor-dependent spread without claiming a parametric variance "
            "model. Spatial GLS corrects residual correlation for linear "
            "estimation but typically does not raise holdout \(R^2\) much over "
            "naive OLS. Residual lag-1 after QBM may remain elevated because "
            "`node_z` is a smooth spatial feature, not a full CosWM whitening "
            "of RF dependence—report it honestly alongside pinball/PI metrics.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    print("Loading data …")
    df = add_design_columns(load_ratios())
    tr, te = seed_grouped_split_indices(df)
    df_tr, df_te = df.iloc[tr], df.iloc[te]
    print(f"  holdout n={len(df_te):,}")

    ols_beta = load_ols_beta_fullsample()
    gls_beta = load_gls_beta()
    ceiling = load_ceiling()

    cmp_rows: list[dict] = []
    pi_rows: list[dict] = []
    spat_rows: list[dict] = []

    for metric in METRICS:
        print(f"Comparing {metric} …")
        y_tr = log_response(df_tr, metric)
        y_te = log_response(df_te, metric)
        m_te = np.isfinite(y_te)
        yt = y_te[m_te]
        dte = df_te.loc[m_te]

        # --- Naive OLS ---
        yhat_ols_tr = predict_linear(df_tr, ols_beta[metric])
        yhat_ols = predict_linear(dte, ols_beta[metric])
        sig = sigma_from_train(y_tr, yhat_ols_tr)
        r2_ols = r2_score(yt, yhat_ols)
        ceil = ceiling.get(metric, np.nan)

        # Homoscedastic quantile bands
        q_ols = {tau: ols_quantile_band(yhat_ols, sig, tau) for tau in TAUS}
        pin50_ols = pinball_loss(yt, q_ols[0.50], 0.50)
        null50 = float(np.quantile(y_tr[np.isfinite(y_tr)], 0.50))
        pseudo50_ols = koenker_pseudo_r2(yt, q_ols[0.50], 0.50, null50)
        cov90_ols = float(np.mean((yt >= q_ols[0.05]) & (yt <= q_ols[0.95])))
        w_ols = q_ols[0.95] - q_ols[0.05]

        cmp_rows.append(
            dict(
                metric=metric,
                model="naive_ols",
                r2=r2_ols,
                rmse=rmse(yt, yhat_ols),
                r2_ceiling=ceil,
                efficiency=r2_ols / ceil if ceil and ceil > 0 else np.nan,
                pinball_q50=pin50_ols,
                pseudo_r2_q50=pseudo50_ols,
                pinball_q05=pinball_loss(yt, q_ols[0.05], 0.05),
                pinball_q95=pinball_loss(yt, q_ols[0.95], 0.95),
            )
        )
        pi_rows.append(
            dict(
                metric=metric,
                model="naive_ols",
                cov90=cov90_ols,
                mean_width=float(np.mean(w_ols)),
                width_min=float(np.min(w_ols)),
                width_max=float(np.max(w_ols)),
                width_ratio=1.0,  # constant width
            )
        )
        spat_rows.append(
            dict(
                metric=metric,
                model="naive_ols",
                **residual_lag1_summary(dte, yt - yhat_ols),
            )
        )

        # --- Spatial GLS ---
        yhat_gls = predict_linear(dte, gls_beta[metric])
        r2_gls = r2_score(yt, yhat_gls)
        yhat_gls_tr = predict_linear(df_tr, gls_beta[metric])
        sig_g = sigma_from_train(y_tr, yhat_gls_tr)
        q_gls = {tau: ols_quantile_band(yhat_gls, sig_g, tau) for tau in TAUS}
        cmp_rows.append(
            dict(
                metric=metric,
                model="spatial_ols",
                r2=r2_gls,
                rmse=rmse(yt, yhat_gls),
                r2_ceiling=ceil,
                efficiency=r2_gls / ceil if ceil and ceil > 0 else np.nan,
                pinball_q50=pinball_loss(yt, q_gls[0.50], 0.50),
                pseudo_r2_q50=koenker_pseudo_r2(yt, q_gls[0.50], 0.50, null50),
                pinball_q05=pinball_loss(yt, q_gls[0.05], 0.05),
                pinball_q95=pinball_loss(yt, q_gls[0.95], 0.95),
            )
        )
        w_gls = q_gls[0.95] - q_gls[0.05]
        pi_rows.append(
            dict(
                metric=metric,
                model="spatial_ols",
                cov90=float(np.mean((yt >= q_gls[0.05]) & (yt <= q_gls[0.95]))),
                mean_width=float(np.mean(w_gls)),
                width_min=float(np.min(w_gls)),
                width_max=float(np.max(w_gls)),
                width_ratio=1.0,
            )
        )
        spat_rows.append(
            dict(
                metric=metric,
                model="spatial_ols",
                **residual_lag1_summary(dte, yt - yhat_gls),
            )
        )

        # --- QBM / mean GBM ---
        mean_m, qmods = load_qbm(metric)
        Xte = dte[QBM_FEATURES]
        yhat_gbm = mean_m.predict(Xte)
        r2_gbm = r2_score(yt, yhat_gbm)
        preds = predict_sorted_q(qmods, Xte)
        # map sorted columns back: after sort, col0≈q05 ... col-1≈q95
        qhat = {tau: preds[:, i] for i, tau in enumerate(TAUS)}
        pin50_q = pinball_loss(yt, qhat[0.50], 0.50)
        cmp_rows.append(
            dict(
                metric=metric,
                model="mean_gbm",
                r2=r2_gbm,
                rmse=rmse(yt, yhat_gbm),
                r2_ceiling=ceil,
                efficiency=r2_gbm / ceil if ceil and ceil > 0 else np.nan,
                pinball_q50=np.nan,
                pseudo_r2_q50=np.nan,
                pinball_q05=np.nan,
                pinball_q95=np.nan,
            )
        )
        cmp_rows.append(
            dict(
                metric=metric,
                model="qbm",
                r2=r2_score(yt, qhat[0.50]),  # median as point
                rmse=rmse(yt, qhat[0.50]),
                r2_ceiling=ceil,
                efficiency=(r2_score(yt, qhat[0.50]) / ceil) if ceil and ceil > 0 else np.nan,
                pinball_q50=pin50_q,
                pseudo_r2_q50=koenker_pseudo_r2(yt, qhat[0.50], 0.50, null50),
                pinball_q05=pinball_loss(yt, qhat[0.05], 0.05),
                pinball_q95=pinball_loss(yt, qhat[0.95], 0.95),
            )
        )
        w_q = qhat[0.95] - qhat[0.05]
        pi_rows.append(
            dict(
                metric=metric,
                model="qbm",
                cov90=float(np.mean((yt >= qhat[0.05]) & (yt <= qhat[0.95]))),
                mean_width=float(np.mean(w_q)),
                width_min=float(np.min(w_q)),
                width_max=float(np.max(w_q)),
                width_ratio=float(np.max(w_q) / np.min(w_q)) if np.min(w_q) > 0 else np.nan,
            )
        )
        spat_rows.append(
            dict(
                metric=metric,
                model="qbm",
                **residual_lag1_summary(dte, yt - qhat[0.50]),
            )
        )
        print(
            f"  R² OLS={r2_ols:.4f} GLS={r2_gls:.4f} GBM={r2_gbm:.4f} "
            f"QBM pin50={pin50_q:.5f} cov90={pi_rows[-1]['cov90']:.3f}"
        )

    cmp = pd.DataFrame(cmp_rows)
    pi = pd.DataFrame(pi_rows)
    spat = pd.DataFrame(spat_rows)

    dest = out_dir("compare_models")
    cmp.to_csv(dest / "comparison_metrics.csv", index=False)
    pi.to_csv(dest / "pi_hetero.csv", index=False)
    spat.to_csv(dest / "residual_spatial_acf.csv", index=False)
    (dest / "summary.md").write_text(build_summary_md(cmp, pi, spat), encoding="utf-8")

    # Copy split reference
    split_src = out_dir("train_qbm") / "seed_split.json"
    if split_src.is_file():
        (dest / "seed_split.json").write_text(split_src.read_text(), encoding="utf-8")

    print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
