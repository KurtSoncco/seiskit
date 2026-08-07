"""Stage-1 OLS mean model for χ ratios on ln(χ).

Fits standardized main effects with naive, seed-cluster, and cell-cluster SEs.
Writes CSV + summary.md under ``figure_dir("chi_ols", "stage1_mean_ols")``.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FACTORS,
    METRICS,
    N_CELLS,
    N_NODES,
    N_SEEDS,
    ZCOLS,
    add_design_columns,
    fmt,
    formula_rhs_main,
    load_ratios,
    log_response,
    out_dir,
)

warnings.filterwarnings("ignore")

N_CV_SPLITS = 5
RANDOM_STATE = 0


def fit_metric(df: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, dict]:
    """Return (effects table, fit metrics dict) for one metric."""
    y = log_response(df, metric)
    work = df.loc[np.isfinite(y)].copy()
    work["y"] = y[np.isfinite(y)]
    rhs = formula_rhs_main()
    formula = f"y ~ {rhs}"

    m = smf.ols(formula, data=work).fit()
    m_seed = smf.ols(formula, data=work).fit(cov_type="cluster", cov_kwds={"groups": work["seed"]})
    m_cell = smf.ols(formula, data=work).fit(cov_type="cluster", cov_kwds={"groups": work["cell"]})

    terms = ["Intercept"] + ZCOLS
    rows = []
    for t in terms:
        se_n = float(m.bse[t])
        se_s = float(m_seed.bse[t])
        se_c = float(m_cell.bse[t])
        rows.append(
            dict(
                metric=metric,
                term=t,
                factor=t.replace("_z", "") if t.endswith("_z") else t,
                coef=float(m.params[t]),
                se_naive=se_n,
                se_seed_cluster=se_s,
                se_cell_cluster=se_c,
                se_infl_seed=se_s / se_n if se_n > 0 else np.nan,
                se_infl_cell=se_c / se_n if se_n > 0 else np.nan,
                t_naive=float(m.tvalues[t]),
                p_naive=float(m.pvalues[t]),
                t_seed_cluster=float(m_seed.tvalues[t]),
                p_seed_cluster=float(m_seed.pvalues[t]),
                t_cell_cluster=float(m_cell.tvalues[t]),
                p_cell_cluster=float(m_cell.pvalues[t]),
            )
        )
    effects = pd.DataFrame(rows)

    # Cell-grouped CV R² (predict held-out cells).
    X = work[ZCOLS].to_numpy(dtype=float)
    yv = work["y"].to_numpy(dtype=float)
    groups = work["cell"].to_numpy()
    gkf = GroupKFold(n_splits=N_CV_SPLITS)
    yhat_cv = np.full_like(yv, np.nan)
    for tr, te in gkf.split(X, yv, groups):
        dtr = work.iloc[tr]
        mcv = smf.ols(formula, data=dtr).fit()
        yhat_cv[te] = mcv.predict(work.iloc[te])
    ss_res = float(np.nansum((yv - yhat_cv) ** 2))
    ss_tot = float(np.nansum((yv - np.nanmean(yv)) ** 2))
    r2_cv = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    metrics = dict(
        metric=metric,
        n_obs=int(len(work)),
        n_cells=int(work["cell"].nunique()),
        n_seeds=int(work["seed"].nunique()),
        r2_insample=float(m.rsquared),
        r2_adj=float(m.rsquared_adj),
        r2_cv_cell=float(r2_cv),
        rmse_insample=float(np.sqrt(np.mean(m.resid**2))),
        sigma2_resid=float(m.scale),
    )
    return effects, metrics


def build_summary_md(effects: pd.DataFrame, fit: pd.DataFrame) -> str:
    lines = [
        "# Stage-1 OLS mean model",
        "",
        "Ordinary least squares for 1D-normalized ratios on the log scale, "
        f"standardized main effects (`{'`, `'.join(FACTORS)}`).",
        "",
        rf"- Design cells: **{N_CELLS}**; nodes \(N_x = {N_NODES}\); seeds \(N_s = {N_SEEDS}\)",
        "- Working scale: \\(Y = \\ln\\chi\\) (non-positive / non-finite dropped)",
        "- Predictors: z-scored main effects only (no interactions)",
        "- Official uncertainty: report **cluster-robust** SEs "
        "(seed and cell); naive SEs are for inflation diagnostics only",
        "",
        "## Output files",
        "",
        "| File | Contents |",
        "|------|----------|",
        "| `mean_effects.csv` | Coefficient, naive / seed-cluster / cell-cluster "
        r"SE, SE inflation, \(t\)/\(p\) |",
        r"| `mean_fit_metrics.csv` | In-sample \(R^2\), adjusted \(R^2\), "
        rf"cell-grouped {N_CV_SPLITS}-fold CV \(R^2\), RMSE |",
        "| `summary.md` | this file |",
        "",
        "## Notation",
        "",
        "Canonical symbols: "
        "`statistical_analysis/full_paper/analysis/NOTATION.md` (`chi_ols` section).",
        "",
        r"Observation at design cell \(k\), seed \(j\), node \(i\):",
        "",
        r"$$",
        r"Y_{kij} = \mathbf{x}_k^\top\boldsymbol{\beta} + \varepsilon_{kij}.",
        r"$$",
        "",
        "Here \\(\\mathbf{x}_k\\) stacks an intercept and the five z-scored factors. "
        "Cluster-robust variance estimators allow arbitrary within-cluster "
        r"correlation (seed \(j\) or cell \(k\)). SE inflation:",
        "",
        r"$$",
        r"\mathrm{SE\,infl.} = "
        r"\frac{\mathrm{SE}_{\mathrm{cluster}}}{\mathrm{SE}_{\mathrm{naive}}}.",
        r"$$",
        "",
        "In-sample \\(R^2 = 1 - \\mathrm{SS}_{\\mathrm{res}}/"
        "\\mathrm{SS}_{\\mathrm{tot}}\\). Cell-grouped CV \\(R^2\\) holds out "
        "entire design cells so that within-cell spatial/seed dependence does "
        "not leak into the held-out score.",
        "",
        "## Fit metrics",
        "",
        r"| Metric | \(n\) | \(R^2\) in-sample | \(R^2\) CV (cell) | RMSE |",
        "|--------|------:|------------------:|------------------:|-----:|",
    ]
    for _, r in fit.iterrows():
        lines.append(
            f"| {r['metric']} | {int(r['n_obs']):,} | "
            f"{fmt(r['r2_insample'])} | {fmt(r['r2_cv_cell'])} | "
            f"{fmt(r['rmse_insample'])} |"
        )

    lines.extend(
        [
            "",
            "## Main-effect coefficients (seed-cluster SE)",
            "",
            "| Metric | Factor | \\(\\hat\\beta\\) | SE (seed) | SE infl. (seed) |",
            "|--------|--------|---------------:|----------:|----------------:|",
        ]
    )
    main = effects[effects["term"].isin(ZCOLS)]
    for _, r in main.iterrows():
        lines.append(
            f"| {r['metric']} | {r['factor']} | {fmt(r['coef'])} | "
            f"{fmt(r['se_seed_cluster'])} | {fmt(r['se_infl_seed'])} |"
        )

    # Conclusions
    lines.extend(["", "## Conclusions", ""])
    for _, r in fit.iterrows():
        m = r["metric"]
        infl = effects.loc[(effects["metric"] == m) & (effects["term"].isin(ZCOLS)), "se_infl_seed"]
        med_infl = float(infl.median()) if len(infl) else np.nan
        abs_coef = effects.loc[(effects["metric"] == m) & (effects["term"].isin(ZCOLS))].assign(
            abs_b=lambda d: d["coef"].abs()
        )
        top = abs_coef.sort_values("abs_b", ascending=False).iloc[0]
        gap = abs(float(r["r2_insample"]) - float(r["r2_cv_cell"]))
        lines.append(
            rf"- **{m}**: in-sample \(R^2 = {fmt(r['r2_insample'])}\), "
            rf"cell-CV \(R^2 = {fmt(r['r2_cv_cell'])}\) "
            f"(gap {fmt(gap)}); median seed-cluster SE inflation "
            f"{fmt(med_infl)}×; largest |β| among main effects is "
            f"**{top['factor']}** ({fmt(top['coef'])})."
        )
    lines.extend(
        [
            "",
            r"Low \(R^2\) with CV matching in-sample indicates design-driven "
            "signal plus large seed/node stochasticity—not mean-model overfitting. "
            "Use cluster-robust SEs for inference; naive SEs understate uncertainty "
            "when seeds or cells induce residual dependence.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    print("Loading join_master.h5 …")
    df = add_design_columns(load_ratios())
    print(f"  rows={len(df):,}, cells={df['cell'].nunique()}")

    effect_parts: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    for metric in METRICS:
        print(f"Fitting {metric} …")
        effects, met = fit_metric(df, metric)
        effect_parts.append(effects)
        metric_rows.append(met)
        print(f"  R²={met['r2_insample']:.4f}, CV R²={met['r2_cv_cell']:.4f}, n={met['n_obs']:,}")

    effects_df = pd.concat(effect_parts, ignore_index=True)
    fit_df = pd.DataFrame(metric_rows)

    dest = out_dir("stage1_mean_ols")
    effects_df.to_csv(dest / "mean_effects.csv", index=False)
    fit_df.to_csv(dest / "mean_fit_metrics.csv", index=False)
    (dest / "summary.md").write_text(build_summary_md(effects_df, fit_df), encoding="utf-8")
    print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
