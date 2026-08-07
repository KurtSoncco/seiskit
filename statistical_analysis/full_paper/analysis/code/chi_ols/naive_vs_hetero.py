r"""Naive OLS vs heteroscedasticity / cluster-robust contrast for χ ratios.

Breusch–Pagan, Levene by factor, SE inflation, and a cell-level variance
regression \(\log s^2_k \sim \mathbf{z}_k\) (descriptor; full CosWM-whitened
Stage-3 lives in Box mixed_model).

Writes CSV + summary.md under ``figure_dir("chi_ols", "naive_vs_hetero")``.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan

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

ALPHA = 0.05
MIN_GROUP = 3


def _levene_groups(groups: list[np.ndarray]) -> tuple[float, float]:
    cleaned = [g[np.isfinite(g)] for g in groups]
    cleaned = [g for g in cleaned if g.size >= MIN_GROUP]
    if len(cleaned) < 2:
        return np.nan, np.nan
    try:
        W, p = stats.levene(*cleaned, center="median")
        return float(W), float(p)
    except Exception:
        return np.nan, np.nan


def assess_metric(df: pd.DataFrame, metric: str) -> dict[str, pd.DataFrame | dict]:
    y = log_response(df, metric)
    work = df.loc[np.isfinite(y)].copy()
    work["y"] = y[np.isfinite(y)]
    formula = f"y ~ {formula_rhs_main()}"

    m = smf.ols(formula, data=work).fit()
    m_seed = smf.ols(formula, data=work).fit(cov_type="cluster", cov_kwds={"groups": work["seed"]})
    m_cell = smf.ols(formula, data=work).fit(cov_type="cluster", cov_kwds={"groups": work["cell"]})

    # Breusch–Pagan on Stage-1 residuals vs design
    X = sm.add_constant(work[ZCOLS])
    lm, lm_p, fstat, f_p = het_breuschpagan(m.resid, np.asarray(X))
    bp = dict(
        metric=metric,
        n_obs=int(len(work)),
        lm_stat=float(lm),
        lm_p=float(lm_p),
        f_stat=float(fstat),
        f_p=float(f_p),
        reject_homo_alpha=bool(lm_p < ALPHA),
        alpha=ALPHA,
    )

    # SE inflation table
    se_rows = []
    for t in ["Intercept"] + ZCOLS:
        se_n = float(m.bse[t])
        se_s = float(m_seed.bse[t])
        se_c = float(m_cell.bse[t])
        se_rows.append(
            dict(
                metric=metric,
                term=t,
                factor=t.replace("_z", "") if t.endswith("_z") else t,
                se_naive=se_n,
                se_seed_cluster=se_s,
                se_cell_cluster=se_c,
                se_infl_seed=se_s / se_n if se_n > 0 else np.nan,
                se_infl_cell=se_c / se_n if se_n > 0 else np.nan,
            )
        )
    se_df = pd.DataFrame(se_rows)

    # Levene by factor on residual magnitudes (absolute residual as spread proxy
    # for factor-level homogeneity of residual scale)
    work = work.copy()
    work["resid"] = m.resid.to_numpy()
    lev_rows = []
    for fac in FACTORS:
        groups = [
            work.loc[work[fac] == lv, "resid"].to_numpy() for lv in sorted(work[fac].unique())
        ]
        W, p = _levene_groups(groups)
        lev_rows.append(
            dict(
                metric=metric,
                factor=fac,
                levene_W=W,
                p=p,
                reject_homo_alpha=bool(np.isfinite(p) and p < ALPHA),
                alpha=ALPHA,
                n_levels=int(work[fac].nunique()),
            )
        )
    lev_df = pd.DataFrame(lev_rows)

    # Cell residual variance regression: log s²_k ~ z_k (cell means of z)
    cell = (
        work.groupby("cell", sort=False)
        .agg(
            s2_resid=("resid", lambda s: float(np.var(s, ddof=1))),
            **{c: (c, "first") for c in FACTORS},
            **{z: (z, "first") for z in ZCOLS},
        )
        .reset_index()
    )
    cell = cell[cell["s2_resid"] > 0].copy()
    cell["log_s2"] = np.log(cell["s2_resid"])
    vform = f"log_s2 ~ {formula_rhs_main()}"
    mv = smf.ols(vform, data=cell).fit()
    var_rows = []
    for t in ["Intercept"] + ZCOLS:
        var_rows.append(
            dict(
                metric=metric,
                term=t,
                factor=t.replace("_z", "") if t.endswith("_z") else t,
                gamma=float(mv.params[t]),
                se=float(mv.bse[t]),
                t=float(mv.tvalues[t]),
                p=float(mv.pvalues[t]),
            )
        )
    var_df = pd.DataFrame(var_rows)
    var_meta = dict(
        metric=metric,
        n_cells=int(len(cell)),
        r2_log_s2=float(mv.rsquared),
        note=(
            "Cell residual variance of Stage-1 OLS on ln(χ); "
            "not CosWM-whitened. Full Stage-3 after spatial whitening: "
            "Box mixed_model/*/stage3_hetero/."
        ),
    )

    return dict(
        bp=bp,
        se=se_df,
        levene=lev_df,
        variance_effects=var_df,
        variance_meta=var_meta,
    )


def build_summary_md(
    bp: pd.DataFrame,
    lev: pd.DataFrame,
    se: pd.DataFrame,
    var_eff: pd.DataFrame,
    var_meta: pd.DataFrame,
) -> str:
    lines = [
        "# Naive OLS vs heteroscedasticity-aware inference",
        "",
        "Contrasts classical (naive) OLS standard errors with cluster-robust "
        "SEs and documents factor-dependent residual variance on \\(Y = \\ln\\chi\\).",
        "",
        rf"- Design: **{N_CELLS}** cells × \(N_x = {N_NODES}\) × \(N_s = {N_SEEDS}\)",
        f"- Homoscedasticity tests at \\(\\alpha = {ALPHA}\\)",
        "- Variance regression uses **unwhitened** Stage-1 residuals "
        "(descriptor). Prefer CosWM whitening before Stage-3 variance in the "
        "full hierarchical pipeline (`mixed_model`).",
        "",
        "## Output files",
        "",
        "| File | Contents |",
        "|------|----------|",
        "| `breusch_pagan.csv` | BP LM / F tests of residual hetero vs design |",
        "| `levene_by_factor.csv` | Brown–Forsythe / Levene on residuals by factor |",
        "| `se_inflation.csv` | Naive vs seed- and cell-cluster SE |",
        "| `variance_effects.csv` | \\(\\hat\\gamma\\) from \\(\\log s^2_k \\sim \\mathbf{z}_k\\) |",
        r"| `variance_fit_metrics.csv` | Cell-level \(R^2\) of log-variance model |",
        "| `summary.md` | this file |",
        "",
        "## Notation",
        "",
        "**Homoscedastic OLS** assumes \\(\\mathrm{Var}(\\varepsilon) = "
        "\\sigma^2 I\\). **Cluster-robust** SEs allow arbitrary correlation "
        "within seed or cell clusters. **Heteroscedasticity** means "
        "\\(\\mathrm{Var}(\\varepsilon\\mid \\mathbf{x})\\) depends on design.",
        "",
        "Breusch–Pagan regresses squared OLS residuals on the design; reject "
        "homoscedasticity when LM \\(p < \\alpha\\).",
        "",
        "Cell variance model (descriptor):",
        "",
        r"$$",
        r"\log s^2_k = \mathbf{z}_k^\top\boldsymbol{\gamma} + u_k,",
        r"$$",
        "",
        r"with \(s^2_k\) the sample variance of Stage-1 residuals in cell \(k\). "
        "Fit variance **after** absorbing spatial structure when doing "
        "formal Stage-3 inference.",
        "",
        "## Breusch–Pagan",
        "",
        r"| Metric | LM | \(p\) | Reject homo? |",
        "|--------|---:|------:|:------------:|",
    ]
    for _, r in bp.iterrows():
        lines.append(
            f"| {r['metric']} | {fmt(r['lm_stat'], 1)} | "
            f"{r['lm_p']:.2e} | {'yes' if r['reject_homo_alpha'] else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## SE inflation (median over main effects)",
            "",
            "| Metric | Median infl. (seed) | Median infl. (cell) | Max infl. (seed) |",
            "|--------|--------------------:|--------------------:|-----------------:|",
        ]
    )
    main = se[se["term"].isin(ZCOLS)]
    for metric in METRICS:
        sub = main[main["metric"] == metric]
        lines.append(
            f"| {metric} | {fmt(sub['se_infl_seed'].median())} | "
            f"{fmt(sub['se_infl_cell'].median())} | "
            f"{fmt(sub['se_infl_seed'].max())} |"
        )

    lines.extend(
        [
            "",
            "## Variance model \\(R^2\\) (\\(\\log s^2_k\\))",
            "",
            "| Metric | \\(n_{\\mathrm{cells}}\\) | \\(R^2\\) |",
            "|--------|------------------------:|--------:|",
        ]
    )
    for _, r in var_meta.iterrows():
        lines.append(f"| {r['metric']} | {int(r['n_cells'])} | {fmt(r['r2_log_s2'])} |")

    lines.extend(["", "## Conclusions", ""])
    n_rej = int(bp["reject_homo_alpha"].sum())
    lines.append(
        f"- Breusch–Pagan rejects homoscedasticity for **{n_rej}/{len(bp)}** "
        f"metrics at \\(\\alpha = {ALPHA}\\)."
    )
    for metric in METRICS:
        sub = main[main["metric"] == metric]
        vm = var_meta[var_meta["metric"] == metric].iloc[0]
        top = (
            var_eff[(var_eff["metric"] == metric) & (var_eff["term"].isin(ZCOLS))]
            .assign(abs_g=lambda d: d["gamma"].abs())
            .sort_values("abs_g", ascending=False)
            .iloc[0]
        )
        lines.append(
            f"- **{metric}**: median seed-cluster SE inflation "
            f"{fmt(sub['se_infl_seed'].median())}×; "
            rf"log-variance \(R^2 = {fmt(vm['r2_log_s2'])}\); "
            f"largest |γ| factor **{top['factor']}** ({fmt(top['gamma'])})."
        )
    lines.extend(
        [
            "",
            "Practical difference vs naive OLS: point estimates \\(\\hat\\beta\\) "
            "are unchanged under a correct mean, but naive SEs and constant-width "
            "intervals are misleading under clustering and heteroscedasticity. "
            "Report cluster-robust SEs for mean inference and a variance model "
            "(ideally after CosWM whitening) for spread.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    print("Loading join_master.h5 …")
    df = add_design_columns(load_ratios())

    bp_rows: list[dict] = []
    lev_parts: list[pd.DataFrame] = []
    se_parts: list[pd.DataFrame] = []
    var_parts: list[pd.DataFrame] = []
    var_meta_rows: list[dict] = []

    for metric in METRICS:
        print(f"Assessing {metric} …")
        res = assess_metric(df, metric)
        bp_rows.append(res["bp"])
        lev_parts.append(res["levene"])
        se_parts.append(res["se"])
        var_parts.append(res["variance_effects"])
        var_meta_rows.append(res["variance_meta"])
        print(f"  BP p={res['bp']['lm_p']:.2e}, var R²={res['variance_meta']['r2_log_s2']:.4f}")

    bp_df = pd.DataFrame(bp_rows)
    lev_df = pd.concat(lev_parts, ignore_index=True)
    se_df = pd.concat(se_parts, ignore_index=True)
    var_df = pd.concat(var_parts, ignore_index=True)
    var_meta_df = pd.DataFrame(var_meta_rows)

    dest = out_dir("naive_vs_hetero")
    bp_df.to_csv(dest / "breusch_pagan.csv", index=False)
    lev_df.to_csv(dest / "levene_by_factor.csv", index=False)
    se_df.to_csv(dest / "se_inflation.csv", index=False)
    var_df.to_csv(dest / "variance_effects.csv", index=False)
    var_meta_df.to_csv(dest / "variance_fit_metrics.csv", index=False)
    (dest / "summary.md").write_text(
        build_summary_md(bp_df, lev_df, se_df, var_df, var_meta_df),
        encoding="utf-8",
    )
    print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
