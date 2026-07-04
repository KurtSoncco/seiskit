# env: python
"""Heteroscedasticity diagnostics — seiskit conference paper diagnostics.

Generates a 2x3 panel covering residual funnels, mean-variance coupling,
eta-squared decompositions, and per-factor spread for channel 50.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import levene, pearsonr
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm

from seiskit.plot_config import apply_style, panel_letter, result_path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_channel50, FACTORS

d50 = load_channel50()

# Standardize factors
Z = d50[FACTORS].copy()
Zs = (Z - Z.mean()) / Z.std()
Zs.columns = [c + "_z" for c in FACTORS]
d50 = pd.concat([d50, Zs], axis=1)


# OLS models
def bp_test(target):
    X = sm.add_constant(d50[[c + "_z" for c in FACTORS]])
    y = d50[target]
    m = sm.OLS(y, X).fit()
    lm, lm_p, F, F_p = het_breuschpagan(m.resid, np.asarray(X))
    return m, lm, lm_p


ols_models = {}
for tgt in ["f_ratio", "abs_TF_ratio", "log_abs"]:
    m, lm, lmp = bp_test(tgt)
    ols_models[tgt] = m

# Levene tests
lev_rows = []
for tgt in ["f_ratio", "abs_TF_ratio"]:
    for fc in FACTORS:
        gs = [d50.loc[d50[fc] == lv, tgt].values for lv in sorted(d50[fc].unique())]
        W, p = levene(*gs, center="median")
        lev_rows.append(dict(target=tgt, factor=fc, levene_W=W, p=p))

# Mean-variance coupling
cell_stats = (
    d50.groupby(FACTORS)
    .agg(
        f_mean=("f_ratio", "mean"),
        f_std=("f_ratio", "std"),
        a_mean=("abs_TF_ratio", "mean"),
        a_std=("abs_TF_ratio", "std"),
    )
    .reset_index()
)


def eta2(dfin, val, fac):
    grand = dfin[val].mean()
    ss_tot = ((dfin[val] - grand) ** 2).sum()
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for _, g in dfin.groupby(fac)[val])
    return ss_b / ss_tot


eta_rows = []
for fc in FACTORS:
    em_a = eta2(cell_stats, "a_mean", fc)
    es_a = eta2(cell_stats, "a_std", fc)
    em_f = eta2(cell_stats, "f_mean", fc)
    es_f = eta2(cell_stats, "f_std", fc)
    eta_rows.append(
        dict(
            factor=fc, absTF_mean_eta2=em_a, absTF_std_eta2=es_a, f_mean_eta2=em_f, f_std_eta2=es_f
        )
    )

eta_df = pd.DataFrame(eta_rows)

# Plot
apply_style(auto_format=True, font_size=10, frame="open")
fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))

# a,b: residual funnel plots (residual vs fitted) for raw abs_TF and log_abs
for ax, tgt, ttl, col in [
    (axes[0, 0], "abs_TF_ratio", "abs TF ratio (raw)", "#C44E52"),
    (axes[0, 1], "log_abs", "log(abs TF ratio)", "#C44E52"),
]:
    m = ols_models[tgt]
    fit = m.fittedvalues.values
    res = m.resid.values
    idx = np.linspace(0, len(fit) - 1, 4000).astype(int)
    ax.plot(fit[idx], res[idx], "o", ms=2, color=col, alpha=0.3)
    ax.axhline(0, color="0.3", lw=1)
    ax.set_xlabel("OLS fitted value")
    ax.set_ylabel("residual")
    ax.set_title(f"{ttl}: residual funnel", loc="left")
panel_letter(axes[0, 0], "a")
panel_letter(axes[0, 1], "b")

# c: mean vs std scatter (design cells)
ax = axes[0, 2]
ax.plot(
    cell_stats.a_mean,
    cell_stats.a_std,
    "o",
    ms=4,
    color="#C44E52",
    alpha=0.6,
    label=f"abs TF (r={pearsonr(cell_stats.a_mean, cell_stats.a_std)[0]:.2f})",
)
ax.plot(
    cell_stats.f_mean,
    cell_stats.f_std,
    "s",
    ms=4,
    color="#4C72B0",
    alpha=0.6,
    label=f"$f$ ratio (r={pearsonr(cell_stats.f_mean, cell_stats.f_std)[0]:.2f})",
)
ax.set_xlabel("design-cell mean")
ax.set_ylabel("design-cell std")
ax.set_title("Mean–variance coupling (243 cells)", loc="left")
ax.legend(fontsize=7, frameon=False, loc="upper left")
panel_letter(ax, "c")

# d,e: eta^2 grouped bars — mean vs std driver decomposition
for ax, (mcol, scol, ttl, pl) in zip(
    [axes[1, 0], axes[1, 1]],
    [
        ("absTF_mean_eta2", "absTF_std_eta2", "abs TF ratio", "d"),
        ("f_mean_eta2", "f_std_eta2", "$f$ ratio", "e"),
    ],
):
    x = np.arange(5)
    w = 0.38
    ax.bar(x - w / 2, eta_df[mcol], w, color="#555555", label="explains MEAN")
    ax.bar(x + w / 2, eta_df[scol], w, color="#E1812C", label="explains STD")
    ax.set_xticks(x)
    ax.set_xticklabels(FACTORS, fontsize=7, rotation=30)
    ax.set_ylabel("$\\eta^2$ (variance explained)")
    ax.set_title(f"{ttl}: mean vs variance drivers", loc="left")
    ax.legend(fontsize=7, frameon=False)
    panel_letter(ax, pl)

# f: per-factor spread bars — std of f_ratio by CoV level (the pure-variance driver)
ax = axes[1, 2]
xpos = 0
colors = plt.cm.tab10(np.linspace(0, 1, 5))
for i, fc in enumerate(["CoV", "aHV", "Height"]):
    g = d50.groupby(fc)["f_ratio"].std()
    xs = np.arange(len(g)) + xpos
    ax.bar(xs, g.values, color=colors[i], label=fc, width=0.8)
    xpos += len(g) + 0.6
ax.set_ylabel("std of $f$ ratio")
ax.set_xticks([])
ax.set_title("$f$ ratio spread by level (CoV, aHV, H)", loc="left")
ax.legend(fontsize=7, frameon=False)
panel_letter(ax, "f")

fig.suptitle(
    "Heteroscedasticity diagnosis (channel 50): variance is factor-dependent, and mean-drivers ≠ variance-drivers",
    fontsize=10,
    y=0.99,
)
fig.tight_layout()
fig.savefig(
    result_path("plots", "heteroscedasticity_diagnostics.png"), dpi=150, bbox_inches="tight"
)
