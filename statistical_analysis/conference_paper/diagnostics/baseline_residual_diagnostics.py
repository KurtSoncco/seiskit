# env: python
"""Baseline residual diagnostics — seiskit conference paper diagnostics.

Generates a 2x3 panel covering residual QQ plots, SE inflation from
seed clustering, variance decomposition, coefficient forest plot,
and predicted vs actual for channel 50 baseline models.
"""

import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf

from seiskit.plot_config import apply_style, panel_letter, result_path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_channel50, FACTORS

d50 = load_channel50()

# Standardize factors
Z = d50[FACTORS].copy()
Zs = (Z - Z.mean()) / Z.std()
Zs.columns = [c + "_z" for c in FACTORS]
d50 = pd.concat([d50, Zs], axis=1)

zcols = [c + "_z" for c in FACTORS]

# OLS with clustered SEs
main = " + ".join(zcols)
inter = " + ".join([f"{a}:{b}" for i, a in enumerate(zcols) for b in zcols[i + 1 :]])
formula_rhs = main + " + " + inter


def fit_ols_cluster(target):
    f = f"{target} ~ {formula_rhs}"
    m = smf.ols(f, data=d50).fit()
    mc = smf.ols(f, data=d50).fit(cov_type="cluster", cov_kwds={"groups": d50["seed"]})
    return m, mc


results = {}
for tgt in ["log_abs", "f_ratio"]:
    m, mc = fit_ols_cluster(tgt)
    results[tgt] = (m, mc)

# MixedLM
mm_full = smf.mixedlm(f"log_abs ~ {formula_rhs}", d50, groups=d50["seed"]).fit(
    method="powell", reml=True
)
vseed = float(mm_full.cov_re.iloc[0, 0])
vres = float(mm_full.scale)
icc_final = vseed / (vseed + vres)
Xb = np.asarray(mm_full.predict(d50))
var_fixed = np.var(Xb)
R2_marg = var_fixed / (var_fixed + vseed + vres)
R2_cond = (var_fixed + vseed) / (var_fixed + vseed + vres)

apply_style(auto_format=True, font_size=10, frame="open")
fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))

m_log = results["log_abs"][0]
m_f = results["f_ratio"][0]

# a: OLS residual QQ (log_abs) - should be near normal
ax = axes[0, 0]
(osm, osr), (sl, ic, rr) = stats.probplot(m_log.resid.values, dist="norm")
idx = np.linspace(0, len(osm) - 1, 3000).astype(int)
ax.plot(osm[idx], osr[idx], "o", ms=2, color="#C44E52", alpha=0.4)
ax.plot([osm.min(), osm.max()], sl * np.array([osm.min(), osm.max()]) + ic, "-", color="0.2", lw=1)
ax.set_title("log(abs TF): OLS residual QQ — near normal", loc="left")
ax.set_xlabel("theoretical q")
ax.set_ylabel("sample q")
panel_letter(ax, "a")

# b: OLS residual QQ (f_ratio) - heavy tails
ax = axes[0, 1]
(osm, osr), (sl, ic, rr) = stats.probplot(m_f.resid.values, dist="norm")
ax.plot(osm[idx], osr[idx], "o", ms=2, color="#4C72B0", alpha=0.4)
ax.plot([osm.min(), osm.max()], sl * np.array([osm.min(), osm.max()]) + ic, "-", color="0.2", lw=1)
ax.set_title("$f$ ratio: OLS residual QQ — heavy tails remain", loc="left")
ax.set_xlabel("theoretical q")
ax.set_ylabel("sample q")
panel_letter(ax, "b")

# c: SE inflation naive vs clustered (main effects, both targets)
ax = axes[0, 2]
terms = zcols
x = np.arange(len(terms))
w = 0.35
infl_log = [results["log_abs"][1].bse[t] / results["log_abs"][0].bse[t] for t in terms]
infl_f = [results["f_ratio"][1].bse[t] / results["f_ratio"][0].bse[t] for t in terms]
ax.bar(x - w / 2, infl_log, w, color="#C44E52", label="log(abs TF)")
ax.bar(x + w / 2, infl_f, w, color="#4C72B0", label="$f$ ratio")
ax.axhline(1, color="0.3", ls="--", lw=1)
ax.set_xticks(x)
ax.set_xticklabels(FACTORS, rotation=30, fontsize=7)
ax.set_ylabel("SE inflation (clustered / naive)")
ax.set_title("Seed clustering inflates SEs up to ~9×", loc="left")
ax.legend(fontsize=7, frameon=False)
panel_letter(ax, "c")

# d: variance components pie-ish stacked (marginal vs seed vs resid) for log_abs
ax = axes[1, 0]
comps = [var_fixed, vseed, vres]
labs = ["fixed factors", "seed (random)", "residual"]
cols = ["#555555", "#C44E52", "#B0B0B0"]
ax.bar([0], [comps[0]], color=cols[0], label=labs[0])
ax.bar([0], [comps[1]], bottom=comps[0], color=cols[1], label=labs[1])
ax.bar([0], [comps[2]], bottom=comps[0] + comps[1], color=cols[2], label=labs[2])
tot = sum(comps)
ax.text(0, comps[0] / 2, f"{comps[0] / tot * 100:.0f}%", ha="center", color="white", fontsize=8)
ax.text(
    0,
    comps[0] + comps[1] / 2,
    f"{comps[1] / tot * 100:.0f}%",
    ha="center",
    color="white",
    fontsize=8,
)
ax.text(
    0, comps[0] + comps[1] + comps[2] / 2, f"{comps[2] / tot * 100:.0f}%", ha="center", fontsize=8
)
ax.set_xticks([])
ax.set_ylabel("variance")
ax.set_xlim(-0.6, 1.4)
ax.set_title("log(abs TF): variance partition", loc="left")
ax.legend(fontsize=7, frameon=False, loc="upper right")
panel_letter(ax, "d")

# e: coefficient forest plot (mixed model, log_abs, main + top interactions)
ax = axes[1, 1]
terms_plot = zcols + [t for t in mm_full.params.index if ":" in t]
coefs = [mm_full.params[t] for t in terms_plot]
ses = [mm_full.bse[t] for t in terms_plot]
order = np.argsort(coefs)
yy = np.arange(len(terms_plot))
lbl_map = {c + "_z": c for c in FACTORS}


def nice(t):
    return " × ".join(lbl_map.get(p, p.replace("_z", "")) for p in t.split(":"))


ax.errorbar(
    [coefs[i] for i in order],
    yy,
    xerr=[1.96 * ses[i] for i in order],
    fmt="o",
    ms=4,
    color="#C44E52",
    ecolor="0.5",
    capsize=2,
)
ax.axvline(0, color="0.3", ls="--", lw=1)
ax.set_yticks(yy)
ax.set_yticklabels([nice(terms_plot[i]) for i in order], fontsize=5.5)
ax.set_xlabel("standardized coefficient")
ax.set_title("log(abs TF): mixed-model effects (95% CI)", loc="left")
panel_letter(ax, "e")

# f: predicted vs actual (mixed model, log_abs)
ax = axes[1, 2]
pred = Xb
act = d50["log_abs"].values
idx2 = np.linspace(0, len(pred) - 1, 4000).astype(int)
ax.plot(pred[idx2], act[idx2], "o", ms=2, color="#C44E52", alpha=0.25)
lo, hi = act.min(), act.max()
ax.plot([lo, hi], [lo, hi], "-", color="0.2", lw=1)
ax.set_xlabel("predicted (fixed effects)")
ax.set_ylabel("actual log(abs TF)")
ax.set_title(f"Fixed-effect fit: marginal $R^2$={R2_marg:.2f}", loc="left")
panel_letter(ax, "f")

fig.suptitle(
    "Baseline linear models (channel 50): log(abs TF) → hierarchical w/ seed RI; $f$ ratio → non-normal residuals persist",
    fontsize=10,
    y=0.99,
)
fig.tight_layout()
fig.savefig(result_path("plots", "baseline_residual_diagnostics.png"), dpi=150, bbox_inches="tight")
print("saved baseline_residual_diagnostics.png")
