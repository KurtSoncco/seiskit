# env: python
"""Normality assessment via QQ plots — seiskit conference paper diagnostics.

Generates a 2x3 panel comparing raw vs log vs Box-Cox transformed QQ plots
for both target variables at channel 50.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import boxcox, boxcox_normmax

from seiskit.plot_config import apply_style, panel_letter, result_path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_channel50, FACTORS

d50 = load_channel50()

d50 = d50.copy()
d50["abs_dev"] = d50["abs_TF_ratio"] - d50.groupby(FACTORS)["abs_TF_ratio"].transform("mean")
d50["logabs_dev"] = np.log(d50["abs_TF_ratio"]) - d50.groupby(FACTORS)["abs_TF_ratio"].transform(
    lambda s: np.log(s).mean()
)
d50["f_dev"] = d50["f_ratio"] - d50.groupby(FACTORS)["f_ratio"].transform("mean")

lam_abs = boxcox_normmax(d50["abs_TF_ratio"].values)
lam_f = boxcox_normmax(d50["f_ratio"].values)


def bc(x, lam):
    return boxcox(x, lmbda=lam)


abs_bc = bc(d50["abs_TF_ratio"].values, lam_abs)
f_bc = bc(d50["f_ratio"].values, lam_f)

d50["abs_bc"] = abs_bc
d50["f_bc"] = f_bc

apply_style(auto_format=True, font_size=10, frame="open")
fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))


def qq(ax, x, title, color):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    (osm, osr), (sl, ic, r) = stats.probplot(x, dist="norm")
    idx = np.linspace(0, len(osm) - 1, min(3000, len(osm))).astype(int)
    ax.plot(osm[idx], osr[idx], "o", ms=2, color=color, alpha=0.4)
    lim = [osm.min(), osm.max()]
    ax.plot(lim, sl * np.array(lim) + ic, "-", color="0.2", lw=1)
    ax.set_title(title, loc="left", fontsize=8.5)
    ax.set_xlabel("theoretical quantiles")
    ax.set_ylabel("sample quantiles")
    ax.text(
        0.05,
        0.92,
        f"skew {stats.skew(x):.2f}\nkurt {stats.kurtosis(x):.2f}",
        transform=ax.transAxes,
        va="top",
        fontsize=6.5,
        bbox=dict(fc="white", ec="0.7", alpha=0.8, boxstyle="round,pad=0.3"),
    )


qq(axes[0, 0], d50["abs_TF_ratio"], "abs TF ratio — raw", "#C44E52")
qq(axes[0, 1], np.log(d50["abs_TF_ratio"]), "abs TF ratio — log", "#C44E52")
qq(
    axes[0, 2],
    d50["abs_TF_ratio"] ** 0 + bc(d50["abs_TF_ratio"].values, lam_abs),
    f"abs TF ratio — Box-Cox λ={lam_abs:.2f}",
    "#C44E52",
)
qq(axes[1, 0], d50["f_ratio"], "$f$ ratio — raw", "#4C72B0")
qq(axes[1, 1], np.log(d50["f_ratio"]), "$f$ ratio — log", "#4C72B0")
qq(axes[1, 2], bc(d50["f_ratio"].values, lam_f), f"$f$ ratio — Box-Cox λ={lam_f:.2f}", "#4C72B0")

for l, ax in zip("abcdef", axes.ravel()):
    panel_letter(ax, l)
fig.suptitle(
    "Normal QQ plots: raw vs transformed targets (channel 50)  ·  points on the line = normal",
    fontsize=10,
    y=1.0,
)
fig.tight_layout()
fig.savefig(result_path("plots", "normality_assessment.png"), dpi=150, bbox_inches="tight")
