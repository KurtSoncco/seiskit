"""Seed independence diagnostic figure.

Shows that seeds are NOT i.i.d. — a random-field realization effect is present.
Produces: seed_independence.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from seiskit.plot_config import apply_style, get_crameri_cmap, panel_letter, result_path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FACTORS, load_channel50, target_color, REF_COLOR, FIG_DPI

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
d50 = load_channel50()
d50["cell"] = d50.groupby(FACTORS).ngroup()
d50["log_abs_resid"] = d50["log_abs"] - d50.groupby(FACTORS)["log_abs"].transform("mean")
d50["f_resid"] = d50["f_ratio"] - d50.groupby(FACTORS)["f_ratio"].transform("mean")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
apply_style(auto_format=True, font_size=10, frame="open")
fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

# a: variance components stacked bar (between-seed vs within)
ax = axes[0, 0]
labels = ["log(abs_TF)", "abs_TF", "f_ratio"]
vb = [0.1604, 0.0126, 0.0009]
vw = [0.2764, 0.0540, 0.0044]
iccs = [0.367, 0.190, 0.167]
x = np.arange(3)
ax.bar(x, vb, color=target_color("log_abs"), label="between-seed")
ax.bar(x, vw, bottom=vb, color=REF_COLOR, label="within-seed (residual)")
for i, ic in enumerate(iccs):
    ax.text(i, vb[i] + vw[i] + 0.005, f"ICC={ic:.2f}", ha="center", fontsize=8, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("variance component")
ax.legend(fontsize=7, frameon=False)
ax.set_title("Variance decomposition (design effect removed)", loc="left")
panel_letter(ax, "a")

# b: per-seed residual mean distribution (log_abs)
ax = axes[0, 1]
sm = d50.groupby("seed")["log_abs_resid"].mean().sort_values()
ax.plot(np.arange(len(sm)), sm.values, "o", ms=4, color=target_color("log_abs"), alpha=0.8)
ax.axhline(0, color="0.3", ls="--", lw=1)
ax.fill_between([0, 99], -sm.std(), sm.std(), color="0.85", alpha=0.5, zorder=0)
ax.set_xlabel("seed (rank-ordered)")
ax.set_ylabel("mean within-cell residual")
ax.set_title("Per-seed residual means: systematic, not zero", loc="left")
ax.text(
    0.05,
    0.92,
    f"between-seed std={sm.std():.2f}\n90× permutation null",
    transform=ax.transAxes,
    va="top",
    fontsize=7,
    bbox=dict(fc="white", ec="0.7", boxstyle="round,pad=0.3"),
)
panel_letter(ax, "b")

# c: seed-aligned cell-cell correlation heatmap
ax = axes[1, 0]
resid = d50["log_abs"] - d50.groupby(FACTORS)["log_abs"].transform("mean")
M = resid.groupby([d50["seed"], d50["cell"]]).mean().unstack()
sub = M.values[:, ::8]
corr = np.corrcoef(sub.T)
im = ax.imshow(corr, cmap=get_crameri_cmap("vik"), vmin=-1, vmax=1, aspect="auto")
ax.set_xlabel("design cell (subsampled)")
ax.set_ylabel("design cell (subsampled)")
ax.set_title(
    f"Seed-aligned cell–cell correlation (mean {np.corrcoef(M.values.T)[np.triu_indices(243, 1)].mean():.2f})",
    loc="left",
)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("correlation")
panel_letter(ax, "c")

# d: design-effect / SE inflation
m = 243
ax = axes[1, 1]
ax.bar(x, [1 + (m - 1) * ic for ic in iccs], color=target_color("f_ratio"))
for i, ic in enumerate(iccs):
    d_ = 1 + (m - 1) * ic
    ax.text(i, d_ + 1.5, f"{d_:.0f}\n({d_**0.5:.0f}× SE)", ha="center", fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("design effect (DEFF)")
ax.set_title("Ignoring seeds understates SE by √DEFF", loc="left")
ax.set_ylim(0, 105)
panel_letter(ax, "d")

fig.suptitle(
    "Seed independence diagnostic: seeds are NOT i.i.d. — a random-field realization effect is present",
    fontsize=10,
    y=0.98,
)
fig.tight_layout()
fig.savefig(result_path("plots", "seed_independence.png"), dpi=FIG_DPI, bbox_inches="tight")
print("saved seed_independence.png")
