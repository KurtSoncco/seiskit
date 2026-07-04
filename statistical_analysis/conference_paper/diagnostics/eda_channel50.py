# env: python
"""EDA for channel 50 (center recorder) — seiskit conference paper diagnostics.

Generates a 3x3 panel figure covering distributions, factor effects,
seed structure, and interaction heatmap for the center recorder.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from seiskit.plot_config import apply_style, panel_letter, result_path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_channel50, FACTORS

d50 = load_channel50()

apply_style(auto_format=True, font_size=10, frame="open")

fig = plt.figure(figsize=(13, 11))
gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.38)

ax = fig.add_subplot(gs[0, 0])
ax.hist(d50["f_ratio"], bins=60, color="#4C72B0", edgecolor="none")
ax.axvline(1.0, color="0.3", ls="--", lw=1)
ax.set_xlabel("$f$ ratio")
ax.set_ylabel("count")
ax.set_title("$f$ ratio — near 1, right tail", loc="left")
panel_letter(ax, "a")

ax = fig.add_subplot(gs[0, 1])
ax.hist(d50["abs_TF_ratio"], bins=80, color="#C44E52", edgecolor="none")
ax.axvline(1.0, color="0.3", ls="--", lw=1)
ax.set_xlabel("abs TF ratio")
ax.set_ylabel("count")
ax.set_title("abs TF ratio — strong right skew", loc="left")
panel_letter(ax, "b")

ax = fig.add_subplot(gs[0, 2])
ax.hist(np.log10(d50["abs_TF_ratio"]), bins=70, color="#C44E52", edgecolor="none", alpha=0.85)
ax.set_xlabel(r"$\log_{10}$(abs TF ratio)")
ax.set_ylabel("count")
ax.set_title("log abs TF ratio — near symmetric", loc="left")
panel_letter(ax, "c")

for j, (tgt, col, ttl) in enumerate(
    [("f_ratio", "#4C72B0", "$f$ ratio"), ("abs_TF_ratio", "#C44E52", "abs TF ratio")]
):
    ax = fig.add_subplot(gs[1, j])
    xpos = 0
    xticks = []
    xlabs = []
    for fc in FACTORS:
        g = d50.groupby(fc)[tgt].mean()
        xs = np.arange(len(g)) + xpos
        ax.plot(xs, g.values, "o-", color=col, ms=5)
        for x, lv in zip(xs, g.index):
            xticks.append(x)
            xlabs.append(f"{lv:g}")
        ax.text(np.mean(xs), ax.get_ylim()[0], "", ha="center")
        xpos += len(g) + 0.8
        ax.axvline(xpos - 1.4, color="0.85", lw=0.8)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabs, fontsize=6)
    ax.set_ylabel(f"mean {ttl}")
    ax.set_title(f"{ttl}: marginal means (Vs1·H·CoV·rH·aHV)", loc="left")
    panel_letter(ax, "de"[j])

ax = fig.add_subplot(gs[1, 2])
xpos = 0
xticks = []
xlabs = []
for fc in FACTORS:
    g = d50.groupby(fc)["f_ratio"].std()
    xs = np.arange(len(g)) + xpos
    ax.plot(xs, g.values, "s-", color="#55A868", ms=5)
    for x, lv in zip(xs, g.index):
        xticks.append(x)
        xlabs.append(f"{lv:g}")
    xpos += len(g) + 0.8
    ax.axvline(xpos - 1.4, color="0.85", lw=0.8)
ax.set_xticks(xticks)
ax.set_xticklabels(xlabs, fontsize=6)
ax.set_ylabel("std of $f$ ratio")
ax.set_title("$f$ ratio: spread — CoV drives it", loc="left")
panel_letter(ax, "f")

ax = fig.add_subplot(gs[2, 0])
cell = d50[
    (d50.Vs1 == 230) & (d50.Height == 50) & (d50.CoV == 0.2) & (d50.rH == 30) & (d50.aHV == 10)
]
ax.plot(cell["seed"], cell["abs_TF_ratio"], "o", ms=3, color="#8172B3", alpha=0.7)
ax.axhline(cell["abs_TF_ratio"].mean(), color="0.3", ls="--", lw=1)
ax.set_xlabel("seed")
ax.set_ylabel("abs TF ratio")
ax.set_title("Seed scatter, one design cell", loc="left")
panel_letter(ax, "g")

ax = fig.add_subplot(gs[2, 1])
sm = d50.groupby("seed")["abs_TF_ratio"].mean()
ax.plot(sm.index, sm.values, "o", ms=3, color="#8172B3", alpha=0.7)
ax.axhline(sm.mean(), color="0.3", ls="--", lw=1, label=f"grand mean {sm.mean():.2f}")
ax.set_xlabel("seed")
ax.set_ylabel("mean abs TF ratio")
ax.set_title(f"Per-seed mean varies 0.08–0.54 (not flat)", loc="left")
ax.legend(loc="upper right", fontsize=6)
panel_letter(ax, "h")

ax = fig.add_subplot(gs[2, 2])
piv = d50.groupby(["Height", "Vs1"])["abs_TF_ratio"].mean().unstack()
im = ax.imshow(piv.values, cmap="viridis", aspect="auto", origin="lower")
ax.set_xticks(range(3))
ax.set_xticklabels([f"{v:g}" for v in piv.columns])
ax.set_yticks(range(3))
ax.set_yticklabels([f"{v:g}" for v in piv.index])
ax.set_xlabel("$V_{s1}$ (m/s)")
ax.set_ylabel("H (m)")
for i in range(3):
    for k in range(3):
        v = piv.values[i, k]
        ax.text(
            k,
            i,
            f"{v:.2f}",
            ha="center",
            va="center",
            color="white" if v < 0.3 else "black",
            fontsize=7,
        )
ax.set_title("mean abs TF: $V_{s1}$ × H", loc="left")
panel_letter(ax, "i")

fig.suptitle(
    "Channel 50 (center recorder): distributions, factor effects, seed structure  ·  n=24,300 (243 cells × 100 seeds)",
    fontsize=10,
    y=0.985,
)
fig.savefig(result_path("plots", "eda_channel50.png"), dpi=150, bbox_inches="tight")
print("re-saved eda_channel50.png; panel h fixed")
