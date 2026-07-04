"""Seed adequacy figure.

Shows Monte Carlo convergence and effect resolution at n=100 seeds.
Produces: seed_adequacy.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from seiskit.plot_config import apply_style, panel_letter, result_path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_channel50, FACTORS

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
d50 = load_channel50()
d = d50.copy()
d["cell"] = d.groupby(FACTORS).ngroup()
S = 100
rng = np.random.default_rng(0)

ns_grid = [10, 20, 30, 50, 75, 100]
B = 500


def conv_curve(vals, stat_fn):
    out = []
    for n in ns_grid:
        est = np.array([stat_fn(rng.choice(vals, size=n, replace=True)) for _ in range(B)])
        out.append(est.std())
    return np.array(out)


conv = {}
for tgt in ["log_abs", "f_ratio"]:
    gv = d.groupby("cell")[tgt]
    cell_std = gv.std()
    hi = cell_std.idxmax()
    md = (cell_std - cell_std.median()).abs().idxmin()
    for tag, cid in [("high_var", hi), ("median_var", md)]:
        vals = d.loc[d.cell == cid, tgt].values
        conv[(tgt, tag, "mean")] = conv_curve(vals, np.mean)
        conv[(tgt, tag, "q95")] = conv_curve(vals, lambda x: np.quantile(x, 0.95))
        conv[(tgt, tag, "q05")] = conv_curve(vals, lambda x: np.quantile(x, 0.05))

# Compute required seeds for given relative-error targets
_req_rows = []
for tgt in ["log_abs", "f_ratio"]:
    g = d.groupby("cell")[tgt]
    cv = (g.std() / g.mean().abs()).values
    for rel_err in [0.05, 0.10]:
        n_req = np.ceil((cv / rel_err) ** 2)
        _req_rows.append(
            dict(
                target=tgt,
                target_rel_err=rel_err,
                n_req_median=int(np.median(n_req)),
                n_req_p90=int(np.percentile(n_req, 90)),
            )
        )
req = pd.DataFrame(_req_rows)


def contrast_se(tgt, factor, block_seed):
    lv = sorted(d[factor].unique())
    lo, hi = lv[0], lv[-1]
    if block_seed:
        per = d.groupby(["seed", factor])[tgt].mean().unstack()
        diff = per[hi] - per[lo]
        return diff.mean(), diff.std() / np.sqrt(len(diff))
    else:
        a = d[d[factor] == hi][tgt]
        b = d[d[factor] == lo][tgt]
        return a.mean() - b.mean(), np.sqrt(a.var() / len(a) + b.var() / len(b))


rows = []
for tgt in ["log_abs", "f_ratio"]:
    for f in ["Vs1", "Height", "CoV", "aHV"]:
        eb, seb = contrast_se(tgt, f, True)
        eu, seu = contrast_se(tgt, f, False)
        rows.append(
            dict(
                target=tgt,
                factor=f,
                effect=round(eb, 4),
                se_seedblocked=round(seb, 5),
                se_naive_pooled=round(seu, 5),
                ratio=round(seu / seb, 2),
            )
        )
contrast_df = pd.DataFrame(rows)
contrast_df["t_stat"] = (contrast_df["effect"].abs() / contrast_df["se_seedblocked"]).round(1)
contrast_df["n_seeds_for_t2"] = np.ceil(100 * (2 / contrast_df["t_stat"]) ** 2).astype(int)

tcol = {"log_abs": "#C44E52", "f_ratio": "#4C72B0"}
nice = {"log_abs": "log(abs_TF)", "f_ratio": "f_ratio"}

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
apply_style(auto_format=True, font_size=10, frame="open")
fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# (a) bootstrap convergence of estimators (worst-case high-var cell)
ax = axes[0, 0]
ls = {"mean": "-", "q05": "--", "q95": ":"}
for tgt in ["log_abs", "f_ratio"]:
    for stat in ["mean", "q05", "q95"]:
        c = conv[(tgt, "high_var", stat)]
        ax.plot(
            ns_grid,
            c,
            ls[stat],
            color=tcol[tgt],
            lw=1.8,
            label=f"{nice[tgt]} {stat}" if stat != "q05" else None,
        )
ref = conv[("log_abs", "high_var", "mean")][0] * np.sqrt(10 / np.array(ns_grid))
ax.plot(ns_grid, ref, "k-", lw=0.8, alpha=0.5, label=r"ideal $1/\sqrt{n}$")
ax.axvline(100, color="0.6", lw=0.8, ls=":")
ax.set_xlabel("number of seeds n")
ax.set_ylabel("bootstrap SE of estimator")
ax.set_title("(worst-case high-variance cell)\nestimator SE shrinks as 1/√n", fontsize=10)
ax.legend(fontsize=6.5, frameon=False, ncol=2)
panel_letter(ax, "a")

# (b) distribution of relative MCSE of the mean across 243 cells at n=100
ax = axes[0, 1]
for tgt in ["log_abs", "f_ratio"]:
    g = d.groupby("cell")[tgt]
    rel = (g.std() / g.mean().abs() / np.sqrt(100)).values * 100
    ax.hist(
        rel, bins=40, alpha=0.6, color=tcol[tgt], label=f"{nice[tgt]} (med {np.median(rel):.1f}%)"
    )
ax.axvline(5, color="0.3", ls="--", lw=1, label="5% target")
ax.set_xlabel("relative Monte Carlo SE of cell mean (%)")
ax.set_ylabel("# design cells")
ax.set_xlim(0, 20)
ax.set_title("Precision of the cell mean at n=100", fontsize=10)
ax.legend(fontsize=7, frameon=False)
panel_letter(ax, "b")

# (c) required n for +/-5% and +/-10% (median & p90 cell)
ax = axes[1, 0]
x = np.arange(2)
w = 0.35
for i, tgt in enumerate(["log_abs", "f_ratio"]):
    r5 = req[(req.target == tgt) & (req.target_rel_err == 0.05)].iloc[0]
    r10 = req[(req.target == tgt) & (req.target_rel_err == 0.10)].iloc[0]
    ax.bar(i - w / 2, [r5.n_req_median], w, color=tcol[tgt], alpha=0.9)
    ax.bar(i + w / 2, [r10.n_req_median], w, color=tcol[tgt], alpha=0.5)
    ax.text(i - w / 2, r5.n_req_median + 2, f"{r5.n_req_median}", ha="center", fontsize=7)
    ax.text(i + w / 2, r10.n_req_median + 2, f"{r10.n_req_median}", ha="center", fontsize=7)
ax.axhline(100, color="green", ls="--", lw=1.2, label="n=100 (used)")
ax.set_xticks(x)
ax.set_xticklabels([nice[t] for t in ["log_abs", "f_ratio"]])
ax.set_ylabel("seeds required (median cell)")
ax.set_title("Seeds needed for ±5% (dark) / ±10% (light)\nmean precision", fontsize=10)
ax.legend(fontsize=7, frameon=False)
panel_letter(ax, "c")

# (d) effect resolution t-stats at n=100
ax = axes[1, 1]
cd = contrast_df.copy()
cd["lab"] = cd.target.map(nice) + "\n" + cd.factor
order = cd.sort_values("t_stat")
colors = [tcol[t] for t in order.target]
ax.barh(range(len(order)), order.t_stat, color=colors)
ax.axvline(2, color="0.3", ls="--", lw=1.2, label="t=2 (significance)")
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order.lab, fontsize=6.5)
ax.set_xlabel("|effect| / SE  (100 seeds as blocks)")
ax.set_title("Every factor effect resolved with\nwide margin at n=100", fontsize=10)
ax.legend(fontsize=7, frameon=False)
panel_letter(ax, "d")

fig.suptitle(
    "Is n=100 seeds enough? Monte Carlo convergence & effect resolution (channel 50)",
    fontsize=12,
    y=1.005,
)
fig.tight_layout()
fig.savefig(result_path("plots", "seed_adequacy.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved seed_adequacy.png")
