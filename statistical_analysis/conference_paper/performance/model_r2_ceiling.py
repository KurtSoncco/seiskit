"""R² ceiling decomposition and calibrated heteroscedastic prediction intervals.

Uses pre-trained quantile models from models/ to avoid redundant training.
"""

import sys
import string
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score

from seiskit.plot_config import apply_style, panel_letter, result_path
from config import (
    load_channel50,
    FACTORS,
    load_quantile_models,
    seed_grouped_split,
)

warnings.filterwarnings("ignore")
apply_style(auto_format=True, font_size=10, frame="open")

d = load_channel50()
d["cell"] = d.groupby(FACTORS).ngroup()

# R² ceiling computation (fast — pure pandas)
rows = []
for tgt in ["log_abs", "f_ratio", "abs_TF_ratio"]:
    y = d[tgt].values
    grand = y.mean()
    cell_mean = d.groupby("cell")[tgt].transform("mean")
    ss_within = ((y - cell_mean) ** 2).sum()
    ss_total = ((y - grand) ** 2).sum()
    ss_between = ss_total - ss_within
    R2_ceiling = ss_between / ss_total
    rows.append(dict(target=tgt, R2_ceiling=R2_ceiling, frac_irreducible=ss_within / ss_total))
ceil = pd.DataFrame(rows)

recorded = {"log_abs": 0.358, "f_ratio": 0.208}

# Cell means for predictability (only 243 rows — very fast)
cm = d.groupby(FACTORS)[["log_abs", "f_ratio"]].mean().reset_index()

# Load pre-trained quantile models instead of re-fitting
tr, te = seed_grouped_split(d, test_size=0.25, seed=0)
Xte = d.iloc[te][FACTORS].values

qmodels = load_quantile_models(taus=[0.05, 0.25, 0.5, 0.75, 0.95])
qres = {}
for tgt in ["log_abs", "f_ratio"]:
    yte = d.iloc[te][tgt].values
    q = {
        tau: qmodels[tgt][tau].predict(Xte)
        for tau in [0.05, 0.25, 0.5, 0.75, 0.95]
        if tau in qmodels[tgt]
    }
    cov90 = np.mean((yte >= q[0.05]) & (yte <= q[0.95]))
    cov50 = np.mean((yte >= q[0.25]) & (yte <= q[0.75]))
    w = q[0.95] - q[0.05]
    qres[tgt] = dict(q=q, cov90=cov90, cov50=cov50, yte=yte, w=w)

C_log, C_f = "#C44E52", "#4C72B0"

fig, ax = plt.subplots(2, 3, figsize=(14, 8.8))

for i, (tgt, col) in enumerate([("log_abs", C_log), ("f_ratio", C_f)]):
    a = ax[i, 0]
    c = ceil.loc[ceil.target == tgt, "R2_ceiling"].values[0]
    rec = recorded[tgt]
    vals = [1.0, c, rec]
    alphas = [0.35, 0.6, 1.0]
    for xpos, v, al, cc in zip([0, 1, 2], vals, alphas, ["#999999", col, col]):
        a.bar(xpos, v, color=cc, alpha=al, edgecolor="k", linewidth=0.8)
    a.set_xticks([0, 1, 2])
    a.set_xticklabels(
        ["Total\nvariance", "Explainable\n(between-cell)", "GBM\ncaptured"], fontsize=9
    )
    a.set_ylim(0, 1.05)
    a.set_ylabel("Variance fraction")
    a.set_title(f"{tgt}: R² ceiling = {c:.2f}")
    a.text(
        2,
        rec + 0.03,
        f"{rec / c * 100:.0f}% of\nexplainable",
        ha="center",
        fontsize=8.5,
        color=col,
        fontweight="bold",
    )
    a.axhline(c, ls=":", color=col, lw=1)
    a.text(
        0,
        0.52,
        f"{(1 - c) * 100:.0f}%\nirreducible\nseed noise",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#555",
    )

for i, (tgt, col) in enumerate([("log_abs", C_log), ("f_ratio", C_f)]):
    a = ax[i, 1]
    X = cm[FACTORS].values
    y = cm[tgt].values
    m = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        random_state=0,
    )
    pred = cross_val_predict(m, X, y, cv=KFold(5, shuffle=True, random_state=0))
    r2 = r2_score(y, pred)
    a.scatter(y, pred, s=14, color=col, alpha=0.5, edgecolor="none")
    lo, hi = min(y.min(), pred.min()), max(y.max(), pred.max())
    a.plot([lo, hi], [lo, hi], "k--", lw=1)
    a.set_xlabel(f"Actual cell-mean {tgt}")
    a.set_ylabel("Predicted")
    a.set_title(f"Deterministic signal: R²={r2:.3f}")

for i, (tgt, col) in enumerate([("log_abs", C_log), ("f_ratio", C_f)]):
    a = ax[i, 2]
    r = qres[tgt]
    q = r["q"]
    yte = r["yte"]
    order = np.argsort(r["w"])
    xx = np.arange(len(order))
    a.fill_between(xx, q[0.05][order], q[0.95][order], color=col, alpha=0.20, label="90% PI")
    a.fill_between(xx, q[0.25][order], q[0.75][order], color=col, alpha=0.40, label="50% PI")
    a.plot(xx, q[0.5][order], color=col, lw=1.2, label="median")
    sub = np.linspace(0, len(order) - 1, 600).astype(int)
    a.scatter(xx[sub], yte[order][sub], s=3, color="k", alpha=0.3, zorder=3)
    a.set_xlabel("Test points (sorted by PI width)")
    a.set_ylabel(tgt)
    a.set_title(f"90%-PI cov={r['cov90']:.2f}")
    a.legend(fontsize=8, loc="upper left", framealpha=0.9)

for j, axx in enumerate(ax.flat):
    panel_letter(axx, string.ascii_lowercase[j])

fig.suptitle(
    "R² ceiling: variance is capped by irreducible seed noise, "
    "but the deterministic signal is highly predictable",
    fontsize=12.5,
    y=1.005,
)
fig.tight_layout()
fig.savefig(result_path("plots", "model_r2_ceiling.png"), dpi=150, bbox_inches="tight")
print("saved model_r2_ceiling.png")
