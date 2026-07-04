"""GBM mean + quantile model performance (channel 50, seed-grouped hold-out).

Uses pre-trained models from models/ to avoid redundant training.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import (
    FACTORS,
    load_channel50,
    load_mean_models,
    load_quantile_models,
    seed_grouped_split,
)
from sklearn.metrics import mean_pinball_loss, r2_score

from seiskit.plot_config import apply_style, panel_letter, result_path

warnings.filterwarnings("ignore")

d50 = load_channel50()
tr, te = seed_grouped_split(d50, test_size=0.25, seed=0)

Xdf = d50[FACTORS]
Xtr_df, Xte_df = Xdf.iloc[tr], Xdf.iloc[te]

taus = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

mean_models = load_mean_models()
quant_models = load_quantile_models(taus=taus)

pinball_rows = []
for tgt in ["log_abs", "f_ratio"]:
    yte = d50.iloc[te][tgt].values
    for q in taus:
        if q in quant_models[tgt]:
            pred = quant_models[tgt][q].predict(Xte_df)
            pl = mean_pinball_loss(yte, pred, alpha=q)
            pinball_rows.append(dict(target=tgt, tau=q, pinball_loss=pl))
pinball_df = pd.DataFrame(pinball_rows)

imp = {}
for tgt in ["log_abs", "f_ratio"]:
    imp[tgt] = pd.Series(
        mean_models[tgt].booster_.feature_importance(importance_type="gain"),
        index=FACTORS,
    )
imp_df = pd.DataFrame(imp)
imp_df = (imp_df / imp_df.sum() * 100).round(1)

apply_style(auto_format=True, font_size=10, frame="open")
fig, axes = plt.subplots(2, 3, figsize=(13, 8))

for ax, tgt, col in [
    (axes[0, 0], "log_abs", "#C44E52"),
    (axes[0, 1], "f_ratio", "#4C72B0"),
]:
    yte = d50.iloc[te][tgt].values
    pred = mean_models[tgt].predict(Xte_df)
    idx = np.linspace(0, len(yte) - 1, 3500).astype(int)
    ax.plot(pred[idx], yte[idx], "o", ms=2, color=col, alpha=0.25)
    lo, hi = yte.min(), yte.max()
    ax.plot([lo, hi], [lo, hi], "-", color="0.2", lw=1)
    r2 = r2_score(yte, pred)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(f"{tgt}: test R²={r2:.2f}")

ax = axes[0, 2]
x = np.arange(2)
w = 0.35
ax.bar(x - w / 2, [0.28, 0.18], w, color="0.6", label="linear")
ax.bar(x + w / 2, [0.358, 0.208], w, color="#55A868", label="LGBM")
ax.set_xticks(x)
ax.set_xticklabels(["log_abs", "f_ratio"])
ax.set_ylabel("test $R^2$")
ax.set_title("GBM vs linear (mean model)")
ax.legend(fontsize=7, frameon=False)

ax = axes[1, 0]
for tgt, col in [("log_abs", "#C44E52"), ("f_ratio", "#4C72B0")]:
    s = pinball_df[pinball_df.target == tgt]
    ax.plot(s["tau"], s["pinball_loss"], "o-", color=col, label=tgt, ms=5)
ax.set_xlabel(r"quantile $\tau$")
ax.set_ylabel("pinball loss (test)")
ax.set_title("Quantile-model pinball loss")
ax.legend(fontsize=7, frameon=False)

ax = axes[1, 1]
for tgt, col in [("log_abs", "#C44E52"), ("f_ratio", "#4C72B0")]:
    yte = d50.iloc[te][tgt].values
    cov = []
    for q in taus:
        if q in quant_models[tgt]:
            cov.append(np.mean(yte <= quant_models[tgt][q].predict(Xte_df)))
    ax.plot(taus[: len(cov)], cov, "o-", color=col, label=tgt, ms=5)
ax.plot([0, 1], [0, 1], "--", color="0.3", lw=1)
ax.set_xlabel(r"nominal $\tau$")
ax.set_ylabel("empirical coverage")
ax.set_title("Quantile calibration (held-out seeds)")
ax.legend(fontsize=7, frameon=False)

ax = axes[1, 2]
x = np.arange(5)
w = 0.38
ax.bar(x - w / 2, imp_df["log_abs"], w, color="#C44E52", label="log_abs")
ax.bar(x + w / 2, imp_df["f_ratio"], w, color="#4C72B0", label="f_ratio")
ax.set_xticks(x)
ax.set_xticklabels(FACTORS, rotation=30, fontsize=7.5)
ax.set_ylabel("gain importance (%)")
ax.set_title("GBM feature importance")
ax.legend(fontsize=7, frameon=False)

for i, ax in enumerate(axes.ravel()):
    panel_letter(ax, chr(97 + i))

fig.suptitle(
    "Gradient boosting (channel 50, seed-grouped hold-out): "
    "performance, quantile calibration, importances",
    fontsize=10,
    y=0.99,
)
fig.tight_layout()
fig.savefig(result_path("plots", "gbm_performance.png"), dpi=150, bbox_inches="tight")
print("saved gbm_performance.png")
