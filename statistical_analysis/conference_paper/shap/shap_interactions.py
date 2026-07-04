"""SHAP interaction heatmaps and dependence plots for both targets."""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import shap

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FACTORS, cached_shap, load_channel50, load_mean_models, seed_grouped_split

from seiskit.plot_config import apply_style, panel_letter, result_path

apply_style(auto_format=True, font_size=10, frame="open")

nice = {"log_abs": "log(abs_TF_ratio)", "f_ratio": "f_ratio"}

d50 = load_channel50()

Xdf = d50[FACTORS]
tr, te = seed_grouped_split(d50)
Xte_df = Xdf.iloc[te]

mean_models = load_mean_models()

N_SHAP = 3000
idx = np.random.RandomState(42).choice(len(Xte_df), min(N_SHAP, len(Xte_df)), replace=False)
X_shap = Xte_df.iloc[idx]

shap_vals, shap_inter = {}, {}
for tgt in ["log_abs", "f_ratio"]:
    model = mean_models[tgt]
    shap_vals[tgt] = cached_shap(
        f"shap_{tgt}_ch50_te",
        lambda m=model: shap.TreeExplainer(m).shap_values(Xte_df),
    )
    shap_inter[tgt] = cached_shap(
        f"shap_inter_{tgt}_ch50",
        lambda m=model: shap.TreeExplainer(m).shap_interaction_values(X_shap),
    )


def top_pairs(si, k=2):
    m = np.abs(si).mean(0).copy()
    np.fill_diagonal(m, 0)
    pairs = []
    for i in range(5):
        for j in range(i + 1, 5):
            pairs.append((m[i, j], FACTORS[i], FACTORS[j]))
    return sorted(pairs, reverse=True)[:k], m


tp = {}
for tgt in ["log_abs", "f_ratio"]:
    tp[tgt], _ = top_pairs(shap_inter[tgt], 3)
    print(tgt, [(f"{a}×{b}", round(v, 4)) for v, a, b in tp[tgt]])

fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.0))
for r, tgt in enumerate(["log_abs", "f_ratio"]):
    si = shap_inter[tgt]
    M = np.abs(si).mean(0)
    ax = axes[r, 0]
    Moff = M.copy()
    np.fill_diagonal(Moff, np.nan)
    im = ax.imshow(Moff, cmap="viridis")
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(FACTORS, rotation=45, ha="right")
    ax.set_yticklabels(FACTORS)
    ax.set_title(f"{nice[tgt]}\nmean |interaction SHAP|", fontsize=10)
    for i in range(5):
        for j in range(5):
            if i != j:
                ax.text(
                    j,
                    i,
                    f"{M[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="white" if M[i, j] < np.nanmax(Moff) * 0.6 else "black",
                    fontsize=6.5,
                )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for c, (val, fa, fb) in enumerate(tp[tgt][:2]):
        ax = axes[r, c + 1]
        fi = FACTORS.index(fa)
        xv = Xte_df[fa].values
        yv = shap_vals[tgt][:, fi]
        cvar = Xte_df[fb].values
        sc = ax.scatter(xv, yv, c=cvar, cmap="coolwarm", s=12, alpha=0.7, edgecolor="none")
        ax.set_xlabel(fa)
        ax.set_ylabel(f"SHAP({fa})")
        ax.set_title(f"{fa} colored by {fb}", fontsize=9)
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(fb, fontsize=7)
        ax.axhline(0, color="0.6", lw=0.7, zorder=0)

for r, tgt in enumerate(["log_abs", "f_ratio"]):
    panel_letter(axes[r, 0], chr(97 + r * 3))

fig.suptitle(
    "SHAP interaction structure — LightGBM mean models (channel 50)",
    fontsize=12,
    y=1.005,
)
fig.tight_layout()
fig.savefig(
    result_path("plots", "shap_interactions.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.close(fig)
print("interactions fig done")
