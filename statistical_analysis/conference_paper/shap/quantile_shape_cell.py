"""
Quantile-specific SHAP, PDP, and Friedman H-statistics for cell-grouped QBMs.

Loads pre-trained quantile LightGBM models (from ``quantile_channel_model.py
--split cell``), then computes TreeSHAP beeswarms, partial dependence,
quantile-specific physics curves (mean |SHAP| vs τ), and Friedman's H²
interaction strength.

Usage
-----
    python quantile_shape_cell.py
    python quantile_shape_cell.py --force-shap

Requires
--------
    models/lgbm_q{tau}_{target}.pkl from::

        python quantile_channel_model.py --split cell
"""

import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

import shap
from seiskit.plot_config import apply_style, panel_letter, result_path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FACTORS, cached_shap, load_channel50, load_quantile_models

TAUS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
TARGETS = {"f_ratio": "f_ratio", "log_abs": "log_abs"}
FORCE_SHAP = "--force-shap" in sys.argv
SHAP_N = 2000
PDP_GRID = 25

# Uniform coloring of factors
FACTOR_COL = {
    "Vs1": cmc.nuuk(0.2),
    "Height": cmc.nuuk(0.8),
    "CoV": cmc.nuuk(0.4),
    "rH": cmc.nuuk(0.6),
    "aHV": cmc.nuuk(0.3),
}

FACTOR_LIM = {
    "Vs1": (100, 360),
    "Height": (15, 100),
    "CoV": (0.1, 0.3),
    "rH": (10, 50),
    "aHV": (1, 50),
}

# Nice labels for targets
NICE = {"log_abs": "log_abs", "f_ratio": "$f$ ratio"}

# Target-specific colors
TCOL = {"log_abs": FACTOR_COL["Vs1"], "f_ratio": FACTOR_COL["Height"]}

# Cell-grouped train split (matches quantile_channel_model.py --split cell)
d50 = load_channel50()
d50["cell"] = d50.groupby(FACTORS).ngroup()

gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
tr_idx, _ = next(gss.split(d50, groups=d50["cell"].values))
train = d50.iloc[tr_idx].copy()

rng = np.random.default_rng(0)
explain_idx = rng.choice(len(train), size=min(SHAP_N, len(train)), replace=False)
X_shap = train[FACTORS].iloc[explain_idx]

models = load_quantile_models(TAUS)
for tgt_key in TARGETS:
    missing = [tau for tau in TAUS if tau not in models.get(tgt_key, {})]
    if missing:
        tags = ", ".join(f"lgbm_q{int(tau * 100):02d}_{tgt_key}" for tau in missing)
        raise FileNotFoundError(
            f"Missing quantile models for {tgt_key}: {tags}. "
            "Run: python quantile_channel_model.py --split cell"
        )


# Compute SHAP values
def compute_shap(model, X):
    return shap.TreeExplainer(model).shap_values(X)


# Compute SHAP values for each target and quantile
shap_vals = defaultdict(dict)
for tgt_key in TARGETS:
    for tau in TAUS:
        key = f"shap_cell_{tgt_key}_q{int(tau * 100):02d}"
        shap_vals[tgt_key][tau] = cached_shap(
            key,
            lambda m=models[tgt_key][tau]: compute_shap(m, X_shap),
            force=FORCE_SHAP,
        )


# Compute partial dependence and Friedman H²
def pd_1d(model, X_ref, feat_idx, grid):
    X = X_ref.copy()
    out = np.empty(len(grid))
    for k, val in enumerate(grid):
        X[:, feat_idx] = val
        out[k] = model.predict(X).mean()
    return out


# Compute partial dependence for 2D grids
def pd_2d(model, X_ref, i, j, gi, gj):
    X = X_ref.copy()
    out = np.zeros((len(gi), len(gj)))
    for ii, vi in enumerate(gi):
        for jj, vj in enumerate(gj):
            X[:, i], X[:, j] = vi, vj
            out[ii, jj] = model.predict(X).mean()
    return out


# Compute Friedman H² interaction strength
def friedman_h2(model, X_ref, i, j, gi, gj):
    pd_ij = pd_2d(model, X_ref, i, j, gi, gj)
    pd_i = pd_1d(model, X_ref, i, gi)
    pd_j = pd_1d(model, X_ref, j, gj)
    inter = pd_ij - pd_i[:, None] - pd_j[None, :]
    den = np.sum(pd_ij**2)
    return float(np.sum(inter**2) / den) if den > 1e-12 else 0.0


# Create grids for PDPs and compute pairs for Friedman H²
X_ref = X_shap.values.copy()
grids = {
    f: np.linspace(X_ref[:, k].min(), X_ref[:, k].max(), PDP_GRID) for k, f in enumerate(FACTORS)
}
pairs = list(combinations(range(len(FACTORS)), 2))

# Apply style to plots
apply_style(auto_format=True, font_size=10, frame="open")

# Plot beeswarm (τ=0.50)
for tgt_key in TARGETS:
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_vals[tgt_key][0.50], X_shap, feature_names=FACTORS, show=False)
    plt.title(f"{NICE[tgt_key]}: SHAP beeswarm (τ=0.50, cell-grouped)")
    plt.tight_layout()
    plt.savefig(
        result_path("plots", f"quantile_shap_beeswarm_{tgt_key}.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()

# Plot quantile-specific physics
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for col, tgt_key in enumerate(TARGETS):
    ax = axes[col]
    for fi, feat in enumerate(FACTORS):
        y = [np.abs(shap_vals[tgt_key][t][:, fi]).mean() for t in TAUS]
        ax.plot(TAUS, y, "o-", color=FACTOR_COL[feat], ms=5, lw=1.5, label=feat)
    ax.set(
        xlabel=r"quantile $\tau$",
        ylabel=r"mean $|$SHAP$|$",
        title=f"{NICE[tgt_key]}: quantile-specific physics",
    )
    ax.legend(fontsize=7, frameon=False, ncol=2)
    panel_letter(ax, "ab"[col])
fig.suptitle("Factor importance across quantiles (cell-grouped QBM)")
fig.tight_layout()
fig.savefig(result_path("plots", "quantile_shap_physics.png"), dpi=150, bbox_inches="tight")
plt.close()

# Plot PDPs at τ = 0.05, 0.50, 0.95
for tgt_key in TARGETS:
    fig, axes = plt.subplots(3, len(FACTORS), figsize=(14, 7), sharey="row", sharex="col")
    for row, tau in enumerate([0.05, 0.50, 0.95]):
        model = models[tgt_key][tau]
        for col, feat in enumerate(FACTORS):
            ax = axes[row, col]
            ax.plot(grids[feat], pd_1d(model, X_ref, col, grids[feat]), color=TCOL[tgt_key], lw=1.8)
            if row == 0:
                ax.set_title(feat, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"τ={tau:g}\nPD")
            if row == 2:
                ax.set_xlabel(feat)
            ax.set_xlim(FACTOR_LIM[feat])
    fig.suptitle(f"{NICE[tgt_key]}: partial dependence (cell-grouped QBM)")
    fig.tight_layout()
    fig.savefig(
        result_path("plots", f"quantile_shap_pdp_{tgt_key}.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()

# Plot Friedman H² heatmaps
for tgt_key in TARGETS:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    im = None
    for k, (ax, tau) in enumerate(zip(axes, [0.05, 0.50, 0.95])):
        model = models[tgt_key][tau]
        h = np.zeros((len(FACTORS), len(FACTORS)))
        for i, j in pairs:
            h[i, j] = h[j, i] = friedman_h2(
                model, X_ref, i, j, grids[FACTORS[i]], grids[FACTORS[j]]
            )
        im = ax.imshow(h, cmap=cmc.tokyo, vmin=0, vmax=max(0.15, h.max()))
        ax.set_xticks(range(len(FACTORS)))
        ax.set_yticks(range(len(FACTORS)))
        ax.set_xticklabels(FACTORS, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(FACTORS, fontsize=8)
        for ii in range(len(FACTORS)):
            for jj in range(len(FACTORS)):
                if ii != jj:
                    ax.text(jj, ii, f"{h[ii, jj]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title(f"τ={tau:g}")
        panel_letter(ax, "abc"[k])

    # Colorbar at the left edge of the figure
    fig.colorbar(im, ax=axes, location="right", shrink=0.8, label=r"$H^2$")

    fig.suptitle(f"{NICE[tgt_key]}: Friedman $H^2$ interaction strength")
    fig.savefig(
        result_path("plots", f"quantile_shap_hstat_{tgt_key}.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
