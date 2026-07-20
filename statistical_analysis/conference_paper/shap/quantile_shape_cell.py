"""
Quantile-specific SHAP, PDP, and Friedman H-statistics for cell-grouped QBMs.

Loads pre-trained quantile LightGBM models from::

    models/lgbm_q{tau}_{target}_cell.pkl

trained with ``python quantile_channel_model.py --split cell``.

Computes TreeSHAP beeswarms, partial dependence, mean |SHAP| vs τ, and
centered Friedman H² interaction strength (range [0, 1]).

Usage
-----
    python quantile_shape_cell.py
    python quantile_shape_cell.py --force-shap
"""

import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

import shap
from seiskit.plot_config import apply_style, get_crameri_cmap, panel_letter, result_path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    FACTOR_PLOT,
    FACTORS,
    FIG_DPI,
    TARGET_COLORS,
    cached_shap,
    load_channel50,
    load_quantile_models,
    quantile_model_stem,
    target_label,
)

TAUS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
TARGETS = {"f_ratio": "f_ratio", "log_abs": "log_abs"}
FORCE_SHAP = "--force-shap" in sys.argv
SHAP_N = 2000
PDP_GRID = 25
SPLIT_BY = "cell"

FACTOR_LIM = {
    "Vs1": {"limits": (100, 360), "step": 40},
    "Height": {"limits": (15, 100), "step": 10},
    "CoV": {"limits": (0.1, 0.3), "step": 0.05},
    "rH": {"limits": (10, 50), "step": 10},
    "aHV": {"limits": (1, 50), "step": 5},
}

NICE = {"log_abs": target_label("log_abs"), "f_ratio": target_label("f_ratio")}
TCOL = TARGET_COLORS

# Cell-grouped train split (matches quantile_channel_model.py --split cell)
d50 = load_channel50()
d50["cell"] = d50.groupby(FACTORS).ngroup()

gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
tr_idx, _ = next(gss.split(d50, groups=d50["cell"].values))
train = d50.iloc[tr_idx].copy()

rng = np.random.default_rng(0)
explain_idx = rng.choice(len(train), size=min(SHAP_N, len(train)), replace=False)
X_shap = train[FACTORS].iloc[explain_idx]

models = load_quantile_models(TAUS, targets=list(TARGETS), split_by=SPLIT_BY)
for tgt_key in TARGETS:
    missing = [tau for tau in TAUS if tau not in models.get(tgt_key, {})]
    if missing:
        tags = ", ".join(
            quantile_model_stem(tgt_key, tau, split_by=SPLIT_BY) for tau in missing
        )
        raise FileNotFoundError(
            f"Missing cell-split quantile models for {tgt_key}: {tags}. "
            "Run: python quantile/quantile_channel_model.py --split cell"
        )


# Compute SHAP values
def compute_shap(model, X):
    return shap.TreeExplainer(model).shap_values(X)


# Compute SHAP values for each target and quantile
shap_vals = defaultdict(dict)
for tgt_key in TARGETS:
    for tau in TAUS:
        key = f"shap_{SPLIT_BY}_{tgt_key}_q{int(tau * 100):02d}"
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


# Compute Friedman H² interaction strength (centered PDPs, range [0, 1]).
# H²_{jk} = Var(f_jk) / Var(f_j + f_k + f_jk) estimated on the PDP grid:
# interaction residual after removing additive main effects.
def friedman_h2(model, X_ref, i, j, gi, gj):
    pd_ij = pd_2d(model, X_ref, i, j, gi, gj)
    pd_i = pd_1d(model, X_ref, i, gi)
    pd_j = pd_1d(model, X_ref, j, gj)
    # Center so additive decomposition is identifiable (prevents H² > 1).
    pd_ij_c = pd_ij - pd_ij.mean()
    pd_i_c = pd_i - pd_i.mean()
    pd_j_c = pd_j - pd_j.mean()
    add = pd_i_c[:, None] + pd_j_c[None, :]
    inter = pd_ij_c - add
    den = np.sum(pd_ij_c**2)
    if den < 1e-12:
        return 0.0
    return float(np.clip(np.sum(inter**2) / den, 0.0, 1.0))


# Create grids for PDPs and compute pairs for Friedman H²
X_ref = X_shap.values.copy()
grids = {
    f: np.linspace(X_ref[:, k].min(), X_ref[:, k].max(), PDP_GRID) for k, f in enumerate(FACTORS)
}
pairs = list(combinations(range(len(FACTORS)), 2))

# Apply style to plots
apply_style(auto_format=True, font_size=10, frame="open")

# Plot beeswarm (τ=0.50) for both targets in a 1x2 figure with one colorbar
beeswarm_cmap = get_crameri_cmap("berlin")
fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
mean_abs = np.mean([np.abs(shap_vals[t][0.50]).mean(0) for t in TARGETS], axis=0)
feature_order = np.argsort(-mean_abs)

for col, tgt_key in enumerate(TARGETS):
    expl = shap.Explanation(
        values=shap_vals[tgt_key][0.50],
        data=X_shap.values,
        feature_names=FACTORS,
    )
    shap.plots.beeswarm(
        expl,
        ax=axes[col],
        show=False,
        color=beeswarm_cmap,
        color_bar=False,
        plot_size=None,
        order=feature_order,
    )
    axes[col].set_title(NICE[tgt_key])
    if col > 0:
        axes[col].set_yticklabels([])
    panel_letter(axes[col], "ab"[col])

sm = mpl_cm.ScalarMappable(cmap=beeswarm_cmap)
sm.set_array([0, 1])
# Locate correctly the colorbar to the right of the figure, without superimposing on the plots
cbar = fig.colorbar(sm, ax=axes, ticks=[0, 1], shrink=0.85, location="right")
cbar.set_ticklabels(["Low", "High"])
cbar.set_label("Feature value")
cbar.ax.tick_params(length=0)

fig.suptitle("SHAP beeswarm (τ=0.50, Cell-Grouped QBM)")
fig.tight_layout()
fig.savefig(result_path("plots", "quantile_shap_beeswarm.png"), dpi=FIG_DPI, bbox_inches="tight")
plt.close()

# Plot quantile-specific physics
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for col, tgt_key in enumerate(TARGETS):
    ax = axes[col]
    for fi, feat in enumerate(FACTORS):
        y = [np.abs(shap_vals[tgt_key][t][:, fi]).mean() for t in TAUS]
        ax.plot(
            TAUS,
            y,
            marker=FACTOR_PLOT[feat]["marker"],
            color=FACTOR_PLOT[feat]["color"],
            ms=5,
            lw=1.5,
            label=feat,
        )
    ax.set(
        xlabel=r"quantile $\tau$",
        ylabel=r"mean $|$SHAP$|$",
    )
    ax.legend(fontsize=10, frameon=False, ncol=2)
    ax.set_title(f"{NICE[tgt_key]}: quantile-specific physics", loc="left")

    # Set xlim and xticks
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.1))

    # For f_ratio, ylim would be from 0 to 0.06, for log_abs, ylim would be from 0 to 0.6
    if tgt_key == "f_ratio":
        ax.set_ylim(0, 0.06)
    elif tgt_key == "log_abs":
        ax.set_ylim(0, 0.6)

    panel_letter(ax, "ab"[col])
fig.suptitle("Factor importance across quantiles (Cell-Grouped QBM)", fontsize=10)
fig.tight_layout()
fig.savefig(result_path("plots", "quantile_shap_physics.png"), dpi=FIG_DPI, bbox_inches="tight")
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
            ax.set_xlim(FACTOR_LIM[feat]["limits"][0], FACTOR_LIM[feat]["limits"][1])
            ax.set_xticks(
                np.arange(
                    FACTOR_LIM[feat]["limits"][0],
                    FACTOR_LIM[feat]["limits"][1] + FACTOR_LIM[feat]["step"],
                    FACTOR_LIM[feat]["step"],
                )
            )
    fig.suptitle(f"{NICE[tgt_key]}: partial dependence (cell-grouped QBM)")
    fig.tight_layout()
    fig.savefig(
        result_path("plots", f"quantile_shap_pdp_{tgt_key}.png"), dpi=FIG_DPI, bbox_inches="tight"
    )
    plt.close()

# Plot Friedman H² heatmaps
for tgt_key in TARGETS:
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    im = None
    for k, (ax, tau) in enumerate(zip(axes, [0.05, 0.50, 0.95])):
        model = models[tgt_key][tau]
        h = np.zeros((len(FACTORS), len(FACTORS)))
        for i, j in pairs:
            h[i, j] = h[j, i] = friedman_h2(
                model, X_ref, i, j, grids[FACTORS[i]], grids[FACTORS[j]]
            )
        im = ax.imshow(h, cmap=get_crameri_cmap("lapaz"), vmin=0, vmax=max(0.15, h.max()))
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
    fig.savefig(result_path("plots", f"quantile_shap_hstat_{tgt_key}.png"), dpi=FIG_DPI)
    plt.close()
