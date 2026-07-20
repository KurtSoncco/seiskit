"""Per-quantile SHAP: which factors drive the tails (channel 50 quantile GBMs)."""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import shap
from seiskit.plot_config import apply_style, get_crameri_cmap, panel_letter, result_path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    FACTOR_COLORS,
    FACTORS,
    FIG_DPI,
    cached_shap,
    load_channel50,
    load_quantile_models,
    seed_grouped_split,
    target_label,
)

apply_style(auto_format=True, font_size=10, frame="open")

fac = FACTORS
taus = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
fcol = FACTOR_COLORS
nice = {"log_abs": target_label("log_abs"), "f_ratio": target_label("f_ratio")}

d50 = load_channel50()

Xdf = d50[FACTORS]
tr, te = seed_grouped_split(d50)
Xte_df = Xdf.iloc[te]

quant_models = load_quantile_models(taus, split_by="seed")

q_importance = {}
q_shapvals = {}
for tgt in ["log_abs", "f_ratio"]:
    mat = np.zeros((len(taus), len(fac)))
    for i, tau in enumerate(taus):
        sv = cached_shap(
            f"qshap_tails_{tgt}_ch50_tau{int(tau * 100):02d}",
            lambda m=quant_models[tgt][tau], X=Xte_df: (shap.TreeExplainer(m).shap_values(X)),
        )
        mat[i] = np.abs(sv).mean(0)
        if tau in (0.05, 0.5, 0.95):
            q_shapvals[(tgt, tau)] = sv
    q_importance[tgt] = pd.DataFrame(mat, index=taus, columns=fac)

fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))

for r, tgt in enumerate(["log_abs", "f_ratio"]):
    ax = axes[r, 0]
    pct = q_importance[tgt].div(q_importance[tgt].sum(1), axis=0) * 100
    for f in fac:
        ax.plot(taus, pct[f].values, "o-", color=fcol[f], ms=4, lw=1.8, label=f)
    ax.set_xlabel("quantile τ")
    ax.set_ylabel("% of total |SHAP|")
    ax.set_title(f"{nice[tgt]}\nfactor importance vs quantile", fontsize=10)
    ax.set_xticks([0.05, 0.25, 0.5, 0.75, 0.95])
    ax.set_xticklabels(["0.05", "0.25", "0.50", "0.75", "0.95"], fontsize=7)
    if r == 0:
        ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper center")
    panel_letter(ax, chr(97 + r * 3))


def dep(ax, tgt, tau, feat, letter, title):
    sv = q_shapvals[(tgt, tau)]
    fi = fac.index(feat)
    xv = Xte_df[feat].values
    yv = sv[:, fi]
    partner = {"CoV": "aHV", "aHV": "CoV", "Vs1": "Height", "Height": "Vs1"}[feat]
    cvar = Xte_df[partner].values
    sc = ax.scatter(xv, yv, c=cvar, cmap=get_crameri_cmap("vik"), s=12, alpha=0.7, edgecolor="none")
    ax.axhline(0, color="0.6", lw=0.7)
    ax.set_xlabel(feat)
    ax.set_ylabel(f"SHAP({feat}) @ τ={tau:g}")
    ax.set_title(title, fontsize=9)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(partner, fontsize=7)
    panel_letter(ax, letter)


dep(
    axes[0, 1],
    "log_abs",
    0.95,
    "Vs1",
    "b",
    "Vs1 at τ=0.95 (upper tail)\nstiffness amplifies the high tail",
)
dep(
    axes[0, 2],
    "log_abs",
    0.05,
    "Height",
    "c",
    "Height at τ=0.05 (lower tail)\nlayer height sets the floor",
)
dep(
    axes[1, 1],
    "f_ratio",
    0.05,
    "CoV",
    "e",
    "CoV at τ=0.05 (lower tail)\nheterogeneity drives de-amplification",
)
dep(
    axes[1, 2],
    "f_ratio",
    0.95,
    "aHV",
    "f",
    "aHV at τ=0.95 (upper tail)\nanisotropy drives amplification",
)

fig.suptitle(
    "Per-quantile SHAP: which factors drive the tails (channel 50 quantile GBMs)",
    fontsize=12,
    y=1.005,
)
fig.tight_layout()
fig.savefig(
    result_path("plots", "quantile_shap_tails.png"),
    dpi=FIG_DPI,
    bbox_inches="tight",
)
plt.close(fig)
print("re-saved")
