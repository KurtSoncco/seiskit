"""Per-quantile SHAP interaction structure (channel 50 quantile GBMs)."""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import shap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    FACTORS,
    FIG_DPI,
    cached_shap,
    load_channel50,
    load_quantile_models,
    seed_grouped_split,
    target_color,
    target_label,
)

from seiskit.plot_config import apply_style, panel_letter, result_path

apply_style(auto_format=True, font_size=10, frame="open")

d50 = load_channel50()

Xdf = d50[FACTORS]
tr, te = seed_grouped_split(d50)
Xte_df = Xdf.iloc[te]

taus = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

quant_models = load_quantile_models(taus, split_by="seed")

N_SHAP = 3000
rng = np.random.default_rng(0)
sub_idx = rng.choice(len(Xte_df), size=min(N_SHAP, len(Xte_df)), replace=False)
Xsub = Xte_df.iloc[sub_idx].reset_index(drop=True)

key_pair = {"log_abs": ("Vs1", "Height"), "f_ratio": ("CoV", "aHV")}
inter_by_tau = {t: {} for t in ["log_abs", "f_ratio"]}

t0 = time.time()
for tgt in ["log_abs", "f_ratio"]:
    for tau in taus:
        si = cached_shap(
            f"qshap_inter_{tgt}_ch50_tau{int(tau * 100):02d}",
            lambda m=quant_models[tgt][tau], X=Xsub: (
                shap.TreeExplainer(m).shap_interaction_values(X)
            ),
        )
        inter_by_tau[tgt][tau] = np.abs(si).mean(0)
print(f"done in {time.time() - t0:.0f}s")

rows = []
for tgt in ["log_abs", "f_ratio"]:
    a, b = key_pair[tgt]
    ia, ib = FACTORS.index(a), FACTORS.index(b)
    for tau in taus:
        M = inter_by_tau[tgt][tau]
        off = M.copy()
        np.fill_diagonal(off, 0)
        total_inter = off.sum()
        total_main = np.diag(M).sum()
        pair = 2 * M[ia, ib]
        rows.append(
            dict(
                target=tgt,
                tau=tau,
                key_pair=f"{a}x{b}",
                pair_abs=M[ia, ib],
                pair_pct_of_interactions=100 * pair / total_inter,
                interaction_fraction=100 * total_inter / (total_main + total_inter),
            )
        )
qint = pd.DataFrame(rows)
qint.to_csv(result_path("data", "quantile_shap_interactions.csv"), index=False)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
tcol = {"log_abs": target_color("log_abs"), "f_ratio": target_color("f_ratio")}
nice = {"log_abs": target_label("log_abs"), "f_ratio": target_label("f_ratio")}

ax = axes[0]
for tgt in ["log_abs", "f_ratio"]:
    s = qint[qint.target == tgt]
    ax.plot(s.tau, s.interaction_fraction, "o-", color=tcol[tgt], lw=2, ms=5, label=nice[tgt])
ax.set_xlabel("quantile τ")
ax.set_ylabel("interaction share of |SHAP| (%)")
ax.set_title("Interactions strengthen toward the tails", fontsize=10)
ax.legend(frameon=False, fontsize=8)
panel_letter(ax, "a")

ax = axes[1]
for tgt in ["log_abs", "f_ratio"]:
    s = qint[qint.target == tgt]
    kp = s.key_pair.iloc[0]
    ax.plot(s.tau, s.pair_abs, "o-", color=tcol[tgt], lw=2, ms=5, label=f"{nice[tgt]}: {kp}")
ax.set_xlabel("quantile τ")
ax.set_ylabel("mean |interaction SHAP| of key pair")
ax.set_title("Dominant pair interaction vs quantile", fontsize=10)
ax.legend(frameon=False, fontsize=8)
panel_letter(ax, "b")

fig.suptitle(
    "Per-quantile SHAP interaction structure (channel 50)",
    fontsize=12,
    y=1.02,
)
fig.tight_layout()
fig.savefig(
    result_path("plots", "quantile_shap_interactions.png"),
    dpi=FIG_DPI,
    bbox_inches="tight",
)
plt.close(fig)
print("saved")
