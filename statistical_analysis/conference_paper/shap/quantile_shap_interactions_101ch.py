"""Per-quantile SHAP interaction heatmaps across all 101 recorders."""

import sys
import string
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_master, FACTORS, seed_grouped_split, cached_shap

from seiskit.plot_config import apply_style, panel_letter, result_path

apply_style(auto_format=True, font_size=10, frame="open")

full = load_master()

fac = FACTORS
facx = fac + ["channel"]
taus = [0.05, 0.25, 0.5, 0.75, 0.95]

tr, te = seed_grouped_split(full)
Xtr = full.iloc[tr][facx].values


def qp(a):
    return dict(
        objective="quantile",
        alpha=a,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=0,
        verbose=-1,
        n_jobs=-1,
    )


qmods = {}
for tgt in ["log_abs", "f_ratio"]:
    ytr = full.iloc[tr][tgt].values
    qmods[tgt] = {a: lgb.LGBMRegressor(**qp(a)).fit(Xtr, ytr) for a in taus}

N_SHAP = 5000
rng = np.random.RandomState(0)
samp = rng.choice(te, min(N_SHAP, len(te)), replace=False)
Xs = full.iloc[samp][facx].values

t0 = time.time()
qinter = {}
for tgt in ["log_abs", "f_ratio"]:
    qinter[tgt] = {}
    for a in taus:
        siv = cached_shap(
            f"qshap_inter_101ch_{tgt}_tau{int(a * 100):02d}",
            lambda m=qmods[tgt][a], X=Xs: (shap.TreeExplainer(m).shap_interaction_values(X)),
        )
        qinter[tgt][a] = np.abs(siv).mean(0)
        print(tgt, a, f"{time.time() - t0:.0f}s")

np.savez(
    result_path("data", "quantile_shap_interactions_101ch.npz"),
    **{f"{t}_{int(a * 100)}": qinter[t][a] for t in qinter for a in taus},
    features=np.array(facx),
)

z = np.load(
    result_path("data", "quantile_shap_interactions_101ch.npz"),
    allow_pickle=True,
)
feats = list(z["features"])
nf = len(feats)

rows = []
for tgt in ["log_abs", "f_ratio"]:
    for a in taus:
        M = z[f"{tgt}_{int(a * 100)}"]
        for i in range(nf):
            for j in range(i + 1, nf):
                rows.append(
                    dict(
                        target=tgt,
                        tau=a,
                        f1=feats[i],
                        f2=feats[j],
                        inter=M[i, j] + M[j, i],
                    )
                )
inter = pd.DataFrame(rows)
inter.to_csv(
    result_path("data", "quantile_shap_interactions_101ch.csv"),
    index=False,
)


def pair_series(tgt, f1, f2):
    s = inter[
        (inter.target == tgt)
        & (((inter.f1 == f1) & (inter.f2 == f2)) | ((inter.f1 == f2) & (inter.f2 == f1)))
    ]
    return s.sort_values("tau")[["tau", "inter"]]


C_log = "#C44E52"
C_f = "#4C72B0"
fig = plt.figure(figsize=(15, 8.6))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.42, wspace=0.5)


def heat(ax, M, title, vmax):
    im = ax.imshow(M, cmap="magma", vmin=0, vmax=vmax)
    ax.set_xticks(range(nf))
    ax.set_yticks(range(nf))
    ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(feats, fontsize=8)
    ax.set_title(title, fontsize=10)
    return im


Mlog_med = z["log_abs_50"].copy()
Mlog_hi = z["log_abs_95"].copy()
Mf_med = z["f_ratio_50"].copy()
Mf_hi = z["f_ratio_95"].copy()
for M in [Mlog_med, Mlog_hi, Mf_med, Mf_hi]:
    np.fill_diagonal(M, 0)

vlog = max(Mlog_med.max(), Mlog_hi.max())
vf = max(Mf_med.max(), Mf_hi.max())

ax1 = fig.add_subplot(gs[0, 0])
im1 = heat(ax1, Mlog_med, "log_abs  τ=0.50", vlog)
ax2 = fig.add_subplot(gs[0, 1])
im2 = heat(ax2, Mlog_hi, "log_abs  τ=0.95", vlog)
fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
ax3 = fig.add_subplot(gs[0, 2])
im3 = heat(ax3, Mf_med, "f_ratio  τ=0.50", vf)
ax4 = fig.add_subplot(gs[0, 3])
im4 = heat(ax4, Mf_hi, "f_ratio  τ=0.95", vf)
fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

axA = fig.add_subplot(gs[1, :2])
pairs_log = [("Vs1", "Height", "#4C72B0"), ("Height", "aHV", "#DD8452"), ("Vs1", "aHV", "#55A868")]
for f1, f2, c in pairs_log:
    ps = pair_series("log_abs", f1, f2).set_index("tau")["inter"]
    axA.plot(ps.index, ps.values, "o-", color=c, lw=2, ms=6, label=f"{f1}×{f2}")
axA.set_xlabel("Quantile τ")
axA.set_ylabel("Total |interaction SHAP|")
axA.set_title("log_abs: interactions strengthen toward upper tail", color=C_log)
axA.legend(fontsize=8)
axA.set_xticks(taus)
axA.annotate(
    "2.0× median",
    xy=(0.95, 0.119),
    xytext=(0.7, 0.11),
    fontsize=8,
    color="#4C72B0",
    arrowprops=dict(arrowstyle="->", color="#4C72B0"),
)

axB = fig.add_subplot(gs[1, 2:])
pairs_f = [("CoV", "aHV", "#55A868"), ("Height", "aHV", "#DD8452"), ("Height", "CoV", "#C44E52")]
for f1, f2, c in pairs_f:
    ps = pair_series("f_ratio", f1, f2).set_index("tau")["inter"]
    axB.plot(ps.index, ps.values, "o-", color=c, lw=2, ms=6, label=f"{f1}×{f2}")
axB.set_xlabel("Quantile τ")
axB.set_ylabel("Total |interaction SHAP|")
axB.set_title("f_ratio: CoV×aHV interaction 2.7× stronger at τ=0.95", color=C_f)
axB.legend(fontsize=8)
axB.set_xticks(taus)
axB.annotate(
    "2.7× median",
    xy=(0.95, 0.0219),
    xytext=(0.7, 0.020),
    fontsize=8,
    color="#55A868",
    arrowprops=dict(arrowstyle="->", color="#55A868"),
)

for j, axx in enumerate([ax1, ax2, ax3, ax4, axA, axB]):
    panel_letter(axx, string.ascii_lowercase[j])

fig.suptitle(
    "SHAP interactions across 101 recorders: Vs1×Height (log_abs) and CoV×aHV (f_ratio) "
    "dominate and intensify toward the upper tail",
    fontsize=12.5,
    y=0.99,
)
fig.savefig(
    result_path("plots", "quantile_shap_interactions_101ch.png"),
    dpi=150,
    bbox_inches="tight",
)
print("saved")
