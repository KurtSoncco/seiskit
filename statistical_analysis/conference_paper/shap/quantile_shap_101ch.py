"""Per-quantile SHAP importance across 101 recorders (trains quantile GBMs on the fly)."""

import string
import sys
import time
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import shap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FACTORS, cached_shap, load_master, seed_grouped_split

from seiskit.plot_config import apply_style, panel_letter, result_path

apply_style(auto_format=True, font_size=10, frame="open")

full = load_master()
full["cell"] = full.groupby(FACTORS).ngroup()

facx = FACTORS + ["channel"]
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
t0 = time.time()
for tgt in ["log_abs", "f_ratio"]:
    ytr = full.iloc[tr][tgt].values
    qmods[tgt] = {a: lgb.LGBMRegressor(**qp(a)).fit(Xtr, ytr) for a in taus}
    print(tgt, "fit", f"{time.time() - t0:.0f}s")

N_SHAP = 5000
rng = np.random.RandomState(0)
samp_idx = rng.choice(te, min(N_SHAP, len(te)), replace=False)
Xs = full.iloc[samp_idx][facx].values

t0 = time.time()
qshap = {}
for tgt in ["log_abs", "f_ratio"]:
    qshap[tgt] = {}
    for a in taus:
        sv = cached_shap(
            f"qshap_101ch_{tgt}_tau{int(a * 100):02d}",
            lambda m=qmods[tgt][a], X=Xs: shap.TreeExplainer(m).shap_values(X),
        )
        qshap[tgt][a] = np.abs(sv).mean(0)
        print(tgt, a, f"{time.time() - t0:.0f}s")

rows = []
for tgt in ["log_abs", "f_ratio"]:
    for a in taus:
        for j, fn in enumerate(facx):
            rows.append(dict(target=tgt, tau=a, feature=fn, mean_abs_shap=qshap[tgt][a][j]))
qshap_df = pd.DataFrame(rows)
qshap_df["pct"] = qshap_df.groupby(["target", "tau"])["mean_abs_shap"].transform(
    lambda x: 100 * x / x.sum()
)

slope_cols = {
    "Vs1": "#4C72B0",
    "Height": "#C44E52",
    "CoV": "#55A868",
    "rH": "#8172B3",
    "aHV": "#DD8452",
    "channel": "#777777",
}

fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
for i, tgt in enumerate(["log_abs", "f_ratio"]):
    a = ax[i]
    d_ = qshap_df[qshap_df.target == tgt]
    for fn in ["Vs1", "Height", "CoV", "rH", "aHV", "channel"]:
        s = d_[d_.feature == fn].sort_values("tau")
        lw = 2.2 if fn != "channel" else 1.6
        ls = "-" if fn != "channel" else "--"
        a.plot(s.tau, s.pct, ls + "o", color=slope_cols[fn], lw=lw, ms=6, label=fn)
    a.set_xlabel("Quantile τ")
    a.set_ylabel("mean|SHAP| (% of total)")
    a.set_title(f"{tgt}", color="#C44E52" if tgt == "log_abs" else "#4C72B0")
    a.legend(fontsize=8, ncol=2)
    a.set_xticks(taus)

ax[0].annotate(
    "Vs1 rises →\nupper-tail driver",
    xy=(0.95, 34.8),
    xytext=(0.55, 30),
    fontsize=8,
    color="#4C72B0",
    arrowprops=dict(arrowstyle="->", color="#4C72B0"),
)
ax[1].annotate(
    "CoV: low-tail",
    xy=(0.05, 46.3),
    xytext=(0.2, 52),
    fontsize=8,
    color="#55A868",
    arrowprops=dict(arrowstyle="->", color="#55A868"),
)
ax[1].annotate(
    "aHV: median+up",
    xy=(0.75, 37),
    xytext=(0.35, 40),
    fontsize=8,
    color="#DD8452",
    arrowprops=dict(arrowstyle="->", color="#DD8452"),
)

for j, axx in enumerate(ax.flat):
    panel_letter(axx, string.ascii_lowercase[j])

fig.suptitle(
    "Per-quantile SHAP importance, pooled across 101 recorders: factor roles shift across the distribution;\n"
    "channel (recorder position) is a minor direct driver (2–3%) at every quantile",
    fontsize=12,
    y=1.02,
)
fig.tight_layout()
fig.savefig(
    result_path("plots", "quantile_shap_101ch.png"),
    dpi=150,
    bbox_inches="tight",
)
print("saved")
