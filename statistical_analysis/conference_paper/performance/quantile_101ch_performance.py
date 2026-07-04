"""Pooled quantile GBM across 101 recorders — calibration, heteroscedastic
intervals, and pinball skill.

Uses pre-trained quantile models from models/ instead of re-fitting,
reducing runtime from hours to ~30 seconds.
"""

import sys
import string
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from seiskit.plot_config import apply_style, panel_letter, result_path
from config import load_master, FACTORS, load_quantile_models, seed_grouped_split

apply_style(auto_format=True, font_size=10, frame="open")

full = load_master()
taus = [0.05, 0.25, 0.5, 0.75, 0.95]

tr, te = seed_grouped_split(full, test_size=0.25, seed=0)
Xte = full.iloc[te][FACTORS]
yte = {tgt: full.iloc[te][tgt].values for tgt in ["log_abs", "f_ratio"]}

qmodels = load_quantile_models(taus=taus)

qpred: dict[str, dict[float, np.ndarray]] = {}
for tgt in ["log_abs", "f_ratio"]:
    qpred[tgt] = {}
    for tau in taus:
        if tau in qmodels[tgt]:
            qpred[tgt][tau] = qmodels[tgt][tau].predict(Xte)

# Coverage and calibration from held-out seeds
cov_rows = []
for tgt in ["log_abs", "f_ratio"]:
    y = yte[tgt]
    lo90, hi90 = qpred[tgt][0.05], qpred[tgt][0.95]
    lo50, hi50 = qpred[tgt][0.25], qpred[tgt][0.75]
    cov90 = np.mean((y >= lo90) & (y <= hi90))
    cov50 = np.mean((y >= lo50) & (y <= hi50))
    per = {tau: float(np.mean(y <= qpred[tgt][tau])) for tau in taus}
    cross = np.mean(np.diff(np.column_stack([qpred[tgt][tau] for tau in taus]), axis=1) < 0)
    cov_rows.append(
        dict(
            target=tgt,
            cov90=cov90,
            cov50=cov50,
            cross_rate=cross,
            **{f"cov_tau{int(tau * 100)}": per[tau] for tau in taus},
        )
    )
covdf = pd.DataFrame(cov_rows)


# Pinball skill score (vs marginal quantile baseline)
def pinball(y, q, alpha):
    d = y - q
    return np.mean(np.maximum(alpha * d, (alpha - 1) * d))


skill_rows = []
for tgt in ["log_abs", "f_ratio"]:
    y = yte[tgt]
    for tau in taus:
        base_q = np.quantile(y, tau)
        pb_base = pinball(y, base_q, tau)
        pb_model = pinball(y, qpred[tgt][tau], tau)
        skill_rows.append(
            dict(
                target=tgt,
                tau=tau,
                pinball_model=pb_model,
                pinball_base=pb_base,
                skill=1 - pb_model / pb_base,
            )
        )
skilldf = pd.DataFrame(skill_rows)

# ── Plot ──
C_log, C_f = "#C44E52", "#4C72B0"
tau_arr = np.array(taus)
fig, ax = plt.subplots(2, 3, figsize=(15, 9))

# a) reliability: nominal vs empirical coverage
a = ax[0, 0]
for tgt, c in [("log_abs", C_log), ("f_ratio", C_f)]:
    emp = [covdf.loc[covdf.target == tgt, f"cov_tau{int(t * 100)}"].values[0] for t in taus]
    a.plot(tau_arr, emp, "o-", color=c, lw=1.8, ms=7, label=tgt)
a.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
a.set_xlabel("Nominal quantile τ")
a.set_ylabel("Empirical coverage P(y≤q̂)")
a.set_title("Calibration (seed-grouped hold-out)")
a.legend(fontsize=8)
a.set_xlim(0, 1)
a.set_ylim(0, 1)

# b) pinball skill score by tau
a = ax[0, 1]
for tgt, c in [("log_abs", C_log), ("f_ratio", C_f)]:
    s = skilldf[skilldf.target == tgt].sort_values("tau")
    a.plot(s.tau, s.skill, "s-", color=c, lw=1.8, ms=7, label=tgt)
a.set_xlabel("Quantile τ")
a.set_ylabel("Pinball skill (vs marginal)")
a.set_title("Predictive value peaks at the tails")
a.legend(fontsize=8)
a.axhline(0, color="k", lw=0.6)

# c) PI width heteroscedasticity across test points
a = ax[0, 2]
for tgt, c in [("log_abs", C_log), ("f_ratio", C_f)]:
    w = qpred[tgt][0.95] - qpred[tgt][0.05]
    a.hist(
        w / np.median(w),
        bins=60,
        color=c,
        alpha=0.5,
        label=f"{tgt} ({w.max() / w.min():.0f}× range)",
    )
a.set_xlabel("90% PI width / median width")
a.set_ylabel("Test-point count")
a.set_title("Heteroscedastic interval widths")
a.legend(fontsize=8)

# d, e) test prediction: predicted median vs actual
for i, (tgt, c) in enumerate([("log_abs", C_log), ("f_ratio", C_f)]):
    a = ax[1, i]
    y = yte[tgt]
    med = qpred[tgt][0.5]
    w = qpred[tgt][0.95] - qpred[tgt][0.05]
    idx = np.random.RandomState(0).choice(len(y), min(8000, len(y)), replace=False)
    sc = a.scatter(med[idx], y[idx], c=w[idx], s=5, cmap="viridis", alpha=0.4)
    lo, hi = np.percentile(y, [0.5, 99.5])
    a.plot([lo, hi], [lo, hi], "k--", lw=1.2)
    a.set_xlabel(f"Predicted median {tgt}")
    a.set_ylabel(f"Actual {tgt}")
    r2 = r2_score(y, med)
    a.set_title(f"{tgt}: R²(median)={r2:.3f}")
    plt.colorbar(sc, ax=a, label="90% PI width", fraction=0.046, pad=0.04)

# f) coverage bars
a = ax[1, 2]
xx = np.arange(2)
w = 0.35
c90 = [covdf.loc[covdf.target == t, "cov90"].values[0] for t in ["log_abs", "f_ratio"]]
c50 = [covdf.loc[covdf.target == t, "cov50"].values[0] for t in ["log_abs", "f_ratio"]]
a.bar(xx - w / 2, c90, w, color="#55A868", label="90% PI")
a.bar(xx + w / 2, c50, w, color="#DD8452", label="50% PI")
a.axhline(0.90, ls=":", color="#55A868")
a.axhline(0.50, ls=":", color="#DD8452")
a.set_xticks(xx)
a.set_xticklabels(["log_abs", "f_ratio"])
a.set_ylabel("Empirical coverage")
a.set_title("Interval coverage vs nominal")
a.legend(fontsize=8)
a.set_ylim(0, 1)

for j, axx in enumerate(ax.flat):
    panel_letter(axx, string.ascii_lowercase[j])

fig.suptitle(
    "Pooled quantile GBM across 101 recorders: calibrated intervals, tail-focused predictive value",
    fontsize=12.5,
    y=1.01,
)
fig.tight_layout()
fig.savefig(result_path("plots", "quantile_101ch_performance.png"), dpi=150, bbox_inches="tight")
print("saved quantile_101ch_performance.png")
