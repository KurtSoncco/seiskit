"""Per-quantile Monte Carlo error and seed convergence at n=100 (channel 50).

Produces a 2×3 panel figure: exact MC error per quantile, convergence curves,
and the number of seeds required per quantile — for both ln(abs TF) and
f_ratio.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from seiskit.plot_config import apply_style, get_crameri_cmap, panel_letter, result_path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FACTORS, FIG_DPI, FIG_WIDTH, figsize, load_channel50, target_color, target_label

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
d50 = load_channel50()

d = d50.copy()
d["cell"] = d.groupby(FACTORS).ngroup()

taus = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
rng = np.random.default_rng(0)
N = 100  # seeds per cell

# ---------------------------------------------------------------------------
# Exact per-quantile Monte Carlo error at n=100
# ---------------------------------------------------------------------------
B = 1000


def per_quantile_error(tgt):
    cells = d["cell"].unique()
    boot_se = np.zeros((len(cells), len(taus)))
    anal_se = np.zeros((len(cells), len(taus)))
    qhat = np.zeros((len(cells), len(taus)))
    for ci, c in enumerate(cells):
        v = d.loc[d.cell == c, tgt].values
        idx = rng.integers(0, N, size=(B, N))
        bs = v[idx]
        bq = np.quantile(bs, taus, axis=1).T
        boot_se[ci] = bq.std(0)
        qhat[ci] = np.quantile(v, taus)
        try:
            kde = gaussian_kde(v)
            f = kde(qhat[ci])
            f[f < 1e-6] = 1e-6
            anal_se[ci] = np.sqrt(np.array(taus) * (1 - np.array(taus))) / (f * np.sqrt(N))
        except Exception:
            anal_se[ci] = np.nan
    return boot_se, anal_se, qhat


err = {}
for tgt in ["log_abs", "f_ratio"]:
    err[tgt] = per_quantile_error(tgt)

rows = []
for tgt in ["log_abs", "f_ratio"]:
    bse, ase, qh = err[tgt]
    for j, tau in enumerate(taus):
        iqr = d[tgt].quantile(0.75) - d[tgt].quantile(0.25)
        rows.append(
            dict(
                target=tgt,
                tau=tau,
                boot_SE_med=np.median(bse[:, j]),
                boot_SE_p90=np.percentile(bse[:, j], 90),
                analytic_SE_med=np.nanmedian(ase[:, j]),
                SE_rel_to_IQR_pct=100 * np.median(bse[:, j]) / iqr,
            )
        )
qerr = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Required seeds per quantile
# ---------------------------------------------------------------------------
rows = []
for tgt in ["log_abs", "f_ratio"]:
    sub = qerr[qerr.target == tgt].set_index("tau")
    se_med = sub.loc[0.50, "boot_SE_med"]
    for tau in taus:
        se = sub.loc[tau, "boot_SE_med"]
        n_match = 100 * (se / se_med) ** 2
        iqr = d[tgt].quantile(0.75) - d[tgt].quantile(0.25)
        se_target = 0.05 * iqr
        n_5pct = 100 * (se / se_target) ** 2
        rows.append(
            dict(
                target=tgt,
                tau=tau,
                n_to_match_median=int(np.ceil(n_match)),
                n_for_SE_5pct_IQR=int(np.ceil(n_5pct)),
            )
        )
nreq_q = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Compute convergence data (bootstrap at varying n)
# ---------------------------------------------------------------------------
ns_grid = np.array([10, 20, 30, 50, 75, 100])
B_conv = 200
conv_q: dict[str, np.ndarray] = {}
for tgt in ["log_abs", "f_ratio"]:
    cells = d["cell"].unique()
    med_se = np.zeros((len(ns_grid), len(taus)))
    for ni, n in enumerate(ns_grid):
        cell_se = np.zeros((len(cells), len(taus)))
        for ci, c in enumerate(cells):
            v = d.loc[d.cell == c, tgt].values
            idx = rng.integers(0, N, size=(B_conv, n))
            bs = v[idx]
            bq = np.quantile(bs, taus, axis=1).T
            cell_se[ci] = bq.std(0)
        med_se[ni] = np.median(cell_se, axis=0)
    conv_q[tgt] = med_se

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
apply_style(auto_format=True, font_size=10, frame="open")

tcol = {"log_abs": target_color("log_abs"), "f_ratio": target_color("f_ratio")}
nice = {"log_abs": target_label("log_abs"), "f_ratio": target_label("f_ratio")}
qcmap = get_crameri_cmap("batlow")(np.linspace(0, 0.9, len(taus)))

fig, axes = plt.subplots(2, 3, figsize=figsize(height=FIG_WIDTH * 0.75))
for r, tgt in enumerate(["log_abs", "f_ratio"]):
    sub = qerr[qerr.target == tgt].set_index("tau")

    # col 0: exact MC error per quantile at n=100, bootstrap vs analytic
    ax = axes[r, 0]
    ax.plot(
        taus, sub["boot_SE_med"], "o-", color=tcol[tgt], lw=2, ms=5, label="bootstrap SE (exact)"
    )
    ax.plot(
        taus, sub["analytic_SE_med"], "s--", color="0.4", lw=1.4, ms=4, label="analytic SE (KDE)"
    )
    ax.fill_between(
        taus,
        sub["boot_SE_med"],
        sub["boot_SE_p90"],
        color=tcol[tgt],
        alpha=0.15,
        label="median→p90 cell",
    )
    ax.set_xlabel("quantile τ")
    ax.set_ylabel("Monte Carlo SE at n=100")
    ax.set_title(f"{nice[tgt]}: exact per-quantile error\n(n=100 seeds)", fontsize=10)
    ax.legend(fontsize=6.5, frameon=False)
    panel_letter(ax, chr(97 + r * 3))

    # col 1: convergence curves per quantile
    ax = axes[r, 1]
    for j, tau in enumerate(taus):
        ax.plot(ns_grid, conv_q[tgt][:, j], "o-", color=qcmap[j], ms=3, lw=1.5, label=f"τ={tau:g}")
    ax.axvline(100, color="green", ls=":", lw=1.2)
    ax.set_xlabel("number of seeds n")
    ax.set_ylabel("median-cell MC SE")
    ax.set_title(f"{nice[tgt]}: convergence vs n\n(√n reference below)", fontsize=10)
    ref = conv_q[tgt][0, 3] * np.sqrt(ns_grid[0] / np.array(ns_grid))
    ax.plot(ns_grid, ref, "k-", lw=0.7, alpha=0.4)
    if r == 0:
        ax.legend(fontsize=6, frameon=False, ncol=2)
    panel_letter(ax, chr(98 + r * 3))

    # col 2: required n per quantile
    ax = axes[r, 2]
    nr = nreq_q[nreq_q.target == tgt].set_index("tau")
    x = np.arange(len(taus))
    w = 0.4
    ax.bar(
        x - w / 2,
        nr["n_to_match_median"],
        w,
        color=tcol[tgt],
        alpha=0.9,
        label="match median precision",
    )
    ax.bar(
        x + w / 2, nr["n_for_SE_5pct_IQR"], w, color=tcol[tgt], alpha=0.45, label="SE = 5% of IQR"
    )
    ax.axhline(100, color="green", ls="--", lw=1.2, label="n=100 (used)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:g}" for t in taus], fontsize=7)
    ax.set_xlabel("quantile τ")
    ax.set_ylabel("seeds required")
    ax.set_title(f"{nice[tgt]}: seeds needed per quantile", fontsize=10)
    ax.legend(fontsize=6.5, frameon=False)
    panel_letter(ax, chr(99 + r * 3))

fig.tight_layout()
fig.savefig(result_path("plots", "quantile_seed_error.png"), dpi=FIG_DPI, bbox_inches="tight")
plt.close(fig)
print("saved quantile_seed_error.png")
