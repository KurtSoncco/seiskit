"""Quantile regression coefficients for log(abs TF ratio) across quantiles.

Produces a 2×3 panel figure showing how each standardised factor's coefficient
varies from τ=0.05 to τ=0.95, with 95 % seed-cluster bootstrap CIs and a
dashed OLS/mixed-model reference line.
"""

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from seiskit.plot_config import apply_style, panel_letter, result_path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FACTORS, FIG_DPI, FIG_WIDTH, factor_color, figsize, load_channel50  # noqa: E402

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
d50 = load_channel50()

# Standardize factors
Z = d50[FACTORS].copy()
Zs = (Z - Z.mean()) / Z.std()
Zs.columns = [c + "_z" for c in FACTORS]
d50 = pd.concat([d50, Zs], axis=1)

zcols = [c + "_z" for c in FACTORS]

main = " + ".join(zcols)
inter = " + ".join([f"{a}:{b}" for i, a in enumerate(zcols) for b in zcols[i + 1 :]])
formula_rhs = main + " + " + inter
main_formula = " + ".join(zcols)

# Within-cell residuals for seed ICC
d50["log_abs_resid"] = d50["log_abs"] - d50.groupby(FACTORS)["log_abs"].transform("mean")
d50["f_resid"] = d50["f_ratio"] - d50.groupby(FACTORS)["f_ratio"].transform("mean")

# Build cell id
d50["cell"] = d50.groupby(FACTORS).ngroup()


# ---------------------------------------------------------------------------
# OLS models with cluster-robust SEs
# ---------------------------------------------------------------------------
def fit_ols_cluster(target):
    f = f"{target} ~ {formula_rhs}"
    m = smf.ols(f, data=d50).fit()
    mc = smf.ols(f, data=d50).fit(cov_type="cluster", cov_kwds={"groups": d50["seed"]})
    return m, mc


results = {}
for tgt in ["log_abs", "f_ratio"]:
    m, mc = fit_ols_cluster(tgt)
    results[tgt] = (m, mc)

# ---------------------------------------------------------------------------
# Mixed model for log_abs
# ---------------------------------------------------------------------------
mm_full = smf.mixedlm(f"log_abs ~ {formula_rhs}", d50, groups=d50["seed"]).fit(
    method="powell", reml=True
)
vseed = float(mm_full.cov_re.iloc[0, 0])
vres = float(mm_full.scale)
Xb = np.asarray(mm_full.predict(d50))
var_fixed = np.var(Xb)
R2_marg = var_fixed / (var_fixed + vseed + vres)
R2_cond = (var_fixed + vseed) / (var_fixed + vseed + vres)

# ---------------------------------------------------------------------------
# Quantile regression point estimates
# ---------------------------------------------------------------------------
taus = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def qr_points(target):
    out = {}
    for t in taus:
        m = smf.quantreg(f"{target} ~ {main_formula}", d50).fit(q=t, max_iter=2000)
        out[t] = m.params
    return pd.DataFrame(out).T


qr_log = qr_points("log_abs")
qr_f = qr_points("f_ratio")

# ---------------------------------------------------------------------------
# Cluster bootstrap
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
seeds = np.arange(1, 101)
seed_groups = {s: d50.index[d50["seed"] == s].values for s in seeds}
n_boot = 100

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cluster_bootstrap_qr(target, n_boot=100):
    cache = CACHE_DIR / f"boot_qr_{target}.npz"
    pcols = ["Intercept"] + zcols
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        recs = []
        for t in taus:
            arr = z[f"tau_{t}"]
            for j, p in enumerate(pcols):
                recs.append(
                    dict(
                        tau=t,
                        term=p,
                        se=np.nanstd(arr[:, j]),
                        lo=np.nanpercentile(arr[:, j], 2.5),
                        hi=np.nanpercentile(arr[:, j], 97.5),
                    )
                )
        return pd.DataFrame(recs)

    boot = {t: [] for t in taus}
    for b in range(n_boot):
        if (b + 1) % 20 == 0:
            print(f"  bootstrap {b + 1}/{n_boot}", flush=True)
        pick = rng.choice(seeds, size=100, replace=True)
        idx = np.concatenate([seed_groups[s] for s in pick])
        dfb = d50.loc[idx]
        for t in taus:
            try:
                m = smf.quantreg(f"{target} ~ {main_formula}", dfb).fit(q=t, max_iter=500)
                boot[t].append(m.params[pcols].values)
            except Exception:
                boot[t].append([np.nan] * len(pcols))
    arrays = {f"tau_{t}": np.array(boot[t]) for t in taus}
    np.savez(cache, **arrays)

    recs = []
    for t in taus:
        arr = arrays[f"tau_{t}"]
        for j, p in enumerate(pcols):
            recs.append(
                dict(
                    tau=t,
                    term=p,
                    se=np.nanstd(arr[:, j]),
                    lo=np.nanpercentile(arr[:, j], 2.5),
                    hi=np.nanpercentile(arr[:, j], 97.5),
                )
            )
    return pd.DataFrame(recs)


boot_log = cluster_bootstrap_qr("log_abs", n_boot)
boot_f = cluster_bootstrap_qr("f_ratio", n_boot)


# ---------------------------------------------------------------------------
# Merge point estimates with bootstrap CIs
# ---------------------------------------------------------------------------
def to_long(qr_df, target):
    r = []
    for t in qr_df.index:
        for p in ["Intercept"] + zcols:
            r.append(dict(target=target, tau=t, term=p, coef=qr_df.loc[t, p]))
    return pd.DataFrame(r)


qr_long = pd.concat([to_long(qr_log, "log_abs"), to_long(qr_f, "f_ratio")], ignore_index=True)
boot_log["target"] = "log_abs"
boot_f["target"] = "f_ratio"
boot_all = pd.concat([boot_log, boot_f], ignore_index=True)
qr_full = qr_long.merge(boot_all, on=["target", "tau", "term"], how="left")
qr_full["sig_95"] = ~((qr_full["lo"] <= 0) & (qr_full["hi"] >= 0))

ols_ref = {
    "log_abs": {c: mm_full.params[c] for c in zcols},
    "f_ratio": {c: results["f_ratio"][0].params[c] for c in zcols},
}

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
apply_style(auto_format=True, font_size=10, frame="open")

factor_col = {f"{f}_z": factor_color(f) for f in FACTORS}


def qr_figure(target, qr_pts, ref, fname, ttl):
    fig, axes = plt.subplots(2, 3, figsize=figsize(height=FIG_WIDTH * 0.75))
    axes = axes.ravel()
    for k, fc in enumerate(zcols):
        ax = axes[k]
        sub = qr_full[(qr_full.target == target) & (qr_full.term == fc)].sort_values("tau")
        ax.plot(sub["tau"], sub["coef"], "o-", color=factor_col[fc], ms=5, lw=1.5, zorder=3)
        ax.fill_between(
            sub["tau"], sub["lo"], sub["hi"], color=factor_col[fc], alpha=0.22, zorder=1
        )
        ax.axhline(0, color="0.4", ls="-", lw=0.8)
        ax.axhline(ref[fc], color="0.3", ls="--", lw=1.2, zorder=2, label="mean-model")
        ax.set_title(fc.replace("_z", ""), loc="left", fontsize=10)
        ax.set_xlabel(r"quantile $\tau$")
        ax.set_ylabel("coefficient")
        if k == 0:
            ax.legend(fontsize=6.5, frameon=False, loc="best")
        panel_letter(ax, "abcdef"[k])
    ax = axes[5]
    ax.axis("off")
    lines = [
        "Reading the curves:",
        "",
        "• Flat curve = pure MEAN (location) effect",
        "  → same shift at every quantile",
        "• Sloped/sign-flipping = VARIANCE effect",
        "  → widens or narrows the distribution",
        "• Dashed line = single mean-model coef",
        "  (what OLS/mixed reports)",
    ]
    ax.text(
        0.02,
        0.95,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
        family="DejaVu Sans",
    )
    fig.tight_layout()
    fig.savefig(fname, dpi=FIG_DPI, bbox_inches="tight")
    print("saved", fname)


qr_figure(
    "log_abs",
    qr_log,
    ols_ref["log_abs"],
    result_path("plots", "quantile_coefficients_abs_TF.png"),
    "Quantile regression: log(abs TF ratio) — coefficients across quantiles "
    "(95% seed-cluster bootstrap CI)",
)
