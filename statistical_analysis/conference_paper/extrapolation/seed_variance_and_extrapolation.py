"""Seed variance decomposition and model interpolation/extrapolation limits (Figure).

Produces a 2×3 panel figure showing variance decomposition, seed-effect
reproducibility, spread attribution, and extrapolation degradation curves.
"""

import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from seiskit.plot_config import apply_style, panel_letter, result_path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    COMPARE_COLOR,
    FACTOR_COLORS,
    FACTORS,
    FIG_DPI,
    FIG_WIDTH,
    REF_COLOR,
    figsize,
    load_channel50,
    target_color,
)

apply_style(auto_format=True, font_size=10, frame="open")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
d50 = load_channel50()
fac = FACTORS

d = d50.copy()
d["cell"] = d.groupby(FACTORS).ngroup()
C = d["cell"].nunique()
S = d["seed"].nunique()


# ---------------------------------------------------------------------------
# Two-way variance decomposition
# ---------------------------------------------------------------------------
def twoway_components(y):
    df = pd.DataFrame({"cell": d["cell"].values, "seed": d["seed"].values, "y": y})
    gm = df["y"].mean()
    cm = df.groupby("cell")["y"].transform("mean")
    sm = df.groupby("seed")["y"].transform("mean")
    resid = df["y"] - cm - sm + gm
    SS_cell = ((df.groupby("cell")["y"].mean() - gm) ** 2).sum() * S
    SS_seed = ((df.groupby("seed")["y"].mean() - gm) ** 2).sum() * C
    SS_res = (resid**2).sum()
    MS_cell = SS_cell / (C - 1)
    MS_seed = SS_seed / (S - 1)
    MS_res = SS_res / ((C - 1) * (S - 1))
    var_seed = max(0, (MS_seed - MS_res) / C)
    var_cell = max(0, (MS_cell - MS_res) / S)
    var_res = MS_res
    tot = var_cell + var_seed + var_res
    return dict(
        var_cell=var_cell,
        var_seed=var_seed,
        var_resid=var_res,
        pct_cell=100 * var_cell / tot,
        pct_seed=100 * var_seed / tot,
        pct_resid=100 * var_res / tot,
        F_seed=MS_seed / MS_res,
        resid=resid.values,
        seed_means=df.groupby("seed")["y"].mean().values - gm,
    )


comp = {}
for tgt, lbl in [("log_abs", "log(abs_TF)"), ("f_ratio", "f_ratio")]:
    comp[tgt] = twoway_components(d[tgt].values)

rows = []
for tgt in ["log_abs", "f_ratio"]:
    cm = d.groupby("cell")[tgt].transform("mean")
    dev = d[tgt] - cm
    g = pd.DataFrame({"cell": d["cell"], "dev": dev, "cm": cm})
    cell_std = g.groupby("cell")["dev"].std()
    cell_mu = g.groupby("cell")["cm"].first()
    r, p = stats.pearsonr(cell_mu, np.log(cell_std))
    piv = d.pivot_table(index="seed", columns="cell", values=tgt)
    piv_c = piv.sub(piv.mean(0), axis=1)
    cols = np.array(piv_c.columns.values, copy=True)
    rng = np.random.default_rng(0)
    rng.shuffle(cols)
    h1, h2 = cols[: len(cols) // 2], cols[len(cols) // 2 :]
    seed_eff1 = piv_c[h1].mean(1)
    seed_eff2 = piv_c[h2].mean(1)
    r_split, _ = stats.pearsonr(seed_eff1, seed_eff2)
    rows.append(
        dict(
            target=tgt,
            r_cellmean_vs_logstd=round(r, 4),
            p_val=p,
            seed_split_half_r=round(r_split, 4),
        )
    )
comp_df = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Interpolation / extrapolation splits
# ---------------------------------------------------------------------------
levels = {f: sorted(d[f].unique()) for f in fac}


def make_Xy(df, tgt):
    return df[fac].values, df[tgt].values


def run_split(tgt, factor, mode):
    lv = levels[factor]
    if mode == "interp":
        train_mask = d[factor].isin([lv[0], lv[2]])
        test_mask = d[factor] == lv[1]
    else:
        train_mask = d[factor].isin([lv[0], lv[1]])
        test_mask = d[factor] == lv[2]
    Xtr, ytr = make_Xy(d[train_mask], tgt)
    Xte, yte = make_Xy(d[test_mask], tgt)

    ols = make_pipeline(
        StandardScaler(), PolynomialFeatures(2, include_bias=False), LinearRegression()
    )
    ols.fit(Xtr, ytr)
    p_ols = ols.predict(Xte)

    gb = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=0,
        verbose=-1,
    )
    gb.fit(Xtr, ytr)
    p_gb = gb.predict(Xte)

    def r2(y, p):
        ss = ((y - p) ** 2).sum()
        return 1 - ss / ((y - y.mean()) ** 2).sum()

    return dict(
        target=tgt,
        factor=factor,
        mode=mode,
        n_test=int(test_mask.sum()),
        R2_OLS=round(r2(yte, p_ols), 3),
        R2_GBM=round(r2(yte, p_gb), 3),
        mean_true=round(yte.mean(), 3),
        mean_gb=round(p_gb.mean(), 3),
        mean_ols=round(p_ols.mean(), 3),
    )


results = []
for tgt, facs_ in [("log_abs", ["Vs1", "Height"]), ("f_ratio", ["CoV", "aHV"])]:
    for f in facs_:
        for mode in ["interp", "extrap"]:
            results.append(run_split(tgt, f, mode))
extrap_df = pd.DataFrame(results)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=figsize(height=FIG_WIDTH * 0.75))

# Panel a: variance decomposition
ax = axes[0, 0]
x = np.arange(2)
w = 0.6
bottom = np.zeros(2)
segs = {
    "design": [comp["log_abs"]["pct_cell"], comp["f_ratio"]["pct_cell"]],
    "seed": [comp["log_abs"]["pct_seed"], comp["f_ratio"]["pct_seed"]],
    "field": [comp["log_abs"]["pct_resid"], comp["f_ratio"]["pct_resid"]],
}
cols_seg = {"design": COMPARE_COLOR, "seed": FACTOR_COLORS["aHV"], "field": REF_COLOR}
for k in ["design", "seed", "field"]:
    ax.bar(x, segs[k], w, bottom=bottom, label=k, color=cols_seg[k], edgecolor="white")
    for xi, (b, v) in enumerate(zip(bottom, segs[k])):
        if v > 4:
            ax.text(
                xi,
                b + v / 2,
                f"{v:.0f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if k != "field" else "0.2",
                fontweight="bold",
            )
    bottom = bottom + np.array(segs[k])
ax.set_xticks(x)
ax.set_xticklabels(["log(abs_TF)", "f_ratio"])
ax.set_ylabel("% of total variance")
ax.set_ylim(0, 100)
ax.set_title("Variance decomposition\n(crossed design × seed)", fontsize=10)
ax.legend(fontsize=7, loc="upper right", frameon=False)
panel_letter(ax, "a")

# Panel b: seed effect reproducibility
ax = axes[0, 1]
for tgt, cc in [("log_abs", target_color("log_abs")), ("f_ratio", target_color("f_ratio"))]:
    piv = d.pivot_table(index="seed", columns="cell", values=tgt)
    piv_c = piv.sub(piv.mean(0), axis=1)
    cols = np.array(piv_c.columns.values, copy=True)
    rng = np.random.default_rng(0)
    rng.shuffle(cols)
    h1, h2 = cols[: len(cols) // 2], cols[len(cols) // 2 :]
    lbl = "log(abs_TF)" if tgt == "log_abs" else "f_ratio"
    r_val = comp_df.set_index("target").loc[tgt, "seed_split_half_r"]
    ax.scatter(
        piv_c[h1].mean(1),
        piv_c[h2].mean(1),
        s=14,
        alpha=0.6,
        color=cc,
        label=f"{lbl} (r={r_val:.2f})",
    )
ax.set_xlabel("seed effect, design half A")
ax.set_ylabel("seed effect, design half B")
ax.set_title("Seed effect reproducibility\n(split-half over design cells)", fontsize=10)
ax.legend(fontsize=7, frameon=False)
ax.axhline(0, color="0.7", lw=0.6)
ax.axvline(0, color="0.7", lw=0.6)
panel_letter(ax, "b")

# Panel c: spread attribution
ax = axes[0, 2]
en = {"Vs1": 0.299, "Height": 0.099, "CoV": 0.144, "rH": 0.060, "aHV": 0.255}
ef = {"Vs1": 0.301, "Height": 0.189, "CoV": 0.084, "rH": 0.021, "aHV": 0.192}
xf = np.arange(5)
ax.bar(xf - 0.2, [en[f] for f in fac], 0.4, label="naive (seed+field)", color=COMPARE_COLOR)
ax.bar(xf + 0.2, [ef[f] for f in fac], 0.4, label="field-only (seed removed)", color="#CCB974")
ax.set_xticks(xf)
ax.set_xticklabels(fac, rotation=45, ha="right")
ax.set_ylabel("η² on log(cell variance)")
ax.set_title("log(abs_TF): spread attribution\nshifts when seed removed", fontsize=10)
ax.legend(fontsize=7, frameon=False)
panel_letter(ax, "c")


# Panels d–f: extrapolation degradation
def plot_ie(ax, tgt, factor, letter):
    lv = levels[factor]
    means = d.groupby(factor)[tgt].mean()
    ax.plot(range(3), means.values, "o-", color="0.2", ms=8, label="observed mean", zorder=5)
    r = extrap_df[
        (extrap_df.target == tgt) & (extrap_df.factor == factor) & (extrap_df["mode"] == "extrap")
    ].iloc[0]
    ax.scatter(
        [2],
        [r["mean_gb"]],
        marker="s",
        s=90,
        color=target_color("log_abs"),
        label="GBM extrap",
        zorder=6,
    )
    ax.scatter(
        [2],
        [r["mean_ols"]],
        marker="^",
        s=90,
        color=target_color("f_ratio"),
        label="OLS extrap",
        zorder=6,
    )
    ax.axvspan(-0.3, 1.3, color="0.9", zorder=0)
    ax.text(0.5, ax.get_ylim()[1], "train", ha="center", va="top", fontsize=7, color="0.4")
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{v:g}" for v in lv])
    ax.set_xlabel(factor)
    ax.set_ylabel(f"mean {'log(abs_TF)' if tgt == 'log_abs' else 'f_ratio'}")
    ax.set_title(
        f"Extrapolate to high {factor}\nGBM R²={r['R2_GBM']:.2f}, OLS R²={r['R2_OLS']:.1f}",
        fontsize=9,
    )
    ax.legend(fontsize=6.5, frameon=False, loc="best")
    panel_letter(ax, letter)


plot_ie(axes[1, 0], "log_abs", "Height", "d")
plot_ie(axes[1, 1], "log_abs", "Vs1", "e")
plot_ie(axes[1, 2], "f_ratio", "aHV", "f")

fig.tight_layout()
fig.savefig(
    result_path("plots", "seed_variance_and_extrapolation.png"), dpi=FIG_DPI, bbox_inches="tight"
)
plt.close(fig)
print("saved seed_variance_and_extrapolation.png")
