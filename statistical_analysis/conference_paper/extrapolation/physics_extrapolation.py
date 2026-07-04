"""Physics-informed reparameterization for extrapolation (Figure).

Compares physics-feature linear model vs raw OLS vs GBM on interpolation
and extrapolation splits of the channel-50 dataset.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_channel50, FACTORS, TARGETS

from seiskit.plot_config import apply_style, panel_letter, result_path

apply_style(auto_format=True, font_size=10, frame="open")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
d50 = load_channel50()
fac = FACTORS

# Physics-motivated features
d = d50.copy()
d["r_v"] = d["rH"] / d["aHV"]
d["f0"] = d["Vs1"] / (4 * d["Height"])
d["ln_f0"] = np.log(d["f0"])
d["n_vert"] = d["Height"] / d["r_v"]
d["ln_nvert"] = np.log(d["n_vert"])
d["scatter"] = d["CoV"] * np.sqrt(d["n_vert"])
phys_feats = ["ln_f0", "ln_nvert", "CoV", "scatter"]

levels = {f: sorted(d[f].unique()) for f in fac}


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def phys_design(df):
    X = np.column_stack(
        [
            df["ln_f0"],
            df["ln_f0"] ** 2,
            df["ln_nvert"],
            df["CoV"],
            df["scatter"],
            df["ln_f0"] * df["CoV"],
        ]
    )
    return X


def run(tgt, factor, mode):
    lv = levels[factor]
    if mode == "interp":
        tr = d[factor].isin([lv[0], lv[2]])
        te = d[factor] == lv[1]
    else:
        tr = d[factor].isin([lv[0], lv[1]])
        te = d[factor] == lv[2]
    dtr, dte = d[tr], d[te]
    ytr, yte = dtr[tgt].values, dte[tgt].values

    # Raw OLS with interactions
    ols = make_pipeline(
        StandardScaler(), PolynomialFeatures(2, include_bias=False), LinearRegression()
    )
    ols.fit(dtr[fac].values, ytr)
    p_ols = ols.predict(dte[fac].values)

    # Physics linear
    ph = make_pipeline(StandardScaler(), LinearRegression())
    ph.fit(phys_design(dtr), ytr)
    p_ph = ph.predict(phys_design(dte))

    # GBM
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
    gb.fit(dtr[fac].values, ytr)
    p_gb = gb.predict(dte[fac].values)

    return dict(
        target=tgt,
        factor=factor,
        mode=mode,
        R2_physics=round(r2(yte, p_ph), 3),
        R2_OLS=round(r2(yte, p_ols), 3),
        R2_GBM=round(r2(yte, p_gb), 3),
    )


# ---------------------------------------------------------------------------
# Run all splits
# ---------------------------------------------------------------------------
res = []
for tgt, facs in [("log_abs", ["Vs1", "Height"]), ("f_ratio", ["CoV", "aHV"])]:
    for f in facs:
        for m in ["interp", "extrap"]:
            res.append(run(tgt, f, m))
phys_df = pd.DataFrame(res)
phys_df.to_csv(result_path("data", "physics_model_extrap.csv"), index=False)

phys_df = pd.read_csv(result_path("data", "physics_model_extrap.csv"))
ext = phys_df[phys_df["mode"] == "extrap"].copy()
ext["label"] = ext.target + "\n" + ext.factor

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# Panel a: extrapolation R2 by model
ax = axes[0]
x = np.arange(len(ext))
w = 0.26
mods = [
    ("R2_physics", "physics features", "#55A868"),
    ("R2_GBM", "GBM (trees)", "#C44E52"),
    ("R2_OLS", "raw OLS+interactions", "#4C72B0"),
]
clip = -1.5
for i, (c, lab, col) in enumerate(mods):
    vals = ext[c].values
    disp = np.clip(vals, clip, 1)
    b = ax.bar(x + (i - 1) * w, disp, w, color=col, label=lab)
    for xi, (v, dv) in enumerate(zip(vals, disp)):
        if v < clip:
            ax.text(
                xi + (i - 1) * w,
                clip + 0.05,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=6.5,
                rotation=90,
                color="white",
                fontweight="bold",
            )
ax.axhline(0, color="0.4", lw=0.8)
ax.set_ylim(clip, 1)
ax.set_xticks(x)
ax.set_xticklabels(ext["label"].values, fontsize=7.5)
ax.set_ylabel("extrapolation R² (clipped at −1.5)")
ax.set_title("Extrapolate beyond trained range:\nphysics features stay bounded", fontsize=10)
ax.legend(fontsize=7, frameon=False, loc="lower left")
panel_letter(ax, "a")

# Panel b: interp vs extrap for physics model only
ax = axes[1]
pv = phys_df.pivot_table(
    index=["target", "factor"], columns="mode", values="R2_physics"
).reset_index()
lbls = (pv.target + "\n" + pv.factor).values
xx = np.arange(len(pv))
ax.bar(xx - 0.2, pv["interp"], 0.4, color="#8172B3", label="interpolate (fill gap)")
ax.bar(xx + 0.2, pv["extrap"], 0.4, color="#DD8452", label="extrapolate (beyond range)")
ax.axhline(0, color="0.4", lw=0.8)
ax.set_xticks(xx)
ax.set_xticklabels(lbls, fontsize=7.5)
ax.set_ylabel("R² (physics model)")
ax.set_title("Physics model: interpolation vs extrapolation", fontsize=10)
ax.legend(fontsize=7, frameon=False)
panel_letter(ax, "b")

fig.suptitle(
    "Physics-informed reparameterization closes the extrapolation gap (channel 50)",
    fontsize=12,
    y=1.02,
)
fig.tight_layout()
fig.savefig(result_path("plots", "physics_extrapolation.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved physics_extrapolation.png")
