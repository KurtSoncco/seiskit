"""Interpolation vs extrapolation prediction scatter plots (Figure).

Produces a 2×2 panel figure showing predicted vs actual values for
three model types under interpolation and extrapolation conditions.
"""

import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FACTORS, load_channel50

from seiskit.plot_config import apply_style, panel_letter, result_path

apply_style(auto_format=True, font_size=10, frame="open")

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
d50 = load_channel50()
fac = FACTORS

d = d50.copy()
d["r_v"] = d["rH"] / d["aHV"]
d["ln_f0"] = np.log(d["Vs1"] / (4 * d["Height"]))
d["n_vert"] = d["Height"] / d["r_v"]
d["ln_nvert"] = np.log(d["n_vert"])
d["scatter"] = d["CoV"] * np.sqrt(d["n_vert"])


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------
def phys_design(df):
    return np.column_stack(
        [
            df["ln_f0"],
            df["ln_f0"] ** 2,
            df["ln_nvert"],
            df["CoV"],
            df["scatter"],
            df["ln_f0"] * df["CoV"],
        ]
    )


def preds(tgt, factor, mode):
    lv = sorted(d[factor].unique())
    if mode == "interp":
        tr = d[factor].isin([lv[0], lv[2]])
        te = d[factor] == lv[1]
    else:
        tr = d[factor].isin([lv[0], lv[1]])
        te = d[factor] == lv[2]
    dtr, dte = d[tr], d[te]
    ytr, yte = dtr[tgt].values, dte[tgt].values

    out = {"y": yte}

    ols = make_pipeline(
        StandardScaler(), PolynomialFeatures(2, include_bias=False), LinearRegression()
    ).fit(dtr[fac].values, ytr)
    out["OLS"] = ols.predict(dte[fac].values)

    ph = make_pipeline(StandardScaler(), LinearRegression()).fit(phys_design(dtr), ytr)
    out["Physics"] = ph.predict(phys_design(dte))

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
    ).fit(dtr[fac].values, ytr)
    out["GBM"] = gb.predict(dte[fac].values)

    return out


# ---------------------------------------------------------------------------
# Generate predictions
# ---------------------------------------------------------------------------
pred_data = {}
for tgt, factor in [("log_abs", "Height"), ("f_ratio", "aHV")]:
    for mode in ["interp", "extrap"]:
        pred_data[(tgt, factor, mode)] = preds(tgt, factor, mode)


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
mcol = {"Physics": "#55A868", "GBM": "#C44E52", "OLS": "#4C72B0"}
nice = {"log_abs": "log(abs_TF)", "f_ratio": "f_ratio"}

fig, axes = plt.subplots(2, 2, figsize=(10, 9.6))
specs = [("log_abs", "Height"), ("f_ratio", "aHV")]
modes = ["interp", "extrap"]
mode_lbl = {
    "interp": "Interpolation (predict middle level)",
    "extrap": "Extrapolation (predict beyond range)",
}

for r, (tgt, factor) in enumerate(specs):
    for c, mode in enumerate(modes):
        ax = axes[r, c]
        pd_ = pred_data[(tgt, factor, mode)]
        y = pd_["y"]
        lo, hi = y.min(), y.max()
        pad = (hi - lo) * 0.35
        for m in ["Physics", "GBM", "OLS"]:
            p = pd_[m]
            ax.scatter(
                y,
                p,
                s=8,
                alpha=0.35,
                color=mcol[m],
                edgecolor="none",
                label=f"{m} (R²={r2(y, p):.2f})",
            )
        lim = [lo - pad, hi + pad]
        ax.plot(lim, lim, "k--", lw=1, alpha=0.6)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"actual {nice[tgt]}")
        ax.set_ylabel("predicted")
        ax.set_title(f"{nice[tgt]} — vary {factor}\n{mode_lbl[mode]}", fontsize=10)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        panel_letter(ax, chr(97 + r * 2 + c))

fig.suptitle(
    "Model predictions on held-out design levels: interpolation vs extrapolation (channel 50)",
    fontsize=12,
    y=1.005,
)
fig.tight_layout()
fig.savefig(result_path("plots", "interp_extrap_predictions.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved interp_extrap_predictions.png")
