"""Spatial 101-channel structure figure.

Shows factor effects are stationary across recorders, position sets the
explainability ceiling, and within-realization spatial scatter is a distinct
variance component.
Produces: spatial_101ch_structure.png
"""

import sys
import string
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import lstsq

from seiskit.plot_config import apply_style, panel_letter, result_path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_master, FACTORS

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
full = load_master()
fac = FACTORS

# Per-channel R2 ceiling and mean/spread spatial profile
rows = []
for ch, g in full.groupby("channel"):
    gg = g.groupby(fac)
    for tgt in ["log_abs", "f_ratio"]:
        y = g[tgt].values
        grand = y.mean()
        cmean = gg[tgt].transform("mean")
        ssw = ((y - cmean.values) ** 2).sum()
        sst = ((y - grand) ** 2).sum()
        rows.append(dict(channel=ch, target=tgt, ceiling=(sst - ssw) / sst, mean=grand, sd=y.std()))
spatial = pd.DataFrame(rows)

full["cell"] = full.groupby(fac).ngroup()

# Balanced crossed variance decomposition
N = len(full)
vc = {}
for tgt in ["log_abs", "f_ratio"]:
    y = full[tgt].values
    gm = y.mean()
    sst = ((y - gm) ** 2).sum()

    def ssm(col):
        return ((full.groupby(col)[tgt].transform("mean").values - gm) ** 2).sum()

    ss_cell = ssm("cell")
    ss_ch = ssm("channel")
    ss_seed = ssm("seed")
    cc = full.groupby(["cell", "channel"])[tgt].transform("mean").values
    ss_cellch = ((cc - gm) ** 2).sum() - ss_cell - ss_ch
    csd = full.groupby(["cell", "seed"])[tgt].transform("mean").values
    ss_cellseed = ((csd - gm) ** 2).sum() - ss_cell - ss_seed
    resid = sst - ss_cell - ss_ch - ss_seed - ss_cellch - ss_cellseed
    vc[tgt] = dict(
        design=ss_cell / sst,
        channel=ss_ch / sst,
        seed=ss_seed / sst,
        design_x_channel=ss_cellch / sst,
        design_x_seed=ss_cellseed / sst,
        other_resid=resid / sst,
    )

for tgt in ["log_abs", "f_ratio"]:
    y = full[tgt].values
    gm = y.mean()
    sst = ((y - gm) ** 2).sum()

    def ssm(col):
        return ((full.groupby(col)[tgt].transform("mean").values - gm) ** 2).sum()

    ss_seed = ssm("seed")
    ss_ch = ssm("channel")
    ss_seedch = (
        ((full.groupby(["seed", "channel"])[tgt].transform("mean").values - gm) ** 2).sum()
        - ss_seed
        - ss_ch
    )
    vc[tgt]["seed_x_channel"] = ss_seedch / sst

vcd = pd.DataFrame(vc).T
vcd["threeway_plus"] = vcd["other_resid"] - vcd["seed_x_channel"]

# Factor slope drift across channels
Z = (full[fac] - full[fac].mean()) / full[fac].std()
coef_by_ch = {t: [] for t in ["log_abs", "f_ratio"]}
chs = sorted(full["channel"].unique())
for ch in chs:
    m = full["channel"].values == ch
    Xz = np.column_stack([np.ones(m.sum()), Z.values[m]])
    for tgt in ["log_abs", "f_ratio"]:
        b, *_ = lstsq(Xz, full[tgt].values[m], rcond=None)
        coef_by_ch[tgt].append(b[1:])
coef_by_ch = {t: np.array(v) for t, v in coef_by_ch.items()}

# Within-realization spatial scatter vs seed scatter
rows = []
for tgt in ["log_abs", "f_ratio"]:
    sp = full.groupby(["cell", "seed"])[tgt].std().mean()
    se = full.groupby(["cell", "channel"])[tgt].std().mean()
    tot = full.groupby("cell")[tgt].std().mean()
    rows.append(
        dict(
            target=tgt,
            spatial_within_real=sp,
            seed_within_chan=se,
            total_within_cell=tot,
            spatial_frac=(sp**2) / (tot**2),
            seed_frac=(se**2) / (tot**2),
        )
    )
wr = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
apply_style(auto_format=True, font_size=10, frame="open")

C_log = "#C44E52"
C_f = "#4C72B0"
sp_l = spatial[spatial.target == "log_abs"].sort_values("channel")
sp_f = spatial[spatial.target == "f_ratio"].sort_values("channel")
fig, ax = plt.subplots(2, 3, figsize=(15, 8.8))

# a) ceiling vs channel
a = ax[0, 0]
a.plot(sp_l.channel, sp_l.ceiling, color=C_log, lw=1.8, label="log_abs")
a.axvline(50, ls=":", color="k", lw=1, alpha=0.6)
a.set_xlabel("Recorder (channel)")
a.set_ylabel("R² ceiling", color=C_log)
a.tick_params(axis="y", labelcolor=C_log)
a2 = a.twinx()
a2.plot(sp_f.channel, sp_f.ceiling, color=C_f, lw=1.8)
a2.set_ylabel("R² ceiling (f_ratio)", color=C_f)
a2.tick_params(axis="y", labelcolor=C_f)
a.set_title("Explainability varies with position")
a.text(50, a.get_ylim()[0], " center", fontsize=8, color="k", va="bottom")

# b) mean profile
a = ax[0, 1]
a.plot(sp_l.channel, sp_l["mean"], color=C_log, lw=1.8)
a.set_xlabel("Recorder (channel)")
a.set_ylabel("mean log_abs", color=C_log)
a.tick_params(axis="y", labelcolor=C_log)
a.axvline(50, ls=":", color="k", lw=1, alpha=0.6)
a2 = a.twinx()
a2.plot(sp_f.channel, sp_f["mean"], color=C_f, lw=1.8)
a2.set_ylabel("mean f_ratio", color=C_f)
a2.tick_params(axis="y", labelcolor=C_f)
a.set_title("Spatial mean profile (edge vs center)")

# c) variance budget stacked bar
a = ax[0, 2]
comps = [
    "design",
    "seed",
    "design_x_seed",
    "seed_x_channel",
    "channel",
    "design_x_channel",
    "threeway_plus",
]
labs = [
    "design",
    "seed",
    "design×seed",
    "seed×chan\n(spatial)",
    "channel",
    "design×chan",
    "3-way+resid",
]
cols_ = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#CCCCCC"]
bottom = np.zeros(2)
x = [0, 1]
for comp, lab, cc in zip(comps, labs, cols_):
    vals = np.array([vcd.loc["log_abs", comp], vcd.loc["f_ratio", comp]]) * 100
    a.bar(x, vals, bottom=bottom, color=cc, label=lab, edgecolor="w", linewidth=0.5)
    bottom += vals
a.set_xticks(x)
a.set_xticklabels(["log_abs", "f_ratio"])
a.set_ylabel("% of total variance")
a.set_title("Crossed variance decomposition")
a.legend(fontsize=7, ncol=1, loc="center left", bbox_to_anchor=(1.0, 0.5))

# d) factor slope drift across channels (log_abs)
a = ax[1, 0]
slope_cols = {
    "Vs1": "#4C72B0",
    "Height": "#C44E52",
    "CoV": "#55A868",
    "rH": "#8172B3",
    "aHV": "#DD8452",
}
for j, fn in enumerate(fac):
    a.plot(chs, coef_by_ch["log_abs"][:, j], color=slope_cols[fn], lw=1.5, label=fn)
a.axhline(0, color="k", lw=0.6)
a.axvline(50, ls=":", color="k", lw=1, alpha=0.5)
a.set_xlabel("Recorder (channel)")
a.set_ylabel("Std. slope")
a.set_title("log_abs: factor effects ~stationary across recorders")
a.legend(fontsize=8, ncol=2)

# e) same for f_ratio
a = ax[1, 1]
for j, fn in enumerate(fac):
    a.plot(chs, coef_by_ch["f_ratio"][:, j], color=slope_cols[fn], lw=1.5, label=fn)
a.axhline(0, color="k", lw=0.6)
a.axvline(50, ls=":", color="k", lw=1, alpha=0.5)
a.set_xlabel("Recorder (channel)")
a.set_ylabel("Std. slope")
a.set_title("f_ratio: factor effects ~stationary")
a.legend(fontsize=8, ncol=2)

# f) within-realization spatial vs seed scatter
a = ax[1, 2]
xlab = ["log_abs", "f_ratio"]
sp_v = [wr[wr.target == t].spatial_within_real.values[0] for t in xlab]
se_v = [wr[wr.target == t].seed_within_chan.values[0] for t in xlab]
xx = np.arange(2)
w = 0.35
a.bar(xx - w / 2, sp_v, w, color="#C44E52", label="across recorders\n(fixed realization)")
a.bar(xx + w / 2, se_v, w, color="#DD8452", label="across seeds\n(fixed recorder)")
a.set_xticks(xx)
a.set_xticklabels(xlab)
a.set_ylabel("Mean within-cell std")
a.set_title("Within-realization spatial scatter\nvs seed scatter")
a.legend(fontsize=8)

for j, axx in enumerate(ax.flat):
    panel_letter(axx, string.ascii_lowercase[j])
fig.suptitle(
    "Spatial (101-recorder) structure: factor effects are stationary across recorders, so one pooled model works;\n"
    "position sets the explainability ceiling, and within-realization spatial scatter is a distinct ~23% variance component",
    fontsize=12,
    y=1.01,
)
fig.tight_layout()
fig.savefig(result_path("plots", "spatial_101ch_structure.png"), dpi=150, bbox_inches="tight")
print("saved")
print("\nVariance budget (% total):")
print(
    (
        vcd[
            [
                "design",
                "seed",
                "design_x_seed",
                "channel",
                "design_x_channel",
                "seed_x_channel",
                "threeway_plus",
            ]
        ]
        * 100
    )
    .round(2)
    .to_string()
)
