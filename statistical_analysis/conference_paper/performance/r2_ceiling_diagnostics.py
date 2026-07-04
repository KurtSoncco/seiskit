import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold, GroupShuffleSplit
from sklearn.metrics import r2_score

from seiskit.plot_config import result_path
from config import load_channel50, FACTORS

d = load_channel50()
d["cell"] = d.groupby(FACTORS).ngroup()

# R2 ceiling
rows = []
for tgt in ["log_abs", "f_ratio", "abs_TF_ratio"]:
    y = d[tgt].values
    grand = y.mean()
    cell_mean = d.groupby("cell")[tgt].transform("mean")
    ss_within = ((y - cell_mean) ** 2).sum()
    ss_total = ((y - grand) ** 2).sum()
    ss_between = ss_total - ss_within
    R2_ceiling = ss_between / ss_total
    rows.append(dict(target=tgt, R2_ceiling=R2_ceiling, frac_irreducible=ss_within / ss_total))
ceil = pd.DataFrame(rows)

# Recorded test R2 from prior GBM work
recorded = {"log_abs": 0.358, "f_ratio": 0.208}

# Cell means for cell-mean prediction
cm = d.groupby(FACTORS)[["log_abs", "f_ratio"]].mean().reset_index()

# Cell stats for scale prediction
cs = (
    d.groupby(FACTORS)
    .agg(
        mu_log=("log_abs", "mean"),
        sd_log=("log_abs", "std"),
        mu_f=("f_ratio", "mean"),
        sd_f=("f_ratio", "std"),
    )
    .reset_index()
)

# Seed-grouped split for quantile models
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
tr, te = next(gss.split(d, groups=d["seed"]))
Xtr, Xte = d.iloc[tr][FACTORS].values, d.iloc[te][FACTORS].values


def lgbm(obj="regression", alpha=None):
    p = dict(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=0,
        verbose=-1,
        objective=obj,
    )
    if alpha is not None:
        p["alpha"] = alpha
    return lgb.LGBMRegressor(**p)


qres = {}
for tgt in ["log_abs", "f_ratio"]:
    ytr, yte = d.iloc[tr][tgt].values, d.iloc[te][tgt].values
    q = {}
    for a in [0.05, 0.25, 0.5, 0.75, 0.95]:
        q[a] = lgbm("quantile", alpha=a).fit(Xtr, ytr).predict(Xte)
    cov90 = np.mean((yte >= q[0.05]) & (yte <= q[0.95]))
    cov50 = np.mean((yte >= q[0.25]) & (yte <= q[0.75]))
    w = q[0.95] - q[0.05]
    qres[tgt] = dict(q=q, cov90=cov90, cov50=cov50, yte=yte, w=w)

# Build summary
summ = []
for tgt in ["log_abs", "f_ratio"]:
    c = ceil.loc[ceil.target == tgt, "R2_ceiling"].values[0]
    X = cm[FACTORS].values
    y = cm[tgt].values
    m = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0
    )
    r2cm = r2_score(y, cross_val_predict(m, X, y, cv=KFold(5, shuffle=True, random_state=0)))
    col = "sd_log" if tgt == "log_abs" else "sd_f"
    Xs = cs[FACTORS].values
    ys = cs[col].values
    ms = HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0
    )
    r2sd = r2_score(ys, cross_val_predict(ms, Xs, ys, cv=KFold(5, shuffle=True, random_state=0)))
    r = qres[tgt]
    summ.append(
        dict(
            target=tgt,
            R2_ceiling=round(c, 3),
            GBM_test_R2=recorded[tgt],
            efficiency_pct=round(recorded[tgt] / c * 100, 0),
            cellmean_R2=round(r2cm, 3),
            scale_R2=round(r2sd, 3),
            sd_range_x=round(ys.max() / ys.min(), 1),
            PI90_coverage=round(r["cov90"], 3),
            PI_width_range_x=round(r["w"].max() / r["w"].min(), 1),
        )
    )

sd = pd.DataFrame(summ)
sd.to_csv(result_path("data", "r2_ceiling_diagnostics.csv"), index=False)
print(sd.to_string(index=False))
