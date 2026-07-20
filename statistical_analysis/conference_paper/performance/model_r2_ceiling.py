"""R² ceiling decomposition and heteroscedastic prediction intervals.

Writes both the figure and a CSV summary (merged from the former
``r2_ceiling_diagnostics.py``). Center recorder only.
"""

from __future__ import annotations

import string
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict

from config import (  # noqa: E402
    FACTORS,
    FIG_DPI,
    REF_COLOR,
    load_channel50,
    load_mean_models,
    load_quantile_models,
    seed_grouped_split,
    target_color,
    target_label,
)
from seiskit.plot_config import apply_style, panel_letter, result_path

warnings.filterwarnings("ignore")


def main() -> None:
    apply_style(auto_format=True, font_size=10, frame="open")
    d = load_channel50()
    d["cell"] = d.groupby(FACTORS).ngroup()

    rows = []
    for tgt in ["log_abs", "f_ratio", "abs_TF_ratio"]:
        y = d[tgt].values
        grand = y.mean()
        cell_mean = d.groupby("cell")[tgt].transform("mean")
        ss_within = ((y - cell_mean) ** 2).sum()
        ss_total = ((y - grand) ** 2).sum()
        ss_between = ss_total - ss_within
        rows.append(
            dict(
                target=tgt,
                R2_ceiling=ss_between / ss_total,
                frac_irreducible=ss_within / ss_total,
            )
        )
    ceil = pd.DataFrame(rows)

    # Prefer live mean-model R² when models exist; fall back to recorded values.
    tr, te = seed_grouped_split(d, test_size=0.25, seed=0)
    Xte = d.iloc[te][FACTORS]
    mean_models = load_mean_models(
        targets=["log_abs", "f_ratio", "abs_TF_ratio"], split_by="seed"
    )
    recorded = {"log_abs": 0.358, "f_ratio": 0.208}
    for tgt in ["log_abs", "f_ratio"]:
        if tgt in mean_models:
            recorded[tgt] = float(
                r2_score(d.iloc[te][tgt].values, mean_models[tgt].predict(Xte))
            )

    cm = d.groupby(FACTORS)[["log_abs", "f_ratio", "abs_TF_ratio"]].mean().reset_index()
    cs = (
        d.groupby(FACTORS)
        .agg(
            mu_log=("log_abs", "mean"),
            sd_log=("log_abs", "std"),
            mu_f=("f_ratio", "mean"),
            sd_f=("f_ratio", "std"),
            sd_raw=("abs_TF_ratio", "std"),
        )
        .reset_index()
    )

    qmodels = load_quantile_models(
        taus=[0.05, 0.25, 0.5, 0.75, 0.95],
        targets=["log_abs", "f_ratio"],
        split_by="seed",
    )
    qres = {}
    for tgt in ["log_abs", "f_ratio"]:
        yte = d.iloc[te][tgt].values
        q = {
            tau: qmodels[tgt][tau].predict(Xte)
            for tau in [0.05, 0.25, 0.5, 0.75, 0.95]
            if tau in qmodels.get(tgt, {})
        }
        if len(q) < 5:
            continue
        cov90 = np.mean((yte >= q[0.05]) & (yte <= q[0.95]))
        cov50 = np.mean((yte >= q[0.25]) & (yte <= q[0.75]))
        w = q[0.95] - q[0.05]
        qres[tgt] = dict(q=q, cov90=cov90, cov50=cov50, yte=yte, w=w)

    # CSV summary (former r2_ceiling_diagnostics)
    summ = []
    for tgt in ["log_abs", "f_ratio"]:
        c = ceil.loc[ceil.target == tgt, "R2_ceiling"].values[0]
        X = cm[FACTORS].values
        y = cm[tgt].values
        m = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0
        )
        r2cm = r2_score(
            y, cross_val_predict(m, X, y, cv=KFold(5, shuffle=True, random_state=0))
        )
        col = "sd_log" if tgt == "log_abs" else "sd_f"
        ys = cs[col].values
        ms = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0
        )
        r2sd = r2_score(
            ys,
            cross_val_predict(ms, cs[FACTORS].values, ys, cv=KFold(5, shuffle=True, random_state=0)),
        )
        row = dict(
            target=tgt,
            R2_ceiling=round(c, 3),
            GBM_test_R2=round(recorded.get(tgt, np.nan), 3),
            efficiency_pct=round(recorded.get(tgt, np.nan) / c * 100, 0),
            cellmean_R2=round(r2cm, 3),
            scale_R2=round(r2sd, 3),
            sd_range_x=round(ys.max() / ys.min(), 1),
        )
        if tgt in qres:
            r = qres[tgt]
            row["PI90_coverage"] = round(r["cov90"], 3)
            row["PI_width_range_x"] = round(r["w"].max() / r["w"].min(), 1)
        summ.append(row)
    sd = pd.DataFrame(summ)
    csv_path = result_path("data", "r2_ceiling_diagnostics.csv")
    sd.to_csv(csv_path, index=False)
    print(sd.to_string(index=False))

    fig, ax = plt.subplots(2, 3, figsize=(14, 8.8))
    for i, tgt in enumerate(["log_abs", "f_ratio"]):
        col = target_color(tgt)
        a = ax[i, 0]
        c = ceil.loc[ceil.target == tgt, "R2_ceiling"].values[0]
        rec = recorded[tgt]
        for xpos, v, al, cc in zip(
            [0, 1, 2], [1.0, c, rec], [0.35, 0.6, 1.0], [REF_COLOR, col, col]
        ):
            a.bar(xpos, v, color=cc, alpha=al, edgecolor="k", linewidth=0.8)
        a.set_xticks([0, 1, 2])
        a.set_xticklabels(
            ["Total\nvariance", "Explainable\n(between-cell)", "GBM\ncaptured"], fontsize=9
        )
        a.set_ylim(0, 1.05)
        a.set_ylabel("Variance fraction")
        a.set_title(f"{target_label(tgt)}: R² ceiling = {c:.2f}")
        a.text(
            2,
            rec + 0.03,
            f"{rec / c * 100:.0f}% of\nexplainable",
            ha="center",
            fontsize=8.5,
            color=col,
            fontweight="bold",
        )
        a.axhline(c, ls=":", color=col, lw=1)
        a.text(
            0,
            0.52,
            f"{(1 - c) * 100:.0f}%\nirreducible\nseed noise",
            ha="center",
            va="center",
            fontsize=8.5,
            color="#555",
        )

    for i, tgt in enumerate(["log_abs", "f_ratio"]):
        col = target_color(tgt)
        a = ax[i, 1]
        X = cm[FACTORS].values
        y = cm[tgt].values
        m = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0
        )
        pred = cross_val_predict(m, X, y, cv=KFold(5, shuffle=True, random_state=0))
        r2 = r2_score(y, pred)
        a.scatter(y, pred, s=14, color=col, alpha=0.5, edgecolor="none")
        lo, hi = min(y.min(), pred.min()), max(y.max(), pred.max())
        a.plot([lo, hi], [lo, hi], "k--", lw=1)
        a.set_xlabel(f"Actual cell-mean {target_label(tgt)}")
        a.set_ylabel("Predicted")
        a.set_title(f"Deterministic signal: R²={r2:.3f}")

    for i, tgt in enumerate(["log_abs", "f_ratio"]):
        a = ax[i, 2]
        if tgt not in qres:
            a.axis("off")
            a.set_title(f"{target_label(tgt)}: quantile models missing")
            continue
        col = target_color(tgt)
        r = qres[tgt]
        q = r["q"]
        yte = r["yte"]
        order = np.argsort(r["w"])
        xx = np.arange(len(order))
        a.fill_between(xx, q[0.05][order], q[0.95][order], color=col, alpha=0.20, label="90% PI")
        a.fill_between(xx, q[0.25][order], q[0.75][order], color=col, alpha=0.40, label="50% PI")
        a.plot(xx, q[0.5][order], color=col, lw=1.2, label="median")
        sub = np.linspace(0, len(order) - 1, 600).astype(int)
        a.scatter(xx[sub], yte[order][sub], s=3, color="k", alpha=0.3, zorder=3)
        a.set_xlabel("Test points (sorted by PI width)")
        a.set_ylabel(target_label(tgt))
        a.set_title(f"90%-PI cov={r['cov90']:.2f}")
        a.legend(fontsize=8, loc="upper left", framealpha=0.9)

    for j, axx in enumerate(ax.flat):
        panel_letter(axx, string.ascii_lowercase[j])

    fig.suptitle(
        "R² ceiling: variance capped by irreducible seed noise; "
        "deterministic signal remains predictable",
        fontsize=12.5,
        y=1.005,
    )
    fig.tight_layout()
    out = result_path("plots", "model_r2_ceiling.png")
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
