"""Reliability ceiling, mean/QBM diagnostics, and heteroscedastic PIs.

Writes a three-block CSV summary plus the figure. Center recorder only.

Blocks
------
1. Mean prediction — reliability ceiling, GBM test R², efficiency, MAE,
   cell-mean R²
2. Distributional prediction — QBM pseudo-R², pinball at τ∈{0.05,0.50,0.95},
   90% PI coverage, mean PI width
3. Heteroscedasticity — PI width range / max÷min ratio, upper÷lower tail
   asymmetry

The reliability ceiling is a population-information (signal-to-total) bound,
not a fitted-model score. Treating it as an R² upper bound for a single draw
requires seed noise to be irreducible and independent of the design factors.
"""

from __future__ import annotations

import string
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from calculation_r_2 import reliability_ceiling_from_replicates  # noqa: E402
from config import (  # noqa: E402
    FACTORS,
    FIG_DPI,
    FIG_WIDTH,
    REF_COLOR,
    figsize,
    load_channel50,
    load_mean_models,
    load_quantile_models,
    seed_grouped_split,
    target_color,
    target_label,
)
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, r2_score
from sklearn.model_selection import KFold, cross_val_predict

from seiskit.plot_config import apply_style, panel_letter, result_path

warnings.filterwarnings("ignore")

EVAL_TARGETS = ["abs_TF_ratio", "log_abs", "f_ratio"]
PINBALL_TAUS = (0.05, 0.50, 0.95)
PI_TAUS = (0.05, 0.25, 0.50, 0.75, 0.95)


def _koenker_pseudo_r2(y_true: np.ndarray, y_pred: np.ndarray, tau: float, y_null: float) -> float:
    """Koenker–Machado R¹(τ): 1 − pinball(model) / pinball(unconditional τ-quantile)."""
    v_model = float(mean_pinball_loss(y_true, y_pred, alpha=tau))
    v_null = float(mean_pinball_loss(y_true, np.full_like(y_true, y_null), alpha=tau))
    if v_null <= 0:
        return np.nan
    return 1.0 - v_model / v_null


def _print_block(title: str, df: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    print(df.to_string(index=False))


def main() -> None:
    apply_style(auto_format=True, font_size=10, frame="open")
    d = load_channel50()
    d["cell"] = d.groupby(FACTORS).ngroup()

    # Reliability / signal-to-total ceilings (population-information bound).
    ceil = pd.DataFrame([reliability_ceiling_from_replicates(d, tgt) for tgt in EVAL_TARGETS])

    tr, te = seed_grouped_split(d, test_size=0.25, seed=0)
    Xte = d.iloc[te][FACTORS]
    mean_models = load_mean_models(targets=EVAL_TARGETS, split_by="seed")
    qmodels = load_quantile_models(taus=list(PI_TAUS), targets=EVAL_TARGETS, split_by="seed")

    cm = d.groupby(FACTORS)[EVAL_TARGETS].mean().reset_index()

    # ---- Quantile predictions / PI diagnostics on the seed hold-out ----
    qres: dict[str, dict] = {}
    for tgt in EVAL_TARGETS:
        if tgt not in qmodels or any(tau not in qmodels[tgt] for tau in PI_TAUS):
            continue
        yte = d.iloc[te][tgt].to_numpy(dtype=float)
        ytr = d.iloc[tr][tgt].to_numpy(dtype=float)
        q = {tau: np.asarray(qmodels[tgt][tau].predict(Xte), dtype=float) for tau in PI_TAUS}
        w90 = q[0.95] - q[0.05]
        upper = q[0.95] - q[0.50]
        lower = q[0.50] - q[0.05]
        with np.errstate(divide="ignore", invalid="ignore"):
            asym = np.where(lower > 0, upper / lower, np.nan)

        pinball = {tau: float(mean_pinball_loss(yte, q[tau], alpha=tau)) for tau in PINBALL_TAUS}
        # Honest null: unconditional τ-quantile from the training split.
        pseudo = {
            tau: _koenker_pseudo_r2(yte, q[tau], tau, float(np.quantile(ytr, tau)))
            for tau in PINBALL_TAUS
        }
        qres[tgt] = dict(
            q=q,
            yte=yte,
            cov90=float(np.mean((yte >= q[0.05]) & (yte <= q[0.95]))),
            cov50=float(np.mean((yte >= q[0.25]) & (yte <= q[0.75]))),
            w=w90,
            mean_width=float(np.mean(w90)),
            width_min=float(np.min(w90)),
            width_max=float(np.max(w90)),
            width_ratio=float(np.max(w90) / np.min(w90)),
            mean_asym=float(np.nanmean(asym)),
            pinball=pinball,
            # Primary QBM pseudo-R²: median Koenker–Machado; also store τ extremes.
            pseudo_r2=pseudo[0.50],
            pseudo_r2_by_tau=pseudo,
        )

    # ---- Block 1: mean prediction ----
    mean_rows = []
    for tgt in EVAL_TARGETS:
        c = float(ceil.loc[ceil.target == tgt, "reliability_ceiling"].iloc[0])
        yte = d.iloc[te][tgt].to_numpy(dtype=float)
        if tgt in mean_models:
            pred = np.asarray(mean_models[tgt].predict(Xte), dtype=float)
            gbm_r2 = float(r2_score(yte, pred))
            mae = float(mean_absolute_error(yte, pred))
        else:
            gbm_r2, mae = np.nan, np.nan

        X = cm[FACTORS].values
        y = cm[tgt].values
        m = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0
        )
        cellmean_r2 = float(
            r2_score(y, cross_val_predict(m, X, y, cv=KFold(5, shuffle=True, random_state=0)))
        )
        mean_rows.append(
            dict(
                target=tgt,
                R2_ceiling=round(c, 3),
                GBM_test_R2=round(gbm_r2, 3),
                efficiency_pct=round(gbm_r2 / c * 100, 0)
                if c > 0 and np.isfinite(gbm_r2)
                else np.nan,
                MAE=round(mae, 4),
                cellmean_R2=round(cellmean_r2, 3),
            )
        )
    mean_df = pd.DataFrame(mean_rows)

    # ---- Block 2: distributional prediction ----
    dist_rows = []
    for tgt in EVAL_TARGETS:
        if tgt not in qres:
            continue
        r = qres[tgt]
        dist_rows.append(
            dict(
                target=tgt,
                QBM_pseudo_R2=round(r["pseudo_r2"], 3),
                pinball_t05=round(r["pinball"][0.05], 5),
                pinball_t50=round(r["pinball"][0.50], 5),
                pinball_t95=round(r["pinball"][0.95], 5),
                PI90_coverage=round(r["cov90"], 3),
                mean_PI90_width=round(r["mean_width"], 4),
            )
        )
    dist_df = pd.DataFrame(dist_rows)

    # ---- Block 3: heteroscedasticity ----
    hetero_rows = []
    for tgt in EVAL_TARGETS:
        if tgt not in qres:
            continue
        r = qres[tgt]
        hetero_rows.append(
            dict(
                target=tgt,
                width_min=round(r["width_min"], 4),
                width_max=round(r["width_max"], 4),
                width_ratio_max_min=round(r["width_ratio"], 1),
                mean_upper_lower_width_ratio=round(r["mean_asym"], 2),
            )
        )
    hetero_df = pd.DataFrame(hetero_rows)

    mean_path = result_path("data", "ceiling_mean_prediction.csv")
    dist_path = result_path("data", "ceiling_distributional_prediction.csv")
    hetero_path = result_path("data", "ceiling_heteroscedasticity.csv")
    mean_df.to_csv(mean_path, index=False)
    dist_df.to_csv(dist_path, index=False)
    hetero_df.to_csv(hetero_path, index=False)

    # Backward-compatible wide summary (former single diagnostics CSV).
    wide = mean_df.merge(dist_df, on="target", how="left").merge(hetero_df, on="target", how="left")
    wide_path = result_path("data", "r2_ceiling_diagnostics.csv")
    wide.to_csv(wide_path, index=False)

    _print_block("Mean prediction", mean_df)
    _print_block("Distributional prediction", dist_df)
    _print_block("Heteroscedasticity", hetero_df)
    print(f"\nsaved {mean_path}")
    print(f"saved {dist_path}")
    print(f"saved {hetero_path}")
    print(f"saved {wide_path}")

    # ---- Figure (unchanged layout) ----
    recorded = {r["target"]: r["GBM_test_R2"] for r in mean_rows}
    ceil_map = {r["target"]: r["R2_ceiling"] for r in mean_rows}

    n_tgt = len(EVAL_TARGETS)
    fig, ax = plt.subplots(n_tgt, 3, figsize=figsize(height=FIG_WIDTH * 0.30 * n_tgt))
    if n_tgt == 1:
        ax = np.asarray([ax])
    for i, tgt in enumerate(EVAL_TARGETS):
        col = target_color(tgt)
        a = ax[i, 0]
        c = ceil_map[tgt]
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
        a.set_title(f"{target_label(tgt)}: reliability ceiling = {c:.2f}")
        if np.isfinite(rec) and c > 0:
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

    for i, tgt in enumerate(EVAL_TARGETS):
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

    for i, tgt in enumerate(EVAL_TARGETS):
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

    fig.tight_layout()
    out = result_path("plots", "model_r2_ceiling.png")
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
