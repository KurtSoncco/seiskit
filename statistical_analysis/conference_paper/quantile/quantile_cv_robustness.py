"""
Cross-validated robustness study for quantile boosting models (channel 50).

Compares two grouping strategies (seed vs cell) under repeated K-fold CV,
and runs a lightweight hyperparameter grid over the most impactful LightGBM
knobs.  All evaluation uses the pinball loss and empirical quantile coverage
on the held-out folds.

Outputs
-------
- Console summary tables (pinball loss, coverage, calibration error)
- ``quantile_cv_robustness.png`` — 4-panel figure:
    (a–b) CV pinball loss by split strategy for each target
    (c–d) CV calibration curves (coverage vs nominal τ) with fold spread

Usage
-----
    python quantile_cv_robustness.py              # default 5-fold, both splits
    python quantile_cv_robustness.py --folds 10   # 10-fold
"""

import sys
from itertools import product
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from seiskit.plot_config import apply_style, panel_letter, result_path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FACTORS, load_channel50

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
N_FOLDS = 5
if "--folds" in sys.argv:
    N_FOLDS = int(sys.argv[sys.argv.index("--folds") + 1])

TAUS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
TARGETS = {"f_ratio": "f_ratio", "log_abs": "log_abs"}

# ---------------------------------------------------------------------------
# Hyperparameter grid — keep it small to stay tractable
# ---------------------------------------------------------------------------
HP_GRID = list(
    product(
        [0.01, 0.05, 0.10],  # learning_rate
        [15, 31, 63],  # num_leaves
        [20, 50],  # min_child_samples
    )
)
HP_NAMES = ["lr", "leaves", "min_child"]

LGB_FIXED = dict(
    subsample=0.8,
    colsample_bytree=0.8,
    verbose=-1,
)
MAX_ROUNDS = 2000
EARLY_STOP = 20
VAL_FRAC = 0.20

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
d50 = load_channel50()
d50["cell"] = d50.groupby(FACTORS).ngroup()

print(f"Data loaded: {len(d50):,} rows, {N_FOLDS}-fold CV\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def pinball_loss(y_true, y_pred, tau):
    e = y_true - y_pred
    return np.mean(np.where(e >= 0, tau * e, (tau - 1) * e))


def _fit_one(X_fit, y_fit, X_val, y_val, params):
    ds_fit = lgb.Dataset(X_fit, label=y_fit, free_raw_data=False)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_fit, free_raw_data=False)
    model = lgb.train(
        params,
        ds_fit,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[ds_val],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOP),
            lgb.log_evaluation(period=0),
        ],
    )
    return model


def _inner_val_split(X_tr, y_tr, groups_tr, rng):
    """80/20 grouped split of the training fold for early stopping."""
    unique_g = np.unique(groups_tr)
    rng.shuffle(unique_g)
    n_val = max(1, int(len(unique_g) * VAL_FRAC))
    val_groups = set(unique_g[:n_val])
    mask = np.array([g in val_groups for g in groups_tr])
    return X_tr[~mask], y_tr[~mask], X_tr[mask], y_tr[mask]


# ---------------------------------------------------------------------------
# CV evaluation for one (target, split_by, hp_combo)
# ---------------------------------------------------------------------------


def cv_evaluate(tgt_col, split_by, lr, leaves, mcs, rng):
    """Run grouped K-fold CV and return per-fold metrics."""
    groups = d50["seed"].values if split_by == "seed" else d50["cell"].values
    gkf = GroupKFold(n_splits=N_FOLDS)

    fold_records = []
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(d50, groups=groups)):
        X_tr = d50.iloc[tr_idx][FACTORS].values
        y_tr = d50.iloc[tr_idx][tgt_col].values
        X_te = d50.iloc[te_idx][FACTORS].values
        y_te = d50.iloc[te_idx][tgt_col].values
        groups_tr = groups[tr_idx]

        X_fit, y_fit, X_val, y_val = _inner_val_split(X_tr, y_tr, groups_tr, rng)

        fold_models = {}
        for tau in TAUS:
            params = {
                **LGB_FIXED,
                "objective": "quantile",
                "metric": "quantile",
                "alpha": tau,
                "learning_rate": lr,
                "num_leaves": leaves,
                "min_child_samples": mcs,
                "seed": 42 + fold,
            }
            fold_models[tau] = _fit_one(X_fit, y_fit, X_val, y_val, params)

        # Crossing-corrected predictions
        raw = np.column_stack([fold_models[t].predict(X_te) for t in TAUS])
        raw.sort(axis=1)

        for j, tau in enumerate(TAUS):
            pl = pinball_loss(y_te, raw[:, j], tau)
            cov = np.mean(y_te <= raw[:, j])
            fold_records.append(
                dict(
                    fold=fold,
                    tau=tau,
                    pinball=pl,
                    coverage=cov,
                    cal_error=abs(cov - tau),
                )
            )

    return pd.DataFrame(fold_records)


# ===================================================================
# Part 1 — Compare split strategies (seed vs cell) with default HPs
# ===================================================================
print("=" * 70)
print("PART 1: Split-strategy comparison (seed vs cell grouped CV)")
print("=" * 70)

default_hp = (0.05, 31, 50)
split_results = {}
for split_by in ["seed", "cell"]:
    print(f"\n  [{split_by}-grouped] ...", flush=True)
    rng = np.random.default_rng(0)
    frames = []
    for tgt_key, tgt_col in TARGETS.items():
        df = cv_evaluate(tgt_col, split_by, *default_hp, rng)
        df["target"] = tgt_key
        frames.append(df)
    split_results[split_by] = pd.concat(frames, ignore_index=True)
    agg = (
        split_results[split_by]
        .groupby(["target", "tau"])
        .agg(
            pinball_mean=("pinball", "mean"),
            pinball_std=("pinball", "std"),
            cov_mean=("coverage", "mean"),
            cov_std=("coverage", "std"),
            cal_err=("cal_error", "mean"),
        )
        .reset_index()
    )
    print(agg.to_string(index=False, float_format="%.4f"))


# ===================================================================
# Part 2 — Hyperparameter grid search (seed-grouped only, median τ)
# ===================================================================
print("\n" + "=" * 70)
print("PART 2: Hyperparameter sensitivity (seed-grouped, τ=0.50)")
print("=" * 70)

hp_records = []
for i, (lr, leaves, mcs) in enumerate(HP_GRID):
    tag = f"lr={lr} leaves={leaves} mcs={mcs}"
    print(f"  [{i + 1}/{len(HP_GRID)}] {tag}", flush=True)
    rng = np.random.default_rng(0)
    for tgt_key, tgt_col in TARGETS.items():
        df = cv_evaluate(tgt_col, "seed", lr, leaves, mcs, rng)
        median_rows = df[df.tau == 0.50]
        hp_records.append(
            dict(
                lr=lr,
                leaves=leaves,
                mcs=mcs,
                target=tgt_key,
                pinball_mean=median_rows["pinball"].mean(),
                pinball_std=median_rows["pinball"].std(),
                cal_err_mean=median_rows["cal_error"].mean(),
            )
        )

hp_df = pd.DataFrame(hp_records)
for tgt_key in TARGETS:
    sub = hp_df[hp_df.target == tgt_key].sort_values("pinball_mean")
    print(f"\n  {tgt_key} — sorted by median pinball loss:")
    print(sub.to_string(index=False, float_format="%.5f"))


# ===================================================================
# Figure
# ===================================================================
apply_style(auto_format=True, font_size=10, frame="open")
fig, axes = plt.subplots(2, 2, figsize=(13, 10))

nice = {"log_abs": r"log |TF|", "f_ratio": "$f$ ratio"}
tcol = {"log_abs": "#C44E52", "f_ratio": "#4C72B0"}
split_ls = {"seed": "-", "cell": "--"}
split_marker = {"seed": "o", "cell": "s"}

# --- Row 0: Pinball loss by split strategy ---
for col_idx, (tgt_key, _) in enumerate(TARGETS.items()):
    ax = axes[0, col_idx]
    for split_by in ["seed", "cell"]:
        sub = split_results[split_by]
        sub = (
            sub[sub.target == tgt_key]
            .groupby("tau")
            .agg(
                m=("pinball", "mean"),
                s=("pinball", "std"),
            )
            .reset_index()
        )
        ax.plot(
            sub["tau"],
            sub["m"],
            marker=split_marker[split_by],
            ls=split_ls[split_by],
            color=tcol[tgt_key],
            ms=5,
            lw=1.5,
            label=f"{split_by}-grouped",
        )
        ax.fill_between(
            sub["tau"],
            sub["m"] - sub["s"],
            sub["m"] + sub["s"],
            color=tcol[tgt_key],
            alpha=0.12,
        )
    ax.set_xlabel(r"quantile $\tau$")
    ax.set_ylabel("pinball loss (CV mean ± 1 sd)")
    ax.set_title(f"{nice[tgt_key]}: pinball loss — seed vs cell split", loc="left")
    ax.legend(fontsize=8, frameon=False)
    panel_letter(ax, chr(97 + col_idx))

# --- Row 1: Calibration curves with fold spread ---
for col_idx, (tgt_key, _) in enumerate(TARGETS.items()):
    ax = axes[1, col_idx]
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, label="ideal")
    for split_by in ["seed", "cell"]:
        sub = split_results[split_by]
        sub = (
            sub[sub.target == tgt_key]
            .groupby("tau")
            .agg(
                m=("coverage", "mean"),
                s=("coverage", "std"),
            )
            .reset_index()
        )
        ax.plot(
            sub["tau"],
            sub["m"],
            marker=split_marker[split_by],
            ls=split_ls[split_by],
            color=tcol[tgt_key],
            ms=5,
            lw=1.5,
            label=f"{split_by}-grouped",
        )
        ax.fill_between(
            sub["tau"],
            sub["m"] - sub["s"],
            sub["m"] + sub["s"],
            color=tcol[tgt_key],
            alpha=0.12,
        )
    ax.set_xlabel(r"nominal $\tau$")
    ax.set_ylabel("observed coverage (CV mean ± 1 sd)")
    ax.set_title(f"{nice[tgt_key]}: calibration — seed vs cell split", loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    panel_letter(ax, chr(99 + col_idx))

fig.suptitle(
    f"QBM robustness: {N_FOLDS}-fold grouped CV — split strategy & calibration",
    fontsize=11,
    y=1.005,
)
fig.tight_layout()
fname = result_path("plots", "quantile_cv_robustness.png")
fig.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved: {fname}")
