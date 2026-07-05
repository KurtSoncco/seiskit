"""
Quantile Boosting Models using LightGBM for channel-50 data.

Fits quantile and mean LightGBM models for both targets (f_ratio, log_abs)
on the center-recorder dataset (24,300 rows).  Two split strategies are
supported via ``--split``:

* ``seed`` (default) — seed-grouped hold-out: evaluates generalisation to
  unseen random-field realisations of the *same* 243 design cells.
* ``cell`` — cell-grouped hold-out: evaluates generalisation to entirely
  unseen physical configurations (material / geometry combos).

Early stopping is used to select the number of boosting rounds per model.
Quantile crossing is fixed at prediction time via isotonic rearrangement
(sorting across the τ dimension per sample).

Outputs
-------
- Trained models saved to ``models/`` as pkl files
- 3x2 diagnostic figure: feature importance, calibration, prediction fans

Usage
-----
    python quantile_channel_model.py                   # seed split, save
    python quantile_channel_model.py --split cell      # cell split
    python quantile_channel_model.py --no-save         # skip model saving
"""

import sys
from collections import defaultdict
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from seiskit.plot_config import apply_style, panel_letter, result_path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FACTORS, MODELS_DIR, load_channel50

# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------
TAUS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
TARGETS = {"f_ratio": "f_ratio", "log_abs": "log_abs"}
SAVE = "--no-save" not in sys.argv

_split_flag = "cell"
if "--split" in sys.argv:
    _split_flag = sys.argv[sys.argv.index("--split") + 1]
SPLIT_BY = _split_flag  # "seed" or "cell"

LGB_PARAMS_BASE = dict(
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    verbose=-1,
)
MAX_ROUNDS = 2000
EARLY_STOPPING_ROUNDS = 50
VAL_FRAC = 0.20  # fraction of training set held for early-stopping validation

# ---------------------------------------------------------------------------
# Data & split
# ---------------------------------------------------------------------------
d50 = load_channel50()

if SPLIT_BY == "cell":
    d50["cell"] = d50.groupby(FACTORS).ngroup()
    groups = d50["cell"].values
    label = "cell-grouped"
else:
    groups = d50["seed"].values
    label = "seed-grouped"

gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
tr_idx, te_idx = next(gss.split(d50, groups=groups))
train, test = d50.iloc[tr_idx].copy(), d50.iloc[te_idx].copy()

print(f"Split: {label}  |  Train: {len(train):,}  Test: {len(test):,}")

# Further split training into fit / validation for early stopping
_val_groups = train["seed"].values
gss_val = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=1)
fit_idx, val_idx = next(gss_val.split(train, groups=_val_groups))
fit_df, val_df = train.iloc[fit_idx], train.iloc[val_idx]

print(f"  └─ Fit: {len(fit_df):,}  Val (early-stop): {len(val_df):,}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def pinball_loss(y_true, y_pred, tau):
    """Quantile (pinball) loss."""
    e = y_true - y_pred
    return np.mean(np.where(e >= 0, tau * e, (tau - 1) * e))


def _fit(params, X_fit, y_fit, X_val, y_val):
    """Train with early stopping and return (model, best_iteration)."""
    ds_fit = lgb.Dataset(X_fit, label=y_fit, free_raw_data=False)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_fit, free_raw_data=False)
    model = lgb.train(
        params,
        ds_fit,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[ds_val],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=0),  # suppress per-round logs
        ],
    )
    return model, model.best_iteration


def fit_quantile_model(X_fit, y_fit, X_val, y_val, tau, seed=42):
    params = {
        **LGB_PARAMS_BASE,
        "objective": "quantile",
        "metric": "quantile",
        "alpha": tau,
        "seed": seed,
    }
    return _fit(params, X_fit, y_fit, X_val, y_val)


def fit_mean_model(X_fit, y_fit, X_val, y_val, seed=42):
    params = {
        **LGB_PARAMS_BASE,
        "objective": "regression",
        "metric": "rmse",
        "seed": seed,
    }
    return _fit(params, X_fit, y_fit, X_val, y_val)


def predict_sorted(models_for_target, X, taus=TAUS):
    """Predict all quantiles and fix crossings via isotonic rearrangement.

    For each sample, the raw per-τ predictions are sorted so that
    q(τ_i) ≤ q(τ_j) whenever τ_i < τ_j.
    """
    raw = np.column_stack([models_for_target[t].predict(X) for t in taus])
    raw.sort(axis=1)  # per-row sort enforces monotonicity across τ
    return raw  # shape (n_samples, len(taus))


# ---------------------------------------------------------------------------
# Fit models
# ---------------------------------------------------------------------------
models: dict[str, dict] = defaultdict(dict)

for tgt_key, tgt_col in TARGETS.items():
    X_fit, y_fit = fit_df[FACTORS], fit_df[tgt_col]
    X_val, y_val = val_df[FACTORS], val_df[tgt_col]
    X_te, y_te = test[FACTORS], test[tgt_col]

    # Mean model
    m_mean, n_iter = fit_mean_model(X_fit, y_fit, X_val, y_val)
    pred_mean = m_mean.predict(X_te)
    rmse = np.sqrt(np.mean((y_te - pred_mean) ** 2))
    models[tgt_key]["mean"] = m_mean
    print(f"[{tgt_key}] mean  RMSE={rmse:.5f}  rounds={n_iter}")

    # Quantile models
    for tau in TAUS:
        m, n_iter = fit_quantile_model(X_fit, y_fit, X_val, y_val, tau)
        models[tgt_key][tau] = m
        pred = m.predict(X_te)
        pl = pinball_loss(y_te.values, pred, tau)
        cov_raw = np.mean(y_te.values <= pred)
        print(
            f"[{tgt_key}] τ={tau:.2f}  pinball={pl:.5f}  "
            f"coverage(raw)={cov_raw:.3f}  rounds={n_iter}"
        )

    # Report crossing-corrected coverage
    preds_sorted = predict_sorted(models[tgt_key], X_te)
    crossings_fixed = int(
        np.any(
            np.diff(
                np.column_stack([models[tgt_key][t].predict(X_te) for t in TAUS]),
                axis=1,
            )
            < 0,
            axis=1,
        ).sum()
    )
    print(
        f"[{tgt_key}] quantile crossings fixed: "
        f"{crossings_fixed}/{len(X_te)} samples "
        f"({100 * crossings_fixed / len(X_te):.1f}%)"
    )
    corrected_cov = [np.mean(y_te.values <= preds_sorted[:, j]) for j in range(len(TAUS))]
    for tau, cc in zip(TAUS, corrected_cov):
        print(f"  τ={tau:.2f}  coverage(corrected)={cc:.3f}")

# ---------------------------------------------------------------------------
# Save models
# ---------------------------------------------------------------------------
if SAVE:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for tgt_key in TARGETS:
        joblib.dump(models[tgt_key]["mean"], MODELS_DIR / f"lgbm_mean_{tgt_key}.pkl")
        for tau in TAUS:
            tag = f"lgbm_q{int(tau * 100):02d}_{tgt_key}"
            joblib.dump(models[tgt_key][tau], MODELS_DIR / f"{tag}.pkl")
    print(f"\nModels saved to {MODELS_DIR}")

# ---------------------------------------------------------------------------
# Figure — diagnostics
# ---------------------------------------------------------------------------
apply_style(auto_format=True, font_size=10, frame="open")
fig, axes = plt.subplots(2, 4, figsize=(13, 5), layout="constrained")

nice = {"log_abs": "log_abs", "f_ratio": "$f$ ratio"}
tcol = {"log_abs": "#C44E52", "f_ratio": "#4C72B0"}

for row_idx, (tgt_key, tgt_col) in enumerate(TARGETS.items()):
    X_te, y_te = test[FACTORS], test[tgt_col].values
    color = tcol[tgt_key]
    preds_sorted = predict_sorted(models[tgt_key], X_te)

    # --- Column 0: Feature importance (gain) for median model ---
    ax = axes[row_idx, 0]
    imp = models[tgt_key][0.50].feature_importance(importance_type="gain")
    imp = imp / imp.sum()
    order = np.argsort(imp)
    ax.barh(np.arange(len(FACTORS)), imp[order], color=color, alpha=0.85)
    ax.set_yticks(np.arange(len(FACTORS)))
    ax.set_yticklabels(np.array(FACTORS)[order])
    ax.set_xlabel("Normalised Gain")
    ax.set_xlim(0, 1)
    ax.set_title(f"{nice[tgt_key]}: feature importance (Median QBM)", loc="left")

    # --- Column 1: Calibration — observed coverage vs nominal (corrected) ---
    ax = axes[row_idx, 1]
    coverages = [np.mean(y_te <= preds_sorted[:, j]) for j in range(len(TAUS))]
    ax.plot(TAUS, coverages, "o-", color=color, ms=6, lw=1.8, label="observed", zorder=3)
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="ideal")
    ax.set_xlabel(r"nominal quantile $\tau$")
    ax.set_ylabel("observed coverage")
    ax.set_title(f"{nice[tgt_key]}: quantile calibration (crossing-corrected)", loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # --- Column 2: Prediction interval fan (crossing-corrected) ---
    ax = axes[row_idx, 2]
    pred_50 = preds_sorted[:, TAUS.index(0.50)]
    pred_05 = preds_sorted[:, TAUS.index(0.05)]
    pred_95 = preds_sorted[:, TAUS.index(0.95)]

    sort_idx = np.argsort(pred_50)
    xs = np.arange(len(sort_idx))

    ax.fill_between(
        xs,
        pred_05[sort_idx],
        pred_95[sort_idx],
        color=color,
        alpha=0.15,
        label="5-95 %",
    )
    ax.plot(xs, pred_50[sort_idx], "-", color=color, lw=0.6, label="median")
    ax.plot(xs, y_te[sort_idx], ",", color="0.3", alpha=0.15, rasterized=True)
    ax.set_xlabel("Test Observations (Sorted by Predicted Median)")
    ax.set_ylabel(nice[tgt_key])
    ax.set_xlim(0, len(sort_idx))

    # For f_Ratio, yaxis would be from 0.6 to 1.4, for log_abs, y axis would be from -4 to 1
    if tgt_key == "f_ratio":
        ax.set_ylim(0.6, 1.4)
    elif tgt_key == "log_abs":
        ax.set_ylim(-4, 1)
    ax.set_title(f"{nice[tgt_key]}: prediction interval fan (Test Set)", loc="left")
    ax.legend(fontsize=10, frameon=False)

    # --- Column 3: Pinball loss vs quantile ---
    ax = axes[row_idx, 3]
    pb_losses = [pinball_loss(y_te, preds_sorted[:, j], TAUS[j]) for j in range(len(TAUS))]
    ax.plot(TAUS, pb_losses, "o-", color=color, ms=6, lw=1.8, label="pinball loss")
    ax.set_xlabel("nominal quantile $\tau$")
    ax.set_ylabel("pinball loss")
    ax.set_title(f"{nice[tgt_key]}: pinball loss vs quantile", loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.set_xlim(0, 1)

    # for f_ratio, yaxis from 1e-3 to 2.25e-2. for log_abs, yaxis from 1e-2 to 0.5.
    if tgt_key == "f_ratio":
        ax.set_ylim(1e-3, 3.0e-2)
    elif tgt_key == "log_abs":
        ax.set_ylim(1e-2, 0.5)

    ax.set_title(f"{nice[tgt_key]}: pinball loss vs quantile", loc="left")


fig.suptitle(
    f"Quantile Boosting Models — LightGBM  ({label} test set)",
    fontsize=11,
)
panel_letter(axes[0, 0], "a")
panel_letter(axes[0, 1], "b")
panel_letter(axes[0, 2], "c")
panel_letter(axes[0, 3], "d")
panel_letter(axes[1, 0], "e")
panel_letter(axes[1, 1], "f")
panel_letter(axes[1, 2], "g")
panel_letter(axes[1, 3], "h")
fig.tight_layout()
fname = result_path("plots", "quantile_channel_model.png")
fig.savefig(fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved: {fname}")
