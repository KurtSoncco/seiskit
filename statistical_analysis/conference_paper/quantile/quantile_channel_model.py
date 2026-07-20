"""Quantile Boosting Models (LightGBM) for the center recorder.

Fits mean and quantile LightGBM models for:

* raw amplitude ``abs_TF_ratio``
* natural-log amplitude ``log_abs = ln(abs_TF_ratio)``
* frequency ratio ``f_ratio``

Split strategies (``--split``):

* ``seed`` (default) — hold out unseen random-field seeds
* ``cell`` — hold out unseen design cells

Outputs trained models under ``models/`` and a diagnostic figure under
``results/quantile/plots/``.
"""

from __future__ import annotations

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
from config import (  # noqa: E402
    DEFAULT_SPLIT_BY,
    DEFAULT_TAUS,
    FACTORS,
    FIG_DPI,
    MODELS_DIR,
    MODEL_TARGETS,
    REF_COLOR,
    load_channel50,
    mean_model_stem,
    quantile_model_stem,
    target_color,
    target_label,
)

TAUS = list(DEFAULT_TAUS)
TARGETS = {t: t for t in MODEL_TARGETS}
SAVE = "--no-save" not in sys.argv

_split_flag = DEFAULT_SPLIT_BY
if "--split" in sys.argv:
    _split_flag = sys.argv[sys.argv.index("--split") + 1]
SPLIT_BY = _split_flag

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
VAL_FRAC = 0.20


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
            lgb.log_evaluation(period=0),
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
    """Predict all quantiles and fix crossings via per-row sorting."""
    raw = np.column_stack([models_for_target[t].predict(X) for t in taus])
    raw.sort(axis=1)
    return raw


def main() -> None:
    d50 = load_channel50()

    if SPLIT_BY == "cell":
        d50 = d50.copy()
        d50["cell"] = d50.groupby(FACTORS).ngroup()
        groups = d50["cell"].values
        label = "cell-grouped"
    elif SPLIT_BY == "seed":
        groups = d50["seed"].values
        label = "seed-grouped"
    else:
        raise ValueError("--split must be 'seed' or 'cell'")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
    tr_idx, te_idx = next(gss.split(d50, groups=groups))
    train, test = d50.iloc[tr_idx].copy(), d50.iloc[te_idx].copy()
    print(f"Split: {label}  |  Train: {len(train):,}  Test: {len(test):,}")

    gss_val = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=1)
    fit_idx, val_idx = next(gss_val.split(train, groups=train["seed"].values))
    fit_df, val_df = train.iloc[fit_idx], train.iloc[val_idx]
    print(f"  └─ Fit: {len(fit_df):,}  Val (early-stop): {len(val_df):,}")

    models: dict[str, dict] = defaultdict(dict)
    for tgt_key, tgt_col in TARGETS.items():
        X_fit, y_fit = fit_df[FACTORS], fit_df[tgt_col]
        X_val, y_val = val_df[FACTORS], val_df[tgt_col]
        X_te, y_te = test[FACTORS], test[tgt_col]

        m_mean, n_iter = fit_mean_model(X_fit, y_fit, X_val, y_val)
        pred_mean = m_mean.predict(X_te)
        rmse = np.sqrt(np.mean((y_te - pred_mean) ** 2))
        models[tgt_key]["mean"] = m_mean
        print(f"[{tgt_key}] mean  RMSE={rmse:.5f}  rounds={n_iter}")

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

    if SAVE:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        for tgt_key in TARGETS:
            mean_stem = mean_model_stem(tgt_key, split_by=SPLIT_BY)
            joblib.dump(models[tgt_key]["mean"], MODELS_DIR / f"{mean_stem}.pkl")
            for tau in TAUS:
                q_stem = quantile_model_stem(tgt_key, tau, split_by=SPLIT_BY)
                joblib.dump(models[tgt_key][tau], MODELS_DIR / f"{q_stem}.pkl")
        print(f"\nModels saved to {MODELS_DIR} with split suffix '_{SPLIT_BY}'")

    # Focus figure on primary comparison: ln amplitude vs frequency (+ raw row)
    apply_style(auto_format=True, font_size=9, frame="open")
    fig, axes = plt.subplots(len(TARGETS), 4, figsize=(14, 3.2 * len(TARGETS)), layout="constrained")
    if len(TARGETS) == 1:
        axes = np.array([axes])

    for row_idx, (tgt_key, tgt_col) in enumerate(TARGETS.items()):
        X_te, y_te = test[FACTORS], test[tgt_col].values
        color = target_color(tgt_key)
        nice = target_label(tgt_key)
        preds_sorted = predict_sorted(models[tgt_key], X_te)

        ax = axes[row_idx, 0]
        imp = models[tgt_key][0.50].feature_importance(importance_type="gain")
        imp = imp / imp.sum()
        order = np.argsort(imp)
        ax.barh(np.arange(len(FACTORS)), imp[order], color=color, alpha=0.85)
        ax.set_yticks(np.arange(len(FACTORS)))
        ax.set_yticklabels(np.array(FACTORS)[order])
        ax.set_xlabel("Normalised Gain")
        ax.set_xlim(0, 1)
        ax.set_title(f"{nice}: feature importance (median)", loc="left")

        ax = axes[row_idx, 1]
        coverages = [np.mean(y_te <= preds_sorted[:, j]) for j in range(len(TAUS))]
        ax.plot(TAUS, coverages, "o-", color=color, ms=5, lw=1.6, label="observed", zorder=3)
        ax.plot([0, 1], [0, 1], "--", color=REF_COLOR, lw=1, label="ideal")
        ax.set_xlabel(r"nominal quantile $\tau$")
        ax.set_ylabel("observed coverage")
        ax.set_title(f"{nice}: calibration", loc="left")
        ax.legend(fontsize=7, frameon=False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax = axes[row_idx, 2]
        pred_50 = preds_sorted[:, TAUS.index(0.50)]
        pred_05 = preds_sorted[:, TAUS.index(0.05)]
        pred_95 = preds_sorted[:, TAUS.index(0.95)]
        sort_idx = np.argsort(pred_50)
        xs = np.arange(len(sort_idx))
        ax.fill_between(
            xs, pred_05[sort_idx], pred_95[sort_idx], color=color, alpha=0.15, label="5–95%"
        )
        ax.plot(xs, pred_50[sort_idx], "-", color=color, lw=0.6, label="median")
        ax.plot(xs, y_te[sort_idx], ",", color="0.3", alpha=0.15, rasterized=True)
        ax.set_xlabel("Test obs. (sorted by median)")
        ax.set_ylabel(nice)
        ax.set_xlim(0, len(sort_idx))
        if tgt_key == "f_ratio":
            ax.set_ylim(0.6, 1.4)
        elif tgt_key == "log_abs":
            ax.set_ylim(-4, 1)
        ax.set_title(f"{nice}: prediction fan", loc="left")
        ax.legend(fontsize=8, frameon=False)

        ax = axes[row_idx, 3]
        pb_losses = [pinball_loss(y_te, preds_sorted[:, j], TAUS[j]) for j in range(len(TAUS))]
        ax.plot(TAUS, pb_losses, "o-", color=color, ms=5, lw=1.6)
        ax.set_xlabel(r"nominal quantile $\tau$")
        ax.set_ylabel("pinball loss")
        ax.set_title(f"{nice}: pinball loss", loc="left")
        ax.set_xlim(0, 1)

    letters = "abcdefghijkl"
    for i, ax in enumerate(axes.ravel()):
        panel_letter(ax, letters[i])

    fig.suptitle(
        f"Quantile Boosting Models — center recorder ({label} test set)",
        fontsize=11,
    )
    fname = result_path("plots", "quantile_channel_model.png")
    fig.savefig(fname, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {fname}")


if __name__ == "__main__":
    main()
