"""GBM mean + quantile performance on the center recorder.

Compares raw amplitude, natural-log amplitude, and frequency ratio on a
seed-grouped hold-out using pre-trained models from ``models/``. Exports
R², MSE, and MAE on each target's native scale and a fair amplitude-only
comparison on the common raw-amplitude scale.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_pinball_loss,
    mean_squared_error,
    r2_score,
)

from config import (  # noqa: E402
    FACTORS,
    FIG_DPI,
    MODEL_TARGETS,
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
    d50 = load_channel50()
    tr, te = seed_grouped_split(d50, test_size=0.25, seed=0)
    Xte_df = d50[FACTORS].iloc[te]
    taus = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

    mean_models = load_mean_models(targets=list(MODEL_TARGETS), split_by="seed")
    quant_models = load_quantile_models(
        taus=taus, targets=list(MODEL_TARGETS), split_by="seed"
    )
    missing_targets = [t for t in MODEL_TARGETS if t not in mean_models]
    if missing_targets:
        raise FileNotFoundError(
            f"Missing mean models for {missing_targets}. Train all targets with: "
            "python quantile/quantile_channel_model.py"
        )
    eval_targets = list(MODEL_TARGETS)

    # Native-scale metrics describe each fitted response, but MSE and MAE are
    # scale-dependent. A second raw-amplitude evaluation back-transforms the
    # log model so raw-vs-log amplitude models can be compared fairly.
    metric_rows = []
    predictions = {}
    for tgt in eval_targets:
        y_true = d50.iloc[te][tgt].to_numpy()
        y_pred = np.asarray(mean_models[tgt].predict(Xte_df))
        predictions[tgt] = y_pred
        metric_rows.append(
            {
                "model_target": tgt,
                "evaluation_target": tgt,
                "evaluation_scale": "native",
                "prediction_transform": "identity",
                "r2": r2_score(y_true, y_pred),
                "mse": mean_squared_error(y_true, y_pred),
                "mae": mean_absolute_error(y_true, y_pred),
            }
        )

    # Duan's smearing estimate corrects the retransformation bias of a model
    # fitted to E[ln(Y)|X]. Estimate it on training rows only.
    Xtr_df = d50[FACTORS].iloc[tr]
    log_train_pred = np.asarray(mean_models["log_abs"].predict(Xtr_df))
    log_train_true = d50.iloc[tr]["log_abs"].to_numpy()
    smearing_factor = float(np.mean(np.exp(log_train_true - log_train_pred)))

    y_amp_raw = d50.iloc[te]["abs_TF_ratio"].to_numpy()
    raw_scale_predictions = {
        "abs_TF_ratio": predictions["abs_TF_ratio"],
        "log_abs": np.exp(predictions["log_abs"]) * smearing_factor,
    }
    for tgt, pred_raw in raw_scale_predictions.items():
        transform = "identity" if tgt == "abs_TF_ratio" else "exp_duan_smearing"
        metric_rows.append(
            {
                "model_target": tgt,
                "evaluation_target": "abs_TF_ratio",
                "evaluation_scale": "raw_amplitude",
                "prediction_transform": transform,
                "smearing_factor": 1.0 if tgt == "abs_TF_ratio" else smearing_factor,
                "r2": r2_score(y_amp_raw, pred_raw),
                "mse": mean_squared_error(y_amp_raw, pred_raw),
                "mae": mean_absolute_error(y_amp_raw, pred_raw),
            }
        )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = result_path("data", "model_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    pinball_rows = []
    for tgt in eval_targets:
        yte = d50.iloc[te][tgt].values
        for q in taus:
            if q in quant_models.get(tgt, {}):
                pred = quant_models[tgt][q].predict(Xte_df)
                pl = mean_pinball_loss(yte, pred, alpha=q)
                pinball_rows.append(dict(target=tgt, tau=q, pinball_loss=pl))
    pinball_df = pd.DataFrame(pinball_rows)

    def _gain_importance(model):
        """Support both native Booster and sklearn LGBMRegressor wrappers."""
        booster = getattr(model, "booster_", model)
        return booster.feature_importance(importance_type="gain")

    imp = {}
    for tgt in eval_targets:
        imp[tgt] = pd.Series(_gain_importance(mean_models[tgt]), index=FACTORS)
    imp_df = pd.DataFrame(imp)
    imp_df = (imp_df / imp_df.sum() * 100).round(1)

    apply_style(auto_format=True, font_size=10, frame="open")
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    # Native-scale predicted vs actual for the two amplitude models.
    plot_tgts = ["abs_TF_ratio", "log_abs"]
    for ax, tgt in zip([axes[0, 0], axes[0, 1]], plot_tgts):
        yte = d50.iloc[te][tgt].values
        pred = predictions[tgt]
        idx = np.linspace(0, len(yte) - 1, 3500).astype(int)
        ax.plot(pred[idx], yte[idx], "o", ms=2, color=target_color(tgt), alpha=0.25)
        lo, hi = yte.min(), yte.max()
        ax.plot([lo, hi], [lo, hi], "-", color=REF_COLOR, lw=1)
        r2 = r2_score(yte, pred)
        mse = mean_squared_error(yte, pred)
        mae = mean_absolute_error(yte, pred)
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_title(
            f"{target_label(tgt)}: R²={r2:.2f}, MSE={mse:.3g}, MAE={mae:.3g}"
        )

    # Fair raw-amplitude comparison of raw-fit vs log-fit mean GBMs.
    ax = axes[0, 2]
    fair = metrics_df[metrics_df["evaluation_scale"] == "raw_amplitude"].copy()
    names = fair["model_target"].tolist()
    vals = fair["r2"].tolist()
    cols = [target_color(n) for n in names]
    ax.bar(np.arange(len(names)), vals, color=cols)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(
        [f"{target_label(n)}\n→ raw" for n in names],
        rotation=0,
        fontsize=8,
    )
    ax.set_ylabel("test $R^2$ on raw amplitude")
    ax.set_title("Fair comparison: raw-fit vs log-fit GBM")
    for i, row in fair.reset_index(drop=True).iterrows():
        ax.text(
            i,
            row["r2"] + 0.01,
            f"MAE={row['mae']:.3f}",
            ha="center",
            fontsize=7,
        )

    ax = axes[1, 0]
    for tgt in eval_targets:
        s = pinball_df[pinball_df.target == tgt]
        if s.empty:
            continue
        ax.plot(
            s["tau"],
            s["pinball_loss"],
            "o-",
            color=target_color(tgt),
            label=target_label(tgt),
            ms=5,
        )
    ax.set_xlabel(r"quantile $\tau$")
    ax.set_ylabel("pinball loss (test)")
    ax.set_title("Quantile-model pinball loss")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1, 1]
    for tgt in eval_targets:
        if tgt not in quant_models or not quant_models[tgt]:
            continue
        yte = d50.iloc[te][tgt].values
        cov = []
        for q in taus:
            if q in quant_models[tgt]:
                cov.append(np.mean(yte <= quant_models[tgt][q].predict(Xte_df)))
        ax.plot(
            taus[: len(cov)],
            cov,
            "o-",
            color=target_color(tgt),
            label=target_label(tgt),
            ms=5,
        )
    ax.plot([0, 1], [0, 1], "--", color=REF_COLOR, lw=1)
    ax.set_xlabel(r"nominal $\tau$")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Quantile calibration (held-out seeds)")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1, 2]
    x = np.arange(len(FACTORS))
    n = len(eval_targets)
    w = 0.8 / max(n, 1)
    for i, tgt in enumerate(eval_targets):
        ax.bar(
            x + (i - (n - 1) / 2) * w,
            imp_df[tgt],
            w,
            color=target_color(tgt),
            label=target_label(tgt),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(FACTORS, rotation=30, fontsize=7.5)
    ax.set_ylabel("gain importance (%)")
    ax.set_title("GBM feature importance")
    ax.legend(fontsize=7, frameon=False)

    for i, ax in enumerate(axes.ravel()):
        panel_letter(ax, chr(97 + i))

    fig.suptitle(
        "Mean GBM benchmarks + QBM calibration (seed hold-out): "
        "raw vs ln amplitude are complementary, not competitors",
        fontsize=10,
        y=0.99,
    )
    fig.tight_layout()
    out = result_path("plots", "gbm_performance.png")
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    print(f"saved {out}")
    print(f"saved {metrics_path}")
    print(metrics_df.to_string(index=False))
    fair = metrics_df[metrics_df["evaluation_scale"] == "raw_amplitude"]
    raw_row = fair[fair["model_target"] == "abs_TF_ratio"].iloc[0]
    log_row = fair[fair["model_target"] == "log_abs"].iloc[0]
    print(
        "\nFair raw-amplitude mean-GBM comparison:\n"
        f"  raw-fit: R²={raw_row.r2:.3f}, MSE={raw_row.mse:.4f}, MAE={raw_row.mae:.4f}\n"
        f"  log-fit→raw (Duan): R²={log_row.r2:.3f}, MSE={log_row.mse:.4f}, "
        f"MAE={log_row.mae:.4f}, smear={log_row.smearing_factor:.3f}\n"
        "Note: QBM is not scored by R²; it models conditional quantiles "
        "(pinball/calibration), complementary to mean GBM."
    )


if __name__ == "__main__":
    main()
