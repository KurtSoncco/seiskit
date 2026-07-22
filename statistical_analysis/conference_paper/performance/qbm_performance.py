"""QBM test performance: metrics CSV + 2×2 hold-out figure.

Evaluates pre-trained quantile models on the seed-grouped hold-out for
``log_abs`` and ``f_ratio``.

Scores (CSV)
------------
* Pseudo-R²(τ) at τ∈{0.05, 0.50, 0.95} vs intercept-only quantile null
* Pinball loss at those τ
* Empirical 90% PI coverage, interval score, WIS
* Mean PI width (sharpness; interpret with coverage)
* Calibration AUC / integrated pinball; median-model MAE / RMSE / R²

Figure (2×2) — edit panel functions below to restyle
----------------------------------------------------
* (a, c) Predicted median vs true + smoothed 90% PI envelope
* (b) Marginal quantile calibration
* (d) Relative PI90 sharpness vs predicted-median percentile
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
from config import (  # noqa: E402
    FACTORS,
    FIG_DPI,
    FIG_WIDTH,
    REF_COLOR,
    load_channel50,
    load_quantile_models,
    seed_grouped_split,
    target_color,
    target_label,
)
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import mean_absolute_error, mean_pinball_loss, mean_squared_error, r2_score

from seiskit.plot_config import apply_style, panel_letter, result_path

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EVAL_TARGETS = ["log_abs", "f_ratio"]
TAUS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
KEY_TAUS = (0.05, 0.50, 0.95)
SCATTER_SEED = 0  # fixed subsample for panels (a)/(c); do not change casually
# Central PIs: (alpha, lower_tau, upper_tau)
CENTRAL_INTERVALS = (
    (0.50, 0.25, 0.75),  # 50% PI
    (0.20, 0.10, 0.90),  # 80% PI
    (0.10, 0.05, 0.95),  # 90% PI
)

PRIMARY_METRIC_COLS = [
    "target",
    "pseudo_R2_t05",
    "pseudo_R2_t50",
    "pseudo_R2_t95",
    "pinball_t05",
    "pinball_t50",
    "pinball_t95",
    "PI90_coverage",
    "mean_interval_score_90",
    "mean_WIS",
    "WIS_skill",
    "mean_PI90_width",
]


# ===========================================================================
# Scoring helpers (no I/O, no plotting)
# ===========================================================================


def koenker_pseudo_r2(y_true: np.ndarray, y_pred: np.ndarray, tau: float, y_null: float) -> float:
    """Koenker–Machado R¹(τ): 1 − pinball(model) / pinball(unconditional τ-quantile)."""
    v_model = float(mean_pinball_loss(y_true, y_pred, alpha=tau))
    v_null = float(mean_pinball_loss(y_true, np.full_like(y_true, y_null), alpha=tau))
    if v_null <= 0:
        return np.nan
    return 1.0 - v_model / v_null


def interval_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> np.ndarray:
    """Gneiting–Raftery interval score for a central (1−α) prediction interval.

    IS_α(l,u,y) = (u−l) + (2/α)(l−y)1{y<l} + (2/α)(y−u)1{y>u}
    """
    width = upper - lower
    below = np.maximum(lower - y, 0.0)
    above = np.maximum(y - upper, 0.0)
    return width + (2.0 / alpha) * below + (2.0 / alpha) * above


def weighted_interval_score(
    y: np.ndarray,
    median: np.ndarray,
    intervals: list[tuple[float, np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Bracher et al. WIS from median + central intervals [(α, l, u), ...].

    WIS = 1/(K+1/2) * [ 0.5 |y−m| + Σ_k (α_k/2) IS_{α_k} ]
    """
    k = len(intervals)
    total = 0.5 * np.abs(y - median)
    for alpha, lower, upper in intervals:
        total = total + (alpha / 2.0) * interval_score(y, lower, upper, alpha)
    return total / (k + 0.5)


def predict_sorted(qmodels_tgt: dict, X, taus: list[float]) -> tuple[np.ndarray, float]:
    """Stack quantile predictions; return sorted array and pre-sort crossing rate."""
    raw = np.column_stack([np.asarray(qmodels_tgt[t].predict(X), dtype=float) for t in taus])
    crossings = float(np.mean(np.any(np.diff(raw, axis=1) < 0, axis=1)))
    return np.sort(raw, axis=1), crossings


def trapz_abs_miscal(taus: np.ndarray, coverages: np.ndarray) -> float:
    """∫ |ĉ(τ) − τ| dτ over the observed tau grid (miscalibration area)."""
    return float(np.trapezoid(np.abs(coverages - taus), taus))


def trapz_pinball(taus: np.ndarray, pinballs: np.ndarray) -> float:
    return float(np.trapezoid(pinballs, taus))


def central_pi_coverage(
    y: np.ndarray,
    preds: np.ndarray,
    tau_idx: dict[float, int],
) -> list[tuple[float, float]]:
    """Empirical coverage for each central PI in ``CENTRAL_INTERVALS``."""
    out: list[tuple[float, float]] = []
    for alpha, lo_tau, hi_tau in CENTRAL_INTERVALS:
        q_lo = preds[:, tau_idx[lo_tau]]
        q_hi = preds[:, tau_idx[hi_tau]]
        emp_cov = float(np.mean((y >= q_lo) & (y <= q_hi)))
        out.append((1.0 - alpha, emp_cov))
    return out


# ===========================================================================
# Plot data shaping (pure; used by panel functions)
# ===========================================================================


def binned_mean(
    x: np.ndarray,
    values: np.ndarray,
    *,
    n_bins: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin ``x`` into equal-width bins; return (centers, mean values)."""
    x = np.asarray(x, dtype=float)
    values = np.asarray(values, dtype=float)
    x_min, x_max = float(np.min(x)), float(np.max(x))
    if x_max <= x_min:
        return np.array([x_min]), np.array([float(np.mean(values))])
    edges = np.linspace(x_min, x_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_id = np.digitize(x, edges[1:-1], right=False)
    means = np.full(len(centers), np.nan)
    for b in range(len(centers)):
        mask = bin_id == b
        if np.any(mask):
            means[b] = float(np.mean(values[mask]))
    ok = np.isfinite(means)
    return centers[ok], means[ok]


def unique_quantile_cells(q05: np.ndarray, q50: np.ndarray, q95: np.ndarray) -> pd.DataFrame:
    """One row per unique predictive triple, sorted by median."""
    return (
        pd.DataFrame({"q05": q05, "q50": q50, "q95": q95})
        .drop_duplicates()
        .sort_values("q50", kind="mergesort")
        .reset_index(drop=True)
    )


def smoothed_pi_envelope(
    q05: np.ndarray,
    q50: np.ndarray,
    q95: np.ndarray,
    *,
    n_bins: int = 20,
    smooth_window: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Binned + rolling-smoothed ``(x=q50, lo=q05, hi=q95)`` envelope."""
    band = unique_quantile_cells(q05, q50, q95)
    x_q = band["q50"].to_numpy()
    x_b, lo_b = binned_mean(x_q, band["q05"].to_numpy(), n_bins=n_bins)
    x_hi, hi_b = binned_mean(x_q, band["q95"].to_numpy(), n_bins=n_bins)
    if len(x_b) != len(x_hi) or not np.allclose(x_b, x_hi):
        hi_b = np.interp(x_b, x_hi, hi_b)
    lo_s = pd.Series(lo_b).rolling(smooth_window, center=True, min_periods=1).mean().to_numpy()
    hi_s = pd.Series(hi_b).rolling(smooth_window, center=True, min_periods=1).mean().to_numpy()
    return x_b, lo_s, hi_s


def relative_width_profile(
    q50: np.ndarray,
    rel_width: np.ndarray,
    *,
    n_bins: int = 16,
    smooth_window: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Relative PI90 width vs within-target percentile of ``q50``."""
    cell = (
        pd.DataFrame({"q50": q50, "rel_width": rel_width})
        .drop_duplicates()
        .sort_values("q50", kind="mergesort")
        .reset_index(drop=True)
    )
    n_c = len(cell)
    x_pct = (np.arange(n_c) + 0.5) / n_c
    x_b, w_b = binned_mean(x_pct, cell["rel_width"].to_numpy(), n_bins=n_bins)
    if len(w_b) >= smooth_window:
        w_b = pd.Series(w_b).rolling(smooth_window, center=True, min_periods=1).mean().to_numpy()
    return x_b, w_b


# ===========================================================================
# Evaluation (metrics + plot payload; no plotting)
# ===========================================================================


def evaluate_target(
    tgt: str,
    *,
    yte: np.ndarray,
    ytr: np.ndarray,
    preds: np.ndarray,
    cross_rate: float,
    taus: list[float] = TAUS,
) -> tuple[dict, dict, dict]:
    """Score one target; return (metric_row, median_row, plot_payload)."""
    tau_arr = np.asarray(taus, dtype=float)
    tau_idx = {t: i for i, t in enumerate(taus)}

    coverages = np.array([float(np.mean(yte <= preds[:, j])) for j in range(len(taus))])
    pinballs = np.array(
        [float(mean_pinball_loss(yte, preds[:, j], alpha=t)) for j, t in enumerate(taus)]
    )
    null_qs = np.array([float(np.quantile(ytr, t)) for t in taus])
    null_pinballs = np.array(
        [
            float(mean_pinball_loss(yte, np.full_like(yte, nq), alpha=t))
            for t, nq in zip(taus, null_qs)
        ]
    )

    cal_auc = trapz_abs_miscal(tau_arr, coverages)
    cal_skill = float(np.clip(1.0 - 2.0 * cal_auc, 0.0, 1.0))
    i_model = trapz_pinball(tau_arr, pinballs)
    i_null = trapz_pinball(tau_arr, null_pinballs)
    pinball_skill = 1.0 - i_model / i_null if i_null > 0 else np.nan

    q05 = preds[:, tau_idx[0.05]]
    q10 = preds[:, tau_idx[0.10]]
    q25 = preds[:, tau_idx[0.25]]
    q50 = preds[:, tau_idx[0.50]]
    q75 = preds[:, tau_idx[0.75]]
    q90 = preds[:, tau_idx[0.90]]
    q95 = preds[:, tau_idx[0.95]]
    w90 = q95 - q05
    upper = q95 - q50
    lower = q50 - q05
    with np.errstate(divide="ignore", invalid="ignore"):
        asym = np.where(lower > 0, upper / lower, np.nan)

    is90 = interval_score(yte, q05, q95, alpha=0.10)
    wis = weighted_interval_score(
        yte,
        q50,
        [
            (0.50, q25, q75),
            (0.20, q10, q90),
            (0.10, q05, q95),
        ],
    )
    null_intervals = [
        (
            alpha,
            np.full_like(yte, float(np.quantile(ytr, lo))),
            np.full_like(yte, float(np.quantile(ytr, hi))),
        )
        for alpha, lo, hi in CENTRAL_INTERVALS
    ]
    wis_null = weighted_interval_score(
        yte, np.full_like(yte, float(np.quantile(ytr, 0.50))), null_intervals
    )
    wis_skill = (
        1.0 - float(np.mean(wis)) / float(np.mean(wis_null)) if np.mean(wis_null) > 0 else np.nan
    )

    pseudo = {
        t: koenker_pseudo_r2(yte, preds[:, tau_idx[t]], t, float(np.quantile(ytr, t)))
        for t in KEY_TAUS
    }

    mae = float(mean_absolute_error(yte, q50))
    rmse = float(np.sqrt(mean_squared_error(yte, q50)))
    r2 = float(r2_score(yte, q50))
    pi90_cov = float(np.mean((yte >= q05) & (yte <= q95)))
    null_w90 = float(np.quantile(ytr, 0.95) - np.quantile(ytr, 0.05))
    rel_width = w90 / null_w90 if null_w90 > 0 else np.full_like(w90, np.nan)

    metric_row = dict(
        target=tgt,
        pseudo_R2_t05=round(pseudo[0.05], 4),
        pseudo_R2_t50=round(pseudo[0.50], 4),
        pseudo_R2_t95=round(pseudo[0.95], 4),
        pinball_t05=round(float(pinballs[tau_idx[0.05]]), 5),
        pinball_t50=round(float(pinballs[tau_idx[0.50]]), 5),
        pinball_t95=round(float(pinballs[tau_idx[0.95]]), 5),
        PI90_coverage=round(pi90_cov, 4),
        mean_interval_score_90=round(float(np.mean(is90)), 4),
        mean_WIS=round(float(np.mean(wis)), 4),
        WIS_skill=round(wis_skill, 4),
        mean_PI90_width=round(float(np.mean(w90)), 4),
        width_min=round(float(np.min(w90)), 4),
        width_max=round(float(np.max(w90)), 4),
        width_ratio_max_min=round(float(np.max(w90) / np.min(w90)), 1),
        mean_upper_lower_width_ratio=round(float(np.nanmean(asym)), 2),
        calibration_AUC=round(cal_auc, 4),
        calibration_skill=round(cal_skill, 4),
        integrated_pinball=round(i_model, 5),
        integrated_pinball_skill=round(pinball_skill, 4),
        quantile_crossing_rate=round(cross_rate, 4),
    )
    for t, cov in zip(taus, coverages):
        metric_row[f"coverage_t{int(t * 100):02d}"] = round(float(cov), 4)

    median_row = dict(
        target=tgt,
        tau=0.50,
        MAE=round(mae, 5),
        RMSE=round(rmse, 5),
        R2=round(r2, 4),
    )

    plot_payload = dict(
        target=tgt,
        y=yte,
        q05=q05,
        q50=q50,
        q95=q95,
        coverages=coverages,
        cal_auc=cal_auc,
        pi90_cov=pi90_cov,
        pi_reliability=central_pi_coverage(yte, preds, tau_idx),
        mae=mae,
        r2=r2,
        rel_width=rel_width,
        mean_rel_width=float(np.nanmean(rel_width)),
    )
    return metric_row, median_row, plot_payload


def evaluate_all(
    test: pd.DataFrame,
    train: pd.DataFrame,
    qmodels: dict[str, dict],
    *,
    targets: list[str] = EVAL_TARGETS,
    taus: list[float] = TAUS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict]]:
    """Evaluate all targets; return metrics_df, median_df, plot_data."""
    metric_rows: list[dict] = []
    median_rows: list[dict] = []
    plot_data: dict[str, dict] = {}

    for tgt in targets:
        yte = test[tgt].to_numpy(dtype=float)
        ytr = train[tgt].to_numpy(dtype=float)
        preds, cross_rate = predict_sorted(qmodels[tgt], test[FACTORS], taus)
        metric_row, median_row, payload = evaluate_target(
            tgt, yte=yte, ytr=ytr, preds=preds, cross_rate=cross_rate, taus=taus
        )
        metric_rows.append(metric_row)
        median_rows.append(median_row)
        plot_data[tgt] = payload

    return pd.DataFrame(metric_rows), pd.DataFrame(median_rows), plot_data


def save_and_print_metrics(
    metrics_df: pd.DataFrame,
    median_df: pd.DataFrame,
) -> tuple[Path, Path]:
    """Write CSVs and print primary tables. Return (metrics_path, median_path)."""
    csv_path = result_path("data", "qbm_performance.csv")
    metrics_df.to_csv(csv_path, index=False)
    median_csv_path = result_path("data", "qbm_median_classical_metrics.csv")
    median_df.to_csv(median_csv_path, index=False)

    print("\n=== Primary QBM scores (seed hold-out) ===")
    print(metrics_df[PRIMARY_METRIC_COLS].to_string(index=False))
    print(f"\nsaved {csv_path}")
    print("\n=== Median model (τ=0.50) classical metrics ===")
    print(median_df.to_string(index=False))
    print(f"\nsaved {median_csv_path}")
    return csv_path, median_csv_path


# ===========================================================================
# Panel plotters — edit these to restyle the figure
# ===========================================================================


def plot_pred_vs_true(
    ax: Axes,
    payload: dict,
    *,
    color: str | None = None,
    scatter_frac: float = 0.08,
    scatter_seed: int = 0,
    n_bins: int = 20,
    smooth_window: int = 5,
) -> None:
    """Panel (a)/(c): predicted median vs true with smoothed 90% PI envelope."""
    y = payload["y"]
    q05, q50, q95 = payload["q05"], payload["q50"], payload["q95"]
    color = color or target_color(payload["target"])
    label = target_label(payload["target"])

    x_env, lo_env, hi_env = smoothed_pi_envelope(
        q05, q50, q95, n_bins=n_bins, smooth_window=smooth_window
    )
    x_min, x_max = float(np.min(q50)), float(np.max(q50))

    ax.fill_between(
        x_env,
        lo_env,
        hi_env,
        color=color,
        alpha=0.22,
        linewidth=0,
        label=r"$q_{0.05}$–$q_{0.95}$",
        zorder=1,
    )
    ax.plot(x_env, lo_env, color=color, lw=1.5, zorder=2)
    ax.plot(x_env, hi_env, color=color, lw=1.5, zorder=2)

    n = len(y)
    n_show = max(200, int(round(scatter_frac * n)))
    sub = np.random.default_rng(scatter_seed).choice(n, size=min(n, n_show), replace=False)
    ax.scatter(
        q50[sub],
        y[sub],
        s=12,
        color="0.1",
        alpha=0.45,
        edgecolor="none",
        rasterized=True,
        zorder=3,
        label="Test observations",
    )
    ax.plot(
        [x_min, x_max], [x_min, x_max], "--", color=REF_COLOR, lw=1.2, label=r"$y=x$", zorder=4
    )  # Ideal line y=x

    x_pad = 0.06 * (x_max - x_min or 1.0)  # Padding for x-axis, 6% of the range
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    y_lo = float(min(y.min(), lo_env.min()))
    y_hi = float(max(y.max(), hi_env.max()))
    y_pad = 0.06 * (y_hi - y_lo or 1.0)
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

    ax.set_xlabel(r"QBM $\hat{q}_{0.50}$")
    ax.set_ylabel(label)
    ax.set_title(
        # Avoid the word "cov" — auto_format maps it to CoV (coeff. of variation).
        f"{label}: PI$_{{90}}$ coverage={payload['pi90_cov']:.2f}, "
        f"$R^2$={payload['r2']:.2f}, MAE={payload['mae']:.3g}",
        loc="left",
    )
    ax.legend(
        fontsize=10,
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.5,
    )


def plot_quantile_calibration(
    ax: Axes,
    plot_data: dict[str, dict],
    *,
    targets: list[str] = EVAL_TARGETS,
    taus: list[float] = TAUS,
) -> None:
    """Panel (b): empirical coverage vs nominal τ."""
    ax.plot([0, 1], [0, 1], "--", color=REF_COLOR, lw=1, label="Ideal coverage")
    for tgt in targets:
        pd_ = plot_data[tgt]
        ax.plot(
            taus,
            pd_["coverages"],
            "o-",
            color=target_color(tgt),
            ms=5,
            lw=1.5,
            label=target_label(tgt),
        )
    ax.set_xlabel(r"nominal $\tau$")
    ax.set_ylabel("Empirical Coverage")
    ax.set_title("Quantile calibration (Seed hold-out Test)", loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(
        fontsize=10,
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.5,
    )


def plot_interval_sharpness(
    ax: Axes,
    plot_data: dict[str, dict],
    *,
    targets: list[str] = EVAL_TARGETS,
    n_bins: int = 16,
    smooth_window: int = 3,
) -> None:
    """Panel (d): relative PI90 width vs predicted-median percentile."""
    ax.axhline(1.0, ls="--", color=REF_COLOR, lw=1, label="Null width")
    for tgt in targets:
        pd_ = plot_data[tgt]
        x_b, w_b = relative_width_profile(
            pd_["q50"],
            pd_["rel_width"],
            n_bins=n_bins,
            smooth_window=smooth_window,
        )
        ax.plot(
            x_b,
            w_b,
            "o-",
            color=target_color(tgt),
            ms=4,
            lw=1.5,
            label=f"{target_label(tgt)} (mean={pd_['mean_rel_width']:.2f})",
        )
    ax.set_xlabel(r"Percentile of QBM $\hat{q}_{0.50}$")
    ax.set_ylabel("Relative PI90 width (vs Null)")
    ax.set_title("Interval sharpness across predicted median", loc="left")
    ax.set_xlim(0, 1)
    ax.legend(
        fontsize=10,
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.5,
    )


def make_performance_figure(
    plot_data: dict[str, dict],
    *,
    targets: list[str] = EVAL_TARGETS,
    scatter_seed: int = SCATTER_SEED,
) -> Figure:
    """Assemble the 2×2 performance figure from ``plot_data``."""
    fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH * 2, FIG_WIDTH * 1.5))

    for i, tgt in enumerate(targets):
        plot_pred_vs_true(
            axes[i, 0],
            plot_data[tgt],
            scatter_seed=scatter_seed + i,
        )

    plot_quantile_calibration(axes[0, 1], plot_data, targets=targets)
    plot_interval_sharpness(axes[1, 1], plot_data, targets=targets)

    for j, axx in enumerate(axes.flat):
        panel_letter(axx, string.ascii_lowercase[j])

    fig.suptitle(
        "QBM hold-out performance: predictive accuracy, calibration, and sharpness",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    return fig


# ===========================================================================
# Entry point
# ===========================================================================


def main() -> None:
    apply_style(auto_format=True, font_size=10, frame="open")

    d = load_channel50()
    tr, te = seed_grouped_split(d, test_size=0.25, seed=0)
    test = d.iloc[te].copy().reset_index(drop=True)
    train = d.iloc[tr]

    qmodels = load_quantile_models(taus=TAUS, targets=EVAL_TARGETS, split_by="seed")
    missing = [t for t in EVAL_TARGETS if any(tau not in qmodels.get(t, {}) for tau in TAUS)]
    if missing:
        raise FileNotFoundError(
            f"Missing quantile models for {missing}. Train with: "
            "python quantile/quantile_channel_model.py"
        )

    metrics_df, median_df, plot_data = evaluate_all(test, train, qmodels)
    save_and_print_metrics(metrics_df, median_df)

    fig = make_performance_figure(plot_data)
    out = result_path("plots", "qbm_performance.png")
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
