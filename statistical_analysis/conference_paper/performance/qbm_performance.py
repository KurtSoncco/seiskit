"""QBM test performance: pseudo-R², pinball, interval scores, cell-ordered PIs.

Evaluates pre-trained quantile models on the seed-grouped hold-out for
``log_abs`` and ``f_ratio``. Primary scores:

* Pseudo-R²(τ) at τ∈{0.05, 0.50, 0.95} vs intercept-only quantile null
* Pinball loss at those τ (training / evaluation loss)
* Empirical 90% PI coverage
* Interval score (90%) and weighted interval score (WIS)
* Mean predictive-interval width (sharpness; interpret with coverage)

Also writes calibration AUC / integrated pinball and a figure with
factorial-cell-ordered 5–95% bands (not test-set order).
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
    DEFAULT_TAUS,
    FACTORS,
    FIG_DPI,
    REF_COLOR,
    load_channel50,
    load_quantile_models,
    seed_grouped_split,
    target_color,
    target_label,
)
from sklearn.metrics import mean_pinball_loss

from seiskit.plot_config import apply_style, panel_letter, result_path

warnings.filterwarnings("ignore")

EVAL_TARGETS = ["log_abs", "f_ratio"]
TAUS = list(DEFAULT_TAUS)
KEY_TAUS = (0.05, 0.50, 0.95)
# Central PIs available from DEFAULT_TAUS: (alpha, lower_tau, upper_tau)
CENTRAL_INTERVALS = (
    (0.50, 0.25, 0.75),  # 50% PI
    (0.20, 0.10, 0.90),  # 80% PI
    (0.10, 0.05, 0.95),  # 90% PI
)


def _koenker_pseudo_r2(y_true: np.ndarray, y_pred: np.ndarray, tau: float, y_null: float) -> float:
    """Koenker–Machado R¹(τ): 1 − pinball(model) / pinball(unconditional τ-quantile)."""
    v_model = float(mean_pinball_loss(y_true, y_pred, alpha=tau))
    v_null = float(mean_pinball_loss(y_true, np.full_like(y_true, y_null), alpha=tau))
    if v_null <= 0:
        return np.nan
    return 1.0 - v_model / v_null


def _interval_score(
    y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float
) -> np.ndarray:
    """Gneiting–Raftery interval score for a central (1−α) prediction interval.

    IS_α(l,u,y) = (u−l) + (2/α)(l−y)1{y<l} + (2/α)(y−u)1{y>u}
    """
    width = upper - lower
    below = np.maximum(lower - y, 0.0)
    above = np.maximum(y - upper, 0.0)
    return width + (2.0 / alpha) * below + (2.0 / alpha) * above


def _weighted_interval_score(
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
        total = total + (alpha / 2.0) * _interval_score(y, lower, upper, alpha)
    return total / (k + 0.5)


def _predict_sorted(qmodels_tgt: dict, X, taus: list[float]) -> tuple[np.ndarray, float]:
    """Stack quantile predictions; return sorted array and pre-sort crossing rate."""
    raw = np.column_stack([np.asarray(qmodels_tgt[t].predict(X), dtype=float) for t in taus])
    crossings = float(np.mean(np.any(np.diff(raw, axis=1) < 0, axis=1)))
    sorted_pred = np.sort(raw, axis=1)
    return sorted_pred, crossings


def _trapz_abs_miscal(taus: np.ndarray, coverages: np.ndarray) -> float:
    """∫ |ĉ(τ) − τ| dτ over the observed tau grid (miscalibration area)."""
    return float(np.trapezoid(np.abs(coverages - taus), taus))


def _trapz_pinball(taus: np.ndarray, pinballs: np.ndarray) -> float:
    return float(np.trapezoid(pinballs, taus))


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

    # Lex order of unique design cells (FACTORS order).
    cells = (
        d[FACTORS].drop_duplicates().sort_values(FACTORS, kind="mergesort").reset_index(drop=True)
    )
    cells["cell_rank"] = np.arange(len(cells))
    test = test.merge(cells, on=FACTORS, how="left")

    tau_arr = np.asarray(TAUS, dtype=float)
    metric_rows = []
    plot_data: dict[str, dict] = {}

    for tgt in EVAL_TARGETS:
        yte = test[tgt].to_numpy(dtype=float)
        ytr = train[tgt].to_numpy(dtype=float)
        Xte = test[FACTORS]
        preds, cross_rate = _predict_sorted(qmodels[tgt], Xte, TAUS)
        tau_idx = {t: i for i, t in enumerate(TAUS)}

        coverages = np.array([float(np.mean(yte <= preds[:, j])) for j in range(len(TAUS))])
        pinballs = np.array(
            [float(mean_pinball_loss(yte, preds[:, j], alpha=t)) for j, t in enumerate(TAUS)]
        )
        null_qs = np.array([float(np.quantile(ytr, t)) for t in TAUS])
        null_pinballs = np.array(
            [
                float(mean_pinball_loss(yte, np.full_like(yte, nq), alpha=t))
                for t, nq in zip(TAUS, null_qs)
            ]
        )

        cal_auc = _trapz_abs_miscal(tau_arr, coverages)
        cal_skill = float(np.clip(1.0 - 2.0 * cal_auc, 0.0, 1.0))
        i_model = _trapz_pinball(tau_arr, pinballs)
        i_null = _trapz_pinball(tau_arr, null_pinballs)
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

        # 90% interval score + WIS over 50/80/90% central bands + median.
        is90 = _interval_score(yte, q05, q95, alpha=0.10)
        wis = _weighted_interval_score(
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
        wis_null = _weighted_interval_score(
            yte, np.full_like(yte, float(np.quantile(ytr, 0.50))), null_intervals
        )
        wis_skill = (
            1.0 - float(np.mean(wis)) / float(np.mean(wis_null))
            if np.mean(wis_null) > 0
            else np.nan
        )

        pseudo = {
            t: _koenker_pseudo_r2(yte, preds[:, tau_idx[t]], t, float(np.quantile(ytr, t)))
            for t in KEY_TAUS
        }

        row = dict(
            target=tgt,
            # Relative improvement vs intercept-only quantile model
            pseudo_R2_t05=round(pseudo[0.05], 4),
            pseudo_R2_t50=round(pseudo[0.50], 4),
            pseudo_R2_t95=round(pseudo[0.95], 4),
            # Actual quantile training / evaluation loss
            pinball_t05=round(float(pinballs[tau_idx[0.05]]), 5),
            pinball_t50=round(float(pinballs[tau_idx[0.50]]), 5),
            pinball_t95=round(float(pinballs[tau_idx[0.95]]), 5),
            # Calibrated uncertainty
            PI90_coverage=round(float(np.mean((yte >= q05) & (yte <= q95))), 4),
            # Proper scoring rules for bands
            mean_interval_score_90=round(float(np.mean(is90)), 4),
            mean_WIS=round(float(np.mean(wis)), 4),
            WIS_skill=round(wis_skill, 4),
            # Sharpness (interpret jointly with coverage)
            mean_PI90_width=round(float(np.mean(w90)), 4),
            width_min=round(float(np.min(w90)), 4),
            width_max=round(float(np.max(w90)), 4),
            width_ratio_max_min=round(float(np.max(w90) / np.min(w90)), 1),
            mean_upper_lower_width_ratio=round(float(np.nanmean(asym)), 2),
            # Extra diagnostics
            calibration_AUC=round(cal_auc, 4),
            calibration_skill=round(cal_skill, 4),
            integrated_pinball=round(i_model, 5),
            integrated_pinball_skill=round(pinball_skill, 4),
            quantile_crossing_rate=round(cross_rate, 4),
        )
        for t, cov in zip(TAUS, coverages):
            row[f"coverage_t{int(t * 100):02d}"] = round(float(cov), 4)
        metric_rows.append(row)

        cell_band = (
            test.assign(q05=q05, q50=q50, q95=q95)
            .groupby("cell_rank", as_index=False)[["q05", "q50", "q95"]]
            .first()
            .sort_values("cell_rank")
        )
        plot_data[tgt] = dict(
            y=yte,
            cell_rank=test["cell_rank"].to_numpy(),
            q05=q05,
            q50=q50,
            q95=q95,
            cell_band=cell_band,
            coverages=coverages,
            pinballs=pinballs,
            cal_auc=cal_auc,
            pinball_skill=pinball_skill,
            mean_is90=float(np.mean(is90)),
            mean_wis=float(np.mean(wis)),
            wis_skill=wis_skill,
            pi90_cov=float(np.mean((yte >= q05) & (yte <= q95))),
            mean_width=float(np.mean(w90)),
        )

    metrics_df = pd.DataFrame(metric_rows)
    csv_path = result_path("data", "qbm_performance.csv")
    metrics_df.to_csv(csv_path, index=False)

    primary_cols = [
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
    print("\n=== Primary QBM scores (seed hold-out) ===")
    print(metrics_df[primary_cols].to_string(index=False))
    print(f"\nsaved {csv_path}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    for i, tgt in enumerate(EVAL_TARGETS):
        ax = axes[i, 0]
        col = target_color(tgt)
        pd_ = plot_data[tgt]
        band = pd_["cell_band"]
        x = band["cell_rank"].to_numpy()
        ax.fill_between(
            x,
            band["q05"],
            band["q95"],
            step="mid",
            color=col,
            alpha=0.22,
            label=r"QBM $q_{0.05}$–$q_{0.95}$",
        )
        ax.plot(x, band["q50"], color=col, lw=1.0, drawstyle="steps-mid", label="QBM median")
        n = len(pd_["y"])
        sub = np.linspace(0, n - 1, min(n, 4000)).astype(int)
        ax.scatter(
            pd_["cell_rank"][sub],
            pd_["y"][sub],
            s=4,
            color="k",
            alpha=0.22,
            zorder=3,
            label="test obs.",
            rasterized=True,
        )
        ax.set_xlabel("Factorial cell rank (Vs1→Height→CoV→rH→aHV)")
        ax.set_ylabel(target_label(tgt))
        ax.set_title(
            f"{target_label(tgt)}: cov90={pd_['pi90_cov']:.2f}, "
            f"IS90={pd_['mean_is90']:.3g}, width={pd_['mean_width']:.3g}",
            loc="left",
        )
        ax.set_xlim(-1, len(cells))
        ax.legend(fontsize=7, frameon=False, loc="upper right")

    ax = axes[0, 1]
    ax.plot([0, 1], [0, 1], "--", color=REF_COLOR, lw=1, label="ideal")
    for tgt in EVAL_TARGETS:
        pd_ = plot_data[tgt]
        ax.plot(
            TAUS,
            pd_["coverages"],
            "o-",
            color=target_color(tgt),
            ms=5,
            lw=1.5,
            label=f"{target_label(tgt)} (AUC={pd_['cal_auc']:.3f})",
        )
    ax.set_xlabel(r"nominal $\tau$")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Quantile calibration (seed hold-out)", loc="left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1, 1]
    for tgt in EVAL_TARGETS:
        pd_ = plot_data[tgt]
        ax.plot(
            TAUS,
            pd_["pinballs"],
            "o-",
            color=target_color(tgt),
            ms=5,
            lw=1.5,
            label=(
                f"{target_label(tgt)} "
                f"(pinball skill={pd_['pinball_skill']:.2f}, WIS skill={pd_['wis_skill']:.2f})"
            ),
        )
    ax.set_xlabel(r"quantile $\tau$")
    ax.set_ylabel("pinball loss (test)")
    ax.set_title("Pinball loss vs τ", loc="left")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=7, frameon=False)

    for j, axx in enumerate(axes.flat):
        panel_letter(axx, string.ascii_lowercase[j])

    fig.suptitle(
        "QBM test performance: cell-ordered predictive intervals "
        "(ordered by design factors, not test index)",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    out = result_path("plots", "qbm_performance.png")
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
