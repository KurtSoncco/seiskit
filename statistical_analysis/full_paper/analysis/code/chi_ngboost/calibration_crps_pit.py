"""Holdout CRPS and PIT calibration for Normal NGBoost (Y = ln χ).

Loads the seed-grouped holdout via ``load_or_make_split``, predicts
Normal(μ, σ) per metric, and reports closed-form CRPS plus PIT =
Φ((y−μ)/σ). Writes under figure_dir("chi_ngboost", "calibration").
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FEATURES,
    METRICS,
    add_design_columns,
    load_or_make_split,
    load_ratios,
    log_response,
    models_dir,
    out_dir,
)
from train_ngboost import predict_params  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import (  # noqa: E402
    add_panel_label,
    apply_full_paper_style,
    figsize,
    metric_color,
    metric_label,
    save_figure,
)

warnings.filterwarnings("ignore")
apply_full_paper_style(auto_format=True, frame="open", grid=False)

N_PIT_BINS = 20
INV_SQRT_PI = 1.0 / np.sqrt(np.pi)


def crps_normal(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Closed-form CRPS for a Normal predictive distribution (Gneiting et al.)."""
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-8)
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    z = (y - mu) / sigma
    return sigma * (z * (2.0 * stats.norm.cdf(z) - 1.0) + 2.0 * stats.norm.pdf(z) - INV_SQRT_PI)


def pit_normal(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-8)
    return stats.norm.cdf((np.asarray(y, dtype=float) - np.asarray(mu, dtype=float)) / sigma)


def _plot_pit_histograms(pit_by_metric: dict[str, np.ndarray], out: Path) -> None:
    n = len(METRICS)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize(height=min(6.5, 2.2 * nrows)),
        squeeze=False,
    )
    for i, metric in enumerate(METRICS):
        ax = axes[i // ncols, i % ncols]
        pit = pit_by_metric[metric]
        ax.hist(
            pit,
            bins=N_PIT_BINS,
            range=(0.0, 1.0),
            density=True,
            color=metric_color(metric),
            edgecolor="white",
            linewidth=0.4,
        )
        ax.axhline(1.0, color="0.35", linewidth=0.6, linestyle="--")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("PIT")
        ax.set_ylabel("Density")
        ax.set_title(metric_label(metric, log=True), fontsize=7)
        add_panel_label(ax, i)
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)
    fig.tight_layout(pad=0.4)
    save_figure(fig, "pit_histograms", out_dir=out)
    plt.close(fig)


def main() -> None:
    out = out_dir("calibration")
    print("Loading data …")
    df = add_design_columns(load_ratios())
    tr, te = load_or_make_split(df)
    X_te = df.iloc[te][FEATURES].to_numpy(dtype=float)

    rows = []
    pit_by_metric: dict[str, np.ndarray] = {}
    detail_rows = []

    for metric in METRICS:
        mpath = models_dir() / f"ngboost_{metric}.pkl"
        if not mpath.is_file():
            raise FileNotFoundError(f"Missing NGBoost model: {mpath}")
        model: NGBRegressor = joblib.load(mpath)
        y = log_response(df.iloc[te], metric)
        m = np.isfinite(y) & np.all(np.isfinite(X_te), axis=1)
        y_m = y[m]
        X_m = X_te[m]
        print(f"CRPS/PIT {metric}  n={len(y_m)} …")
        mu, sigma = predict_params(model, X_m)
        sigma = np.maximum(sigma, 1e-8)
        crps = crps_normal(y_m, mu, sigma)
        pit = pit_normal(y_m, mu, sigma)
        pit_by_metric[metric] = pit

        # Uniform PIT → E[PIT]=0.5, Var=1/12; KS vs Uniform(0,1)
        ks = stats.kstest(pit, "uniform")
        rows.append(
            {
                "metric": metric,
                "n_holdout": int(len(y_m)),
                "mean_crps": float(np.mean(crps)),
                "median_crps": float(np.median(crps)),
                "mean_pit": float(np.mean(pit)),
                "std_pit": float(np.std(pit, ddof=0)),
                "pit_var_target": 1.0 / 12.0,
                "frac_pit_lt_0_1": float(np.mean(pit < 0.1)),
                "frac_pit_gt_0_9": float(np.mean(pit > 0.9)),
                "ks_stat": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
            }
        )
        # Compact per-decile occupancy for diagnostics
        for d in range(10):
            lo, hi = d / 10.0, (d + 1) / 10.0
            detail_rows.append(
                {
                    "metric": metric,
                    "pit_bin": f"[{lo:.1f},{hi:.1f})",
                    "frac": float(np.mean((pit >= lo) & (pit < hi)))
                    if d < 9
                    else float(np.mean((pit >= lo) & (pit <= hi))),
                    "expected": 0.1,
                }
            )

    tab = pd.DataFrame(rows)
    detail = pd.DataFrame(detail_rows)
    tab.to_csv(out / "crps_pit_summary.csv", index=False)
    detail.to_csv(out / "pit_decile_occupancy.csv", index=False)
    _plot_pit_histograms(pit_by_metric, out)

    lines = [
        "# NGBoost CRPS and PIT calibration",
        "",
        "## Definitions",
        "",
        r"- Predictive family: Normal NGBoost on \(Y=\ln\chi\) with parameters "
        r"\((\mu(\mathbf{x}),\sigma(\mathbf{x}))\).",
        r"- Holdout: same seed-grouped split as `chi_qbm` (`load_or_make_split`).",
        r"- CRPS (closed form for Normal): "
        r"\(\mathrm{CRPS}=\sigma\bigl[z(2\Phi(z)-1)+2\phi(z)-1/\sqrt{\pi}\bigr]\), "
        r"\(z=(y-\mu)/\sigma\).",
        r"- PIT: \(u=\Phi((y-\mu)/\sigma)\). Under a calibrated continuous forecast, "
        r"\(u\sim\mathrm{Uniform}(0,1)\).",
        r"- KS: one-sample Kolmogorov–Smirnov test of PIT against Uniform(0,1).",
        "",
        "## Mean CRPS and PIT diagnostics",
        "",
        tab.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Calibration notes",
        "",
        "- Flat PIT histograms (density ≈ 1) and mean PIT near 0.5 indicate good "
        "probabilistic calibration on the holdout.",
        "- U-shaped PIT implies underdispersed forecasts (σ too small); "
        "inverse-U implies overdispersion.",
        "- Systematic left/right skew in PIT indicates biased μ.",
        "- CRPS is a proper scoring rule combining sharpness and calibration; "
        "lower is better. Compare across metrics only with care — response scales differ.",
        "- Residual spatial lag-1 (see `train_ngboost`) is a separate issue: "
        "well-calibrated marginal Normal predictions can still leave short-range dependence.",
        "",
        "## Output files",
        "",
        "| File | Content |",
        "|------|---------|",
        "| `crps_pit_summary.csv` | mean/median CRPS, PIT moments, KS |",
        "| `pit_decile_occupancy.csv` | empirical vs expected PIT decile mass |",
        "| `pit_histograms.pdf` | PIT density histograms by metric |",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(tab.to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
