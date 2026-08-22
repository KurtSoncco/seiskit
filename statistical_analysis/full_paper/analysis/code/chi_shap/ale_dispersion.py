"""1D ALE on matched dispersion / upper-tail targets (Fig18).

Two dual-model overlays per metric (same physical quantity on each figure):

1. ``ale_scale_<metric>.pdf`` — conditional scale
   - NGBoost σ(x)
   - QBM Normal-equivalent scale from quantiles:
     (q_{0.95} − q_{0.05}) / (2 Φ^{−1}(0.95))

2. ``ale_q95_<metric>.pdf`` — upper quantile of Y=ln χ
   - QBM τ=0.95
   - NGBoost μ(x) + Φ^{−1}(0.95) σ(x)

Do **not** overlay σ against q95 (incommensurable).

Writes under ``figure_dir("chi_shap", "ale_dispersion")``.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ale_effects import (  # noqa: E402
    ALE_N_BINS,
    ALE_SAMPLE_SEED,
    ALE_SUBSAMPLE_N,
    _as_booster,
    _predict_qbm,
    ale_1d,
)
from common import (  # noqa: E402
    FEATURES,
    METRICS,
    add_design_columns,
    load_or_make_split,
    load_ratios,
    ngboost_model_path,
    out_dir,
    qbm_model_path,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import (  # noqa: E402
    add_panel_label,
    apply_full_paper_style,
    factor_color,
    figsize,
    metric_label,
    save_figure,
)

warnings.filterwarnings("ignore")
apply_full_paper_style(auto_format=True, frame="open", grid=False)

Z95 = float(norm.ppf(0.95))

# (figure_stem_suffix, title, [(tag, linestyle, legend), ...])
FIGURES = (
    (
        "scale",
        r"conditional scale $\sigma$",
        (
            ("ngboost_sigma", "-", r"NGBoost $\sigma$"),
            ("qbm_sigma_proxy", "--", r"QBM $(q_{0.95}-q_{0.05})/(2z_{0.95})$"),
        ),
    ),
    (
        "q95",
        r"upper quantile $q_{0.95}$",
        (
            ("qbm_q95", "-", r"QBM $\tau=0.95$"),
            ("ngboost_q95", "--", r"NGBoost $\mu+z_{0.95}\sigma$"),
        ),
    ),
)


def _predict_ngb_sigma(model: NGBRegressor, X: np.ndarray) -> np.ndarray:
    dist = model.pred_dist(np.asarray(X, dtype=float))
    return np.maximum(np.asarray(dist.scale, dtype=float).ravel(), 1e-8)


def _predict_ngb_q95(model: NGBRegressor, X: np.ndarray) -> np.ndarray:
    dist = model.pred_dist(np.asarray(X, dtype=float))
    mu = np.asarray(dist.loc, dtype=float).ravel()
    sig = np.maximum(np.asarray(dist.scale, dtype=float).ravel(), 1e-8)
    return mu + Z95 * sig


def _load_predictors(metric: str) -> dict[str, object]:
    """Return predict_fn keyed by target tag."""
    out: dict[str, object] = {}

    npath = ngboost_model_path(metric)
    if not npath.is_file():
        raise FileNotFoundError(npath)
    ngb: NGBRegressor = joblib.load(npath)
    out["ngboost_sigma"] = lambda X, m=ngb: _predict_ngb_sigma(m, X)
    out["ngboost_q95"] = lambda X, m=ngb: _predict_ngb_q95(m, X)

    q05 = qbm_model_path("q05", metric)
    q95 = qbm_model_path("q95", metric)
    if not q05.is_file() or not q95.is_file():
        raise FileNotFoundError(f"Need QBM q05 and q95 for {metric}")
    b05 = _as_booster(joblib.load(q05))
    b95 = _as_booster(joblib.load(q95))
    out["qbm_q95"] = lambda X, b=b95: _predict_qbm(b, X)

    def _qbm_sigma_proxy(X, lo=b05, hi=b95):
        return (_predict_qbm(hi, X) - _predict_qbm(lo, X)) / (2.0 * Z95)

    out["qbm_sigma_proxy"] = _qbm_sigma_proxy
    return out


def _plot_overlay(
    curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    *,
    metric: str,
    stem: str,
    title_qty: str,
    series: tuple[tuple[str, str, str], ...],
    out: Path,
) -> None:
    n = len(FEATURES)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize(height=min(6.5, 2.0 * nrows)),
        squeeze=False,
    )
    for i, feat in enumerate(FEATURES):
        ax = axes[i // ncols, i % ncols]
        color = factor_color(feat) if feat != "node_z" else "0.35"
        for tag, ls, lab in series:
            x, y = curves[tag][feat]
            ax.plot(
                x,
                y,
                color=color,
                linestyle=ls,
                linewidth=1.0,
                label=lab if i == 0 else None,
            )
        ax.axhline(0.0, color="0.55", linewidth=0.5, linestyle=":")
        ax.set_xlabel(feat.replace("_z", r" $z$") if feat.endswith("_z") else feat)
        ax.set_ylabel(r"ALE")
        add_panel_label(ax, i)
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=2,
            fontsize=6,
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
        )
    fig.suptitle(
        f"{metric_label(metric, log=True)} — {title_qty}",
        fontsize=7,
        y=1.06,
    )
    fig.tight_layout(pad=0.35)
    save_figure(fig, f"ale_{stem}_{metric}", out_dir=out)
    plt.close(fig)


def main() -> None:
    out = out_dir("ale_dispersion")
    print("Loading data …")
    df = add_design_columns(load_ratios())
    _, te = load_or_make_split(df)
    rng = np.random.default_rng(ALE_SAMPLE_SEED)
    te = np.asarray(te)
    if len(te) > ALE_SUBSAMPLE_N:
        te = rng.choice(te, size=ALE_SUBSAMPLE_N, replace=False)
    X = df.iloc[te][FEATURES].to_numpy(dtype=float)

    rows = []
    meta = {
        "subsample_n": int(len(te)),
        "n_bins": ALE_N_BINS,
        "z95": Z95,
        "figures": [],
    }

    all_tags = sorted({tag for _, _, series in FIGURES for tag, _, _ in series})

    for metric in METRICS:
        print(f"ALE dispersion/tail {metric} …")
        predictors = _load_predictors(metric)
        curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        for tag in all_tags:
            predict_fn = predictors[tag]
            curves[tag] = {}
            for j, feat in enumerate(FEATURES):
                grid, ale = ale_1d(predict_fn, X, j, n_bins=ALE_N_BINS)
                curves[tag][feat] = (grid, ale)
                for x, y in zip(grid, ale):
                    rows.append(
                        {
                            "metric": metric,
                            "target": tag,
                            "feature": feat,
                            "x": float(x),
                            "effect": float(y),
                        }
                    )
            meta["figures"].append({"metric": metric, "target": tag})

        for stem, title_qty, series in FIGURES:
            _plot_overlay(
                curves,
                metric=metric,
                stem=stem,
                title_qty=title_qty,
                series=series,
                out=out,
            )

    tab = pd.DataFrame(rows)
    tab.to_csv(out / "ale_dispersion_curves.csv", index=False)
    amp = (
        tab.groupby(["metric", "target", "feature"], as_index=False)["effect"]
        .agg(effect_min="min", effect_max="max")
        .assign(effect_range=lambda d: d["effect_max"] - d["effect_min"])
        .sort_values(["metric", "target", "effect_range"], ascending=[True, True, False])
    )
    amp.to_csv(out / "ale_dispersion_effect_range.csv", index=False)
    (out / "ale_dispersion_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Legacy mismatched overlay ale_dispersion_<metric>.pdf may remain on Box;
    # prefer ale_scale_* / ale_q95_* (matched dual-model figures).

    lines = [
        "# ALE on matched dispersion and upper-tail targets (Fig18)",
        "",
        "## Matched comparisons",
        "",
        r"1. **Scale** (`ale_scale_<metric>.pdf`): NGBoost $\sigma$ vs QBM "
        r"$(q_{0.95}-q_{0.05})/(2z_{0.95})$.",
        r"2. **Upper quantile** (`ale_q95_<metric>.pdf`): QBM $\tau=0.95$ vs "
        r"NGBoost $\mu+z_{0.95}\sigma$.",
        "",
        "σ and q95 are never overlaid on the same axes.",
        "",
        "## Effect amplitude",
        "",
        amp.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Output files",
        "",
        "| File | Content |",
        "|------|---------|",
        "| `ale_scale_<metric>.pdf` | Dual-model ALE on scale |",
        "| `ale_q95_<metric>.pdf` | Dual-model ALE on $q_{0.95}$ |",
        "| `ale_dispersion_curves.csv` | curve points |",
        "| `ale_dispersion_effect_range.csv` | amplitudes |",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
