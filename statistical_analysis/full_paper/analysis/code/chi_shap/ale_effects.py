"""1D ALE (fallback: PDP) marginal effects on median IM (Y = ln χ).

Prefers QBM τ=0.50 LightGBM; falls back to NGBoost μ when q50 models are
missing. Implements a simple 1D ALE estimator (no external ALE package).
Nature PDFs under figure_dir("chi_shap", "ale_effects").
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ngboost import NGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
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

ALE_SUBSAMPLE_N = 4000
ALE_N_BINS = 20
ALE_SAMPLE_SEED = 4
PDP_GRID_N = 25


def _as_booster(obj) -> lgb.Booster:
    if isinstance(obj, lgb.Booster):
        return obj
    if hasattr(obj, "booster_"):
        return obj.booster_
    return obj


def _predict_qbm(booster: lgb.Booster, X: np.ndarray) -> np.ndarray:
    return np.asarray(booster.predict(X), dtype=float).ravel()


def _predict_ngb_mu(model: NGBRegressor, X: np.ndarray) -> np.ndarray:
    dist = model.pred_dist(np.asarray(X, dtype=float))
    return np.asarray(dist.loc, dtype=float).ravel()


def ale_1d(
    predict_fn,
    X: np.ndarray,
    j: int,
    *,
    n_bins: int = ALE_N_BINS,
) -> tuple[np.ndarray, np.ndarray]:
    """Centered 1D accumulated local effects for feature column *j*.

    Returns (grid_centers, ale_values). Falls back to PDP if a feature has
    too few unique values for binning.
    """
    xj = np.asarray(X[:, j], dtype=float)
    uniq = np.unique(xj[np.isfinite(xj)])
    if uniq.size < 3:
        return pdp_1d(predict_fn, X, j, n_grid=min(PDP_GRID_N, max(uniq.size, 2)))

    # Quantile edges; drop duplicate edges for discrete-ish features
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(xj, qs))
    if edges.size < 3:
        return pdp_1d(predict_fn, X, j, n_grid=min(PDP_GRID_N, edges.size))

    effects = []
    centers = []
    for b in range(edges.size - 1):
        lo, hi = edges[b], edges[b + 1]
        if b < edges.size - 2:
            mask = (xj >= lo) & (xj < hi)
        else:
            mask = (xj >= lo) & (xj <= hi)
        if not np.any(mask):
            continue
        X_lo = X[mask].copy()
        X_hi = X[mask].copy()
        X_lo[:, j] = lo
        X_hi[:, j] = hi
        delta = predict_fn(X_hi) - predict_fn(X_lo)
        effects.append(float(np.mean(delta)))
        centers.append(0.5 * (lo + hi))

    if len(effects) < 2:
        return pdp_1d(predict_fn, X, j, n_grid=PDP_GRID_N)

    ale = np.cumsum(np.asarray(effects, dtype=float))
    ale = ale - float(np.mean(ale))
    return np.asarray(centers, dtype=float), ale


def pdp_1d(
    predict_fn,
    X: np.ndarray,
    j: int,
    *,
    n_grid: int = PDP_GRID_N,
) -> tuple[np.ndarray, np.ndarray]:
    """Centered 1D partial dependence for feature column *j*."""
    xj = np.asarray(X[:, j], dtype=float)
    grid = np.unique(np.quantile(xj, np.linspace(0.0, 1.0, n_grid)))
    vals = np.empty(grid.size, dtype=float)
    X_work = X.copy()
    for i, g in enumerate(grid):
        X_work[:, j] = g
        vals[i] = float(np.mean(predict_fn(X_work)))
    vals = vals - float(np.mean(vals))
    return grid, vals


def _load_predictor(metric: str):
    """Return (predict_fn, model_tag, method_note)."""
    qpath = qbm_model_path("q50", metric)
    if qpath.is_file():
        booster = _as_booster(joblib.load(qpath))
        return (
            lambda X, b=booster: _predict_qbm(b, X),
            "qbm_q50",
            "ale",
        )
    npath = ngboost_model_path(metric)
    if npath.is_file():
        model: NGBRegressor = joblib.load(npath)
        return (
            lambda X, m=model: _predict_ngb_mu(m, X),
            "ngboost_mu",
            "ale",
        )
    raise FileNotFoundError(f"No QBM q50 or NGBoost model for {metric}")


def _plot_metric_ale(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    metric: str,
    model_tag: str,
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
        x, y = curves[feat]
        color = factor_color(feat) if feat != "node_z" else "0.35"
        ax.plot(x, y, color=color, linewidth=1.0)
        ax.axhline(0.0, color="0.55", linewidth=0.5, linestyle="--")
        ax.set_xlabel(feat.replace("_z", r" $z$") if feat.endswith("_z") else feat)
        ax.set_ylabel(r"ALE on $Y$")
        add_panel_label(ax, i)
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)
    fig.suptitle(
        f"{metric_label(metric, log=True)}  ({model_tag})",
        fontsize=7,
        y=1.01,
    )
    fig.tight_layout(pad=0.35)
    save_figure(fig, f"ale_{metric}", out_dir=out)
    plt.close(fig)


def main() -> None:
    out = out_dir("ale_effects")
    print("Loading data …")
    df = add_design_columns(load_ratios())
    tr, te = load_or_make_split(df)
    rng = np.random.default_rng(ALE_SAMPLE_SEED)
    te = np.asarray(te)
    if len(te) > ALE_SUBSAMPLE_N:
        te = rng.choice(te, size=ALE_SUBSAMPLE_N, replace=False)
    X = df.iloc[te][FEATURES].to_numpy(dtype=float)

    row_tabs = []
    meta = {"subsample_n": int(len(te)), "n_bins": ALE_N_BINS, "models": []}

    for metric in METRICS:
        predict_fn, model_tag, method = _load_predictor(metric)
        print(f"ALE {metric} via {model_tag} …")
        curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for j, feat in enumerate(FEATURES):
            grid, ale = ale_1d(predict_fn, X, j, n_bins=ALE_N_BINS)
            curves[feat] = (grid, ale)
            for x, y in zip(grid, ale):
                row_tabs.append(
                    {
                        "metric": metric,
                        "model": model_tag,
                        "method": method,
                        "feature": feat,
                        "x": float(x),
                        "effect": float(y),
                    }
                )
        _plot_metric_ale(curves, metric=metric, model_tag=model_tag, out=out)
        meta["models"].append({"metric": metric, "model": model_tag, "method": method})

    tab = pd.DataFrame(row_tabs)
    tab.to_csv(out / "ale_curves.csv", index=False)
    (out / "ale_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Effect range summary
    amp = (
        tab.groupby(["metric", "model", "feature"], as_index=False)["effect"]
        .agg(effect_min="min", effect_max="max")
        .assign(effect_range=lambda d: d["effect_max"] - d["effect_min"])
        .sort_values(["metric", "effect_range"], ascending=[True, False])
    )
    amp.to_csv(out / "ale_effect_range.csv", index=False)

    lines = [
        "# ALE / PDP marginal effects (median IM)",
        "",
        "## Definitions",
        "",
        r"- Response: \(Y=\ln\chi\) (conditional median / Normal mean).",
        r"- Primary model: QBM LightGBM at \(\tau=0.50\); fallback: NGBoost \(\mu\).",
        r"- Estimator: 1D accumulated local effects (ALE) on quantile bins of each "
        r"feature, centered to mean zero. If binning fails, centered 1D PDP is used.",
        r"- Features: z-scored design factors + `node_z`. Evaluation on a seed-holdout subsample.",
        "",
        "## Effect amplitude (max − min of ALE curve)",
        "",
        amp.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Conclusions",
        "",
        "- ALE isolates the local marginal effect of each design/spatial feature on the "
        "median (or μ) surface without requiring extrapolated PDP corners.",
        "- Large `node_z` ALE slopes indicate a smooth spatial trend in the emulator; "
        "short-range residual dependence is diagnosed separately.",
        "",
        "## Output files",
        "",
        "| File | Content |",
        "|------|---------|",
        "| `ale_curves.csv` | ALE/PDP curves per metric × feature |",
        "| `ale_effect_range.csv` | amplitude summary |",
        "| `ale_<metric>.pdf` | Nature-width multi-panel ALE plots |",
        "| `ale_meta.json` | subsample / model tags |",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(amp.head(20).to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
