"""Nature SHAP beeswarm for central tendency (Fig15).

One condensed 2×5 figure: rows = QBM τ=0.50 and NGBoost μ, columns = χ
metrics. Design factors have three experimental levels, so points are coloured
Low / Medium / High (node position is tertiled) with a discrete legend instead
of a colour bar — same encoding as the conference beeswarm.

Writes ``shap_beeswarm_central.pdf`` under
``figure_dir("chi_shap", "shap_beeswarm")``. Pass ``--force`` to recompute
TreeSHAP / permutation SHAP instead of loading the npz cache.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from ngboost import NGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FEATURES,
    METRICS,
    add_design_columns,
    load_or_make_split,
    load_ratios,
    make_shap_sample,
    ngboost_model_path,
    out_dir,
    qbm_model_path,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import (  # noqa: E402
    LABEL_FONTSIZE,
    TICK_LABELSIZE,
    add_panel_label,
    apply_full_paper_style,
    figsize,
    metric_label,
    save_figure,
)
from seiskit.plot_config import get_crameri_cmap  # noqa: E402

warnings.filterwarnings("ignore")
apply_full_paper_style(auto_format=True, frame="boxed", grid=False)

BOX_SPINE_LW = 1.0
BOX_SPINE_COLOR = "0.15"
NICE_STEPS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0)

FORCE = "--force" in sys.argv
NGB_EXPLAIN_N = 400
NGB_BG_N = 100
FEATURE_DISPLAY = {
    "Vs1_z": r"$V_{s1}$",
    "Height_z": r"$H$",
    "CoV_z": "CoV",
    "rH_z": r"$r_h$",
    "aHV_z": r"$a_{hv}$",
    "node_z": "node",
}
LEVEL_LABELS = ("Low", "Medium", "High")
ROW_TITLES = (r"QBM $\tau=0.50$", r"NGBoost $\mu$")


def _as_booster(obj) -> lgb.Booster:
    if isinstance(obj, lgb.Booster):
        return obj
    if hasattr(obj, "booster_"):
        return obj.booster_
    return obj


def _qbm_shap(metric: str, X_ex: np.ndarray) -> np.ndarray:
    path = qbm_model_path("q50", metric)
    if not path.is_file():
        raise FileNotFoundError(path)
    booster = _as_booster(joblib.load(path))
    return np.asarray(shap.TreeExplainer(booster).shap_values(X_ex), dtype=float)


def _ngb_mu_shap(metric: str, X_bg: np.ndarray, X_ex: np.ndarray) -> np.ndarray:
    path = ngboost_model_path(metric)
    if not path.is_file():
        raise FileNotFoundError(path)
    model: NGBRegressor = joblib.load(path)

    def predict(X):
        dist = model.pred_dist(np.asarray(X, dtype=float))
        return np.asarray(dist.loc, dtype=float).ravel()

    explainer = shap.Explainer(predict, X_bg, algorithm="permutation")
    explanation = explainer(X_ex, max_evals=2 * X_ex.shape[1] + 1)
    return np.asarray(explanation.values, dtype=float)


def _level_codes(X: np.ndarray) -> np.ndarray:
    """Map each column to {0, 1, 2} = low / medium / high.

    Three unique values (the design-factor levels) are ranked in sort order.
    Continuous columns (``node_z``) are split at empirical tertiles.
    """
    out = np.empty(X.shape, dtype=np.float64)
    for j in range(X.shape[1]):
        col = np.asarray(X[:, j], dtype=float)
        finite = col[np.isfinite(col)]
        uniq = np.unique(finite)
        if uniq.size <= 3:
            mapping = {float(v): float(i) for i, v in enumerate(np.sort(uniq))}
            out[:, j] = np.fromiter((mapping[float(v)] for v in col), dtype=np.float64, count=len(col))
        else:
            q1, q2 = np.quantile(finite, [1.0 / 3.0, 2.0 / 3.0])
            out[:, j] = np.digitize(col, [q1, q2]).astype(np.float64)
    return out


def _beeswarm_offsets(
    x: np.ndarray,
    *,
    rng: np.random.Generator,
    nbins: int = 40,
    max_spread: float = 0.36,
) -> np.ndarray:
    """Density-based vertical jitter (violin of dots, not uniform noise)."""
    n = len(x)
    y = np.zeros(n, dtype=float)
    finite = np.isfinite(x)
    if int(finite.sum()) < 2:
        return (rng.random(n) - 0.5) * 0.12
    xv = x[finite]
    lo, hi = np.quantile(xv, [0.002, 0.998])
    if not np.isfinite(lo) or hi <= lo:
        y[finite] = (rng.random(int(finite.sum())) - 0.5) * 0.12
        return y
    bins = np.linspace(lo, hi, nbins + 1)
    idx = np.clip(np.digitize(np.clip(x, lo, hi), bins) - 1, 0, nbins - 1)
    for b in range(nbins):
        m = finite & (idx == b)
        k = int(m.sum())
        if k <= 1:
            continue
        span = max_spread * min(1.0, 0.20 + 0.80 * (k / 24.0))
        offs = np.linspace(-span, span, k)
        rng.shuffle(offs)
        y[m] = offs
    return y


def _level_cmap() -> tuple[ListedColormap, list]:
    base = get_crameri_cmap("managua", reverse=True)
    colors = [base(0.05), base(0.50), base(0.95)]
    return ListedColormap(colors), colors


def _box_axes(ax: plt.Axes) -> None:
    """Conference-style closed frame: all four spines, dark hairline."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(BOX_SPINE_LW)
        spine.set_color(BOX_SPINE_COLOR)


def _nice_limit(half: float) -> float:
    """Smallest 1–2–2.5–5-style half-range that fits *half* with a little air.

    Keeps 0 centred and lands ticks on round numbers so the beeswarm fills the
    box instead of sitting in a clipped or empty quantile window.
    """
    need = max(float(half), 1e-6)
    exp = np.floor(np.log10(need))
    base = 10.0 ** exp
    for m in NICE_STEPS:
        cand = m * base
        if cand + 1e-12 >= need:
            return float(cand)
    return float(10.0 * base)


def _beeswarm_panel(
    ax: plt.Axes,
    shap_values: np.ndarray,
    levels: np.ndarray,
    *,
    order: np.ndarray,
    cmap: ListedColormap,
    x_lim: tuple[float, float],
    rng: np.random.Generator,
) -> None:
    n_feat = len(order)
    for row, j in enumerate(order):
        vals = shap_values[:, j]
        y = row + _beeswarm_offsets(vals, rng=rng)
        ax.scatter(
            vals,
            y,
            c=levels[:, j],
            cmap=cmap,
            vmin=0.0,
            vmax=2.0,
            s=3.0,
            alpha=0.55,
            linewidths=0,
            rasterized=True,
            zorder=0,
            clip_on=True,
        )
    ax.axvline(0.0, color="0.55", lw=0.5, ls="--", zorder=1)
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels([FEATURE_DISPLAY.get(FEATURES[j], FEATURES[j]) for j in order])
    ax.set_ylim(-0.55, n_feat - 0.45)
    lo, hi = x_lim
    ax.set_xlim(lo, hi)
    ticks = [lo, 0.0, hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.tick_params(labelsize=TICK_LABELSIZE, length=2.0, width=0.6, direction="out")
    ax.tick_params("y", length=0)
    ax.tick_params(top=False, right=False)
    ax.set_rasterization_zorder(1)
    _box_axes(ax)


def _column_xlims(qbm: dict[str, np.ndarray], ngb: dict[str, np.ndarray]) -> dict[str, tuple[float, float]]:
    """Symmetric per-metric limits, shared by QBM and NGBoost in that column."""
    x_lims: dict[str, tuple[float, float]] = {}
    for metric in METRICS:
        half = max(
            float(np.max(np.abs(qbm[metric]))),
            float(np.max(np.abs(ngb[metric]))),
        )
        L = _nice_limit(half)
        x_lims[metric] = (-L, L)
    return x_lims


def _feature_order(qbm: dict[str, np.ndarray], ngb: dict[str, np.ndarray]) -> np.ndarray:
    """Shared y-order: most important overall at the top of every panel."""
    acc = np.zeros(len(FEATURES), dtype=float)
    n = 0
    for store in (qbm, ngb):
        for metric in METRICS:
            acc += np.mean(np.abs(store[metric]), axis=0)
            n += 1
    return np.argsort(acc / max(n, 1))


def plot_central_beeswarm(
    qbm: dict[str, np.ndarray],
    ngb: dict[str, np.ndarray],
    X_qbm: np.ndarray,
    X_ngb: np.ndarray,
    *,
    out: Path,
) -> None:
    cmap, level_colors = _level_cmap()
    levels_q = _level_codes(X_qbm)
    levels_n = _level_codes(X_ngb)
    order = _feature_order(qbm, ngb)
    x_lims = _column_xlims(qbm, ngb)
    print("  xlims:", {m: (round(a, 4), round(b, 4)) for m, (a, b) in x_lims.items()})
    rng = np.random.default_rng(0)

    n_metrics = len(METRICS)
    fig = plt.figure(figsize=figsize(height=5.15))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.09, 1.0],
        hspace=0.04,
        top=0.99,
        bottom=0.08,
        left=0.12,
        right=0.985,
    )
    ax_leg = fig.add_subplot(gs[0])
    ax_leg.set_axis_off()
    gs_plots = gs[1].subgridspec(2, n_metrics, hspace=0.28, wspace=0.22)
    axes = np.array(
        [[fig.add_subplot(gs_plots[r, c]) for c in range(n_metrics)] for r in range(2)]
    )

    stores = ((qbm, levels_q), (ngb, levels_n))
    for row, ((store, levels), row_title) in enumerate(zip(stores, ROW_TITLES)):
        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            _beeswarm_panel(
                ax,
                store[metric],
                levels,
                order=order,
                cmap=cmap,
                x_lim=x_lims[metric],
                rng=rng,
            )
            add_panel_label(ax, row * n_metrics + col, x=0.97, y=0.97, alpha=0.75)
            if row == 0:
                ax.set_title(metric_label(metric, log=True), fontsize=LABEL_FONTSIZE, pad=3)
                ax.tick_params(labelbottom=False)
                ax.set_xlabel("")
            else:
                ax.set_xlabel("SHAP value" if col == n_metrics // 2 else "", fontsize=LABEL_FONTSIZE, labelpad=2)
            if col == 0:
                ax.set_ylabel(row_title, fontsize=LABEL_FONTSIZE, labelpad=2)
            else:
                ax.tick_params(axis="y", left=False, labelleft=False, length=0)
                ax.set_ylabel("")
            _box_axes(ax)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=c,
            markeredgecolor="none",
            markersize=6,
            label=lab,
        )
        for c, lab in zip(level_colors, LEVEL_LABELS)
    ]
    leg = ax_leg.legend(
        handles,
        list(LEVEL_LABELS),
        loc="center",
        ncol=3,
        fontsize=LABEL_FONTSIZE,
        frameon=False,
        columnspacing=0.8,
        handletextpad=0.25,
        borderpad=0.0,
        labelspacing=0.0,
        handlelength=0.9,
        title="Feature value",
        title_fontsize=LABEL_FONTSIZE,
    )
    if hasattr(leg, "_legend_box") and leg._legend_box is not None:
        leg._legend_box.sep = 0.15

    save_figure(fig, "shap_beeswarm_central", out_dir=out)
    plt.close(fig)


def _cache_path(out: Path) -> Path:
    return out / "shap_values_cache.npz"


def _load_cache(path: Path) -> dict[str, np.ndarray] | None:
    if FORCE or not path.is_file():
        return None
    data = np.load(path, allow_pickle=False)
    needed = [f"qbm_{m}" for m in METRICS] + [f"ngb_{m}" for m in METRICS] + ["X_qbm", "X_ngb"]
    if any(k not in data.files for k in needed):
        return None
    return {k: np.asarray(data[k]) for k in needed}


def _compute_shap(
    df: pd.DataFrame,
    bg_idx: np.ndarray,
    ex_idx: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    X_ex = df.iloc[ex_idx][FEATURES].to_numpy(dtype=float)

    rng = np.random.default_rng(3)
    if len(ex_idx) > NGB_EXPLAIN_N:
        ngb_ex = rng.choice(ex_idx, size=NGB_EXPLAIN_N, replace=False)
    else:
        ngb_ex = ex_idx
    if len(bg_idx) > NGB_BG_N:
        ngb_bg = rng.choice(bg_idx, size=NGB_BG_N, replace=False)
    else:
        ngb_bg = bg_idx
    X_bg_n = df.iloc[ngb_bg][FEATURES].to_numpy(dtype=float)
    X_ex_n = df.iloc[ngb_ex][FEATURES].to_numpy(dtype=float)

    qbm: dict[str, np.ndarray] = {}
    ngb: dict[str, np.ndarray] = {}
    for metric in METRICS:
        print(f"Beeswarm {metric}: QBM q50 …")
        qbm[metric] = _qbm_shap(metric, X_ex)
        print(f"Beeswarm {metric}: NGBoost μ …")
        ngb[metric] = _ngb_mu_shap(metric, X_bg_n, X_ex_n)
    return qbm, ngb, X_ex, X_ex_n


def _drop_per_metric_pdfs(out: Path) -> None:
    for metric in METRICS:
        for ext in ("pdf", "png"):
            path = out / f"shap_beeswarm_{metric}.{ext}"
            if path.is_file():
                path.unlink()


def main() -> None:
    out = out_dir("shap_beeswarm")
    print("Loading data …")
    df = add_design_columns(load_ratios())
    _, te = load_or_make_split(df)
    bg_idx, ex_idx, sample_meta = make_shap_sample(df, te)

    cached = _load_cache(_cache_path(out))
    if cached is not None:
        print(f"Loaded SHAP cache {_cache_path(out)}")
        qbm = {m: cached[f"qbm_{m}"] for m in METRICS}
        ngb = {m: cached[f"ngb_{m}"] for m in METRICS}
        X_qbm, X_ngb = cached["X_qbm"], cached["X_ngb"]
    else:
        qbm, ngb, X_qbm, X_ngb = _compute_shap(df, bg_idx, ex_idx)
        np.savez_compressed(
            _cache_path(out),
            X_qbm=X_qbm,
            X_ngb=X_ngb,
            **{f"qbm_{m}": qbm[m] for m in METRICS},
            **{f"ngb_{m}": ngb[m] for m in METRICS},
        )
        print(f"Wrote {_cache_path(out)}")

    plot_central_beeswarm(qbm, ngb, X_qbm, X_ngb, out=out)
    _drop_per_metric_pdfs(out)

    meta = {
        "sample": sample_meta,
        "ngboost_explain_n": int(X_ngb.shape[0]),
        "qbm_explain_n": int(X_qbm.shape[0]),
        "metrics": list(METRICS),
        "feature_levels": "low/medium/high (design factors: 3 experimental levels; node_z: tertiles)",
        "figure": "shap_beeswarm_central.pdf",
        "layout": "2 rows (QBM τ=0.50, NGBoost μ) × 5 metrics",
        "xlims": {m: list(_column_xlims(qbm, ngb)[m]) for m in METRICS},
        "forced": FORCE,
    }
    (out / "shap_beeswarm_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    lines = [
        "# SHAP beeswarm summaries (Fig15)",
        "",
        "Single condensed Nature beeswarm for **central** tendency, all χ metrics:",
        r"- Row 1: QBM LightGBM TreeSHAP at $\tau=0.50$",
        r"- Row 2: NGBoost permutation SHAP on predictive mean $\mu$",
        r"- Columns: $f_0^N$, $|TF|_0^N$, $\mathrm{PGA}^N$, $\mathrm{SA}^N$, $I_a^N$",
        "",
        "Feature values are encoded as **Low / Medium / High** (no colour bar):",
        "- Five design factors: the three experimental levels, ranked by value.",
        "- `node_z`: empirical tertiles of array position.",
        "",
        "Y-order is shared (mean $|$SHAP$|$ averaged over models and metrics).",
        "X-limits are symmetric about 0, shared within each metric column, and snapped",
        "to round tick values so clouds fill the box without clipping extrema.",
        "Panels use a closed (boxed) frame with extra spacing, as in the conference beeswarm.",
        "",
        "## Output files",
        "",
        "| File | Content |",
        "|------|---------|",
        "| `shap_beeswarm_central.pdf` | 2×5 condensed beeswarm |",
        "| `shap_values_cache.npz` | TreeSHAP / permutation SHAP arrays |",
        "| `shap_beeswarm_meta.json` | subsample sizes |",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
