"""Nature SHAP beeswarm per NGBoost partition (Fig15).

One 2×5 figure per partition (between-seed, within-seed): rows = NGBoost μ and
NGBoost log σ, columns = χ metrics. Design factors have three experimental
levels, so points are coloured Low / Medium / High with a discrete legend
instead of a colour bar — same encoding as the conference beeswarm.

Writes ``shap_beeswarm.pdf`` under
``figure_dir("chi_shap", "shap_beeswarm", <partition>)``. Pass ``--force`` to
recompute permutation SHAP instead of loading the npz cache.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FEATURE_DISPLAY,
    METRICS,
    NGB_FEATURES,
    PARTITION_LABELS,
    PARTITIONS,
    load_partition,
    out_dir,
    partition_shap_sample,
)
from shap_ngboost import _LogSigmaModel, _MuModel, load_model  # noqa: E402

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
EXPLAIN_N = 400
BG_N = 100
LEVEL_LABELS = ("Low", "Medium", "High")
# (cache key, row title)
ROWS = (("mu", r"NGBoost $\mu$"), ("log_sigma", r"NGBoost $\log\sigma$"))


def _shap_values(predict_fn, X_bg: np.ndarray, X_ex: np.ndarray) -> np.ndarray:
    explainer = shap.Explainer(predict_fn, X_bg, algorithm="permutation")
    explanation = explainer(X_ex, max_evals=2 * X_ex.shape[1] + 1)
    return np.asarray(explanation.values, dtype=float)


def _level_codes(X: np.ndarray) -> np.ndarray:
    """Map each three-level column to {0, 1, 2} = low / medium / high."""
    out = np.empty(X.shape, dtype=np.float64)
    for j in range(X.shape[1]):
        col = np.asarray(X[:, j], dtype=float)
        uniq = np.sort(np.unique(col[np.isfinite(col)]))
        mapping = {float(v): float(i) for i, v in enumerate(uniq)}
        out[:, j] = np.fromiter((mapping[float(v)] for v in col), dtype=np.float64, count=len(col))
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
    """Smallest 1–2–2.5–5-style half-range that fits *half* with a little air."""
    need = max(float(half), 1e-6)
    base = 10.0 ** np.floor(np.log10(need))
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
        ax.scatter(
            vals,
            row + _beeswarm_offsets(vals, rng=rng),
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
    ax.set_yticklabels([FEATURE_DISPLAY.get(NGB_FEATURES[j], NGB_FEATURES[j]) for j in order])
    ax.set_ylim(-0.55, n_feat - 0.45)
    lo, hi = x_lim
    ax.set_xlim(lo, hi)
    ax.set_xticks([lo, 0.0, hi])
    ax.set_xticklabels([f"{t:g}" for t in (lo, 0.0, hi)])
    ax.tick_params(labelsize=TICK_LABELSIZE, length=2.0, width=0.6, direction="out")
    ax.tick_params("y", length=0)
    ax.tick_params(top=False, right=False)
    ax.set_rasterization_zorder(1)
    _box_axes(ax)


def _panel_xlims(
    store: dict[str, dict[str, np.ndarray]], rows=ROWS
) -> dict[tuple[str, str], tuple]:
    """Symmetric limits per (row target, metric); μ and log σ scales differ."""
    out = {}
    for key, _ in rows:
        for metric in METRICS:
            L = _nice_limit(float(np.max(np.abs(store[key][metric]))))
            out[(key, metric)] = (-L, L)
    return out


def _feature_order(store: dict[str, dict[str, np.ndarray]], rows=ROWS) -> np.ndarray:
    """Shared y-order: most important overall at the top of every panel."""
    acc = np.zeros(len(NGB_FEATURES), dtype=float)
    n = 0
    for key, _ in rows:
        for metric in METRICS:
            per = np.mean(np.abs(store[key][metric]), axis=0)
            acc += per / (float(per.sum()) or 1.0)
            n += 1
    return np.argsort(acc / max(n, 1))


def plot_beeswarm(
    store: dict[str, dict[str, np.ndarray]],
    X: np.ndarray | dict[str, np.ndarray],
    *,
    out: Path,
    partition: str | None = None,
    rows=ROWS,
    legend_title: str | None = None,
    stem: str = "shap_beeswarm",
) -> dict:
    """2 × metrics beeswarm; *X* may be one array or one array per row key."""
    cmap, level_colors = _level_cmap()
    levels = (
        {k: _level_codes(v) for k, v in X.items()}
        if isinstance(X, dict)
        else {k: _level_codes(X) for k, _ in rows}
    )
    order = _feature_order(store, rows)
    x_lims = _panel_xlims(store, rows)
    if legend_title is None:
        legend_title = f"{PARTITION_LABELS[partition]} — feature value"
    rng = np.random.default_rng(0)

    n_metrics = len(METRICS)
    fig = plt.figure(figsize=figsize(height=5.15))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[0.11, 1.0], hspace=0.04, top=0.99, bottom=0.08, left=0.12, right=0.985
    )
    ax_leg = fig.add_subplot(gs[0])
    ax_leg.set_axis_off()
    gs_plots = gs[1].subgridspec(2, n_metrics, hspace=0.28, wspace=0.22)
    axes = np.array([[fig.add_subplot(gs_plots[r, c]) for c in range(n_metrics)] for r in range(2)])

    for row, (key, row_title) in enumerate(rows):
        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            _beeswarm_panel(
                ax,
                store[key][metric],
                levels[key],
                order=order,
                cmap=cmap,
                x_lim=x_lims[(key, metric)],
                rng=rng,
            )
            add_panel_label(ax, row * n_metrics + col, x=0.97, y=0.97, alpha=0.75)
            if row == 0:
                ax.set_title(metric_label(metric, log=True), fontsize=LABEL_FONTSIZE, pad=3)
            ax.set_xlabel(
                "SHAP value" if (row == 1 and col == n_metrics // 2) else "",
                fontsize=LABEL_FONTSIZE,
                labelpad=2,
            )
            if col == 0:
                ax.set_ylabel(row_title, fontsize=LABEL_FONTSIZE, labelpad=2)
            else:
                ax.tick_params(axis="y", left=False, labelleft=False, length=0)

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
    ax_leg.legend(
        handles,
        list(LEVEL_LABELS),
        loc="center",
        ncol=3,
        fontsize=LABEL_FONTSIZE,
        frameon=False,
        columnspacing=0.8,
        handletextpad=0.25,
        borderpad=0.0,
        handlelength=0.9,
        title=legend_title,
        title_fontsize=LABEL_FONTSIZE,
    )
    save_figure(fig, stem, out_dir=out)
    plt.close(fig)
    return {f"{k}|{m}": list(v) for (k, m), v in x_lims.items()}


def _compute(df, partition: str) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray, dict]:
    bg_idx, ex_idx, sample_meta = partition_shap_sample(
        df, partition, explain_n=EXPLAIN_N, bg_n=BG_N
    )
    X_bg = df.iloc[bg_idx][NGB_FEATURES].to_numpy(dtype=float)
    X_ex = df.iloc[ex_idx][NGB_FEATURES].to_numpy(dtype=float)
    store: dict[str, dict[str, np.ndarray]] = {"mu": {}, "log_sigma": {}}
    for metric in METRICS:
        model = load_model(partition, metric)
        print(f"Beeswarm [{partition}] {metric}: μ …")
        store["mu"][metric] = _shap_values(_MuModel(model).predict, X_bg, X_ex)
        print(f"Beeswarm [{partition}] {metric}: log σ …")
        store["log_sigma"][metric] = _shap_values(_LogSigmaModel(model).predict, X_bg, X_ex)
    return store, X_ex, sample_meta


def run_partition(partition: str) -> None:
    out = out_dir("shap_beeswarm", partition)
    cache = out / "shap_values_cache.npz"
    keys = [f"{k}_{m}" for k, _ in ROWS for m in METRICS]
    store = None
    sample_meta: dict = {}
    if not FORCE and cache.is_file():
        data = np.load(cache, allow_pickle=False)
        if all(k in data.files for k in [*keys, "X"]):
            print(f"Loaded SHAP cache {cache}")
            store = {k: {m: np.asarray(data[f"{k}_{m}"]) for m in METRICS} for k, _ in ROWS}
            X = np.asarray(data["X"])
    if store is None:
        print(f"Loading {PARTITION_LABELS[partition]} …")
        df = load_partition(partition)
        store, X, sample_meta = _compute(df, partition)
        np.savez_compressed(
            cache, X=X, **{f"{k}_{m}": store[k][m] for k, _ in ROWS for m in METRICS}
        )
        print(f"Wrote {cache}")

    xlims = plot_beeswarm(store, X, out=out, partition=partition)
    meta = {
        "partition": partition,
        "sample": sample_meta,
        "explain_n": int(X.shape[0]),
        "metrics": list(METRICS),
        "feature_levels": "low/medium/high (three experimental levels)",
        "figure": "shap_beeswarm.pdf",
        "layout": "2 rows (NGBoost μ, NGBoost log σ) × 5 metrics",
        "xlims": xlims,
        "forced": FORCE,
    }
    (out / "shap_beeswarm_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    lines = [
        f"# SHAP beeswarm — {PARTITION_LABELS[partition]} (Fig15)",
        "",
        r"- Row 1: NGBoost permutation SHAP on predictive mean $\mu$",
        r"- Row 2: NGBoost permutation SHAP on $\log\sigma$ "
        + ("(between-seed dispersion)" if partition == "between" else "(within-seed dispersion)"),
        r"- Columns: $f_0^N$, $|TF|_0^N$, $\mathrm{PGA}^N$, $\mathrm{SA}^N$, $I_a^N$",
        "- Feature values: Low / Medium / High = the three experimental levels.",
        "- Y-order shared across panels; x-limits symmetric per panel (μ and log σ differ in scale).",
        "",
        "| File | Content |",
        "|------|---------|",
        "| `shap_beeswarm.pdf` | 2×5 beeswarm |",
        "| `shap_values_cache.npz` | permutation SHAP arrays |",
        "| `shap_beeswarm_meta.json` | subsample sizes, limits |",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


def main() -> None:
    for partition in PARTITIONS:
        run_partition(partition)


if __name__ == "__main__":
    main()
