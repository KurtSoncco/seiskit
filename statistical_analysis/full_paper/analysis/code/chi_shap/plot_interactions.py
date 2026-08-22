"""Friedman H ranking + TreeSHAP / proxy interaction heatmaps (Fig19).

One Nature 5×3 figure: rows = χ metrics, columns = Friedman H bars,
QBM TreeSHAP heatmap, NGBoost product-proxy heatmap. All data axes share
the same physical height; heatmap columns share feature order and limits;
Friedman bars share pair order and H limits.

Reads existing CSVs from exceedance_friedman, shap_qbm, shap_ngboost and
writes under ``figure_dir("chi_shap", "interactions")``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator, ScalarFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURES, METRICS, out_dir  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import (  # noqa: E402
    BOX_ROOT,
    LABEL_FONTSIZE,
    TICK_LABELSIZE,
    add_panel_label,
    apply_full_paper_style,
    figsize,
    metric_color,
    metric_label,
    save_figure,
)
from seiskit.plot_config import get_crameri_cmap  # noqa: E402

apply_full_paper_style(auto_format=True, frame="boxed", grid=False)

FRIEDMAN = (
    BOX_ROOT
    / "full_paper"
    / "figures"
    / "chi_ngboost"
    / "exceedance_friedman"
    / "friedman_H_pairs.csv"
)
QBM_INT = (
    BOX_ROOT / "full_paper" / "figures" / "chi_shap" / "shap_qbm" / "shap_interactions_top.csv"
)
NGB_INT = (
    BOX_ROOT / "full_paper" / "figures" / "chi_shap" / "shap_ngboost" / "shap_interactions_top.csv"
)

DISPLAY = {
    "Vs1_z": r"$V_{s1}$",
    "Height_z": r"$H$",
    "CoV_z": "CoV",
    "rH_z": r"$r_h$",
    "aHV_z": r"$a_{hv}$",
    "node_z": "node",
}

COL_TITLES = (
    r"Friedman $H$",
    r"QBM TreeSHAP $|\phi_{jk}|$",
    r"NGBoost proxy $|\phi_i\phi_j|$",
)
def _labels() -> list[str]:
    return [DISPLAY.get(f, f) for f in FEATURES]


def _canon_pair(a: str, b: str) -> tuple[str, str]:
    ia, ib = FEATURES.index(a), FEATURES.index(b)
    return (a, b) if ia < ib else (b, a)


def _pair_label(a: str, b: str) -> str:
    return f"{DISPLAY.get(a, a)} × {DISPLAY.get(b, b)}"


def _shared_pairs(friedman: pd.DataFrame) -> list[tuple[str, str]]:
    """Unordered pairs, weakest at the bottom so the strongest sit at the top."""
    keys = [_canon_pair(a, b) for a, b in zip(friedman["feature_i"], friedman["feature_j"])]
    tmp = friedman.assign(a=[k[0] for k in keys], b=[k[1] for k in keys])
    mean_h = tmp.groupby(["a", "b"], sort=False)["H"].mean().sort_values(ascending=True)
    return [(a, b) for a, b in mean_h.index]


def _h_aligned(friedman: pd.DataFrame, metric: str, pairs: list[tuple[str, str]]) -> np.ndarray:
    sub = friedman[friedman["metric"] == metric]
    lookup: dict[tuple[str, str], float] = {}
    for _, row in sub.iterrows():
        lookup[_canon_pair(row["feature_i"], row["feature_j"])] = float(row["H"])
    return np.array([lookup.get(p, np.nan) for p in pairs], dtype=float)


def _select_pairs(tab: pd.DataFrame, metric: str, *, preferred: str, fallback: str | None) -> pd.DataFrame:
    q = tab[(tab["metric"] == metric) & (tab["target"] == preferred)]
    if q.empty and fallback is not None:
        q = tab[(tab["metric"] == metric) & (tab["target"] == fallback)]
    return q


def _matrix_from_pairs(
    pairs: pd.DataFrame,
    *,
    value_col: str,
    feature_i: str = "feature_i",
    feature_j: str = "feature_j",
) -> np.ndarray:
    p = len(FEATURES)
    idx = {f: i for i, f in enumerate(FEATURES)}
    mat = np.full((p, p), np.nan, dtype=float)
    for _, row in pairs.iterrows():
        i = idx.get(row[feature_i])
        j = idx.get(row[feature_j])
        if i is None or j is None:
            continue
        v = float(row[value_col])
        mat[i, j] = v
        mat[j, i] = v
    np.fill_diagonal(mat, np.nan)
    return mat


def _style_heatmap(ax: plt.Axes, n: int, *, show_x: bool, show_y: bool) -> None:
    labels = _labels()
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("auto")
    if show_x:
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=TICK_LABELSIZE)
    else:
        ax.set_xticklabels([])
    if show_y:
        ax.set_yticklabels(labels, fontsize=TICK_LABELSIZE)
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=TICK_LABELSIZE, length=2.0, width=0.6, pad=1.2)
    ax.tick_params(top=False, right=False)


_LEFT = 0.125
_RIGHT = 0.935
_CBAR_WIDTH = 0.012
_CBAR_PAD = 0.008
_GUTTER = 0.08


def _layout_panels(fig: plt.Figure, axes: np.ndarray) -> None:
    """Equal y-axis length, square heatmaps, equal gaps between column stacks."""
    fig.canvas.draw()
    fw, fh = fig.get_size_inches()
    height = min(ax.get_position().height for ax in axes.ravel())
    sq_w = height * fh / fw
    y0s = [
        axes[r, 0].get_position().y0
        + 0.5 * (axes[r, 0].get_position().height - height)
        for r in range(axes.shape[0])
    ]
    heat_stack = sq_w + _CBAR_PAD + _CBAR_WIDTH
    span = _RIGHT - _LEFT
    gutter = _GUTTER
    bar_w = span - 2.0 * gutter - 2.0 * heat_stack
    min_bar = 0.90 * sq_w
    if bar_w < min_bar:
        bar_w = min_bar
        gutter = (span - bar_w - 2.0 * heat_stack) / 2.0

    x_bar = _LEFT
    x_qbm = x_bar + bar_w + gutter
    x_ngb = x_qbm + heat_stack + gutter
    xs = (x_bar, x_qbm, x_ngb)
    widths = (bar_w, sq_w, sq_w)
    for r in range(axes.shape[0]):
        for c in range(3):
            axes[r, c].set_position([xs[c], y0s[r], widths[c], height])


def _add_cbar(fig: plt.Figure, ax: plt.Axes, im) -> None:
    bb = ax.get_position()
    cax = fig.add_axes([bb.x1 + _CBAR_PAD, bb.y0, _CBAR_WIDTH, bb.height])
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-2, 2))
    cbar.ax.yaxis.set_major_formatter(fmt)
    off = cbar.ax.yaxis.get_offset_text()
    off.set_fontsize(TICK_LABELSIZE - 0.5)
    off.set_horizontalalignment("left")
    off.set_x(0.5)
    cbar.ax.tick_params(labelsize=TICK_LABELSIZE - 0.5, pad=1.0, length=2.0, width=0.5)
    cbar.set_label("")


def plot_interactions_grid(
    friedman: pd.DataFrame,
    qbm: pd.DataFrame,
    ngb: pd.DataFrame,
    *,
    out: Path,
) -> None:
    pairs = _shared_pairs(friedman)
    n_pairs = len(pairs)
    n_feat = len(FEATURES)
    pair_labels = [_pair_label(a, b) for a, b in pairs]
    h_lim = 0.5
    n_metrics = len(METRICS)
    last = n_metrics - 1

    cmaps = []
    for name, reverse in (("lajolla", True), ("oslo", True)):
        cmap = get_crameri_cmap(name, reverse=reverse).copy()
        cmap.set_bad("#f4f4f4")
        cmaps.append(cmap)

    fig = plt.figure(figsize=figsize(height=6.65))
    gs = fig.add_gridspec(
        n_metrics,
        3,
        wspace=0.30,
        hspace=0.48,
        left=_LEFT,
        right=_RIGHT,
        top=0.955,
        bottom=0.05,
    )
    axes = np.empty((n_metrics, 3), dtype=object)
    for r in range(n_metrics):
        for c in range(3):
            axes[r, c] = fig.add_subplot(
                gs[r, c],
                sharex=axes[0, c] if r else None,
                sharey=axes[0, c] if r else None,
            )

    images: list[tuple[plt.Axes, object]] = []
    for r, metric in enumerate(METRICS):
        # Column 0: Friedman H, shared pair order and xlim
        ax = axes[r, 0]
        hvals = _h_aligned(friedman, metric, pairs)
        y = np.arange(n_pairs, dtype=float)
        finite = np.isfinite(hvals)
        ax.barh(
            y[finite],
            hvals[finite],
            color=metric_color(metric),
            height=0.68,
            edgecolor="none",
        )
        ax.set_yticks(y)
        ax.set_yticklabels(pair_labels, fontsize=TICK_LABELSIZE)
        ax.set_ylim(-0.55, n_pairs - 0.45)
        ax.set_xlim(0.0, h_lim)
        ax.set_xticks([0.0, 0.25, 0.5])
        ax.tick_params(labelsize=TICK_LABELSIZE, length=2.0, width=0.6, pad=1.2)
        ax.tick_params(top=False, right=False)
        ax.set_ylabel(metric_label(metric, log=True), fontsize=LABEL_FONTSIZE, labelpad=3)
        if r == 0:
            ax.set_title(COL_TITLES[0], fontsize=LABEL_FONTSIZE, pad=8)
        if r == last:
            ax.set_xlabel(r"Friedman $H$", fontsize=LABEL_FONTSIZE, labelpad=2)
        else:
            ax.tick_params(labelbottom=False)
            ax.set_xlabel("")

        q = _select_pairs(qbm, metric, preferred="q50", fallback="mean")
        n = _select_pairs(ngb, metric, preferred="mu", fallback=None)
        mats = (
            _matrix_from_pairs(q, value_col="mean_abs_interaction"),
            _matrix_from_pairs(n, value_col="mean_abs_interaction"),
        )
        for c, (mat, cmap) in enumerate(zip(mats, cmaps), start=1):
            ax = axes[r, c]
            finite = mat[np.isfinite(mat)]
            vmax = float(np.max(finite)) if finite.size else 1.0
            if vmax <= 0:
                vmax = 1.0
            im = ax.imshow(
                np.ma.masked_invalid(mat),
                cmap=cmap,
                norm=Normalize(vmin=0.0, vmax=vmax),
                aspect="auto",
                interpolation="nearest",
                origin="upper",
            )
            _style_heatmap(ax, n_feat, show_x=(r == last), show_y=True)
            if r == 0:
                ax.set_title(COL_TITLES[c], fontsize=LABEL_FONTSIZE, pad=8)
            images.append((ax, im))

        add_panel_label(axes[r, 0], r * 3, x=0.97, y=0.96, alpha=0.75)
        add_panel_label(axes[r, 1], r * 3 + 1, x=0.97, y=0.96, alpha=0.75)
        add_panel_label(axes[r, 2], r * 3 + 2, x=0.97, y=0.96, alpha=0.75)

    _layout_panels(fig, axes)
    for ax, im in images:
        _add_cbar(fig, ax, im)

    save_figure(fig, "interactions", out_dir=out)
    plt.close(fig)


def main() -> None:
    out = out_dir("interactions")
    friedman = pd.read_csv(FRIEDMAN)
    qbm = pd.read_csv(QBM_INT)
    ngb = pd.read_csv(NGB_INT)

    print("Interactions 5×3 grid …")
    plot_interactions_grid(friedman, qbm, ngb, out=out)

    for stale in out.glob("interactions_*.pdf"):
        stale.unlink()
        print(f"Removed {stale.name}")

    top = (
        friedman.sort_values(["metric", "H"], ascending=[True, False])
        .groupby("metric", as_index=False)
        .head(5)
    )
    top.to_csv(out / "friedman_H_top5.csv", index=False)

    lines = [
        "# Parameter interactions (Fig19)",
        "",
        "Single 5×3 figure (`interactions.pdf`):",
        "- Rows: χ metrics (same order as Fig15).",
        r"- Columns: Friedman $H$ bars, QBM TreeSHAP $|\phi_{jk}|$, NGBoost product-proxy.",
        "- Friedman bars share pair order (union of computed pairs, ranked by mean $H$) and $H$ limits.",
        "- Heatmaps share feature order and axis limits; colour scales are per panel (magnitudes are not comparable across metrics).",
        "- All data axes are forced to the same physical height.",
        "",
        "## Top Friedman pairs",
        "",
        top.to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
