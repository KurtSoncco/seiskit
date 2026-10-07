r"""Parallel-slope variance figure across design factors.

Reads ``cell_summary.csv`` and writes ``parallel_slopes`` under
``figure_dir("chi_variables", "variance_factor_effects")``: a metric × factor
grid of \(\overline{s}_W\), \(\sigma_\mu\), \(\sigma_{\mathrm{total}}\)
(rms of log-χ; CSV columns stay ``s_*``).

Medians are taken over the orthogonal design factors at each level.

Usage::

    python variance_factor_effects.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import (  # noqa: E402
    DATA_LINEWIDTH,
    FACTORS,
    LABEL_FONTSIZE,
    TICK_LABELSIZE,
    add_panel_label,
    apply_full_paper_style,
    figsize,
    figure_dir,
    metric_color,
    metric_label,
    save_figure,
)

apply_full_paper_style(auto_format=True, frame="open", grid=False)

METRICS = ("f_ratio", "abs_TF_ratio", "PGA_ratio", "PSA_ratio", "Ia_ratio")

FACTOR_XLABELS = {
    "Vs1": r"$V_{s1}$ (m/s)",
    "Height": r"$H$ (m)",
    "CoV": "CoV",
    "rH": r"$r_h$ (m)",
    "aHV": r"$a_{hv}$",
}

# Bars stay s (̄s_W, ̄s_B). Former σ² terms display as σ (rms), not s.
COMP_STYLES = {
    "s_W_bar": {"ls": "-", "marker": "o", "label": r"$\overline{s}_W$"},
    "s_mu": {"ls": ":", "marker": "s", "label": r"$\sigma_\mu$"},
    "s_total": {"ls": "--", "marker": "D", "label": r"$\sigma_{\mathrm{total}}$"},
}
LINE_KW = {"markersize": 3.5, "lw": DATA_LINEWIDTH, "alpha": 0.95}

GRID_ALPHA = 0.18
LEGEND_FRAME = {
    "frameon": True,
    "fancybox": False,
    "framealpha": 0.75,
    "facecolor": "white",
    "edgecolor": "none",
    "borderpad": 0.25,
}


def load_cell_summary() -> pd.DataFrame:
    path = figure_dir("chi_variables", "central_variability") / "cell_summary.csv"
    df = pd.read_csv(path)
    if "metric" not in df.columns:
        raise ValueError(f"Expected 'metric' column in {path}")
    return df


def _format_level(value: float) -> str:
    return f"{int(value)}" if float(value).is_integer() else f"{value:g}"


def plot_parallel_slopes(df: pd.DataFrame, out_dir: Path) -> None:
    nrows, ncols = len(METRICS), len(FACTORS)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize(aspect=0.92), sharex="col", sharey="row")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.12, top=0.90, wspace=0.12, hspace=0.18)

    for r, metric in enumerate(METRICS):
        sub = df[df["metric"] == metric]
        color = metric_color(metric)
        row_vals: list[float] = []
        for c, factor in enumerate(FACTORS):
            ax = axes[r, c]
            grouped = sub.groupby(sub[factor].astype(float))[list(COMP_STYLES)].median()
            levels = grouped.index.to_numpy()
            for col, style in COMP_STYLES.items():
                meds = grouped[col].to_numpy()
                ax.plot(
                    levels,
                    meds,
                    color=color,
                    ls=style["ls"],
                    marker=style["marker"],
                    zorder=5,
                    **LINE_KW,
                )
                row_vals.extend(meds[np.isfinite(meds)].tolist())

            # Natural (linear) scale at true factor levels — not categorical.
            span = float(levels[-1] - levels[0]) if len(levels) > 1 else max(float(levels[0]), 1.0)
            ax.set_xlim(levels[0] - 0.12 * span, levels[-1] + 0.12 * span)
            ax.set_xticks(levels)
            ax.set_xticklabels([_format_level(v) for v in levels])
            ax.grid(True, which="major", axis="y", alpha=GRID_ALPHA, lw=0.6)
            ax.set_axisbelow(True)
            add_panel_label(ax, r * ncols + c, alpha=0.75)

            if c == 0:
                ax.set_ylabel(metric_label(metric, log=True), fontsize=LABEL_FONTSIZE)
            else:
                ax.tick_params(labelleft=False)
            if r == nrows - 1:
                ax.set_xlabel(FACTOR_XLABELS[factor], fontsize=LABEL_FONTSIZE)
            else:
                ax.tick_params(labelbottom=False)

        if row_vals:
            lo, hi = min(row_vals), max(row_vals)
            pad = 0.12 * max(hi - lo, 1e-3)
            axes[r, 0].set_ylim(max(0.0, lo - pad), hi + pad)  # sharey="row"

    handles = [
        Line2D([0], [0], color="0.25", ls=s["ls"], marker=s["marker"], label=s["label"], **LINE_KW)
        for s in COMP_STYLES.values()
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        fontsize=TICK_LABELSIZE,
        bbox_to_anchor=(0.5, 0.995),
        handlelength=4,
        **LEGEND_FRAME,
    )
    save_figure(fig, "parallel_slopes", out_dir=out_dir)
    plt.close(fig)


def main() -> None:
    df = load_cell_summary()
    out_dir = figure_dir("chi_variables", "variance_factor_effects")
    print(f"Loaded cell_summary.csv (rows={len(df):,}); writing parallel_slopes → {out_dir}")
    plot_parallel_slopes(df, out_dir)


if __name__ == "__main__":
    main()
