"""Histograms of \\(Y_{ij}=\\ln\\chi_{ij}\\) with matching Normal densities.

Single figure, rows = samples (never pooled with each other), columns = metrics:

- top — ``center_node_all_seeds``: center node, all seeds (between-seed)
- bottom — ``one_seed_all_nodes``: first seed, all nodes (within-seed)

Writes ``hist_distributions.pdf`` under
``figure_dir("chi_variables", "distribution_histograms")``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

_FULL = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_FULL))
sys.path.insert(0, str(_CODE))

from _shared import METRICS, Sample, load_ratios, log_response, select_sample  # noqa: E402
from config import (  # noqa: E402
    DATA_LINEWIDTH,
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

N_BINS = 60
HIST_ALPHA = 0.55
GRID_ALPHA = 0.18
FIT_COLOR = "0.15"

ROWS: tuple[tuple[Sample, str], ...] = (
    ("center_node_all_seeds", "Center node,\nall seeds"),
    ("one_seed_all_nodes", "One seed,\nall nodes"),
)


def _normal_pdf(y: np.ndarray, xs: np.ndarray) -> np.ndarray:
    mu, sd = float(np.mean(y)), float(np.std(y, ddof=0))
    return np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def main() -> None:
    df = load_ratios()
    # s = "center_node_all_seeds"  # Define the sample identifier 's'
    samples = {s: select_sample(df, s) for s, _ in ROWS}

    metrics = list(METRICS)
    ncols = len(metrics)
    fig, axes = plt.subplots(
        len(ROWS),
        ncols,
        figsize=figsize(aspect=0.5),
        sharex="col",
        sharey="col",
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.12, top=0.90, wspace=0.28, hspace=0.12)

    for c, metric in enumerate(metrics):
        ys = {}
        for s, _ in ROWS:
            y = log_response(samples[s], metric)
            ys[s] = y[np.isfinite(y)]
        lo, hi = np.percentile(np.concatenate(list(ys.values())), [0.5, 99.5])
        bins = np.linspace(lo, hi, N_BINS)
        xs = np.linspace(lo, hi, 400)

        for row, (sample, row_title) in enumerate(ROWS):
            ax = axes[row, c]
            y = ys[sample]
            ax.hist(
                y,
                bins=bins,
                density=True,
                color=metric_color(metric),
                alpha=HIST_ALPHA,
                edgecolor="none",
            )
            ax.plot(xs, _normal_pdf(y, xs), color=FIT_COLOR, lw=DATA_LINEWIDTH * 0.9, ls="--")
            if row == len(ROWS) - 1:
                ax.set_xlabel(metric_label(metric, log=True), fontsize=LABEL_FONTSIZE)
            if c == 0:
                ax.set_ylabel(f"{row_title}\nDensity", fontsize=LABEL_FONTSIZE)
            ax.set_xlim(lo, hi)
            ax.grid(True, axis="y", alpha=GRID_ALPHA, lw=0.5)
            ax.set_axisbelow(True)
            add_panel_label(ax, row * ncols + c, alpha=0.75)

    fig.legend(
        handles=[
            Line2D([0], [0], color=FIT_COLOR, lw=DATA_LINEWIDTH * 0.9, ls="--", label="Normal fit")
        ],
        loc="upper right",
        fontsize=TICK_LABELSIZE,
        bbox_to_anchor=(0.99, 1.0),
        frameon=True,
        fancybox=False,
        framealpha=0.75,
        facecolor="white",
        edgecolor="none",
        borderpad=0.25,
    )
    save_figure(
        fig, "hist_distributions", out_dir=figure_dir("chi_variables", "distribution_histograms")
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
