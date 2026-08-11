r"""Parallel-slope variance figures across design factors.

Reads ``cell_summary.csv`` and writes Nature PDFs under
``figure_dir("chi_variables", "variance_factor_effects")/parallel_slopes/``:

metric × factor lines for \(\overline{s}_W\), \(\sigma_\mu\),
\(\sigma_{\mathrm{total}}\) (rms of log-χ; CSV columns stay ``s_*``).

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
FREQ_METRICS = ("f_ratio", "abs_TF_ratio")
IM_METRICS = ("PGA_ratio", "PSA_ratio", "Ia_ratio")

FACTOR_XLABELS = {
    "Vs1": r"$V_{s1}$ (m/s)",
    "Height": r"$H$ (m)",
    "CoV": "CoV",
    "rH": r"$r_h$ (m)",
    "aHV": r"$a_{hv}$",
}

COMP_COLS = ("s_W_bar", "s_mu", "s_total")
# Bars stay s (̄s_W, ̄s_B). Former σ² terms display as σ (rms), not s.
COMP_STYLES = {
    "s_W_bar": {"ls": "-", "marker": "o", "label": r"$\overline{s}_W$", "lw": DATA_LINEWIDTH},
    "s_mu": {"ls": "--", "marker": "s", "label": r"$\sigma_\mu$", "lw": DATA_LINEWIDTH},
    "s_total": {"ls": "-", "marker": "D", "label": r"$\sigma_{\mathrm{total}}$", "lw": 1.2},
}

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


def _out(*parts: str) -> Path:
    return figure_dir("chi_variables", "variance_factor_effects", *parts)


def _format_level(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value)}"
    return f"{value:g}"


def _median_by_factor(
    df: pd.DataFrame, metric: str, factor: str, col: str
) -> tuple[np.ndarray, np.ndarray]:
    sub = df[df["metric"] == metric]
    levels = np.sort(sub[factor].unique().astype(float))
    meds = np.array(
        [float(sub.loc[sub[factor] == lv, col].median()) for lv in levels],
        dtype=float,
    )
    return levels, meds


def _comp_handles(cols: tuple[str, ...] = COMP_COLS) -> list[Line2D]:
    handles = []
    for col in cols:
        style = COMP_STYLES[col]
        handles.append(
            Line2D(
                [0],
                [0],
                color="0.25",
                ls=style["ls"],
                marker=style["marker"],
                markersize=3.5,
                lw=style.get("lw", DATA_LINEWIDTH),
                alpha=style.get("alpha", 0.95),
                label=style["label"],
            )
        )
    return handles


def plot_parallel_slopes(df: pd.DataFrame) -> None:
    out_dir = _out("parallel_slopes")
    _plot_parallel_grid(
        df,
        metrics=FREQ_METRICS,
        cols=COMP_COLS,
        stem="parallel_slopes_freq",
        aspect=0.48,
        out_dir=out_dir,
    )
    _plot_parallel_grid(
        df,
        metrics=IM_METRICS,
        cols=COMP_COLS,
        stem="parallel_slopes_im",
        aspect=0.68,
        out_dir=out_dir,
    )
    _plot_parallel_grid(
        df,
        metrics=METRICS,
        cols=COMP_COLS,
        stem="parallel_slopes_all",
        aspect=0.92,
        out_dir=out_dir,
    )


def _plot_parallel_grid(
    df: pd.DataFrame,
    *,
    metrics: tuple[str, ...],
    cols: tuple[str, ...],
    stem: str,
    aspect: float,
    out_dir: Path,
) -> None:
    nrows, ncols = len(metrics), len(FACTORS)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize(aspect=aspect),
        sharex="col",
        sharey="row",
        constrained_layout=False,
    )
    axes = np.asarray(axes).reshape(nrows, ncols)

    fig.subplots_adjust(
        left=0.09,
        right=0.98,
        bottom=0.12,
        top=0.90,
        wspace=0.12,
        hspace=0.18,
    )

    panel_i = 0
    for r, metric in enumerate(metrics):
        color = metric_color(metric)
        row_vals: list[float] = []
        for c, factor in enumerate(FACTORS):
            ax = axes[r, c]
            levels, _ = _median_by_factor(df, metric, factor, cols[0])
            for col in cols:
                _, meds = _median_by_factor(df, metric, factor, col)
                style = COMP_STYLES[col]
                ax.plot(
                    levels,
                    meds,
                    color=color,
                    ls=style["ls"],
                    marker=style["marker"],
                    markersize=3.5,
                    lw=style.get("lw", DATA_LINEWIDTH),
                    alpha=style.get("alpha", 0.95),
                    zorder=5,
                )
                row_vals.extend(meds.tolist())

            # Natural (linear) scale at true factor levels — not categorical.
            span = float(levels[-1] - levels[0]) if len(levels) > 1 else max(float(levels[0]), 1.0)
            pad = 0.12 * span
            ax.set_xlim(float(levels[0]) - pad, float(levels[-1]) + pad)
            ax.set_xticks(levels)
            ax.set_xticklabels([_format_level(v) for v in levels])
            ax.grid(True, which="major", axis="y", alpha=GRID_ALPHA, lw=0.6)
            ax.set_axisbelow(True)
            add_panel_label(ax, panel_i, alpha=0.75)
            panel_i += 1

            if c == 0:
                ax.set_ylabel(metric_label(metric, log=True), fontsize=LABEL_FONTSIZE)
            else:
                ax.tick_params(labelleft=False)

            if r == nrows - 1:
                ax.set_xlabel(FACTOR_XLABELS[factor], fontsize=LABEL_FONTSIZE)
            else:
                ax.tick_params(labelbottom=False)

        finite = [v for v in row_vals if np.isfinite(v)]
        if finite:
            lo, hi = min(finite), max(finite)
            span = max(hi - lo, 1e-3)
            pad = 0.12 * span
            for c in range(ncols):
                axes[r, c].set_ylim(max(0.0, lo - pad), hi + pad)

    fig.legend(
        handles=_comp_handles(cols),
        loc="upper center",
        ncol=len(cols),
        fontsize=TICK_LABELSIZE,
        bbox_to_anchor=(0.5, 0.995),
        **LEGEND_FRAME,
    )
    save_figure(fig, stem, out_dir=out_dir)
    plt.close(fig)


def main() -> None:
    print("Loading cell_summary.csv …")
    df = load_cell_summary()
    print(f"  rows={len(df):,}")
    out_dir = _out("parallel_slopes")
    print(f"  → {out_dir}")
    print("Writing parallel_slopes …")
    plot_parallel_slopes(df)
    print(f"Done → {out_dir}")


if __name__ == "__main__":
    main()
