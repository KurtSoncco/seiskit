r"""Variability figures from per-cell central_variability summaries.

Reads ``cell_summary.csv`` (no HDF5 recompute) and writes Nature-styled PDFs
under ``figure_dir("chi_variables", "variability")``:

- ``decomposition/`` — stacked seed/node variance fractions (+ s_total boxes)
- ``components_by_factor/`` — rms components vs design factors (freq / IM)
- ``fractions_by_factor/`` — seed- and node-split fractions vs factors
- ``factor_cross/`` — rH / CoV / aHV sweeps of s_total and seed fractions
  at fixed (Height, Vs1)

Usage::

    python variability_plots.py
    python variability_plots.py --family decomposition
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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

H_LIST = [15.0, 50.0, 100.0]
VS1_LIST = [100.0, 230.0, 360.0]
CENTER = (30.0, 0.2, 10.0)  # (rH, CoV, aHV)

# Factor-cross columns: (xlabel, log_x, levels, list of (rh, cov, ahv))
FACTOR_COLS: list[tuple[str, bool, list[float], list[tuple[float, float, float]]]] = [
    (
        r"$r_h$ (m)",
        False,
        [10.0, 30.0, 50.0],
        [(10.0, 0.2, 10.0), (30.0, 0.2, 10.0), (50.0, 0.2, 10.0)],
    ),
    (
        "CoV",
        False,
        [0.1, 0.2, 0.3],
        [(30.0, 0.1, 10.0), (30.0, 0.2, 10.0), (30.0, 0.3, 10.0)],
    ),
    (
        r"$a_{hv}$",
        True,
        [1.0, 10.0, 50.0],
        [(30.0, 0.2, 1.0), (30.0, 0.2, 10.0), (30.0, 0.2, 50.0)],
    ),
]

GRID_ALPHA = 0.18
TEXT_BBOX = {"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.6}
LEGEND_FRAME = {
    "frameon": True,
    "fancybox": False,
    "framealpha": 0.75,
    "facecolor": "white",
    "edgecolor": "none",
    "borderpad": 0.25,
}

# Component line styles (rms σ_ln)
# Bars stay s (̄s_W, ̄s_B). Former σ² terms display as σ (rms), not s.
COMP_STYLES = {
    "s_W_bar": {"ls": "-", "marker": "o", "label": r"$\overline{s}_W$"},
    "s_mu": {"ls": "--", "marker": "s", "label": r"$\sigma_\mu$"},
    "s_B_bar": {"ls": ":", "marker": "^", "label": r"$\overline{s}_B$"},
    "s_total": {"ls": "-", "marker": "D", "label": r"$\sigma_{\mathrm{total}}$", "lw": 1.2},
    "s_nu": {"ls": "-.", "marker": "x", "label": r"$\sigma_\nu$", "alpha": 0.45},
}

# Seed/node fraction colors (neutral, not metric-specific)
FRAC_COLORS = {
    "frac_W_seed": "#4477AA",
    "frac_mu": "#EE6677",
    "frac_B_node": "#4477AA",
    "frac_nu": "#CCBB44",
}
FRAC_LABELS = {
    "frac_W_seed": r"$f_W$",
    "frac_mu": r"$f_\mu$",
    "frac_B_node": r"$f_B$",
    "frac_nu": r"$f_\nu$",
}

FAMILIES = (
    "decomposition",
    "components_by_factor",
    "fractions_by_factor",
    "factor_cross",
)


def load_cell_summary() -> pd.DataFrame:
    path = figure_dir("chi_variables", "central_variability") / "cell_summary.csv"
    df = pd.read_csv(path)
    if "metric" not in df.columns:
        raise ValueError(f"Expected 'metric' column in {path}")
    return df


def _format_level(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value)}"
    return f"{value:g}"


def _out(*parts: str) -> Path:
    return figure_dir("chi_variables", "variability", *parts)


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


# ---------------------------------------------------------------------------
# 1. Decomposition
# ---------------------------------------------------------------------------


def plot_decomposition(df: pd.DataFrame) -> None:
    out_dir = _out("decomposition")
    _plot_stacked_split(
        df,
        bottom_col="frac_W_seed",
        top_col="frac_mu",
        bottom_label=FRAC_LABELS["frac_W_seed"],
        top_label=FRAC_LABELS["frac_mu"],
        stem="seed_split_stacked",
        out_dir=out_dir,
    )
    _plot_stacked_split(
        df,
        bottom_col="frac_B_node",
        top_col="frac_nu",
        bottom_label=FRAC_LABELS["frac_B_node"],
        top_label=FRAC_LABELS["frac_nu"],
        stem="node_split_stacked",
        out_dir=out_dir,
    )


def _plot_stacked_split(
    df: pd.DataFrame,
    *,
    bottom_col: str,
    top_col: str,
    bottom_label: str,
    top_label: str,
    stem: str,
    out_dir: Path,
) -> None:
    """Two-row figure: stacked fractions + s_total boxplots."""
    fig = plt.figure(figsize=figsize(aspect=0.55))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[1.15, 1.0],
        hspace=0.28,
        left=0.14,
        right=0.98,
        bottom=0.10,
        top=0.97,
    )
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    y_pos = np.arange(len(METRICS))
    bottoms = []
    tops = []
    s_totals = []
    for metric in METRICS:
        sub = df[df["metric"] == metric]
        bottoms.append(float(sub[bottom_col].median()))
        tops.append(float(sub[top_col].median()))
        s_totals.append(sub["s_total"].to_numpy(dtype=float))

    bottoms_a = np.asarray(bottoms)
    tops_a = np.asarray(tops)
    colors = [metric_color(m) for m in METRICS]

    # Bottom segment: lighter face of metric color
    ax0.barh(
        y_pos,
        bottoms_a,
        height=0.65,
        color=colors,
        edgecolor="0.25",
        linewidth=0.6,
        alpha=0.45,
        hatch="",
        label=bottom_label,
    )
    ax0.barh(
        y_pos,
        tops_a,
        left=bottoms_a,
        height=0.65,
        color=colors,
        edgecolor="0.25",
        linewidth=0.6,
        alpha=0.90,
        hatch="//",
        label=top_label,
    )
    for i, (b, t, st) in enumerate(zip(bottoms_a, tops_a, [float(np.median(s)) for s in s_totals])):
        ax0.text(
            min(b + t + 0.02, 1.02),
            i,
            rf"$\sigma_{{\mathrm{{total}}}}={st:.3f}$",
            va="center",
            ha="left",
            fontsize=TICK_LABELSIZE,
            bbox=TEXT_BBOX,
        )

    ax0.set_yticks(y_pos)
    ax0.set_yticklabels([metric_label(m) for m in METRICS], fontsize=LABEL_FONTSIZE)
    ax0.set_xlim(0.0, 1.18)
    ax0.set_xlabel(r"Fraction of $\sigma^2_{\mathrm{total}}$", fontsize=LABEL_FONTSIZE)
    ax0.set_ylim(-0.6, len(METRICS) - 0.4)
    ax0.axvline(1.0, color="0.5", lw=0.5, ls=":")
    ax0.grid(True, axis="x", alpha=GRID_ALPHA, lw=0.6)
    ax0.set_axisbelow(True)
    add_panel_label(ax0, 0, alpha=0.75)

    # Legend distinguishing bottom vs top via alpha/hatch (shared gray proxy)
    legend_handles = [
        Patch(facecolor="0.55", edgecolor="0.25", alpha=0.45, label=bottom_label),
        Patch(facecolor="0.55", edgecolor="0.25", alpha=0.90, hatch="//", label=top_label),
    ]
    ax0.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=TICK_LABELSIZE,
        **LEGEND_FRAME,
    )

    # s_total boxplots
    bp = ax1.boxplot(
        s_totals,
        positions=y_pos,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        whis=(10, 90),
        vert=True,
        manage_ticks=False,
    )
    for box, color in zip(bp["boxes"], colors):
        box.set_facecolor(color)
        box.set_edgecolor(color)
        box.set_alpha(0.45)
        box.set_linewidth(0.9)
    for key in ("whiskers", "caps"):
        for artist in bp[key]:
            artist.set_color("0.35")
            artist.set_linewidth(0.8)
    for med in bp["medians"]:
        med.set_color("0.15")
        med.set_linewidth(1.1)

    ax1.set_xticks(y_pos)
    ax1.set_xticklabels([metric_label(m) for m in METRICS], fontsize=LABEL_FONTSIZE)
    ax1.set_ylabel(r"$\sigma_{\mathrm{total}}$", fontsize=LABEL_FONTSIZE)
    ax1.set_xlim(-0.6, len(METRICS) - 0.4)
    ax1.grid(True, axis="y", alpha=GRID_ALPHA, lw=0.6)
    ax1.set_axisbelow(True)
    add_panel_label(ax1, 1, alpha=0.75)

    save_figure(fig, stem, out_dir=out_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Components by factor
# ---------------------------------------------------------------------------


def plot_components_by_factor(df: pd.DataFrame) -> None:
    out_dir = _out("components_by_factor")
    _plot_metric_factor_grid(
        df,
        metrics=FREQ_METRICS,
        cols=("s_W_bar", "s_mu", "s_B_bar", "s_total"),
        include_s_nu=True,
        stem="components_freq",
        aspect=0.55,
        out_dir=out_dir,
        y_share_row=True,
    )
    _plot_metric_factor_grid(
        df,
        metrics=IM_METRICS,
        cols=("s_W_bar", "s_mu", "s_B_bar", "s_total"),
        include_s_nu=True,
        stem="components_im",
        aspect=0.72,
        out_dir=out_dir,
        y_share_row=True,
    )


def _plot_metric_factor_grid(
    df: pd.DataFrame,
    *,
    metrics: tuple[str, ...],
    cols: tuple[str, ...],
    include_s_nu: bool,
    stem: str,
    aspect: float,
    out_dir: Path,
    y_share_row: bool,
    ylim: tuple[float, float] | None = None,
) -> None:
    plot_cols = list(cols) + (["s_nu"] if include_s_nu else [])
    nrows, ncols = len(metrics), len(FACTORS)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize(aspect=aspect),
        sharex="col",
        sharey="row" if y_share_row else False,
        constrained_layout=False,
    )
    if nrows == 1:
        axes = np.asarray(axes).reshape(1, -1)

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
            for col in plot_cols:
                levels, meds = _median_by_factor(df, metric, factor, col)
                style = COMP_STYLES[col]
                kw = dict(
                    color=color,
                    ls=style["ls"],
                    marker=style["marker"],
                    markersize=3.5,
                    lw=style.get("lw", DATA_LINEWIDTH),
                    alpha=style.get("alpha", 0.95),
                    zorder=5 if col != "s_nu" else 3,
                )
                ax.plot(levels, meds, **kw)
                row_vals.extend(meds.tolist())

            ax.set_xticks(levels)
            ax.set_xticklabels([_format_level(v) for v in levels])
            pad = 0.15 * (levels[-1] - levels[0]) if len(levels) > 1 else 0.5
            ax.set_xlim(levels[0] - pad, levels[-1] + pad)
            ax.grid(True, which="major", axis="y", alpha=GRID_ALPHA, lw=0.6)
            ax.set_axisbelow(True)
            add_panel_label(ax, panel_i, alpha=0.75)
            panel_i += 1

            if c == 0:
                ax.set_ylabel(metric_label(metric), fontsize=LABEL_FONTSIZE)
            else:
                ax.tick_params(labelleft=False)

            if r == nrows - 1:
                ax.set_xlabel(factor, fontsize=LABEL_FONTSIZE)
            else:
                ax.tick_params(labelbottom=False)

        if ylim is not None:
            for c in range(ncols):
                axes[r, c].set_ylim(*ylim)
        elif y_share_row and row_vals:
            finite = [v for v in row_vals if np.isfinite(v)]
            if finite:
                lo, hi = min(finite), max(finite)
                span = max(hi - lo, 1e-3)
                pad = 0.12 * span
                for c in range(ncols):
                    axes[r, c].set_ylim(max(0.0, lo - pad), hi + pad)

    handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            ls=COMP_STYLES[col]["ls"],
            marker=COMP_STYLES[col]["marker"],
            markersize=3.5,
            lw=COMP_STYLES[col].get("lw", DATA_LINEWIDTH),
            alpha=COMP_STYLES[col].get("alpha", 0.95),
            label=COMP_STYLES[col]["label"],
        )
        for col in plot_cols
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        fontsize=TICK_LABELSIZE,
        bbox_to_anchor=(0.5, 0.995),
        **LEGEND_FRAME,
    )

    save_figure(fig, stem, out_dir=out_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Fractions by factor
# ---------------------------------------------------------------------------


def plot_fractions_by_factor(df: pd.DataFrame) -> None:
    out_dir = _out("fractions_by_factor")
    _plot_fraction_grid(
        df,
        metrics=METRICS,
        bottom_col="frac_W_seed",
        top_col="frac_mu",
        stem="fractions_seed_split",
        aspect=0.85,
        out_dir=out_dir,
    )
    _plot_fraction_grid(
        df,
        metrics=METRICS,
        bottom_col="frac_B_node",
        top_col="frac_nu",
        stem="fractions_node_split",
        aspect=0.85,
        out_dir=out_dir,
    )


def _plot_fraction_grid(
    df: pd.DataFrame,
    *,
    metrics: tuple[str, ...],
    bottom_col: str,
    top_col: str,
    stem: str,
    aspect: float,
    out_dir: Path,
) -> None:
    """Metric × factor grid of stacked median fractions (fills to ~1)."""
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
        bottom=0.10,
        top=0.97,
        wspace=0.12,
        hspace=0.18,
    )

    c_bot = FRAC_COLORS[bottom_col]
    c_top = FRAC_COLORS[top_col]
    panel_i = 0
    for r, metric in enumerate(metrics):
        for c, factor in enumerate(FACTORS):
            ax = axes[r, c]
            levels, f_bot = _median_by_factor(df, metric, factor, bottom_col)
            _, f_top = _median_by_factor(df, metric, factor, top_col)
            (0.28 * (levels[-1] - levels[0]) / max(len(levels) - 1, 1) if len(levels) > 1 else 0.3)
            # Use categorical bar positions for clarity when levels are uneven
            x = np.arange(len(levels), dtype=float)
            ax.bar(
                x,
                f_bot,
                width=0.55,
                color=c_bot,
                edgecolor="0.25",
                linewidth=0.5,
                alpha=0.85,
                hatch="",
                label=FRAC_LABELS[bottom_col],
            )
            ax.bar(
                x,
                f_top,
                bottom=f_bot,
                width=0.55,
                color=c_top,
                edgecolor="0.25",
                linewidth=0.5,
                alpha=0.85,
                hatch="",
                label=FRAC_LABELS[top_col],
            )
            ax.set_xticks(x)
            ax.set_xticklabels([_format_level(v) for v in levels])
            ax.set_ylim(0.0, 1.05)
            ax.axhline(1.0, color="0.6", lw=0.4, ls=":")
            ax.grid(True, axis="y", alpha=GRID_ALPHA, lw=0.6)
            ax.set_axisbelow(True)
            add_panel_label(ax, panel_i, alpha=0.75)
            panel_i += 1

            if c == 0:
                ax.set_ylabel(metric_label(metric), fontsize=LABEL_FONTSIZE)
            else:
                ax.tick_params(labelleft=False)

            if r == nrows - 1:
                ax.set_xlabel(factor, fontsize=LABEL_FONTSIZE)
            else:
                ax.tick_params(labelbottom=False)

    handles = [
        Patch(facecolor=c_bot, edgecolor="0.25", alpha=0.85, label=FRAC_LABELS[bottom_col]),
        Patch(facecolor=c_top, edgecolor="0.25", alpha=0.85, label=FRAC_LABELS[top_col]),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        fontsize=TICK_LABELSIZE,
        bbox_to_anchor=(0.55, 0.935),
        **LEGEND_FRAME,
    )

    save_figure(fig, stem, out_dir=out_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Factor cross (per H, Vs1)
# ---------------------------------------------------------------------------


def plot_factor_cross(df: pd.DataFrame) -> None:
    out_s = _out("factor_cross", "s_total")
    out_f = _out("factor_cross", "seed_frac")
    for h in H_LIST:
        for vs1 in VS1_LIST:
            print(f"  factor_cross H={h:.0f}, Vs1={vs1:.0f} …")
            _plot_s_total_cross(df, h=h, vs1=vs1, out_dir=out_s)
            _plot_seed_frac_cross(df, h=h, vs1=vs1, out_dir=out_f)


def _cell_row(
    df: pd.DataFrame,
    *,
    h: float,
    vs1: float,
    rh: float,
    cov: float,
    ahv: float,
    metric: str,
) -> pd.Series | None:
    mask = (
        (df["Height"] == h)
        & (df["Vs1"] == vs1)
        & (df["rH"] == rh)
        & (df["CoV"] == cov)
        & (df["aHV"] == ahv)
        & (df["metric"] == metric)
    )
    sub = df.loc[mask]
    if sub.empty:
        return None
    return sub.iloc[0]


def _cross_header(fig: plt.Figure, h: float, vs1: float) -> None:
    rh0, cov0, ahv0 = CENTER
    fig.text(
        0.5,
        0.985,
        rf"($H = {h:.0f}$ m, $V_{{s1}} = {vs1:.0f}$ m/s; "
        rf"others fixed at $r_h = {rh0:.0f}$ m, CoV $= {cov0:g}$, "
        rf"$a_{{hv}} = {ahv0:.0f}$)",
        ha="center",
        va="top",
        fontsize=TICK_LABELSIZE,
        bbox=TEXT_BBOX,
    )


def _plot_s_total_cross(
    df: pd.DataFrame,
    *,
    h: float,
    vs1: float,
    out_dir: Path,
) -> None:
    fig = plt.figure(figsize=figsize(aspect=0.42))
    gs = fig.add_gridspec(1, 3, wspace=0.10, left=0.09, right=0.995, bottom=0.18, top=0.90)
    axes = [fig.add_subplot(gs[0, c]) for c in range(3)]
    _cross_header(fig, h, vs1)

    panel_i = 0
    y_all: list[float] = []
    for c, (xlabel, log_x, levels, cells) in enumerate(FACTOR_COLS):
        ax = axes[c]
        levels_arr = np.asarray(levels, dtype=float)
        for metric in METRICS:
            ys = []
            for rh, cov, ahv in cells:
                row = _cell_row(df, h=h, vs1=vs1, rh=rh, cov=cov, ahv=ahv, metric=metric)
                ys.append(float(row["s_total"]) if row is not None else np.nan)
            y_all.extend(ys)
            ax.plot(
                levels_arr,
                ys,
                color=metric_color(metric),
                ls="-",
                marker="o",
                markersize=4,
                lw=DATA_LINEWIDTH,
                label=metric_label(metric),
            )
        if log_x:
            ax.set_xscale("log")
        ax.set_xticks(levels_arr)
        ax.set_xticklabels([_format_level(v) for v in levels_arr])
        if log_x:
            ax.set_xlim(levels_arr[0] * 10 ** (-0.08), levels_arr[-1] * 10 ** (0.08))
        else:
            pad = 0.15 * (levels_arr[-1] - levels_arr[0])
            ax.set_xlim(levels_arr[0] - pad, levels_arr[-1] + pad)
        ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
        ax.grid(True, axis="y", alpha=GRID_ALPHA, lw=0.6)
        ax.set_axisbelow(True)
        add_panel_label(ax, panel_i, alpha=0.75)
        panel_i += 1
        if c == 0:
            ax.set_ylabel(r"$\sigma_{\mathrm{total}}$", fontsize=LABEL_FONTSIZE)
        else:
            ax.tick_params(labelleft=False)

    finite = [v for v in y_all if np.isfinite(v)]
    if finite:
        lo, hi = min(finite), max(finite)
        span = max(hi - lo, 1e-3)
        pad = 0.12 * span
        for ax in axes:
            ax.set_ylim(max(0.0, lo - pad), hi + pad)

    handles = [
        Line2D(
            [0],
            [0],
            color=metric_color(m),
            marker="o",
            ls="-",
            lw=DATA_LINEWIDTH,
            markersize=4,
            label=metric_label(m),
        )
        for m in METRICS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        fontsize=TICK_LABELSIZE,
        bbox_to_anchor=(0.55, 0.01),
        **LEGEND_FRAME,
    )

    stem = f"s_total_cross_h{h:.0f}_vs1_{vs1:.0f}"
    save_figure(fig, stem, out_dir=out_dir)
    plt.close(fig)


def _plot_seed_frac_cross(
    df: pd.DataFrame,
    *,
    h: float,
    vs1: float,
    out_dir: Path,
) -> None:
    """Stacked f_W / f_μ bars per metric at each factor level (3 panels)."""
    fig = plt.figure(figsize=figsize(aspect=0.48))
    gs = fig.add_gridspec(1, 3, wspace=0.14, left=0.09, right=0.995, bottom=0.18, top=0.90)
    axes = [fig.add_subplot(gs[0, c]) for c in range(3)]
    _cross_header(fig, h, vs1)

    n_met = len(METRICS)
    bar_w = 0.14
    c_w = FRAC_COLORS["frac_W_seed"]
    c_mu = FRAC_COLORS["frac_mu"]
    panel_i = 0

    for c, (xlabel, _log_x, levels, cells) in enumerate(FACTOR_COLS):
        ax = axes[c]
        n_lv = len(levels)
        # Group offset per level; within group, one bar per metric
        x_base = np.arange(n_lv, dtype=float)
        offsets = (np.arange(n_met) - (n_met - 1) / 2.0) * bar_w

        for mi, metric in enumerate(METRICS):
            f_w = []
            f_mu = []
            for rh, cov, ahv in cells:
                row = _cell_row(df, h=h, vs1=vs1, rh=rh, cov=cov, ahv=ahv, metric=metric)
                if row is None:
                    f_w.append(np.nan)
                    f_mu.append(np.nan)
                else:
                    f_w.append(float(row["frac_W_seed"]))
                    f_mu.append(float(row["frac_mu"]))
            xpos = x_base + offsets[mi]
            ax.bar(
                xpos,
                f_w,
                width=bar_w * 0.92,
                color=c_w,
                edgecolor="0.25",
                linewidth=0.4,
                alpha=0.9,
                hatch="",
            )
            ax.bar(
                xpos,
                f_mu,
                bottom=f_w,
                width=bar_w * 0.92,
                color=c_mu,
                edgecolor="0.25",
                linewidth=0.4,
                alpha=0.9,
                hatch="",
            )
            # Metric color tick on top of each stack for identification
            tops = np.asarray(f_w) + np.asarray(f_mu)
            ax.scatter(
                xpos,
                tops + 0.03,
                s=10,
                color=metric_color(metric),
                marker="v",
                zorder=6,
                linewidths=0,
            )

        ax.set_xticks(x_base)
        ax.set_xticklabels([_format_level(v) for v in levels])
        ax.set_ylim(0.0, 1.12)
        ax.axhline(1.0, color="0.6", lw=0.4, ls=":")
        ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
        ax.grid(True, axis="y", alpha=GRID_ALPHA, lw=0.6)
        ax.set_axisbelow(True)
        add_panel_label(ax, panel_i, alpha=0.75)
        panel_i += 1
        if c == 0:
            ax.set_ylabel(r"Fraction of $\sigma^2_{\mathrm{total}}$", fontsize=LABEL_FONTSIZE)
        else:
            ax.tick_params(labelleft=False)

    handles = [
        Patch(facecolor=c_w, edgecolor="0.25", alpha=0.9, label=FRAC_LABELS["frac_W_seed"]),
        Patch(facecolor=c_mu, edgecolor="0.25", alpha=0.9, label=FRAC_LABELS["frac_mu"]),
        *[
            Line2D(
                [0],
                [0],
                color=metric_color(m),
                marker="v",
                ls="none",
                markersize=5,
                label=metric_label(m),
            )
            for m in METRICS
        ],
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=7,
        fontsize=TICK_LABELSIZE,
        bbox_to_anchor=(0.55, 0.005),
        **LEGEND_FRAME,
    )

    stem = f"seed_frac_cross_h{h:.0f}_vs1_{vs1:.0f}"
    save_figure(fig, stem, out_dir=out_dir)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(families: tuple[str, ...] | None = None) -> None:
    families = families or FAMILIES
    for name in families:
        if name not in FAMILIES:
            raise SystemExit(f"Unknown family {name!r}; choose from {FAMILIES}")

    print("Loading cell_summary.csv …")
    df = load_cell_summary()
    print(f"  rows={len(df):,}")
    root = _out()
    print(f"  → {root}")

    if "decomposition" in families:
        print("Writing decomposition …")
        plot_decomposition(df)

    if "components_by_factor" in families:
        print("Writing components_by_factor …")
        plot_components_by_factor(df)

    if "fractions_by_factor" in families:
        print("Writing fractions_by_factor …")
        plot_fractions_by_factor(df)

    if "factor_cross" in families:
        print("Writing factor_cross …")
        plot_factor_cross(df)

    print(f"Done → {root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        action="append",
        choices=list(FAMILIES),
        help="Restrict to one or more figure families (repeatable).",
    )
    args = parser.parse_args()
    main(tuple(args.family) if args.family else None)
