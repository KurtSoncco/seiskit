"""Central-tendency profile figures for χ ratios (abs_TF_ratio by default).

Nature-width figures under ``figure_dir("chi_variables", "central_profiles")``,
one PDF per (Height, Vs1):

1. Node profiles — 1×3 factor row (vary CoV / rH / aHV about the base cell
    rH=30, CoV=0.2, aHV=10). Each panel overlays the seed-geomean node profile
    for the three levels of that factor (low→high = blue solid / green dashed /
    red dash-dot), with geometric ±1 log-SD shown as translucent filled bands.
2. Seed profiles — geomean and median across nodes vs seed (3×3, legacy).

Factor-sweep geomean boxplots live in ``geomean_factor_cross.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.legend import Legend
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import (  # noqa: E402
    BOX_ROOT,
    DATA_LINEWIDTH,
    LABEL_FONTSIZE,
    TICK_LABELSIZE,
    TOL_BRIGHT,
    add_panel_label,
    apply_full_paper_style,
    figsize,
    figure_dir,
    metric_color,
    metric_label,
    save_figure,
)

apply_full_paper_style(auto_format=True, frame="open", grid=False)

DATA_PATH = BOX_ROOT / "peak_analysis" / "join_master.h5"
METRIC = "abs_TF_ratio"
LOG_Y_METRICS = frozenset({"abs_TF_ratio", "Ia_ratio"})

H_LIST = [15.0, 50.0, 100.0]
VS1_LIST = [100.0, 230.0, 360.0]

# (rH, CoV, aHV) for panels (a)–(i), row-major — same as TF qualitative
# (used only by the legacy 3×3 seed-profile figure).
PANELS: list[tuple[float, float, float]] = [
    (10.0, 0.2, 10.0),
    (30.0, 0.2, 10.0),
    (50.0, 0.2, 10.0),
    (30.0, 0.1, 10.0),
    (30.0, 0.2, 10.0),
    (30.0, 0.3, 10.0),
    (30.0, 0.2, 1.0),
    (30.0, 0.2, 10.0),
    (30.0, 0.2, 50.0),
]

# Base design cell for the 1×3 node-profile row; each panel sweeps one factor.
BASE_CELL: dict[str, float] = {"rH": 30.0, "CoV": 0.2, "aHV": 10.0}
PLOT_NODE_MIN = -100.0
PLOT_NODE_MAX = 100.0
X_TICKS = [-100, -50, 0, 50, 100]  # node offsets for x-ticks
VARY_PANELS: list[tuple[str, list[float]]] = [
    ("CoV", [0.1, 0.2, 0.3]),
    ("rH", [10.0, 30.0, 50.0]),
    ("aHV", [1.0, 10.0, 50.0]),
]

GRID_ALPHA = 0.18
Y_LIM = (1e-2, 1e0)
TEXT_BBOX = {"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.8}
LEGEND_FRAME = {
    "frameon": True,
    "fancybox": False,
    "framealpha": 0.75,
    "facecolor": "white",
    "edgecolor": "none",
    "borderpad": 0.3,
}

DOC_WIDTH, FIG_HEIGHT = figsize(aspect=0.88)


def load_ratios(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load joined ratio table; rename channel → node."""
    cols = [
        "Vs1",
        "Height",
        "CoV",
        "rH",
        "aHV",
        "channel",
        "seed",
        METRIC,
    ]
    with h5py.File(path, "r") as f:
        g = f["master"]
        df = pd.DataFrame({c: g[c][:] for c in cols})
    return df.rename(columns={"channel": "node"})


def _panel_param_text(rh: float, cov: float, ahv: float) -> str:
    return (
        rf"$r_h = {rh:.0f}$ m" + "\n"
        rf"$\mathrm{{CoV}} = {cov:g}$" + "\n"
        rf"$a_{{hv}} = {ahv:.0f}$"
    )


def _stem(kind: str, h: float, vs1: float) -> str:
    short = METRIC.replace("_ratio", "")
    return f"{short}_{kind}_h{h:.0f}_vs1_{vs1:.0f}"


def cell_matrix(
    df_hv: pd.DataFrame, rh: float, cov: float, ahv: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (nodes, seeds, chi[node, seed]) for one design cell."""
    mask = (df_hv["rH"] == rh) & (df_hv["CoV"] == cov) & (df_hv["aHV"] == ahv)
    sub = df_hv.loc[mask]
    piv = sub.pivot(index="node", columns="seed", values=METRIC)
    arr = piv.to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        arr = np.where(np.isfinite(arr) & (arr > 0), arr, np.nan)
    return piv.index.to_numpy(), piv.columns.to_numpy(), arr


def node_profiles(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Geomean, median, and log-SD across seeds (axis=1) at each node.

    ``log_std`` is the standard deviation of ``ln(chi)`` across seeds; the
    multiplicative (geometric) ±1-SD band around the geomean is
    ``[geo / exp(log_std), geo * exp(log_std)]``.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        log_a = np.log(arr)
    geo = np.exp(np.nanmean(log_a, axis=1))
    med = np.nanmedian(arr, axis=1)
    log_std = np.nanstd(log_a, axis=1)
    return geo, med, log_std


# Per-level line styling, shared across factors (low / base / high level).
_LEVEL_PALETTE: list[tuple[str, str]] = [
    (TOL_BRIGHT["blue"], "-"),
    (TOL_BRIGHT["green"], "--"),
    (TOL_BRIGHT["red"], "-."),
]
_LEVEL_MARKERS = ["o", "s", "^"]


def _level_label(factor: str, val: float) -> str:
    if factor == "CoV":
        return rf"$\mathrm{{CoV}} = {val:g}$"
    if factor == "rH":
        return rf"$r_h = {val:.0f}$ m"
    return rf"$a_{{hv}} = {val:.0f}$"


def _level_styles(factor: str, levels: list[float]) -> dict[float, dict[str, str]]:
    """Map each factor level to its visual style (low→high)."""
    return {
        lv: {
            "color": color,
            "ls": ls,
            "marker": marker,
            "label": _level_label(factor, lv),
        }
        for lv, (color, ls), marker in zip(levels, _LEVEL_PALETTE, _LEVEL_MARKERS)
    }


def _held_text(factor: str) -> str:
    """Base-cell values for the two factors held fixed in a panel."""
    held = [
        (rf"$r_h = {BASE_CELL['rH']:.0f}$ m", "rH"),
        (rf"$\mathrm{{CoV}} = {BASE_CELL['CoV']:g}$", "CoV"),
        (rf"$a_{{hv}} = {BASE_CELL['aHV']:.0f}$", "aHV"),
    ]
    return "\n".join(txt for txt, key in held if key != factor)


def seed_profiles(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Geomean and median across nodes (axis=0) at each seed."""
    with np.errstate(invalid="ignore", divide="ignore"):
        log_a = np.log(arr)
    geo = np.exp(np.nanmean(log_a, axis=0))
    med = np.nanmedian(arr, axis=0)
    return geo, med


def _make_3x3_figure(
    *,
    h: float,
    vs1: float,
    legend_handles: list,
) -> tuple[plt.Figure, np.ndarray]:
    fig = plt.figure(figsize=(DOC_WIDTH, FIG_HEIGHT))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.05, 1.0],
        hspace=0.02,
        left=0.08,
        right=0.995,
        bottom=0.06,
        top=0.99,
    )
    header = fig.add_subplot(gs[0, 0])
    header.axis("off")
    gs_panels = gs[1, 0].subgridspec(3, 3, wspace=0.06, hspace=0.08)
    axes = np.empty((3, 3), dtype=object)
    for r in range(3):
        for c in range(3):
            # Share x and y across the full 3×3 (tick labels only on first column / bottom row).
            sharex = axes[0, 0] if (r, c) != (0, 0) else None
            sharey = axes[0, 0] if (r, c) != (0, 0) else None
            axes[r, c] = fig.add_subplot(gs_panels[r, c], sharex=sharex, sharey=sharey)

    header.text(
        0.5,
        0.95,
        rf"($H = {h:.0f}$ m, $V_{{s1}} = {vs1:.0f}$ m/s; {metric_label(METRIC)})",
        transform=header.transAxes,
        ha="center",
        va="top",
        fontsize=TICK_LABELSIZE,
        bbox=TEXT_BBOX,
    )
    header.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        fontsize=TICK_LABELSIZE,
        handlelength=1.8,
        columnspacing=1.0,
        borderaxespad=0.0,
        labelspacing=0.1,
        bbox_to_anchor=(0.5, -0.05),
        **LEGEND_FRAME,
    )
    return fig, axes


def _annotate_panel(ax: plt.Axes, i: int, rh: float, cov: float, ahv: float) -> None:
    add_panel_label(ax, i, alpha=0.75)
    ax.text(
        0.02,
        0.97,
        _panel_param_text(rh, cov, ahv),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=TICK_LABELSIZE,
        linespacing=1.15,
        zorder=6,
        bbox=TEXT_BBOX,
    )
    ax.tick_params(labelsize=TICK_LABELSIZE)
    ax.grid(True, which="major", alpha=GRID_ALPHA, lw=0.6)
    ax.set_axisbelow(True)


def _apply_shared_ylims(axes: np.ndarray, *, log_y: bool) -> None:
    """Set fixed shared y-limits on the master axis (propagates to all panels)."""
    ax = axes[0, 0]
    if log_y:
        ax.set_yscale("log")
    ax.set_ylim(*Y_LIM)


def _make_1x3_figure(*, h: float, vs1: float) -> tuple[plt.Figure, np.ndarray]:
    """1×3 factor row with a shared (log) y-axis and a top (H, Vs1) title."""
    width, height = figsize(aspect=0.36)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(width, height),
        sharey=True,
        gridspec_kw=dict(
            wspace=0.06, left=0.08, right=0.995, bottom=0.16, top=0.86
        ),
    )
    # fig.text (not Figure.suptitle) to bypass the auto-format label patch,
    # which would re-substitute tokens inside the already-LaTeX title.
    fig.text(
        0.5,
        0.99,
        rf"($H = {h:.0f}$ m, $V_{{s1}} = {vs1:.0f}$ m/s; {metric_label(METRIC)})",
        ha="center",
        va="top",
        fontsize=TICK_LABELSIZE,
    )
    return fig, axes


def plot_node_profiles(df: pd.DataFrame, *, h: float, vs1: float, out_dir: Path) -> Path:
    """Node profiles as a 1×3 factor row for one (Height, Vs1) pair.

    Panels sweep CoV, then rH, then aHV about the base cell
    (rH=30, CoV=0.2, aHV=10). Within a panel the seed-geomean node profile is
    drawn for each of the three factor levels styled low→high via
    ``_level_styles`` (color, line style, and marker), with geometric ±1
    log-SD shown as translucent filled bands.
    """
    out_dir = out_dir / "node_profiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_y = METRIC in LOG_Y_METRICS
    df_hv = df[(df["Height"] == h) & (df["Vs1"] == vs1)]

    fig, axes = _make_1x3_figure(h=h, vs1=vs1)

    for i, (ax, (factor, levels)) in enumerate(zip(axes, VARY_PANELS)):
        styles = _level_styles(factor, levels)
        handles: list[Line2D] = []
        for j, level in enumerate(levels):
            st = styles[level]
            cell = {**BASE_CELL, factor: level}
            nodes, _seeds, arr = cell_matrix(
                df_hv, cell["rH"], cell["CoV"], cell["aHV"]
            )
            geo, _med, log_std = node_profiles(arr)
            gsd = np.exp(log_std)  # geometric ±1 log-SD factor
            offset = np.linspace(PLOT_NODE_MIN, PLOT_NODE_MAX, len(nodes))
            ax.plot(
                offset,
                geo,
                color=st["color"],
                ls=st["ls"],
                lw=DATA_LINEWIDTH,
                marker=st["marker"],
                markevery=10,
                ms=2.5,
                markerfacecolor="white",
                markeredgewidth=0.6,
                zorder=4,
            )
            # Geometric ±1 log-SD boundaries around the geomean profile.
            ax.plot(
                offset,
                geo / gsd,
                color=st["color"],
                ls=":",
                lw=1.0,
                alpha=1.0,
                zorder=3,
            )
            ax.plot(
                offset,
                geo * gsd,
                color=st["color"],
                ls=":",
                lw=1.0,
                alpha=1.0,
                zorder=3,
            )
            handles.append(
                Line2D(
                    [0], [0],
                    color=st["color"],
                    ls=st["ls"],
                    lw=DATA_LINEWIDTH,
                    marker=st["marker"],
                    markersize=3,
                    markerfacecolor="white",
                    markeredgewidth=0.6,
                    label=st["label"],
                )
            )

        add_panel_label(ax, i, alpha=0.75)
        # Legend built directly (not ax.legend) to bypass the auto-format patch,
        # which would nest ``$...$`` inside the already-LaTeX level labels.
        leg = Legend(
            ax,
            handles,
            [h.get_label() for h in handles],
            loc="upper left",
            fontsize=TICK_LABELSIZE * 0.9,
            handlelength=1.4,
            borderaxespad=0.3,
            labelspacing=0.2,
            **LEGEND_FRAME,
        )
        ax.add_artist(leg)
        ax.text(
            0.02, 0.03, _held_text(factor),
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=TICK_LABELSIZE, linespacing=1.15, zorder=6, bbox=TEXT_BBOX,
        )
        ax.tick_params(labelsize=TICK_LABELSIZE)
        ax.grid(True, which="major", alpha=GRID_ALPHA, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xticks(X_TICKS)
        ax.axvline(0.0, color="0.5", lw=0.6, ls=":", zorder=1)
        ax.set_xlabel("Distance from center (nodes)", fontsize=LABEL_FONTSIZE)
        if i == 0:
            ax.set_ylabel(metric_label(METRIC), fontsize=LABEL_FONTSIZE)
        else:
            ax.tick_params(labelleft=False)

    if log_y:
        axes[0].set_yscale("log")
    axes[0].set_ylim(*Y_LIM)

    paths = save_figure(fig, _stem("node_profile", h, vs1), out_dir=out_dir)
    plt.close(fig)
    return paths[0]


def plot_seed_profiles(df: pd.DataFrame, *, h: float, vs1: float, out_dir: Path) -> Path:
    out_dir = out_dir / "seed_profiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    color = metric_color(METRIC)
    log_y = METRIC in LOG_Y_METRICS
    df_hv = df[(df["Height"] == h) & (df["Vs1"] == vs1)]

    legend_handles = [
        Line2D([0], [0], color=color, ls="-", lw=DATA_LINEWIDTH, label="Geomean across nodes"),
        Line2D([0], [0], color=color, ls="--", lw=DATA_LINEWIDTH, label="Median across nodes"),
    ]
    fig, axes = _make_3x3_figure(
        h=h,
        vs1=vs1,
        legend_handles=legend_handles,
    )

    cell_data: list[tuple] = []
    for rh, cov, ahv in PANELS:
        nodes, seeds, arr = cell_matrix(df_hv, rh, cov, ahv)
        geo, med = seed_profiles(arr)
        cell_data.append((seeds, geo, med))

    for i, ((rh, cov, ahv), (seeds, geo, med)) in enumerate(zip(PANELS, cell_data)):
        ax = axes.flat[i]
        ax.plot(seeds, geo, color=color, ls="-", lw=DATA_LINEWIDTH, zorder=4)
        ax.plot(seeds, med, color=color, ls="--", lw=DATA_LINEWIDTH, zorder=4)
        _annotate_panel(ax, i, rh, cov, ahv)

        row, col = divmod(i, 3)
        if row == 2:
            ax.set_xlabel("Seed", fontsize=LABEL_FONTSIZE)
        else:
            ax.tick_params(labelbottom=False)
        if col == 0:
            ax.set_ylabel(metric_label(METRIC), fontsize=LABEL_FONTSIZE)
        else:
            ax.tick_params(labelleft=False)

    _apply_shared_ylims(axes, log_y=log_y)

    paths = save_figure(fig, _stem("seed_profile", h, vs1), out_dir=out_dir)
    plt.close(fig)
    return paths[0]


def main() -> None:
    out_dir = figure_dir("chi_variables", "central_profiles")
    print(f"Loading {DATA_PATH} …")
    df = load_ratios()
    print(f"  rows={len(df):,}  metric={METRIC}")

    for h in H_LIST:
        for vs1 in VS1_LIST:
            print(f"  H={h:.0f}, Vs1={vs1:.0f} …")
            p1 = plot_node_profiles(df, h=h, vs1=vs1, out_dir=out_dir)
            # p2 = plot_seed_profiles(df, h=h, vs1=vs1, out_dir=out_dir)
            print(f"    {p1.name}")
            # print(f"    {p2.name}")

    print(f"Done → {out_dir}")


if __name__ == "__main__":
    main()
