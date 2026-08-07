"""Two-panel model-scheme figure for a representative Vs realization.

Panel (a): outside-domain free-field BC strips + domain flanks + main window,
with dimension arrows for Lx,var and L_array. Panel (b): zoom with spike
geophones (center-only or full array).

Paper targets:
  - Full paper → array receivers (FIG_WIDTH = 7 in, 600 dpi)
  - Conference paper → center receiver (6.5 in, 600 dpi)

Produces: ``model_scheme_{center,array}.{pdf,svg}`` under
``complete/full_paper/figures/model_scheme/``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import h5py
import hdf5plugin  # noqa: F401 — registers Blosc2 filter
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    ANNOTATION_FONTSIZE,
    DATA_LINEWIDTH,
    FIG_WIDTH,
    LABEL_FONTSIZE,
    PANEL_LABEL_SIZE,
    TICK_LABELSIZE,
    add_panel_label,
    apply_full_paper_style,
    figsize,
    figure_dir,
    save_figure,
)

from seiskit.plot_config import COLORBLIND_COLORS, get_crameri_cmap

apply_full_paper_style(auto_format=True, frame="open", grid=False)

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
H5_PATH = Path(
    "/mnt/box/GIG Lab - UC Berkeley/Projects/Statistical Analysis/h=50/results/run_4050.h5"
)
OUT_DIR = figure_dir("model_scheme")

# Conference paper column width; both targets use 600 dpi for embedded imshow.
CONF_FIG_WIDTH = 6.5
SAVE_DPI = 600
CONF_FIGSIZE = (CONF_FIG_WIDTH, CONF_FIG_WIDTH * 0.75)
EXPORT_FORMATS = ("pdf", "svg")

# Conference paper isn't Nature-constrained; all text there is a flat 10 pt
# (overrides the full paper's LABEL_FONTSIZE/TICK_LABELSIZE/ANNOTATION_FONTSIZE/
# PANEL_LABEL_SIZE, which stay 7/6/6/8 pt for the "full" target).
CONFERENCE_FONTSIZE = 10

VMIN, VMAX = 140.0, 340.0
LX_VAR = 500.0
ZOOM_HALF = 100.0
N_RECEIVERS = 101
RECEIVER_COLOR = COLORBLIND_COLORS[1]
SPIKE_DEPTH = 1.0  # m — shaft length into the ground
SPIKE_MS = 10  # circle markersize

# Domain windows (FF strips are outside the domain)
WIN_LEFT = (-750.0, -700.0)
WIN_CENTER = (-300.0, 300.0)
WIN_RIGHT = (700.0, 750.0)
# Display width ratios: FF strips are 1-column data but widened for visibility
FF_WIDTH = 10.0
FLANK_WIDTH = WIN_LEFT[1] - WIN_LEFT[0]
MAIN_WIDTH = WIN_CENTER[1] - WIN_CENTER[0]
CBAR_WIDTH = 20.0

ReceiverMode = Literal["center", "array"]
PaperTarget = Literal["conference", "full"]

# Which receiver layout each manuscript uses
PAPER_RECEIVERS: dict[PaperTarget, ReceiverMode] = {
    "full": "array",
    "conference": "center",
}
PAPER_SIZE: dict[PaperTarget, tuple[float, float]] = {
    "full": figsize(),
    "conference": CONF_FIGSIZE,
}
PAPER_DPI: dict[PaperTarget, int] = {
    "full": SAVE_DPI,
    "conference": SAVE_DPI,
}
PAPER_WIDTH: dict[PaperTarget, float] = {
    "full": FIG_WIDTH,
    "conference": CONF_FIG_WIDTH,
}

# Matches the serif/STIX stack every other statistical_analysis/conference_paper/
# script gets from seiskit.plot_config.apply_style()'s defaults — this file uses
# apply_full_paper_style()'s Nature sans-serif style instead, so the conference
# target needs an explicit override to stay visually consistent with the rest
# of that paper's figures.
CONFERENCE_SERIF_FONTS = ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"]


def _set_paper_font(paper: PaperTarget) -> None:
    """Switch the active font family to match *paper*'s figures elsewhere."""
    apply_full_paper_style(auto_format=True, frame="open", grid=False)
    if paper == "conference":
        mpl.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": CONFERENCE_SERIF_FONTS,
                "mathtext.fontset": "stix",
            }
        )


def load_vs(path: Path) -> tuple[np.ndarray, float, float, float, float]:
    """Return Vs_realization_2D and grid attrs (Lx, Lz, dx, dz)."""
    with h5py.File(path, "r") as f:
        vs = np.asarray(f["Vs_realization_2D"][:], dtype=np.float64)
        g = f["grid"].attrs
        lx = float(g["Lx"])
        lz = float(g["Lz"])
        dx = float(g["dx"])
        dz = float(g["dz"])
    return vs, lx, lz, dx, dz


def _extent(x0: float, x1: float, lz: float) -> tuple[float, float, float, float]:
    return (x0, x1, lz, 0.0)


def _xaxis_on_top(ax: plt.Axes) -> None:
    ax.xaxis.tick_top()
    ax.tick_params(
        axis="x",
        which="both",
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False,
    )
    ax.spines["top"].set_visible(True)
    ax.spines["bottom"].set_visible(False)


def _no_xaxis(ax: plt.Axes) -> None:
    """Hide x-axis entirely (free-field BC strips)."""
    ax.tick_params(
        axis="x",
        which="both",
        top=False,
        bottom=False,
        labeltop=False,
        labelbottom=False,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)


def _draw_boxes(ax: plt.Axes, lz: float) -> None:
    ax.add_patch(
        Rectangle(
            (-LX_VAR / 2, 0.0),
            LX_VAR,
            lz,
            fill=False,
            linestyle="--",
            linewidth=DATA_LINEWIDTH,
            edgecolor="black",
            zorder=5,
        )
    )

    ax.add_patch(
        Rectangle(
            (-ZOOM_HALF, 0.0),
            2 * ZOOM_HALF,
            lz,
            fill=False,
            linestyle="--",
            linewidth=DATA_LINEWIDTH,
            edgecolor="red",
            zorder=6,
        )
    )


def _dimension_arrow(
    ax: plt.Axes,
    x0: float,
    x1: float,
    z: float,
    label: str,
    *,
    color: str = "black",
    fontsize: float = ANNOTATION_FONTSIZE,
) -> None:
    """Horizontal <-> dimension with centered label."""
    ax.annotate(
        "",
        xy=(x1, z),
        xytext=(x0, z),
        arrowprops={
            "arrowstyle": "<->",
            "color": color,
            "lw": 0.9,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=8,
    )
    ax.text(
        0.5 * (x0 + x1),
        z,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=color,
        zorder=9,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 0.8},
    )


def _draw_spike_geophones(
    ax: plt.Axes, x_positions: np.ndarray, top_position: float = -1.5
) -> None:
    """Minimalist spike geophones: thin shaft + surface circle."""
    x_positions = np.asarray(x_positions, dtype=np.float64)
    for x in x_positions:
        ax.plot(
            [x, x],
            [top_position, SPIKE_DEPTH],
            color=RECEIVER_COLOR,
            linewidth=0.7,
            solid_capstyle="round",
            zorder=8,
            clip_on=False,
        )
    ax.scatter(
        x_positions,
        np.zeros_like(x_positions) + top_position,
        s=SPIKE_MS,
        c=RECEIVER_COLOR,
        edgecolors="k",
        linewidths=0.35,
        zorder=9,
        clip_on=False,
    )


def plot_model_scheme(
    vs: np.ndarray,
    lx: float,
    lz: float,
    dx: float,
    *,
    receivers: ReceiverMode = "array",
    figsize: tuple[float, float] | None = None,
    paper: PaperTarget = "full",
) -> plt.Figure:
    """Build the two-panel model-scheme figure."""
    _set_paper_font(paper)
    if paper == "conference":
        fs_label = fs_tick = fs_annot = fs_panel = CONFERENCE_FONTSIZE
    else:
        fs_label, fs_tick, fs_annot, fs_panel = (
            LABEL_FONTSIZE,
            TICK_LABELSIZE,
            ANNOTATION_FONTSIZE,
            PANEL_LABEL_SIZE,
        )

    _nz, nx = vs.shape
    x0 = -lx / 2.0
    x1 = x0 + nx * dx
    full_extent = _extent(x0, x1, lz)

    cmap = get_crameri_cmap("navia", reverse=False).copy()
    cmap.set_over("gray")

    if figsize is None:
        figsize = PAPER_SIZE["full" if receivers == "array" else "conference"]
    fig = plt.figure(figsize=figsize)
    # Top: FF | flank | main | flank | FF | cbar
    # Bottom: panel (b) spans domain columns (1:4), cbar shared
    gs = fig.add_gridspec(
        2,
        6,
        width_ratios=[
            FF_WIDTH,
            FLANK_WIDTH,
            MAIN_WIDTH,
            FLANK_WIDTH,
            FF_WIDTH,
            CBAR_WIDTH,
        ],
        height_ratios=[1.0, 1.0],
        wspace=0.22,
        hspace=0.38,
        left=0.08,
        right=0.90,
        top=0.88,
        bottom=0.04,
    )

    ax_ff_l = fig.add_subplot(gs[0, 0])
    ax_fl = fig.add_subplot(gs[0, 1], sharey=ax_ff_l)
    ax_main = fig.add_subplot(gs[0, 2], sharey=ax_ff_l)
    ax_fr = fig.add_subplot(gs[0, 3], sharey=ax_ff_l)
    ax_ff_r = fig.add_subplot(gs[0, 4], sharey=ax_ff_l)
    # Span all five content columns so outer width matches the top row
    # (FF + flanks + main + wspace gaps). xlim still [-100, 100].
    ax_b = fig.add_subplot(gs[1, 0:5])
    cax = fig.add_subplot(gs[:, 5])

    imshow_kw = dict(
        cmap=cmap,
        vmin=VMIN,
        vmax=VMAX,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
    )
    yticks = np.arange(0, 61, 10)

    # --- Free-field BC strips (outside domain) ---
    ff_l = vs[:, 0:1]
    ff_r = vs[:, -1:]
    ax_ff_l.imshow(ff_l, extent=_extent(0.0, dx, lz), **imshow_kw)
    ax_ff_r.imshow(ff_r, extent=_extent(0.0, dx, lz), **imshow_kw)
    for ax_ff in (ax_ff_l, ax_ff_r):
        ax_ff.set_xlim(0.0, dx)
        ax_ff.set_ylim(lz, 0.0)
        ax_ff.grid(False)
        ax_ff.tick_params(labelsize=fs_tick)
        _no_xaxis(ax_ff)

    ax_ff_l.set_yticks(yticks)
    ax_ff_l.set_ylabel("Depth (m)", fontsize=fs_label)
    ax_ff_r.tick_params(left=False, labelleft=False)

    # --- Domain flanks + main ---
    domain_axes = (ax_fl, ax_main, ax_fr)
    domain_wins = (WIN_LEFT, WIN_CENTER, WIN_RIGHT)
    for ax, (xa, xb) in zip(domain_axes, domain_wins):
        ax.imshow(vs, extent=full_extent, **imshow_kw)
        ax.set_xlim(xa, xb)
        ax.set_ylim(lz, 0.0)
        ax.grid(False)
        ax.tick_params(labelsize=fs_tick)
        _xaxis_on_top(ax)

    ax_fl.set_xticks([-750])
    ax_fr.set_xticks([750])
    ax_main.set_xticks([-300, 0, 300])

    # Open seams between domain panels
    ax_fl.spines["left"].set_visible(False)
    ax_fl.spines["right"].set_visible(False)
    ax_main.spines["left"].set_visible(False)
    ax_main.spines["right"].set_visible(False)
    ax_fr.spines["left"].set_visible(False)
    ax_fl.tick_params(left=False, labelleft=False)
    ax_main.tick_params(left=False, labelleft=False)
    ax_fr.tick_params(left=False, labelleft=False)
    ax_ff_l.spines["right"].set_visible(False)
    ax_ff_r.spines["left"].set_visible(False)

    _draw_boxes(ax_main, lz)
    _dimension_arrow(
        ax_main,
        -LX_VAR / 2,
        LX_VAR / 2,
        52.0,
        r"$500\,\mathrm{m}$",
        color="black",
        fontsize=fs_annot,
    )

    # Let's add "See (b). inside the red box"
    ax_main.text(
        -ZOOM_HALF + 5,
        2,  # Same height as subfigure label
        "See (b)",
        ha="left",
        va="top",
        color="red",
        fontsize=fs_label,
        transform=ax_main.transData,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.7,
            "pad": 0.8,
        },
    )

    add_panel_label(ax_main, 0, fontsize=fs_panel)

    # Full-model label centered over domain columns (flanks + main), not FF
    pos_fl = ax_fl.get_position()
    pos_fr = ax_fr.get_position()
    # fig.text(
    #     0.5 * (pos_fl.x0 + pos_fr.x1),
    #     0.96,
    #     r"Full model $L_x = 1500\,\mathrm{m}$",
    #     ha="center",
    #     va="top",
    #     fontsize=8,
    #     transform=fig.transFigure,
    # )

    # --- Panel (b) ---
    im = ax_b.imshow(vs, extent=full_extent, **imshow_kw)
    ax_b.set_xlim(-ZOOM_HALF, ZOOM_HALF)
    ax_b.set_ylim(lz, 0.0)
    ax_b.set_yticks(yticks)
    ax_b.grid(False)
    ax_b.tick_params(labelsize=fs_tick)
    _xaxis_on_top(ax_b)
    ax_b.set_xticks([-100, -50, 0, 50, 100])
    ax_b.set_ylabel("Depth (m)", fontsize=fs_label)

    if receivers == "center":
        rx = np.array([0.0])
    else:
        rx = np.linspace(-ZOOM_HALF, ZOOM_HALF, N_RECEIVERS)
    _draw_spike_geophones(ax_b, rx)
    add_panel_label(ax_b, 1, fontsize=fs_panel)

    fig.text(
        0.5 * (pos_fl.x0 + pos_fr.x1),
        0.95,
        "Distance from center (m)",
        ha="center",
        va="top",
        fontsize=fs_label,
        transform=fig.transFigure,
    )

    cbar = fig.colorbar(im, cax=cax, extend="max")
    cbar.set_label(r"$V_{s1}$ (m/s)", fontsize=fs_label)
    cbar.set_ticks(np.arange(VMIN, VMAX + 1, 40))
    cbar.ax.tick_params(labelsize=fs_tick)

    # Robust width match: pin panel (b) to the outer edges of the FF strips
    # (same total width as the 5-panel top row, including wspace gaps).
    fig.canvas.draw()
    bb_l = ax_ff_l.get_position()
    bb_r = ax_ff_r.get_position()
    bb_b = ax_b.get_position()
    ax_b.set_position([bb_l.x0, bb_b.y0, bb_r.x1 - bb_l.x0, bb_b.height])

    return fig


def main() -> None:
    vs, lx, lz, dx, _dz = load_vs(H5_PATH)

    # Full paper → array; conference paper → center (with matching page widths).
    for paper in ("full", "conference"):
        mode = PAPER_RECEIVERS[paper]
        expect_w = PAPER_WIDTH[paper]
        dpi = PAPER_DPI[paper]
        fig = plot_model_scheme(
            vs, lx, lz, dx, receivers=mode, figsize=PAPER_SIZE[paper], paper=paper
        )
        w, h = fig.get_size_inches()
        if abs(w - expect_w) > 1e-6:
            raise ValueError(f"{paper}: figsize width {w:.3f} != {expect_w}")
        paths = save_figure(
            fig,
            f"model_scheme_{mode}",
            out_dir=OUT_DIR,
            formats=EXPORT_FORMATS,
            dpi=dpi,
        )
        plt.close(fig)
        for p in paths:
            print(f"  verified canvas {w:.2f}×{h:.2f} in @ {dpi} dpi → {p.name}")


if __name__ == "__main__":
    main()
