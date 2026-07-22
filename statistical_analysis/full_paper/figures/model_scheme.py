"""Two-panel model-scheme figure for a representative Vs realization.

Panel (a): outside-domain free-field BC strips + domain flanks + main window,
with dimension arrows for Lx,var and L_array. Panel (b): zoom with spike
geophones (center-only or full array).

Paper targets:
  - Full paper → array receivers, DOC_WIDTH = 7 in
  - Conference paper → center receiver, DOC_WIDTH = 6.5 in

Produces: model_scheme_{center,array}.{png,pdf,svg} under Box complete/figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import h5py
import hdf5plugin  # noqa: F401 — registers Blosc2 filter
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox

from seiskit.plot_config import (
    COLORBLIND_COLORS,
    add_subfigure_label,
    apply_style,
    get_crameri_cmap,
)

apply_style(auto_format=True, font_size=10, frame="open")

mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["pdf.fonttype"] = 42

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
H5_PATH = Path(
    "/mnt/box/GIG Lab - UC Berkeley/Projects/Statistical Analysis/h=50/results/run_4050.h5"
)
OUT_DIR = Path(
    "/mnt/box/GIG Lab - UC Berkeley/Projects/Statistical Analysis/complete/figures/model_scheme"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

ASPECT_RATIO = 0.75  # 3:2; full-paper height stays under ~170 mm at 7 in width

# Conference paper: single center geophone
CONF_DOC_WIDTH = 6.5  # inches
CONF_FIG_HEIGHT = CONF_DOC_WIDTH * ASPECT_RATIO

# Full paper: full receiver array
FULL_DOC_WIDTH = 7.0  # inches
FULL_FIG_HEIGHT = FULL_DOC_WIDTH * ASPECT_RATIO

VMIN, VMAX = 140.0, 340.0
LX_VAR = 500.0
ZOOM_HALF = 100.0
N_RECEIVERS = 101
RECEIVER_COLOR = COLORBLIND_COLORS[1]
TICK_LABELSIZE = 8
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
    "full": (FULL_DOC_WIDTH, FULL_FIG_HEIGHT),
    "conference": (CONF_DOC_WIDTH, CONF_FIG_HEIGHT),
}


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
            linewidth=1.1,
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
            linewidth=1.1,
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
        fontsize=7,
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
) -> plt.Figure:
    """Build the two-panel model-scheme figure."""
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
        ax_ff.tick_params(labelsize=TICK_LABELSIZE)
        _no_xaxis(ax_ff)

    ax_ff_l.set_yticks(yticks)
    ax_ff_l.set_ylabel("Depth (m)", fontsize=9)
    ax_ff_r.tick_params(left=False, labelleft=False)

    # --- Domain flanks + main ---
    domain_axes = (ax_fl, ax_main, ax_fr)
    domain_wins = (WIN_LEFT, WIN_CENTER, WIN_RIGHT)
    for ax, (xa, xb) in zip(domain_axes, domain_wins):
        ax.imshow(vs, extent=full_extent, **imshow_kw)
        ax.set_xlim(xa, xb)
        ax.set_ylim(lz, 0.0)
        ax.grid(False)
        ax.tick_params(labelsize=TICK_LABELSIZE)
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
        r"$L_{x,\mathrm{var}} = 500\,\mathrm{m}$",
        color="black",
    )

    _dimension_arrow(
        ax_main,
        -ZOOM_HALF,
        ZOOM_HALF,
        58.0,
        r"$L_{\mathrm{ROI}} = 200\,\mathrm{m}$",
        color="red",
    )

    # Let's add "See (b). inside the red box"
    ax_main.text(
        -ZOOM_HALF + 5,
        2,  # Same height as subfigure label
        "See (b)",
        ha="left",
        va="top",
        color="red",
        fontsize=9,
        transform=ax_main.transData,
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.7,
            "pad": 0.8,
        },
    )

    add_subfigure_label(ax_main, 0, x=0.02)

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
    ax_b.tick_params(labelsize=TICK_LABELSIZE)
    _xaxis_on_top(ax_b)
    ax_b.set_xticks([-100, -50, 0, 50, 100])
    ax_b.set_ylabel("Depth (m)", fontsize=9)

    if receivers == "center":
        rx = np.array([0.0])
    else:
        rx = np.linspace(-ZOOM_HALF, ZOOM_HALF, N_RECEIVERS)
    _draw_spike_geophones(ax_b, rx)
    add_subfigure_label(ax_b, 1, x=0.02)

    fig.text(
        0.5 * (pos_fl.x0 + pos_fr.x1),
        0.95,
        "Distance from center (m)",
        ha="center",
        va="top",
        fontsize=9,
        transform=fig.transFigure,
    )

    cbar = fig.colorbar(im, cax=cax, extend="max")
    cbar.set_label(r"$V_{s1}$ (m/s)", fontsize=9)
    cbar.set_ticks(np.arange(VMIN, VMAX + 1, 40))
    cbar.ax.tick_params(labelsize=TICK_LABELSIZE)

    # Robust width match: pin panel (b) to the outer edges of the FF strips
    # (same total width as the 5-panel top row, including wspace gaps).
    fig.canvas.draw()
    bb_l = ax_ff_l.get_position()
    bb_r = ax_ff_r.get_position()
    bb_b = ax_b.get_position()
    ax_b.set_position([bb_l.x0, bb_b.y0, bb_r.x1 - bb_l.x0, bb_b.height])

    return fig


def _export(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    w, h = fig.get_size_inches()
    bbox = Bbox.from_bounds(0, 0, w, h)
    for ext in ("svg", "pdf", "png"):
        path = OUT_DIR / f"{stem}.{ext}"
        kw: dict = {"bbox_inches": bbox}
        if ext == "png":
            kw["dpi"] = 300
        fig.savefig(path, **kw)
        print(f"Wrote {path} ({w:.2f} x {h:.2f} in)")


def main() -> None:
    vs, lx, lz, dx, _dz = load_vs(H5_PATH)

    # Full paper → array; conference paper → center (with matching page widths).
    for paper in ("full", "conference"):
        mode = PAPER_RECEIVERS[paper]
        fig = plot_model_scheme(vs, lx, lz, dx, receivers=mode, figsize=PAPER_SIZE[paper])
        _export(fig, f"model_scheme_{mode}")
        plt.close(fig)


if __name__ == "__main__":
    main()
