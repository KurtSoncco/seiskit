"""Two-panel Ricker wavelet figure for the full paper.

Panel (a): time-domain pulse with a broken x-axis (0–1 s | 14–15 s).
Panel (b): unnormalized Fourier amplitude spectrum on log-log axes.

Produces: ricker_wave.pdf under
``complete/full_paper/figures/descriptions/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (  # noqa: E402
    DATA_LINEWIDTH,
    LABEL_FONTSIZE,
    TICK_LABELSIZE,
    add_panel_label,
    apply_full_paper_style,
    figsize,
    figure_dir,
    save_figure,
)

from seiskit.utils import compute_ricker

apply_full_paper_style(auto_format=True, frame="open", grid=False)

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
OUT_DIR = figure_dir("descriptions")

FREQ = 3.0
T_SHIFT = 0.5
DURATION = 15.0
DT = 0.001

# Shorter aspect than default 4:3 — time/FAS pair is wide and short
FIG_HEIGHT = figsize(aspect=0.42)[1]
LINEWIDTH = DATA_LINEWIDTH
GRID_KW = dict(color="0.75", linestyle=":", linewidth=0.4, alpha=0.7)


def _generate_signal() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return time, Ricker pulse, frequency, and |rfft| amplitude."""
    t = np.arange(0.0, DURATION + DT, DT)
    v = compute_ricker(FREQ, T_SHIFT, DURATION, DT)
    fas = np.abs(np.fft.rfft(v))
    f = np.fft.rfftfreq(len(v), DT)
    return t, v, f, fas


def _draw_axis_break(fig: plt.Figure, ax_left: plt.Axes, ax_right: plt.Axes) -> None:
    """``---//---`` break marks at bottom, middle, and top of the cut.

    The ``//`` slashes are a fixed-size marker centered in the gap; the
    horizontal dashes are drawn in figure coordinates from the right edge of
    *ax_left* to the left edge of *ax_right*, so they always span the gap
    regardless of the gridspec spacing.
    """
    slash_ms = 7  # marker size (points) of the // symbol
    verts = [
        (-0.9, -1.0),
        (0.1, 1.0),  # first diagonal slash
        (-0.1, -1.0),
        (0.9, 1.0),  # second diagonal slash
    ]
    codes = [
        mpath.Path.MOVETO,
        mpath.Path.LINETO,
        mpath.Path.MOVETO,
        mpath.Path.LINETO,
    ]
    slash_marker = mpath.Path(verts, codes)

    pos_l = ax_left.get_position()
    pos_r = ax_right.get_position()
    x0, x1 = pos_l.x1, pos_r.x0  # gap edges in figure coords
    x_mid = 0.5 * (x0 + x1)
    ys = [pos_l.y0, 0.363 * (pos_l.y0 + pos_l.y1)]

    # Clearance so the dashes stop just short of the slashes
    fig_w_pts = fig.get_size_inches()[0] * 72.0
    half_gap = 0.9 * slash_ms / fig_w_pts

    fig.add_artist(
        Line2D(
            [x_mid] * len(ys),
            ys,
            transform=fig.transFigure,
            marker=slash_marker,
            markersize=slash_ms,
            linestyle="none",
            color="k",
            mec="k",
            mew=DATA_LINEWIDTH,
            mfc="none",
            clip_on=False,
        )
    )
    dash_kw = dict(
        transform=fig.transFigure, color="k", lw=DATA_LINEWIDTH, clip_on=False
    )  # Definition of the  horizontal dash line
    dash_length = 0.4  # You control the length of the lines here
    for y in ys:
        fig.add_artist(
            Line2D([x0, x_mid - half_gap * dash_length], [y, y], **dash_kw)
        )  # You control the length of the lines here
        fig.add_artist(
            Line2D([x_mid + half_gap * dash_length, x1], [y, y], **dash_kw)
        )  # You control the length of the lines here


def _style_time_axes(ax_left: plt.Axes, ax_right: plt.Axes) -> None:
    """Open-frame styling (left+bottom spines) and dotted grid for panel (a)."""
    for ax in (ax_left, ax_right):
        ax.set_ylim(-0.50, 1.25)
        ax.set_yticks(np.arange(-0.50, 1.26, 0.25))
        ax.tick_params(axis="both", which="both", labelsize=TICK_LABELSIZE)
        ax.grid(True, which="major", **GRID_KW)
        ax.set_axisbelow(True)

    ax_left.set_xlim(0.0, 1.05)
    ax_left.set_xticks([0, 1])

    ax_right.set_xlim(14.0, 15.0)
    ax_right.set_xticks([14, 15])
    # The facing edge of the right segment carries no spine or ticks
    ax_right.spines["left"].set_visible(False)
    ax_right.tick_params(axis="y", which="both", left=False, labelleft=False)


def _style_fas_axis(ax: plt.Axes) -> None:
    """Open-frame log-log styling with major/minor dotted grid for panel (b)."""
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-1, 1e1)
    ax.set_ylim(1e-2, 1e3)

    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.tick_params(axis="both", which="both", labelsize=TICK_LABELSIZE)
    ax.grid(True, which="major", **GRID_KW)
    ax.grid(True, which="minor", **GRID_KW)
    ax.set_axisbelow(True)


def plot_ricker_wave(t: np.ndarray, v: np.ndarray, f: np.ndarray, fas: np.ndarray) -> plt.Figure:
    """Build the side-by-side time / FAS figure."""
    fig = plt.figure(figsize=figsize(height=FIG_HEIGHT))
    # Outer grid separates panel (a) from panel (b); the broken time axes
    # live in a tight nested grid so only they sit close together.
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.15, 1.0],
        wspace=0.35,
        left=0.09,
        right=0.97,
        top=0.92,
        bottom=0.20,
    )
    gs_time = outer[0, 0].subgridspec(1, 2, width_ratios=[2.4, 1.0], wspace=0.17)

    ax_left = fig.add_subplot(gs_time[0, 0])
    ax_right = fig.add_subplot(gs_time[0, 1], sharey=ax_left)
    ax_fas = fig.add_subplot(outer[0, 1])

    ax_left.plot(t, v, color="k", linewidth=LINEWIDTH, solid_capstyle="round")
    ax_right.plot(t, v, color="k", linewidth=LINEWIDTH, solid_capstyle="round")
    _style_time_axes(ax_left, ax_right)
    _draw_axis_break(fig, ax_left, ax_right)

    ax_left.set_ylabel(r"$v(t)$ (m/s)", fontsize=LABEL_FONTSIZE)
    # Shared x-label centered under the broken-axis pair
    pos_l = ax_left.get_position()
    pos_r = ax_right.get_position()
    fig.text(
        0.5 * (pos_l.x0 + pos_r.x1),
        0.064,
        "Time (s)",
        ha="center",
        va="bottom",
        fontsize=LABEL_FONTSIZE,
        transform=fig.transFigure,
    )

    # Plot beyond the displayed band; xlim clips so the curve meets the frame
    mask = (f > 0.0) & (f <= 12.0)
    ax_fas.plot(f[mask], fas[mask], color="k", linewidth=LINEWIDTH, solid_capstyle="round")
    _style_fas_axis(ax_fas)
    ax_fas.set_xlabel("Frequency (Hz)", fontsize=LABEL_FONTSIZE)
    ax_fas.set_ylabel("Fourier amplitude (g-s)", fontsize=LABEL_FONTSIZE)

    add_panel_label(ax_left, 0)
    add_panel_label(ax_fas, 1)

    return fig


def main() -> None:
    t, v, f, fas = _generate_signal()
    fig = plot_ricker_wave(t, v, f, fas)
    save_figure(fig, "ricker_wave", out_dir=OUT_DIR)
    plt.close(fig)


if __name__ == "__main__":
    main()
