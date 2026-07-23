"""2x2 Vs realizations across r_h and a_hv for the full paper.

Panels (a)–(d) share one seed and a soil-focused batlow color scale with
gray bedrock overshoot. Produces: vs_rh_realizations.pdf under
``complete/full_paper/figures/vs_rh_realizations/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    ANNOTATION_FONTSIZE,
    LABEL_FONTSIZE,
    TICK_LABELSIZE,
    add_panel_label,
    apply_full_paper_style,
    figsize,
    figure_dir,
    save_figure,
)

from seiskit.gaussian_field import create_vs_realization
from seiskit.plot_config import get_crameri_cmap

apply_full_paper_style(auto_format=True, frame="open", grid=False)

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
OUT_DIR = figure_dir("vs_rh_realizations")

VMIN, VMAX = 150.0, 350.0
VS1, VS2 = 230.0, 1500.0
CV = 0.2
SEED = 42

LX = 500.0
LX_VAR = 500.0
LZ = 60.0
DX = DZ = 1.0
DZ_1D = 1.0
SOIL_THICKNESS = 50.0
BEDROCK_THICKNESS = 10.0

# (rH, aHV) for panels (a)–(d)
PANEL_PARAMS: list[tuple[float, float]] = [
    (10.0, 1.0),
    (10.0, 50.0),
    (50.0, 1.0),
    (50.0, 50.0),
]


def _vs_profile() -> np.ndarray:
    n_soil = int(SOIL_THICKNESS / DZ_1D)
    n_rock = int(BEDROCK_THICKNESS / DZ_1D)
    return np.array([VS1] * n_soil + [VS2] * n_rock, dtype=np.float64)


def generate_panel_fields() -> list[np.ndarray]:
    """Generate the four Vs realizations (same seed, different rH/aHV)."""
    profile = _vs_profile()
    fields: list[np.ndarray] = []
    for rH, aHV in PANEL_PARAMS:
        vs, _x, _z, _h, _mask = create_vs_realization(
            Vs_profile=profile,
            Lx=LX,
            Lx_variability=LX_VAR,
            Lz=LZ,
            dx=DX,
            dz=DZ,
            rH=rH,
            aHV=aHV,
            CV=CV,
            seed=SEED,
            dz_1D=DZ_1D,
            interlayer_amplitude=0.0,
        )
        fields.append(vs)
    return fields


def _param_annotation(ax: plt.Axes, rH: float, aHV: float) -> None:
    """Parameter box in the upper-right corner (sentence case, ≤7 pt)."""
    ax.text(
        0.98,
        0.98,
        rf"$r_h = {rH:.0f}\,\mathrm{{m}}$" + "\n" + rf"$a_{{hv}} = {aHV:.0f}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=ANNOTATION_FONTSIZE,
        color="black",
        zorder=7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
    )


def _xaxis_on_top(ax: plt.Axes, *, label: str | None = None) -> None:
    """Place ticks and optional label on the top spine."""
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.tick_params(
        axis="x",
        which="both",
        top=True,
        bottom=False,
        labeltop=True,
        labelbottom=False,
        pad=1.5,
    )
    ax.spines["top"].set_visible(True)
    ax.spines["bottom"].set_visible(False)
    if label is not None:
        ax.set_xlabel(label)


def plot_vs_rh_realizations(fields: list[np.ndarray]) -> plt.Figure:
    """Build the 2x2 figure with a shared colorbar."""
    cmap = get_crameri_cmap("navia", reverse=False).copy()
    cmap.set_over("gray")

    extent = (-LX / 2.0, LX / 2.0, LZ, 0.0)
    imshow_kw = dict(
        cmap=cmap,
        vmin=VMIN,
        vmax=VMAX,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=extent,
    )

    fig = plt.figure(figsize=figsize())
    # Tight margins: top strip only for shared distance label + ticks
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.0, 0.04],
        height_ratios=[1.0, 1.0],
        wspace=0.15,
        hspace=0.05,
        left=0.07,
        right=0.92,
        top=0.90,
        bottom=0.04,
    )

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    cax = fig.add_subplot(gs[:, 2])

    xticks = np.arange(-250, 251, 100)
    im = None
    for i, (ax, vs, (rH, aHV)) in enumerate(zip(axes, fields, PANEL_PARAMS)):
        im = ax.imshow(vs, **imshow_kw)
        ax.set_xlim(-LX / 2.0, LX / 2.0)
        ax.set_ylim(LZ, 0.0)
        ax.set_xticks(xticks)
        ax.set_yticks(np.arange(0, 61, 20))
        ax.grid(False)

        row, col = divmod(i, 2)
        if row == 0:
            _xaxis_on_top(ax)
            ax.set_xticklabels([f"{t:g}" for t in xticks], fontsize=TICK_LABELSIZE)
        else:
            ax.tick_params(
                axis="x",
                which="both",
                bottom=False,
                top=False,
                labelbottom=False,
                labeltop=False,
            )
            ax.spines["bottom"].set_visible(False)
            ax.spines["top"].set_visible(False)

        if col == 0:
            ax.set_ylabel("Depth (m)", fontsize=LABEL_FONTSIZE, labelpad=2)
        else:
            ax.tick_params(labelleft=False)

        ax.tick_params(axis="y", labelsize=TICK_LABELSIZE, pad=1.5)

        _param_annotation(ax, rH, aHV)
        add_panel_label(ax, i)

    # Shared x label tucked just above the top-row tick labels
    pos0 = axes[0].get_position()
    pos1 = axes[1].get_position()
    fig.text(
        0.5 * (pos0.x0 + pos1.x1),
        pos0.y1 + 0.035,
        "Distance from center (m)",
        ha="center",
        va="bottom",
        fontsize=LABEL_FONTSIZE,
        transform=fig.transFigure,
        clip_on=False,
    )

    assert im is not None
    cbar = fig.colorbar(im, cax=cax, extend="max")
    cbar.set_label(r"$V_{s1}$ (m/s)", fontsize=LABEL_FONTSIZE, labelpad=4)
    cbar.set_ticks(np.arange(VMIN, VMAX + 1, 50))
    cbar.ax.tick_params(labelsize=TICK_LABELSIZE, pad=1.5)

    return fig


def main() -> None:
    fields = generate_panel_fields()
    fig = plot_vs_rh_realizations(fields)
    save_figure(fig, "vs_rh_realizations", out_dir=OUT_DIR)
    plt.close(fig)


if __name__ == "__main__":
    main()
