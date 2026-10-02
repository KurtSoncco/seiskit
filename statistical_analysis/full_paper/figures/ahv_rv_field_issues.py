"""Three figures for the a_hv effect and the two r_v failures of the FFT field.

1. ahv_realizations — same seed and r_h, three a_hv. Shows the anisotropy
   the parameter is meant to produce.
2. rv_unresolved_correlation — r_h fixed at 10 m. When r_v drops to the
   grid spacing, the horizontal correlation becomes too long.
3. rv_periodic_correlation — r_h = 50 m. When r_v is comparable to the
   vertical FFT period, centering locks the field onto a 500 m wave and
   the correlation crosses zero.

Soil thickness 50 m, 10 m bedrock, 1 m grid, V_s1 = 230 m/s, CoV = 0.2.
These are the settings of the h=50 Box runs. Curves are the ensemble
correlation of the Gaussian field inside generate_gaussian_field_fft,
which is what the stored V_s arrays follow.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    ANNOTATION_FONTSIZE,
    DATA_LINEWIDTH,
    LABEL_FONTSIZE,
    TICK_LABELSIZE,
    TOL_BRIGHT,
    add_panel_label,
    apply_full_paper_style,
    figsize,
    figure_dir,
    save_figure,
)

from seiskit.gaussian_field import create_vs_realization, generate_gaussian_field_fft
from seiskit.plot_config import get_crameri_cmap

apply_full_paper_style(auto_format=True, frame="open", grid=False)

OUT_DIR = figure_dir("ahv_rv_field")

VS1, VS2 = 230.0, 1500.0
CV = 0.2
DX = DZ = 1.0
SOIL = 50
BEDROCK = 10
NZ = SOIL + BEDROCK
NX = 500
LZ = float(NZ)
LX = float(NX)
VMIN, VMAX = 150.0, 350.0
N_ENS = 40
MAP_SEED = 7

# a_hv = 1, 10, 50
AHV_COLORS = {
    1.0: TOL_BRIGHT["blue"],
    10.0: TOL_BRIGHT["green"],
    50.0: TOL_BRIGHT["red"],
}


def _profile() -> np.ndarray:
    return np.array([VS1] * SOIL + [VS2] * BEDROCK, dtype=np.float64)


def vs_map(rh: float, ahv: float, seed: int) -> np.ndarray:
    vs, *_ = create_vs_realization(
        Vs_profile=_profile(),
        Lx=LX,
        Lx_variability=LX,
        Lz=LZ,
        dx=DX,
        dz=DZ,
        rH=rh,
        aHV=ahv,
        CV=CV,
        seed=seed,
        dz_1D=DZ,
        interlayer_amplitude=0.0,
    )
    return vs


def ensemble(rh: float, ahv: float) -> np.ndarray:
    """Soil rows of the centered Gaussian field, shape (n, soil, nx)."""
    fields = np.empty((N_ENS, SOIL, NX), dtype=np.float64)
    for i in range(N_ENS):
        g = generate_gaussian_field_fft(
            NX, NZ, DX, DZ, rh, ahv, np.random.default_rng(10_000 + i)
        )
        fields[i] = g[:SOIL]
    return fields


def acf_h(fields: np.ndarray, max_lag: int) -> np.ndarray:
    c0 = float(np.mean(fields * fields))
    rho = np.empty(max_lag + 1)
    rho[0] = 1.0
    for lag in range(1, max_lag + 1):
        rho[lag] = float(np.mean(fields[:, :, :-lag] * fields[:, :, lag:]) / c0)
    return rho


def acf_v(fields: np.ndarray, max_lag: int) -> np.ndarray:
    c0 = float(np.mean(fields * fields))
    rho = np.empty(max_lag + 1)
    rho[0] = 1.0
    for lag in range(1, max_lag + 1):
        rho[lag] = float(np.mean(fields[:, :-lag, :] * fields[:, lag:, :]) / c0)
    return rho


def _style_map(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
    ax.grid(False)
    ax.tick_params(labelsize=TICK_LABELSIZE, width=0.5, length=2.5, pad=1.5)


def _annotate(ax: plt.Axes, lines: str) -> None:
    ax.text(
        0.02,
        0.97,
        lines,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=ANNOTATION_FONTSIZE,
        color="black",
        zorder=7,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.2},
    )


def figure_realizations() -> plt.Figure:
    """Fixed r_h = 10 m. a_hv stretches the field horizontally and thins it vertically."""
    rh = 10.0
    ahvs = (1.0, 10.0, 50.0)
    fields = [vs_map(rh, ahv, MAP_SEED) for ahv in ahvs]

    cmap = get_crameri_cmap("navia", reverse=False).copy()
    cmap.set_over("0.55")
    extent = (-LX / 2, LX / 2, LZ, 0.0)

    fig = plt.figure(figsize=figsize(height=2.15))
    gs = fig.add_gridspec(
        1,
        4,
        width_ratios=[1, 1, 1, 0.045],
        wspace=0.08,
        left=0.055,
        right=0.90,
        top=0.78,
        bottom=0.18,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    im = None
    xticks = np.arange(-200, 201, 200)
    for i, (ax, vs, ahv) in enumerate(zip(axes, fields, ahvs)):
        im = ax.imshow(
            vs,
            cmap=cmap,
            vmin=VMIN,
            vmax=VMAX,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            extent=extent,
            rasterized=True,
        )
        ax.set_xlim(-LX / 2, LX / 2)
        ax.set_ylim(LZ, 0.0)
        ax.set_xticks(xticks)
        ax.set_yticks([0, 25, 50])
        _style_map(ax)
        rv = rh / ahv
        _annotate(
            ax,
            rf"$r_h = {rh:.0f}\,\mathrm{{m}}$"
            + "\n"
            + rf"$a_{{hv}} = {ahv:.0f}$"
            + "\n"
            + rf"$r_v = {rv:.1f}\,\mathrm{{m}}$",
        )
        add_panel_label(ax, i)
        if i == 0:
            ax.set_ylabel("Depth (m)", fontsize=LABEL_FONTSIZE, labelpad=2)
        else:
            ax.tick_params(labelleft=False)
        ax.set_xlabel("Distance from center (m)", fontsize=LABEL_FONTSIZE, labelpad=2)
    assert im is not None
    cbar = fig.colorbar(im, cax=cax, extend="max")
    cbar.set_label(r"Soil $V_s$ (m/s)", fontsize=LABEL_FONTSIZE, labelpad=3)
    cbar.set_ticks([150, 250, 350])
    cbar.ax.tick_params(labelsize=TICK_LABELSIZE, width=0.5, length=2.5)
    return fig


def _line_axes(ax: plt.Axes) -> None:
    ax.tick_params(labelsize=TICK_LABELSIZE, width=0.5, length=2.5, pad=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)


def figure_unresolved() -> plt.Figure:
    """r_h = 10 m. Smaller r_v makes the horizontal correlation too strong."""
    rh = 10.0
    ahvs = (1.0, 10.0, 50.0)
    max_h, max_v = 40, 20
    lags_h = np.arange(max_h + 1)
    lags_v = np.arange(max_v + 1)
    stored = {ahv: ensemble(rh, ahv) for ahv in ahvs}

    fig = plt.figure(figsize=figsize(height=2.55))
    gs = fig.add_gridspec(1, 2, wspace=0.28, left=0.08, right=0.98, top=0.92, bottom=0.20)
    ax_h = fig.add_subplot(gs[0, 0])
    ax_v = fig.add_subplot(gs[0, 1])

    ax_h.plot(
        lags_h,
        np.exp(-lags_h / rh),
        color="0.25",
        lw=DATA_LINEWIDTH,
        ls=(0, (3, 1.5)),
        zorder=1,
    )
    for ahv in ahvs:
        color = AHV_COLORS[ahv]
        rho_h = acf_h(stored[ahv], max_h)
        rho_v = acf_v(stored[ahv], max_v)
        rv = rh / ahv
        ax_h.plot(lags_h, rho_h, color=color, lw=DATA_LINEWIDTH, zorder=2)
        ax_v.plot(
            lags_v,
            np.exp(-lags_v / rv),
            color=color,
            lw=DATA_LINEWIDTH,
            ls=(0, (3, 1.5)),
            zorder=1,
        )
        ax_v.plot(lags_v, rho_v, color=color, lw=DATA_LINEWIDTH, zorder=2)

    ax_h.set_xlim(0, max_h)
    ax_v.set_xlim(0, max_v)
    ax_h.set_ylim(0, 1.05)
    ax_v.set_ylim(0, 1.05)
    ax_h.set_xlabel("Horizontal separation (m)", fontsize=LABEL_FONTSIZE)
    ax_v.set_xlabel("Vertical separation (m)", fontsize=LABEL_FONTSIZE)
    ax_h.set_ylabel("Correlation", fontsize=LABEL_FONTSIZE)
    _line_axes(ax_h)
    _line_axes(ax_v)
    add_panel_label(ax_h, 0)
    add_panel_label(ax_v, 1)

    handles = [
        Line2D([0], [0], color=AHV_COLORS[ahv], lw=DATA_LINEWIDTH, label=rf"$a_{{hv}} = {ahv:.0f}$")
        for ahv in ahvs
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color="0.25",
            lw=DATA_LINEWIDTH,
            ls=(0, (3, 1.5)),
            label="Exponential",
        )
    )
    ax_h.legend(handles=handles, loc="lower right", fontsize=ANNOTATION_FONTSIZE, handlelength=2.2)
    ax_h.text(
        0.03,
        0.55,
        rf"$r_h = {rh:.0f}\,\mathrm{{m}}$",
        transform=ax_h.transAxes,
        fontsize=ANNOTATION_FONTSIZE,
        va="center",
    )
    ax_v.text(
        0.97,
        0.08,
        r"Dashed: $\exp(-z/r_v)$",
        transform=ax_v.transAxes,
        fontsize=ANNOTATION_FONTSIZE,
        ha="right",
        va="bottom",
    )
    return fig


def figure_periodic() -> plt.Figure:
    """r_h = 50 m. a_hv = 1 puts r_v on the FFT period and the correlation changes sign."""
    rh = 50.0
    # Pick a seed whose soil has a clear left-right contrast, so the 500 m
    # mode is visible. Half the variance sits in that mode on average.
    best_seed, best_score = MAP_SEED, -1.0
    best_vs = vs_map(rh, 1.0, MAP_SEED)
    for seed in range(1, 25):
        vs = vs_map(rh, 1.0, seed)
        soil = vs[:SOIL]
        score = abs(float(soil[:, : NX // 2].mean() - soil[:, NX // 2 :].mean()))
        if score > best_score:
            best_seed, best_score, best_vs = seed, score, vs

    cmap = get_crameri_cmap("navia", reverse=False).copy()
    cmap.set_over("0.55")
    extent = (-LX / 2, LX / 2, LZ, 0.0)

    fig = plt.figure(figsize=figsize(height=3.55))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[0.85, 1.15],
        width_ratios=[1, 0.035],
        hspace=0.38,
        wspace=0.05,
        left=0.09,
        right=0.92,
        top=0.94,
        bottom=0.12,
    )
    ax_map = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])
    ax = fig.add_subplot(gs[1, 0])

    im = ax_map.imshow(
        best_vs,
        cmap=cmap,
        vmin=VMIN,
        vmax=VMAX,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=extent,
        rasterized=True,
    )
    ax_map.set_xlim(-LX / 2, LX / 2)
    ax_map.set_ylim(LZ, 0.0)
    ax_map.axvline(-100, color="black", lw=0.7, ls=(0, (2, 1.2)), zorder=3)
    ax_map.axvline(100, color="black", lw=0.7, ls=(0, (2, 1.2)), zorder=3)
    ax_map.text(
        0,
        3.5,
        "recorders",
        ha="center",
        va="top",
        fontsize=ANNOTATION_FONTSIZE,
        color="black",
        zorder=4,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.6},
    )
    ax_map.set_xticks(np.arange(-200, 201, 100))
    ax_map.set_yticks([0, 25, 50])
    _style_map(ax_map)
    ax_map.set_ylabel("Depth (m)", fontsize=LABEL_FONTSIZE, labelpad=2)
    ax_map.set_xlabel("Distance from center (m)", fontsize=LABEL_FONTSIZE, labelpad=2)
    _annotate(
        ax_map,
        rf"$r_h = {rh:.0f}\,\mathrm{{m}}$"
        + "\n"
        + rf"$a_{{hv}} = 1$"
        + "\n"
        + rf"$r_v = {rh:.0f}\,\mathrm{{m}}$",
    )
    add_panel_label(ax_map, 0, x=0.98, y=0.06)
    cbar = fig.colorbar(im, cax=cax, extend="max")
    cbar.set_label(r"Soil $V_s$ (m/s)", fontsize=LABEL_FONTSIZE, labelpad=3)
    cbar.set_ticks([150, 250, 350])
    cbar.ax.tick_params(labelsize=TICK_LABELSIZE, width=0.5, length=2.5)

    max_lag = 250
    lags = np.arange(max_lag + 1)
    cases = ((1.0, AHV_COLORS[1.0]), (10.0, AHV_COLORS[10.0]))
    for ahv, color in cases:
        rho = acf_h(ensemble(rh, ahv), max_lag)
        ax.plot(lags, rho, color=color, lw=DATA_LINEWIDTH, zorder=2)
    ax.plot(
        lags,
        np.exp(-lags / rh),
        color="0.25",
        lw=DATA_LINEWIDTH,
        ls=(0, (3, 1.5)),
        zorder=1,
    )
    ax.axhline(0.0, color="0.6", lw=0.5, zorder=0)
    ax.set_xlim(0, max_lag)
    ax.set_ylim(-0.35, 1.05)
    ax.set_xlabel("Horizontal separation (m)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Correlation", fontsize=LABEL_FONTSIZE)
    _line_axes(ax)
    add_panel_label(ax, 1)
    handles = [
        Line2D(
            [0],
            [0],
            color=AHV_COLORS[1.0],
            lw=DATA_LINEWIDTH,
            label=r"$a_{hv} = 1$ ($r_v = 50$ m)",
        ),
        Line2D(
            [0],
            [0],
            color=AHV_COLORS[10.0],
            lw=DATA_LINEWIDTH,
            label=r"$a_{hv} = 10$ ($r_v = 5$ m)",
        ),
        Line2D(
            [0],
            [0],
            color="0.25",
            lw=DATA_LINEWIDTH,
            ls=(0, (3, 1.5)),
            label=r"Exponential, $r_h = 50$ m",
        ),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.84),
        fontsize=ANNOTATION_FONTSIZE,
        handlelength=2.4,
    )
    # best_seed is chosen only so the map shows the mode; the curve is an ensemble.
    del best_seed
    return fig


def main() -> None:
    figs = {
        "ahv_realizations": figure_realizations(),
        "rv_unresolved_correlation": figure_unresolved(),
        "rv_periodic_correlation": figure_periodic(),
    }
    for stem, fig in figs.items():
        save_figure(fig, stem, out_dir=OUT_DIR)
        png = OUT_DIR / f"{stem}.png"
        fig.savefig(png, dpi=160)
        print(f"Wrote {png}")
        plt.close(fig)


if __name__ == "__main__":
    main()
