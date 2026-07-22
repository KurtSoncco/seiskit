"""3x3 center-node |TF| figure: all seeds vs geomean±1σ_ln vs 1D baseline.

H = 50 m, Vs1 = 230 m/s. Rows vary r_h, CoV, and a_hv; the center column is
the shared baseline cell (r_h=30, CoV=0.2, a_hv=10).

Produces:
  complete/full_paper/figures/qualitative/
    tf_raw_3x3_center_node_all_seeds_h50_vs1_230_node_50.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (  # noqa: E402
    DATA_LINEWIDTH,
    FIG_DPI,
    LABEL_FONTSIZE,
    TICK_LABELSIZE,
    add_panel_label,
    apply_full_paper_style,
    figsize,
    figure_dir,
)

apply_full_paper_style(auto_format=True, frame="open", grid=False)

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
BOX = Path("/mnt/box/GIG Lab - UC Berkeley/Projects/Statistical Analysis")
TF_DIR = BOX / "h=50" / "transfer_function_results"
BASE_1D = BOX / "h=50" / "base_cases" / "base_case_tf_Vs1230.npz"
OUT_DIR = figure_dir("qualitative")
OUT_STEM = "tf_raw_3x3_center_node_all_seeds_h50_vs1_230_node_50"

H = 50.0
VS1 = 230.0
CENTER_CH = 50
N_SEEDS = 100

VS1_LIST = [100.0, 230.0, 360.0]
COV_LIST = [0.1, 0.2, 0.3]
RH_LIST = [10.0, 30.0, 50.0]
AHV_LIST = [1.0, 10.0, 50.0]

# (rH, CoV, aHV) for panels (a)–(i), row-major
PANELS: list[tuple[float, float, float]] = [
    # Top: vary r_h (CoV=0.2, aHV=10)
    (10.0, 0.2, 10.0),
    (30.0, 0.2, 10.0),
    (50.0, 0.2, 10.0),
    # Mid: vary CoV (rH=30, aHV=10)
    (30.0, 0.1, 10.0),
    (30.0, 0.2, 10.0),
    (30.0, 0.3, 10.0),
    # Bot: vary a_hv (rH=30, CoV=0.2)
    (30.0, 0.2, 1.0),
    (30.0, 0.2, 10.0),
    (30.0, 0.2, 50.0),
]

DOC_WIDTH, FIG_HEIGHT = figsize(aspect=0.95)

FREQ_LIM = (1e-1, 1e1)
TF_LIM = (1e-1, 1e3)

# Wong / Nature colorblind palette — also separable in B&W via linestyle + fill
COLOR_SAMPLES = "0.55"
COLOR_GEO = "#D55E00"  # Wong vermillion
COLOR_1D = "#000000"  # black baseline: strongest B&W cue; dotted linestyle


def cell_start(vs1: float, cov: float, rh: float, ahv: float) -> int:
    """Index of the first seed for a factorial cell (order: Vs1→CoV→rH→aHV→seed)."""
    i = VS1_LIST.index(vs1)
    j = COV_LIST.index(cov)
    k = RH_LIST.index(rh)
    m = AHV_LIST.index(ahv)
    return (((i * len(COV_LIST) + j) * len(RH_LIST) + k) * len(AHV_LIST) + m) * N_SEEDS


def _panel_param_text(rh: float, cov: float, ahv: float) -> str:
    return (
        rf"$r_h = {rh:.0f}$ m" + "\n"
        rf"$\mathrm{{CoV}} = {cov:g}$" + "\n"
        rf"$a_{{hv}} = {ahv:.0f}$"
    )


def _geomean_band(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return geomean, lo, hi for log|TF| ±1σ across seeds."""
    log_tf = np.log(np.clip(stack, 1e-12, None))
    geo = np.exp(np.mean(log_tf, axis=0))
    sigma_ln = np.std(log_tf, axis=0, ddof=1)
    lo = geo * np.exp(-sigma_ln)
    hi = geo * np.exp(sigma_ln)
    return geo, lo, hi


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load freq, memmapped 2D TF, and 1D baseline (freq, |TF|)."""
    freq = np.asarray(np.load(TF_DIR / "freq.npz")["freq"], dtype=np.float64)
    tf_all = np.load(TF_DIR / "tf_per_sample.npy", mmap_mode="r")
    base = np.load(BASE_1D)
    freq_1d = np.asarray(base["freq"], dtype=np.float64)
    tf_1d = np.asarray(base["tf_magnitude"], dtype=np.float64)
    return freq, tf_all, freq_1d, tf_1d


def plot_tf_panel(
    ax: plt.Axes,
    freq: np.ndarray,
    stack: np.ndarray,
    freq_1d: np.ndarray,
    tf_1d: np.ndarray,
) -> None:
    """Draw 2D samples, geomean±1σ (log|TF|), and 1D baseline on *ax*."""
    geo, lo, hi = _geomean_band(stack)

    # 2D samples — light gray solid (B&W: dense bundle)
    for seed_tf in stack:
        ax.plot(
            freq,
            seed_tf,
            color=COLOR_SAMPLES,
            lw=0.35,
            alpha=0.30,
            zorder=1,
        )

    # Geomean ±1σ: fill is the B&W-readable range cue; dashed line for geomean
    ax.fill_between(
        freq,
        lo,
        hi,
        facecolor=COLOR_GEO,
        alpha=0.28,
        edgecolor="none",
        zorder=3,
        label="_nolegend_",
    )
    ax.plot(freq, geo, color=COLOR_GEO, ls="--", lw=DATA_LINEWIDTH, zorder=4)
    ax.plot(freq, lo, color=COLOR_GEO, ls="--", lw=0.6, alpha=0.65, zorder=4)
    ax.plot(freq, hi, color=COLOR_GEO, ls="--", lw=0.6, alpha=0.65, zorder=4)

    # 1D baseline — black dotted (distinct in B&W without relying on hue)
    ax.plot(freq_1d, tf_1d, color=COLOR_1D, lw=DATA_LINEWIDTH, zorder=2, dashes=(1, 1.5))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*FREQ_LIM)
    ax.set_ylim(*TF_LIM)
    ax.grid(True, which="major", alpha=0.18, lw=0.6)
    ax.grid(True, which="minor", alpha=0.08, lw=0.4)


def plot_figure(
    freq: np.ndarray,
    tf_all: np.ndarray,
    freq_1d: np.ndarray,
    tf_1d: np.ndarray,
) -> plt.Figure:
    """Build the 3×3 log–log |TF| figure."""
    fig = plt.figure(figsize=(DOC_WIDTH, FIG_HEIGHT))
    # Header row for title+legend, then 3×3 data panels (maximizes panel area)
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.12, 1.0],
        hspace=0.04,
        left=0.09,
        right=0.99,
        bottom=0.07,
        top=0.98,
    )
    header = fig.add_subplot(gs[0, 0])
    header.axis("off")
    gs_panels = gs[1, 0].subgridspec(3, 3, wspace=0.10, hspace=0.16)
    axes = np.empty((3, 3), dtype=object)
    for r in range(3):
        for c in range(3):
            sharex = axes[0, 0] if (r, c) != (0, 0) else None
            sharey = axes[0, 0] if (r, c) != (0, 0) else None
            axes[r, c] = fig.add_subplot(gs_panels[r, c], sharex=sharex, sharey=sharey)

    legend_handles: list | None = None

    for i, (rh, cov, ahv) in enumerate(PANELS):
        ax = axes.flat[i]
        i0 = cell_start(VS1, cov, rh, ahv)
        stack = np.asarray(tf_all[i0 : i0 + N_SEEDS, CENTER_CH, :], dtype=np.float64)
        plot_tf_panel(ax, freq, stack, freq_1d, tf_1d)

        # Subfigure letter: top-left (Nature: 8 pt bold lowercase)
        add_panel_label(ax, i)

        # Parameters: top-right (mirrors vs_rh_realizations annotation placement)
        ax.text(
            0.98,
            0.97,
            _panel_param_text(rh, cov, ahv),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=TICK_LABELSIZE,
            linespacing=1.25,
            zorder=6,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
        )

        ax.tick_params(labelsize=TICK_LABELSIZE)
        row, col = divmod(i, 3)
        if row == 2:
            ax.set_xlabel("Frequency (Hz)", fontsize=LABEL_FONTSIZE)
        else:
            ax.tick_params(labelbottom=False)
        if col == 0:
            ax.set_ylabel(r"$TF$", fontsize=LABEL_FONTSIZE)
        else:
            ax.tick_params(labelleft=False)

        if legend_handles is None:
            legend_handles = [
                Line2D([0], [0], color=COLOR_SAMPLES, lw=DATA_LINEWIDTH, label="2D samples"),
                Patch(
                    facecolor=COLOR_GEO,
                    edgecolor=COLOR_GEO,
                    alpha=0.35,
                    linestyle="--",
                    linewidth=DATA_LINEWIDTH,
                    label=r"Geomean $\pm 1\sigma$ ($\log\vert TF\vert$)",
                ),
                Line2D(
                    [0],
                    [0],
                    color=COLOR_1D,
                    ls=":",
                    lw=DATA_LINEWIDTH,
                    label="1D (baseline model)",
                ),
            ]

    assert legend_handles is not None
    # Header: sentence case, no bold on ordinary text (Nature)
    header.text(
        0.5,
        1.0,
        "Sensitivity of transfer function to parameters",
        transform=header.transAxes,
        ha="center",
        va="top",
        fontsize=LABEL_FONTSIZE,
        fontweight="normal",
    )
    header.text(
        0.5,
        0.68,
        rf"($H = {H:.0f}$ m, $V_{{s1}} = {VS1:.0f}$ m/s; "
        rf"center node, {N_SEEDS} seeds per case)",
        transform=header.transAxes,
        ha="center",
        va="top",
        fontsize=TICK_LABELSIZE,
    )
    header.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=TICK_LABELSIZE,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.2,
        borderaxespad=0.0,
        labelspacing=0.15,
        bbox_to_anchor=(0.5, 0.0),
    )

    return fig


def main() -> None:
    freq, tf_all, freq_1d, tf_1d = load_data()
    fig = plot_figure(freq, tf_all, freq_1d, tf_1d)

    import matplotlib as mpl

    print(
        f"Fonts: family={mpl.rcParams['font.family']}, "
        f"sans={mpl.rcParams['font.sans-serif'][:2]}, "
        f"mathtext={mpl.rcParams['mathtext.fontset']}"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf",):
        path = OUT_DIR / f"{OUT_STEM}.{ext}"
        fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.12)
        print(f"Wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
