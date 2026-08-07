"""Shared helpers for qualitative |TF| sensitivity figures.

Three sampling modes (gray curves + geomean±1σ over that pool)::

  center_node_all_seeds  — node 50 × all seeds
  one_seed_all_nodes     — seed 0 × all nodes
  all_seeds_all_nodes    — all seeds × all nodes (flattened)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# full_paper/ is four levels up from this file's parent (qualitative/)
_FULL_PAPER = Path(__file__).resolve().parents[2]
if str(_FULL_PAPER) not in sys.path:
    sys.path.insert(0, str(_FULL_PAPER))

from config import (  # noqa: E402
    BOX_ROOT,
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
BOX = BOX_ROOT.parent  # complete/ → Statistical Analysis campaign root

H_LIST = [15.0, 50.0, 100.0]
VS1_LIST = [100.0, 230.0, 360.0]
COV_LIST = [0.1, 0.2, 0.3]
RH_LIST = [10.0, 30.0, 50.0]
AHV_LIST = [1.0, 10.0, 50.0]

CENTER_CH = 50
N_SEEDS = 100
N_NODES = 101
SEED_IDX = 0

Mode = Literal["center_node_all_seeds", "one_seed_all_nodes", "all_seeds_all_nodes"]

MODE_SUBTITLE: dict[Mode, str] = {
    "center_node_all_seeds": f"center node, {N_SEEDS} seeds per case",
    "one_seed_all_nodes": f"seed {SEED_IDX}, all {N_NODES} nodes",
    "all_seeds_all_nodes": f"all {N_SEEDS} seeds × all {N_NODES} nodes",
}

# (rH, CoV, aHV) for 3×3 panels (a)–(i), row-major
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

DOC_WIDTH_3X3, FIG_HEIGHT_3X3 = figsize(aspect=0.95)

FREQ_LIM = (1e-1, 1e1)
TF_LIM = (1e-1, 1e3)

COLOR_SAMPLES = "0.55"
COLOR_GEO = "#D55E00"
COLOR_1D = "#000000"


def case_figure_dir(case: Mode, layout: str = "3x3") -> Path:
    """Return ``figures/qualitative/<case>/3x3/`` (cross layout removed)."""
    if layout != "3x3":
        raise ValueError("Only 3x3 layout is supported (cross removed per review)")
    return figure_dir("qualitative", case, "3x3")


def out_stem(layout: str, h: float, vs1: float) -> str:
    if layout != "3x3":
        raise ValueError("Only 3x3 layout is supported")
    return f"tf_raw_3x3_h{h:.0f}_vs1_{vs1:.0f}"


def tf_dir(h: float) -> Path:
    return BOX / f"h={h:.0f}" / "transfer_function_results"


def base_1d_path(h: float, vs1: float) -> Path:
    return BOX / f"h={h:.0f}" / "base_cases" / f"base_case_tf_Vs1{vs1:.0f}.npz"


def cell_start(vs1: float, cov: float, rh: float, ahv: float) -> int:
    """Index of the first seed for a factorial cell (order: Vs1→CoV→rH→aHV→seed)."""
    i = VS1_LIST.index(vs1)
    j = COV_LIST.index(cov)
    k = RH_LIST.index(rh)
    m = AHV_LIST.index(ahv)
    return (((i * len(COV_LIST) + j) * len(RH_LIST) + k) * len(AHV_LIST) + m) * N_SEEDS


# Cap gray sample curves so all_seeds_all_nodes (~10k) does not OOM matplotlib.
MAX_PLOT_CURVES = 200


def stack_for_case(tf_all: np.ndarray, i0: int, mode: Mode) -> np.ndarray:
    """Return ``(n_curves, n_freq)`` stack for *mode* starting at cell index *i0*.

    ``all_seeds_all_nodes`` uses float32 to limit RAM (~3 GB mmap × full copy).
    """
    if mode == "center_node_all_seeds":
        return np.asarray(tf_all[i0 : i0 + N_SEEDS, CENTER_CH, :], dtype=np.float64)
    if mode == "one_seed_all_nodes":
        return np.asarray(tf_all[i0 + SEED_IDX, :, :], dtype=np.float64)
    if mode == "all_seeds_all_nodes":
        block = np.asarray(tf_all[i0 : i0 + N_SEEDS, :, :], dtype=np.float32)
        return block.reshape(N_SEEDS * block.shape[1], -1)
    raise ValueError(f"unknown mode: {mode!r}")


def _panel_param_text(rh: float, cov: float, ahv: float) -> str:
    return (
        rf"$r_h = {rh:.0f}$ m" + "\n"
        rf"$\mathrm{{CoV}} = {cov:g}$" + "\n"
        rf"$a_{{hv}} = {ahv:.0f}$"
    )


def _geomean_band(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return geomean, lo, hi for log|TF| ±1σ across curves (axis 0)."""
    log_tf = np.log(np.clip(stack, 1e-12, None))
    geo = np.exp(np.mean(log_tf, axis=0))
    sigma_ln = np.std(log_tf, axis=0, ddof=1)
    lo = geo * np.exp(-sigma_ln)
    hi = geo * np.exp(sigma_ln)
    return geo, lo, hi


def _load_freq(h: float) -> np.ndarray:
    """Load frequency axis; fall back to tf_geomean.npz when freq.npz is absent."""
    d = tf_dir(h)
    freq_npz = d / "freq.npz"
    if freq_npz.exists():
        return np.asarray(np.load(freq_npz)["freq"], dtype=np.float64)
    geo_npz = d / "tf_geomean.npz"
    if geo_npz.exists():
        return np.asarray(np.load(geo_npz)["freq"], dtype=np.float64)
    raise FileNotFoundError(f"No frequency axis found under {d}")


def load_data(h: float, vs1: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load freq, memmapped 2D TF, and 1D baseline (freq, |TF|) for one (H, Vs1)."""
    d = tf_dir(h)
    freq = _load_freq(h)
    tf_all = np.load(d / "tf_per_sample.npy", mmap_mode="r")
    base = np.load(base_1d_path(h, vs1))
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

    # Subsample gray curves for plotting (full stack still used for geomean band)
    n = int(stack.shape[0])
    if n > MAX_PLOT_CURVES:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=MAX_PLOT_CURVES, replace=False)
        plot_stack = stack[idx]
    else:
        plot_stack = stack
    ax.plot(
        freq,
        plot_stack.T,
        color=COLOR_SAMPLES,
        lw=0.35,
        alpha=0.30,
        zorder=1,
    )

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

    ax.plot(freq_1d, tf_1d, color=COLOR_1D, lw=DATA_LINEWIDTH, zorder=2, dashes=(1, 1.5))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*FREQ_LIM)
    ax.set_ylim(*TF_LIM)
    ax.grid(True, which="major", alpha=0.18, lw=0.6)
    ax.grid(True, which="minor", alpha=0.08, lw=0.4)


def _legend_handles() -> list:
    return [
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


def _subtitle(h: float, vs1: float, mode: Mode) -> str:
    return (
        rf"($H = {h:.0f}$ m, $V_{{s1}} = {vs1:.0f}$ m/s; "
        rf"{MODE_SUBTITLE[mode]})"
    )


# ---------------------------------------------------------------------------
# 3×3 layout
# ---------------------------------------------------------------------------
def plot_3x3_figure(
    freq: np.ndarray,
    tf_all: np.ndarray,
    freq_1d: np.ndarray,
    tf_1d: np.ndarray,
    *,
    h: float,
    vs1: float,
    mode: Mode,
) -> plt.Figure:
    """Build the 3×3 log–log |TF| figure for one (H, Vs1) and *mode*."""
    fig = plt.figure(figsize=(DOC_WIDTH_3X3, FIG_HEIGHT_3X3))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.07, 1.0],
        hspace=0.02,
        left=0.09,
        right=0.99,
        bottom=0.07,
        top=0.98,
    )
    header = fig.add_subplot(gs[0, 0])
    header.axis("off")
    gs_panels = gs[1, 0].subgridspec(3, 3, wspace=0.10, hspace=0.14)
    axes = np.empty((3, 3), dtype=object)
    for r in range(3):
        for c in range(3):
            sharex = axes[0, 0] if (r, c) != (0, 0) else None
            sharey = axes[0, 0] if (r, c) != (0, 0) else None
            axes[r, c] = fig.add_subplot(gs_panels[r, c], sharex=sharex, sharey=sharey)

    legend_handles: list | None = None

    for i, (rh, cov, ahv) in enumerate(PANELS):
        ax = axes.flat[i]
        i0 = cell_start(vs1, cov, rh, ahv)
        stack = stack_for_case(tf_all, i0, mode)
        plot_tf_panel(ax, freq, stack, freq_1d, tf_1d)

        add_panel_label(ax, i)
        ax.text(
            0.02,
            0.97,
            _panel_param_text(rh, cov, ahv),
            transform=ax.transAxes,
            ha="left",
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
            legend_handles = _legend_handles()

    assert legend_handles is not None
    header.text(
        0.5,
        0.95,
        _subtitle(h, vs1, mode),
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


def save_3x3(
    h: float,
    vs1: float,
    *,
    mode: Mode,
    out_dir: Path,
) -> Path:
    freq, tf_all, freq_1d, tf_1d = load_data(h, vs1)
    fig = plot_3x3_figure(freq, tf_all, freq_1d, tf_1d, h=h, vs1=vs1, mode=mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{out_stem('3x3', h, vs1)}.pdf"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Wrote {path}")
    return path


def run_3x3(mode: Mode, out_dir: Path | None = None) -> list[Path]:
    import matplotlib as mpl
    from tqdm import tqdm

    dest = out_dir or case_figure_dir(mode, "3x3")
    print(
        f"Fonts: family={mpl.rcParams['font.family']}, "
        f"sans={mpl.rcParams['font.sans-serif'][:2]}, "
        f"mathtext={mpl.rcParams['mathtext.fontset']}"
    )
    jobs = [(h, vs1) for h in H_LIST for vs1 in VS1_LIST]
    written: list[Path] = []
    for h, vs1 in tqdm(jobs, desc=f"tf 3x3 [{mode}]"):
        written.append(save_3x3(h, vs1, mode=mode, out_dir=dest))
    print(f"Done: {len(written)} figures → {dest}")
    return written
