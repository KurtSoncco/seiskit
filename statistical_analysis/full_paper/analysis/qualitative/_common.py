"""Shared helpers for qualitative |TF| sensitivity figures.

Four sampling modes (samples + geomean±1σ over that pool)::

  center_node_one_seed   — node 50 × seed 0 (single realization)
  center_node_all_seeds  — node 50 × all seeds
  one_seed_all_nodes     — seed 0 × all nodes
  all_seeds_all_nodes    — all seeds × all nodes (flattened)

Each 3×3 figure is an \\(r_h \\times a_{hv}\\) grid. Within a panel, all CoV
levels are overlaid (Paul Tol Bright color + linestyle); the homogeneous 1D
baseline is a thick black line.
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
    TOL_BRIGHT,
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

Mode = Literal[
    "center_node_one_seed",
    "center_node_all_seeds",
    "one_seed_all_nodes",
    "all_seeds_all_nodes",
]

MODE_SUBTITLE: dict[Mode, str] = {
    "center_node_one_seed": f"center node, seed {SEED_IDX}",
    "center_node_all_seeds": f"center node, {N_SEEDS} seeds per case",
    "one_seed_all_nodes": f"seed {SEED_IDX}, all {N_NODES} nodes",
    "all_seeds_all_nodes": f"all {N_SEEDS} seeds × all {N_NODES} nodes",
}

# 3×3 panels (a)–(i): rows = r_h, cols = a_hv (CoV overlaid in each panel)
PANELS: list[tuple[float, float]] = [(rh, ahv) for rh in RH_LIST for ahv in AHV_LIST]

DOC_WIDTH_3X3, FIG_HEIGHT_3X3 = figsize(aspect=0.95)

FREQ_LIM = (1e-1, 1e1)
TF_LIM = (5e-1, 5e2)

# Paul Tol Bright: colorblind-safe blue / green / vermillion (plus linestyle)
COV_STYLE: dict[float, dict[str, str]] = {
    0.1: {"color": TOL_BRIGHT["blue"], "ls": "-", "label": "CoV = 0.1"},
    0.2: {"color": TOL_BRIGHT["green"], "ls": "--", "label": "CoV = 0.2"},
    0.3: {"color": TOL_BRIGHT["red"], "ls": "-.", "label": "CoV = 0.3"},
}
COLOR_1D = "#000000"
LW_1D = 1.75


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


# Cap sample curves per panel (split across CoV overlays) so all_seeds_all_nodes
# (~10k/cell) does not OOM matplotlib.
MAX_PLOT_CURVES = 200


def stack_for_case(tf_all: np.ndarray, i0: int, mode: Mode) -> np.ndarray:
    """Return ``(n_curves, n_freq)`` stack for *mode* starting at cell index *i0*.

    ``all_seeds_all_nodes`` uses float32 to limit RAM (~3 GB mmap × full copy).
    """
    if mode == "center_node_one_seed":
        # Shape (1, n_freq) so geomean band collapses to the single curve.
        return np.asarray(
            tf_all[i0 + SEED_IDX : i0 + SEED_IDX + 1, CENTER_CH, :],
            dtype=np.float64,
        )
    if mode == "center_node_all_seeds":
        return np.asarray(tf_all[i0 : i0 + N_SEEDS, CENTER_CH, :], dtype=np.float64)
    if mode == "one_seed_all_nodes":
        return np.asarray(tf_all[i0 + SEED_IDX, :, :], dtype=np.float64)
    if mode == "all_seeds_all_nodes":
        block = np.asarray(tf_all[i0 : i0 + N_SEEDS, :, :], dtype=np.float32)
        return block.reshape(N_SEEDS * block.shape[1], -1)
    raise ValueError(f"unknown mode: {mode!r}")


def _panel_param_text(rh: float, ahv: float) -> str:
    return rf"$r_h = {rh:.0f}$ m" + "\n" + rf"$a_{{hv}} = {ahv:.0f}$"


def _geomean_band(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return geomean, lo, hi for log|TF| ±1σ across curves (axis 0)."""
    log_tf = np.log(np.clip(stack, 1e-12, None))
    geo = np.exp(np.mean(log_tf, axis=0))
    sigma_ln = np.std(log_tf, axis=0, ddof=1)
    lo = geo * np.exp(-sigma_ln)
    hi = geo * np.exp(sigma_ln)
    return geo, lo, hi


def _subsample_stack(stack: np.ndarray, n_max: int, *, seed: int) -> np.ndarray:
    """Subsample curves for display; full *stack* still feeds the geomean band."""
    n = int(stack.shape[0])
    if n <= n_max:
        return stack
    rng = np.random.default_rng(seed)
    return stack[rng.choice(n, size=n_max, replace=False)]


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
    tf_all: np.ndarray,
    freq_1d: np.ndarray,
    tf_1d: np.ndarray,
    *,
    vs1: float,
    rh: float,
    ahv: float,
    mode: Mode,
) -> None:
    """Overlay CoV samples + geomean±1σ (log|TF|) and the 1D baseline on *ax*."""
    max(1, MAX_PLOT_CURVES // len(COV_LIST))
    for i_cov, cov in enumerate(COV_LIST):
        style = COV_STYLE[cov]
        color = style["color"]
        ls = style["ls"]
        i0 = cell_start(vs1, cov, rh, ahv)
        stack = stack_for_case(tf_all, i0, mode)
        geo, lo, hi = _geomean_band(stack)
        # plot_stack = _subsample_stack(stack, n_max, seed=i_cov)

        # ax.plot(
        #    freq,
        #    plot_stack.T,
        #    color=color,
        #    lw=0.35,
        #    alpha=0.22,
        #    zorder=2,
        # )

        ax.fill_between(
            freq,
            lo,
            hi,
            facecolor=color,
            alpha=0.18,
            edgecolor="none",
            zorder=3,
            label="_nolegend_",
        )
        ax.plot(freq, geo, color=color, ls=ls, lw=DATA_LINEWIDTH, zorder=4)
        # ax.plot(freq, lo, color=color, ls=ls, lw=0.55, alpha=0.70, zorder=4)
        # ax.plot(freq, hi, color=color, ls=ls, lw=0.55, alpha=0.70, zorder=4)

    ax.plot(freq_1d, tf_1d, color=COLOR_1D, ls="-", lw=LW_1D, zorder=1)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*FREQ_LIM)
    ax.set_ylim(*TF_LIM)
    ax.grid(True, which="major", alpha=0.18, lw=0.6)
    ax.grid(True, which="minor", alpha=0.08, lw=0.4)


def _legend_handles() -> list:
    """CoV encodes color+linestyle for samples and geomean±1σ; 1D is black."""
    handles: list = [
        Line2D(
            [0],
            [0],
            color=COV_STYLE[cov]["color"],
            ls=COV_STYLE[cov]["ls"],
            lw=DATA_LINEWIDTH,
            label=COV_STYLE[cov]["label"],
        )
        for cov in COV_LIST
    ]
    handles.append(
        Patch(
            facecolor=TOL_BRIGHT["gray"],
            edgecolor="none",
            alpha=0.45,
            label=r"Geomean $\pm 1\sigma$ ($\ln(\left| TF \right|)$)",
        )
    )
    handles.append(
        Line2D(
            [0],
            [0],
            color=COLOR_1D,
            ls="-",
            lw=LW_1D,
            label="1D (baseline model)",
        )
    )
    return handles


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
    r"""Build the 3×3 log–log |TF| figure for one (H, Vs1) and *mode*.

    Rows vary \(r_h\), columns vary \(a_{hv}\); CoV levels share each panel.
    """
    fig = plt.figure(figsize=(DOC_WIDTH_3X3, FIG_HEIGHT_3X3))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.09, 1.0],
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

    for i, (rh, ahv) in enumerate(PANELS):
        ax = axes.flat[i]
        plot_tf_panel(
            ax,
            freq,
            tf_all,
            freq_1d,
            tf_1d,
            vs1=vs1,
            rh=rh,
            ahv=ahv,
            mode=mode,
        )

        add_panel_label(ax, i)
        ax.text(
            0.02,
            0.97,
            _panel_param_text(rh, ahv),
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
            ax.set_ylabel(r"$\left| TF \right|$", fontsize=LABEL_FONTSIZE)
        else:
            ax.tick_params(labelleft=False)

        if legend_handles is None:
            legend_handles = _legend_handles()

    assert legend_handles is not None
    header.text(
        0.5,
        0.98,
        _subtitle(h, vs1, mode),
        transform=header.transAxes,
        ha="center",
        va="top",
        fontsize=TICK_LABELSIZE,
    )
    header.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        fontsize=TICK_LABELSIZE,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.0,
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
