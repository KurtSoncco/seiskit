"""Shared helpers for qualitative |TF| sensitivity figures.

Three sampling modes (gray curves + geomean±1σ over that pool)::

  center_node_all_seeds  — node 50 × all seeds
  one_seed_all_nodes     — seed 0 × all nodes
  all_seeds_all_nodes    — all seeds × all nodes (flattened)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

# full_paper/ is four levels up from this file's parent (qualitative/)
_FULL_PAPER = Path(__file__).resolve().parents[2]
if str(_FULL_PAPER) not in sys.path:
    sys.path.insert(0, str(_FULL_PAPER))

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
DOC_WIDTH_CROSS, FIG_HEIGHT_CROSS = figsize(aspect=0.90)

FREQ_LIM = (1e-1, 1e1)
TF_LIM = (1e-1, 1e3)

COLOR_SAMPLES = "0.55"
COLOR_GEO = "#D55E00"
COLOR_1D = "#000000"

# Cross-layout accents (Paul Tol Bright / conference FACTOR_COLORS)
ACCENTS: dict[str, str] = {
    "r_h": "#CCBB44",
    "CoV": "#228833",
    "a_hv": "#66CCEE",
}
ACCENT_BASELINE = "#333333"

_PARAM_LABEL: dict[str, str] = {
    "r_h": r"$r_h$",
    "CoV": r"$\mathrm{CoV}$",
    "a_hv": r"$a_{hv}$",
}
_PARAM_FMT: dict[str, str] = {
    "r_h": "{:.0f} m",
    "CoV": "{:g}",
    "a_hv": "{:.0f}",
}

META_CELL = (0, 2)
KEY_CELL = (2, 0)

BASELINE = {"r_h": 30.0, "CoV": 0.2, "a_hv": 10.0}
SWEEPS = {
    "r_h": (10.0, 50.0),
    "CoV": (0.1, 0.3),
    "a_hv": (1.0, 50.0),
}


def case_figure_dir(case: Mode, layout: Literal["3x3", "cross"]) -> Path:
    """Return ``figures/qualitative/<case>/<layout>/``."""
    return figure_dir("qualitative", case, layout)


def out_stem(layout: Literal["3x3", "cross"], h: float, vs1: float) -> str:
    return f"tf_raw_{layout}_h{h:.0f}_vs1_{vs1:.0f}"


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


def stack_for_case(tf_all: np.ndarray, i0: int, mode: Mode) -> np.ndarray:
    """Return ``(n_curves, n_freq)`` stack for *mode* starting at cell index *i0*."""
    if mode == "center_node_all_seeds":
        return np.asarray(tf_all[i0 : i0 + N_SEEDS, CENTER_CH, :], dtype=np.float64)
    if mode == "one_seed_all_nodes":
        return np.asarray(tf_all[i0 + SEED_IDX, :, :], dtype=np.float64)
    if mode == "all_seeds_all_nodes":
        block = np.asarray(tf_all[i0 : i0 + N_SEEDS, :, :], dtype=np.float64)
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

    # One call for all curves (needed for all_seeds_all_nodes ~10k lines)
    ax.plot(
        freq,
        stack.T,
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

    dest = out_dir or case_figure_dir(mode, "3x3")
    print(
        f"Fonts: family={mpl.rcParams['font.family']}, "
        f"sans={mpl.rcParams['font.sans-serif'][:2]}, "
        f"mathtext={mpl.rcParams['mathtext.fontset']}"
    )
    written: list[Path] = []
    for h in H_LIST:
        for vs1 in VS1_LIST:
            print(f"=== 3×3 | {mode} | H={h:.0f} m, Vs1={vs1:.0f} m/s ===")
            written.append(save_3x3(h, vs1, mode=mode, out_dir=dest))
    print(f"Done: {len(written)} figures → {dest}")
    return written


# ---------------------------------------------------------------------------
# Cross layout
# ---------------------------------------------------------------------------
def _case_tuple(case: Mapping[str, float]) -> tuple[float, float, float]:
    return (float(case["r_h"]), float(case["CoV"]), float(case["a_hv"]))


def _vary_one(baseline: Mapping[str, float], param: str, value: float) -> dict[str, float]:
    out = dict(baseline)
    out[param] = float(value)
    return out


def build_cell_to_case(
    baseline: Mapping[str, float],
    sweeps: Mapping[str, tuple[float, float]],
    *,
    row_param: str = "r_h",
    col_param: str = "CoV",
    diagonal_param: str = "a_hv",
) -> dict[tuple[int, int], dict[str, float]]:
    """Map GridSpec (row, col) → case dict for the shared-center cross layout."""
    assigned = {row_param, col_param, diagonal_param}
    if assigned != set(sweeps.keys()):
        raise ValueError(
            f"row/col/diagonal params {assigned!r} must match sweeps keys {set(sweeps.keys())!r}"
        )
    if len(assigned) != 3:
        raise ValueError("row_param, col_param, diagonal_param must be distinct")
    for p in assigned:
        if p not in baseline:
            raise ValueError(f"baseline missing parameter {p!r}")

    lo_row, hi_row = sweeps[row_param]
    lo_col, hi_col = sweeps[col_param]
    lo_diag, hi_diag = sweeps[diagonal_param]

    cell_to_case: dict[tuple[int, int], dict[str, float]] = {
        (1, 1): dict(baseline),
        (1, 0): _vary_one(baseline, row_param, lo_row),
        (1, 2): _vary_one(baseline, row_param, hi_row),
        (0, 1): _vary_one(baseline, col_param, lo_col),
        (2, 1): _vary_one(baseline, col_param, hi_col),
        (0, 0): _vary_one(baseline, diagonal_param, lo_diag),
        (2, 2): _vary_one(baseline, diagonal_param, hi_diag),
    }
    _assert_mapping(cell_to_case, baseline)
    return cell_to_case


def _assert_mapping(
    cell_to_case: Mapping[tuple[int, int], Mapping[str, float]],
    baseline: Mapping[str, float],
) -> None:
    if len(cell_to_case) != 7:
        raise ValueError(f"expected 7 occupied cells, got {len(cell_to_case)}")
    tuples = [_case_tuple(c) for c in cell_to_case.values()]
    if len(set(tuples)) != 7:
        raise ValueError(f"expected 7 unique cases, got {len(set(tuples))}: {tuples}")
    base_t = _case_tuple(baseline)
    n_base = sum(1 for t in tuples if t == base_t)
    if n_base != 1:
        raise ValueError(f"baseline case must appear exactly once, found {n_base}")
    if _case_tuple(cell_to_case[(1, 1)]) != base_t:
        raise ValueError("center cell (1,1) must be the baseline")


def _varying_param(case: Mapping[str, float], baseline: Mapping[str, float]) -> str | None:
    diffs = [p for p in ("r_h", "CoV", "a_hv") if case[p] != baseline[p]]
    if not diffs:
        return None
    if len(diffs) != 1:
        raise ValueError(f"case differs in multiple params from baseline: {diffs}")
    return diffs[0]


def _panel_annotation(case: Mapping[str, float], baseline: Mapping[str, float]) -> str:
    varying = _varying_param(case, baseline)
    if varying is None:
        return "baseline (shared)\n" + "\n".join(
            f"{_PARAM_LABEL[p]} = {_PARAM_FMT[p].format(case[p])}" for p in ("r_h", "CoV", "a_hv")
        )
    return f"{_PARAM_LABEL[varying]} = {_PARAM_FMT[varying].format(case[varying])}"


def _draw_accent_bar(ax: plt.Axes, color: str) -> None:
    ax.add_patch(
        Rectangle(
            (0.0, 1.0),
            0.22,
            0.035,
            transform=ax.transAxes,
            facecolor=color,
            edgecolor="none",
            clip_on=False,
            zorder=8,
        )
    )


def _fill_metadata_cell(
    ax: plt.Axes,
    baseline: Mapping[str, float],
    sweeps: Mapping[str, tuple[float, float]],
    *,
    row_param: str,
    col_param: str,
    diagonal_param: str,
) -> None:
    ax.axis("off")
    sweep_order = (row_param, col_param, diagonal_param)
    lines: list[str] = ["Fixed per sweep:"]
    for p in sweep_order:
        fixed = [q for q in ("r_h", "CoV", "a_hv") if q != p]
        fixed_str = ", ".join(
            f"{_PARAM_LABEL[q]}={_PARAM_FMT[q].format(baseline[q])}" for q in fixed
        )
        lines.append(f"  {_PARAM_LABEL[p]}: {fixed_str}")

    ax.text(
        0.0,
        1.0,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=LABEL_FONTSIZE,
        linespacing=1.35,
    )
    ax.legend(
        handles=_legend_handles(),
        loc="lower left",
        fontsize=LABEL_FONTSIZE,
        frameon=False,
        handlelength=2.0,
        labelspacing=0.35,
        borderaxespad=0.0,
        bbox_to_anchor=(0.0, 0.0),
    )


def _fill_key_cell(
    ax: plt.Axes,
    *,
    row_param: str,
    col_param: str,
    diagonal_param: str,
    accent: Mapping[str, str],
) -> None:
    ax.axis("off")
    ax.text(
        0.0,
        1.0,
        "Each parameter increases\nTop-left → bottom-right\n· Center = baseline",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=LABEL_FONTSIZE,
        linespacing=1.3,
    )
    y0 = 0.48
    dy = 0.16
    for i, p in enumerate((row_param, col_param, diagonal_param)):
        y = y0 - i * dy
        ax.add_patch(
            FancyBboxPatch(
                (0.0, y - 0.02),
                0.12,
                0.08,
                boxstyle="round,pad=0.01,rounding_size=0.01",
                transform=ax.transAxes,
                facecolor=accent[p],
                edgecolor="none",
                clip_on=False,
            )
        )
        ax.text(
            0.16,
            y + 0.02,
            _PARAM_LABEL[p],
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=LABEL_FONTSIZE,
        )


def _tick_label_cells(
    occupied: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    x_cells: set[tuple[int, int]] = set()
    y_cells: set[tuple[int, int]] = set()
    for col in range(3):
        rows = [r for r, c in occupied if c == col]
        if rows:
            x_cells.add((max(rows), col))
    for row in range(3):
        cols = [c for r, c in occupied if r == row]
        if cols:
            y_cells.add((row, min(cols)))
    return x_cells, y_cells


def make_cross_figure(
    freq: np.ndarray,
    tf_all: np.ndarray,
    freq_1d: np.ndarray,
    tf_1d: np.ndarray,
    baseline: Mapping[str, float],
    sweeps: Mapping[str, tuple[float, float]],
    *,
    h: float,
    vs1: float,
    mode: Mode,
    row_param: str = "r_h",
    col_param: str = "CoV",
    diagonal_param: str = "a_hv",
    accent: Mapping[str, str] | None = None,
) -> plt.Figure:
    """Build the 7-panel shared-center cross |TF| figure for one (H, Vs1) and *mode*."""
    accents = dict(accent or ACCENTS)
    cell_to_case = build_cell_to_case(
        baseline,
        sweeps,
        row_param=row_param,
        col_param=col_param,
        diagonal_param=diagonal_param,
    )
    occupied = set(cell_to_case.keys())
    x_tick_cells, y_tick_cells = _tick_label_cells(occupied)

    fig = plt.figure(figsize=(DOC_WIDTH_CROSS, FIG_HEIGHT_CROSS), constrained_layout=True)
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.96))
    fig.text(
        0.5,
        0.985,
        _subtitle(h, vs1, mode),
        ha="center",
        va="top",
        fontsize=TICK_LABELSIZE,
    )
    gs = fig.add_gridspec(3, 3, wspace=0.06, hspace=0.08)

    ax_meta = fig.add_subplot(gs[META_CELL])
    ax_key = fig.add_subplot(gs[KEY_CELL])
    _fill_metadata_cell(
        ax_meta,
        baseline,
        sweeps,
        row_param=row_param,
        col_param=col_param,
        diagonal_param=diagonal_param,
    )
    _fill_key_cell(
        ax_key,
        row_param=row_param,
        col_param=col_param,
        diagonal_param=diagonal_param,
        accent=accents,
    )

    axes: dict[tuple[int, int], plt.Axes] = {}
    share_ref: plt.Axes | None = None
    letter_i = 0
    for r in range(3):
        for c in range(3):
            if (r, c) not in occupied:
                continue
            ax = fig.add_subplot(gs[r, c], sharex=share_ref, sharey=share_ref)
            if share_ref is None:
                share_ref = ax
            axes[(r, c)] = ax

            case = cell_to_case[(r, c)]
            rh, cov, ahv = _case_tuple(case)
            i0 = cell_start(vs1, cov, rh, ahv)
            stack = stack_for_case(tf_all, i0, mode)
            plot_tf_panel(ax, freq, stack, freq_1d, tf_1d)

            varying = _varying_param(case, baseline)
            accent_color = ACCENT_BASELINE if varying is None else accents[varying]
            _draw_accent_bar(ax, accent_color)
            add_panel_label(ax, letter_i)
            letter_i += 1

            ax.text(
                0.02,
                0.97,
                _panel_annotation(case, baseline),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=TICK_LABELSIZE,
                linespacing=1.25,
                zorder=6,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.75,
                    "pad": 1.0,
                },
            )

            ax.tick_params(labelsize=TICK_LABELSIZE)
            if (r, c) not in x_tick_cells:
                ax.tick_params(labelbottom=False)
            if (r, c) not in y_tick_cells:
                ax.tick_params(labelleft=False)

    drawn = [_case_tuple(cell_to_case[k]) for k in sorted(axes)]
    if len(drawn) != 7 or len(set(drawn)) != 7:
        raise RuntimeError(f"inconsistent drawn cases: {drawn}")
    if sum(1 for t in drawn if t == _case_tuple(baseline)) != 1:
        raise RuntimeError("baseline was not drawn exactly once")

    fig.supxlabel("Frequency (Hz)", fontsize=LABEL_FONTSIZE)
    fig.supylabel(r"$TF$", fontsize=LABEL_FONTSIZE)
    return fig


def save_cross(
    h: float,
    vs1: float,
    *,
    mode: Mode,
    out_dir: Path,
) -> Path:
    freq, tf_all, freq_1d, tf_1d = load_data(h, vs1)
    fig = make_cross_figure(
        freq,
        tf_all,
        freq_1d,
        tf_1d,
        BASELINE,
        SWEEPS,
        h=h,
        vs1=vs1,
        mode=mode,
        row_param="r_h",
        col_param="CoV",
        diagonal_param="a_hv",
        accent=ACCENTS,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{out_stem('cross', h, vs1)}.pdf"
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(f"Wrote {path}")
    return path


def run_cross(mode: Mode, out_dir: Path | None = None) -> list[Path]:
    import matplotlib as mpl

    dest = out_dir or case_figure_dir(mode, "cross")
    print(
        f"Fonts: family={mpl.rcParams['font.family']}, "
        f"sans={mpl.rcParams['font.sans-serif'][:2]}, "
        f"mathtext={mpl.rcParams['mathtext.fontset']}"
    )
    written: list[Path] = []
    for h in H_LIST:
        for vs1 in VS1_LIST:
            print(f"=== cross | {mode} | H={h:.0f} m, Vs1={vs1:.0f} m/s ===")
            written.append(save_cross(h, vs1, mode=mode, out_dir=dest))
    print(f"Done: {len(written)} figures → {dest}")
    return written
