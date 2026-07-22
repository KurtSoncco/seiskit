"""Cross-layout |TF| figure: shared-center one-at-a-time sensitivity sweep.

H = 50 m, Vs1 = 230 m/s. Baseline (r_h=30, CoV=0.2, a_hv=10) is drawn once at
the center of a 3x3 GridSpec; the six surrounding occupied cells show low/high
levels of each parameter. Cells (0,2) and (2,0) hold metadata / key text.

Produces:
  complete/full_paper/figures/qualitative/
    tf_raw_cross_center_node_all_seeds_h50_vs1_230_node_50.pdf
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle

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


def _load_sibling():
    """Import the 3×3 sibling module (not a package) for shared helpers."""
    path = Path(__file__).resolve().parent / "tf_center_all_seeds_3x3.py"
    spec = importlib.util.spec_from_file_location("tf_center_all_seeds_3x3", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load sibling module at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_sib = _load_sibling()
CENTER_CH = _sib.CENTER_CH
COLOR_1D = _sib.COLOR_1D
COLOR_GEO = _sib.COLOR_GEO
COLOR_SAMPLES = _sib.COLOR_SAMPLES
H = _sib.H
N_SEEDS = _sib.N_SEEDS
VS1 = _sib.VS1
cell_start = _sib.cell_start
load_data = _sib.load_data
plot_tf_panel = _sib.plot_tf_panel

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
OUT_DIR = figure_dir("qualitative")
OUT_STEM = "tf_raw_cross_center_node_all_seeds_h50_vs1_230_node_50"

DOC_WIDTH, FIG_HEIGHT = figsize(aspect=0.90)

# Accent colors for each sweep — standardized parameter colors from
# statistical_analysis/conference_paper/config.py (FACTOR_COLORS, Paul Tol Bright)
ACCENTS: dict[str, str] = {
    "r_h": "#CCBB44",  # Tol Bright yellow (rH)
    "CoV": "#228833",  # Tol Bright green (CoV)
    "a_hv": "#66CCEE",  # Tol Bright cyan (aHV)
}
ACCENT_BASELINE = "#333333"

# Pretty labels / formatters for panel annotations
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

# Text cells in the 3×3 grid
META_CELL = (0, 2)
KEY_CELL = (2, 0)


def _case_tuple(case: Mapping[str, float]) -> tuple[float, float, float]:
    """Canonical (r_h, CoV, a_hv) ordering for uniqueness checks / slicing."""
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
    """Map GridSpec (row, col) → case dict for the shared-center cross layout.

    Layout (rows top→bottom, cols left→right)::

        (0,0) diag LOW     (0,1) col LOW      (0,2) [metadata]
        (1,0) row LOW      (1,1) BASELINE     (1,2) row HIGH
        (2,0) [key]        (2,1) col HIGH     (2,2) diag HIGH
    """
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
    """Return the single parameter that differs from baseline, or None if equal."""
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
    """Short colored bar in the panel top-left (open-frame style has no top spine)."""
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

    # Fixed context for each sweep (the two params held at baseline)
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

    # Direction convention (no arrows)
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

    # 3-row color key
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
    """Lowest occupied panel per column (x), left-most occupied per row (y)."""
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
    row_param: str = "r_h",
    col_param: str = "CoV",
    diagonal_param: str = "a_hv",
    accent: Mapping[str, str] | None = None,
) -> plt.Figure:
    """Build the 7-panel shared-center cross |TF| figure."""
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

    fig = plt.figure(figsize=(DOC_WIDTH, FIG_HEIGHT), constrained_layout=True)
    # Reserve a top strip for the two-line title; constrained_layout only
    # auto-reserves space for suptitle, not fig.text.
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.925))
    fig.text(
        0.5,
        0.985,
        "Sensitivity of transfer function to parameters",
        ha="center",
        va="top",
        fontsize=LABEL_FONTSIZE,
        fontweight="normal",
    )
    fig.text(
        0.5,
        0.952,
        rf"($H = {H:.0f}$ m, $V_{{s1}} = {VS1:.0f}$ m/s; "
        rf"center node, {N_SEEDS} seeds per case)",
        ha="center",
        va="top",
        fontsize=TICK_LABELSIZE,
    )
    gs = fig.add_gridspec(3, 3, wspace=0.06, hspace=0.08)

    # Text cells
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

    # Data panels — share axes; create in row-major order for letter indexing
    axes: dict[tuple[int, int], plt.Axes] = {}
    share_ref: plt.Axes | None = None
    letter_i = 0
    for r in range(3):
        for c in range(3):
            if (r, c) not in occupied:
                continue
            ax = fig.add_subplot(
                gs[r, c],
                sharex=share_ref,
                sharey=share_ref,
            )
            if share_ref is None:
                share_ref = ax
            axes[(r, c)] = ax

            case = cell_to_case[(r, c)]
            rh, cov, ahv = _case_tuple(case)
            i0 = cell_start(VS1, cov, rh, ahv)
            stack = np.asarray(tf_all[i0 : i0 + N_SEEDS, CENTER_CH, :], dtype=np.float64)
            plot_tf_panel(ax, freq, stack, freq_1d, tf_1d)

            varying = _varying_param(case, baseline)
            accent_color = ACCENT_BASELINE if varying is None else accents[varying]
            _draw_accent_bar(ax, accent_color)

            add_panel_label(ax, letter_i)
            letter_i += 1

            ax.text(
                0.98,
                0.97,
                _panel_annotation(case, baseline),
                transform=ax.transAxes,
                ha="right",
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

    # Sanity: baseline once, 7 unique cases drawn
    drawn = [_case_tuple(cell_to_case[k]) for k in sorted(axes)]
    if len(drawn) != 7 or len(set(drawn)) != 7:
        raise RuntimeError(f"inconsistent drawn cases: {drawn}")
    if sum(1 for t in drawn if t == _case_tuple(baseline)) != 1:
        raise RuntimeError("baseline was not drawn exactly once")

    fig.supxlabel("Frequency (Hz)", fontsize=LABEL_FONTSIZE)
    fig.supylabel(r"$TF$", fontsize=LABEL_FONTSIZE)

    return fig


def main() -> None:
    baseline = {"r_h": 30.0, "CoV": 0.2, "a_hv": 10.0}
    sweeps = {
        "r_h": (10.0, 50.0),
        "CoV": (0.1, 0.3),
        "a_hv": (1.0, 50.0),
    }

    freq, tf_all, freq_1d, tf_1d = load_data()
    fig = make_cross_figure(
        freq,
        tf_all,
        freq_1d,
        tf_1d,
        baseline,
        sweeps,
        row_param="r_h",
        col_param="CoV",
        diagonal_param="a_hv",
        accent=ACCENTS,
    )

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
