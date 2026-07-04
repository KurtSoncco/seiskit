"""Reusable helpers for subfigure labels, title formatting, legend placement,
and clean axis limits.
"""

from __future__ import annotations

import string
from typing import Iterable, Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from seiskit.plot_config.style import FONT_SIZE, TITLE_SIZE

# Subfigure label sequence: a, b, c, ...
_SUBFIG_LABELS = list(string.ascii_lowercase)


def panel_letter(
    ax: Axes,
    letter: str,
    *,
    fontsize: int | None = None,
    x: float = 0.02,
    y: float = 0.98,
) -> None:
    """Place a bare lowercase panel letter (a, b, c, ...) in the axes.

    Lighter-weight alternative to :func:`add_subfigure_label` — no
    parentheses, positioned flush with the top-left corner.
    """
    ax.text(
        x,
        y,
        letter.lower(),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontweight="bold",
        fontsize=fontsize or FONT_SIZE,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.5},
        zorder=10,
    )


def add_subfigure_label(
    ax: Axes,
    index: int,
    *,
    x: float = 0.02,
    y: float = 0.97,
    fontsize: int | None = None,
    alpha: float = 0.75,
) -> None:
    """Place a sequential subfigure label (a, b, c, ...) inside the axes.

    The label has a white bounding box with partial transparency so that
    underlying data remains visible.

    Parameters
    ----------
    ax:
        Matplotlib Axes to annotate.
    index:
        0-based subfigure index (0 → "a", 1 → "b", ...).
    x, y:
        Position in axes coordinates (upper-left by default).
    fontsize:
        Override the default label font size.
    alpha:
        Background box transparency (0 = fully transparent).
    """
    label = _SUBFIG_LABELS[index % len(_SUBFIG_LABELS)]
    ax.text(
        x,
        y,
        f"({label})",
        transform=ax.transAxes,
        fontsize=fontsize or FONT_SIZE,
        fontweight="bold",
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="none",
            alpha=alpha,
        ),
    )


def format_title(
    main: str,
    subtitle: str | None = None,
    *,
    ax: Axes | None = None,
    fig: Figure | None = None,
) -> None:
    """Set a bold main title with an optional non-bold subtitle beneath it.

    Exactly one of *ax* or *fig* should be provided.  When *fig* is used the
    title is set via ``fig.suptitle``; otherwise ``ax.set_title`` is used.
    """
    if ax is not None:
        full = main
        if subtitle:
            full = f"$\\bf{{{main}}}$\n{subtitle}"
            ax.set_title(full, fontsize=TITLE_SIZE)
        else:
            ax.set_title(main, fontsize=TITLE_SIZE, fontweight="bold")
    elif fig is not None:
        if subtitle:
            fig.suptitle(
                f"$\\bf{{{main}}}$\n{subtitle}",
                fontsize=TITLE_SIZE,
                y=1.02,
            )
        else:
            fig.suptitle(main, fontsize=TITLE_SIZE, fontweight="bold", y=1.02)


def place_legend(
    ax: Axes,
    *,
    position: str = "bottom",
    ncol: int | None = None,
    **kwargs,
) -> None:
    """Place the legend at the *top* or *bottom* of the plotting area.

    The legend is positioned outside the data region to avoid occluding
    data points.

    Parameters
    ----------
    ax:
        Target axes.
    position:
        ``"top"`` or ``"bottom"``.
    ncol:
        Number of columns.  ``None`` auto-selects based on handle count.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    if ncol is None:
        ncol = min(len(handles), 4)

    loc_map = {
        "top": dict(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=ncol),
        "bottom": dict(loc="upper left", bbox_to_anchor=(0.0, -0.12), ncol=ncol),
    }
    legend_kw = loc_map.get(position, loc_map["bottom"])
    legend_kw.update(kwargs)
    ax.legend(**legend_kw)


def set_clean_axis_limits(
    ax: Axes,
    data_arrays: Sequence[np.ndarray] | None = None,
    *,
    axis: str = "both",
    pad_fraction: float = 0.0,
) -> None:
    """Snap axis limits to clean data boundaries.

    When *data_arrays* are given the limits are derived from the union of
    min/max across all arrays.  When omitted the current axis limits are
    left unchanged but ticks are snapped to the endpoints.

    Parameters
    ----------
    ax:
        Target axes.
    data_arrays:
        One or more 1-D arrays whose extent should define the limits.
    axis:
        ``"x"``, ``"y"``, or ``"both"``.
    pad_fraction:
        Fraction of the data range to add as padding (0 = exact fit).
    """
    if data_arrays is not None and len(data_arrays) > 0:
        flat = np.concatenate([np.asarray(a).ravel() for a in data_arrays])
        finite = flat[np.isfinite(flat)]
        if finite.size == 0:
            return
        lo, hi = float(finite.min()), float(finite.max())
        pad = (hi - lo) * pad_fraction
        lo -= pad
        hi += pad
    else:
        lo, hi = None, None

    if axis in ("x", "both") and lo is not None:
        ax.set_xlim(lo, hi)
    if axis in ("y", "both") and lo is not None:
        ax.set_ylim(lo, hi)


def annotate_comment(
    ax: Axes,
    text: str,
    *,
    x: float = 0.98,
    y: float = 0.95,
    fontsize: int | None = None,
    alpha: float = 0.6,
) -> None:
    """Place a text annotation in the upper-right corner of *ax*.

    Uses a white background box with the given *alpha* transparency,
    suitable for brief notes or context strings that would otherwise
    crowd a title.
    """
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=fontsize or (FONT_SIZE - 2),
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": alpha,
            "boxstyle": "round,pad=0.3",
        },
        zorder=10,
    )


def hatch_bars(
    ax_or_fig: Axes | Figure,
    patterns: Sequence[str] | None = None,
) -> None:
    """Apply distinct hatch patterns to bar groups on *ax_or_fig*.

    Each ``BarContainer`` on an axes receives a different pattern from
    *patterns* (defaults to the module-level ``HATCH_PATTERNS``).
    Useful for making bar charts distinguishable in B&W print.
    """
    from seiskit.plot_config.style import HATCH_PATTERNS

    pats = list(patterns or HATCH_PATTERNS)

    axes: list[Axes]
    if isinstance(ax_or_fig, Figure):
        axes = list(ax_or_fig.axes)
    else:
        axes = [ax_or_fig]

    for ax in axes:
        for i, container in enumerate(ax.containers):
            h = pats[i % len(pats)]
            for patch in container:
                patch.set_hatch(h)
                if h:
                    patch.set_edgecolor("black")


def enforce_clean_axis_limits(
    ax: Axes,
    *,
    x_values: Sequence[float] | None = None,
    y_values: Sequence[float] | None = None,
    x_bounds: Sequence[float] | None = None,
    y_bounds: Sequence[float] | None = None,
    x_ticks: Iterable[float] | None = None,
    y_ticks: Iterable[float] | None = None,
) -> None:
    """Set axis limits and ticks with explicit bounds or data-driven ranges."""
    if x_bounds is not None:
        ax.set_xlim(float(x_bounds[0]), float(x_bounds[1]))
    elif x_values is not None and len(x_values) > 0:
        xv = np.asarray(x_values, dtype=float)
        ax.set_xlim(float(np.nanmin(xv)), float(np.nanmax(xv)))

    if y_bounds is not None:
        ax.set_ylim(float(y_bounds[0]), float(y_bounds[1]))
    elif y_values is not None and len(y_values) > 0:
        yv = np.asarray(y_values, dtype=float)
        ax.set_ylim(float(np.nanmin(yv)), float(np.nanmax(yv)))

    if x_ticks is not None:
        ax.set_xticks(list(x_ticks))
    if y_ticks is not None:
        ax.set_yticks(list(y_ticks))
