"""Centralized publication-quality plot configuration for seiskit.

This package provides standardized styling, colormaps, nomenclature, and
helper functions so every figure in the project shares a consistent,
colorblind-friendly, publication-ready appearance.

Usage::

    from seiskit.plot_config import apply_style, get_crameri_cmap, LABEL_MAP, add_subfigure_label
    apply_style()  # call once at the top of any plotting script

For conference-paper scripts that want zero-boilerplate label formatting::

    apply_style(auto_format=True, font_size=10, frame="open")
"""

from seiskit.plot_config.colormaps import get_crameri_cmap
from seiskit.plot_config.helpers import (
    add_subfigure_label,
    annotate_comment,
    enforce_clean_axis_limits,
    format_title,
    hatch_bars,
    panel_letter,
    place_legend,
    set_clean_axis_limits,
)
from seiskit.plot_config.labels import (
    LABEL_MAP,
    format_label,
    rename_channel,
    to_title_case,
)
from seiskit.plot_config.paths import result_path, set_results_root
from seiskit.plot_config.style import (
    COLORBLIND_COLORS,
    COLORBLIND_PALETTE,
    FONT_SIZE,
    HATCH_PATTERNS,
    apply_style,
    get_model_color,
    get_plotly_layout,
)

__all__ = [
    # Style
    "apply_style",
    "COLORBLIND_PALETTE",
    "COLORBLIND_COLORS",
    "FONT_SIZE",
    "get_model_color",
    "get_plotly_layout",
    # Colormaps
    "get_crameri_cmap",
    # Labels
    "LABEL_MAP",
    "format_label",
    "rename_channel",
    "to_title_case",
    # Helpers
    "add_subfigure_label",
    "annotate_comment",
    "hatch_bars",
    "HATCH_PATTERNS",
    "panel_letter",
    "format_title",
    "place_legend",
    "set_clean_axis_limits",
    "enforce_clean_axis_limits",
    # Paths
    "result_path",
    "set_results_root",
]
