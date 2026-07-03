"""Core style configuration: rcParams, colorblind-friendly palettes, font setup.

Call :func:`apply_style` once at the start of any script to globally set
matplotlib rcParams for publication-quality figures.
"""

from __future__ import annotations

import itertools
from typing import Iterator

import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    import seaborn as sns

    _SEABORN_AVAILABLE = True
except ImportError:  # pragma: no cover
    sns = None  # type: ignore[assignment]
    _SEABORN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Uniform text size (points) — identical for axis labels, tick labels, legend
# ---------------------------------------------------------------------------
FONT_SIZE: int = 12
TITLE_SIZE: int = 14

# ---------------------------------------------------------------------------
# Colorblind-friendly discrete palette
# ---------------------------------------------------------------------------
if _SEABORN_AVAILABLE and sns is not None:
    COLORBLIND_PALETTE = sns.color_palette("colorblind", as_cmap=False)
    COLORBLIND_COLORS: list[str] = [
        f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
        for r, g, b in COLORBLIND_PALETTE
    ]
else:
    COLORBLIND_PALETTE = None
    COLORBLIND_COLORS = [
        "#0173b2",
        "#de8f05",
        "#029e73",
        "#cc78bc",
        "#56b4e9",
        "#ece133",
        "#0072b2",
        "#d55e00",
    ]

_MODEL_COLORS: dict[str, str] = {
    "My_New_Run": COLORBLIND_COLORS[0],
    "PLAXIS": COLORBLIND_COLORS[1],
    "OpenSeesPy_Prev": COLORBLIND_COLORS[2],
}

_fallback_cycle: Iterator[str] = itertools.cycle(COLORBLIND_COLORS)


def get_model_color(model_name: str) -> str:
    """Return a deterministic colorblind-friendly hex colour for *model_name*."""
    return _MODEL_COLORS.get(model_name, next(_fallback_cycle))


# ---------------------------------------------------------------------------
# Apply global matplotlib rcParams
# ---------------------------------------------------------------------------

def apply_style() -> None:
    """Set matplotlib rcParams for publication-quality output.

    * Times New Roman font family (with STIX fallback)
    * Uniform font sizes across labels, ticks, legends
    * LaTeX-capable mathtext
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    mpl.rcParams.update(
        {
            # Font
            "font.family": "serif",
            "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            # Uniform sizing
            "font.size": FONT_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "figure.titlesize": TITLE_SIZE,
            # High-quality output
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "figure.dpi": 100,
        }
    )


# ---------------------------------------------------------------------------
# Plotly layout defaults (mirrors rcParams for Plotly figures)
# ---------------------------------------------------------------------------

def get_plotly_layout(**overrides) -> dict:
    """Return a Plotly ``update_layout`` dict matching the publication style.

    Any key-word argument is merged on top of the defaults so callers can
    override individual fields.
    """
    defaults = {
        "font": dict(family="Times New Roman, STIXGeneral, serif", size=FONT_SIZE),
        "title_font": dict(
            family="Times New Roman, STIXGeneral, serif",
            size=TITLE_SIZE,
        ),
        "legend": dict(
            font=dict(
                family="Times New Roman, STIXGeneral, serif", size=FONT_SIZE
            ),
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5,
        ),
    }
    defaults.update(overrides)
    return defaults
