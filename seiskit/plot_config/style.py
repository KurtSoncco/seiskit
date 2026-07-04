"""Core style configuration: rcParams, colorblind-friendly palettes, font setup.

Call :func:`apply_style` once at the start of any script to globally set
matplotlib rcParams for publication-quality figures.

When ``auto_format=True`` is passed, Axes/Figure methods are monkey-patched
to automatically apply LaTeX label substitution, Title Case, uniform font
sizes, and legend placement — eliminating boilerplate in analysis scripts.
"""

from __future__ import annotations

import itertools
import re
from typing import Iterator, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

try:
    import seaborn as sns

    _SEABORN_AVAILABLE = True
except ImportError:  # pragma: no cover
    sns = None  # type: ignore[assignment]
    _SEABORN_AVAILABLE = False

from seiskit.plot_config.colormaps import get_crameri_cmap
from seiskit.plot_config.labels import format_label, to_title_case

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
FONT_SIZE: int = 12
TITLE_SIZE: int = 14

# ---------------------------------------------------------------------------
# Colorblind-friendly discrete palette
# ---------------------------------------------------------------------------
_WONG_CYCLE = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#000000",
]

if _SEABORN_AVAILABLE and sns is not None:
    COLORBLIND_PALETTE = sns.color_palette("colorblind", as_cmap=False)
    COLORBLIND_COLORS: list[str] = [
        f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}" for r, g, b in COLORBLIND_PALETTE
    ]
else:
    COLORBLIND_PALETTE = None
    COLORBLIND_COLORS = list(_WONG_CYCLE)

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
# Auto-format monkeypatch system (opt-in via apply_style(auto_format=True))
# ---------------------------------------------------------------------------
_MONKEYPATCHED = False
_ORIG_SET_XLABEL = Axes.set_xlabel
_ORIG_SET_YLABEL = Axes.set_ylabel
_ORIG_SET_TITLE = Axes.set_title
_ORIG_LEGEND = Axes.legend
_ORIG_SET_XTICKLABELS = Axes.set_xticklabels
_ORIG_SET_YTICKLABELS = Axes.set_yticklabels
_ORIG_IMSHOW = Axes.imshow
_ORIG_SUPTITLE = Figure.suptitle

_ACTIVE_FONT_SIZE: int = FONT_SIZE


def _patched_set_xlabel(self, xlabel, *args, **kwargs):
    kwargs.setdefault("fontsize", _ACTIVE_FONT_SIZE)
    return _ORIG_SET_XLABEL(self, to_title_case(format_label(xlabel)), *args, **kwargs)


def _patched_set_ylabel(self, ylabel, *args, **kwargs):
    kwargs.setdefault("fontsize", _ACTIVE_FONT_SIZE)
    return _ORIG_SET_YLABEL(self, to_title_case(format_label(ylabel)), *args, **kwargs)


def _patched_set_title(self, label, *args, **kwargs):
    kwargs.setdefault("fontsize", _ACTIVE_FONT_SIZE)
    kwargs.setdefault("fontweight", "bold")
    if not isinstance(label, str):
        return _ORIG_SET_TITLE(self, label, *args, **kwargs)
    # Strip subtitles: split on newline, em-dash, or colon and keep only the main part
    main = label.split("\n")[0]
    for sep in ("\u2014", " \u2014 ", " — "):
        if sep in main:
            main = main.split(sep)[0]
            break
    main = main.strip()
    formatted = to_title_case(format_label(main))
    return _ORIG_SET_TITLE(self, formatted, *args, **kwargs)


def _patched_legend(self, *args, **kwargs):
    kwargs.setdefault("loc", "best")
    kwargs.setdefault("frameon", True)
    kwargs.setdefault("fancybox", True)
    kwargs.setdefault("framealpha", 0.6)
    kwargs.setdefault("edgecolor", "none")
    kwargs.setdefault("fontsize", _ACTIVE_FONT_SIZE)
    leg = _ORIG_LEGEND(self, *args, **kwargs)
    if leg is not None:
        for txt in leg.get_texts():
            txt.set_text(format_label(txt.get_text()))
            txt.set_fontsize(kwargs["fontsize"])
    return leg


def _patched_set_xticklabels(self, labels, *args, **kwargs):
    kwargs.setdefault("fontsize", _ACTIVE_FONT_SIZE)
    mapped = [format_label(str(x)) for x in labels]
    return _ORIG_SET_XTICKLABELS(self, mapped, *args, **kwargs)


def _patched_set_yticklabels(self, labels, *args, **kwargs):
    kwargs.setdefault("fontsize", _ACTIVE_FONT_SIZE)
    mapped = [format_label(str(x)) for x in labels]
    return _ORIG_SET_YTICKLABELS(self, mapped, *args, **kwargs)


def _patched_imshow(self, X, *args, **kwargs):
    if "cmap" not in kwargs:
        kwargs["cmap"] = get_crameri_cmap("batlow")
    return _ORIG_IMSHOW(self, X, *args, **kwargs)


def _patched_suptitle(self, t, *args, **kwargs):
    kwargs.setdefault("fontweight", "bold")
    kwargs.setdefault("fontsize", _ACTIVE_FONT_SIZE)
    return _ORIG_SUPTITLE(self, to_title_case(format_label(t)), *args, **kwargs)


def _install_monkeypatches() -> None:
    global _MONKEYPATCHED
    if _MONKEYPATCHED:
        return
    Axes.set_xlabel = _patched_set_xlabel
    Axes.set_ylabel = _patched_set_ylabel
    Axes.set_title = _patched_set_title
    Axes.legend = _patched_legend
    Axes.set_xticklabels = _patched_set_xticklabels
    Axes.set_yticklabels = _patched_set_yticklabels
    Axes.imshow = _patched_imshow
    Figure.suptitle = _patched_suptitle
    _MONKEYPATCHED = True


# ---------------------------------------------------------------------------
# Public style entry point
# ---------------------------------------------------------------------------


def apply_style(
    *,
    auto_format: bool = False,
    font_size: int | None = None,
    frame: str = "boxed",
    grid: bool = True,
    cmap_name: str = "batlow",
) -> None:
    """Set matplotlib rcParams for publication-quality output.

    Parameters
    ----------
    auto_format:
        When ``True``, monkey-patch Axes/Figure methods to auto-apply
        LaTeX label substitution, Title Case, uniform sizing, and legend
        placement.  Leaves other seiskit scripts unaffected when ``False``.
    font_size:
        Base font size in points.  ``None`` keeps the module default (12).
    frame:
        Spine style: ``"open"`` (left+bottom only), ``"boxed"`` (all four),
        or ``"none"`` (no spines).
    grid:
        Whether to show grid lines.
    cmap_name:
        Default Crameri colormap name for ``imshow``.
    """
    global _ACTIVE_FONT_SIZE

    if frame not in {"open", "boxed", "none"}:
        raise ValueError(f"frame must be 'open', 'boxed', or 'none'; got {frame!r}")

    fs = font_size if font_size is not None else FONT_SIZE
    _ACTIVE_FONT_SIZE = fs

    boxed = frame == "boxed"

    rc: dict = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": fs,
        "axes.titlesize": fs,
        "axes.labelsize": fs,
        "xtick.labelsize": fs,
        "ytick.labelsize": fs,
        "legend.fontsize": fs,
        "figure.titlesize": fs,
        "axes.titleweight": "bold",
        "axes.labelweight": "normal",
        "axes.spines.top": boxed,
        "axes.spines.right": boxed,
        "axes.spines.left": frame != "none",
        "axes.spines.bottom": frame != "none",
        "axes.grid": bool(grid),
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "legend.frameon": False,
        "lines.linewidth": 1.4,
        "axes.prop_cycle": mpl.cycler(color=_WONG_CYCLE),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "figure.autolayout": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    mpl.rcParams.update(rc)

    if auto_format:
        _install_monkeypatches()


# ---------------------------------------------------------------------------
# Plotly layout defaults (mirrors rcParams for Plotly figures)
# ---------------------------------------------------------------------------


def get_plotly_layout(**overrides) -> dict:
    """Return a Plotly ``update_layout`` dict matching the publication style."""
    defaults = {
        "font": dict(family="Times New Roman, STIXGeneral, serif", size=FONT_SIZE),
        "title_font": dict(
            family="Times New Roman, STIXGeneral, serif",
            size=TITLE_SIZE,
        ),
        "legend": dict(
            font=dict(family="Times New Roman, STIXGeneral, serif", size=FONT_SIZE),
            orientation="h",
            yanchor="top",
            y=-0.12,
            xanchor="center",
            x=0.5,
        ),
    }
    defaults.update(overrides)
    return defaults
