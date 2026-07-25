"""Shared configuration for full-paper figures and analysis plots.

Nature figure criteria
----------------------
Geometry
  - Figure width: **7.0 in** (matches the LaTeX document; ≈178 mm).
  - Aspect typically 4:3 (``0.75``) or 3:2 (``≈ 0.667``); height ≤ **170 mm**.

Fonts (sans-serif only inside graphics)
  - Body text: Arial or Helvetica (DejaVu Sans fallback on Linux).
  - Math / Greek: sans mathtext (Symbol when available; else matching sans).
  - Never Times / Computer Modern / other serif faces in figures.

Text sizes (final printed size)
  - Axes, legends, labels: **5–7 pt** (we use 6 pt ticks, 7 pt labels).
  - Panel letters (a, b, c, …): **8 pt**, lowercase, bold.
  - No bold/italic on ordinary labels; italics only for math variables, etc.
  - Sentence case for axis labels and legends.

Line weights
  - Axes, ticks, error bars: **0.25–1 pt** (no hairlines).

Export
  - PDF only, at ≥ **350 dpi** (for rasterized artists embedded in the PDF).
  - PNG/SVG are not required for the full paper.

Conference-paper figures (6.5 in, serif OK for that venue) live under
``conference_paper/config.py``.

Scripts under ``figures/`` or ``analysis/`` typically do::

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[N]))  # → full_paper/
    from config import figsize, save_figure, apply_full_paper_style, figure_dir

    apply_full_paper_style()
    out = figure_dir("vs_rh_realizations")  # → …/complete/full_paper/figures/…

Box output layout
-----------------
``complete/full_paper/figures/<topic>/`` for manuscript figures.
Conference analysis stays under ``conference_paper/results/<topic>/{plots,data}/``
(see ``conference_paper/config.py``); Box mirror is ``complete/conference_paper/``.
"""

from __future__ import annotations

import string
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.transforms import Bbox

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BOX_ROOT = Path("/mnt/box/GIG Lab - UC Berkeley/Projects/Statistical Analysis/complete")
FULL_PAPER_ROOT = Path(__file__).resolve().parent

# Box layout: complete/full_paper/figures/<topic>/...
PAPER_OUT_ROOT = BOX_ROOT / "full_paper"
FIGURES_DIR = PAPER_OUT_ROOT / "figures"


def figure_dir(*parts: str) -> Path:
    """Return ``FIGURES_DIR / parts...``, creating the directory if needed.

    Examples
    --------
    ``figure_dir("model_scheme")``
    ``figure_dir("descriptions")``
    ``figure_dir("qualitative")``
    """
    path = FIGURES_DIR.joinpath(*parts) if parts else FIGURES_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Figure geometry (full paper / Nature)
# ---------------------------------------------------------------------------
# LaTeX document figure width is 7 in; Nature's printed max is ~178 mm (≈7.01 in).
FIG_WIDTH = 7.0  # inches — matches the LaTeX document
MAX_WIDTH_MM = FIG_WIDTH * 25.4  # ≈ 177.8 mm
ASPECT_RATIO = 0.75  # 4:3 (height / width); 3:2 ≈ 0.667 also allowed
ASPECT_RATIO_3_2 = 2.0 / 3.0
MAX_HEIGHT_MM = 170.0  # hard limit for journal submission
MAX_HEIGHT_IN = MAX_HEIGHT_MM / 25.4  # ≈ 6.69 in
FIG_DPI = 350  # minimum for raster content embedded in full-paper PDFs
FIG_FORMATS: tuple[str, ...] = ("pdf",)

# ---------------------------------------------------------------------------
# Typography (Nature)
# ---------------------------------------------------------------------------
# Prefer Arial/Helvetica; DejaVu Sans is the usual Linux fallback.
SANS_FONTS = ["Arial", "Helvetica", "DejaVu Sans"]

TICK_LABELSIZE = 6  # Nature: 5–7 pt
LABEL_FONTSIZE = 7  # axes, legends, colorbar — Nature: 5–7 pt
ANNOTATION_FONTSIZE = 6
PANEL_LABEL_SIZE = 8  # bold parenthesized (a), (b), ...
FIG_FONTSIZE = LABEL_FONTSIZE  # alias for older call sites

# ---------------------------------------------------------------------------
# Line weights (Nature: 0.25–1 pt for axes / ticks / error bars)
# ---------------------------------------------------------------------------
AXIS_LINEWIDTH = 0.6
TICK_WIDTH = 0.5
TICK_MAJOR_SIZE = 2.5
TICK_MINOR_SIZE = 1.5
DATA_LINEWIDTH = 1.0  # keep data strokes within the 1 pt ceiling
SPINE_LINEWIDTH = AXIS_LINEWIDTH

# ---------------------------------------------------------------------------
# Paul Tol palettes — Bright for factors, Muted for χ metrics
# ---------------------------------------------------------------------------
# Bright purple/gray reserved for reference overlays (not factors).
# Muted cyan/green/sand/rose skipped for metrics (too close to Bright aHV/CoV/rH/Height).
FACTORS = ["Vs1", "Height", "CoV", "rH", "aHV"]
METRICS = ("f_ratio", "abs_TF_ratio", "PGA_ratio", "PSA_ratio", "Ia_ratio")

TOL_BRIGHT = {
    "blue": "#4477AA",
    "red": "#EE6677",
    "green": "#228833",
    "yellow": "#CCBB44",
    "cyan": "#66CCEE",
    "purple": "#AA3377",
    "gray": "#BBBBBB",
}

TOL_MUTED = {
    "indigo": "#332288",
    "cyan": "#88CCEE",
    "teal": "#44AA99",
    "green": "#117733",
    "olive": "#999933",
    "sand": "#DDCC77",
    "rose": "#CC6677",
    "wine": "#882255",
    "purple": "#AA4499",
    "pale_grey": "#DDDDDD",
}

FACTOR_COLORS = {
    "Vs1": TOL_BRIGHT["blue"],
    "Height": TOL_BRIGHT["red"],
    "CoV": TOL_BRIGHT["green"],
    "rH": TOL_BRIGHT["yellow"],
    "aHV": TOL_BRIGHT["cyan"],
}

METRIC_COLORS = {
    "f_ratio": TOL_MUTED["indigo"],
    "abs_TF_ratio": TOL_MUTED["teal"],
    "PGA_ratio": TOL_MUTED["wine"],
    "PSA_ratio": TOL_MUTED["purple"],
    "Ia_ratio": TOL_MUTED["olive"],
}

# 1D-normalized ratios χ; superscript N matches conference LABEL_MAP convention.
METRIC_LABELS = {
    "f_ratio": r"$f_0^N$",
    "abs_TF_ratio": r"$TF_0^N$",
    "PGA_ratio": r"$\mathrm{PGA}^N$",
    "PSA_ratio": r"$\mathrm{PSA}^N$",
    "Ia_ratio": r"$I_a^N$",
}

REF_COLOR = TOL_BRIGHT["gray"]
COMPARE_COLOR = TOL_BRIGHT["purple"]


def factor_color(name: str) -> str:
    """Return Paul Tol Bright color for a factor (strips ``_z`` suffix)."""
    key = name[:-2] if name.endswith("_z") else name
    return FACTOR_COLORS[key]


def metric_color(name: str) -> str:
    """Return Paul Tol Muted color for a χ metric column."""
    return METRIC_COLORS[name]


def metric_label(name: str, *, log: bool = False) -> str:
    """Return the display LaTeX label for a χ metric column.

    When *log* is True, wrap as ``$\\ln(...)$`` (e.g. ``$\\ln(f_0^N)$``).
    """
    from seiskit.plot_config.labels import as_ln_label

    label = METRIC_LABELS[name]
    return as_ln_label(label) if log else label


_SUBFIG_LABELS = list(string.ascii_lowercase)


def figsize(
    height: float | None = None,
    *,
    aspect: float | None = None,
) -> tuple[float, float]:
    """Return ``(FIG_WIDTH, height)`` for matplotlib ``figsize``.

    When *height* is omitted, uses ``FIG_WIDTH * aspect`` (default
    ``ASPECT_RATIO``). Raises if the resulting height exceeds
    ``MAX_HEIGHT_MM``.
    """
    if height is None:
        height = FIG_WIDTH * (ASPECT_RATIO if aspect is None else aspect)
    height = float(height)
    if height > MAX_HEIGHT_IN + 1e-6:
        raise ValueError(
            f"Figure height {height:.3f} in ({height * 25.4:.1f} mm) exceeds "
            f"the full-paper maximum of {MAX_HEIGHT_MM:.0f} mm."
        )
    return (FIG_WIDTH, height)


def apply_export_rc() -> None:
    """Embed TrueType text in PDF/SVG (editable in Illustrator)."""
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42


def apply_full_paper_style(
    *,
    auto_format: bool = True,
    frame: str = "open",
    grid: bool = False,
) -> None:
    """Apply Nature-compliant fonts, sizes, and line weights.

    Calls :func:`seiskit.plot_config.apply_style` for shared helpers
    (optional auto-format), then overrides serif defaults with sans-serif
    Arial/Helvetica and Nature text/line limits.
    """
    from seiskit.plot_config import apply_style

    apply_style(
        auto_format=auto_format,
        font_size=LABEL_FONTSIZE,
        frame=frame,
        grid=grid,
    )

    # Resolve first installed sans face (Arial/Helvetica preferred).
    available = {f.name for f in mpl.font_manager.fontManager.ttflist}
    body = next((f for f in SANS_FONTS if f in available), "DejaVu Sans")
    # Nature's "Symbol for Greek" is an Illustrator convention. Matplotlib
    # mathtext uses the matching sans face so Latin + Greek stay sans-serif
    # (never Times / Computer Modern).
    use_dejavu_math = body == "DejaVu Sans"

    rc_fonts: dict = {
        # --- fonts (sans only; no Times / Computer Modern) ---
        "font.family": "sans-serif",
        "font.sans-serif": SANS_FONTS,
        "font.size": LABEL_FONTSIZE,
        "axes.titlesize": LABEL_FONTSIZE,
        "axes.labelsize": LABEL_FONTSIZE,
        "xtick.labelsize": TICK_LABELSIZE,
        "ytick.labelsize": TICK_LABELSIZE,
        "legend.fontsize": LABEL_FONTSIZE,
        "figure.titlesize": LABEL_FONTSIZE,
        "axes.titleweight": "normal",  # no bold on ordinary labels
        "axes.labelweight": "normal",
        "mathtext.default": "it",  # variables italic; use \\mathrm for units
    }
    if use_dejavu_math:
        rc_fonts["mathtext.fontset"] = "dejavusans"
    else:
        rc_fonts.update(
            {
                "mathtext.fontset": "custom",
                "mathtext.rm": body,
                "mathtext.it": f"{body}:italic",
                "mathtext.bf": f"{body}:bold",
                "mathtext.sf": body,
                "mathtext.tt": body,
                "mathtext.cal": body,
            }
        )

    mpl.rcParams.update(
        {
            **rc_fonts,
            # --- line weights (0.25–1 pt) ---
            "axes.linewidth": AXIS_LINEWIDTH,
            "lines.linewidth": DATA_LINEWIDTH,
            "patch.linewidth": AXIS_LINEWIDTH,
            "xtick.major.width": TICK_WIDTH,
            "ytick.major.width": TICK_WIDTH,
            "xtick.minor.width": max(0.25, TICK_WIDTH * 0.75),
            "ytick.minor.width": max(0.25, TICK_WIDTH * 0.75),
            "xtick.major.size": TICK_MAJOR_SIZE,
            "ytick.major.size": TICK_MAJOR_SIZE,
            "xtick.minor.size": TICK_MINOR_SIZE,
            "ytick.minor.size": TICK_MINOR_SIZE,
            "grid.linewidth": 0.4,
            # --- export ---
            "figure.dpi": 150,
            "savefig.dpi": FIG_DPI,
            "savefig.bbox": None,  # full-paper scripts use fixed canvas
            "savefig.pad_inches": 0.0,
        }
    )
    apply_export_rc()


def add_panel_label(
    ax: Axes,
    index: int,
    *,
    x: float = 0.98,
    y: float = 0.98,
    alpha: float = 0.7,
) -> None:
    """Panel letter: ``(a)``, ``(b)``, ... bold 8 pt, top-right by default."""
    label = _SUBFIG_LABELS[index % len(_SUBFIG_LABELS)]
    ax.text(
        x,
        y,
        f"({label})",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=PANEL_LABEL_SIZE,
        fontweight="bold",
        fontfamily="sans-serif",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": alpha,
        },
        zorder=10,
    )


def save_figure(
    fig: plt.Figure,
    stem: str,
    *,
    out_dir: Path = FIGURES_DIR,
    formats: tuple[str, ...] = FIG_FORMATS,
    dpi: int = FIG_DPI,
) -> list[Path]:
    """Save *fig* at its canvas size (no tight crop).

    Full paper defaults to PDF only. Uses a fixed ``Bbox`` matching
    ``fig.get_size_inches()`` so layout margins stay as designed.
    ``dpi`` applies to rasterized artists embedded in the PDF.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    w, h = fig.get_size_inches()
    bbox = Bbox.from_bounds(0, 0, w, h)
    written: list[Path] = []
    for ext in formats:
        path = out_dir / f"{stem}.{ext}"
        kw: dict = {"bbox_inches": bbox, "dpi": dpi}
        fig.savefig(path, **kw)
        print(f"Wrote {path} ({w:.2f} x {h:.2f} in, {dpi} dpi)")
        written.append(path)
    return written
