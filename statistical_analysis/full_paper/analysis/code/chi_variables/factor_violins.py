"""Violin grids of χ ratios across design-factor levels.

Two Nature-width figures under ``figure_dir("chi_variables", "factor_violins")``:

- ``chi_violins_freq.pdf`` — rows ``f_ratio``, ``abs_TF_ratio``; cols factors
- ``chi_violins_im.pdf`` — rows ``PGA_ratio``, ``PSA_ratio``, ``Ia_ratio``

Each panel: all node×seed observations at that factor level (marginal over
other factors), violin shape from a fixed subsample, P5 / median / P95 from
the full finite sample connected across levels.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import (  # noqa: E402
    BOX_ROOT,
    DATA_LINEWIDTH,
    FACTORS,
    LABEL_FONTSIZE,
    add_panel_label,
    apply_full_paper_style,
    figsize,
    figure_dir,
    metric_color,
    metric_label,
    save_figure,
)

apply_full_paper_style(auto_format=True, frame="open", grid=False)

DATA_PATH = BOX_ROOT / "peak_analysis" / "join_master.h5"
METRICS = ("f_ratio", "abs_TF_ratio", "PGA_ratio", "PSA_ratio", "Ia_ratio")
FREQ_METRICS = ("f_ratio", "abs_TF_ratio")
IM_METRICS = ("PGA_ratio", "PSA_ratio", "Ia_ratio")

VIOLIN_N = 8_000
RNG = np.random.default_rng(42)
VIOLIN_ALPHA = 0.60
LINE_ALPHA = 0.95
LOG_Y_METRICS = frozenset({"abs_TF_ratio", "Ia_ratio"})
GRID_ALPHA = 0.5

PCT_STYLES = {
    "median": {"ls": "-", "label": "Median"},
    "p5": {"ls": "--", "label": "P5"},
    "p95": {"ls": ":", "label": "P95"},
}

Y_LIMS = {
    "f_ratio": (0.4, 2.0),
    "abs_TF_ratio": (1e-2, 1e1),
    "PGA_ratio": (0.0, 2.5),
    "PSA_ratio": (0.0, 2.5),
    "Ia_ratio": (1e-2, 1e1),
}


def load_ratios(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load joined ratio table; rename channel → node."""
    cols = [
        "Vs1",
        "Height",
        "CoV",
        "rH",
        "aHV",
        "channel",
        "seed",
        *METRICS,
    ]
    with h5py.File(path, "r") as f:
        g = f["master"]
        df = pd.DataFrame({c: g[c][:] for c in cols})
    return df.rename(columns={"channel": "node"})


def _format_level(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value)}"
    return f"{value:g}"


def _finite_positive(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x) & (x > 0)]


def _subsample(x: np.ndarray, n_max: int = VIOLIN_N) -> np.ndarray:
    if x.size <= n_max:
        return x
    idx = RNG.choice(x.size, size=n_max, replace=False)
    return x[idx]


def level_stats(
    df: pd.DataFrame,
    metric: str,
    factor: str,
) -> tuple[list[float], list[np.ndarray], list[tuple[float, float, float]]]:
    """Return (levels, violin_samples, (p5, med, p95) per level)."""
    levels = sorted(df[factor].unique())
    samples: list[np.ndarray] = []
    pcts: list[tuple[float, float, float]] = []
    for lv in levels:
        full = _finite_positive(df.loc[df[factor] == lv, metric].to_numpy())
        if full.size == 0:
            samples.append(np.array([np.nan]))
            pcts.append((np.nan, np.nan, np.nan))
            continue
        p5, med, p95 = np.percentile(full, [5, 50, 95])
        samples.append(_subsample(full))
        pcts.append((float(p5), float(med), float(p95)))
    return [float(lv) for lv in levels], samples, pcts


def _style_violins(parts: dict, color: str) -> None:
    bodies = parts.get("bodies", [])
    for body in bodies:
        if isinstance(body, PolyCollection):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(VIOLIN_ALPHA)
            body.set_linewidth(0.6)


def _violin_width(levels: list[float]) -> float:
    """Width in data units ≈ 35% of the smallest adjacent gap."""
    arr = np.asarray(levels, dtype=float)
    if arr.size < 2:
        return 0.4 * abs(arr[0]) if arr.size else 1.0
    return 0.35 * float(np.min(np.diff(arr)))


def draw_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    factor: str,
    *,
    log_y: bool,
) -> None:
    color = metric_color(metric)
    levels, samples, pcts = level_stats(df, metric, factor)
    # True numeric factor levels (not equal category indices).
    positions = np.asarray(levels, dtype=float)
    width = _violin_width(levels)

    plot_data = [s for s in samples if s.size and np.isfinite(s).any()]
    plot_pos = [p for s, p in zip(samples, positions) if s.size and np.isfinite(s).any()]
    if plot_data:
        parts = ax.violinplot(
            plot_data,
            positions=plot_pos,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            widths=width,
        )
        _style_violins(parts, color)

    p5s = [t[0] for t in pcts]
    meds = [t[1] for t in pcts]
    p95s = [t[2] for t in pcts]
    kw = dict(color=color, linewidth=DATA_LINEWIDTH, alpha=LINE_ALPHA, zorder=5)
    ax.plot(positions, meds, PCT_STYLES["median"]["ls"], **kw)
    ax.plot(positions, p5s, PCT_STYLES["p5"]["ls"], **kw)
    ax.plot(positions, p95s, PCT_STYLES["p95"]["ls"], **kw)

    ax.set_xticks(positions)
    ax.set_xticklabels([_format_level(lv) for lv in levels])
    pad = 0.6 * width
    ax.set_xlim(positions[0] - pad, positions[-1] + pad)
    ax.set_ylim(*Y_LIMS[metric])
    if log_y:
        ax.set_yscale("log")

    ax.grid(True, which="major", alpha=GRID_ALPHA, linewidth=0.4)
    ax.set_axisbelow(True)


def make_figure(
    df: pd.DataFrame,
    metrics: tuple[str, ...],
    *,
    stem: str,
    aspect: float,
    out_dir: Path,
) -> list[Path]:
    nrows, ncols = len(metrics), len(FACTORS)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize(aspect=aspect),
        sharex="col",
        sharey="row",
        constrained_layout=False,
    )
    if nrows == 1:
        axes = np.asarray(axes).reshape(1, -1)

    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.10,
        top=0.92,
        wspace=0.12,
        hspace=0.18,
    )

    panel_i = 0
    for r, metric in enumerate(metrics):
        log_y = metric in LOG_Y_METRICS
        for c, factor in enumerate(FACTORS):
            ax = axes[r, c]
            draw_panel(ax, df, metric, factor, log_y=log_y)
            add_panel_label(ax, panel_i)
            panel_i += 1

            if c == 0:
                ax.set_ylabel(metric_label(metric), fontsize=LABEL_FONTSIZE)
            else:
                ax.tick_params(labelleft=False)

            if r == nrows - 1:
                # Bare name: auto_format → LABEL_MAP via format_label
                ax.set_xlabel(factor, fontsize=LABEL_FONTSIZE)
            else:
                ax.tick_params(labelbottom=False)

    # Shared legend (line styles); color is metric-specific in panels
    handles = [
        Line2D(
            [0],
            [0],
            color="0.2",
            ls=PCT_STYLES[k]["ls"],
            lw=DATA_LINEWIDTH,
            label=PCT_STYLES[k]["label"],
        )
        for k in ("median", "p5", "p95")
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=LABEL_FONTSIZE,
        bbox_to_anchor=(0.5, 0.995),
    )

    return save_figure(fig, stem, out_dir=out_dir)


def main() -> None:
    out_dir = figure_dir("chi_variables", "factor_violins")
    print(f"Loading {DATA_PATH} …")
    df = load_ratios()
    print(f"  rows={len(df):,}")

    print("Writing freq violins …")
    make_figure(
        df,
        FREQ_METRICS,
        stem="chi_violins_freq",
        aspect=0.55,
        out_dir=out_dir,
    )
    plt.close("all")

    print("Writing IM violins …")
    make_figure(
        df,
        IM_METRICS,
        stem="chi_violins_im",
        aspect=0.75,
        out_dir=out_dir,
    )
    plt.close("all")


if __name__ == "__main__":
    main()
