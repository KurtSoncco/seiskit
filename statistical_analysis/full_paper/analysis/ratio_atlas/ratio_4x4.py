"""4×4 atlas of 1D-normalized |TF| and PGA ratios vs distance from center.

One fixed factorial cell; each panel is a single seed. Each metric is its own
figure, so |TF| and PGA are not overlaid. Default cell is the paper reference
(Vs1=230, H=50, CoV=0.2, r_h=30, a_hv=10). The default run writes six pages
per metric (seeds 1–16, 17–32, …, 81–96). Seeds 97–100 are omitted because a
page holds 16 seeds.

Writes PDFs under ``figure_dir("ratio_atlas")`` on Box.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

_FULL_PAPER = Path(__file__).resolve().parents[2]
_CODE = _FULL_PAPER / "analysis" / "code"
for _p in (_FULL_PAPER, _CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _shared import CENTER_NODE, DATA_PATH, DX_M  # noqa: E402
from config import (  # noqa: E402
    DATA_LINEWIDTH,
    LABEL_FONTSIZE,
    REF_COLOR,
    TICK_LABELSIZE,
    add_panel_label,
    apply_full_paper_style,
    figsize,
    figure_dir,
    metric_color,
    metric_label,
    save_figure,
)

apply_full_paper_style(auto_format=True, frame="open", grid=False)

METRICS = ("abs_TF_ratio", "PGA_ratio")
N_PANEL = 4
N_PANELS = N_PANEL * N_PANEL

DEFAULT_VS1 = 230.0
DEFAULT_HEIGHT = 50.0
DEFAULT_COV = 0.2
DEFAULT_RH = 30.0
DEFAULT_AHV = 10.0
DEFAULT_BLOCKS = 6
N_SEEDS_AVAILABLE = 100

GRID_ALPHA = 0.18
TEXT_BBOX = {"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.6}
LEGEND_FRAME = {
    "frameon": True,
    "fancybox": False,
    "framealpha": 0.75,
    "facecolor": "white",
    "edgecolor": "none",
    "borderpad": 0.3,
}

DOC_WIDTH, FIG_HEIGHT = figsize(aspect=0.92)


def parse_seeds(spec: str) -> list[int]:
    """Parse ``1-16`` or ``1,2,5`` into exactly 16 seed ids."""
    seeds: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError(f"seed range {part!r} is inverted")
            seeds.extend(range(lo, hi + 1))
        else:
            seeds.append(int(part))
    if len(seeds) != N_PANELS:
        raise ValueError(f"need exactly {N_PANELS} seeds for a 4×4 atlas, got {len(seeds)}")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed list contains duplicates")
    return seeds


def _fmt_num(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def seed_blocks(n_blocks: int, *, first: int = 1) -> list[list[int]]:
    """Consecutive non-overlapping groups of 16 seed ids."""
    if n_blocks < 1:
        raise ValueError("n_blocks must be at least 1")
    last = first + n_blocks * N_PANELS - 1
    if first < 1 or last > N_SEEDS_AVAILABLE:
        raise ValueError(
            f"{n_blocks} blocks of {N_PANELS} starting at seed {first} "
            f"exceed seeds 1–{N_SEEDS_AVAILABLE}"
        )
    return [list(range(first + i * N_PANELS, first + (i + 1) * N_PANELS)) for i in range(n_blocks)]


def out_stem(
    vs1: float,
    height: float,
    cov: float,
    rh: float,
    ahv: float,
    seed_ids: list[int],
    metric: str,
) -> str:
    short = "abs_TF" if metric == "abs_TF_ratio" else "PGA"
    return (
        f"ratio_4x4_{short}_h{height:.0f}_vs1_{vs1:.0f}"
        f"_cov{_fmt_num(cov)}_rh{rh:.0f}_ahv{ahv:.0f}"
        f"_seeds{seed_ids[0]}-{seed_ids[-1]}"
    )


def load_cell(
    *,
    vs1: float,
    height: float,
    cov: float,
    rh: float,
    ahv: float,
    path: Path = DATA_PATH,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Return (channels, seeds, {metric: values}) for one factorial cell.

    Arrays are flat and aligned. Channels are the raw node ids.
    """
    with h5py.File(path, "r") as f:
        g = f["master"]
        mask = (
            (g["Vs1"][:] == vs1)
            & (g["Height"][:] == height)
            & np.isclose(g["CoV"][:], cov)
            & (g["rH"][:] == rh)
            & (g["aHV"][:] == ahv)
        )
        if not np.any(mask):
            raise ValueError(
                f"no rows for Vs1={vs1:g}, H={height:g}, CoV={cov:g}, rH={rh:g}, aHV={ahv:g}"
            )
        channels = np.asarray(g["channel"][:][mask], dtype=int)
        seeds = np.asarray(g["seed"][:][mask], dtype=int)
        series = {metric: np.asarray(g[metric][:][mask], dtype=float) for metric in METRICS}
    return channels, seeds, series


def _seed_profile(
    channels: np.ndarray,
    seeds: np.ndarray,
    values: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (distance_m, values) for one seed, sorted along the array."""
    sel = seeds == seed
    if not np.any(sel):
        raise ValueError(f"seed {seed} is missing from this cell")
    x = (channels[sel].astype(float) - CENTER_NODE) * DX_M
    y = values[sel]
    order = np.argsort(x)
    return x[order], y[order]


def _shared_ylim(profiles: list[dict[str, np.ndarray]]) -> tuple[float, float]:
    """Limits covering the plotted curves, padded, and including 0 and 1."""
    chunks = [y for prof in profiles for y in prof.values()]
    vals = np.concatenate(chunks)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return (0.0, 1.05)
    ymin = min(0.0, float(np.min(finite)))
    ymax = float(np.max(finite))
    pad = 0.06 * (ymax - ymin if ymax > ymin else 1.0)
    return ymin, ymax + pad


def plot_atlas(
    channels: np.ndarray,
    seeds: np.ndarray,
    series: dict[str, np.ndarray],
    seed_ids: list[int],
    metric: str,
    *,
    vs1: float,
    height: float,
    cov: float,
    rh: float,
    ahv: float,
    ylim: tuple[float, float] | None = None,
) -> plt.Figure:
    profiles: list[tuple[np.ndarray, np.ndarray]] = []
    for seed in seed_ids:
        profiles.append(_seed_profile(channels, seeds, series[metric], seed))
    if ylim is None:
        ylim = _shared_ylim([{metric: y} for _x, y in profiles])
    x0 = profiles[0][0]
    xlim = (float(x0[0]), float(x0[-1]))
    color = metric_color(metric)

    fig = plt.figure(figsize=(DOC_WIDTH, FIG_HEIGHT))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.07, 1.0],
        hspace=0.02,
        left=0.09,
        right=0.995,
        bottom=0.08,
        top=0.99,
    )
    header = fig.add_subplot(gs[0, 0])
    header.axis("off")
    gs_panels = gs[1, 0].subgridspec(N_PANEL, N_PANEL, wspace=0.16, hspace=0.10)
    axes = np.empty((N_PANEL, N_PANEL), dtype=object)
    for r in range(N_PANEL):
        for c in range(N_PANEL):
            sharex = axes[0, 0] if (r, c) != (0, 0) else None
            sharey = axes[0, 0] if (r, c) != (0, 0) else None
            axes[r, c] = fig.add_subplot(gs_panels[r, c], sharex=sharex, sharey=sharey)

    for i, (seed, (x, y)) in enumerate(zip(seed_ids, profiles)):
        ax = axes.flat[i]
        ax.axhline(1.0, color=REF_COLOR, lw=DATA_LINEWIDTH, zorder=1)
        ax.plot(x, y, color=color, lw=DATA_LINEWIDTH, zorder=3)
        add_panel_label(ax, i, alpha=0.75)
        ax.text(
            0.02,
            0.97,
            f"seed {seed}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=TICK_LABELSIZE,
            zorder=6,
            bbox=TEXT_BBOX,
        )
        ax.tick_params(labelsize=TICK_LABELSIZE)
        ax.grid(True, which="major", alpha=GRID_ALPHA, lw=0.6)
        ax.set_axisbelow(True)
        row, col = divmod(i, N_PANEL)
        if row != N_PANEL - 1:
            ax.tick_params(labelbottom=False)
        if col != 0:
            ax.tick_params(labelleft=False)

    xpad = 18.0
    axes[0, 0].set_xlim(xlim[0] - xpad, xlim[1] + xpad)
    axes[0, 0].set_ylim(*ylim)
    axes[0, 0].set_xticks([-100, 0, 100])
    fig.supxlabel("Distance from center (m)", fontsize=LABEL_FONTSIZE, y=0.01)
    fig.supylabel(metric_label(metric), fontsize=LABEL_FONTSIZE, x=0.01)

    header.text(
        0.5,
        0.98,
        (
            rf"$V_{{s1}} = {vs1:.0f}$ m/s, $H = {height:.0f}$ m, "
            rf"$\mathrm{{CoV}} = {cov:g}$, $r_h = {rh:.0f}$ m, $a_{{hv}} = {ahv:.0f}$"
        ),
        transform=header.transAxes,
        ha="center",
        va="top",
        fontsize=TICK_LABELSIZE,
    )
    handles = [
        Line2D([0], [0], color=color, lw=DATA_LINEWIDTH, label=metric_label(metric)),
    ]
    if ylim[0] <= 1.0 <= ylim[1]:
        handles.append(Line2D([0], [0], color=REF_COLOR, lw=DATA_LINEWIDTH, label="1D baseline"))
    header.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        fontsize=TICK_LABELSIZE,
        handlelength=1.8,
        columnspacing=1.0,
        borderaxespad=0.0,
        labelspacing=0.1,
        bbox_to_anchor=(0.5, 0.0),
        **LEGEND_FRAME,
    )
    return fig


def _ylim_for_seeds(
    channels: np.ndarray,
    seeds: np.ndarray,
    series: dict[str, np.ndarray],
    seed_ids: list[int],
    metric: str,
) -> tuple[float, float]:
    """One y-range for every page of one metric, so seed blocks share a scale."""
    chunks: list[np.ndarray] = []
    for seed in seed_ids:
        _, y = _seed_profile(channels, seeds, series[metric], seed)
        chunks.append(y)
    finite = np.concatenate(chunks)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (0.0, 1.05)
    ymin = min(0.0, float(np.min(finite)))
    ymax = float(np.max(finite))
    pad = 0.06 * (ymax - ymin if ymax > ymin else 1.0)
    return ymin, ymax + pad


def main() -> None:
    p = argparse.ArgumentParser(description="4×4 seed atlas of |TF| and PGA ratios.")
    p.add_argument("--vs1", type=float, default=DEFAULT_VS1)
    p.add_argument("--height", type=float, default=DEFAULT_HEIGHT)
    p.add_argument("--cov", type=float, default=DEFAULT_COV)
    p.add_argument("--rh", type=float, default=DEFAULT_RH)
    p.add_argument("--ahv", type=float, default=DEFAULT_AHV)
    p.add_argument(
        "--seeds",
        default=None,
        help="One page of exactly 16 seed ids, e.g. 1-16. Overrides --blocks.",
    )
    p.add_argument(
        "--blocks",
        type=int,
        default=DEFAULT_BLOCKS,
        help="Consecutive 16-seed pages starting at seed 1 (default: 6, seeds 1–96).",
    )
    args = p.parse_args()
    if args.seeds is not None:
        blocks = [parse_seeds(args.seeds)]
    else:
        blocks = seed_blocks(args.blocks)

    channels, seeds, series = load_cell(
        vs1=args.vs1,
        height=args.height,
        cov=args.cov,
        rh=args.rh,
        ahv=args.ahv,
    )
    all_seeds = [seed for block in blocks for seed in block]
    out_dir = figure_dir("ratio_atlas")
    for metric in METRICS:
        ylim = _ylim_for_seeds(channels, seeds, series, all_seeds, metric)
        for seed_ids in blocks:
            fig = plot_atlas(
                channels,
                seeds,
                series,
                seed_ids,
                metric,
                vs1=args.vs1,
                height=args.height,
                cov=args.cov,
                rh=args.rh,
                ahv=args.ahv,
                ylim=ylim,
            )
            stem = out_stem(args.vs1, args.height, args.cov, args.rh, args.ahv, seed_ids, metric)
            save_figure(fig, stem, out_dir=out_dir)
            plt.close(fig)


if __name__ == "__main__":
    main()
