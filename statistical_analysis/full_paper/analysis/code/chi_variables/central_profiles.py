"""Central-tendency profile figures for χ ratios (abs_TF_ratio by default).

Two Nature-width figure families under
``figure_dir("chi_variables", "central_profiles")``, each as 3×3 panels
matching the TF qualitative cross layout (vary rH / CoV / aHV; center column
shared), one PDF per (Height, Vs1):

1. Node profiles — geomean and median across seeds vs node
2. Seed profiles — geomean and median across nodes vs seed

Factor-sweep geomean boxplots live in ``geomean_factor_cross.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import (  # noqa: E402
    BOX_ROOT,
    DATA_LINEWIDTH,
    LABEL_FONTSIZE,
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

DATA_PATH = BOX_ROOT / "peak_analysis" / "join_master.h5"
METRIC = "abs_TF_ratio"
LOG_Y_METRICS = frozenset({"abs_TF_ratio", "Ia_ratio"})

H_LIST = [15.0, 50.0, 100.0]
VS1_LIST = [100.0, 230.0, 360.0]

# (rH, CoV, aHV) for panels (a)–(i), row-major — same as TF qualitative
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

N_SEED_TRACES = 20
RNG = np.random.default_rng(42)
SAMPLE_COLOR = "0.55"
SAMPLE_ALPHA = 0.5
SAMPLE_LW = 0.35
GRID_ALPHA = 0.18
Y_LIM = (1e-2, 1e1)
TEXT_BBOX = {"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.8}
LEGEND_FRAME = {
    "frameon": True,
    "fancybox": False,
    "framealpha": 0.75,
    "facecolor": "white",
    "edgecolor": "none",
    "borderpad": 0.3,
}

DOC_WIDTH, FIG_HEIGHT = figsize(aspect=0.88)


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
        METRIC,
    ]
    with h5py.File(path, "r") as f:
        g = f["master"]
        df = pd.DataFrame({c: g[c][:] for c in cols})
    return df.rename(columns={"channel": "node"})


def _panel_param_text(rh: float, cov: float, ahv: float) -> str:
    return (
        rf"$r_h = {rh:.0f}$ m" + "\n"
        rf"$\mathrm{{CoV}} = {cov:g}$" + "\n"
        rf"$a_{{hv}} = {ahv:.0f}$"
    )


def _stem(kind: str, h: float, vs1: float) -> str:
    short = METRIC.replace("_ratio", "")
    return f"{short}_{kind}_h{h:.0f}_vs1_{vs1:.0f}"


def cell_matrix(
    df_hv: pd.DataFrame, rh: float, cov: float, ahv: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (nodes, seeds, chi[node, seed]) for one design cell."""
    mask = (df_hv["rH"] == rh) & (df_hv["CoV"] == cov) & (df_hv["aHV"] == ahv)
    sub = df_hv.loc[mask]
    piv = sub.pivot(index="node", columns="seed", values=METRIC)
    arr = piv.to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        arr = np.where(np.isfinite(arr) & (arr > 0), arr, np.nan)
    return piv.index.to_numpy(), piv.columns.to_numpy(), arr


def node_profiles(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Geomean and median across seeds (axis=1) at each node."""
    with np.errstate(invalid="ignore", divide="ignore"):
        log_a = np.log(arr)
    geo = np.exp(np.nanmean(log_a, axis=1))
    med = np.nanmedian(arr, axis=1)
    return geo, med


def seed_profiles(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Geomean and median across nodes (axis=0) at each seed."""
    with np.errstate(invalid="ignore", divide="ignore"):
        log_a = np.log(arr)
    geo = np.exp(np.nanmean(log_a, axis=0))
    med = np.nanmedian(arr, axis=0)
    return geo, med


def _make_3x3_figure(
    *,
    h: float,
    vs1: float,
    legend_handles: list,
) -> tuple[plt.Figure, np.ndarray]:
    fig = plt.figure(figsize=(DOC_WIDTH, FIG_HEIGHT))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.05, 1.0],
        hspace=0.02,
        left=0.08,
        right=0.995,
        bottom=0.06,
        top=0.99,
    )
    header = fig.add_subplot(gs[0, 0])
    header.axis("off")
    gs_panels = gs[1, 0].subgridspec(3, 3, wspace=0.06, hspace=0.08)
    axes = np.empty((3, 3), dtype=object)
    for r in range(3):
        for c in range(3):
            # Share x and y across the full 3×3 (tick labels only on first column / bottom row).
            sharex = axes[0, 0] if (r, c) != (0, 0) else None
            sharey = axes[0, 0] if (r, c) != (0, 0) else None
            axes[r, c] = fig.add_subplot(gs_panels[r, c], sharex=sharex, sharey=sharey)

    header.text(
        0.5,
        0.95,
        rf"($H = {h:.0f}$ m, $V_{{s1}} = {vs1:.0f}$ m/s; {metric_label(METRIC)})",
        transform=header.transAxes,
        ha="center",
        va="top",
        fontsize=TICK_LABELSIZE,
        bbox=TEXT_BBOX,
    )
    header.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        fontsize=TICK_LABELSIZE,
        handlelength=1.8,
        columnspacing=1.0,
        borderaxespad=0.0,
        labelspacing=0.1,
        bbox_to_anchor=(0.5, -0.05),
        **LEGEND_FRAME,
    )
    return fig, axes


def _annotate_panel(ax: plt.Axes, i: int, rh: float, cov: float, ahv: float) -> None:
    add_panel_label(ax, i, alpha=0.75)
    ax.text(
        0.02,
        0.97,
        _panel_param_text(rh, cov, ahv),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=TICK_LABELSIZE,
        linespacing=1.15,
        zorder=6,
        bbox=TEXT_BBOX,
    )
    ax.tick_params(labelsize=TICK_LABELSIZE)
    ax.grid(True, which="major", alpha=GRID_ALPHA, lw=0.6)
    ax.set_axisbelow(True)


def _apply_shared_ylims(axes: np.ndarray, *, log_y: bool) -> None:
    """Set fixed shared y-limits on the master axis (propagates to all panels)."""
    ax = axes[0, 0]
    if log_y:
        ax.set_yscale("log")
    ax.set_ylim(*Y_LIM)


def plot_node_profiles(df: pd.DataFrame, *, h: float, vs1: float, out_dir: Path) -> Path:
    out_dir = out_dir / "node_profiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    color = metric_color(METRIC)
    log_y = METRIC in LOG_Y_METRICS
    df_hv = df[(df["Height"] == h) & (df["Vs1"] == vs1)]

    legend_handles = [
        Line2D([0], [0], color=SAMPLE_COLOR, lw=DATA_LINEWIDTH, label="Seeds (subset)"),
        Line2D([0], [0], color=color, ls="-", lw=DATA_LINEWIDTH, label="Geomean across seeds"),
        Line2D([0], [0], color=color, ls="--", lw=DATA_LINEWIDTH, label="Median across seeds"),
    ]
    fig, axes = _make_3x3_figure(
        h=h,
        vs1=vs1,
        legend_handles=legend_handles,
    )

    cell_data: list[tuple] = []
    for rh, cov, ahv in PANELS:
        nodes, seeds, arr = cell_matrix(df_hv, rh, cov, ahv)
        geo, med = node_profiles(arr)
        cell_data.append((nodes, seeds, arr, geo, med))

    for i, ((rh, cov, ahv), (nodes, seeds, arr, geo, med)) in enumerate(zip(PANELS, cell_data)):
        ax = axes.flat[i]
        # faint seed traces
        n_seeds = arr.shape[1]
        n_show = min(N_SEED_TRACES, n_seeds)
        idx = RNG.choice(n_seeds, size=n_show, replace=False) if n_seeds else []
        for j in idx:
            ax.plot(
                nodes,
                arr[:, j],
                color=SAMPLE_COLOR,
                lw=SAMPLE_LW,
                alpha=SAMPLE_ALPHA,
                zorder=1,
            )
        ax.plot(nodes, geo, color=color, ls="-", lw=DATA_LINEWIDTH, zorder=4)
        ax.plot(nodes, med, color=color, ls="--", lw=DATA_LINEWIDTH, zorder=4)
        _annotate_panel(ax, i, rh, cov, ahv)

        row, col = divmod(i, 3)
        if row == 2:
            ax.set_xlabel("Node", fontsize=LABEL_FONTSIZE)
        else:
            ax.tick_params(labelbottom=False)
        if col == 0:
            ax.set_ylabel(metric_label(METRIC), fontsize=LABEL_FONTSIZE)
        else:
            ax.tick_params(labelleft=False)

    _apply_shared_ylims(axes, log_y=log_y)

    paths = save_figure(fig, _stem("node_profile", h, vs1), out_dir=out_dir)
    plt.close(fig)
    return paths[0]


def plot_seed_profiles(df: pd.DataFrame, *, h: float, vs1: float, out_dir: Path) -> Path:
    out_dir = out_dir / "seed_profiles"
    out_dir.mkdir(parents=True, exist_ok=True)
    color = metric_color(METRIC)
    log_y = METRIC in LOG_Y_METRICS
    df_hv = df[(df["Height"] == h) & (df["Vs1"] == vs1)]

    legend_handles = [
        Line2D([0], [0], color=color, ls="-", lw=DATA_LINEWIDTH, label="Geomean across nodes"),
        Line2D([0], [0], color=color, ls="--", lw=DATA_LINEWIDTH, label="Median across nodes"),
    ]
    fig, axes = _make_3x3_figure(
        h=h,
        vs1=vs1,
        legend_handles=legend_handles,
    )

    cell_data: list[tuple] = []
    for rh, cov, ahv in PANELS:
        nodes, seeds, arr = cell_matrix(df_hv, rh, cov, ahv)
        geo, med = seed_profiles(arr)
        cell_data.append((seeds, geo, med))

    for i, ((rh, cov, ahv), (seeds, geo, med)) in enumerate(zip(PANELS, cell_data)):
        ax = axes.flat[i]
        ax.plot(seeds, geo, color=color, ls="-", lw=DATA_LINEWIDTH, zorder=4)
        ax.plot(seeds, med, color=color, ls="--", lw=DATA_LINEWIDTH, zorder=4)
        _annotate_panel(ax, i, rh, cov, ahv)

        row, col = divmod(i, 3)
        if row == 2:
            ax.set_xlabel("Seed", fontsize=LABEL_FONTSIZE)
        else:
            ax.tick_params(labelbottom=False)
        if col == 0:
            ax.set_ylabel(metric_label(METRIC), fontsize=LABEL_FONTSIZE)
        else:
            ax.tick_params(labelleft=False)

    _apply_shared_ylims(axes, log_y=log_y)

    paths = save_figure(fig, _stem("seed_profile", h, vs1), out_dir=out_dir)
    plt.close(fig)
    return paths[0]


def main() -> None:
    out_dir = figure_dir("chi_variables", "central_profiles")
    print(f"Loading {DATA_PATH} …")
    df = load_ratios()
    print(f"  rows={len(df):,}  metric={METRIC}")

    for h in H_LIST:
        for vs1 in VS1_LIST:
            print(f"  H={h:.0f}, Vs1={vs1:.0f} …")
            p1 = plot_node_profiles(df, h=h, vs1=vs1, out_dir=out_dir)
            # p2 = plot_seed_profiles(df, h=h, vs1=vs1, out_dir=out_dir)
            print(f"    {p1.name}")
            # print(f"    {p2.name}")

    print(f"Done → {out_dir}")


if __name__ == "__main__":
    main()
