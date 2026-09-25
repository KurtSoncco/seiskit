r"""Direct log-standard-deviation profiles for seed and node geomeans.

Writes one Nature-width PDF for a selected (Height, Vs1) pair. Each panel
contains grouped bars for every chi metric:

* ``std_j(ln(G_seed,j / G_global))`` measures realization variability.
* ``std_i(ln(G_node,i / G_global))`` measures spatial variability.

The panel layout matches the other chi-variable figures: a 3x3 factor grid
varying r_h, CoV, and a_hv. Outputs are written under
``figure_dir("chi_variables", "geomean_log_std")``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import (  # noqa: E402
    BOX_ROOT,
    LABEL_FONTSIZE,
    METRICS,
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
# One case by design; change these two values for another Height/Vs1 pair.
HEIGHT = 15.0
VS1 = 100.0

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

METRIC_GAP = 1.0
SEED_OFFSET = -0.18
NODE_OFFSET = 0.18
BAR_WIDTH = 0.30
SEED_HATCH = "..."
NODE_HATCH = "///"
Y_LIM = (0.0, 0.4)
GRID_ALPHA = 0.18
TEXT_BBOX = {"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.6}
LEGEND_FRAME = {
    "frameon": True,
    "fancybox": False,
    "framealpha": 0.75,
    "facecolor": "white",
    "edgecolor": "none",
    "borderpad": 0.25,
}

DOC_WIDTH, FIG_HEIGHT = figsize(aspect=0.88)


def load_ratios(path: Path = DATA_PATH) -> pd.DataFrame:
    cols = ["Vs1", "Height", "CoV", "rH", "aHV", "channel", "seed", *METRICS]
    with h5py.File(path, "r") as f:
        group = f["master"]
        df = pd.DataFrame({column: group[column][:] for column in cols})
    return df.rename(columns={"channel": "node"})


def cell_matrix(
    df_hv: pd.DataFrame,
    rh: float,
    cov: float,
    ahv: float,
    metric: str,
) -> np.ndarray:
    mask = (df_hv["rH"] == rh) & (df_hv["CoV"] == cov) & (df_hv["aHV"] == ahv)
    sub = df_hv.loc[mask]
    pivot = sub.pivot(index="node", columns="seed", values=metric)
    values = pivot.to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(values) & (values > 0), values, np.nan)


def seed_geomeans(values: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.exp(np.nanmean(np.log(values), axis=0))


def node_geomeans(values: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.exp(np.nanmean(np.log(values), axis=1))


def log_std(values: np.ndarray) -> float:
    clean = values[np.isfinite(values) & (values > 0)]
    if clean.size == 0:
        return float("nan")
    return float(np.std(np.log(clean)))


def geomean_log_stds(values: np.ndarray) -> tuple[float, float]:
    return log_std(seed_geomeans(values)), log_std(node_geomeans(values))


def _panel_param_text(rh: float, cov: float, ahv: float) -> str:
    return (
        rf"$r_h = {rh:.0f}$ m" + "\n"
        rf"$\mathrm{{CoV}} = {cov:g}$" + "\n"
        rf"$a_{{hv}} = {ahv:.0f}$"
    )


def _make_figure(*, height: float, vs1: float) -> tuple[plt.Figure, np.ndarray]:
    fig = plt.figure(figsize=(DOC_WIDTH, FIG_HEIGHT))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.06, 1.0],
        hspace=0.02,
        left=0.08,
        right=0.995,
        bottom=0.06,
        top=0.99,
    )
    header = fig.add_subplot(grid[0, 0])
    header.axis("off")
    panel_grid = grid[1, 0].subgridspec(3, 3, wspace=0.08, hspace=0.10)
    axes = np.empty((3, 3), dtype=object)
    for row in range(3):
        for col in range(3):
            sharex = axes[0, 0] if (row, col) != (0, 0) else None
            sharey = axes[0, 0] if (row, col) != (0, 0) else None
            axes[row, col] = fig.add_subplot(panel_grid[row, col], sharex=sharex, sharey=sharey)

    header.text(
        0.5,
        0.95,
        rf"($H = {height:.0f}$ m, $V_{{s1}} = {vs1:.0f}$ m/s)",
        transform=header.transAxes,
        ha="center",
        va="top",
        fontsize=TICK_LABELSIZE,
        bbox=TEXT_BBOX,
    )
    header.legend(
        handles=[
            Patch(
                facecolor="0.65",
                edgecolor="0.2",
                hatch=SEED_HATCH,
                label=r"$\sigma_{\ln(G_{\mathrm{seed}}/G_{\mathrm{global}})}$",
            ),
            Patch(
                facecolor="0.65",
                edgecolor="0.2",
                hatch=NODE_HATCH,
                label=r"$\sigma_{\ln(G_{\mathrm{node}}/G_{\mathrm{global}})}$",
            ),
        ],
        loc="lower center",
        ncol=2,
        fontsize=TICK_LABELSIZE,
        handlelength=1.8,
        handleheight=1.4,
        columnspacing=1.0,
        borderaxespad=0.0,
        labelspacing=0.1,
        bbox_to_anchor=(0.5, -0.15),
        **LEGEND_FRAME,
    )
    return fig, axes


def _annotate_panel(ax: plt.Axes, index: int, rh: float, cov: float, ahv: float) -> None:
    add_panel_label(ax, index, alpha=0.75)
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
    ax.grid(True, which="major", axis="y", alpha=GRID_ALPHA, lw=0.6)
    ax.set_axisbelow(True)


def plot_log_std(df: pd.DataFrame, *, height: float, vs1: float, out_dir: Path) -> Path:
    df_hv = df[(df["Height"] == height) & (df["Vs1"] == vs1)]
    fig, axes = _make_figure(height=height, vs1=vs1)
    tick_pos = np.arange(len(METRICS), dtype=float) * METRIC_GAP
    tick_labels = [metric_label(metric) for metric in METRICS]

    for index, (rh, cov, ahv) in enumerate(PANELS):
        ax = axes.flat[index]
        seed_values: list[float] = []
        node_values: list[float] = []
        colors: list[str] = []
        for metric in METRICS:
            seed_std, node_std = geomean_log_stds(cell_matrix(df_hv, rh, cov, ahv, metric))
            seed_values.append(seed_std)
            node_values.append(node_std)
            colors.append(metric_color(metric))

        seed_heights = np.nan_to_num(seed_values, nan=0.0)
        node_heights = np.nan_to_num(node_values, nan=0.0)
        ax.bar(
            tick_pos + SEED_OFFSET,
            seed_heights,
            width=BAR_WIDTH,
            color=colors,
            alpha=0.72,
            hatch=SEED_HATCH,
            edgecolor="0.2",
            linewidth=0.55,
            label=r"$\sigma_{\ln(G_{\mathrm{seed}}/G_{\mathrm{global}})}$",
        )
        ax.bar(
            tick_pos + NODE_OFFSET,
            node_heights,
            width=BAR_WIDTH,
            color=colors,
            alpha=0.72,
            hatch=NODE_HATCH,
            edgecolor="0.2",
            linewidth=0.55,
            label=r"$\sigma_{\ln(G_{\mathrm{node}}/G_{\mathrm{global}})}$",
        )
        ax.set_xlim(tick_pos[0] - 0.55, tick_pos[-1] + 0.55)
        ax.set_ylim(*Y_LIM)
        ax.set_xticks(tick_pos)
        _annotate_panel(ax, index, rh, cov, ahv)

        row, col = divmod(index, 3)
        if row == 2:
            ax.set_xticklabels(tick_labels, fontsize=TICK_LABELSIZE - 0.5)
        else:
            ax.tick_params(labelbottom=False)
        if col == 0:
            ax.set_ylabel(r"$\sigma_{\ln(G/G_{\mathrm{global}})}$", fontsize=LABEL_FONTSIZE)
            ax.tick_params(labelleft=True)
        else:
            ax.tick_params(labelleft=False)

    stem = f"geomean_log_std_h{height:.0f}_vs1_{vs1:.0f}"
    paths = save_figure(fig, stem, out_dir=out_dir)
    plt.close(fig)
    return paths[0]


def build_summary_md(written: Path, *, height: float, vs1: float) -> str:
    return "\n".join(
        [
            "# Direct log-standard-deviation profiles",
            "",
            rf"One 3x3 factor-grid figure for $H={height:.0f}$ m and $V_{{s1}}={vs1:.0f}$ m/s.",
            "",
            r"- Seed bars: \(\sigma_{\ln(G_{\mathrm{seed}}/G_{\mathrm{global}})}=\operatorname{std}_j[\ln(G_{\mathrm{seed},j}/G_{\mathrm{global}})]\)",
            r"- Node bars: \(\sigma_{\ln(G_{\mathrm{node}}/G_{\mathrm{global}})}=\operatorname{std}_i[\ln(G_{\mathrm{node},i}/G_{\mathrm{global}})]\)",
            r"- Since \(G_{\mathrm{global}}\) is constant within a cell, these equal the log-SD of the unnormalized geomeans.",
            r"- Standard deviations use the population convention (\(ddof=0\)) and ignore invalid/non-positive values.",
            r"- Shared linear y-axis: \([0.0, 0.4]\)",
            "",
            "## Output",
            "",
            "| File | Content |",
            "| --- | --- |",
            f"| `{written.name}` | Direct seed/node log-standard-deviation profiles |",
            "",
        ]
    )


def main() -> None:
    out_dir = figure_dir("chi_variables", "geomean_log_std")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("geomean_log_std_*.pdf"):
        old.unlink()
        print(f"  removed {old.name}")

    print(f"Loading {DATA_PATH} ...")
    df = load_ratios()
    print(f"  rows={len(df):,}")
    print(f"  H={HEIGHT:.0f}, Vs1={VS1:.0f} -> {out_dir}")
    written = plot_log_std(df, height=HEIGHT, vs1=VS1, out_dir=out_dir)
    print(f"    {written.name}")

    summary_path = out_dir / "summary.md"
    summary_path.write_text(build_summary_md(written, height=HEIGHT, vs1=VS1), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
