r"""Relative seed/node geomean boxplots for all χ metrics.

One Nature-width PDF per (Height, Vs1) as a 3×3 factor grid (same layout
as TF qualitative / central profiles: vary \(r_h\), CoV, \(a_{hv}\); center
column shared):

- \(G_{\mathrm{seed},j}/G_{\mathrm{global}}\) and
  \(G_{\mathrm{node},i}/G_{\mathrm{global}}\)
- Native boxplots (IQR box, median line, \(P_{10}\)–\(P_{90}\) whiskers,
  fliers beyond) drawn straight from each seed/node cloud
- Distinct hatch for seed vs node (also in the legend)
- Shared \(y\) across panels (relative scale)

Writes under ``figure_dir("chi_variables", "geomean_relative")``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import (  # noqa: E402
    BOX_ROOT,
    DATA_LINEWIDTH,
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
# Bold black hatch lines for clear contrast against the light strip fills.
plt.rcParams["hatch.linewidth"] = 0.9
plt.rcParams["hatch.color"] = "black"

DATA_PATH = BOX_ROOT / "peak_analysis" / "join_master.h5"
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

BOX_WIDTH = 0.32
METRIC_GAP = 1.0
SEED_OFFSET = -0.22
NODE_OFFSET = 0.22
# Native Matplotlib hatch strings, shared by legend glyphs and panel boxes.
SEED_HATCH = "..."
NODE_HATCH = "///"
SEED_FACE_ALPHA = 0.22
NODE_FACE_ALPHA = 0.16
HATCH_EDGE = "0.2"
# Whiskers reach the 10th/90th percentile of each box's own cloud (clipped to
# the data range) instead of the default 1.5x-IQR rule, so the whisker span
# keeps the same P10-P90 "spread" definition the figure used previously while
# the box itself now also shows the IQR and true outliers beyond it.
WHIS = (10, 90)
Y_LIM = (0.1, 10.0)
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
        g = f["master"]
        df = pd.DataFrame({c: g[c][:] for c in cols})
    return df.rename(columns={"channel": "node"})


def _panel_param_text(rh: float, cov: float, ahv: float) -> str:
    return (
        rf"$r_h = {rh:.0f}$ m" + "\n"
        rf"$\mathrm{{CoV}} = {cov:g}$" + "\n"
        rf"$a_{{hv}} = {ahv:.0f}$"
    )


def cell_matrix(
    df_hv: pd.DataFrame,
    rh: float,
    cov: float,
    ahv: float,
    metric: str,
) -> np.ndarray:
    mask = (df_hv["rH"] == rh) & (df_hv["CoV"] == cov) & (df_hv["aHV"] == ahv)
    sub = df_hv.loc[mask]
    piv = sub.pivot(index="node", columns="seed", values=metric)
    arr = piv.to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(arr) & (arr > 0), arr, np.nan)


def seed_geomeans(arr: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.exp(np.nanmean(np.log(arr), axis=0))


def node_geomeans(arr: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.exp(np.nanmean(np.log(arr), axis=1))


def overall_geomean(arr: np.ndarray) -> float:
    with np.errstate(invalid="ignore", divide="ignore"):
        flat = arr[np.isfinite(arr) & (arr > 0)]
    if flat.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(flat))))


def _relative_cloud(vals: np.ndarray, g_global: float) -> np.ndarray:
    if not np.isfinite(g_global) or g_global <= 0:
        return np.asarray([], dtype=float)
    clean = vals[np.isfinite(vals) & (vals > 0)]
    return clean / g_global


def _face_rgba(color: str, face_alpha: float) -> tuple[float, float, float, float]:
    r, g, b = to_rgb(color)
    return (r, g, b, face_alpha)


def _draw_boxplot(
    ax: plt.Axes,
    data: list[np.ndarray],
    positions: list[float],
    colors: list[str],
    *,
    hatch: str,
    face_alpha: float,
) -> None:
    """Draw one seed/node boxplot group (all metrics in a panel) at once."""
    keep = [(d, p, c) for d, p, c in zip(data, positions, colors) if d.size > 0]
    if not keep:
        return
    kept_data, kept_pos, kept_colors = zip(*keep)
    bp = ax.boxplot(
        kept_data,
        positions=kept_pos,
        widths=BOX_WIDTH,
        whis=WHIS,
        showfliers=True,
        patch_artist=True,
        manage_ticks=False,
        boxprops={"edgecolor": HATCH_EDGE, "linewidth": 0.65},
        medianprops={"color": "0.1", "linewidth": DATA_LINEWIDTH, "solid_capstyle": "butt"},
        whiskerprops={"color": HATCH_EDGE, "linewidth": 0.65},
        capprops={"color": HATCH_EDGE, "linewidth": 0.65},
        flierprops={
            "marker": ".",
            "markersize": 2.2,
            "markerfacecolor": HATCH_EDGE,
            "markeredgecolor": "none",
            "alpha": 0.6,
        },
        zorder=3,
    )
    for box, color in zip(bp["boxes"], kept_colors):
        box.set_facecolor(_face_rgba(color, face_alpha))
        box.set_hatch(hatch)


def _legend_handles() -> list:
    return [
        Patch(
            facecolor=_face_rgba("0.65", SEED_FACE_ALPHA),
            edgecolor=HATCH_EDGE,
            hatch=SEED_HATCH,
            linewidth=0.8,
            label=r"$G_{\mathrm{seed}}/G_{\mathrm{global}}$",
        ),
        Patch(
            facecolor=_face_rgba("0.65", NODE_FACE_ALPHA),
            edgecolor=HATCH_EDGE,
            hatch=NODE_HATCH,
            linewidth=0.8,
            label=r"$G_{\mathrm{node}}/G_{\mathrm{global}}$",
        ),
        # Line2D([0], [0], color="0.1", lw=DATA_LINEWIDTH, label="Median"),
        Line2D(
            [0],
            [0],
            color="0.25",
            ls="--",
            lw=DATA_LINEWIDTH,
            label=r"$G/G_{\mathrm{global}}=1$",
        ),
    ]


def _make_3x3_figure(*, h: float, vs1: float) -> tuple[plt.Figure, np.ndarray]:
    fig = plt.figure(figsize=(DOC_WIDTH, FIG_HEIGHT))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.06, 1.0],
        hspace=0.02,
        left=0.08,
        right=0.995,
        bottom=0.06,
        top=0.99,
    )
    header = fig.add_subplot(gs[0, 0])
    header.axis("off")
    gs_panels = gs[1, 0].subgridspec(3, 3, wspace=0.08, hspace=0.10)
    axes = np.empty((3, 3), dtype=object)
    for r in range(3):
        for c in range(3):
            sharex = axes[0, 0] if (r, c) != (0, 0) else None
            sharey = axes[0, 0] if (r, c) != (0, 0) else None
            axes[r, c] = fig.add_subplot(gs_panels[r, c], sharex=sharex, sharey=sharey)

    header.text(
        0.5,
        0.95,
        rf"($H = {h:.0f}$ m, $V_{{s1}} = {vs1:.0f}$ m/s)",
        transform=header.transAxes,
        ha="center",
        va="top",
        fontsize=TICK_LABELSIZE,
        bbox=TEXT_BBOX,
    )
    header.legend(
        handles=_legend_handles(),
        loc="lower center",
        ncol=4,
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
    ax.grid(True, which="major", axis="y", alpha=GRID_ALPHA, lw=0.6)
    ax.set_axisbelow(True)


def plot_geomean_relative(
    df: pd.DataFrame,
    *,
    h: float,
    vs1: float,
    out_dir: Path,
) -> Path:
    df_hv = df[(df["Height"] == h) & (df["Vs1"] == vs1)]
    fig, axes = _make_3x3_figure(h=h, vs1=vs1)

    tick_pos = [float(i) * METRIC_GAP for i in range(len(METRICS))]
    tick_lab = [metric_label(m) for m in METRICS]

    # Pass 1: gather relative clouds (raw, not reduced to percentiles — the
    # boxplot computes its own quartiles/whiskers/fliers from these).
    panel_data: list[list[tuple]] = []
    for rh, cov, ahv in PANELS:
        cell_rows: list[tuple] = []
        for metric in METRICS:
            arr = cell_matrix(df_hv, rh, cov, ahv, metric)
            g_global = overall_geomean(arr)
            rel_seed = _relative_cloud(seed_geomeans(arr), g_global)
            rel_node = _relative_cloud(node_geomeans(arr), g_global)
            cell_rows.append((metric, rel_seed, rel_node, metric_color(metric)))
        panel_data.append(cell_rows)

    # Set y-scale to log
    axes[0, 0].set_yscale("log")
    y0, y1 = Y_LIM
    axes[0, 0].set_ylim(y0, y1)

    # Pass 2: draw one boxplot group per seed/node per panel.
    for i, ((rh, cov, ahv), rows) in enumerate(zip(PANELS, panel_data)):
        ax = axes.flat[i]
        colors = [color for _, _, _, color in rows]
        seed_data = [rel_seed for _, rel_seed, _, _ in rows]
        node_data = [rel_node for _, _, rel_node, _ in rows]
        seed_pos = [x + SEED_OFFSET for x in tick_pos]
        node_pos = [x + NODE_OFFSET for x in tick_pos]
        _draw_boxplot(ax, seed_data, seed_pos, colors, hatch=SEED_HATCH, face_alpha=SEED_FACE_ALPHA)
        _draw_boxplot(ax, node_data, node_pos, colors, hatch=NODE_HATCH, face_alpha=NODE_FACE_ALPHA)

        ax.axhline(1.0, color="0.25", ls="--", lw=DATA_LINEWIDTH, zorder=2)
        ax.set_xlim(tick_pos[0] - 0.55, tick_pos[-1] + 0.55)
        ax.set_xticks(tick_pos)
        _annotate_panel(ax, i, rh, cov, ahv)

        row, col = divmod(i, 3)
        if row == 2:
            ax.set_xticklabels(tick_lab, fontsize=TICK_LABELSIZE - 0.5)
        else:
            ax.tick_params(labelbottom=False)
        if col == 0:
            ax.set_ylabel(r"$G/G_{\mathrm{global}}$", fontsize=LABEL_FONTSIZE)
            ax.tick_params(labelleft=True)
        else:
            ax.tick_params(labelleft=False)

    stem = f"geomean_relative_h{h:.0f}_vs1_{vs1:.0f}"
    paths = save_figure(fig, stem, out_dir=out_dir)
    plt.close(fig)
    return paths[0]


def build_summary_md(written: list[Path]) -> str:
    lines = [
        "# Relative seed / node geomean boxplots",
        "",
        "3×3 factor-grid comparison of seed and node geomeans for all χ metrics, "
        r"normalized by the cell overall geomean \(G_{\mathrm{global}}\). "
        "Panel layout matches TF qualitative / central profiles "
        r"(\(r_h\), CoV, \(a_{hv}\) sweeps; center column shared).",
        "",
        r"- Seed cloud: \(G_{\mathrm{seed},j}/G_{\mathrm{global}}\) "
        r"with \(G_{\mathrm{seed},j}=\exp(N_x^{-1}\sum_i\ln\chi_{ij})\) "
        f"(hatch `{SEED_HATCH}`)",
        r"- Node cloud: \(G_{\mathrm{node},i}/G_{\mathrm{global}}\) "
        r"with \(G_{\mathrm{node},i}=\exp(N_s^{-1}\sum_j\ln\chi_{ij})\) "
        f"(hatch `{NODE_HATCH}`)",
        r"- \(G_{\mathrm{global}}=\exp(\overline{\ln\chi})\) over the cell",
        r"- Shared log \(y\)-axis: \([10^{-1},\,10^{1}]\) across all panels / \((H,V_{s1})\)",
        r"- Boxplots drawn straight from each cloud (native Matplotlib "
        r"``ax.boxplot``): box spans the IQR, whiskers reach the "
        r"\(P_{10}\)–\(P_{90}\) range (clipped to the data), dots beyond are "
        r"fliers",
        "- Horizontal tick: median; dashed: unity",
        "",
        "## Outputs",
        "",
        "| File | Content |",
        "| --- | --- |",
    ]
    for p in written:
        lines.append(f"| `{p.name}` | 3×3 relative seed/node boxplots, all metrics |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    out_dir = figure_dir("chi_variables", "geomean_relative")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("geomean_relative_*.pdf"):
        old.unlink()
        print(f"  removed {old.name}")

    print(f"Loading {DATA_PATH} …")
    df = load_ratios()
    print(f"  rows={len(df):,}")
    print(f"  → {out_dir}")

    written: list[Path] = []
    for h in H_LIST:
        for vs1 in VS1_LIST:
            print(f"  H={h:.0f}, Vs1={vs1:.0f} …")
            p = plot_geomean_relative(df, h=h, vs1=vs1, out_dir=out_dir)
            written.append(p)
            print(f"    {p.name}")

    md_path = out_dir / "summary.md"
    md_path.write_text(build_summary_md(written), encoding="utf-8")
    print(f"Wrote {md_path}")
    print(f"Done → {out_dir}")


if __name__ == "__main__":
    main()
