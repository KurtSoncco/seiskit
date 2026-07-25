r"""Factor-sweep boxplots of seed and node geomeans (all χ metrics).

For each (Height, Vs1) and geomean kind (seed / node), two Nature-width
figures under ``figure_dir("chi_variables", "geomean_factor_cross")``:

- ``geomean_{kind}_freq_cross_…`` — rows ``f_ratio``, ``abs_TF_ratio``
- ``geomean_{kind}_im_cross_…`` — rows ``PGA_ratio``, ``PSA_ratio``, ``Ia_ratio``

Columns sweep \(r_h\), CoV, and \(a_{hv}\) with the other two factors held at
the center cell (30 m, 0.2, 10). Factor levels use **numeric** x positions
(log-x for \(a_{hv}\)). Each cell: boxplots of the geomean cloud + connected
overall geomean \(G\).

Unlike ``factor_violins.py`` (raw χ, marginal over other factors), these
figures show geomean summaries conditional on the center cell.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
METRICS = ("f_ratio", "abs_TF_ratio", "PGA_ratio", "PSA_ratio", "Ia_ratio")
FREQ_METRICS = ("f_ratio", "abs_TF_ratio")
IM_METRICS = ("PGA_ratio", "PSA_ratio", "Ia_ratio")
LOG_Y_METRICS = frozenset({"abs_TF_ratio", "Ia_ratio"})

H_LIST = [15.0, 50.0, 100.0]
VS1_LIST = [100.0, 230.0, 360.0]

CENTER = (30.0, 0.2, 10.0)  # (rH, CoV, aHV)

# Column factor sweeps: (bare xlabel, log_x, levels as the varying coordinate,
#  list of (rh, cov, ahv) cells in level order)
FACTOR_COLS: list[tuple[str, bool, list[float], list[tuple[float, float, float]]]] = [
    (
        r"$r_h$ (m)",
        False,
        [10.0, 30.0, 50.0],
        [(10.0, 0.2, 10.0), (30.0, 0.2, 10.0), (50.0, 0.2, 10.0)],
    ),
    (
        "CoV",
        False,
        [0.1, 0.2, 0.3],
        [(30.0, 0.1, 10.0), (30.0, 0.2, 10.0), (30.0, 0.3, 10.0)],
    ),
    (
        r"$a_{hv}$",
        True,
        [1.0, 10.0, 50.0],
        [(30.0, 0.2, 1.0), (30.0, 0.2, 10.0), (30.0, 0.2, 50.0)],
    ),
]

Y_PCT_LO, Y_PCT_HI = 1.0, 99.0
Y_PAD_FRAC = 0.12
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

GeomeanKind = Literal["seed", "node"]
MetricGroup = Literal["freq", "im"]


def load_ratios(path: Path = DATA_PATH) -> pd.DataFrame:
    cols = ["Vs1", "Height", "CoV", "rH", "aHV", "channel", "seed", *METRICS]
    with h5py.File(path, "r") as f:
        g = f["master"]
        df = pd.DataFrame({c: g[c][:] for c in cols})
    return df.rename(columns={"channel": "node"})


def _format_level(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value)}"
    return f"{value:g}"


def _stem(kind: GeomeanKind, group: MetricGroup, h: float, vs1: float) -> str:
    return f"geomean_{kind}_{group}_cross_h{h:.0f}_vs1_{vs1:.0f}"


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


def node_geomeans(arr: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.exp(np.nanmean(np.log(arr), axis=1))


def seed_geomeans(arr: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.exp(np.nanmean(np.log(arr), axis=0))


def overall_geomean(arr: np.ndarray) -> float:
    with np.errstate(invalid="ignore", divide="ignore"):
        flat = arr[np.isfinite(arr) & (arr > 0)]
    if flat.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(flat))))


def _tight_ylim(arrays: list[np.ndarray], *, log_y: bool) -> tuple[float, float]:
    chunks = [a[np.isfinite(a) & (a > 0)] for a in arrays]
    chunks = [c for c in chunks if c.size]
    if not chunks:
        return (0.05, 1.0) if log_y else (0.0, 1.0)
    all_v = np.concatenate(chunks)
    lo = float(np.percentile(all_v, Y_PCT_LO))
    hi = float(np.percentile(all_v, Y_PCT_HI))
    if not np.isfinite(lo) or (log_y and lo <= 0):
        lo = float(np.nanmin(all_v[all_v > 0])) if log_y else float(np.nanmin(all_v))
    if not np.isfinite(hi) or hi <= lo:
        hi = float(np.nanmax(all_v))
    if log_y:
        log_lo, log_hi = np.log10(lo), np.log10(hi)
        span = max(log_hi - log_lo, 0.15)
        pad = Y_PAD_FRAC * span
        return (10 ** (log_lo - pad), 10 ** (log_hi + pad))
    span = max(hi - lo, 1e-3)
    pad = Y_PAD_FRAC * span
    return (max(lo - pad, 0.0) if lo >= 0 else lo - pad, hi + pad)


def _box_widths(levels: np.ndarray, *, log_x: bool) -> np.ndarray:
    """Box widths in data units (~35% of smallest adjacent gap)."""
    levels = np.asarray(levels, dtype=float)
    if levels.size < 2:
        w = 0.4 * abs(levels[0]) if levels.size else 1.0
        return np.full(max(levels.size, 1), w)
    if log_x:
        dlog = float(np.min(np.diff(np.log10(levels))))
        frac = 0.35 * dlog
        return levels * (10 ** (frac / 2) - 10 ** (-frac / 2))
    w = 0.35 * float(np.min(np.diff(levels)))
    return np.full(levels.size, w)


def _x_pad(levels: np.ndarray, widths: np.ndarray, *, log_x: bool) -> tuple[float, float]:
    levels = np.asarray(levels, dtype=float)
    widths = np.asarray(widths, dtype=float)
    if log_x:
        lo = levels[0] * 10 ** (-0.08)
        hi = levels[-1] * 10 ** (0.08)
        return (lo, hi)
    pad0 = 0.6 * float(widths[0])
    pad1 = 0.6 * float(widths[-1])
    return (float(levels[0] - pad0), float(levels[-1] + pad1))


def _style_boxplot(bp: dict, color: str) -> None:
    for box in bp["boxes"]:
        box.set_facecolor(color)
        box.set_edgecolor(color)
        box.set_linewidth(0.9)
        box.set_alpha(0.35)
    for key in ("whiskers", "caps"):
        for artist in bp[key]:
            artist.set_color(color)
            artist.set_linewidth(0.9)
    for med in bp["medians"]:
        med.set_color("0.15")
        med.set_linewidth(1.1)


def _make_grid(
    *,
    nrows: int,
    h: float,
    vs1: float,
    legend_handles: list,
) -> tuple[plt.Figure, np.ndarray]:
    aspect = 0.55 if nrows == 2 else 0.72
    fig = plt.figure(figsize=figsize(aspect=aspect))
    # Compact case subtitle; bottom reserved for xlabels + legend.
    top = 0.94 if nrows == 2 else 0.95
    bottom = 0.14 if nrows == 2 else 0.11
    gs = fig.add_gridspec(
        nrows,
        3,
        wspace=0.08,
        hspace=0.12,
        left=0.10,
        right=0.995,
        bottom=bottom,
        top=top,
    )
    axes = np.empty((nrows, 3), dtype=object)
    for r in range(nrows):
        for c in range(3):
            axes[r, c] = fig.add_subplot(gs[r, c])

    rh0, cov0, ahv0 = CENTER
    fig.text(
        0.5,
        0.985,
        rf"($H = {h:.0f}$ m, $V_{{s1}} = {vs1:.0f}$ m/s; "
        rf"others fixed at $r_h = {rh0:.0f}$ m, CoV $= {cov0:g}$, "
        rf"$a_{{hv}} = {ahv0:.0f}$)",
        ha="center",
        va="top",
        fontsize=TICK_LABELSIZE,
        bbox=TEXT_BBOX,
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        fontsize=TICK_LABELSIZE,
        handlelength=1.5,
        columnspacing=1.5,
        borderaxespad=0.0,
        bbox_to_anchor=(0.55, 0.005),
        **LEGEND_FRAME,
    )
    return fig, axes


def plot_geomean_factor_cross(
    df: pd.DataFrame,
    *,
    h: float,
    vs1: float,
    kind: GeomeanKind,
    metrics: tuple[str, ...],
    group: MetricGroup,
    out_dir: Path,
) -> Path:
    df_hv = df[(df["Height"] == h) & (df["Vs1"] == vs1)]
    geomean_fn = seed_geomeans if kind == "seed" else node_geomeans
    if kind == "seed":
        box_label = r"$G^{\mathrm{seed}}_j$ (P10–P90)"
    else:
        box_label = r"$G^{\mathrm{node}}_i$ (P10–P90)"
    legend_handles = [
        Patch(facecolor="0.55", edgecolor="0.35", alpha=0.35, label=box_label),
        Line2D(
            [0],
            [0],
            color="0.15",
            marker="D",
            ls="-",
            lw=DATA_LINEWIDTH,
            markersize=4,
            label=r"Overall $G$",
        ),
    ]
    fig, axes = _make_grid(
        nrows=len(metrics),
        h=h,
        vs1=vs1,
        legend_handles=legend_handles,
    )

    panel_i = 0
    for r, metric in enumerate(metrics):
        color = metric_color(metric)
        log_y = metric in LOG_Y_METRICS
        row_ylim_vals: list[np.ndarray] = []
        row_cache: list[tuple] = []

        for c, (xlabel, log_x, levels, cells) in enumerate(FACTOR_COLS):
            levels_arr = np.asarray(levels, dtype=float)
            clouds: list[np.ndarray] = []
            g_line: list[float] = []
            for rh, cov, ahv in cells:
                arr = cell_matrix(df_hv, rh, cov, ahv, metric)
                vals = geomean_fn(arr)
                clean = vals[np.isfinite(vals) & (vals > 0)]
                clouds.append(clean)
                g_all = overall_geomean(arr)
                g_line.append(g_all)
                row_ylim_vals.append(clean)
                if np.isfinite(g_all):
                    row_ylim_vals.append(np.asarray([g_all]))
            row_cache.append((xlabel, log_x, levels_arr, clouds, g_line))

        y_lim = _tight_ylim(row_ylim_vals, log_y=log_y)

        for c, (xlabel, log_x, levels_arr, clouds, g_line) in enumerate(row_cache):
            ax = axes[r, c]
            widths = _box_widths(levels_arr, log_x=log_x)
            plot_data = [cld if cld.size else np.asarray([np.nan]) for cld in clouds]
            bp = ax.boxplot(
                plot_data,
                positions=levels_arr,
                widths=widths,
                patch_artist=True,
                showfliers=False,
                whis=(10, 90),
                manage_ticks=False,
            )
            _style_boxplot(bp, color)

            g_arr = np.asarray(g_line, dtype=float)
            ax.plot(
                levels_arr,
                g_arr,
                color="0.15",
                ls="-",
                lw=DATA_LINEWIDTH,
                marker="D",
                markersize=4,
                zorder=5,
            )

            if log_x:
                ax.set_xscale("log")
            ax.set_xticks(levels_arr)
            ax.set_xticklabels([_format_level(v) for v in levels_arr])
            ax.set_xlim(*_x_pad(levels_arr, widths, log_x=log_x))

            if log_y:
                ax.set_yscale("log")
            ax.set_ylim(*y_lim)

            add_panel_label(ax, panel_i, alpha=0.75)
            panel_i += 1
            ax.tick_params(labelsize=TICK_LABELSIZE)
            ax.grid(True, which="major", axis="y", alpha=GRID_ALPHA, lw=0.6)
            ax.set_axisbelow(True)

            if c == 0:
                ax.set_ylabel(metric_label(metric), fontsize=LABEL_FONTSIZE)
                ax.tick_params(labelleft=True)
            else:
                ax.tick_params(labelleft=False)
                plt.setp(ax.get_yticklabels(), visible=False)

            if r == len(metrics) - 1:
                ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
            else:
                ax.tick_params(labelbottom=False)

        # Re-assert after all ylim/scale changes on the row.
        for c in range(1, 3):
            axes[r, c].tick_params(labelleft=False)
            plt.setp(axes[r, c].get_yticklabels(), visible=False)

    paths = save_figure(fig, _stem(kind, group, h, vs1), out_dir=out_dir)
    plt.close(fig)
    return paths[0]


def main() -> None:
    out_dir = figure_dir("chi_variables", "geomean_factor_cross")
    # Erase previous versions (old single-metric and prior layouts).
    if out_dir.is_dir():
        for old in out_dir.glob("*.pdf"):
            old.unlink()
            print(f"  removed {old.name}")

    print(f"Loading {DATA_PATH} …")
    df = load_ratios()
    print(f"  rows={len(df):,}")
    print(f"  → {out_dir}")

    for h in H_LIST:
        for vs1 in VS1_LIST:
            print(f"  H={h:.0f}, Vs1={vs1:.0f} …")
            for kind in ("seed", "node"):
                p_freq = plot_geomean_factor_cross(
                    df,
                    h=h,
                    vs1=vs1,
                    kind=kind,  # type: ignore[arg-type]
                    metrics=FREQ_METRICS,
                    group="freq",
                    out_dir=out_dir,
                )
                p_im = plot_geomean_factor_cross(
                    df,
                    h=h,
                    vs1=vs1,
                    kind=kind,  # type: ignore[arg-type]
                    metrics=IM_METRICS,
                    group="im",
                    out_dir=out_dir,
                )
                print(f"    {p_freq.name}")
                print(f"    {p_im.name}")

    print(f"Done → {out_dir}")


if __name__ == "__main__":
    main()
