r"""Seed, node, and global geomean profiles across the three design factors.

For one fixed ``(Height, Vs1)`` case, the figure has three panels showing
the effects of ``r_h``, CoV, and ``a_hv``. All five chi metrics are shown in
each panel. Metric color identifies the response; line style identifies the
seed geomean cloud, node geomean cloud, or global geomean.

Outputs are written under ``figure_dir("chi_variables", "geomean_factor_profiles")``.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
    METRICS,
    TICK_LABELSIZE,
    apply_full_paper_style,
    figsize,
    figure_dir,
    metric_color,
    metric_label,
    save_figure,
)

apply_full_paper_style(auto_format=True, frame="open", grid=False)

DATA_PATH = BOX_ROOT / "peak_analysis" / "join_master.h5"
HEIGHT = 15.0
VS1 = 100.0
CENTER = (30.0, 0.2, 10.0)  # (rH, CoV, aHV)
FACTOR_SWEEPS = [
    (r"$r_h$ (m)", [10.0, 30.0, 50.0], [(10.0, 0.2, 10.0), (30.0, 0.2, 10.0), (50.0, 0.2, 10.0)]),
    ("CoV", [0.1, 0.2, 0.3], [(30.0, 0.1, 10.0), (30.0, 0.2, 10.0), (30.0, 0.3, 10.0)]),
    (r"$a_{hv}$", [1.0, 10.0, 50.0], [(30.0, 0.2, 1.0), (30.0, 0.2, 10.0), (30.0, 0.2, 50.0)]),
]
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


def load_ratios(path: Path = DATA_PATH) -> pd.DataFrame:
    cols = ["Vs1", "Height", "CoV", "rH", "aHV", "channel", "seed", *METRICS]
    with h5py.File(path, "r") as f:
        group = f["master"]
        df = pd.DataFrame({column: group[column][:] for column in cols})
    return df.rename(columns={"channel": "node"})


def cell_matrix(df: pd.DataFrame, *, rh: float, cov: float, ahv: float, metric: str) -> np.ndarray:
    mask = (
        (df["Height"] == HEIGHT)
        & (df["Vs1"] == VS1)
        & (df["rH"] == rh)
        & (df["CoV"] == cov)
        & (df["aHV"] == ahv)
    )
    pivot = df.loc[mask].pivot(index="node", columns="seed", values=metric)
    values = pivot.to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(values) & (values > 0), values, np.nan)


def seed_geomeans(values: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.exp(np.nanmean(np.log(values), axis=0))


def node_geomeans(values: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.exp(np.nanmean(np.log(values), axis=1))


def overall_geomean(values: np.ndarray) -> float:
    clean = values[np.isfinite(values) & (values > 0)]
    if clean.size == 0:
        return float("nan")
    return float(np.exp(np.mean(np.log(clean))))


def cloud_summary(values: np.ndarray) -> tuple[float, float, float]:
    clean = values[np.isfinite(values) & (values > 0)]
    if clean.size == 0:
        return float("nan"), float("nan"), float("nan")
    return tuple(float(np.percentile(clean, q)) for q in (10, 50, 90))


def _legend_handles() -> list:
    handles = [
        Line2D([0], [0], color="0.15", lw=DATA_LINEWIDTH, ls="-", label="Seed median"),
        Line2D([0], [0], color="0.15", lw=DATA_LINEWIDTH, ls="--", label="Node median"),
        Line2D([0], [0], color="0.05", lw=1.3, ls=":", label="Global geomean"),
        Patch(facecolor="0.45", edgecolor="none", alpha=0.16, label="P10--P90 cloud"),
    ]
    handles.extend(
        Line2D([0], [0], color=metric_color(metric), lw=DATA_LINEWIDTH, label=metric_label(metric))
        for metric in METRICS
    )
    return handles


def plot_profiles(df: pd.DataFrame, *, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=figsize(aspect=0.48), squeeze=False)
    axes_flat = axes.flat

    for ax, (xlabel, levels, cells) in zip(axes_flat, FACTOR_SWEEPS):
        x_values = np.asarray(levels, dtype=float)
        for metric in METRICS:
            color = metric_color(metric)
            seed_stats: list[tuple[float, float, float]] = []
            node_stats: list[tuple[float, float, float]] = []
            global_values: list[float] = []
            for rh, cov, ahv in cells:
                values = cell_matrix(df, rh=rh, cov=cov, ahv=ahv, metric=metric)
                seed_stats.append(cloud_summary(seed_geomeans(values)))
                node_stats.append(cloud_summary(node_geomeans(values)))
                global_values.append(overall_geomean(values))

            seed_array = np.asarray(seed_stats, dtype=float)
            node_array = np.asarray(node_stats, dtype=float)
            ax.fill_between(x_values, seed_array[:, 0], seed_array[:, 2], color=color, alpha=0.08, linewidth=0)
            ax.fill_between(x_values, node_array[:, 0], node_array[:, 2], color=color, alpha=0.08, linewidth=0)
            ax.plot(x_values, seed_array[:, 1], color=color, lw=DATA_LINEWIDTH, ls="-", marker="o", ms=3.5)
            ax.plot(x_values, node_array[:, 1], color=color, lw=DATA_LINEWIDTH, ls="--", marker="s", ms=3.5)
            ax.plot(x_values, global_values, color=color, lw=1.1, ls=":", marker="D", ms=3.0)

        ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
        ax.set_xticks(x_values)
        ax.tick_params(labelsize=TICK_LABELSIZE)
        ax.grid(True, which="major", axis="y", alpha=GRID_ALPHA, lw=0.6)
        ax.set_axisbelow(True)

    axes_flat[0].set_ylabel("Geomean ratio", fontsize=LABEL_FONTSIZE)
    fig.text(
        0.5,
        0.985,
        rf"($H={HEIGHT:.0f}$ m, $V_{{s1}}={VS1:.0f}$ m/s; other factors held at center levels)",
        ha="center",
        va="top",
        fontsize=TICK_LABELSIZE,
        bbox=TEXT_BBOX,
    )
    fig.legend(
        handles=_legend_handles(),
        loc="lower center",
        ncol=5,
        fontsize=TICK_LABELSIZE,
        handlelength=1.6,
        columnspacing=1.0,
        borderaxespad=0.0,
        bbox_to_anchor=(0.5, -0.04),
        **LEGEND_FRAME,
    )
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.20, top=0.90, wspace=0.18)

    paths = save_figure(fig, "geomean_factor_profiles", out_dir=out_dir)
    plt.close(fig)
    return paths[0]


def build_summary_md(written: Path) -> str:
    return "\n".join(
        [
            "# Geomean profiles across design factors",
            "",
            rf"Fixed case: $H={HEIGHT:.0f}$ m and $V_{{s1}}={VS1:.0f}$ m/s. The three panels vary $r_h$, CoV, and $a_{{hv}}$ independently.",
            "",
            r"- Solid circles: seed-geomean median; dashed squares: node-geomean median; dotted diamonds: global geomean",
            r"- Shaded bands are P10--P90 across the corresponding seed/node geomean cloud",
            r"- Color identifies the five response metrics",
            "",
            "## Output",
            "",
            "| File | Content |",
            "| --- | --- |",
            f"| `{written.name}` | Three factor panels with all metrics |",
            "",
        ]
    )


def main() -> None:
    out_dir = figure_dir("chi_variables", "geomean_factor_profiles")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("geomean_factor_profiles_*.pdf"):
        old.unlink()
        print(f"  removed {old.name}")

    print(f"Loading {DATA_PATH} ...")
    df = load_ratios()
    print(f"  rows={len(df):,}")
    written = plot_profiles(df, out_dir=out_dir)
    print(f"    {written.name}")
    summary_path = out_dir / "summary.md"
    summary_path.write_text(build_summary_md(written), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
