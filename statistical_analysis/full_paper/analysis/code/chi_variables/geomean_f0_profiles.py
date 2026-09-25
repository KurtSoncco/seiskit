r"""Seed, node, and global geomean profiles versus the site frequency.

For the center design cell ``(r_h, CoV, a_hv) = (30 m, 0.2, 10)``, this
script compares geomean profiles across all ``(Vs1, H)`` cases while varying
one design factor at a time. The horizontal coordinate is the calculated
quarter-wave frequency

``f_0 = Vs1 / (4 H)``.

Each factor gets a separate five-row by three-column figure. Curves identify
factor levels; columns show seed, node, and global geomeans. No additional
H/Vs1 encoding is used: the horizontal coordinate is only the calculated
frequency.
Outputs are written under ``figure_dir("chi_variables", "geomean_f0_profiles")``.
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
CENTER = {"rH": 30.0, "CoV": 0.2, "aHV": 10.0}
H_LIST = [15.0, 50.0, 100.0]
VS1_LIST = [100.0, 230.0, 360.0]
FACTOR_SWEEPS = {
    "rH": [10.0, 30.0, 50.0],
    "CoV": [0.1, 0.2, 0.3],
    "aHV": [1.0, 10.0, 50.0],
}
FACTOR_LABELS = {"rH": r"$r_h$", "CoV": "CoV", "aHV": r"$a_{hv}$"}
GRID_ALPHA = 0.18
F0_XLIM = (0.1, 10.0)
F0_XTICKS = np.asarray([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
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


def extract_matrix(
    df: pd.DataFrame,
    *,
    height: float,
    vs1: float,
    rh: float,
    cov: float,
    ahv: float,
    metric: str,
) -> np.ndarray:
    mask = (
        (df["Height"] == height)
        & (df["Vs1"] == vs1)
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


def _cloud_summary(values: np.ndarray) -> tuple[float, float, float]:
    clean = values[np.isfinite(values) & (values > 0)]
    if clean.size == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.percentile(clean, 10)),
        float(np.median(clean)),
        float(np.percentile(clean, 90)),
    )


def _case_order() -> list[tuple[float, float, float]]:
    cases = [(height, vs1, vs1 / (4.0 * height)) for height in H_LIST for vs1 in VS1_LIST]
    return sorted(cases, key=lambda case: case[2])


def _summary_value(values: np.ndarray, summary: str) -> float:
    if summary == "seed":
        cloud = seed_geomeans(values)
    elif summary == "node":
        cloud = node_geomeans(values)
    else:
        cloud = np.asarray([overall_geomean(values)])
    clean = cloud[np.isfinite(cloud) & (cloud > 0)]
    return float(np.median(clean)) if clean.size else float("nan")


def compute_global_ylims(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Return shared linear y-limits for each metric across all plotted cells."""
    limits: dict[str, tuple[float, float]] = {}
    cases = _case_order()
    for metric in METRICS:
        values: list[float] = []
        for height, vs1, _ in cases:
            for factor, sweep_values in FACTOR_SWEEPS.items():
                for sweep_value in sweep_values:
                    params = CENTER.copy()
                    params[factor] = sweep_value
                    matrix = extract_matrix(
                        df,
                        height=height,
                        vs1=vs1,
                        rh=params["rH"],
                        cov=params["CoV"],
                        ahv=params["aHV"],
                        metric=metric,
                    )
                    values.extend(
                        summary
                        for summary in (
                            _summary_value(matrix, "seed"),
                            _summary_value(matrix, "node"),
                            _summary_value(matrix, "global"),
                        )
                        if np.isfinite(summary) and summary > 0
                    )
        if values:
            limits[metric] = (min(values) * 0.9, max(values) * 1.1)
        else:
            limits[metric] = (0.1, 10.0)
    return limits


def effect_comparison(df: pd.DataFrame) -> list[dict[str, object]]:
    """Compare f0 range with factor ranges averaged across all f0 cases."""
    rows: list[dict[str, object]] = []
    cases = _case_order()
    for metric in METRICS:
        for summary in ("seed", "node", "global"):
            f0_values = [
                _summary_value(
                    extract_matrix(
                        df,
                        height=height,
                        vs1=vs1,
                        rh=CENTER["rH"],
                        cov=CENTER["CoV"],
                        ahv=CENTER["aHV"],
                        metric=metric,
                    ),
                    summary,
                )
                for height, vs1, _ in _case_order()
            ]
            f0_range = float(np.ptp(np.log(f0_values)))
            factor_ranges: dict[str, float] = {}
            for factor, sweep_values in FACTOR_SWEEPS.items():
                case_ranges: list[float] = []
                for height, vs1, _ in cases:
                    sweep_results: list[float] = []
                    for sweep_value in sweep_values:
                        params = CENTER.copy()
                        params[factor] = sweep_value
                        matrix = extract_matrix(
                            df,
                            height=height,
                            vs1=vs1,
                            rh=params["rH"],
                            cov=params["CoV"],
                            ahv=params["aHV"],
                            metric=metric,
                        )
                        sweep_results.append(_summary_value(matrix, summary))
                    case_ranges.append(float(np.ptp(np.log(sweep_results))))
                factor_ranges[factor] = float(np.mean(case_ranges))
            strongest_factor = max(factor_ranges, key=factor_ranges.get)
            rows.append(
                {
                    "metric": metric,
                    "summary": summary,
                    "f0_range": f0_range,
                    **factor_ranges,
                    "strongest_factor": strongest_factor,
                    "f0_dominant": f0_range > max(factor_ranges.values()),
                }
            )
    return rows


def plot_factor_profile(
    df: pd.DataFrame,
    *,
    factor: str,
    ylims: dict[str, tuple[float, float]],
    out_dir: Path,
) -> Path:
    cases = _case_order()
    f0_values = np.asarray([case[2] for case in cases], dtype=float)
    fig, axes = plt.subplots(
        len(METRICS),
        3,
        figsize=figsize(aspect=0.88),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    column_titles = [
        r"Seed geomean $G^{\mathrm{seed}}_j$",
        r"Node geomean $G^{\mathrm{node}}_i$",
        r"Global geomean $G$",
    ]
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title, fontsize=TICK_LABELSIZE)

    styles = [("-", "o"), ("--", "s"), (":", "D")]
    sweep_values = FACTOR_SWEEPS[factor]
    factor_label = FACTOR_LABELS[factor]
    for row, metric in enumerate(METRICS):
        color = metric_color(metric)
        seed_ax, node_ax, global_ax = axes[row]
        for ax in (seed_ax, node_ax, global_ax):
            ax.grid(True, which="major", axis="y", alpha=GRID_ALPHA, lw=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=TICK_LABELSIZE)
            ax.set_xscale("log")
            ax.set_xlim(*F0_XLIM)
            ax.set_xticks(F0_XTICKS)
            ax.set_ylim(*ylims[metric])
            if row == len(METRICS) - 1:
                ax.set_xticklabels([f"{value:g}" for value in F0_XTICKS])
            else:
                ax.tick_params(labelbottom=False)

        for index, sweep_value in enumerate(sweep_values):
            params = CENTER.copy()
            params[factor] = sweep_value
            line_style, marker = styles[index]
            seed_values: list[float] = []
            node_values: list[float] = []
            global_values: list[float] = []
            for height, vs1, _ in cases:
                matrix = extract_matrix(
                    df,
                    height=height,
                    vs1=vs1,
                    rh=params["rH"],
                    cov=params["CoV"],
                    ahv=params["aHV"],
                    metric=metric,
                )
                seed_values.append(_summary_value(matrix, "seed"))
                node_values.append(_summary_value(matrix, "node"))
                global_values.append(_summary_value(matrix, "global"))

            seed_ax.plot(f0_values, seed_values, color=color, lw=DATA_LINEWIDTH, ls=line_style, marker=marker, ms=3.2)
            node_ax.plot(f0_values, node_values, color=color, lw=DATA_LINEWIDTH, ls=line_style, marker=marker, ms=3.2)
            global_ax.plot(f0_values, global_values, color=color, lw=DATA_LINEWIDTH, ls=line_style, marker=marker, ms=3.0)

        seed_ax.set_ylabel(metric_label(metric), fontsize=TICK_LABELSIZE)
        node_ax.tick_params(labelleft=False)
        global_ax.tick_params(labelleft=False)
        if row == len(METRICS) - 1:
            for ax in (seed_ax, node_ax, global_ax):
                ax.set_xlabel(r"$f_{0,\mathrm{calc}} = V_{s1}/(4H)$ (Hz)", fontsize=LABEL_FONTSIZE)

    fig.text(
        0.5,
        0.985,
        rf"Varying {factor_label}; other factors fixed at center levels; shared y-limits per metric",
        ha="center",
        va="top",
        fontsize=TICK_LABELSIZE,
        bbox=TEXT_BBOX,
    )
    fig.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="0.15",
                lw=DATA_LINEWIDTH,
                ls=styles[index][0],
                marker=styles[index][1],
                label=f"{factor_label} = {value:g}",
            )
            for index, value in enumerate(sweep_values)
        ],
        loc="lower center",
        ncol=3,
        fontsize=TICK_LABELSIZE,
        handlelength=1.7,
        columnspacing=1.1,
        borderaxespad=0.0,
        bbox_to_anchor=(0.5, 0.03),
        **LEGEND_FRAME,
    )
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.21, top=0.94, wspace=0.16, hspace=0.16)

    paths = save_figure(fig, f"geomean_f0_profiles_{factor}", out_dir=out_dir)
    plt.close(fig)
    return paths[0]


def build_summary_md(written: list[Path], df: pd.DataFrame) -> str:
    comparison = effect_comparison(df)
    dominant_count = sum(bool(row["f0_dominant"]) for row in comparison)
    metric_dominance = {
        metric: all(row["f0_dominant"] for row in comparison if row["metric"] == metric)
        for metric in METRICS
    }
    lines = [
        "# Geomean profiles versus calculated site frequency",
        "",
        rf"Center design cell: $r_h={CENTER['rH']:.0f}$ m, CoV $={CENTER['CoV']:g}$, $a_{{hv}}={CENTER['aHV']:.0f}$.",
        r"The x-coordinate is the calculated quarter-wave frequency \(f_{0,\mathrm{calc}}=V_{s1}/(4H)\) on a logarithmic scale from 0.1 to 10 Hz.",
        "",
        r"- Each factor has a separate figure; curves show its three levels across all nine \((H,V_{s1})\) cases.",
        r"- Columns show the median seed geomean, median node geomean, and global geomean.",
        r"- No H/Vs1 encoding is applied; the horizontal coordinate is only \(f_{0,\mathrm{calc}}\).",
        r"- Y-limits are shared across the three figures for each metric.",
        "",
        "## Effect-size comparison",
        "",
        "To compare changes on a multiplicative scale, the table below reports the log-range "
        "of the median geomean across the nine calculated-frequency cases. Factor ranges are "
        "averaged across all nine frequency cases, so no arbitrary fixed H/Vs1 baseline is used.",
        "",
        rf"The calculated-frequency effect is larger than all three factor effects for "
        rf"{dominant_count} of {len(comparison)} geomean summaries. It dominates all three "
        + "summaries for: "
        + ", ".join(metric for metric, is_dominant in metric_dominance.items() if is_dominant)
        + ".",
        "",
        "| Metric | Summary | $f_{0,\\mathrm{calc}}$ | $r_h$ | CoV | $a_{hv}$ | Strongest factor | $f_0$ dominant |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in comparison:
        lines.append(
            f"| `{row['metric']}` | {row['summary']} | {row['f0_range']:.3f} | "
            f"{row['rH']:.3f} | {row['CoV']:.3f} | {row['aHV']:.3f} | "
            f"{row['strongest_factor']} | {'yes' if row['f0_dominant'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "A larger log-range means a larger multiplicative change in the plotted geomean. "
            "This is a descriptive comparison, not an independent variance-based sensitivity "
            "analysis: the frequency sweep changes both $H$ and $V_{s1}$, while factor ranges "
            "are averaged over the same nine $(H,V_{s1})$ cases.",
            "",
            "## Output",
            "",
            "| File | Content |",
            "| --- | --- |",
        ]
    )
    lines.extend(f"| `{path.name}` | Factor profile across calculated $f_{{0,\\mathrm{{calc}}}}$ |" for path in written)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    out_dir = figure_dir("chi_variables", "geomean_f0_profiles")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("geomean_f0_profiles*.pdf"):
        old.unlink()
        print(f"  removed {old.name}")

    print(f"Loading {DATA_PATH} ...")
    df = load_ratios()
    print(f"  rows={len(df):,}")
    print("Computing shared y-axis limits ...")
    ylims = compute_global_ylims(df)
    written: list[Path] = []
    for factor in FACTOR_SWEEPS:
        print(f"  generating {factor} profile ...")
        path = plot_factor_profile(df, factor=factor, ylims=ylims, out_dir=out_dir)
        written.append(path)
        print(f"    {path.name}")
    summary_path = out_dir / "summary.md"
    summary_path.write_text(build_summary_md(written, df), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
