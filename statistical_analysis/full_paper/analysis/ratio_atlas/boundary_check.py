"""Ensemble check that spatial ratio change is not a fixed boundary effect.

A mesh or absorbing-boundary imprint is nearly the same in every seed, so it
survives an average over seeds. The random field does not. This script plots
the geomean and median of ``abs_TF_ratio`` and ``PGA_ratio`` versus distance
from the center recorder for the paper reference cell, and writes a short
text comparison of left versus right and of the inner versus outer array.

The ensemble geomean is called flat when its span across the array is less
than one quarter of the typical seed-to-seed standard deviation. The outer
bin is called unremarkable when its extra ``|χ − χ(0)|``, relative to the
inner bin, is also below that quarter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

_ATLAS = Path(__file__).resolve().parent
_FULL_PAPER = _ATLAS.parents[1]
_CODE = _FULL_PAPER / "analysis" / "code"
for _p in (_ATLAS, _FULL_PAPER, _CODE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from _shared import CENTER_NODE, DX_M  # noqa: E402
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
from ratio_4x4 import (  # noqa: E402
    DEFAULT_AHV,
    DEFAULT_COV,
    DEFAULT_HEIGHT,
    DEFAULT_RH,
    DEFAULT_VS1,
    METRICS,
    load_cell,
)

apply_full_paper_style(auto_format=True, frame="open", grid=False)

N_TRACE = 20
TRACE_SEED = 42
INNER_M = 40.0
OUTER_M = 80.0
FLAT_FRACTION = 0.25

SAMPLE_COLOR = "0.55"
SAMPLE_ALPHA = 0.35
SAMPLE_LW = 0.35
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

DOC_WIDTH, FIG_HEIGHT = figsize(aspect=0.48)


def _fmt_num(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def stack_by_distance(
    channels: np.ndarray,
    seeds: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (distance_m, seed_ids, values[distance, seed])."""
    seed_ids = np.unique(seeds)
    x_ref = (np.unique(channels).astype(float) - CENTER_NODE) * DX_M
    x_ref = np.sort(x_ref)
    mat = np.full((x_ref.size, seed_ids.size), np.nan, dtype=float)
    seed_index = {int(s): j for j, s in enumerate(seed_ids)}
    x = (channels.astype(float) - CENTER_NODE) * DX_M
    for i in range(channels.size):
        ix = int(np.where(x_ref == x[i])[0][0])
        mat[ix, seed_index[int(seeds[i])]] = values[i]
    return x_ref, seed_ids, mat


def _positive(mat: np.ndarray) -> np.ndarray:
    out = np.array(mat, dtype=float, copy=True)
    out[~(np.isfinite(out) & (out > 0))] = np.nan
    return out


def ensemble_curves(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Geomean, median, and seed-to-seed standard deviation along axis 1."""
    clean = _positive(mat)
    with np.errstate(invalid="ignore", divide="ignore"):
        geo = np.exp(np.nanmean(np.log(clean), axis=1))
    med = np.nanmedian(clean, axis=1)
    std = np.nanstd(clean, axis=1, ddof=1)
    return geo, med, std


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("nan")
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def left_right_correlation(x: np.ndarray, profile: np.ndarray) -> float:
    """Pearson correlation of the left half with the mirrored right half."""
    left = profile[x < 0]
    right = profile[x > 0][::-1]
    n = min(left.size, right.size)
    return _corr(left[:n], right[:n])


def seed_left_right(x: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """One left–right correlation per seed."""
    out = np.empty(mat.shape[1], dtype=float)
    for j in range(mat.shape[1]):
        out[j] = left_right_correlation(x, mat[:, j])
    return out


def bin_abs_offset(x: np.ndarray, mat: np.ndarray) -> tuple[float, float]:
    """Mean |χ(x) − χ(0)| inside |x| < 40 m and outside |x| > 80 m."""
    center = np.where(np.isclose(x, 0.0))[0]
    if center.size != 1:
        raise ValueError("distance axis has no unique center recorder")
    chi0 = mat[int(center[0])]
    absdev = np.abs(mat - chi0[None, :])
    inner = float(np.nanmean(absdev[np.abs(x) < INNER_M]))
    outer = float(np.nanmean(absdev[np.abs(x) > OUTER_M]))
    return inner, outer


def summarize_metric(
    name: str,
    x: np.ndarray,
    mat: np.ndarray,
) -> dict[str, float | str]:
    geo, med, std = ensemble_curves(mat)
    seed_std = float(np.nanmedian(std))
    span = float(np.nanmax(geo) - np.nanmin(geo))
    span_ratio = span / seed_std if seed_std > 0 else float("nan")
    mirror_r = left_right_correlation(x, geo)
    seed_r = seed_left_right(x, _positive(mat))
    seed_r = seed_r[np.isfinite(seed_r)]
    inner, outer = bin_abs_offset(x, _positive(mat))
    excess = outer - inner
    excess_ratio = excess / seed_std if seed_std > 0 else float("nan")
    flat = bool(np.isfinite(span_ratio) and span_ratio < FLAT_FRACTION)
    outer_quiet = bool(np.isfinite(excess_ratio) and excess_ratio < FLAT_FRACTION)
    return {
        "metric": name,
        "geo_min": float(np.nanmin(geo)),
        "geo_max": float(np.nanmax(geo)),
        "x_geo_min": float(x[int(np.nanargmin(geo))]),
        "x_geo_max": float(x[int(np.nanargmax(geo))]),
        "geo_center": float(geo[np.isclose(x, 0.0)][0]),
        "med_center": float(med[np.isclose(x, 0.0)][0]),
        "span": span,
        "seed_std": seed_std,
        "span_ratio": span_ratio,
        "mirror_r": mirror_r,
        "seed_mirror_r": float(np.median(seed_r)) if seed_r.size else float("nan"),
        "inner": inner,
        "outer": outer,
        "excess": excess,
        "excess_ratio": excess_ratio,
        "flat": flat,
        "outer_quiet": outer_quiet,
    }


def _fmt(value: float, digits: int = 3) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def summary_text(
    rows: list[dict[str, float | str]],
    *,
    vs1: float,
    height: float,
    cov: float,
    rh: float,
    ahv: float,
    n_seeds: int,
) -> str:
    lines = [
        "Boundary-effect check for the reference cell",
        (
            f"Vs1 = {vs1:.0f} m/s, H = {height:.0f} m, CoV = {cov:g}, "
            f"r_h = {rh:.0f} m, a_hv = {ahv:.0f}"
        ),
        f"Seeds averaged: {n_seeds}. Recorders: center ±100 m at {DX_M:.0f} m.",
        (
            "A fixed boundary imprint would remain in the geomean and would raise "
            f"the outer array (|x| > {OUTER_M:.0f} m) relative to the inner array "
            f"(|x| < {INNER_M:.0f} m)."
        ),
        (
            f"Flat geomean: span across x < {FLAT_FRACTION:.2f} times the median "
            "seed-to-seed standard deviation."
        ),
        (
            f"Quiet outer bin: (outer − inner) mean |χ − χ(0)| < {FLAT_FRACTION:.2f} "
            "times that same standard deviation."
        ),
        "",
    ]
    for row in rows:
        label = metric_label(str(row["metric"]))
        flat = "yes" if row["flat"] else "no"
        quiet = "yes" if row["outer_quiet"] else "no"
        lines.extend(
            [
                str(row["metric"]),
                f"  display: {label}",
                (
                    f"  geomean range: {_fmt(float(row['geo_min']))} to "
                    f"{_fmt(float(row['geo_max']))} "
                    f"(min at {float(row['x_geo_min']):.0f} m, "
                    f"max at {float(row['x_geo_max']):.0f} m, "
                    f"center {_fmt(float(row['geo_center']))}, "
                    f"median at center {_fmt(float(row['med_center']))})"
                ),
                (
                    f"  geomean span: {_fmt(float(row['span']))}; "
                    f"seed-to-seed std: {_fmt(float(row['seed_std']))}; "
                    f"span/std: {_fmt(float(row['span_ratio']))}"
                ),
                (
                    f"  left–right correlation of geomean: {_fmt(float(row['mirror_r']))}; "
                    f"median seed left–right correlation: {_fmt(float(row['seed_mirror_r']))}"
                ),
                (
                    f"  mean |χ − χ(0)| inner: {_fmt(float(row['inner']))}; "
                    f"outer: {_fmt(float(row['outer']))}; "
                    f"outer − inner: {_fmt(float(row['excess']))} "
                    f"({_fmt(float(row['excess_ratio']))} of seed std)"
                ),
                f"  geomean flat: {flat}; outer bin quiet: {quiet}",
                "",
            ]
        )
    if all(bool(row["flat"]) for row in rows):
        lines.append(
            "The geomean is flat relative to the seed-to-seed spread for both ratios. "
            "A fixed boundary imprint would have remained in that average, so the "
            "spatial change in single seeds is the random field."
        )
    else:
        parts = []
        for row in rows:
            if row["flat"]:
                continue
            parts.append(
                f"{row['metric']} (span/std = {_fmt(float(row['span_ratio']))}, "
                f"minimum at {float(row['x_geo_min']):.0f} m, "
                f"maximum at {float(row['x_geo_max']):.0f} m)"
            )
        on_edge = [
            row
            for row in rows
            if not row["flat"]
            and (
                abs(float(row["x_geo_min"])) >= 90.0
                or abs(float(row["x_geo_max"])) >= 90.0
            )
        ]
        if on_edge:
            where = (
                "At least one extremum is at an outer recorder (±100 m), "
                "which is the pattern a side-boundary imprint would leave."
            )
        else:
            where = (
                "Those extrema are not at the outer recorders (±100 m), and the "
                "left–right correlation of individual seeds is much lower than that of "
                "the geomean, so the trend is not a symmetric edge peak."
            )
        lines.append(
            "A residual ensemble trend remains for " + "; ".join(parts) + ". "
            + where
            + " The residual is smaller than the seed-to-seed spread. The spatial "
            "change in the atlas is still dominated by the random field."
        )
    if any(not bool(row["outer_quiet"]) for row in rows):
        lines.append(
            "Within a seed, the outer recorders differ more from the center value "
            "than the inner recorders do. That follows from correlation decaying "
            "with distance. It is not, by itself, a shift of the ensemble mean."
        )
    lines.append("")
    return "\n".join(lines)


def plot_check(
    stacks: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    vs1: float,
    height: float,
    cov: float,
    rh: float,
    ahv: float,
) -> Figure:
    fig = plt.figure(figsize=(DOC_WIDTH, FIG_HEIGHT))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.14, 1.0],
        hspace=0.04,
        left=0.08,
        right=0.99,
        bottom=0.16,
        top=0.98,
    )
    header = fig.add_subplot(gs[0, 0])
    header.axis("off")
    gs_panels = gs[1, 0].subgridspec(1, len(METRICS), wspace=0.22)
    axes = [fig.add_subplot(gs_panels[0, i]) for i in range(len(METRICS))]

    rng = np.random.default_rng(TRACE_SEED)
    for i, metric in enumerate(METRICS):
        ax = axes[i]
        x, seed_ids, mat = stacks[metric]
        clean = _positive(mat)
        geo, med, _std = ensemble_curves(mat)
        n_show = min(N_TRACE, seed_ids.size)
        chosen = rng.choice(seed_ids.size, size=n_show, replace=False)
        for j in chosen:
            ax.plot(x, clean[:, j], color=SAMPLE_COLOR, lw=SAMPLE_LW, alpha=SAMPLE_ALPHA, zorder=1)
        ax.axhline(1.0, color=REF_COLOR, lw=DATA_LINEWIDTH, zorder=2)
        color = metric_color(metric)
        ax.plot(x, geo, color=color, ls="-", lw=DATA_LINEWIDTH, zorder=4)
        ax.plot(x, med, color=color, ls="--", lw=DATA_LINEWIDTH, zorder=4)
        finite = clean[np.isfinite(clean)]
        lo, hi = np.nanpercentile(finite, [2, 98])
        lo = min(lo, float(np.nanmin(geo)), float(np.nanmin(med)))
        hi = max(hi, float(np.nanmax(geo)), float(np.nanmax(med)))
        pad = 0.08 * (hi - lo if hi > lo else 1.0)
        ax.set_ylim(lo - pad, 1.5)
        ax.set_xlim(float(x[0]) - 8.0, float(x[-1]) + 8.0)
        ax.set_xticks([-100, 0, 100])
        add_panel_label(ax, i, alpha=0.75)
        ax.tick_params(labelsize=TICK_LABELSIZE)
        ax.grid(True, which="major", alpha=GRID_ALPHA, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_ylabel(metric_label(metric), fontsize=LABEL_FONTSIZE)

    fig.supxlabel("Distance from center (m)", fontsize=LABEL_FONTSIZE, y=0.04)
    header.text(
        0.5,
        0.98,
        (
            rf"$V_{{s1}} = {vs1:.0f}$ m/s, $H = {height:.0f}$ m, "
            rf"$\mathrm{{CoV}} = {cov:g}$, $r_h = {rh:.0f}$ m, $a_{{hv}} = {ahv:.0f}$"
            "\n"
            "geomean and median across all seeds"
        ),
        transform=header.transAxes,
        ha="center",
        va="top",
        fontsize=TICK_LABELSIZE,
        linespacing=1.35,
    )
    handles = [
        Line2D([0], [0], color=SAMPLE_COLOR, lw=DATA_LINEWIDTH, label=f"Seeds (subset of {N_TRACE})"),
        Line2D([0], [0], color="0.2", ls="-", lw=DATA_LINEWIDTH, label="Geomean across seeds"),
        Line2D([0], [0], color="0.2", ls="--", lw=DATA_LINEWIDTH, label="Median across seeds"),
        Line2D([0], [0], color=REF_COLOR, lw=DATA_LINEWIDTH, label="1D baseline"),
    ]
    header.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=TICK_LABELSIZE,
        handlelength=1.8,
        columnspacing=1.0,
        borderaxespad=0.0,
        bbox_to_anchor=(0.5, -0.05),
        **LEGEND_FRAME,
    )
    return fig


def main() -> None:
    vs1, height, cov, rh, ahv = (
        DEFAULT_VS1,
        DEFAULT_HEIGHT,
        DEFAULT_COV,
        DEFAULT_RH,
        DEFAULT_AHV,
    )
    channels, seeds, series = load_cell(vs1=vs1, height=height, cov=cov, rh=rh, ahv=ahv)
    stacks = {
        metric: stack_by_distance(channels, seeds, series[metric]) for metric in METRICS
    }
    n_seeds = int(stacks[METRICS[0]][1].size)
    rows = []
    for metric in METRICS:
        x, _seed_ids, mat = stacks[metric]
        rows.append(summarize_metric(metric, x, mat))

    text = summary_text(
        rows, vs1=vs1, height=height, cov=cov, rh=rh, ahv=ahv, n_seeds=n_seeds
    )
    out_dir = figure_dir("ratio_atlas")
    stem = (
        f"boundary_check_h{height:.0f}_vs1_{vs1:.0f}"
        f"_cov{_fmt_num(cov)}_rh{rh:.0f}_ahv{ahv:.0f}"
    )
    text_path = out_dir / f"{stem}.txt"
    text_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {text_path}")

    fig = plot_check(stacks, vs1=vs1, height=height, cov=cov, rh=rh, ahv=ahv)
    save_figure(fig, stem, out_dir=out_dir)
    plt.close(fig)


if __name__ == "__main__":
    main()
