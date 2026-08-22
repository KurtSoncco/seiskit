"""Histograms of \\(Y_{ij}=\\ln\\chi_{ij}\\) (no Q–Q plots).

Per metric:

- Pooled histogram over all node×seed draws
- Optional within-cell residual histogram (Y − cell mean)
- Factor-level overlays for CoV (marginal)

References existing Shapiro / normality CSV from ``node_ratio_normality.py``
when present (``figure_dir("chi_variables")/normality_results.csv``).

Writes Nature PDFs + brief ``summary.md`` under
``figure_dir("chi_variables", "distribution_histograms")``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

_FULL = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_FULL))
sys.path.insert(0, str(_CODE))

from _shared import (  # noqa: E402
    METRICS,
    N_CELLS,
    N_NODES,
    N_SEEDS,
    add_design_columns,
    load_ratios,
    log_response,
)
from config import (  # noqa: E402
    DATA_LINEWIDTH,
    FACTORS,
    LABEL_FONTSIZE,
    TICK_LABELSIZE,
    TOL_BRIGHT,
    add_panel_label,
    apply_full_paper_style,
    figsize,
    figure_dir,
    metric_color,
    metric_label,
    save_figure,
)

apply_full_paper_style(auto_format=True, frame="open", grid=False)

FREQ_METRICS = ("f_ratio", "abs_TF_ratio")
IM_METRICS = ("PGA_ratio", "PSA_ratio", "Ia_ratio")
COV_LEVELS = (0.1, 0.2, 0.3)
COV_COLORS = {
    0.1: TOL_BRIGHT["blue"],
    0.2: TOL_BRIGHT["red"],
    0.3: TOL_BRIGHT["green"],
}
N_BINS = 60
HIST_ALPHA = 0.55
GRID_ALPHA = 0.18
MAX_HIST_SAMPLES = 200_000
RNG = np.random.default_rng(0)

LEGEND_FRAME = {
    "frameon": True,
    "fancybox": False,
    "framealpha": 0.75,
    "facecolor": "white",
    "edgecolor": "none",
    "borderpad": 0.25,
}

NORMALITY_CSV = figure_dir("chi_variables") / "normality_results.csv"


def _out() -> Path:
    return figure_dir("chi_variables", "distribution_histograms")


def _subsample(x: np.ndarray, n_max: int = MAX_HIST_SAMPLES) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= n_max:
        return x
    idx = RNG.choice(x.size, size=n_max, replace=False)
    return x[idx]


def _finite_y(df: pd.DataFrame, metric: str) -> np.ndarray:
    return log_response(df, metric)


def within_cell_residual_y(df: pd.DataFrame, metric: str) -> np.ndarray:
    y = log_response(df, metric)
    work = df.loc[np.isfinite(y), list(FACTORS)].copy()
    work["y"] = y[np.isfinite(y)]
    cell_mean = work.groupby(list(FACTORS), sort=False)["y"].transform("mean")
    return (work["y"] - cell_mean).to_numpy(dtype=float)


def load_shapiro_summary() -> pd.DataFrame | None:
    if not NORMALITY_CSV.is_file():
        return None
    raw = pd.read_csv(NORMALITY_CSV)
    # Focus: residual ln (closest to within-cell Y normality)
    sub = raw[(raw["transform"] == "ln") & (raw["residual"] == True)]  # noqa: E712
    if sub.empty:
        sub = raw[raw["transform"] == "ln"]
    rows = []
    for metric in METRICS:
        m = sub[sub["metric"] == metric]
        if m.empty:
            continue
        rows.append(
            {
                "metric": metric,
                "n_nodes": int(m["node"].nunique()),
                "median_shapiro_p": float(m["shapiro_p"].median()),
                "frac_shapiro_gt05": float((m["shapiro_p"] > 0.05).mean()),
                "median_abs_skew": float(m["skew"].abs().median()),
                "frac_verdict_lognormal": float((m["verdict"] == "lognormal").mean())
                if "verdict" in m.columns
                else np.nan,
            }
        )
    return pd.DataFrame(rows) if rows else None


def plot_pooled_and_residual(df: pd.DataFrame) -> None:
    """Two-row figure: pooled Y and within-cell residual, one col per metric."""
    out = _out()
    metrics = list(METRICS)
    ncols = len(metrics)
    fig, axes = plt.subplots(
        2,
        ncols,
        figsize=figsize(aspect=0.55),
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.90, wspace=0.22, hspace=0.35)

    for c, metric in enumerate(metrics):
        y = _finite_y(df, metric)
        y = _subsample(y[np.isfinite(y)])
        r = _subsample(within_cell_residual_y(df, metric))
        color = metric_color(metric)

        for row, series, title in (
            (0, y, r"Pooled $Y=\ln\chi$"),
            (1, r, r"Within-cell residual"),
        ):
            ax = axes[row, c]
            if series.size:
                ax.hist(
                    series,
                    bins=N_BINS,
                    density=True,
                    color=color,
                    alpha=HIST_ALPHA,
                    edgecolor="none",
                )
                mu, sd = float(np.mean(series)), float(np.std(series, ddof=0))
                if sd > 0:
                    xs = np.linspace(mu - 4 * sd, mu + 4 * sd, 200)
                    pdf = (1.0 / (sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sd) ** 2)
                    ax.plot(xs, pdf, color="0.15", lw=DATA_LINEWIDTH * 0.9, ls="--")
            ax.set_title(
                metric_label(metric, log=True) if row == 0 else "",
                fontsize=LABEL_FONTSIZE,
            )
            if row == 1:
                ax.set_xlabel(r"$Y$", fontsize=LABEL_FONTSIZE)
            if c == 0:
                ax.set_ylabel(f"{title}\nDensity", fontsize=LABEL_FONTSIZE)
            ax.grid(True, axis="y", alpha=GRID_ALPHA, lw=0.5)
            ax.set_axisbelow(True)
            add_panel_label(ax, row * ncols + c, alpha=0.75)

    fig.suptitle(
        r"Distribution of $Y_{ij}=\ln\chi_{ij}$ (dashed: matching Normal)",
        fontsize=LABEL_FONTSIZE,
        y=0.98,
    )
    save_figure(fig, "hist_pooled_and_residual", out_dir=out)
    plt.close(fig)


def plot_by_cov(df: pd.DataFrame) -> None:
    """Overlay CoV-level histograms of Y for each metric."""
    out = _out()
    metrics = list(METRICS)
    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=figsize(aspect=0.35),
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.18, top=0.86, wspace=0.22)

    for i, metric in enumerate(metrics):
        ax = axes[i]
        # Shared bin edges from pooled sample
        y_all = _subsample(_finite_y(df, metric))
        y_all = y_all[np.isfinite(y_all)]
        if y_all.size == 0:
            continue
        lo, hi = np.percentile(y_all, [0.5, 99.5])
        bins = np.linspace(lo, hi, N_BINS)
        for cov in COV_LEVELS:
            y = log_response(df.loc[np.isclose(df["CoV"].astype(float), cov)], metric)
            y = _subsample(y[np.isfinite(y)])
            if y.size == 0:
                continue
            ax.hist(
                y,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=DATA_LINEWIDTH,
                color=COV_COLORS[cov],
                label=f"{cov:g}",
            )
        ax.set_title(metric_label(metric, log=True), fontsize=LABEL_FONTSIZE)
        ax.set_xlabel(r"$Y$", fontsize=LABEL_FONTSIZE)
        if i == 0:
            ax.set_ylabel("Density", fontsize=LABEL_FONTSIZE)
        ax.grid(True, axis="y", alpha=GRID_ALPHA, lw=0.5)
        add_panel_label(ax, i, alpha=0.75)

    handles = [
        Line2D([0], [0], color=COV_COLORS[c], lw=DATA_LINEWIDTH, label=f"{c:g}") for c in COV_LEVELS
    ]
    fig.legend(
        handles=handles,
        title="CoV",
        loc="lower center",
        ncol=3,
        fontsize=TICK_LABELSIZE,
        title_fontsize=TICK_LABELSIZE,
        bbox_to_anchor=(0.5, 0.02),
        **LEGEND_FRAME,
    )
    fig.suptitle(
        r"Marginal $Y$ density by CoV (step histograms)",
        fontsize=LABEL_FONTSIZE,
        y=0.98,
    )
    save_figure(fig, "hist_by_cov", out_dir=out)
    plt.close(fig)


def plot_metric_groups(df: pd.DataFrame) -> None:
    """Separate freq / IM pooled histograms (taller panels)."""
    out = _out()
    for stem, metrics in (("freq", FREQ_METRICS), ("im", IM_METRICS)):
        fig, axes = plt.subplots(
            1,
            len(metrics),
            figsize=figsize(aspect=0.45),
            constrained_layout=False,
        )
        if len(metrics) == 1:
            axes = np.asarray([axes])
        fig.subplots_adjust(left=0.10, right=0.98, bottom=0.14, top=0.88, wspace=0.25)
        for i, metric in enumerate(metrics):
            ax = axes[i]
            y = _subsample(_finite_y(df, metric))
            y = y[np.isfinite(y)]
            color = metric_color(metric)
            if y.size:
                ax.hist(
                    y,
                    bins=N_BINS,
                    density=True,
                    color=color,
                    alpha=HIST_ALPHA,
                    edgecolor="none",
                )
                mu, sd = float(np.mean(y)), float(np.std(y, ddof=0))
                if sd > 0:
                    xs = np.linspace(mu - 4 * sd, mu + 4 * sd, 200)
                    pdf = (1.0 / (sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xs - mu) / sd) ** 2)
                    ax.plot(xs, pdf, color="0.15", lw=DATA_LINEWIDTH, ls="--")
            ax.set_xlabel(metric_label(metric, log=True), fontsize=LABEL_FONTSIZE)
            if i == 0:
                ax.set_ylabel("Density", fontsize=LABEL_FONTSIZE)
            ax.grid(True, axis="y", alpha=GRID_ALPHA, lw=0.5)
            add_panel_label(ax, i, alpha=0.75)
        fig.suptitle(
            rf"Pooled $Y$ histograms — {stem}",
            fontsize=LABEL_FONTSIZE,
            y=0.97,
        )
        save_figure(fig, f"hist_pooled_{stem}", out_dir=out)
        plt.close(fig)


def _descriptive_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        y = _finite_y(df, metric)
        y = y[np.isfinite(y)]
        r = within_cell_residual_y(df, metric)
        r = r[np.isfinite(r)]
        rows.append(
            {
                "metric": metric,
                "n_pooled": int(y.size),
                "mean_Y": float(np.mean(y)) if y.size else np.nan,
                "std_Y": float(np.std(y, ddof=0)) if y.size else np.nan,
                "skew_Y": float(pd.Series(y).skew()) if y.size > 2 else np.nan,
                "mean_resid": float(np.mean(r)) if r.size else np.nan,
                "std_resid": float(np.std(r, ddof=0)) if r.size else np.nan,
                "skew_resid": float(pd.Series(r).skew()) if r.size > 2 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_summary_md(desc: pd.DataFrame, shapiro: pd.DataFrame | None) -> str:
    lines = [
        "# Distribution histograms of \\(Y=\\ln\\chi\\)",
        "",
        "Histograms of the working scale \\(Y_{ij}=\\ln\\chi_{ij}\\) "
        "(pooled and within-cell residual). **No Q–Q plots** — formal "
        "normality tests live in `node_ratio_normality.py`.",
        "",
        rf"- Design: \(N_x={N_NODES}\), \(N_s={N_SEEDS}\), "
        f"{N_CELLS} cells; factors `{', '.join(FACTORS)}`",
        "- Dashed curves: Normal density matched to sample mean / SD",
        "",
        "## Descriptive moments",
        "",
        "| Metric | mean Y | sd Y | skew Y | sd residual | skew residual |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, r in desc.iterrows():
        lines.append(
            "| {metric} | {mean_Y:.3f} | {std_Y:.3f} | {skew_Y:.3f} | "
            "{std_resid:.3f} | {skew_resid:.3f} |".format(**r)
        )

    lines.extend(["", "## Reference: Shapiro on residual ln (existing CSV)", ""])
    if shapiro is None or shapiro.empty:
        lines.append(
            f"`{NORMALITY_CSV.name}` not found under "
            f"`figure_dir('chi_variables')`; run `node_ratio_normality.py` "
            "to regenerate. Histogram interpretation proceeds without it."
        )
    else:
        lines.extend(
            [
                "From `normality_results.csv` (per-node residual ln transform):",
                "",
                "| Metric | med Shapiro p | frac p>0.05 | med |skew| | frac verdict=lognormal |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, r in shapiro.iterrows():
            lines.append(
                "| {metric} | {median_shapiro_p:.3g} | {frac_shapiro_gt05:.3f} | "
                "{median_abs_skew:.3f} | {frac_verdict_lognormal:.3f} |".format(**r)
            )
        # Quick IM vs f0 note
        ia = shapiro.loc[shapiro["metric"] == "Ia_ratio"]
        f0 = shapiro.loc[shapiro["metric"] == "f_ratio"]
        if len(ia) and len(f0):
            lines.extend(
                [
                    "",
                    "Relative log-normality (residual ln, by median |skew|): "
                    f"`Ia_ratio` med|skew|={float(ia['median_abs_skew'].iloc[0]):.3f} vs "
                    f"`f_ratio`={float(f0['median_abs_skew'].iloc[0]):.3f}.",
                ]
            )

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "| File | Content |",
            "| --- | --- |",
            "| `hist_pooled_and_residual.pdf` | Pooled Y + within-cell residual |",
            "| `hist_by_cov.pdf` | CoV-stratified step histograms |",
            "| `hist_pooled_{freq,im}.pdf` | Grouped metric panels |",
            "| `distribution_moments.csv` | Skew / SD table |",
            "",
            "Formal Shapiro / KS / Anderson results remain in "
            "`normality_results.csv` — this script only visualizes densities.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    print("Loading ratios via _shared.load_ratios …")
    df = load_ratios()
    df = add_design_columns(df, include_node_z=False)
    print(f"Loaded {len(df):,} rows; cells={df['cell'].nunique()}")

    out = _out()
    desc = _descriptive_table(df)
    desc_path = out / "distribution_moments.csv"
    desc.to_csv(desc_path, index=False)

    shapiro = load_shapiro_summary()
    if shapiro is not None:
        shapiro.to_csv(out / "shapiro_reference_summary.csv", index=False)
        print(f"Referenced Shapiro CSV: {NORMALITY_CSV}")
    else:
        print(f"Shapiro CSV not found: {NORMALITY_CSV}")

    print("Plotting …")
    plot_pooled_and_residual(df)
    plot_by_cov(df)
    plot_metric_groups(df)

    md = build_summary_md(desc, shapiro)
    md_path = out / "summary.md"
    md_path.write_text(md, encoding="utf-8")
    print()
    print(md)
    print(f"Wrote {desc_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
