"""Between-seed spatial coherence of χ ratios (distinct from lag-ACF).

Per design cell × metric on \(Y_{ij}=\\ln\\chi_{ij}\):

1. Node means \(\\nu_i\) and between-seed SDs \(s_{B,i}\)
2. Cross-covariance / correlation across seeds:
   \(C_{ik}=N_s^{-1}\\sum_j (Y_{ij}-\\nu_i)(Y_{kj}-\\nu_k)\),
   \(\\rho_{ik}=C_{ik}/(s_{B,i}\,s_{B,k})\)
3. Lag-distance binning of \(\\rho_{ik}\) (mean ρ vs lag)

This is **not** the within-seed lag ACF in ``spatial_acf.py``: here correlation
is across realizations \(j\) at fixed node pairs \((i,k)\).

Writes CSV + Nature PDFs + ``summary.md`` under
``figure_dir("chi_spatial", "spatial_coherence")``.
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

from config import (  # noqa: E402
    DATA_LINEWIDTH,
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
from _shared import (  # noqa: E402
    CENTER_NODE,
    DX_M,
    FACTORS,
    METRICS,
    N_CELLS,
    N_NODES,
    N_SEEDS,
    load_ratios,
    log_response,
)

apply_full_paper_style(auto_format=True, frame="open", grid=False)

H_MAX_M = (N_NODES - 1) * DX_M  # 200 m
LAG_SUMMARY_M = (2.0, 10.0, 20.0, 50.0, 100.0, 200.0)

# Center Height / Vs1 for factor-sweep figures
REF_H = 50.0
REF_VS1 = 230.0
CENTER_RH, CENTER_COV, CENTER_AHV = 30.0, 0.2, 10.0

FACTOR_SWEEPS: list[tuple[str, str, bool, list[float], dict[str, float]]] = [
    (
        "CoV",
        "CoV",
        False,
        [0.1, 0.2, 0.3],
        {"rH": CENTER_RH, "aHV": CENTER_AHV},
    ),
    (
        "rH",
        r"$r_h$ (m)",
        False,
        [10.0, 30.0, 50.0],
        {"CoV": CENTER_COV, "aHV": CENTER_AHV},
    ),
    (
        "aHV",
        r"$a_{hv}$",
        True,
        [1.0, 10.0, 50.0],
        {"CoV": CENTER_COV, "rH": CENTER_RH},
    ),
]

SWEEP_COLORS = [
    TOL_BRIGHT["blue"],
    TOL_BRIGHT["red"],
    TOL_BRIGHT["green"],
]

GRID_ALPHA = 0.18
LEGEND_FRAME = {
    "frameon": True,
    "fancybox": False,
    "framealpha": 0.75,
    "facecolor": "white",
    "edgecolor": "none",
    "borderpad": 0.25,
}


def _out() -> Path:
    return figure_dir("chi_spatial", "spatial_coherence")


def pivot_ln(df_cell: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (Y, node_index) with Y shape (N_x, N_s)."""
    chi = df_cell.pivot(index="node", columns="seed", values=metric)
    nodes = chi.index.to_numpy(dtype=float)
    arr = chi.to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        Y = np.where(np.isfinite(arr) & (arr > 0), np.log(arr), np.nan)
    return Y, nodes


def between_seed_C_rho(
    Y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute C_ik, rho_ik, nu_i, s_B,i (population / ddof=0 over seeds)."""
    n_nodes, n_seeds = Y.shape
    nu = np.nanmean(Y, axis=1)
    Z = Y - nu[:, np.newaxis]

    if np.all(np.isfinite(Y)):
        C = (Z @ Z.T) / float(n_seeds)
    else:
        C = np.full((n_nodes, n_nodes), np.nan)
        for i in range(n_nodes):
            zi = Z[i]
            mi = np.isfinite(zi)
            for k in range(i, n_nodes):
                both = mi & np.isfinite(Z[k])
                n_ok = int(both.sum())
                if n_ok < 2:
                    continue
                c = float(np.mean(zi[both] * Z[k, both]))
                C[i, k] = c
                C[k, i] = c

    s_B = np.sqrt(np.diag(C))
    with np.errstate(invalid="ignore", divide="ignore"):
        den = np.outer(s_B, s_B)
        rho = np.where(den > 0, C / den, np.nan)
    np.fill_diagonal(rho, 1.0)
    return C, rho, nu, s_B


def lag_bin_rho(rho: np.ndarray, nodes: np.ndarray) -> pd.DataFrame:
    """Mean / SD / n_pairs of ρ_ik vs lag distance (m)."""
    node_idx = np.asarray(nodes, dtype=float)
    di, dk = np.meshgrid(node_idx, node_idx, indexing="ij")
    lag_steps = np.abs(di - dk).astype(int)
    # Unique positive lags (exclude diagonal)
    max_lag = int(lag_steps.max())
    rows: list[dict] = []
    for step in range(1, max_lag + 1):
        mask = lag_steps == step
        vals = rho[mask]
        vals = vals[np.isfinite(vals)]
        h_m = float(step * DX_M)
        rows.append(
            {
                "lag_steps": step,
                "h_m": h_m,
                "rho_mean": float(np.mean(vals)) if vals.size else np.nan,
                "rho_std": float(np.std(vals, ddof=0)) if vals.size else np.nan,
                "rho_median": float(np.median(vals)) if vals.size else np.nan,
                "n_pairs": int(vals.size),
            }
        )
    return pd.DataFrame(rows)


def _rho_at_lag(lag_df: pd.DataFrame, h_m: float) -> float:
    hit = lag_df.loc[np.isclose(lag_df["h_m"], h_m), "rho_mean"]
    return float(hit.iloc[0]) if len(hit) else float("nan")


def assess_cell_metric(
    df_cell: pd.DataFrame,
    metric: str,
    cell_keys: dict,
) -> tuple[dict, list[dict]]:
    """Return (cell summary row, lag-binned rows)."""
    Y, nodes = pivot_ln(df_cell, metric)
    n_finite = int(np.isfinite(Y).sum())
    empty_sum = {
        **cell_keys,
        "metric": metric,
        "n_nodes": int(Y.shape[0]),
        "n_seeds": int(Y.shape[1]),
        "n_finite": n_finite,
        "s_B_bar": np.nan,
        "mean_abs_rho_offdiag": np.nan,
        "mean_rho_offdiag": np.nan,
        **{f"rho_h{int(h)}m": np.nan for h in LAG_SUMMARY_M},
        "ok": False,
    }
    if n_finite < 8 or Y.shape[0] < 3:
        return empty_sum, []

    C, rho, nu, s_B = between_seed_C_rho(Y)
    lag_df = lag_bin_rho(rho, nodes)

    off = ~np.eye(rho.shape[0], dtype=bool)
    rho_off = rho[off]
    rho_off = rho_off[np.isfinite(rho_off)]

    summary = {
        **cell_keys,
        "metric": metric,
        "n_nodes": int(np.isfinite(nu).sum()),
        "n_seeds": int(Y.shape[1]),
        "n_finite": n_finite,
        "s_B_bar": float(np.nanmean(s_B)),
        "mean_abs_rho_offdiag": float(np.mean(np.abs(rho_off))) if rho_off.size else np.nan,
        "mean_rho_offdiag": float(np.mean(rho_off)) if rho_off.size else np.nan,
        **{f"rho_h{int(h)}m": _rho_at_lag(lag_df, h) for h in LAG_SUMMARY_M},
        "ok": True,
    }

    lag_rows = [
        {**cell_keys, "metric": metric, **rec}
        for rec in lag_df.to_dict(orient="records")
    ]
    return summary, lag_rows


def _format_level(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value)}"
    return f"{value:g}"


def plot_coherence_factor_sweeps(lag: pd.DataFrame) -> None:
    """Mean ρ vs lag across CoV / rH / aHV at center Height, Vs1."""
    out = _out()
    for metric in METRICS:
        fig, axes = plt.subplots(
            1,
            3,
            figsize=figsize(aspect=0.38),
            sharey=True,
            constrained_layout=False,
        )
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.18, top=0.88, wspace=0.18)

        for ax_i, (factor, xlabel, _log_x, levels, fixed) in enumerate(FACTOR_SWEEPS):
            ax = axes[ax_i]
            for lv_i, lv in enumerate(levels):
                mask = (
                    (lag["metric"] == metric)
                    & np.isclose(lag["Height"].astype(float), REF_H)
                    & np.isclose(lag["Vs1"].astype(float), REF_VS1)
                    & np.isclose(lag[factor].astype(float), float(lv))
                )
                for k, v in fixed.items():
                    mask &= np.isclose(lag[k].astype(float), float(v))
                sub = lag.loc[mask].sort_values("h_m")
                if sub.empty:
                    continue
                ax.plot(
                    sub["h_m"].to_numpy(dtype=float),
                    sub["rho_mean"].to_numpy(dtype=float),
                    color=SWEEP_COLORS[lv_i % len(SWEEP_COLORS)],
                    lw=DATA_LINEWIDTH,
                    label=_format_level(lv),
                )
            ax.axhline(0.0, color="0.6", lw=0.4)
            ax.set_xlim(0.0, H_MAX_M)
            ax.set_ylim(-0.2, 1.05)
            ax.set_xlabel("Lag distance (m)", fontsize=LABEL_FONTSIZE)
            if ax_i == 0:
                ax.set_ylabel(r"Mean $\rho_{ik}$", fontsize=LABEL_FONTSIZE)
            ax.set_title(xlabel, fontsize=LABEL_FONTSIZE)
            ax.grid(True, alpha=GRID_ALPHA, lw=0.5)
            ax.set_axisbelow(True)
            ax.legend(
                title=xlabel,
                fontsize=TICK_LABELSIZE,
                title_fontsize=TICK_LABELSIZE,
                loc="upper right",
                **LEGEND_FRAME,
            )
            add_panel_label(ax, ax_i, alpha=0.75)

        fig.suptitle(
            rf"Between-seed coherence — {metric_label(metric, log=True)} "
            rf"($H={_format_level(REF_H)}$ m, $V_{{s1}}={_format_level(REF_VS1)}$ m/s)",
            fontsize=LABEL_FONTSIZE,
            y=0.98,
        )
        save_figure(fig, f"coherence_vs_lag_{metric}", out_dir=out)
        plt.close(fig)


def plot_coherence_overview(lag: pd.DataFrame) -> None:
    """All metrics at the center design cell: mean ρ vs lag."""
    out = _out()
    fig, ax = plt.subplots(figsize=figsize(aspect=0.55))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.14, top=0.92)

    mask = (
        np.isclose(lag["Height"].astype(float), REF_H)
        & np.isclose(lag["Vs1"].astype(float), REF_VS1)
        & np.isclose(lag["CoV"].astype(float), CENTER_COV)
        & np.isclose(lag["rH"].astype(float), CENTER_RH)
        & np.isclose(lag["aHV"].astype(float), CENTER_AHV)
    )
    for metric in METRICS:
        sub = lag.loc[mask & (lag["metric"] == metric)].sort_values("h_m")
        if sub.empty:
            continue
        ax.plot(
            sub["h_m"].to_numpy(dtype=float),
            sub["rho_mean"].to_numpy(dtype=float),
            color=metric_color(metric),
            lw=DATA_LINEWIDTH,
            label=metric_label(metric, log=True),
        )
    ax.axhline(0.0, color="0.6", lw=0.4)
    ax.set_xlim(0.0, H_MAX_M)
    ax.set_ylim(-0.2, 1.05)
    ax.set_xlabel("Lag distance (m)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(r"Mean $\rho_{ik}$", fontsize=LABEL_FONTSIZE)
    ax.grid(True, alpha=GRID_ALPHA, lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(fontsize=TICK_LABELSIZE, loc="upper right", **LEGEND_FRAME)
    ax.set_title(
        "Between-seed coherence at center cell "
        f"(H={_format_level(REF_H)}, Vs1={_format_level(REF_VS1)}, "
        f"CoV={_format_level(CENTER_COV)}, rH={_format_level(CENTER_RH)}, "
        f"aHV={_format_level(CENTER_AHV)})",
        fontsize=LABEL_FONTSIZE,
    )
    save_figure(fig, "coherence_vs_lag_center_all_metrics", out_dir=out)
    plt.close(fig)


def plot_median_envelope(lag: pd.DataFrame) -> None:
    """Median ± IQR of lag-ρ across all 243 cells, per metric."""
    out = _out()
    fig, axes = plt.subplots(
        1,
        len(METRICS),
        figsize=figsize(aspect=0.32),
        sharey=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.20, top=0.88, wspace=0.12)

    for i, metric in enumerate(METRICS):
        ax = axes[i]
        sub = lag[lag["metric"] == metric]
        g = sub.groupby("h_m", sort=True)["rho_mean"]
        h = g.mean().index.to_numpy(dtype=float)
        med = g.median().to_numpy(dtype=float)
        q25 = g.quantile(0.25).to_numpy(dtype=float)
        q75 = g.quantile(0.75).to_numpy(dtype=float)
        color = metric_color(metric)
        ax.fill_between(h, q25, q75, color=color, alpha=0.25, lw=0)
        ax.plot(h, med, color=color, lw=DATA_LINEWIDTH)
        ax.axhline(0.0, color="0.6", lw=0.4)
        ax.set_xlim(0.0, H_MAX_M)
        ax.set_ylim(-0.2, 1.05)
        ax.set_xlabel("Lag (m)", fontsize=LABEL_FONTSIZE)
        if i == 0:
            ax.set_ylabel(r"Mean $\rho_{ik}$", fontsize=LABEL_FONTSIZE)
        ax.set_title(metric_label(metric, log=True), fontsize=LABEL_FONTSIZE)
        ax.grid(True, alpha=GRID_ALPHA, lw=0.5)
        add_panel_label(ax, i, alpha=0.75)

    fig.suptitle(
        r"Between-seed coherence: median / IQR over design cells",
        fontsize=LABEL_FONTSIZE,
        y=0.98,
    )
    handles = [
        Line2D([0], [0], color="0.35", lw=DATA_LINEWIDTH, label="Median"),
        Line2D([0], [0], color="0.35", lw=6, alpha=0.25, label="IQR"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        fontsize=TICK_LABELSIZE,
        bbox_to_anchor=(0.5, 0.02),
        **LEGEND_FRAME,
    )
    save_figure(fig, "coherence_vs_lag_median_iqr", out_dir=out)
    plt.close(fig)


def build_summary_md(cell: pd.DataFrame, lag: pd.DataFrame) -> str:
    lines = [
        "# Between-seed spatial coherence",
        "",
        "Cross-seed spatial coherence of \(Y_{ij}=\\ln\\chi_{ij}\) "
        f"for `{', '.join(METRICS)}` from `join_master.h5`.",
        "",
        "## Distinction from lag-ACF (`spatial_acf.py`)",
        "",
        "| Quantity | Indexing | Meaning |",
        "| --- | --- | --- |",
        "| **Lag-ACF** (`spatial_acf.py`) | within seed \(j\), "
        "along nodes | Pearson autocorrelation of the demeaned node series "
        "vs lag \(h\) |",
        "| **Coherence** (this script) | across seeds \(j\), "
        "fixed node pair \((i,k)\) | "
        r"\(C_{ik}=\mathrm{mean}_j[(Y_{ij}-\nu_i)(Y_{kj}-\nu_k)]\), "
        r"\(\rho_{ik}=C_{ik}/(s_{B,i}\,s_{B,k})\) |",
        "",
        "Lag-ACF asks how similar neighbouring *nodes* are within one "
        "realization. Coherence asks whether *seed-to-seed* fluctuations "
        "at node \(i\) track those at node \(k\).",
        "",
        f"- Design cells: **{int(cell['cell'].nunique())}** "
        f"(expected {N_CELLS}); nodes \(N_x={N_NODES}\), "
        f"seeds \(N_s={N_SEEDS}\); \(dx={DX_M:g}\) m",
        r"- \(s_{B,i}=\sqrt{C_{ii}}\); lag bins use \(|i-k|\,dx\)",
        "",
        "## Short-lag coherence (median over cells)",
        "",
        "| Metric | med ρ̄(2 m) | med ρ̄(10 m) | med ρ̄(50 m) | med ρ̄(100 m) | "
        "med mean\\|ρ\\| off-diag |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metric in METRICS:
        m = cell[cell["metric"] == metric]

        def med_col(col: str) -> float:
            s = m[col]
            return float(s.median()) if len(s) else float("nan")

        lines.append(
            "| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                metric,
                med_col("rho_h2m"),
                med_col("rho_h10m"),
                med_col("rho_h50m"),
                med_col("rho_h100m"),
                med_col("mean_abs_rho_offdiag"),
            )
        )

    # Factor sensitivity at center H/Vs1
    lines.extend(
        [
            "",
            f"## Factor sweeps at \(H={REF_H:g}\) m, \(V_{{s1}}={REF_VS1:g}\) m/s",
            "",
            "Mean lag-ρ at 10 m for center-fixed companion factors:",
            "",
            "| Sweep | Level | "
            + " | ".join(METRICS)
            + " |",
            "| --- | ---: | "
            + " | ".join(["---:" ] * len(METRICS))
            + " |",
        ]
    )
    for factor, _xlabel, _log_x, levels, fixed in FACTOR_SWEEPS:
        for lv in levels:
            vals = []
            for metric in METRICS:
                mask = (
                    (lag["metric"] == metric)
                    & np.isclose(lag["Height"].astype(float), REF_H)
                    & np.isclose(lag["Vs1"].astype(float), REF_VS1)
                    & np.isclose(lag[factor].astype(float), float(lv))
                    & np.isclose(lag["h_m"].astype(float), 10.0)
                )
                for k, v in fixed.items():
                    mask &= np.isclose(lag[k].astype(float), float(v))
                sub = lag.loc[mask, "rho_mean"]
                vals.append(f"{float(sub.iloc[0]):.3f}" if len(sub) else "—")
            lines.append(f"| {factor} | {_format_level(lv)} | " + " | ".join(vals) + " |")

    med_rho2 = float(cell["rho_h2m"].median())
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "| File | Content |",
            "| --- | --- |",
            "| `coherence_cell_summary.csv` | Per cell×metric short-lag ρ "
            "and off-diagonal means |",
            "| `coherence_lag_binned.csv` | Mean / SD / median ρ vs lag "
            "for every cell×metric |",
            "| `coherence_vs_lag_*.pdf` | Nature figures (factor sweeps, "
            "center overview, median/IQR) |",
            "",
            "## Conclusions",
            "",
            f"1. **Between-seed fluctuations are spatially coherent.** "
            f"Median short-lag ρ̄(2 m) across cells/metrics is "
            f"**{med_rho2:.3f}**, so seed anomalies persist over many nodes.",
            "",
            "2. **Coherence ≠ lag-ACF.** Strong within-seed ACF "
            "(`spatial_acf.py`) and strong between-seed \(\\rho_{ik}\) "
            "are related but answer different questions; both caution "
            "against treating node×seed draws as iid.",
            "",
            "3. **Design factors modulate the decay** of ρ̄(h) with lag "
            "(see CoV / \(r_h\) / \(a_{hv}\) sweep figures at center "
            f"\(H\), \(V_{{s1}}\)).",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    print("Loading ratios via _shared.load_ratios …")
    df = load_ratios()
    print(f"Loaded {len(df):,} rows")

    df = df.copy()
    df["cell"] = df.groupby(list(FACTORS), sort=False).ngroup()
    n_cells = int(df["cell"].nunique())
    print(f"Design cells: {n_cells} (expected {N_CELLS})")

    # Warm-up log_response path (ensures shared helper is exercised)
    _ = log_response(df.head(10), METRICS[0])

    _CODE = Path(__file__).resolve().parent.parent
    if str(_CODE) not in sys.path:
        sys.path.insert(0, str(_CODE))
    from _shared import parallel_map  # noqa: E402

    grouped = list(df.groupby("cell", sort=True))

    def _one(item):
        _, g = item
        cell_keys = {f: g[f].iloc[0] for f in FACTORS}
        cell_keys["cell"] = int(g["cell"].iloc[0])
        cell_rows, lag_rows = [], []
        for metric in METRICS:
            summary, lags = assess_cell_metric(g, metric, cell_keys)
            cell_rows.append(summary)
            lag_rows.extend(lags)
        return cell_rows, lag_rows

    results = parallel_map(_one, grouped, desc="spatial_coherence cells")
    cell_rows: list[dict] = []
    lag_rows: list[dict] = []
    for c, lags in results:
        cell_rows.extend(c)
        lag_rows.extend(lags)

    cell = pd.DataFrame(cell_rows)
    lag = pd.DataFrame(lag_rows)

    out = _out()
    cell_path = out / "coherence_cell_summary.csv"
    lag_path = out / "coherence_lag_binned.csv"
    md_path = out / "summary.md"

    cell.to_csv(cell_path, index=False)
    lag.to_csv(lag_path, index=False)

    print("Plotting …")
    plot_coherence_overview(lag)
    plot_coherence_factor_sweeps(lag)
    plot_median_envelope(lag)

    summary = build_summary_md(cell, lag)
    md_path.write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print(f"Wrote {cell_path}")
    print(f"Wrote {lag_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
