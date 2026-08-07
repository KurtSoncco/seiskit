"""Small-multiple heatmaps of variance fractions across CoV × rH.

Reads ``cell_summary.csv`` from ``figure_dir("chi_variables", "central_variability")``
when present; otherwise computes a minimal fraction table from ``join_master.h5``.

Annotates \(R^2_{\\mathrm{ceiling}}\) from
``figure_dir("chi_ols", "r2_ceiling")/reliability_ceiling.csv`` when available.

Writes Nature PDFs + ``summary.md`` under
``figure_dir("chi_variables", "variance_heatmaps")``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

_FULL = Path(__file__).resolve().parents[3]
_CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_FULL))
sys.path.insert(0, str(_CODE))

from config import (  # noqa: E402
    LABEL_FONTSIZE,
    TICK_LABELSIZE,
    add_panel_label,
    apply_full_paper_style,
    figsize,
    figure_dir,
    metric_label,
    save_figure,
)
from _shared import (  # noqa: E402
    CHI_OLS_CEILING,
    FACTORS,
    METRICS,
    N_CELLS,
    load_ratios,
    log_response,
)

apply_full_paper_style(auto_format=True, frame="open", grid=False)

H_LIST = [15.0, 50.0, 100.0]
VS1_LIST = [100.0, 230.0, 360.0]
COV_LEVELS = [0.1, 0.2, 0.3]
RH_LEVELS = [10.0, 30.0, 50.0]
CENTER_AHV = 10.0

FRAC_SPECS = (
    ("frac_W_seed", r"$f_{W\mid\mathrm{seed}}$"),
    ("frac_mu", r"$f_\mu$"),
    ("frac_B_node", r"$f_{B\mid\mathrm{node}}$"),
    ("frac_nu", r"$f_\nu$"),
)

CELL_SUMMARY = figure_dir("chi_variables", "central_variability") / "cell_summary.csv"


def _out() -> Path:
    return figure_dir("chi_variables", "variance_heatmaps")


def _format_level(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value)}"
    return f"{value:g}"


def compute_minimal_fracs(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal seed-split fractions when cell_summary.csv is missing."""
    work = df.copy()
    work["cell"] = work.groupby(list(FACTORS), sort=False).ngroup()
    rows: list[dict] = []
    for (_, g) in work.groupby("cell", sort=True):
        keys = {f: g[f].iloc[0] for f in FACTORS}
        for metric in METRICS:
            chi = g.pivot(index="node", columns="seed", values=metric)
            arr = chi.to_numpy(dtype=float)
            with np.errstate(invalid="ignore", divide="ignore"):
                Y = np.where(np.isfinite(arr) & (arr > 0), np.log(arr), np.nan)
            flat = Y[np.isfinite(Y)]
            if flat.size < 8:
                continue
            mu_j = np.nanmean(Y, axis=0)
            s2_W_bar = float(np.nanmean(np.nanvar(Y, axis=0, ddof=0)))
            s2_mu = float(np.nanvar(mu_j[np.isfinite(mu_j)], ddof=0))
            s2_B_bar = float(np.nanmean(np.nanvar(Y, axis=1, ddof=0)))
            nu_i = np.nanmean(Y, axis=1)
            s2_nu = float(np.nanvar(nu_i[np.isfinite(nu_i)], ddof=0))
            s2_total = float(np.nanvar(flat, ddof=0))
            if s2_total <= 0:
                continue
            rows.append(
                {
                    **keys,
                    "metric": metric,
                    "frac_W_seed": s2_W_bar / s2_total,
                    "frac_mu": s2_mu / s2_total,
                    "frac_B_node": s2_B_bar / s2_total,
                    "frac_nu": s2_nu / s2_total,
                    "s_total": float(np.sqrt(s2_total)),
                }
            )
    return pd.DataFrame(rows)


def load_cell_summary() -> tuple[pd.DataFrame, str]:
    if CELL_SUMMARY.is_file():
        df = pd.read_csv(CELL_SUMMARY)
        return df, str(CELL_SUMMARY)
    print(f"cell_summary.csv not found at {CELL_SUMMARY}; computing minimal fracs …")
    df = compute_minimal_fracs(load_ratios())
    return df, "computed_minimal"


def load_ceilings() -> dict[str, float]:
    path = CHI_OLS_CEILING
    if not path.is_file():
        # Fallback relative to figure_dir
        alt = figure_dir("chi_ols", "r2_ceiling") / "reliability_ceiling.csv"
        path = alt if alt.is_file() else path
    if not path.is_file():
        return {}
    c = pd.read_csv(path)
    full = c[c["scope"] == "full"] if "scope" in c.columns else c
    return {str(r["metric"]): float(r["reliability_ceiling"]) for _, r in full.iterrows()}


def _matrix(
    df: pd.DataFrame,
    *,
    metric: str,
    height: float,
    vs1: float,
    ahv: float,
    col: str,
) -> np.ndarray:
    """Return (n_CoV, n_rH) matrix of *col*."""
    out = np.full((len(COV_LEVELS), len(RH_LEVELS)), np.nan)
    sub = df[
        (df["metric"] == metric)
        & np.isclose(df["Height"].astype(float), height)
        & np.isclose(df["Vs1"].astype(float), vs1)
        & np.isclose(df["aHV"].astype(float), ahv)
    ]
    for i, cov in enumerate(COV_LEVELS):
        for j, rh in enumerate(RH_LEVELS):
            hit = sub[
                np.isclose(sub["CoV"].astype(float), cov)
                & np.isclose(sub["rH"].astype(float), rh)
            ]
            if len(hit):
                out[i, j] = float(hit[col].iloc[0])
    return out


def plot_frac_heatmaps(
    df: pd.DataFrame,
    ceilings: dict[str, float],
    *,
    frac_col: str,
    frac_label: str,
    stem: str,
) -> None:
    """One figure per metric: 3×3 Height×Vs1 panels of CoV×rH heatmaps."""
    out = _out()
    cmap = plt.get_cmap("viridis")
    for metric in METRICS:
        fig, axes = plt.subplots(
            len(H_LIST),
            len(VS1_LIST),
            figsize=figsize(aspect=0.85),
            constrained_layout=False,
        )
        fig.subplots_adjust(left=0.10, right=0.90, bottom=0.10, top=0.90, wspace=0.18, hspace=0.22)

        mats = []
        for r, h in enumerate(H_LIST):
            for c, vs1 in enumerate(VS1_LIST):
                mats.append(_matrix(df, metric=metric, height=h, vs1=vs1, ahv=CENTER_AHV, col=frac_col))
        finite = np.concatenate([m[np.isfinite(m)] for m in mats if np.isfinite(m).any()])
        if finite.size:
            vmin, vmax = float(np.min(finite)), float(np.max(finite))
            if abs(vmax - vmin) < 1e-6:
                vmin, vmax = max(0.0, vmin - 0.05), min(1.0, vmax + 0.05)
        else:
            vmin, vmax = 0.0, 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)

        panel_i = 0
        last_im = None
        for r, h in enumerate(H_LIST):
            for c, vs1 in enumerate(VS1_LIST):
                ax = axes[r, c]
                mat = mats[panel_i]
                last_im = ax.imshow(
                    mat,
                    origin="lower",
                    aspect="auto",
                    cmap=cmap,
                    norm=norm,
                    extent=(
                        -0.5,
                        len(RH_LEVELS) - 0.5,
                        -0.5,
                        len(COV_LEVELS) - 0.5,
                    ),
                )
                # Annotate cells
                for i in range(len(COV_LEVELS)):
                    for j in range(len(RH_LEVELS)):
                        v = mat[i, j]
                        if not np.isfinite(v):
                            continue
                        # Contrast text
                        tcolor = "white" if (v - vmin) / max(vmax - vmin, 1e-9) > 0.55 else "0.1"
                        ax.text(
                            j,
                            i,
                            f"{v:.2f}",
                            ha="center",
                            va="center",
                            fontsize=TICK_LABELSIZE - 0.5,
                            color=tcolor,
                        )
                ax.set_xticks(range(len(RH_LEVELS)))
                ax.set_xticklabels([_format_level(v) for v in RH_LEVELS])
                ax.set_yticks(range(len(COV_LEVELS)))
                ax.set_yticklabels([_format_level(v) for v in COV_LEVELS])
                if r == len(H_LIST) - 1:
                    ax.set_xlabel(r"$r_h$ (m)", fontsize=LABEL_FONTSIZE)
                else:
                    ax.tick_params(labelbottom=False)
                if c == 0:
                    ax.set_ylabel("CoV", fontsize=LABEL_FONTSIZE)
                else:
                    ax.tick_params(labelleft=False)
                ax.set_title(
                    rf"$H={_format_level(h)}$, $V_{{s1}}={_format_level(vs1)}$",
                    fontsize=TICK_LABELSIZE,
                )
                add_panel_label(ax, panel_i, alpha=0.7)
                panel_i += 1

        # Colorbar
        cax = fig.add_axes([0.92, 0.18, 0.018, 0.60])
        cb = fig.colorbar(last_im, cax=cax)
        cb.set_label(frac_label, fontsize=LABEL_FONTSIZE)
        cb.ax.tick_params(labelsize=TICK_LABELSIZE)

        ceil = ceilings.get(metric)
        ceil_txt = (
            rf"$R^2_{{\mathrm{{ceiling}}}}={ceil:.3f}$ (full array)"
            if ceil is not None and np.isfinite(ceil)
            else r"$R^2_{\mathrm{ceiling}}$ unavailable"
        )
        fig.suptitle(
            rf"{frac_label} — {metric_label(metric, log=True)}; "
            rf"$a_{{hv}}={_format_level(CENTER_AHV)}$. {ceil_txt}",
            fontsize=LABEL_FONTSIZE,
            y=0.97,
        )
        save_figure(fig, f"{stem}_{metric}", out_dir=out)
        plt.close(fig)


def plot_seed_split_pair(df: pd.DataFrame, ceilings: dict[str, float]) -> None:
    """Compact figure: frac_W and frac_mu at center Height/Vs1, all metrics."""
    out = _out()
    h_c, vs_c = 50.0, 230.0
    nrows, ncols = 2, len(METRICS)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize(aspect=0.48),
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.14, top=0.86, wspace=0.20, hspace=0.30)

    cmap = plt.get_cmap("viridis")
    panel_i = 0
    last_im = None
    for r, (frac_col, frac_label) in enumerate(
        (("frac_W_seed", r"$f_{W\mid\mathrm{seed}}$"), ("frac_mu", r"$f_\mu$"))
    ):
        mats = [
            _matrix(df, metric=m, height=h_c, vs1=vs_c, ahv=CENTER_AHV, col=frac_col)
            for m in METRICS
        ]
        finite = np.concatenate([m[np.isfinite(m)] for m in mats if np.isfinite(m).any()])
        vmin = float(np.min(finite)) if finite.size else 0.0
        vmax = float(np.max(finite)) if finite.size else 1.0
        if abs(vmax - vmin) < 1e-6:
            vmin, vmax = max(0.0, vmin - 0.05), min(1.0, vmax + 0.05)
        norm = Normalize(vmin=vmin, vmax=vmax)
        for c, metric in enumerate(METRICS):
            ax = axes[r, c]
            mat = mats[c]
            last_im = ax.imshow(
                mat,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                norm=norm,
            )
            for i in range(len(COV_LEVELS)):
                for j in range(len(RH_LEVELS)):
                    v = mat[i, j]
                    if not np.isfinite(v):
                        continue
                    tcolor = "white" if (v - vmin) / max(vmax - vmin, 1e-9) > 0.55 else "0.1"
                    ax.text(
                        j,
                        i,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=TICK_LABELSIZE - 0.5,
                        color=tcolor,
                    )
            ax.set_xticks(range(len(RH_LEVELS)))
            ax.set_yticks(range(len(COV_LEVELS)))
            if r == nrows - 1:
                ax.set_xticklabels([_format_level(v) for v in RH_LEVELS])
                ax.set_xlabel(r"$r_h$", fontsize=LABEL_FONTSIZE)
            else:
                ax.set_xticklabels([])
                ax.set_title(metric_label(metric, log=True), fontsize=LABEL_FONTSIZE)
            if c == 0:
                ax.set_yticklabels([_format_level(v) for v in COV_LEVELS])
                ax.set_ylabel(f"{frac_label}\nCoV", fontsize=LABEL_FONTSIZE)
            else:
                ax.set_yticklabels([])
            add_panel_label(ax, panel_i, alpha=0.7)
            panel_i += 1

    cax = fig.add_axes([0.92, 0.20, 0.015, 0.55])
    cb = fig.colorbar(last_im, cax=cax)
    cb.set_label("Fraction", fontsize=LABEL_FONTSIZE)
    cb.ax.tick_params(labelsize=TICK_LABELSIZE)

    ceil_bits = [
        f"{metric_label(m)}: {ceilings[m]:.3f}"
        for m in METRICS
        if m in ceilings and np.isfinite(ceilings[m])
    ]
    ceil_line = (
        r"$R^2_{\mathrm{ceiling}}$ (full): " + "; ".join(ceil_bits)
        if ceil_bits
        else r"$R^2_{\mathrm{ceiling}}$ unavailable"
    )
    fig.suptitle(
        rf"Seed-split fractions at $H={_format_level(h_c)}$, "
        rf"$V_{{s1}}={_format_level(vs_c)}$, $a_{{hv}}={_format_level(CENTER_AHV)}$. "
        + ceil_line,
        fontsize=LABEL_FONTSIZE - 0.5,
        y=0.97,
    )
    save_figure(fig, "frac_seed_split_center_HV", out_dir=out)
    plt.close(fig)


def build_summary_md(
    df: pd.DataFrame,
    source: str,
    ceilings: dict[str, float],
) -> str:
    lines = [
        "# Variance-fraction heatmaps",
        "",
        "Small-multiple heatmaps of law-of-total-variance fractions "
        r"(\(f_{W\mid\mathrm{seed}}\), \(f_\mu\), …) across CoV × \(r_h\) "
        f"at fixed \(a_{{hv}}={CENTER_AHV:g}\).",
        "",
        f"- Source: `{source}`",
        f"- Cells in table: **{df.groupby(list(FACTORS)).ngroup().nunique() if set(FACTORS).issubset(df.columns) else '—'}** "
        f"(design expects {N_CELLS})",
        "",
        "## Reliability ceiling callout",
        "",
    ]
    if ceilings:
        lines.extend(
            [
                "From `reliability_ceiling.csv` (scope=full):",
                "",
                "| Metric | \(R^2_{\\mathrm{ceiling}}\) |",
                "| --- | ---: |",
            ]
        )
        for m in METRICS:
            if m in ceilings:
                lines.append(f"| {m} | {ceilings[m]:.4f} |")
        lines.append("")
        lines.append(
            "Ceiling is the population-information bound on design-only "
            "prediction of a single \(Y\) draw; within-cell noise "
            r"(\(\approx f_{W\mid\mathrm{seed}}+f_\mu\) structure) is irreducible."
        )
    else:
        lines.append(
            "`reliability_ceiling.csv` not found; heatmaps omit numeric "
            r"\(R^2_{\mathrm{ceiling}}\) annotations. Run `chi_ols/r2_ceiling.py`."
        )

    # Center-cell snapshot
    lines.extend(
        [
            "",
            f"## Center slice (\(H=50\), \(V_{{s1}}=230\), \(a_{{hv}}={CENTER_AHV:g}\))",
            "",
            "| Metric | med \(f_W\) | med \(f_\\mu\) |",
            "| --- | ---: | ---: |",
        ]
    )
    sub = df[
        np.isclose(df["Height"].astype(float), 50.0)
        & np.isclose(df["Vs1"].astype(float), 230.0)
        & np.isclose(df["aHV"].astype(float), CENTER_AHV)
    ]
    for metric in METRICS:
        m = sub[sub["metric"] == metric]
        lines.append(
            "| {} | {:.3f} | {:.3f} |".format(
                metric,
                float(m["frac_W_seed"].median()) if len(m) else float("nan"),
                float(m["frac_mu"].median()) if len(m) else float("nan"),
            )
        )

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "| File | Content |",
            "| --- | --- |",
            "| `heatmap_frac_W_seed_<metric>.pdf` | CoV×rH over Height×Vs1 |",
            "| `heatmap_frac_mu_<metric>.pdf` | Same for \(f_\\mu\) |",
            "| `heatmap_frac_B_node_<metric>.pdf` / `frac_nu_*` | Node-split |",
            "| `frac_seed_split_center_HV.pdf` | Compact center H/Vs1 panel |",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    df, source = load_cell_summary()
    print(f"Loaded cell summary ({len(df):,} rows) from {source}")
    ceilings = load_ceilings()
    if ceilings:
        print(f"Loaded R²_ceiling for {len(ceilings)} metrics")
    else:
        print("R²_ceiling CSV not found; continuing without annotation values")

    out = _out()
    # Persist the table used (useful when computed_minimal)
    if source == "computed_minimal":
        df.to_csv(out / "cell_fracs_minimal.csv", index=False)

    print("Plotting …")
    for frac_col, frac_label in FRAC_SPECS:
        plot_frac_heatmaps(
            df,
            ceilings,
            frac_col=frac_col,
            frac_label=frac_label,
            stem=f"heatmap_{frac_col}",
        )
    plot_seed_split_pair(df, ceilings)

    md = build_summary_md(df, source, ceilings)
    md_path = out / "summary.md"
    md_path.write_text(md, encoding="utf-8")
    print()
    print(md)
    print(f"Wrote outputs under {out}")


if __name__ == "__main__":
    main()
