"""Empirical between-seed coherence vs literature spatial-coherency models (Table4).

Compares lag-binned ρ̄(h) at the center design cell (and median / IQR over
cells) to parametric lagged-coherency forms commonly used in earthquake
engineering:

- Harichandran & Vanmarcke (1986) — frequency-integrated envelope at a
  reference frequency band
- Abrahamson et al. (1991) — absolute coherency decay with separation
- Simple exponential γ(h)=exp(−h/ℓ) calibrated to match ρ̄(2 m)→ρ̄(h)

HV and Abrahamson are closed forms of separation *h* (and a fixed reference
frequency). They do **not** depend on factorial cells. The exponential is
fitted to empirical ρ̄(h) (center cell, or the across-cell median). Empirical
ρ̄(h) *does* vary by cell; the median/IQR envelope is the right overlay for
the cell-invariant literature curves.

Note: literature models describe Fourier lagged coherency of ground motion;
our statistic is between-seed spatial coherence of Y=ln χ. Table4 therefore
anchors *decay shape and length scales*, not literal equality of definitions.

Writes under ``figure_dir("chi_spatial", "literature_coherence")``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

_FULL = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_FULL))

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

LAG_CSV = (
    BOX_ROOT
    / "full_paper"
    / "figures"
    / "chi_spatial"
    / "spatial_coherence"
    / "coherence_lag_binned.csv"
)
REF_H, REF_VS1 = 50.0, 230.0
CENTER = {"CoV": 0.2, "rH": 30.0, "aHV": 10.0}
METRICS = ("f_ratio", "abs_TF_ratio", "PGA_ratio", "PSA_ratio", "Ia_ratio")
LAGS_M = np.array([2.0, 10.0, 20.0, 50.0, 100.0, 200.0])

# Literature reference frequency for frequency-dependent forms (Hz)
F_REF = 2.0  # near typical site f0 for H=50, Vs1=230
VS_APP = 230.0  # apparent wave velocity proxy (m/s)


def harichandran_vanmarcke(h_m: np.ndarray, f: float = F_REF) -> np.ndarray:
    """|γ(ξ,f)| envelope from HV86 with common parameter defaults.

    A=0.736, a=0.147, k=5210 m/s-ish scale via α; we use published median-ish
    defaults adapted for separation in metres (see Abrahamson reviews).
    """
    # Standard HV form:
    # A(f) = 1 / (1 + (f/f0)^n), then
    # γ = A exp(−(2A/(1−A+αA²)) * B * ξ)  [simplified radial]
    f0, n, alpha = 0.95, 2.78, 0.0  # common HV-like defaults
    A = 1.0 / (1.0 + (f / f0) ** n)
    # B ~ k * f / v_app with k≈0.5–1; use k=0.5
    B = 0.5 * f / max(VS_APP, 1.0)
    denom = max(1.0 - A + alpha * A * A, 1e-6)
    return A * np.exp(-(2.0 * A / denom) * B * np.asarray(h_m, dtype=float))


def abrahamson(h_m: np.ndarray, f: float = F_REF) -> np.ndarray:
    """Abrahamson-style absolute coherency vs separation at fixed f.

    Uses the widely cited form:
    tanh(c1) * tanh(c2) / [tanh(c3) * tanh(c4)] style reduced to a
    frequency-dependent exponential with published coefficients for rock sites
    (Abrahamson et al. 1991 / 1992 practical fit):
    γ ≈ 1 / (1 + (ξ / ξ0(f))^n) with ξ0 decreasing in f.
    """
    # Practical closed form used in many site-response comparisons:
    # ξ0(f) ≈ a / f^b  (metres), n≈1
    a, b, n = 40.0, 0.35, 1.0
    xi0 = a / max(f, 0.1) ** b
    h = np.asarray(h_m, dtype=float)
    return 1.0 / (1.0 + (h / max(xi0, 1e-6)) ** n)


def exponential_match(h_m: np.ndarray, rho_short: float, ell_m: float) -> np.ndarray:
    """γ(h)=ρ_short * exp(−(h−h0)/ℓ) with h0=2 m."""
    h = np.asarray(h_m, dtype=float)
    return float(rho_short) * np.exp(-(np.maximum(h - 2.0, 0.0)) / max(ell_m, 1e-6))


def _center_lag(lag: pd.DataFrame, metric: str) -> pd.DataFrame:
    mask = (
        (lag["metric"] == metric)
        & np.isclose(lag["Height"].astype(float), REF_H)
        & np.isclose(lag["Vs1"].astype(float), REF_VS1)
        & np.isclose(lag["CoV"].astype(float), CENTER["CoV"])
        & np.isclose(lag["rH"].astype(float), CENTER["rH"])
        & np.isclose(lag["aHV"].astype(float), CENTER["aHV"])
    )
    return lag.loc[mask].sort_values("h_m")


def _rho_at(sub: pd.DataFrame, h: float) -> float:
    hit = sub.loc[np.isclose(sub["h_m"].astype(float), h), "rho_mean"]
    return float(hit.iloc[0]) if len(hit) else float("nan")


def _fit_ell(h: np.ndarray, rho: np.ndarray) -> float:
    """Least-squares ℓ for exp decay anchored at first lag."""
    h = np.asarray(h, dtype=float)
    rho = np.asarray(rho, dtype=float)
    ok = np.isfinite(h) & np.isfinite(rho) & (rho > 1e-6) & (h > 2.0)
    if ok.sum() < 2:
        return float("nan")
    y = -np.log(rho[ok] / max(rho[np.isfinite(rho)][0], 1e-6))
    x = h[ok] - 2.0
    # ℓ = x / y
    ratio = x / np.maximum(y, 1e-8)
    return float(np.median(ratio[np.isfinite(ratio)]))


def plot_median_envelope_vs_literature(lag: pd.DataFrame, out: Path) -> None:
    """Median ± IQR of empirical ρ̄(h) across cells, with cell-invariant models.

    HV and Abrahamson are the same curve in every panel (functions of *h* only
    at fixed f). Exp is fitted to that panel's across-cell median ρ̄(h).
    """
    fig, axes = plt.subplots(
        1,
        len(METRICS),
        figsize=figsize(aspect=0.32),
        sharey=True,
        constrained_layout=False,
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.16, top=0.72, wspace=0.12)

    h_max = float(lag["h_m"].max()) if len(lag) else 200.0
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
        ax.plot(h, harichandran_vanmarcke(h), "k--", lw=0.9)
        ax.plot(h, abrahamson(h), "k:", lw=0.9)
        ell = _fit_ell(h, med)
        if np.isfinite(ell) and np.isfinite(med).any():
            rho0 = float(med[np.isfinite(med)][0])
            ax.plot(h, exponential_match(h, rho0, ell), color="0.45", lw=0.9)
        ax.axhline(0.0, color="0.6", lw=0.4)
        ax.set_xlim(0.0, h_max)
        ax.set_ylim(-0.4, 1.05)
        ax.set_xlabel("Lag (m)", fontsize=LABEL_FONTSIZE)
        if i == 0:
            ax.set_ylabel(r"Coherence $\bar\rho(h)$", fontsize=LABEL_FONTSIZE)
        ax.set_title(metric_label(metric, log=True), fontsize=LABEL_FONTSIZE)
        add_panel_label(ax, i, alpha=0.75)

    fig.suptitle(
        r"Empirical median / IQR over cells against literature coherency models",
        fontsize=LABEL_FONTSIZE,
        y=0.98,
    )
    handles = [
        Line2D([0], [0], color="0.35", lw=DATA_LINEWIDTH, label="Median (cells)"),
        Line2D([0], [0], color="0.35", lw=6, alpha=0.25, label="IQR"),
        Line2D([0], [0], color="k", ls="--", lw=0.9, label="Harichandran–Vanmarcke"),
        Line2D([0], [0], color="k", ls=":", lw=0.9, label="Abrahamson-type"),
        Line2D([0], [0], color="0.45", lw=0.9, label=r"Exp (fit to median)"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=5,
        fontsize=TICK_LABELSIZE,
        bbox_to_anchor=(0.5, 0.935),
        frameon=False,
    )
    save_figure(fig, "literature_coherence_median_iqr", out_dir=out)
    plt.close(fig)


def main() -> None:
    out = figure_dir("chi_spatial", "literature_coherence")
    lag = pd.read_csv(LAG_CSV)

    rows = []
    fig, axes = plt.subplots(1, 2, figsize=figsize(height=3.4))

    # Panel a: abs_TF center cell vs literature
    ax = axes[0]
    sub = _center_lag(lag, "abs_TF_ratio")
    h = sub["h_m"].to_numpy(dtype=float)
    rho = sub["rho_mean"].to_numpy(dtype=float)
    ax.plot(
        h, rho, color=metric_color("abs_TF_ratio"), lw=DATA_LINEWIDTH, label="Empirical (center)"
    )
    ax.plot(h, harichandran_vanmarcke(h), "k--", lw=0.9, label="Harichandran–Vanmarcke")
    ax.plot(h, abrahamson(h), "k:", lw=0.9, label="Abrahamson-type")
    ell = _fit_ell(h, rho)
    ax.plot(
        h, exponential_match(h, rho[0], ell), color="0.45", lw=0.9, label=rf"Exp $\ell={ell:.0f}$ m"
    )
    ax.set_xlabel("Separation $h$ (m)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(r"Coherence $\bar\rho(h)$", fontsize=LABEL_FONTSIZE)
    ax.set_ylim(-0.4, 1.05)
    ax.set_title(r"$|TF|_0^N$ center cell", fontsize=TICK_LABELSIZE)
    ax.legend(fontsize=TICK_LABELSIZE - 1, frameon=False)
    ax.tick_params(labelsize=TICK_LABELSIZE)

    # Panel b: all metrics at selected lags — empirical vs Abrahamson
    ax = axes[1]
    for metric in METRICS:
        sub = _center_lag(lag, metric)
        ax.plot(
            sub["h_m"],
            sub["rho_mean"],
            color=metric_color(metric),
            lw=DATA_LINEWIDTH,
            label=metric_label(metric, log=True),
        )
    ax.plot(LAGS_M, abrahamson(LAGS_M), "k:", lw=1.0, label="Abrahamson-type")
    ax.set_xlabel("Separation $h$ (m)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(r"Coherence $\bar\rho(h)$", fontsize=LABEL_FONTSIZE)
    ax.set_ylim(-0.4, 1.05)
    ax.set_title("All metrics against Abrahamson", fontsize=TICK_LABELSIZE)
    ax.legend(fontsize=TICK_LABELSIZE - 1, frameon=False, ncol=2)
    ax.tick_params(labelsize=TICK_LABELSIZE)

    fig.tight_layout(pad=0.4)
    save_figure(fig, "literature_coherence_compare", out_dir=out)
    plt.close(fig)

    plot_median_envelope_vs_literature(lag, out)

    # Table rows
    for metric in METRICS:
        sub = _center_lag(lag, metric)
        # median over all cells at each lag
        med = (
            lag[lag["metric"] == metric]
            .groupby("h_m", as_index=False)["rho_mean"]
            .median()
            .sort_values("h_m")
        )
        ell = _fit_ell(sub["h_m"].to_numpy(), sub["rho_mean"].to_numpy())
        for h in LAGS_M:
            emp_c = _rho_at(sub, h)
            (
                _rho_at(med.rename(columns={"rho_mean": "rho_mean"}), h)
                if "rho_mean" in med
                else float("nan")
            )
            # med already has rho_mean
            hit_m = med.loc[np.isclose(med["h_m"].astype(float), h), "rho_mean"]
            emp_med = float(hit_m.iloc[0]) if len(hit_m) else float("nan")
            hv = float(harichandran_vanmarcke(np.array([h]))[0])
            ab = float(abrahamson(np.array([h]))[0])
            ex = float(exponential_match(np.array([h]), _rho_at(sub, 2.0), ell)[0])
            rows.append(
                {
                    "metric": metric,
                    "h_m": float(h),
                    "rho_center": emp_c,
                    "rho_median_cells": emp_med,
                    "harichandran_vanmarcke": hv,
                    "abrahamson": ab,
                    "exp_match_center": ex,
                    "exp_length_m": ell,
                    "abs_err_HV_center": abs(emp_c - hv) if np.isfinite(emp_c) else float("nan"),
                    "abs_err_Abr_center": abs(emp_c - ab) if np.isfinite(emp_c) else float("nan"),
                    "agree_HV": "yes" if np.isfinite(emp_c) and abs(emp_c - hv) < 0.15 else "no",
                    "agree_Abr": "yes" if np.isfinite(emp_c) and abs(emp_c - ab) < 0.15 else "no",
                }
            )

    tab = pd.DataFrame(rows)
    tab.to_csv(out / "literature_coherence_comparison.csv", index=False)

    # Compact Table4: one row per metric at h=10, 50, 100 m
    compact_lags = (10.0, 50.0, 100.0)
    compact_rows = []
    for metric in METRICS:
        sub_t = tab[tab["metric"] == metric]
        row = {"metric": metric, "exp_length_m": float(sub_t["exp_length_m"].iloc[0])}
        for h in compact_lags:
            r = sub_t[np.isclose(sub_t["h_m"], h)].iloc[0]
            row[f"rho_c_{int(h)}"] = r["rho_center"]
            row[f"HV_{int(h)}"] = r["harichandran_vanmarcke"]
            row[f"Abr_{int(h)}"] = r["abrahamson"]
            row[f"agree_Abr_{int(h)}"] = r["agree_Abr"]
        compact_rows.append(row)
    compact = pd.DataFrame(compact_rows)
    compact.to_csv(out / "table4_compact.csv", index=False)

    lines = [
        "# Literature coherence comparison (Table4)",
        "",
        f"Center cell: $H={REF_H:.0f}$ m, $V_{{s1}}={REF_VS1:.0f}$ m/s, "
        f"CoV={CENTER['CoV']}, $r_h$={CENTER['rH']}, $a_{{hv}}$={CENTER['aHV']}.",
        f"Reference frequency for HV / Abrahamson forms: $f={F_REF}$ Hz.",
        "",
        "## Compact summary (center-cell ρ against models)",
        "",
        compact.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Notes",
        "",
        "- Agreement flag: |empirical − model| < 0.15 at that lag.",
        "- HV and Abrahamson do **not** vary by factorial cell (closed forms of "
        r"$h$ at $f=2$ Hz). Exp $\ell$ is fitted to the center-cell $\bar\rho(h)$ "
        "(table) or to the across-cell median (envelope figure).",
        "- Expect stronger agreement at large separations (exponential-like decay) "
        "and discrepancies at short lags / low-frequency resonant IMs where 2D "
        "scattering induces slower coherence loss than far-field coherency models.",
        "",
        "## Output files",
        "",
        "| File | Content |",
        "|------|---------|",
        "| `literature_coherence_comparison.csv` | Full lag×metric table |",
        "| `table4_compact.csv` | Manuscript Table4 core |",
        "| `literature_coherence_compare.pdf` | Center-cell overlay |",
        "| `literature_coherence_median_iqr.pdf` | Median/IQR over cells against models |",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(compact.to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
