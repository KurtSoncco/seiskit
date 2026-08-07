"""Field-stat recovery + CoV-swept Vs realization panels for the full paper.

1. ``vs_cov_realizations.pdf`` — 1×3 panels sweeping CoV at fixed (rH, aHV)
2. ``field_stat_recovery.csv`` + summary — realized CoV and horizontal
   correlation length vs targets for the factorial panel set
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    ANNOTATION_FONTSIZE,
    LABEL_FONTSIZE,
    add_panel_label,
    apply_full_paper_style,
    figsize,
    figure_dir,
    save_figure,
)

from seiskit.gaussian_field import create_vs_realization
from seiskit.plot_config import get_crameri_cmap

apply_full_paper_style(auto_format=True, frame="open", grid=False)

OUT = figure_dir("vs_rh_realizations")

VMIN, VMAX = 150.0, 350.0
VS1, VS2 = 230.0, 1500.0
SEED = 42
LX = 500.0
LX_VAR = 500.0
LZ = 60.0
DX = DZ = 1.0
DZ_1D = 1.0
SOIL_THICKNESS = 50.0
BEDROCK_THICKNESS = 10.0

COV_SWEEP = [0.1, 0.2, 0.3]
RH_FIXED, AHV_FIXED = 30.0, 10.0

# Recovery grid: CoV × rH at fixed aHV
RECOVERY_COV = [0.1, 0.2, 0.3]
RECOVERY_RH = [10.0, 30.0, 50.0]
RECOVERY_AHV = 10.0


def _vs_profile() -> np.ndarray:
    n_soil = int(SOIL_THICKNESS / DZ_1D)
    n_rock = int(BEDROCK_THICKNESS / DZ_1D)
    return np.array([VS1] * n_soil + [VS2] * n_rock, dtype=np.float64)


def _soil_mask_vs(vs: np.ndarray) -> np.ndarray:
    """Soil columns only (exclude bedrock overshoot)."""
    n_soil = int(SOIL_THICKNESS / DZ)
    return vs[:n_soil, :]


def realized_cov(vs: np.ndarray) -> float:
    soil = _soil_mask_vs(vs)
    mu = float(np.mean(soil))
    if mu <= 0:
        return float("nan")
    return float(np.std(soil, ddof=0) / mu)


def empirical_corr_length_h(vs: np.ndarray, max_lag_m: float = 100.0) -> float:
    """Rough horizontal correlation length: lag where mean ACF first drops below 1/e."""
    soil = _soil_mask_vs(vs)
    # demean each depth row, pool lag products horizontally
    n_z, n_x = soil.shape
    max_lag = min(int(max_lag_m / DX), n_x - 2)
    if max_lag < 2:
        return float("nan")
    sum_prod = np.zeros(max_lag, dtype=float)
    n_pairs = np.zeros(max_lag, dtype=float)
    for iz in range(n_z):
        row = soil[iz].astype(float)
        row = row - row.mean()
        var = float(np.dot(row, row))
        if var <= 0:
            continue
        for lag in range(1, max_lag + 1):
            a, b = row[:-lag], row[lag:]
            sum_prod[lag - 1] += float(np.dot(a, b))
            n_pairs[lag - 1] += float(len(a))
    with np.errstate(invalid="ignore", divide="ignore"):
        # normalize by lag-0 energy proxy
        rho0 = sum_prod  # will divide by n and by var proxy using lag products
        # Use Pearson per lag averaged: rebuild simply
    rhos = []
    for lag in range(1, max_lag + 1):
        vals = []
        for iz in range(n_z):
            row = soil[iz].astype(float)
            a, b = row[:-lag], row[lag:]
            if a.size < 3:
                continue
            a = a - a.mean()
            b = b - b.mean()
            den = np.sqrt(np.sum(a**2) * np.sum(b**2))
            if den > 0:
                vals.append(float(np.sum(a * b) / den))
        rhos.append(float(np.mean(vals)) if vals else float("nan"))
    thr = 1.0 / np.e
    for lag, rho in enumerate(rhos, start=1):
        if np.isfinite(rho) and rho < thr:
            return float(lag * DX)
    return float(max_lag * DX)


def make_field(rH: float, aHV: float, CV: float, seed: int = SEED) -> np.ndarray:
    vs, *_ = create_vs_realization(
        Vs_profile=_vs_profile(),
        Lx=LX,
        Lx_variability=LX_VAR,
        Lz=LZ,
        dx=DX,
        dz=DZ,
        rH=rH,
        aHV=aHV,
        CV=CV,
        seed=seed,
        dz_1D=DZ_1D,
        interlayer_amplitude=0.0,
    )
    return vs


def plot_cov_sweep() -> None:
    cmap = get_crameri_cmap("navia", reverse=False).copy()
    cmap.set_over("gray")
    extent = (-LX / 2.0, LX / 2.0, LZ, 0.0)
    fig, axes = plt.subplots(1, 3, figsize=figsize(aspect=0.38), sharey=True)
    for i, (ax, cov) in enumerate(zip(axes, COV_SWEEP)):
        vs = make_field(RH_FIXED, AHV_FIXED, cov)
        im = ax.imshow(
            vs,
            cmap=cmap,
            vmin=VMIN,
            vmax=VMAX,
            aspect="auto",
            interpolation="nearest",
            origin="upper",
            extent=extent,
        )
        ax.set_xlabel(r"$x$ (m)")
        if i == 0:
            ax.set_ylabel(r"$z$ (m)")
        ax.text(
            0.02,
            0.98,
            rf"CoV $= {cov:g}$" + "\n" + rf"$r_h={RH_FIXED:.0f}$, $a_{{hv}}={AHV_FIXED:.0f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=ANNOTATION_FONTSIZE,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.8},
        )
        add_panel_label(ax, i)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label=r"$V_s$ (m/s)")
    save_figure(fig, OUT / "vs_cov_realizations")
    plt.close(fig)


def run_recovery() -> pd.DataFrame:
    rows = []
    for cov in RECOVERY_COV:
        for rH in RECOVERY_RH:
            vs = make_field(rH, RECOVERY_AHV, cov)
            # Target horizontal length ≈ rH for isotropic-style fields (document as target)
            rows.append(
                {
                    "CoV_target": cov,
                    "rH_target_m": rH,
                    "aHV": RECOVERY_AHV,
                    "CoV_realized": realized_cov(vs),
                    "corr_length_h_m": empirical_corr_length_h(vs),
                    "seed": SEED,
                }
            )
    df = pd.DataFrame(rows)
    df["CoV_rel_err"] = (df["CoV_realized"] - df["CoV_target"]) / df["CoV_target"]
    df["rH_rel_err"] = (df["corr_length_h_m"] - df["rH_target_m"]) / df["rH_target_m"]
    df.to_csv(OUT / "field_stat_recovery.csv", index=False)
    lines = [
        "# Field-stat recovery",
        "",
        "Realized soil CoV and empirical horizontal correlation length (lag to 1/e)",
        f"for seed={SEED}, aHV={RECOVERY_AHV}.",
        "",
        f"- Median |CoV relative error|: {df['CoV_rel_err'].abs().median():.3f}",
        f"- Median |rH relative error|: {df['rH_rel_err'].abs().median():.3f}",
        "",
        "See `field_stat_recovery.csv` and `vs_cov_realizations.pdf`.",
        "",
    ]
    (OUT / "field_stat_recovery_summary.md").write_text("\n".join(lines), encoding="utf-8")
    return df


def main() -> None:
    print("CoV-swept Vs panels …")
    plot_cov_sweep()
    print("Field-stat recovery …")
    df = run_recovery()
    print(df.to_string(index=False))
    print(f"Wrote → {OUT}")


if __name__ == "__main__":
    main()
