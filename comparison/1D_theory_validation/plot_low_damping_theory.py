"""Show that usual light-damping 1D AF_within peaks are ~1/ξ, not ~25.

Hallal OpenSees global_avg uses Q-based ξ ≈ 3–5%, which caps the first-mode
within peak near 2/(πξ) ≈ 15–25. Textbook 1D theory with ξ ~ 0.5–1% reaches ~60–130.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from seiskit.plot_config import apply_style
from seiskit.theory.layered_1d_tf import Layer, RockHalfspace, layered_transfer_function

OUT = Path(__file__).resolve().parent / "results" / "rayleigh_equivalent"
RHO = 2000.0
XI_LIGHT = (0.005, 0.01)


def peak_rigid_approx(xi: float) -> float:
    """First-mode |AF_within| ≈ 2/(πξ) for small hysteretic damping, rigid rock."""
    return 2.0 / (np.pi * xi)


def theory_xi(freq: np.ndarray, vs1: float, H: float, vs2: float, xi_s: float, xi_r: float = 0.0):
    _, af, _ = layered_transfer_function(
        freq,
        [Layer(float(H), float(vs1), RHO, float(xi_s))],
        RockHalfspace(float(vs2), RHO, float(xi_r)),
    )
    return np.asarray(af, dtype=float)


def main() -> None:
    apply_style()
    rows = json.loads((OUT / "profiles_metrics.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.8), sharex=True)
    fig.subplots_adjust(top=0.90, hspace=0.42, wspace=0.22)

    for ax, row in zip(axes.ravel(), rows):
        d = np.load(row["npz"])
        freq = d["freq"]
        vs1, H, vs2 = float(d["vs1"]), float(d["H"]), float(d["vs2"])
        ax.loglog(freq, d["af_os"], color="0.45", lw=1.4, label="OpenSees 1D (Q-Rayleigh)")
        ax.loglog(
            freq,
            d["af_th_rayleigh"],
            color="#0072B2",
            lw=1.8,
            label=r"Theory Rayleigh $\xi(f)$",
        )
        colors = ("#D55E00", "#CC79A7")
        for xi, c in zip(XI_LIGHT, colors):
            af = theory_xi(freq, vs1, H, vs2, xi, 0.0)
            ax.loglog(freq, af, color=c, lw=1.5, ls="--", label=fr"Theory $\xi$={100*xi:.1f}%")
        ax.axvline(row["f0"], color="0.4", ls="--", lw=0.7)
        ax.set_xlim(0.1, 10)
        ax.set_ylim(0.5, 200)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(
            fr"{row['label']}: $V_{{s1}}$={row['vs1']:.0f}, $H$={row['H']:.0f} m"
            + "\n"
            + fr"$f_0$={row['f0']:.2f} Hz, $\xi_Q$={100*row['xi_soil_Q']:.2f}%",
            fontsize=9,
        )
        ax.annotate(
            fr"$2/(\pi\xi_Q)$={peak_rigid_approx(row['xi_soil_Q']):.0f}"
            + "\n"
            + fr"$2/(\pi\cdot 0.5\%)$={peak_rigid_approx(0.005):.0f}",
            xy=(0.03, 0.97),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="0.8"),
        )
        if ax is axes[0, 0]:
            ax.legend(loc="center right", fontsize=7.5, frameon=True)

    for ax in axes[-1]:
        ax.set_xlabel("Frequency (Hz)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$|AF_\mathrm{within}|$")
    fig.suptitle(
        r"Light-damping 1D theory peaks near $2/(\pi\xi)$ ($\xi$=0.5% $\Rightarrow$ $\sim$127); "
        r"OpenSees Q-Rayleigh caps peaks near 15–25",
        fontsize=11,
    )
    out_grid = OUT / "af_light_damping_theory.png"
    fig.savefig(out_grid, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Peak vs ξ for the demo column.
    demo = next(r for r in rows if r["name"] == "demo_230_15")
    d = np.load(demo["npz"])
    freq, vs1, H, vs2 = d["freq"], float(d["vs1"]), float(d["H"]), float(d["vs2"])
    xis = np.array([0.003, 0.005, 0.0075, 0.01, 0.015, 0.025, 0.036, 0.05])
    peaks = [float(np.max(theory_xi(freq, vs1, H, vs2, xi, 0.0))) for xi in xis]

    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    ax.plot(100 * xis, peaks, "o-", color="#0072B2", lw=1.8, ms=6, label="Thomson–Haskell (this column)")
    ax.plot(100 * xis, peak_rigid_approx(xis), "--", color="0.35", lw=1.4, label=r"$2/(\pi\xi)$ (rigid-rock)")
    ax.axhline(float(np.nanmax(d["af_os"])), color="0.45", lw=1.3, label="OpenSees Q-Rayleigh peak")
    ax.axvline(100 * demo["xi_soil_Q"], color="#009E73", ls=":", lw=1.4, label=fr"Soil $\xi_Q$={100*demo['xi_soil_Q']:.2f}%")
    ax.axvline(2.5, color="#D55E00", ls=":", lw=1.4, label="Config ζ=2.5%")
    ax.set_xlabel("Hysteretic damping ξ (%)")
    ax.set_ylabel(r"Peak $|AF_\mathrm{within}|$")
    ax.set_title(fr"Demo column $V_{{s1}}$=230 m/s, $H$=15 m: peak vs ξ")
    ax.set_xlim(0, 5.5)
    ax.set_ylim(0, 220)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    out_peak = OUT / "af_peak_vs_xi.png"
    fig.savefig(out_peak, dpi=160)
    plt.close(fig)
    print(out_grid)
    print(out_peak)


if __name__ == "__main__":
    main()
