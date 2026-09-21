"""Dmult-scaled Campbell ξ_Q: one 1D Thomson–Haskell TF + Pearson boxplot.

No Toro / Passeri / Vs randomization. Dmult is a standalone damping multiplier
from Vs contrast:

    Dmult = clip(-1.3 * (Vs2/Vs1) + 13.90, 2, 10)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from seiskit.damping import compute_damping_from_Q, compute_quality_factor
from seiskit.plot_config import apply_style
from seiskit.profile_randomization import (
    dmult_from_vs_contrast,
    vs_contrast,
)
from seiskit.theory.layered_1d_tf import Layer, RockHalfspace, layered_transfer_function

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "rayleigh_equivalent"
_RV = ROOT.parent / "Response_Variability"
if str(_RV) not in sys.path:
    sys.path.insert(0, str(_RV))

from sobol_base_cases import generate_base_cases  # noqa: E402

# Hallal-equivalent 2-layer median column (travel-time soil + rock)
H_SOIL = 88.30
H_ROCK = 15.70
VS_ROCK = 2298.12
RHO_SOIL = 1800.0
RHO_ROCK = 2200.0
N_PEARSON = 200
SAMPLER_SEED = 42
FREQ_LO = 0.1
FREQ_HI = 30.0
N_FREQ = 800


def travel_time_vs() -> float:
    h = np.array([10.59, 18.28, 11.12, 31.90, 16.41])
    vs = np.array([139.19, 174.73, 305.03, 254.69, 334.65])
    return float(np.sum(h) / np.sum(h / vs))


def xi_of_vs(vs: float, scale: float = 1.0) -> float:
    return float(scale * compute_damping_from_Q(compute_quality_factor(float(vs))))


def dmult_from_contrast(vs1: float, vs2: float) -> tuple[float, float]:
    return vs_contrast(vs1, vs2), dmult_from_vs_contrast(vs1, vs2)


def two_layer_tf(
    freq: np.ndarray,
    vs1: float,
    H: float,
    vs2: float,
    *,
    soil_scale: float,
    rock_scale: float = 1.0,
) -> np.ndarray:
    layers = [Layer(float(H), float(vs1), RHO_SOIL, xi_of_vs(vs1, soil_scale))]
    rock = RockHalfspace(float(vs2), RHO_ROCK, xi_of_vs(vs2, rock_scale))
    _, aw, _ = layered_transfer_function(freq, layers, rock)
    return np.asarray(aw, dtype=float)


def pearson_ln(a: np.ndarray, b: np.ndarray) -> float:
    x = np.log(np.clip(np.asarray(a, float), 1e-12, None))
    y = np.log(np.clip(np.asarray(b, float), 1e-12, None))
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return np.nan
    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def summarize(name: str, r: np.ndarray) -> None:
    r = r[np.isfinite(r)]
    print(
        f"{name:12s}  n={len(r)}  mean={np.mean(r):.3f}  "
        f"p50={np.median(r):.3f}  p16–p84={np.percentile(r,16):.3f}–{np.percentile(r,84):.3f}  "
        f"min={np.min(r):.3f}  max={np.max(r):.3f}"
    )


def damping_profile(vs1: float, H: float, vs2: float, *, scale: float = 1.0):
    """global_avg-style ξ staircase: soil harmonic ξ_Q(vs1), rock ξ_Q(vs2)."""
    xi_s = xi_of_vs(vs1, scale)
    xi_r = xi_of_vs(vs2, scale)
    z = np.array([0.0, H, H, H + H_ROCK])
    xi = np.array([xi_s, xi_s, xi_r, xi_r])
    return z, xi, xi_s, xi_r


def plot_vs_d_tf(
    vs1: float,
    H: float,
    vs2: float,
    contrast: float,
    dmult: float,
    freq: np.ndarray,
    af_q: np.ndarray,
    af_d: np.ndarray,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8), constrained_layout=True)
    z_max = H + H_ROCK

    # --- Vs ---
    ax = axes[0]
    z_vs = np.array([0.0, H, H, z_max])
    vs_stair = np.array([vs1, vs1, vs2, vs2])
    ax.plot(vs_stair, z_vs, color="#0072B2", lw=2.0)
    ax.set_ylim(z_max, 0.0)
    ax.set_xlabel(r"$V_s$ (m/s)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(r"$V_s$ profile")
    ax.grid(True, alpha=0.3)
    ax.annotate(
        fr"Contrast $=V_{{s2}}/V_{{s1}}$={contrast:.2f}" + "\n"
        + fr"$D_{{\mathrm{{mult}}}}$={dmult:.2f}" + "\n"
        + fr"$V_{{s1}}$={vs1:.1f} m/s" + "\n"
        + fr"$H$={H:.2f} m" + "\n"
        + fr"$V_{{s2}}$={vs2:.1f} m/s",
        xy=(0.97, 0.03),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="0.8"),
    )

    # --- Damping D = ξ (global_avg base vs Dmult × whole profile) ---
    ax = axes[1]
    z_d, xi_base, xi_s, xi_r = damping_profile(vs1, H, vs2, scale=1.0)
    _, xi_dm, xi_s_d, xi_r_d = damping_profile(vs1, H, vs2, scale=dmult)
    ax.plot(100.0 * xi_base, z_d, color="0.35", lw=1.8, ls="--", label=r"base $\xi_Q$")
    ax.plot(
        100.0 * xi_dm,
        z_d,
        color="#D55E00",
        lw=2.0,
        label=fr"$D_{{\mathrm{{mult}}}}\,\xi_Q$",
    )
    ax.set_ylim(z_max, 0.0)
    ax.set_xlabel(r"Damping $D=\xi$ (%)")
    ax.set_title("Damping profile")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    ax.annotate(
        fr"soil: {100*xi_s:.2f}% → {100*xi_s_d:.2f}%" + "\n"
        + fr"rock: {100*xi_r:.2f}% → {100*xi_r_d:.2f}%",
        xy=(0.97, 0.97),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="0.8"),
    )

    # --- TF ---
    ax = axes[2]
    ax.loglog(freq, af_q, color="0.35", lw=1.5, ls="--", label=r"$\xi_Q$ (Campbell)")
    ax.loglog(
        freq,
        af_d,
        color="#D55E00",
        lw=2.0,
        label=fr"$D_{{\mathrm{{mult}}}}\,\xi_Q$  ($D_{{\mathrm{{mult}}}}$={dmult:.2f})",
    )
    ax.set_xlim(FREQ_LO, FREQ_HI)
    ax.set_ylim(0.5, 200)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"$|AF_\mathrm{within}|$")
    ax.set_title("Thomson–Haskell 1D TF")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=8, frameon=True)
    ax.annotate(
        fr"Contrast={contrast:.2f},  $D_{{\mathrm{{mult}}}}$={dmult:.2f}" + "\n"
        + fr"$A_1(\xi_Q)$={np.max(af_q):.1f}" + "\n"
        + fr"$A_1(D_{{\mathrm{{mult}}}})$={np.max(af_d):.1f}",
        xy=(0.97, 0.97),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="0.8"),
    )

    fig.suptitle(
        r"$D_{\mathrm{mult}}=\mathrm{clip}(-1.3\,V_{s2}/V_{s1}+13.90,\ 2,\ 10)$"
        r"  —  whole-profile $\xi$ (soil + rock)",
        fontsize=11,
    )
    out = OUT / "af_dmult_campbell.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def plot_pearson_box(r: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(4.6, 4.8), constrained_layout=True)
    data = r[np.isfinite(r)]
    bp = ax.boxplot(
        [data],
        tick_labels=[r"$D_{\mathrm{mult}}$"],
        patch_artist=True,
        widths=0.45,
        medianprops=dict(color="0.15", lw=1.6),
        whiskerprops=dict(color="0.25"),
        capprops=dict(color="0.25"),
        flierprops=dict(
            marker="o",
            ms=3.5,
            markerfacecolor="0.5",
            markeredgecolor="none",
            alpha=0.7,
        ),
    )
    bp["boxes"][0].set_facecolor("#D55E00")
    bp["boxes"][0].set_alpha(0.75)
    bp["boxes"][0].set_edgecolor("0.2")
    ax.set_ylim(-0.15, 1.05)
    ax.set_ylabel(r"Pearson $r(\ln|AF|_{D_{\mathrm{mult}}},\ \ln|AF|_{\xi_Q})$")
    ax.set_title(fr"$D_{{\mathrm{{mult}}}}$ vs $\xi_Q$ 1D  ($n$={len(data)} columns)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.annotate(
        fr"p50={np.median(data):.2f}",
        xy=(1, np.median(data)),
        xytext=(1.28, np.median(data)),
        fontsize=8,
        va="center",
        color="0.2",
    )
    out = OUT / "dmult_vs_1d_pearson_box.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    freq = np.logspace(np.log10(FREQ_LO), np.log10(FREQ_HI), N_FREQ)

    # --- Figure 1: Hallal-equivalent column, one Dmult TF ---
    vs1 = travel_time_vs()
    H = H_SOIL
    vs2 = VS_ROCK
    contrast, dmult = dmult_from_contrast(vs1, vs2)
    af_q = two_layer_tf(freq, vs1, H, vs2, soil_scale=1.0, rock_scale=1.0)
    af_d = two_layer_tf(freq, vs1, H, vs2, soil_scale=dmult, rock_scale=dmult)
    print(
        f"Hallal column: Vs1={vs1:.2f}  H={H:.2f}  Vs2={vs2:.2f}\n"
        f"  Contrast={contrast:.3f}  Dmult={dmult:.3f}\n"
        f"  A1(xi_Q)={np.max(af_q):.2f}  A1(Dmult)={np.max(af_d):.2f}"
    )
    out_tf = plot_vs_d_tf(vs1, H, vs2, contrast, dmult, freq, af_q, af_d)
    print(out_tf)

    # --- Figure 2: Pearson over 200 independent 2-layer columns ---
    cases = generate_base_cases(N_PEARSON, sampler_seed=SAMPLER_SEED)
    rows = []
    r = np.empty(N_PEARSON)
    for i, case in enumerate(cases):
        c, dm = dmult_from_contrast(case.vs1, case.vs2)
        af_base = two_layer_tf(freq, case.vs1, case.H, case.vs2, soil_scale=1.0, rock_scale=1.0)
        af_dm = two_layer_tf(freq, case.vs1, case.H, case.vs2, soil_scale=dm, rock_scale=dm)
        r[i] = pearson_ln(af_dm, af_base)
        rows.append((i, case.vs1, case.H, case.vs2, c, dm, r[i]))

    print(f"\nDmult Pearson vs xi_Q 1D  n={N_PEARSON}  (independent 2-layer columns)")
    summarize("Dmult", r)
    csv_path = OUT / "dmult_vs_1d_pearson.csv"
    np.savetxt(
        csv_path,
        np.array(rows),
        delimiter=",",
        header="id,Vs1,H,Vs2,contrast,Dmult,pearson_r",
        comments="",
    )
    print(csv_path)
    out_box = plot_pearson_box(r)
    print(out_box)


if __name__ == "__main__":
    main()
