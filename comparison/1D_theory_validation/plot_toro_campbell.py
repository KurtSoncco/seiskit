"""Three Toro ensembles: Vs-only, Vs+H, and full (NHPP+H+Vs). Campbell vs 0.25 ξ_Q."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from seiskit.damping import compute_damping_from_Q, compute_quality_factor
from seiskit.plot_config import apply_style
from seiskit.profile_randomization import (
    ProfileRandomizationConfig,
    build_base_case_profile,
    generate_toro_profile,
)
from seiskit.theory.layered_1d_tf import Layer, RockHalfspace, layered_transfer_function

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "rayleigh_equivalent"

H_SOIL = 88.30
H_ROCK = 15.70
VS_ROCK = 2298.12
RHO_SOIL = 1800.0
RHO_ROCK = 2200.0
DZ = 0.5
N_REAL = 80
COV = 0.20
SIGMA_LN_VS = COV
XI_SCALE_Q = 1.0
XI_SCALE_DMIN = 0.25
SEED0 = 101


@dataclass(frozen=True)
class ToroCase:
    key: str
    filename: str
    title: str
    vs_title: str
    randomize_layer_thickness: bool
    randomize_bedrock_depth: bool


CASES = (
    ToroCase(
        "vs",
        "af_toro_campbell_vs.png",
        r"Toro $V_s$ only (fixed $H$, one soil layer)",
        r"Toro $V_s$ only",
        False,
        False,
    ),
    ToroCase(
        "vs_H",
        "af_toro_campbell_vs_H.png",
        r"Toro $V_s$ + depth to bedrock",
        r"Toro $V_s$ + $H$",
        False,
        True,
    ),
    ToroCase(
        "full",
        "af_toro_campbell_full.png",
        r"Full Toro (NHPP + interface + $V_s$)",
        r"Full Toro",
        True,
        True,
    ),
)


def travel_time_vs() -> float:
    h = np.array([10.59, 18.28, 11.12, 31.90, 16.41])
    vs = np.array([139.19, 174.73, 305.03, 254.69, 334.65])
    return float(np.sum(h) / np.sum(h / vs))


def xi_of_vs(vs: float, scale: float) -> float:
    return float(scale * compute_damping_from_Q(compute_quality_factor(float(vs))))


def vs_depth_to_layers(vs_depth: np.ndarray, n_soil: int, dz: float, xi_scale: float):
    vs = np.asarray(vs_depth, dtype=float).ravel()
    n_soil = int(np.clip(n_soil, 1, len(vs) - 1))
    soil = vs[:n_soil]
    rock_vs = float(np.median(vs[n_soil:]))
    layers: list[Layer] = []
    i = 0
    while i < n_soil:
        j = i + 1
        while j < n_soil and abs(soil[j] - soil[i]) <= 1e-6 * max(1.0, abs(soil[i])):
            j += 1
        v = float(soil[i])
        layers.append(Layer((j - i) * dz, v, RHO_SOIL, xi_of_vs(v, xi_scale)))
        i = j
    rock = RockHalfspace(rock_vs, RHO_ROCK, xi_of_vs(rock_vs, xi_scale))
    return layers, rock


def tf_profile(vs_depth, n_soil, dz, freq, xi_scale):
    layers, rock = vs_depth_to_layers(vs_depth, n_soil, dz, xi_scale)
    _, aw, _ = layered_transfer_function(freq, layers, rock)
    return aw


def ensemble_stats(soil_stack, sigma_target: float) -> dict:
    vs1 = np.array([float(np.asarray(row).ravel()[0]) for row in soil_stack])
    ln_vs1 = np.log(vs1)
    return {
        "sigma_ln_vs1": float(np.std(ln_vs1, ddof=1)),
        "cov_vs1": float(np.std(vs1, ddof=1) / np.mean(vs1)),
        "mean_vs1": float(np.mean(vs1)),
        "target": sigma_target,
    }


def tf_panel(ax, freq, af, med, p16, p84, af_base, title, color):
    for row in af:
        ax.loglog(freq, row, color="0.72", lw=0.55, alpha=0.7, zorder=1)
    ax.fill_between(freq, p16, p84, color=color, alpha=0.18, zorder=2, label="16–84%")
    ax.loglog(freq, med, color=color, lw=2.0, zorder=4, label="geomean")
    ax.loglog(freq, af_base, color="0.15", lw=1.4, ls="--", zorder=5, label="base (no rand.)")
    ax.set_xlim(0.2, 12)
    ax.set_ylim(0.5, 200)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"$|AF_\mathrm{within}|$")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=7.5, frameon=True)
    ax.annotate(
        fr"$A_1$ p50={np.median(np.max(af, axis=1)):.0f}",
        xy=(0.97, 0.97),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="0.8"),
    )


def run_case(case: ToroCase, vs_mean: float, freq: np.ndarray) -> None:
    cfg = ProfileRandomizationConfig(
        vs_mean=vs_mean,
        thickness=H_SOIL,
        dz=DZ,
        cov=COV,
        vs_bedrock=VS_ROCK,
        bedrock_thickness=H_ROCK,
        sigma_ln_vs=SIGMA_LN_VS,
        use_full_model=True,
        randomize_layer_thickness=case.randomize_layer_thickness,
        randomize_bedrock_depth=case.randomize_bedrock_depth,
        vary_bedrock_vs=False,
    )
    print(f"\n=== {case.key}: NHPP={case.randomize_layer_thickness}  H={case.randomize_bedrock_depth} ===")

    af_q = np.empty((N_REAL, len(freq)))
    af_d = np.empty((N_REAL, len(freq)))
    profiles = []
    soil_rows = []
    n_layers = []
    interfaces = []

    for k in range(N_REAL):
        rng = np.random.default_rng(SEED0 + k)
        prof = generate_toro_profile(cfg, rng)
        profiles.append(prof)
        n_soil = int(prof.n_soil_samples)
        soil_rows.append(prof.vs_depth[:n_soil].copy())
        n_layers.append(len({round(float(v), 3) for v in prof.vs_depth[:n_soil]}))
        interfaces.append(float(prof.interface_depth))
        af_q[k] = tf_profile(prof.vs_depth, n_soil, DZ, freq, XI_SCALE_Q)
        af_d[k] = tf_profile(prof.vs_depth, n_soil, DZ, freq, XI_SCALE_DMIN)

    stats = ensemble_stats(soil_rows, SIGMA_LN_VS)
    print(
        f"  σ_ln(Vs1)={stats['sigma_ln_vs1']:.3f}  CoV(Vs1)={stats['cov_vs1']:.3f}  "
        f"(target {SIGMA_LN_VS:.3f}, mean Vs1={stats['mean_vs1']:.1f})\n"
        f"  soil layers: median {np.median(n_layers):.0f}  range {min(n_layers)}–{max(n_layers)}\n"
        f"  interface H: median {np.median(interfaces):.1f} m  "
        f"range {min(interfaces):.1f}–{max(interfaces):.1f}"
    )
    peaks_q = np.max(af_q, axis=1)
    peaks_d = np.max(af_d, axis=1)
    print(
        f"  A1 ξ_Q     p50={np.median(peaks_q):.1f}  "
        f"p16–p84={np.percentile(peaks_q,16):.1f}–{np.percentile(peaks_q,84):.1f}\n"
        f"  A1 0.25ξ_Q p50={np.median(peaks_d):.1f}  "
        f"p16–p84={np.percentile(peaks_d,16):.1f}–{np.percentile(peaks_d,84):.1f}"
    )

    med_q = np.exp(np.mean(np.log(np.clip(af_q, 1e-12, None)), axis=0))
    med_d = np.exp(np.mean(np.log(np.clip(af_d, 1e-12, None)), axis=0))
    p16_q, p84_q = np.percentile(af_q, [16, 84], axis=0)
    p16_d, p84_d = np.percentile(af_d, [16, 84], axis=0)
    base = build_base_case_profile(cfg)
    n_soil_base = max(1, int(round(H_SOIL / DZ)))
    af_base_q = tf_profile(base, n_soil_base, DZ, freq, XI_SCALE_Q)
    af_base_d = tf_profile(base, n_soil_base, DZ, freq, XI_SCALE_DMIN)

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.3), constrained_layout=True)
    ax = axes[0]
    for prof in profiles:
        vs = prof.vs_depth
        z = (np.arange(len(vs)) + 0.5) * DZ
        ax.plot(vs, z, color="0.75", lw=0.7, alpha=0.85)
    z_b = (np.arange(len(base)) + 0.5) * DZ
    ax.plot(base, z_b, color="#0072B2", lw=2.0, label="median column")
    ax.set_ylim(H_SOIL + H_ROCK, 0.0)
    ax.set_xlim(0.0, 2800.0)
    ax.set_xlabel(r"$V_s$ (m/s)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(fr"{case.vs_title}  ($\sigma_{{\ln V_s}}$=CoV={SIGMA_LN_VS:.2f})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    tf_panel(axes[1], freq, af_q, med_q, p16_q, p84_q, af_base_q, r"Campbell $\xi_Q=1/(2Q)$", "#0072B2")
    tf_panel(
        axes[2],
        freq,
        af_d,
        med_d,
        p16_d,
        p84_d,
        af_base_d,
        r"$0.25\,\xi_Q$  (Fig. 6 $D_{\min}$)",
        "#D55E00",
    )
    fig.suptitle(
        fr"{case.title}  $V_{{s,\mathrm{{med}}}}$={vs_mean:.0f} m/s,  $n$={N_REAL}  |  "
        fr"ensemble $\sigma_{{\ln V_{{s1}}}}$={stats['sigma_ln_vs1']:.3f}  (target {SIGMA_LN_VS:.3f})",
        fontsize=11,
    )
    out = OUT / case.filename
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    vs_mean = travel_time_vs()
    freq = np.logspace(np.log10(0.2), np.log10(12.0), 700)
    print(f"vs_mean={vs_mean:.1f} m/s  n={N_REAL}  sigma_ln_vs=CoV={SIGMA_LN_VS:.3f}")
    for case in CASES:
        run_case(case, vs_mean, freq)


if __name__ == "__main__":
    main()
