"""Check the hysteretic-with-ξ_R(ω) idea against algebra, this codebase, and exact Rayleigh."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from seiskit.plot_config import apply_style
from seiskit.theory.layered_1d_tf import Layer, RockHalfspace, layered_transfer_function

OUT = Path(__file__).resolve().parent / "results" / "rayleigh_equivalent"
RHO = 2000.0
XI_TARGET = 0.025
# Sobol 44: several modes inside 0.1–10 Hz so high-frequency overdamping is visible.
VS1, H, VS2 = 228.3, 56.3, 1107.0
F0 = VS1 / (4.0 * H)
F_MOTION = 3.0


def alpha_beta(xi: float, f1: float, f2: float) -> tuple[float, float]:
    w1, w2 = 2.0 * np.pi * f1, 2.0 * np.pi * f2
    return 2.0 * xi * w1 * w2 / (w1 + w2), 2.0 * xi / (w1 + w2)


def xi_r(freq: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    w = 2.0 * np.pi * np.clip(np.asarray(freq, dtype=float), 1e-9, None)
    return alpha / (2.0 * w) + beta * w / 2.0


def tf_hysteretic(freq: np.ndarray, xi) -> np.ndarray:
    xi = np.broadcast_to(np.asarray(xi, dtype=float), np.shape(freq))
    af = np.empty(len(freq))
    for i, (f, z) in enumerate(zip(freq, xi)):
        _, aw, _ = layered_transfer_function(
            f,
            [Layer(H, VS1, RHO, float(z))],
            RockHalfspace(VS2, RHO, 0.0),
        )
        af[i] = float(aw[0])
    return af


def tf_viscous_rayleigh(freq: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    af = np.empty(len(freq))
    g = RHO * VS1 * VS1
    for i, f in enumerate(freq):
        w = 2.0 * np.pi * float(f)
        rho_c = RHO * (1.0 - 1j * alpha / w)
        g_c = g * (1.0 + 1j * w * beta)
        vs_c = np.sqrt(g_c / rho_c)
        if vs_c.real < 0:
            vs_c = -vs_c
        kh = (w / vs_c) * H
        af[i] = float(np.abs(1.0 / np.cos(kh)))
    return af


def main() -> None:
    apply_style()
    freq = np.logspace(np.log10(0.1), 1.0, 1200)
    f1_code, f2_code = min(F0, F_MOTION), 10.0
    f1_idea, f2_idea = F0, F_MOTION  # as written: f0 and motion frequency

    a_code, b_code = alpha_beta(XI_TARGET, f1_code, f2_code)
    a_idea, b_idea = alpha_beta(XI_TARGET, f1_idea, f2_idea)

    af_const = tf_hysteretic(freq, XI_TARGET)
    af_hyst_code = tf_hysteretic(freq, xi_r(freq, a_code, b_code))
    af_hyst_idea = tf_hysteretic(freq, xi_r(freq, a_idea, b_idea))
    af_visc_code = tf_viscous_rayleigh(freq, a_code, b_code)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.15), constrained_layout=True)

    ax = axes[0]
    ax.semilogx(freq, 100 * xi_r(freq, a_code, b_code), color="#0072B2", lw=2.0, label=fr"code: $f_1$={f1_code:.2f}, $f_2$=10 Hz")
    ax.semilogx(freq, 100 * xi_r(freq, a_idea, b_idea), color="#D55E00", lw=1.7, ls="--", label=fr"idea: $f_1$=$f_0$={f1_idea:.2f}, $f_2$=3 Hz")
    ax.axhline(100 * XI_TARGET, color="0.35", ls=":", lw=1.3, label=r"target $\xi$=2.5%")
    ax.axvline(f1_code, color="#0072B2", ls=":", lw=0.8)
    ax.axvline(f2_code, color="#0072B2", ls=":", lw=0.8)
    ax.axvline(f1_idea, color="#D55E00", ls=":", lw=0.8)
    ax.axvline(f2_idea, color="#D55E00", ls=":", lw=0.8)
    ax.set_xlim(0.1, 10)
    ax.set_ylim(0, 8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"$\xi_R(f)$ (%)")
    ax.set_title("U-curve depends on the two anchors")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.loglog(freq, af_const, color="0.35", lw=1.6, label=r"hysteretic const. $\xi$=2.5%")
    ax.loglog(freq, af_hyst_code, color="#0072B2", lw=2.0, label=r"$G^*=\rho V_s^2(1+2i\xi_R)$, anchors  $f_1$, 10 Hz")
    ax.loglog(freq, af_hyst_idea, color="#D55E00", lw=1.5, ls="--", label=r"$G^*=\rho V_s^2(1+2i\xi_R)$, anchors $f_0$, 3 Hz")
    ax.set_xlim(0.1, 10)
    ax.set_ylim(0.5, 40)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"$|AF_\mathrm{within}|$")
    ax.set_title(fr"Sobol 44  $f_0$={F0:.2f} Hz: high-mode suppression")
    ax.legend(fontsize=7.2, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[2]
    ax.loglog(freq, af_hyst_code, color="#0072B2", lw=2.0, label="hysteretic $\\xi_R(\\omega)$")
    ax.loglog(freq, af_visc_code, color="#009E73", lw=1.6, ls="--", label=r"exact viscous $G(1+i\omega\beta)$")
    ax.set_xlim(0.1, 10)
    ax.set_ylim(0.5, 40)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"$|AF_\mathrm{within}|$")
    ax.set_title("Substitution ≈ viscous Rayleigh")
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        r"Idea check: replace constant $\xi$ by $\xi_R(\omega)$ in $G^*$ "
        fr"($V_{{s1}}$={VS1:.0f} m/s, $H$={H:.0f} m)",
        fontsize=11,
    )
    out = OUT / "idea_check_xiR_in_Gstar.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)

    def peak_near(af, f_mode, frac=0.15):
        m = (freq > f_mode * (1 - frac)) & (freq < f_mode * (1 + frac))
        return float(np.max(af[m])) if np.any(m) else float("nan")

    modes = [(2 * n - 1) * F0 for n in range(1, 6)]
    print(f"f0={F0:.3f}  code anchors=({f1_code:.3f}, {f2_code:.1f})  idea anchors=({f1_idea:.3f}, {f2_idea:.1f})")
    print(f"{'mode':>8} {'f':>7} {'const':>8} {'xiR_code':>10} {'xiR_idea':>10} {'xi_code%':>9} {'xi_idea%':>9}")
    for i, fm in enumerate(modes, 1):
        if fm > 10:
            break
        xc = float(xi_r(np.array([fm]), a_code, b_code)[0])
        xi = float(xi_r(np.array([fm]), a_idea, b_idea)[0])
        print(
            f"{i:8d} {fm:7.2f} {peak_near(af_const, fm):8.2f} "
            f"{peak_near(af_hyst_code, fm):10.2f} {peak_near(af_hyst_idea, fm):10.2f} "
            f"{100*xc:9.2f} {100*xi:9.2f}"
        )
    print(out)


if __name__ == "__main__":
    main()
