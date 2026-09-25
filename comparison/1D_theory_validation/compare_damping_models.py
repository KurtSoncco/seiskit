"""Are OpenSees Rayleigh and the theoretical TF using the same damping?

OpenSees: viscous C = αM + βK, with (α, β) from Q(Vs) at (f1, f2).
Previous theory: hysteretic Vs* = Vs sqrt(1+2i ξ(f)), ξ(f)=α/(2ω)+βω/2.
This script also builds the exact frequency-domain Rayleigh (complex G and ρ).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from seiskit.damping import compute_rayleigh_coefficients
from seiskit.plot_config import apply_style
from seiskit.theory.layered_1d_tf import Layer, RockHalfspace, layered_transfer_function

OUT = Path(__file__).resolve().parent / "results" / "rayleigh_equivalent"
RHO = 2000.0
MOTION_FREQ = 3.0
F2 = 10.0


def _sqrtc(z: complex) -> complex:
    s = np.sqrt(z)
    return s if s.real >= 0.0 else -s


def rayleigh_complex_vs_rho(vs: float, rho: float, omega: float, alpha: float, beta: float):
    """Exact 1D Rayleigh: G* = G(1+iωβ), ρ* = ρ(1−iα/ω)."""
    g = rho * vs * vs
    rho_c = rho * (1.0 - 1j * alpha / omega)
    g_c = g * (1.0 + 1j * omega * beta)
    vs_c = _sqrtc(g_c / rho_c)
    return vs_c, rho_c


def hysteretic_tf(freq, vs1, H, vs2, alpha_s, beta_s, alpha_r, beta_r):
    af = np.empty(len(freq))
    for i, f in enumerate(freq):
        w = 2.0 * np.pi * float(f)
        xi_s = alpha_s / (2.0 * w) + beta_s * w / 2.0
        xi_r = alpha_r / (2.0 * w) + beta_r * w / 2.0
        _, aw, _ = layered_transfer_function(
            f,
            [Layer(H, vs1, RHO, float(xi_s))],
            RockHalfspace(vs2, RHO, float(xi_r)),
        )
        af[i] = float(aw[0])
    return af


def viscous_rayleigh_tf(freq, vs1, H, vs2, alpha_s, beta_s, alpha_r, beta_r):
    """Thomson–Haskell with exact Rayleigh complex G, ρ (not hysteretic ξ)."""
    af = np.empty(len(freq))
    for i, f in enumerate(freq):
        w = 2.0 * np.pi * float(f)
        if w == 0.0:
            af[i] = 1.0
            continue
        vs_s, rho_s = rayleigh_complex_vs_rho(vs1, RHO, w, alpha_s, beta_s)
        k = w / vs_s
        kh = k * H
        c, s = np.cos(kh), np.sin(kh)
        gk = 1j * w * rho_s * vs_s
        u_base = c  # u_surf=1, τ_surf=0 → u_base = cos
        tau_base = -gk * s
        af[i] = float(np.abs(1.0 / u_base))
    return af


def rel_err(a, b, mask):
    r = np.abs(a[mask] - b[mask]) / np.clip(b[mask], 1e-12, None)
    return float(np.median(r)), float(np.percentile(r, 95))


def main() -> None:
    apply_style()
    rows = json.loads((OUT / "profiles_metrics.json").read_text())
    summary = []

    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.6), sharex=True)
    fig.subplots_adjust(top=0.90, hspace=0.42, wspace=0.22)

    for ax, row in zip(axes.ravel(), rows):
        d = np.load(row["npz"])
        freq = np.asarray(d["freq"], dtype=float)
        vs1, H, vs2 = float(d["vs1"]), float(d["H"]), float(d["vs2"])
        f0 = vs1 / (4.0 * H)
        f1 = min(f0, MOTION_FREQ)
        a_s, b_s = compute_rayleigh_coefficients(row["xi_soil_Q"], f1, F2)
        a_r, b_r = compute_rayleigh_coefficients(row["xi_rock_Q"], f1, F2)

        af_os = np.asarray(d["af_os"], dtype=float)
        af_hyst = hysteretic_tf(freq, vs1, H, vs2, a_s, b_s, a_r, b_r)
        af_visc = viscous_rayleigh_tf(freq, vs1, H, vs2, a_s, b_s, a_r, b_r)

        band = (freq >= 0.5 * f0) & (freq <= min(8.0, 3.0 * f0)) & np.isfinite(af_os)
        med_h, p95_h = rel_err(af_os, af_hyst, band)
        med_v, p95_v = rel_err(af_os, af_visc, band)
        med_hv, p95_hv = rel_err(af_hyst, af_visc, band)
        i_os = int(np.nanargmax(af_os))
        summary.append(
            {
                "name": row["name"],
                "f1": f1,
                "alpha_soil": a_s,
                "beta_soil": b_s,
                "peak_os": float(af_os[i_os]),
                "peak_hysteretic": float(np.nanmax(af_hyst)),
                "peak_viscous": float(np.nanmax(af_visc)),
                "rel_peak_hyst": float(np.nanmax(af_hyst) / af_os[i_os] - 1.0),
                "rel_peak_visc": float(np.nanmax(af_visc) / af_os[i_os] - 1.0),
                "rel_peak_hyst_vs_visc": float(np.nanmax(af_hyst) / np.nanmax(af_visc) - 1.0),
                "median_rel_os_hyst": med_h,
                "median_rel_os_visc": med_v,
                "median_rel_hyst_visc": med_hv,
                "p95_rel_os_hyst": p95_h,
                "p95_rel_os_visc": p95_v,
            }
        )

        ax.loglog(freq, af_os, color="0.45", lw=1.4, label="OpenSees Rayleigh")
        ax.loglog(freq, af_visc, color="#0072B2", lw=2.0, label="Theory viscous Rayleigh")
        ax.loglog(freq, af_hyst, color="#D55E00", lw=1.4, ls="--", label=r"Theory hysteretic $\xi(f)$")
        ax.axvline(f0, color="0.4", ls="--", lw=0.7)
        ax.set_xlim(0.1, 10)
        ax.set_ylim(0.5, 40)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(
            fr"{row['label']}: $f_0$={f0:.2f} Hz, $\xi_Q$={100*row['xi_soil_Q']:.2f}%"
            + "\n"
            + fr"$\Delta A$ visc={100*summary[-1]['rel_peak_visc']:+.1f}%, "
            + fr"hyst={100*summary[-1]['rel_peak_hyst']:+.1f}%",
            fontsize=9,
        )
        if ax is axes[0, 0]:
            ax.legend(loc="center right", fontsize=7.5, frameon=True)

    for ax in axes[-1]:
        ax.set_xlabel("Frequency (Hz)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$|AF_\mathrm{within}|$")
    fig.suptitle(
        "OpenSees Rayleigh (αM+βK) vs frequency-domain viscous Rayleigh vs hysteretic ξ(f)",
        fontsize=11,
    )
    out_png = OUT / "af_damping_model_check.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Detail: demo relative error.
    demo = next(s for s in summary if s["name"] == "demo_230_15")
    d = np.load(OUT / "demo_230_15.npz")
    freq = np.asarray(d["freq"], dtype=float)
    vs1, H, vs2 = float(d["vs1"]), float(d["H"]), float(d["vs2"])
    f0 = vs1 / (4.0 * H)
    f1 = min(f0, MOTION_FREQ)
    a_s, b_s = compute_rayleigh_coefficients(float(d["xi_soil_Q"]), f1, F2)
    a_r, b_r = compute_rayleigh_coefficients(float(d["xi_rock_Q"]), f1, F2)
    af_os = np.asarray(d["af_os"], dtype=float)
    af_hyst = hysteretic_tf(freq, vs1, H, vs2, a_s, b_s, a_r, b_r)
    af_visc = viscous_rayleigh_tf(freq, vs1, H, vs2, a_s, b_s, a_r, b_r)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2), constrained_layout=True)
    ax = axes[0]
    ax.loglog(freq, af_os, color="0.45", lw=1.5, label="OpenSees")
    ax.loglog(freq, af_visc, color="#0072B2", lw=2.0, label="Theory viscous Rayleigh")
    ax.loglog(freq, af_hyst, color="#D55E00", lw=1.5, ls="--", label=r"Theory hysteretic $\xi(f)$")
    ax.axvline(f0, color="0.4", ls="--", lw=0.8)
    ax.set_xlim(1.0, 8.0)
    ax.set_ylim(0.5, 30)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"$|AF_\mathrm{within}|$")
    ax.set_title("Demo column, zoom on f0")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    err_v = 100 * (af_os - af_visc) / np.clip(af_visc, 1e-12, None)
    err_h = 100 * (af_os - af_hyst) / np.clip(af_hyst, 1e-12, None)
    err_m = 100 * (af_hyst - af_visc) / np.clip(af_visc, 1e-12, None)
    ax.semilogx(freq, err_v, color="#0072B2", lw=1.6, label="OpenSees − viscous")
    ax.semilogx(freq, err_h, color="#D55E00", lw=1.4, ls="--", label="OpenSees − hysteretic")
    ax.semilogx(freq, err_m, color="0.35", lw=1.2, ls=":", label="hysteretic − viscous")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.axvline(f0, color="0.4", ls="--", lw=0.8)
    ax.set_xlim(1.0, 8.0)
    ax.set_ylim(-8, 8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Relative difference (%)")
    ax.set_title("Same (α, β); models are not identical")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    out_zoom = OUT / "af_damping_model_zoom.png"
    fig.savefig(out_zoom, dpi=160)
    plt.close(fig)

    (OUT / "damping_model_check.json").write_text(json.dumps({"demo": demo, "all": summary}, indent=2))
    print(json.dumps(summary, indent=2))
    print(out_png)
    print(out_zoom)


if __name__ == "__main__":
    main()
