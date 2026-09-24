"""Fit hysteretic ξ (no Taborda–Bielak Q) so the 1D TF matches the Hallal digitized curve."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import find_peaks

from seiskit.plot_config import apply_style
from seiskit.theory.layered_1d_tf import Layer, RockHalfspace, layered_transfer_function

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "rayleigh_equivalent"
XI_ROCK = 0.002  # small-strain rock, not Q(Vs)


def load_extracted():
    path = ROOT / "plot_extracted_profile.py"
    spec = importlib.util.spec_from_file_location("extracted_profile", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["extracted_profile"] = mod
    spec.loader.exec_module(mod)
    return mod


def soil_rows(mod):
    return mod.build_column()[2][:-1]


def af_hysteretic(freq, rows, xi_of_f) -> np.ndarray:
    """Thomson–Haskell with hysteretic G* = ρ Vs² (1+2iξ), ξ may depend on f."""
    freq = np.atleast_1d(np.asarray(freq, dtype=float))
    af = np.empty(freq.shape, dtype=float)
    rock = RockHalfspace(2298.12, 2200.0, XI_ROCK)
    for i, f in enumerate(freq):
        xi = float(xi_of_f(float(f)))
        xi = float(np.clip(xi, 1e-4, 0.2))
        layers = [Layer(h, vs, rho, xi) for _, _, _, h, vs, rho, _ in rows]
        _, aw, _ = layered_transfer_function(f, layers, rock)
        af[i] = float(aw[0])
    return af


def log_misfit(af, a_obs) -> float:
    return float(np.mean((np.log(np.clip(af, 1e-12, None)) - np.log(np.clip(a_obs, 1e-12, None))) ** 2))


def hallal_peaks(f, a, fmin=0.55, fmax=10.0):
    m = (f >= fmin) & (f <= fmax)
    idx, props = find_peaks(a[m], prominence=1.2, distance=8)
    return [(float(f[m][i]), float(a[m][i])) for i in idx]


def invert_xi_at_peak(rows, f_peak, a_peak) -> float:
    band = np.logspace(np.log10(0.93 * f_peak), np.log10(1.07 * f_peak), 80)

    def err(log_xi):
        xi = float(np.exp(log_xi))
        af = af_hysteretic(band, rows, lambda _f, z=xi: z)
        return (float(np.max(af)) - a_peak) ** 2

    res = minimize_scalar(err, bounds=(np.log(0.001), np.log(0.08)), method="bounded")
    return float(np.exp(res.x))


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    mod = load_extracted()
    rows = soil_rows(mod)

    raw = np.loadtxt(mod.HALLAL_RAW, delimiter=",", skiprows=1)
    f_h, a_h = mod.clean_digitized_tf(raw[:, 0], raw[:, 1])
    peaks = hallal_peaks(f_h, a_h)
    print("Hallal peaks:")
    for f, a in peaks:
        print(f"  {f:.3f} Hz   A={a:.2f}")

    # --- constant hysteretic ξ (same in every soil layer) ---
    def misfit_const(log_xi):
        xi = float(np.exp(log_xi))
        af = af_hysteretic(f_h, rows, lambda _f, z=xi: z)
        return log_misfit(af, a_h)

    res_c = minimize_scalar(misfit_const, bounds=(np.log(0.002), np.log(0.04)), method="bounded")
    xi_c = float(np.exp(res_c.x))
    af_c = af_hysteretic(f_h, rows, lambda _f, z=xi_c: z)

    # --- ξ(f) = ξ1 (f / 1 Hz)^{-α}  (still hysteretic, mild f-dependence) ---
    def misfit_pow(p):
        xi1, alpha = p
        af = af_hysteretic(f_h, rows, lambda f, z=xi1, a=alpha: z * (f / 1.0) ** (-a))
        return log_misfit(af, a_h)

    res_p = minimize(
        misfit_pow,
        x0=np.array([0.01, 0.15]),
        bounds=[(0.002, 0.05), (-0.3, 1.0)],
        method="L-BFGS-B",
    )
    xi1, alpha = (float(res_p.x[0]), float(res_p.x[1]))
    af_p = af_hysteretic(f_h, rows, lambda f, z=xi1, a=alpha: z * (f / 1.0) ** (-a))

    # --- invert one hysteretic ξ at each Hallal peak, then interpolate ---
    xi_peaks = []
    for fp, ap in peaks:
        z = invert_xi_at_peak(rows, fp, ap)
        xi_peaks.append(z)
        print(f"peak {fp:.3f} Hz  Hallal={ap:.1f}  →  ξ={100*z:.3f}%")
    f_pk = np.array([p[0] for p in peaks])
    z_pk = np.array(xi_peaks)

    def xi_interp(f):
        return float(np.exp(np.interp(np.log(f), np.log(f_pk), np.log(z_pk))))

    af_i = af_hysteretic(f_h, rows, xi_interp)

    freq_plot = np.logspace(np.log10(0.45), np.log10(11.0), 1600)
    af_c_p = af_hysteretic(freq_plot, rows, lambda _f, z=xi_c: z)
    af_p_p = af_hysteretic(freq_plot, rows, lambda f, z=xi1, a=alpha: z * (f / 1.0) ** (-a))
    af_i_p = af_hysteretic(freq_plot, rows, xi_interp)

    def peak_table(name, freq, af):
        print(f"\n{name}")
        print(f"{'f_H':>7} {'Hallal':>8} {'model':>8} {'ratio':>7}")
        for fp, ap in peaks:
            ib = np.argmin(np.abs(freq - fp))
            # local max in ±8%
            m = (freq > 0.92 * fp) & (freq < 1.08 * fp)
            am = float(np.max(af[m])) if np.any(m) else float(af[ib])
            print(f"{fp:7.2f} {ap:8.1f} {am:8.1f} {ap/am:7.2f}")

    print(f"\nbest constant ξ = {100*xi_c:.3f}%   log-MSE={log_misfit(af_c, a_h):.4f}")
    print(f"power-law ξ(f) = {100*xi1:.3f}% × f^{{-{alpha:.3f}}}   log-MSE={log_misfit(af_p, a_h):.4f}")
    print(f"peak-wise ξ(f) interpolated           log-MSE={log_misfit(af_i, a_h):.4f}")
    peak_table("constant ξ", f_h, af_c)
    peak_table("power-law ξ(f)", f_h, af_p)
    peak_table("peak-wise ξ(f)", f_h, af_i)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), constrained_layout=True)

    ax = axes[0]
    ax.plot(raw[:, 0], raw[:, 1], ".", color="0.78", ms=2.4, alpha=0.45, zorder=1)
    ax.loglog(f_h, a_h, color="0.12", lw=2.3, zorder=3, label="Hallal cleaned")
    ax.loglog(
        freq_plot,
        af_c_p,
        color="#0072B2",
        lw=1.5,
        zorder=4,
        label=fr"hysteretic $\xi$={100*xi_c:.2f}% const.",
    )
    ax.loglog(
        freq_plot,
        af_p_p,
        color="#D55E00",
        lw=1.7,
        ls="--",
        zorder=5,
        label=fr"hysteretic $\xi(f)={100*xi1:.2f}\,f^{{-{alpha:.2f}}}$%",
    )
    ax.loglog(
        freq_plot,
        af_i_p,
        color="#009E73",
        lw=1.6,
        ls=":",
        zorder=5,
        label=r"hysteretic $\xi(f)$ peak-wise",
    )
    ax.set_xlim(0.45, 11)
    ax.set_ylim(0.8, 160)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"$|AF_\mathrm{within}|$")
    ax.set_title("Hysteretic $G^*=\\rho V_s^2(1+2i\\xi)$  (no Taborda–Bielak $Q$)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=7.5, frameon=True)

    ax = axes[1]
    ax.semilogx(freq_plot, 100 * np.array([xi_c] * len(freq_plot)), color="#0072B2", lw=1.5, label="constant")
    ax.semilogx(
        freq_plot,
        100 * xi1 * (freq_plot / 1.0) ** (-alpha),
        color="#D55E00",
        lw=1.7,
        ls="--",
        label="power law",
    )
    ax.semilogx(
        freq_plot,
        100 * np.array([xi_interp(f) for f in freq_plot]),
        color="#009E73",
        lw=1.6,
        ls=":",
        label="peak-wise",
    )
    ax.plot(f_pk, 100 * z_pk, "o", color="#009E73", ms=6, zorder=5)
    ax.set_xlim(0.45, 11)
    ax.set_ylim(0.0, 2.2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"Hysteretic $\xi$ (%)")
    ax.set_title("Damping used in the soil stack")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    out = OUT / "af_hysteretic_fit.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
