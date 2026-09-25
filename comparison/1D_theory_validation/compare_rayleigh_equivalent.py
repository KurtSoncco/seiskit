"""OpenSees 1D Rayleigh vs closed-form TF with equivalent hysteretic ξ(f).

Hallal 1D uses ``damping_method="global_avg"``: Q(Vs) → ξ, then Rayleigh
(α, β) at ``(min(f0, motion_freq), 10 Hz)``. ``AnalysisConfig.damping_zeta``
(0.025) is **not** the soil damping that OpenSees applies.

Theory is evaluated directly from Thomson–Haskell (no Konno–Ohmachi).
OpenSees AF is the raw FAS ratio (also unsmoothed) so the overlay is fair.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from seiskit.analysis import run_opensees_analysis
from seiskit.builder import build_model_data
from seiskit.config import AnalysisConfig
from seiskit.damping import (
    compute_damping_from_Q,
    compute_quality_factor,
    compute_rayleigh_coefficients,
)
from seiskit.plot_config import apply_style
from seiskit.theory.layered_1d_tf import Layer, RockHalfspace, layered_transfer_function
from seiskit.ttf.acc2FAS2 import acc2FAS2

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "rayleigh_equivalent"

# Hallal-like 2-layer column (matches profile_randomization demo / RV 1D).
VS1 = 230.0
H = 15.0
VS2 = 1500.0
BEDROCK_THICKNESS = 10.0
RHO = 2000.0
DZ = 0.5
MOTION_FREQ = 3.0  # RV M1
CONFIG_ZETA = 0.025  # AnalysisConfig.damping_zeta (unused by global_avg)


def rayleigh_xi(freq: np.ndarray | float, alpha: float, beta: float) -> np.ndarray:
    """Equivalent viscous damping ξ(ω) = α/(2ω) + βω/2."""
    f = np.atleast_1d(np.asarray(freq, dtype=float))
    w = 2.0 * np.pi * np.clip(f, 1e-6, None)
    return alpha / (2.0 * w) + beta * w / 2.0


def ttf_unsmoothed(
    surface: np.ndarray,
    base: np.ndarray,
    dt: float,
    *,
    fmin: float = 0.1,
    fmax: float = 10.0,
    n_points: int = 800,
) -> tuple[np.ndarray, np.ndarray]:
    """|FAS_surface / FAS_base| with no Konno–Ohmachi smoothing."""
    fas_s, freq_raw = acc2FAS2(surface, dt)
    fas_b, _ = acc2FAS2(base, dt)
    freq = np.logspace(np.log10(fmin), np.log10(fmax), n_points)
    interp_s = interp1d(freq_raw, fas_s, bounds_error=False, fill_value=np.nan)
    interp_b = interp1d(freq_raw, fas_b, bounds_error=False, fill_value=np.nan)
    num = interp_s(freq)
    den = interp_b(freq)
    af = np.divide(num, den, out=np.full_like(num, np.nan), where=np.isfinite(den) & (den > 1e-20))
    return freq, af


def layered_tf_const_xi(
    freq: np.ndarray,
    *,
    xi_soil: float,
    xi_rock: float,
) -> np.ndarray:
    _, af_within, _ = layered_transfer_function(
        freq,
        [Layer(H, VS1, RHO, xi_soil)],
        RockHalfspace(VS2, RHO, xi_rock),
    )
    return np.asarray(af_within, dtype=float)


def layered_tf_rayleigh(
    freq: np.ndarray,
    *,
    alpha_s: float,
    beta_s: float,
    alpha_r: float,
    beta_r: float,
) -> np.ndarray:
    """Per-frequency hysteretic ξ matching Rayleigh (α, β) of soil and rock."""
    af = np.empty(len(freq), dtype=float)
    xi_s = rayleigh_xi(freq, alpha_s, beta_s)
    xi_r = rayleigh_xi(freq, alpha_r, beta_r)
    for i, (fi, zs, zr) in enumerate(zip(freq, xi_s, xi_r)):
        _, aw, _ = layered_transfer_function(
            fi,
            [Layer(H, VS1, RHO, float(zs))],
            RockHalfspace(VS2, RHO, float(zr)),
        )
        af[i] = float(aw[0])
    return af


def build_grids() -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray]:
    n_soil = max(1, int(round(H / DZ)))
    n_rock = max(1, int(round(BEDROCK_THICKNESS / DZ)))
    vs = np.array([VS1] * n_soil + [VS2] * n_rock, dtype=float).reshape(-1, 1)
    rho = np.full_like(vs, RHO)
    nu = np.full_like(vs, 0.3)
    mask = np.zeros_like(vs, dtype=bool)
    mask[n_soil:] = True
    lz = float(vs.shape[0] * DZ)
    interface_y = float(n_rock * DZ)
    return vs, rho, nu, lz, interface_y, mask


def _load_center_pair(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = sorted(run_dir.glob("center_node_y*_dof1_accel.txt"))
    if len(files) < 2:
        raise FileNotFoundError(f"Need ≥2 center recorders in {run_dir}, found {files}")

    def y_of(p: Path) -> float:
        return float(p.name.split("_dof")[0].replace("center_node_y", ""))

    files = sorted(files, key=y_of)
    base = np.loadtxt(files[0])
    surf = np.loadtxt(files[-1])
    return base[:, 0], surf[:, 1], base[:, 1]


def _rel_err(freq: np.ndarray, a: np.ndarray, b: np.ndarray, f_lo: float, f_hi: float) -> dict:
    mask = (freq >= f_lo) & (freq <= f_hi) & np.isfinite(a) & np.isfinite(b) & (b > 0.05)
    if not np.any(mask):
        return {"median": float("nan"), "p95": float("nan"), "n": 0}
    rel = np.abs(a[mask] - b[mask]) / b[mask]
    return {"median": float(np.median(rel)), "p95": float(np.percentile(rel, 95)), "n": int(mask.sum())}


def main() -> int:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)

    f0 = VS1 / (4.0 * H)
    f1 = min(f0, MOTION_FREQ)
    f2 = 10.0
    xi_soil_q = compute_damping_from_Q(compute_quality_factor(VS1))
    xi_rock_q = compute_damping_from_Q(compute_quality_factor(VS2))
    alpha_s, beta_s = compute_rayleigh_coefficients(xi_soil_q, f1, f2)
    alpha_r, beta_r = compute_rayleigh_coefficients(xi_rock_q, f1, f2)

    vs, rho, nu, lz, interface_y, mask = build_grids()
    config = AnalysisConfig(
        Ly=lz,
        Lx=DZ,
        hx=DZ,
        dt=0.001,
        duration=30.0,
        motion_freq=MOTION_FREQ,
        motion_t_shift=0.5,
        damping_freqs=(f1, f2),
        damping_zeta=CONFIG_ZETA,
        damping_method="global_avg",
        boundary_condition_type="1D",
        record_center_nodes=True,
        center_node_y_positions=[interface_y, lz],
        record_all_surface_nodes=False,
        element_type="4node",
        solver_type="UmfPack",
    )
    model = build_model_data(config, vs, rho, nu, bedrock_mask=mask)
    run_id = "hallal_2layer_global_avg"
    print(run_opensees_analysis(config, model, run_id=run_id, output_dir=str(OUT)))

    t, surf, base = _load_center_pair(OUT / run_id)
    dt = float(t[1] - t[0]) if len(t) > 1 else config.dt
    freq, af_os = ttf_unsmoothed(surf, base, dt, fmax=10.0)

    af_th_025 = layered_tf_const_xi(freq, xi_soil=CONFIG_ZETA, xi_rock=0.0)
    af_th_q = layered_tf_const_xi(freq, xi_soil=xi_soil_q, xi_rock=xi_rock_q)
    af_th_ray = layered_tf_rayleigh(
        freq, alpha_s=alpha_s, beta_s=beta_s, alpha_r=alpha_r, beta_r=beta_r
    )

    f_lo, f_hi = 0.5, 8.0
    metrics = {
        "f0": f0,
        "f1": f1,
        "f2": f2,
        "config_zeta": CONFIG_ZETA,
        "xi_soil_Q": xi_soil_q,
        "xi_rock_Q": xi_rock_q,
        "alpha_soil": alpha_s,
        "beta_soil": beta_s,
        "xi_rayleigh_soil_at": {
            "f1": float(rayleigh_xi(f1, alpha_s, beta_s)[0]),
            "f0": float(rayleigh_xi(f0, alpha_s, beta_s)[0]),
            "f2": float(rayleigh_xi(f2, alpha_s, beta_s)[0]),
        },
        "rel_err_vs_rayleigh_eq": _rel_err(freq, af_os, af_th_ray, f_lo, f_hi),
        "rel_err_vs_const_Q": _rel_err(freq, af_os, af_th_q, f_lo, f_hi),
        "rel_err_vs_const_0p025": _rel_err(freq, af_os, af_th_025, f_lo, f_hi),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    i_os = int(np.nanargmax(af_os))
    peak_os = float(af_os[i_os])
    peak_f = float(freq[i_os])
    peak_ray = float(np.nanmax(af_th_ray))
    peak_q = float(np.nanmax(af_th_q))
    peak_025 = float(np.nanmax(af_th_025))
    metrics["peak_AF"] = {
        "f_peak_Hz": peak_f,
        "OpenSees": peak_os,
        "theory_rayleigh_eq": peak_ray,
        "theory_const_Q": peak_q,
        "theory_const_0p025": peak_025,
        "rel_peak_rayleigh": (peak_ray - peak_os) / peak_os,
        "rel_peak_const_Q": (peak_q - peak_os) / peak_os,
        "rel_peak_0p025": (peak_025 - peak_os) / peak_os,
    }

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), constrained_layout=True)

    ax = axes[0]
    ax.semilogx(freq, af_os, color="0.45", lw=1.15, label="OpenSees 1D (raw FAS ratio)")
    ax.semilogx(freq, af_th_ray, color="#0072B2", lw=2.0, label=r"Theory Rayleigh $\xi(f)$")
    ax.semilogx(
        freq,
        af_th_q,
        color="#009E73",
        lw=1.5,
        ls="--",
        label=fr"Theory const. $\xi$={xi_soil_q:.3f}",
    )
    ax.semilogx(
        freq,
        af_th_025,
        color="#D55E00",
        lw=1.5,
        ls=":",
        label=r"Theory const. $\xi$=0.025",
    )
    ax.axvline(f0, color="0.4", ls="--", lw=0.8)
    ax.set_xlim(0.2, 10)
    ax.set_ylim(0, 28)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"$|AF_\mathrm{within}|$")
    ax.set_title("2-layer TF (no Konno–Ohmachi)")
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    ax.grid(True, which="both", alpha=0.3)
    ax.annotate(
        fr"peaks @ {peak_f:.2f} Hz" + "\n"
        + fr"OS {peak_os:.1f}  Rayl. {peak_ray:.1f}" + "\n"
        + fr"Q-const {peak_q:.1f}  0.025 {peak_025:.1f}",
        xy=(0.98, 0.97),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.8"),
    )

    ax = axes[1]
    f_xi = np.logspace(np.log10(0.2), 1, 400)
    ax.semilogx(
        f_xi,
        100 * rayleigh_xi(f_xi, alpha_s, beta_s),
        color="#0072B2",
        lw=2.0,
        label="Soil Rayleigh",
    )
    ax.semilogx(
        f_xi,
        100 * rayleigh_xi(f_xi, alpha_r, beta_r),
        color="#56B4E9",
        lw=1.6,
        label="Rock Rayleigh",
    )
    ax.axhline(100 * xi_soil_q, color="#009E73", ls="--", lw=1.4, label=fr"Soil Q ξ={100*xi_soil_q:.2f}%")
    ax.axhline(100 * CONFIG_ZETA, color="#D55E00", ls=":", lw=1.6, label="Config ζ=2.5%")
    ax.axvline(f1, color="0.4", ls="--", lw=0.8)
    ax.axvline(f2, color="0.4", ls=":", lw=0.8)
    ax.set_xlim(0.2, 10)
    ax.set_ylim(0, 8)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Damping ratio (%)")
    ax.set_title(fr"Rayleigh at $f_1$={f1:.2f} Hz, $f_2$={f2:.0f} Hz")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    ax.grid(True, which="both", alpha=0.3)

    fig.suptitle(
        fr"$V_{{s1}}$={VS1:.0f} m/s, $H$={H:.0f} m, $V_{{s2}}$={VS2:.0f} m/s, "
        fr"$f_0$={f0:.2f} Hz  |  OpenSees global_avg",
        fontsize=10,
    )
    out_png = OUT / "af_rayleigh_equivalent.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    np.savez(
        OUT / "af_data.npz",
        freq=freq,
        af_os=af_os,
        af_th_rayleigh=af_th_ray,
        af_th_const_Q=af_th_q,
        af_th_const_0p025=af_th_025,
    )

    print(json.dumps(metrics, indent=2))
    print(f"Wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
