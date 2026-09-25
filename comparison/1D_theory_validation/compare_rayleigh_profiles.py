"""OpenSees 1D Rayleigh vs unsmoothed theory for several 2-layer Vs profiles."""

from __future__ import annotations

import json
from dataclasses import dataclass
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
    compute_average_damping_harmonic,
    compute_damping_from_Q,
    compute_quality_factor,
    compute_rayleigh_coefficients,
)
from seiskit.plot_config import apply_style
from seiskit.theory.layered_1d_tf import Layer, RockHalfspace, layered_transfer_function
from seiskit.ttf.acc2FAS2 import acc2FAS2

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "rayleigh_equivalent"
RHO = 2000.0
DZ = 0.5
BEDROCK_THICKNESS = 10.0
MOTION_FREQ = 3.0
CONFIG_ZETA = 0.025
DT = 0.002


@dataclass(frozen=True)
class ProfileCase:
    name: str
    vs1: float
    H: float
    vs2: float
    label: str


# Spread of RV 2-layer columns: shallow-stiff to deep-soft, plus the demo case.
CASES: tuple[ProfileCase, ...] = (
    ProfileCase("demo_230_15", 230.0, 15.0, 1500.0, "demo"),
    ProfileCase("s37", 267.7, 15.7, 1274.0, "Sobol 37"),
    ProfileCase("s19", 143.4, 21.8, 1027.0, "Sobol 19"),
    ProfileCase("s44", 228.3, 56.3, 1107.0, "Sobol 44"),
    ProfileCase("s10", 315.8, 82.7, 958.0, "Sobol 10"),
    ProfileCase("s36", 124.2, 99.0, 1035.0, "Sobol 36"),
)


def rayleigh_xi(freq: np.ndarray | float, alpha: float, beta: float) -> np.ndarray:
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
    n_points: int = 600,
) -> tuple[np.ndarray, np.ndarray]:
    fas_s, freq_raw = acc2FAS2(surface, dt)
    fas_b, _ = acc2FAS2(base, dt)
    freq = np.logspace(np.log10(fmin), np.log10(fmax), n_points)
    interp_s = interp1d(freq_raw, fas_s, bounds_error=False, fill_value=np.nan)
    interp_b = interp1d(freq_raw, fas_b, bounds_error=False, fill_value=np.nan)
    num = interp_s(freq)
    den = interp_b(freq)
    af = np.divide(num, den, out=np.full_like(num, np.nan), where=np.isfinite(den) & (den > 1e-20))
    return freq, af


def _load_center_pair(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = sorted(run_dir.glob("center_node_y*_dof1_accel.txt"))
    if len(files) < 2:
        raise FileNotFoundError(f"Need ≥2 center recorders in {run_dir}")

    def y_of(p: Path) -> float:
        return float(p.name.split("_dof")[0].replace("center_node_y", ""))

    files = sorted(files, key=y_of)
    base = np.loadtxt(files[0])
    surf = np.loadtxt(files[-1])
    return base[:, 0], surf[:, 1], base[:, 1]


def build_grids(case: ProfileCase) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray]:
    n_soil = max(1, int(round(case.H / DZ)))
    n_rock = max(1, int(round(BEDROCK_THICKNESS / DZ)))
    vs = np.array([case.vs1] * n_soil + [case.vs2] * n_rock, dtype=float).reshape(-1, 1)
    rho = np.full_like(vs, RHO)
    nu = np.full_like(vs, 0.3)
    mask = np.zeros_like(vs, dtype=bool)
    mask[n_soil:] = True
    lz = float(vs.shape[0] * DZ)
    interface_y = float(n_rock * DZ)
    return vs, rho, nu, lz, interface_y, mask


def soil_rock_xi(vs: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    soil = vs[~mask]
    rock = vs[mask]
    xi_s = compute_average_damping_harmonic([compute_quality_factor(float(v)) for v in soil.ravel()])
    xi_r = compute_damping_from_Q(compute_quality_factor(float(np.median(rock))))
    return float(xi_s), float(xi_r)


def theory_const(freq: np.ndarray, case: ProfileCase, xi_s: float, xi_r: float) -> np.ndarray:
    _, af, _ = layered_transfer_function(
        freq,
        [Layer(case.H, case.vs1, RHO, xi_s)],
        RockHalfspace(case.vs2, RHO, xi_r),
    )
    return np.asarray(af, dtype=float)


def theory_rayleigh(
    freq: np.ndarray,
    case: ProfileCase,
    alpha_s: float,
    beta_s: float,
    alpha_r: float,
    beta_r: float,
) -> np.ndarray:
    af = np.empty(len(freq), dtype=float)
    xi_s = rayleigh_xi(freq, alpha_s, beta_s)
    xi_r = rayleigh_xi(freq, alpha_r, beta_r)
    for i, (fi, zs, zr) in enumerate(zip(freq, xi_s, xi_r)):
        _, aw, _ = layered_transfer_function(
            fi,
            [Layer(case.H, case.vs1, RHO, float(zs))],
            RockHalfspace(case.vs2, RHO, float(zr)),
        )
        af[i] = float(aw[0])
    return af


def _rel_peak(a: np.ndarray, b: np.ndarray) -> float:
    pa, pb = float(np.nanmax(a)), float(np.nanmax(b))
    return (pa - pb) / pb if pb > 0 else float("nan")


def run_case(case: ProfileCase, *, force: bool = False) -> dict:
    f0 = case.vs1 / (4.0 * case.H)
    f1 = min(f0, MOTION_FREQ)
    f2 = 10.0
    duration = 50.0 if f0 < 1.0 else 30.0
    run_id = f"profile_{case.name}"
    run_dir = OUT / run_id
    vs, rho, nu, lz, interface_y, mask = build_grids(case)
    xi_s, xi_r = soil_rock_xi(vs, mask)
    alpha_s, beta_s = compute_rayleigh_coefficients(xi_s, f1, f2)
    alpha_r, beta_r = compute_rayleigh_coefficients(xi_r, f1, f2)

    have_rec = run_dir.exists() and len(list(run_dir.glob("center_node_y*_dof1_accel.txt"))) >= 2
    if force or not have_rec:
        config = AnalysisConfig(
            Ly=lz,
            Lx=DZ,
            hx=DZ,
            dt=DT,
            duration=duration,
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
        print(run_opensees_analysis(config, model, run_id=run_id, output_dir=str(OUT)))

    t, surf, base = _load_center_pair(run_dir)
    dt = float(t[1] - t[0]) if len(t) > 1 else DT
    freq, af_os = ttf_unsmoothed(surf, base, dt)
    af_ray = theory_rayleigh(freq, case, alpha_s, beta_s, alpha_r, beta_r)
    af_q = theory_const(freq, case, xi_s, xi_r)
    af_025 = theory_const(freq, case, CONFIG_ZETA, 0.0)

    npz_path = OUT / f"{case.name}.npz"
    np.savez(
        npz_path,
        freq=freq,
        af_os=af_os,
        af_th_rayleigh=af_ray,
        af_th_const_Q=af_q,
        af_th_const_0p025=af_025,
        vs_profile=vs.ravel(),
        dz=DZ,
        vs1=case.vs1,
        H=case.H,
        vs2=case.vs2,
        f0=f0,
        xi_soil_Q=xi_s,
        xi_rock_Q=xi_r,
    )
    i_os = int(np.nanargmax(af_os))
    return {
        "name": case.name,
        "label": case.label,
        "vs1": case.vs1,
        "H": case.H,
        "vs2": case.vs2,
        "f0": f0,
        "f1": f1,
        "xi_soil_Q": xi_s,
        "xi_rock_Q": xi_r,
        "f_peak_os": float(freq[i_os]),
        "peak_os": float(af_os[i_os]),
        "peak_ray": float(np.nanmax(af_ray)),
        "peak_q": float(np.nanmax(af_q)),
        "peak_025": float(np.nanmax(af_025)),
        "rel_peak_ray": _rel_peak(af_ray, af_os),
        "rel_peak_q": _rel_peak(af_q, af_os),
        "rel_peak_025": _rel_peak(af_025, af_os),
        "npz": str(npz_path),
    }


def plot_all(rows: list[dict]) -> Path:
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.6), sharex=True)
    fig.subplots_adjust(top=0.90, hspace=0.42, wspace=0.22)
    for ax, row in zip(axes.ravel(), rows):
        d = np.load(row["npz"])
        freq, af_os = d["freq"], d["af_os"]
        ax.loglog(freq, af_os, color="0.45", lw=1.15, label="OpenSees 1D")
        ax.loglog(freq, d["af_th_rayleigh"], color="#0072B2", lw=1.9, label=r"Theory Rayleigh $\xi(f)$")
        ax.loglog(freq, d["af_th_const_Q"], color="#009E73", lw=1.3, ls="--", label=r"Theory const. $Q$")
        ax.loglog(freq, d["af_th_const_0p025"], color="#D55E00", lw=1.3, ls=":", label=r"Theory $\xi$=0.025")
        ax.axvline(row["f0"], color="0.4", ls="--", lw=0.7)
        ax.set_xlim(0.1, 10)
        ax.set_ylim(0.5, 40)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_title(
            fr"{row['label']}: $V_{{s1}}$={row['vs1']:.0f}, $H$={row['H']:.0f} m, "
            fr"$V_{{s2}}$={row['vs2']:.0f}" + "\n"
            + fr"$f_0$={row['f0']:.2f} Hz, $\xi_Q$={100*row['xi_soil_Q']:.2f}%",
            fontsize=9,
        )
        if ax is axes[0, 0]:
            ax.legend(loc="center right", fontsize=7.5, frameon=True)
        ax.annotate(
            fr"$\Delta A_\mathrm{{Rayl}}$={100*row['rel_peak_ray']:+.1f}%" + "\n"
            + fr"$\Delta A_{{0.025}}$={100*row['rel_peak_025']:+.1f}%",
            xy=(0.03, 0.97),
            xycoords="axes fraction",
            ha="left",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="0.8"),
        )
    for ax in axes[-1]:
        ax.set_xlabel("Frequency (Hz)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$|AF_\mathrm{within}|$")
    fig.suptitle(
        "OpenSees 1D global_avg Rayleigh vs unsmoothed Thomson–Haskell  (2-layer Vs profiles)",
        fontsize=11,
    )
    out = OUT / "af_rayleigh_profiles.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 6.2), constrained_layout=True)
    colors = ["#000000", "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
    for i, row in enumerate(rows):
        d = np.load(row["npz"])
        vs = d["vs_profile"]
        depth = (np.arange(len(vs)) + 0.5) * float(d["dz"])
        ax.step(vs, depth, where="post", color=colors[i % len(colors)], lw=1.6, label=row["label"])
    ax.set_ylim(ax.get_ylim()[1], 0)
    ax.set_xlabel(r"$V_s$ (m/s)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("2-layer columns used in the TF comparison")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    profiles_png = OUT / "vs_profiles_compared.png"
    fig.savefig(profiles_png, dpi=160)
    plt.close(fig)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [run_case(case) for case in CASES]
    (OUT / "profiles_metrics.json").write_text(json.dumps(rows, indent=2))
    out = plot_all(rows)
    print(json.dumps(rows, indent=2))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
