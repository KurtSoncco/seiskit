"""1D linear site response: OpenSees vs closed-form multilayer TF.

Runs a single-column ``boundary_condition_type="1D"`` analysis for 2-, 3-, and
4-layer soil profiles and compares AF(f) to Kramer / Thomson–Haskell theory.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from seiskit.analysis import run_opensees_analysis
from seiskit.builder import build_model_data
from seiskit.config import AnalysisConfig
from seiskit.theory.layered_1d_tf import Layer, RockHalfspace, layered_transfer_function
from seiskit.ttf.acc2FAS2 import acc2FAS2
from seiskit.ttf.TTF import TTF
from seiskit.utils import compute_ricker

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


@dataclass(frozen=True)
class CaseSpec:
    name: str
    layers: tuple[Layer, ...]
    rock: RockHalfspace


CASES: tuple[CaseSpec, ...] = (
    CaseSpec(
        name="2layer",
        layers=(Layer(30.0, 200.0, 2000.0, 0.0),),
        rock=RockHalfspace(1000.0, 2200.0, 0.0),
    ),
    CaseSpec(
        name="3layer",
        layers=(
            Layer(15.0, 180.0, 1900.0, 0.0),
            Layer(25.0, 300.0, 2000.0, 0.0),
        ),
        rock=RockHalfspace(1000.0, 2200.0, 0.0),
    ),
    CaseSpec(
        name="4layer",
        layers=(
            Layer(10.0, 150.0, 1800.0, 0.0),
            Layer(15.0, 220.0, 1900.0, 0.0),
            Layer(20.0, 350.0, 2000.0, 0.0),
        ),
        rock=RockHalfspace(1200.0, 2200.0, 0.0),
    ),
)

CASES_XI = tuple(
    CaseSpec(
        name=c.name + "_xi02",
        layers=tuple(Layer(L.H, L.Vs, L.rho, 0.02) for L in c.layers),
        rock=RockHalfspace(c.rock.Vs, c.rock.rho, 0.0),
    )
    for c in CASES
)


def build_column_grids(
    layers: tuple[Layer, ...],
    rock: RockHalfspace,
    hx: float,
    rock_buffer: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Build (nz, 1) Vs/rho/nu grids, surface → bedrock.

    Returns vs, rho, nu, Ly, interface_y, vs_min_soil.
    """
    rows_vs: list[float] = []
    rows_rho: list[float] = []
    for layer in layers:
        n = max(1, int(round(layer.H / hx)))
        rows_vs.extend([layer.Vs] * n)
        rows_rho.extend([layer.rho] * n)
    n_rock = max(1, int(round(rock_buffer / hx)))
    rows_vs.extend([rock.Vs] * n_rock)
    rows_rho.extend([rock.rho] * n_rock)

    vs = np.asarray(rows_vs, dtype=float).reshape(-1, 1)
    rho = np.asarray(rows_rho, dtype=float).reshape(-1, 1)
    nu = np.full_like(vs, 0.3)
    Ly = float(len(rows_vs) * hx)
    interface_y = float(n_rock * hx)
    vs_min = float(min(L.Vs for L in layers))
    return vs, rho, nu, Ly, interface_y, vs_min


def _load_center_pair(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    files = sorted(run_dir.glob("center_node_y*_dof1_accel.txt"))
    if len(files) < 2:
        raise FileNotFoundError(f"Need ≥2 center recorders in {run_dir}, found {files}")

    def y_of(p: Path) -> float:
        stem = p.name.split("_dof")[0]
        return float(stem.replace("center_node_y", ""))

    files = sorted(files, key=y_of)
    base = np.loadtxt(files[0])
    surf = np.loadtxt(files[-1])
    return base[:, 0], surf[:, 1], base[:, 1], y_of(files[0]), y_of(files[-1])


def _fas_tf(surface: np.ndarray, reference: np.ndarray, dt: float, vsmin: float, dz: float):
    return TTF(
        surface,
        reference,
        dt=dt,
        n_points=800,
        Vsmin=vsmin,
        dz=dz,
        smooth_coeff=200,
    )


def _rel_error_metrics(
    freq: np.ndarray,
    af_os: np.ndarray,
    af_th: np.ndarray,
    f_lo: float,
    f_hi: float,
) -> dict:
    mask = (freq >= f_lo) & (freq <= f_hi) & np.isfinite(af_os) & np.isfinite(af_th)
    mask &= af_th > 0.05
    if not np.any(mask):
        return {"median_rel_err": np.nan, "p95_rel_err": np.nan, "n": 0}
    rel = np.abs(af_os[mask] - af_th[mask]) / af_th[mask]
    return {
        "median_rel_err": float(np.median(rel)),
        "p95_rel_err": float(np.percentile(rel, 95)),
        "max_rel_err": float(np.max(rel)),
        "n": int(np.sum(mask)),
    }


def _peak_freq(freq: np.ndarray, af: np.ndarray, f_lo: float, f_hi: float) -> float:
    mask = (freq >= f_lo) & (freq <= f_hi)
    if not np.any(mask):
        return float("nan")
    return float(freq[mask][np.argmax(af[mask])])


def _travel_time_f0(layers: tuple[Layer, ...]) -> float:
    tt = sum(L.H / L.Vs for L in layers)
    return 1.0 / (4.0 * tt)


def run_case(
    case: CaseSpec,
    *,
    hx: float = 1.0,
    dt: float = 5e-4,
    duration: float = 20.0,
    motion_freq: float = 2.0,
    rock_buffer: float | None = None,
    skip_opensees: bool = False,
) -> dict:
    """Run one 1D OpenSees column and compare AF to closed-form theory."""
    if rock_buffer is None:
        rock_buffer = max(2.0 * hx, 2.0)

    vs, rho, nu, Ly, interface_y, vs_min = build_column_grids(
        case.layers, case.rock, hx, rock_buffer
    )
    xi_soil = case.layers[0].xi
    f0_est = _travel_time_f0(case.layers)
    if xi_soil == 0.0:
        damping_method = "none"
        damping_zeta = 0.0
        damp_freqs = (max(0.5, motion_freq * 0.5), max(5.0, motion_freq * 3.0))
    else:
        damping_method = "uniform_soil_only"
        damping_zeta = float(xi_soil)
        damp_freqs = (f0_est, 3.0 * f0_est)

    config = AnalysisConfig(
        Ly=Ly,
        Lx=hx,
        hx=hx,
        dt=dt,
        duration=duration,
        motion_freq=motion_freq,
        motion_t_shift=1.5 / motion_freq,
        damping_method=damping_method,
        damping_zeta=damping_zeta,
        damping_freqs=damp_freqs,
        boundary_condition_type="1D",
        record_center_nodes=True,
        center_node_y_positions=[interface_y, Ly],
        record_all_surface_nodes=False,
        element_type="4node",
        solver_type="UmfPack",
    )

    run_id = case.name
    out_dir = OUT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if not skip_opensees:
        model = build_model_data(
            config, vs, rho, nu, bedrock_mask=(vs >= case.rock.Vs * 0.99)
        )
        print(run_opensees_analysis(config, model, run_id=run_id, output_dir=str(OUT)))
    elif not any(out_dir.glob("center_node_y*_dof1_accel.txt")):
        raise FileNotFoundError(f"No recorder output for {run_id}")

    t, surf, base, y_base, y_surf = _load_center_pair(out_dir)
    dt_rec = float(t[1] - t[0]) if len(t) > 1 else dt
    duration_eff = float(t[-1]) if len(t) else duration

    freq_os, af_within_os = _fas_tf(surf, base, dt_rec, vs_min, hx)

    a_inc = compute_ricker(
        config.motion_freq, config.motion_t_shift, duration_eff, config.dt
    )
    n = min(len(surf), len(a_inc))
    freq_o, af_outcrop_os = _fas_tf(surf[:n], 2.0 * a_inc[:n], dt_rec, vs_min, hx)

    _, af_within_th, _ = layered_transfer_function(freq_os, list(case.layers), case.rock)
    _, _, af_outcrop_th = layered_transfer_function(freq_o, list(case.layers), case.rock)

    fas_inc, f_inc = acc2FAS2(a_inc, config.dt)
    fas_n = fas_inc / max(float(np.max(fas_inc)), 1e-30)
    energized = f_inc[fas_n >= 0.10]
    f_lo = max(0.15, 0.25 * f0_est)
    f_hi_energy = float(energized[-1]) if energized.size else 4.0 * f0_est
    f_hi = min(float(freq_os[-1]) * 0.9, f_hi_energy, vs_min / (10.0 * hx), 3.0 * f0_est)

    metrics_within = _rel_error_metrics(freq_os, af_within_os, af_within_th, f_lo, f_hi)
    metrics_outcrop = _rel_error_metrics(freq_o, af_outcrop_os, af_outcrop_th, f_lo, f_hi)

    f_mode_hi = min(f_hi, 2.0 * f0_est)
    f_peak_os = _peak_freq(freq_o, af_outcrop_os, f_lo, f_mode_hi)
    f_peak_th = _peak_freq(freq_o, af_outcrop_th, f_lo, f_mode_hi)
    peak_rel = abs(f_peak_os - f_peak_th) / max(f_peak_th, 1e-12)
    i_th = int(np.argmin(np.abs(freq_o - f_peak_th)))
    peak_amp_rel = abs(af_outcrop_os[i_th] - af_outcrop_th[i_th]) / max(
        af_outcrop_th[i_th], 1e-12
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].semilogx(freq_os, af_within_th, "k-", lw=2, label="Theory within")
    axes[0].semilogx(freq_os, af_within_os, "r--", lw=1.5, label="OpenSees 1D")
    axes[0].set_xlabel("Frequency [Hz]")
    axes[0].set_ylabel("|AF| within")
    axes[0].set_title(f"{case.name} within (y_base={y_base:.1f})")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].semilogx(freq_o, af_outcrop_th, "k-", lw=2, label="Theory outcrop")
    axes[1].semilogx(freq_o, af_outcrop_os, "r--", lw=1.5, label="OpenSees 1D")
    axes[1].set_xlabel("Frequency [Hz]")
    axes[1].set_ylabel("|AF| outcrop")
    axes[1].set_title(f"{case.name} outcrop")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "af_comparison.png", dpi=150)
    plt.close(fig)

    np.savez(
        out_dir / "af_data.npz",
        freq_within=freq_os,
        af_within_os=af_within_os,
        af_within_th=af_within_th,
        freq_outcrop=freq_o,
        af_outcrop_os=af_outcrop_os,
        af_outcrop_th=af_outcrop_th,
    )

    result = {
        "case": case.name,
        "Ly": Ly,
        "interface_y": interface_y,
        "y_base_recorded": y_base,
        "y_surf_recorded": y_surf,
        "hx": hx,
        "dt": dt,
        "f_lo": f_lo,
        "f_hi": f_hi,
        "f0_est": f0_est,
        "f_peak_os": f_peak_os,
        "f_peak_th": f_peak_th,
        "peak_freq_rel_err": peak_rel,
        "peak_amp_rel_err": float(peak_amp_rel),
        "within": metrics_within,
        "outcrop": metrics_outcrop,
        "layers": [asdict(L) for L in case.layers],
        "rock": asdict(case.rock),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def _passes_tolerance(r: dict) -> bool:
    """Gate on AF_within; also require AF_outcrop for undamped cases."""
    within = r.get("within") or {}
    out = r.get("outcrop") or {}
    w_med = within.get("median_rel_err", 1.0)
    w_p95 = within.get("p95_rel_err", 1.0)
    if w_med is None or w_p95 is None:
        return False
    if not (w_med < 0.05 and w_p95 < 0.15 and r.get("peak_freq_rel_err", 1.0) < 0.02):
        return False
    if "xi" in str(r.get("case", "")):
        return True
    o_med = out.get("median_rel_err", 1.0)
    o_p95 = out.get("p95_rel_err", 1.0)
    return (
        o_med is not None
        and o_p95 is not None
        and o_med < 0.05
        and o_p95 < 0.15
        and r.get("peak_amp_rel_err", 1.0) < 0.10
    )


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    case_filter = None
    skip_opensees = "--recompute-only" in argv
    for a in argv:
        if a.startswith("--case="):
            case_filter = a.split("=", 1)[1]

    OUT.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for case in CASES + CASES_XI:
        if case_filter and case.name != case_filter and not (
            case_filter.endswith("*") and case.name.startswith(case_filter[:-1])
        ):
            continue
        print(f"\n=== {case.name} ===")
        results.append(run_case(case, skip_opensees=skip_opensees))

    csv_path = OUT / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "case",
                "within_median_rel_err",
                "within_p95_rel_err",
                "outcrop_median_rel_err",
                "peak_freq_rel_err",
                "peak_amp_rel_err",
                "f_peak_os",
                "f_peak_th",
                "pass",
            ]
        )
        for r in results:
            passed = _passes_tolerance(r)
            w.writerow(
                [
                    r["case"],
                    r["within"].get("median_rel_err"),
                    r["within"].get("p95_rel_err"),
                    r["outcrop"].get("median_rel_err"),
                    r["peak_freq_rel_err"],
                    r.get("peak_amp_rel_err"),
                    r["f_peak_os"],
                    r["f_peak_th"],
                    passed,
                ]
            )

    print(f"\nWrote {csv_path}")
    for r in results:
        flag = "PASS" if _passes_tolerance(r) else "FAIL"
        print(
            f"{r['case']:14s}  "
            f"within med={r['within'].get('median_rel_err', float('nan')):.4f}  "
            f"outcrop med={r['outcrop'].get('median_rel_err', float('nan')):.4f}  "
            f"peak_f={r['peak_freq_rel_err']:.4f}  [{flag}]"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
