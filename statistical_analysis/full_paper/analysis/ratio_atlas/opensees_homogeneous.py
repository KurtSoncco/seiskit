"""Homogeneous OpenSees check at H = 15 m.

One 2D run with CoV = 0, using the H = 15 m campaign geometry (Lx = 1500 m,
side and base absorbers, recorders at center ±100 m), plus the matching 1D
column. Surface PGA across the array divided by the 1D surface PGA is flat
near 1 when the array does not see the boundaries.

Recorder files are written under /tmp so the solve does not stream to Box.
The PDF, text summary, and a NumPy archive of the surface records are copied
to ``figure_dir("ratio_atlas", "opensees_h15")``.
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

# OpenSeesPy's Linux wheel needs its bundled Fortran runtime on the loader path.
_OSEES_LIB = (
    Path(sys.prefix)
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
    / "openseespylinux"
    / "lib"
)
for _lib_name in (
    "libquadmath.so.0",
    "libgfortran.so.4",
    "libgomp.so.1",
    "libblas.so.3",
    "liblapack.so.3",
):
    ctypes.CDLL(str(_OSEES_LIB / _lib_name), mode=ctypes.RTLD_GLOBAL)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

_ATLAS = Path(__file__).resolve().parent
_FULL_PAPER = _ATLAS.parents[1]
if str(_FULL_PAPER) not in sys.path:
    sys.path.insert(0, str(_FULL_PAPER))

from config import (  # noqa: E402
    DATA_LINEWIDTH,
    LABEL_FONTSIZE,
    REF_COLOR,
    TICK_LABELSIZE,
    apply_full_paper_style,
    figsize,
    figure_dir,
    metric_color,
    save_figure,
)

from seiskit.analysis import run_opensees_analysis  # noqa: E402
from seiskit.builder import build_model_data  # noqa: E402
from seiskit.config import AnalysisConfig  # noqa: E402
from seiskit.gaussian_field import (  # noqa: E402
    _extend_profile,
    _generate_vs_variability_field,
)

apply_full_paper_style(auto_format=True, frame="open", grid=False)

# H = 15 m campaign geometry (emulator_first).
THICKNESS = 15.0
VS1 = 230.0
VS2 = 1500.0
DX = 1.0
DZ = 1.0
LX_VARIABILITY = 500.0
BC_WIDTH = 500.0
LX = LX_VARIABILITY + 2 * BC_WIDTH
BEDROCK = 10.0
MOTION_FREQ = 3.0
DAMPING_ZETA = 0.025
RHO = 2000.0
NU = 0.3
NODES_EACH_SIDE = 50
RECORDER_SPACING = 2.0
RH = 30.0
AHV = 10.0
COV = 0.0
SEED = 1

WORK = Path("/tmp/opensees_h15_homogeneous")
DOC_WIDTH, FIG_HEIGHT = figsize(aspect=0.48)


def _duration_s(vs1: float, thickness: float) -> float:
    f0 = vs1 / (4.0 * thickness)
    return 50.0 if f0 < 1.0 else 30.0


def homogeneous_field() -> tuple[np.ndarray, np.ndarray, float]:
    """Return (Vs[nz, nx], bedrock_mask, Lz) for the CoV = 0 domain."""
    layer_1 = int(THICKNESS / DZ)
    layer_2 = int(BEDROCK / DZ)
    vs_profile = np.array([VS1] * layer_1 + [VS2] * layer_2)
    lz = float(vs_profile.size * DZ)
    vs_var, _, _, _, bedrock_var = _generate_vs_variability_field(
        vs_profile,
        LX_VARIABILITY,
        lz,
        DX,
        DZ,
        RH,
        AHV,
        COV,
        seed=SEED,
        dz_1D=DZ,
        interlayer_seed=14,
        interlayer_amplitude=0.0,
    )
    vs, _ = _extend_profile(vs_var, Lx=LX, dx=DX)
    bedrock, _ = _extend_profile(bedrock_var.astype(float), Lx=LX, dx=DX)
    return vs, bedrock.astype(bool), lz


def _config_2d(lz: float, duration: float) -> AnalysisConfig:
    f0 = VS1 / (4.0 * THICKNESS)
    return AnalysisConfig(
        Ly=lz,
        Lx=LX,
        hx=DX,
        dt=0.01,
        duration=duration,
        motion_freq=MOTION_FREQ,
        motion_t_shift=0.5,
        damping_freqs=(min(f0, MOTION_FREQ), 10.0),
        damping_zeta=DAMPING_ZETA,
        damping_method="global_avg",
        boundary_condition_type="2D",
        record_center_nodes=False,
        center_node_y_positions=[2.0, lz],
        record_lateral_span_at_center_depths=(NODES_EACH_SIDE, RECORDER_SPACING),
        record_all_surface_nodes=False,
        element_type="4node",
        solver_type="Mumps",
    )


def _config_1d(lz: float, duration: float) -> AnalysisConfig:
    f0 = VS1 / (4.0 * THICKNESS)
    return AnalysisConfig(
        Ly=lz,
        Lx=DX,
        hx=DX,
        dt=0.01,
        duration=duration,
        motion_freq=MOTION_FREQ,
        motion_t_shift=0.5,
        damping_freqs=(min(f0, MOTION_FREQ), 10.0),
        damping_zeta=DAMPING_ZETA,
        damping_method="global_avg",
        boundary_condition_type="1D",
        record_center_nodes=True,
        center_node_y_positions=[lz],
        record_all_surface_nodes=False,
        element_type="4node",
        solver_type="Mumps",
    )


def recorder_x(config: AnalysisConfig) -> np.ndarray:
    """Horizontal coordinates of the lateral-span recorders, centered on the domain."""
    ndivx = int(config.Lx / config.hx) + 2
    i_rec = int(ndivx / 2)
    step = max(1, round(RECORDER_SPACING / config.hx))
    i_min = max(1, i_rec - NODES_EACH_SIDE * step)
    i_max = min(ndivx - 1, i_rec + NODES_EACH_SIDE * step)
    idx = np.arange(i_min, i_max + 1, step)
    return (idx.astype(float) - 1.0) * config.hx - config.Lx / 2.0


def _load_accel(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1:]


def _pga(accel: np.ndarray) -> np.ndarray:
    return np.max(np.abs(accel), axis=0)


def _run(config: AnalysisConfig, vs: np.ndarray, bedrock: np.ndarray, run_id: str) -> str:
    rho = np.full_like(vs, RHO, dtype=float)
    nu = np.full_like(vs, NU, dtype=float)
    model = build_model_data(config, vs, rho, nu, bedrock_mask=bedrock)
    status = run_opensees_analysis(config, model, run_id, str(WORK))
    print(status)
    if not str(status).startswith("Finished"):
        raise RuntimeError(status)
    return status


def plot_ratio(x: np.ndarray, ratio: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(DOC_WIDTH, FIG_HEIGHT))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.90)
    ax.axhline(1.0, color=REF_COLOR, lw=DATA_LINEWIDTH, zorder=1)
    ax.plot(x, ratio, color=metric_color("PGA_ratio"), lw=DATA_LINEWIDTH, zorder=3)
    ax.set_xlim(-108, 108)
    ax.set_xticks([-100, 0, 100])
    pad = 0.05 * max(float(np.ptp(ratio)), 0.02)
    mid = 0.5 * (float(np.min(ratio)) + float(np.max(ratio)))
    half = max(0.5 * float(np.ptp(ratio)) + pad, 0.02)
    ax.set_ylim(mid - half, mid + half)
    ax.set_xlabel("Distance from center (m)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(r"$\mathrm{PGA}/\mathrm{PGA}_{1D}$", fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_LABELSIZE)
    ax.grid(True, which="major", alpha=0.18, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(
        rf"$H = {THICKNESS:.0f}$ m, $V_{{s1}} = {VS1:.0f}$ m/s, CoV = 0",
        fontsize=TICK_LABELSIZE,
    )
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=metric_color("PGA_ratio"),
                lw=DATA_LINEWIDTH,
                label="2D homogeneous",
            ),
            Line2D([0], [0], color=REF_COLOR, lw=DATA_LINEWIDTH, label="1D column"),
        ],
        frameon=False,
        fontsize=TICK_LABELSIZE,
        loc="best",
    )
    return fig


def summary_text(x: np.ndarray, ratio: np.ndarray, pga_1d: float) -> str:
    center = int(np.argmin(np.abs(x)))
    ends = (0, -1)
    span = float(np.max(ratio) - np.min(ratio))
    lines = [
        "Homogeneous OpenSees check (CoV = 0)",
        (
            f"H = {THICKNESS:.0f} m, Vs1 = {VS1:.0f} m/s, Vs2 = {VS2:.0f} m/s, "
            f"Lx = {LX:.0f} m, BC width = {BC_WIDTH:.0f} m"
        ),
        f"Recorders: {x.size} at {RECORDER_SPACING:.0f} m, from {x[0]:.0f} m to {x[-1]:.0f} m.",
        f"1D surface PGA: {pga_1d:.6e}",
        (
            f"PGA/PGA_1D: min {np.min(ratio):.6f} at {x[int(np.argmin(ratio))]:.0f} m, "
            f"max {np.max(ratio):.6f} at {x[int(np.argmax(ratio))]:.0f} m, "
            f"center {ratio[center]:.6f}, span {span:.6f}"
        ),
        (
            f"Ends / center: left {ratio[ends[0]] / ratio[center]:.6f}, "
            f"right {ratio[ends[1]] / ratio[center]:.6f}"
        ),
        "",
    ]
    if span < 0.01 and abs(float(ratio[center]) - 1.0) < 0.02:
        lines.append(
            "The array is flat and matches the 1D column. The side absorbers and "
            "the homogeneous pad do not imprint a spatial trend on center ±100 m."
        )
    elif span < 0.01:
        lines.append(
            "The array is spatially flat, so the boundaries are not creating a "
            "trend across center ±100 m. The level differs from the 1D column "
            f"by {float(ratio[center]) - 1.0:+.4f} at the center."
        )
    else:
        lines.append(
            "PGA varies across the array in this homogeneous model. That spatial "
            "change is a boundary or domain effect, because CoV = 0 removes the "
            "random field."
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    out_dir = figure_dir("ratio_atlas", "opensees_h15")
    stem = "pga_ratio_h15_vs1_230_cov0"
    saved = out_dir / f"{stem}.npz"
    if saved.is_file():
        data = np.load(saved)
        x = np.asarray(data["x_m"], dtype=float)
        ratio = np.asarray(data["ratio"], dtype=float)
        pga_1d = float(np.asarray(data["pga_1d"]))
        text = summary_text(x, ratio, pga_1d)
        print(text)
        (out_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
        fig = plot_ratio(x, ratio)
        save_figure(fig, stem, out_dir=out_dir)
        plt.close(fig)
        return

    WORK.mkdir(parents=True, exist_ok=True)
    vs, bedrock, lz = homogeneous_field()
    col_spread = float(np.max(np.std(vs, axis=1)))
    print(f"Vs field {vs.shape}, Lz={lz:.0f} m, column std max={col_spread:.3e}")
    if col_spread > 1e-6:
        raise RuntimeError(f"CoV = 0 field is not laterally uniform (std {col_spread})")

    duration = _duration_s(VS1, THICKNESS)
    print(f"Duration {duration:.0f} s")

    vs_1d = vs[:, vs.shape[1] // 2 : vs.shape[1] // 2 + 1].copy()
    bedrock_1d = bedrock[:, bedrock.shape[1] // 2 : bedrock.shape[1] // 2 + 1].copy()

    print("Running 1D column")
    _run(_config_1d(lz, duration), vs_1d, bedrock_1d, "homogeneous_1d")
    print("Running 2D homogeneous domain")
    cfg_2d = _config_2d(lz, duration)
    _run(cfg_2d, vs, bedrock, "homogeneous_2d")

    surface_2d = WORK / "homogeneous_2d" / f"row_y{lz:.2f}_dof1_accel.txt"
    surface_1d = WORK / "homogeneous_1d" / f"center_node_y{lz:.2f}_dof1_accel.txt"
    time_2d, accel_2d = _load_accel(surface_2d)
    _time_1d, accel_1d = _load_accel(surface_1d)
    x = recorder_x(cfg_2d)
    if accel_2d.shape[1] != x.size:
        raise RuntimeError(f"recorder count {accel_2d.shape[1]} != x count {x.size}")
    pga = _pga(accel_2d)
    pga_1d = float(_pga(accel_1d)[0])
    ratio = pga / pga_1d

    text = summary_text(x, ratio, pga_1d)
    print(text)
    out_dir = figure_dir("ratio_atlas", "opensees_h15")
    stem = "pga_ratio_h15_vs1_230_cov0"
    (out_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
    np.savez(
        out_dir / f"{stem}.npz",
        time_2d=time_2d,
        x_m=x,
        pga=pga,
        pga_1d=np.array(pga_1d),
        ratio=ratio,
        accel_2d=accel_2d,
        accel_1d=accel_1d,
    )
    fig = plot_ratio(x, ratio)
    save_figure(fig, stem, out_dir=out_dir)
    plt.close(fig)


if __name__ == "__main__":
    main()
