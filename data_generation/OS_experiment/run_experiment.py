"""
OS_experiment: 32 simulations with three heights, RF seed varied within each height.

- 10 simulations: height 15 m, seeds 1-10
- 10 simulations: height 100 m, seeds 1-10
- 12 simulations: height 57 m, seeds 1-12

Fixed (same as rf_seed_experiment): Vs1=230, Vs2=1000, rH=30, CV=0.20,
1x1 discretization, damping_method="global_avg".

Total: 32 tasks (indices 0-31).
"""

import argparse
import csv
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

from seiskit.analysis import run_opensees_analysis
from seiskit.builder import build_model_data
from seiskit.config import AnalysisConfig
from seiskit.damping import (
    compute_average_damping_harmonic,
    compute_damping_from_Q,
    compute_quality_factor,
)
from seiskit.gaussian_field import (
    _extend_profile,
    _generate_vs_variability_field,
)
from seiskit.plot_results import plot_damping_realization, plot_realization
from seiskit.solver_utils import get_solver_info


TOTAL_COMBINATIONS = 32  # 10 + 10 + 12


def _configure_slurm_environment() -> None:
    """Configure threading and report SLURM context when running under SLURM."""
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if not os.getenv(var):
            os.environ[var] = "1"

    job_id = os.getenv("SLURM_JOB_ID", "-")
    task_id = os.getenv("PARALLEL_SEQ", "-")
    node = os.getenv("SLURMD_NODENAME", os.uname().nodename)
    print(f"[slurm] job_id={job_id} task_slot={task_id} node={node}")


def _install_sigterm_handler():
    """Install a SIGTERM handler to flush logs and exit cleanly."""

    def _handler(signum, frame):  # noqa: ARG001
        try:
            print("[signal] Received SIGTERM. Attempting graceful shutdown...")
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(143)

    try:
        signal.signal(signal.SIGTERM, _handler)
    except Exception:
        pass


def _fmt_hms(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    total_seconds = int(seconds)
    return f"{total_seconds // 3600:02d}:{(total_seconds % 3600) // 60:02d}:{total_seconds % 60:02d}"


def _index_to_thickness_seed(index: int) -> tuple[int, int]:
    """
    Map index 0-31 to (thickness, seed).
    - 0-9:   thickness=15,  seed=1..10
    - 10-19: thickness=100, seed=1..10
    - 20-31: thickness=57,  seed=1..12
    """
    if index < 0 or index >= TOTAL_COMBINATIONS:
        raise IndexError(
            f"Index {index} is out of range for {TOTAL_COMBINATIONS} tasks "
            f"(valid 0..{TOTAL_COMBINATIONS - 1})."
        )
    if index < 10:
        return 15, index + 1
    if index < 20:
        return 100, index - 9
    return 57, index - 19


def run_case(index: int = 0):
    """
    Run a single case given the index (0-31).

    Returns:
        Result status message
    """
    t0 = time.time()

    thickness, seed = _index_to_thickness_seed(index)

    # Fixed parameters (same as rf_seed_experiment)
    case_type = "1x1_4node"
    element_type = "4node"
    dx_base, dz_base = 1.0, 1.0
    aHV = 10.0
    interlayer_seed = 14
    Vs2 = 1000.0
    dz_1D = 1.0
    motion_freq = 3.0
    Lx_variability = 500.0
    BC_width = 500.0
    Lx = Lx_variability + 2 * BC_width
    interlayer_amplitude = 2.0 / 2
    damping_method_idx = "global_avg"

    Vs1 = 230.0
    rH = 30.0
    CV = 0.20
    dx, dz = dx_base, dz_base

    # Create Vs_profile_1D
    bedrock_depth = 10.0
    layer_1_count = int(thickness / dz_1D)
    layer_2_count = int(bedrock_depth / dz_1D)
    Vs_profile_1D = np.array([Vs1] * layer_1_count + [Vs2] * layer_2_count)
    Lz = (layer_1_count + layer_2_count) * dz_1D

    f0 = Vs1 / (4 * thickness)
    duration = 50.0 if f0 < 1.0 else 30.0
    damping_freq_first = min(f0, motion_freq)
    damping_freqs = (damping_freq_first, 10.0)

    output_dir = f"results/Vs1_{Vs1:.0f}_thickness_{thickness:.0f}_rH_{rH:.0f}_CV_{CV:.3f}_s{seed}"

    max_retries = 5
    for attempt in range(max_retries):
        try:
            os.makedirs(output_dir, exist_ok=True)
            break
        except (OSError, IOError):
            if attempt == max_retries - 1:
                raise
            time.sleep(0.1 * (attempt + 1))

    np.savetxt(f"{output_dir}/Vs_profile_1D.txt", Vs_profile_1D)

    realization_str = f"s{seed:02d}"
    task_id = (
        f"{case_type}_Vs1{Vs1:.0f}_th{thickness:.0f}_rH{rH:.0f}_CV{CV:.3f}_s{seed}"
    )

    print(f"[{case_type}] Starting task {task_id} (index={index})")
    print(f"  Case: {case_type}, Element type: {element_type}")
    print(f"  Layer 1 height: {thickness:.0f} m")
    print(f"  Vs1 = {Vs1} m/s, Vs2 = {Vs2} m/s, Total depth Lz = {Lz} m")
    print(f"  dx={dx} m, dz={dz} m")
    print(f"  Damping method: {damping_method_idx}")
    print(f"  CV = {CV}, seed = {seed}, realization = {realization_str}")
    print(f"  Motion frequency = {motion_freq} Hz")
    print(f"  f0 = {f0:.4f} Hz")
    print(f"  Duration = {duration:.1f} seconds")
    print(f"  Damping frequencies = {damping_freqs} Hz")
    print(
        f"  Lx_variability = {Lx_variability} m, BC_width = {BC_width} m, Total Lx = {Lx} m"
    )

    t_field_start = time.time()
    print(f"[{case_type}] Generating VS field with seed={seed}")
    np.random.seed(seed)
    Vs_realization, _, _, _, bedrock_mask_var = _generate_vs_variability_field(
        Vs_profile_1D,
        Lx_variability,
        Lz,
        dx_base,
        dz_base,
        rH,
        aHV,
        CV,
        seed=seed,
        dz_1D=dz_1D,
        interlayer_seed=interlayer_seed,
        interlayer_amplitude=interlayer_amplitude,
    )
    field_generation_time = time.time() - t_field_start

    rediscretization_time = 0.0

    Vs_extended, _ = _extend_profile(Vs_realization, Lx=Lx, dx=dx)
    bedrock_mask_extended, _ = _extend_profile(
        bedrock_mask_var.astype(float), Lx=Lx, dx=dx
    )
    bedrock_mask_extended = bedrock_mask_extended.astype(bool)

    np.savetxt(f"{output_dir}/Vs_array_2D.txt", Vs_extended)

    plot_realization(
        Vs_profile_1D,
        Vs_extended,
        Lx,
        Lz,
        dx,
        dz,
        save_path=f"{output_dir}/Vs_realization.png",
        title=f"Vs Realization (Vs1={Vs1:.0f} m/s, thickness={thickness:.0f} m, rH={rH:.0f} m, CV={CV:.3f}, seed={seed}, realization={realization_str}, damping_method={damping_method_idx})",
        bedrock_mask=bedrock_mask_extended,
    )

    config = AnalysisConfig(
        Ly=Lz,
        Lx=Lx,
        hx=dx,
        dt=0.01,
        duration=duration,
        motion_freq=motion_freq,
        motion_t_shift=0.5,
        damping_freqs=damping_freqs,
        damping_zeta=0.025,
        damping_method=damping_method_idx,
        boundary_condition_type="2D",
        record_center_nodes=True,
        center_node_y_positions=[2.0, Lz],
        record_all_surface_nodes=False,
        element_type=element_type,
        solver_type="Mumps",
    )

    plot_damping_realization(
        Vs_extended,
        damping_method_idx,
        Lx,
        Lz,
        dx,
        dz,
        save_path=f"{output_dir}/Damping_realization.png",
        title=f"Damping Realization (damping_method={damping_method_idx}, CV={CV:.3f}, seed={seed}, realization={realization_str})",
        bedrock_mask=bedrock_mask_extended,
        config=config,
    )

    zeta_grid = np.zeros_like(Vs_extended)
    soil_mask = ~bedrock_mask_extended

    if damping_method_idx == "global_avg":
        soil_Vs = Vs_extended[soil_mask]
        if len(soil_Vs) > 0:
            Q_values_soil = [compute_quality_factor(vs) for vs in soil_Vs]
            avg_damping_soil = compute_average_damping_harmonic(Q_values_soil)
            zeta_grid[soil_mask] = avg_damping_soil

        if np.any(bedrock_mask_extended):
            Q_bedrock = compute_quality_factor(Vs2)
            zeta_grid[bedrock_mask_extended] = compute_damping_from_Q(Q_bedrock)
    elif damping_method_idx == "uniform_soil_only":
        zeta_grid[soil_mask] = config.damping_zeta
        if np.any(bedrock_mask_extended):
            zeta_grid[bedrock_mask_extended] = 0.0075
    else:
        zeta_grid.fill(config.damping_zeta)

    np.savetxt(f"{output_dir}/damping_zeta_grid.txt", zeta_grid)

    solver_info = get_solver_info(config)
    solver_type_str = solver_info["solver_type"]
    if solver_info["mumps_parallel_procs"]:
        solver_type_str += f" (parallel, {solver_info['mumps_parallel_procs']} procs)"
    print(f"  Solver: {solver_type_str}")
    if solver_info["mumps_icntl"]:
        print(f"  MUMPS ICNTL parameters: {solver_info['mumps_icntl']}")
    print(f"  Domain: {Lx}m x {Lz}m")
    ndivx = int(Lx / dx)
    ndivy = int(Lz / dx)
    print(f"  Elements: {ndivx} x {ndivy} = {ndivx * ndivy}")
    print(f"  Time step: {config.dt}s, Duration: {config.duration}s")
    print(f"  Motion frequency: {config.motion_freq} Hz")
    print(f"  Damping method: {config.damping_method}")
    print(f"  Recording center nodes at Y positions: {config.center_node_y_positions}")

    rho = np.ones_like(Vs_extended) * 2000.0
    nu = np.ones_like(Vs_extended) * 0.3

    t_model_start = time.time()
    model_data = build_model_data(config, Vs_extended, rho, nu)
    model_build_time = time.time() - t_model_start

    t_analysis_start = time.time()
    print(f"[{case_type}] Running OpenSees for {task_id} -> {output_dir}")
    result = run_opensees_analysis(config, model_data, task_id, output_dir)
    analysis_time = time.time() - t_analysis_start

    total_time = time.time() - t0

    timing_file = Path(f"results/timing_data_task_{index}.csv")
    timing_file.parent.mkdir(parents=True, exist_ok=True)

    with open(timing_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "case_type",
                "Vs1",
                "thickness",
                "rH",
                "CV",
                "seed",
                "task_id",
                "total_time_sec",
                "field_generation_time_sec",
                "rediscretization_time_sec",
                "model_build_time_sec",
                "analysis_time_sec",
                "status",
            ]
        )
        writer.writerow(
            [
                case_type,
                f"{Vs1:.1f}",
                f"{thickness:.1f}",
                f"{rH:.1f}",
                f"{CV:.3f}",
                str(seed),
                task_id,
                f"{total_time:.3f}",
                f"{field_generation_time:.3f}",
                f"{rediscretization_time:.3f}",
                f"{model_build_time:.3f}",
                f"{analysis_time:.3f}",
                result,
            ]
        )

    print(f"[{case_type}] Done: {result} | Wall time: {_fmt_hms(total_time)}")
    print("  Timing breakdown:")
    print(f"    Field generation: {_fmt_hms(field_generation_time)}")
    print(f"    Model building: {_fmt_hms(model_build_time)}")
    print(f"    OpenSees analysis: {_fmt_hms(analysis_time)}")
    print(f"    Total: {_fmt_hms(total_time)}")
    print(f"  Timing data saved to: {timing_file}")

    return result


def _parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Run OS_experiment (32 tasks: H15×10, H100×10, H57×12).")
    p.add_argument(
        "--index",
        type=int,
        default=None,
        help="Task index 0-31. Required when run via gnu-parallel.",
    )
    return p.parse_args()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    program_start = time.time()

    _configure_slurm_environment()
    _install_sigterm_handler()

    args = _parse_args()

    idx = args.index
    if idx is None:
        env_idx = os.getenv("SLURM_ARRAY_TASK_ID") or os.getenv("PARALLEL_SEQ")
        if env_idx not in (None, ""):
            try:
                idx = int(env_idx)
                if os.getenv("PARALLEL_SEQ"):
                    idx -= 1  # gnu-parallel slot is 1-based
            except ValueError:
                idx = None
        if idx is None or idx < 0 or idx >= TOTAL_COMBINATIONS:
            print(
                "Error: --index 0-31 must be provided or set SLURM_ARRAY_TASK_ID/PARALLEL_SEQ",
                file=sys.stderr,
            )
            sys.exit(1)

    run_case(idx)

    print(f"\n[program] Total wall time: {_fmt_hms(time.time() - program_start)}")
    sys.exit(0)
