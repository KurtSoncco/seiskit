"""
Sensitivity analysis for spatially variable soil properties.

In this case, we will try:
- 2 different values of Vs1: 100, 360 m/s
- 2 different values of thickness: 14, 100 m
- 2 different values of rH: 10, 50 m
- 2 different values of CV: 0.1, 0.3
- 5 different seeds: 10, 20, 30, 40, 50
- 1 value of aHV = 10 m
- 1 value of interlayer_seed = 14, with height of 2 meters
- 1 value of Vs2 = 1500 m/s, only extend it 10 m from the Vs1 profile
- 1 value of motion_freq = 3.0 Hz (Ricker wave central frequency)
- Duration: 50.0 seconds if f0 < 1 Hz, else 30.0 seconds
- Damping frequencies = (min(f0, 3.0), 10.0) Hz (uses f0 or 3Hz, whichever is smaller)
- 1 value of Damping method: "global_avg"
- Boundary condition type = "2D"
- Record center nodes at surface and base after 2 meters of the bottom (2 m from the bottom)
- Element type = "4node"
- Solver type = "Mumps"

Total number of combinations: 2 * 2 * 2 * 2 * 5 = 80
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


def _configure_slurm_environment() -> None:
    """Configure threading and report SLURM context when running under SLURM."""
    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        for var in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            if not os.getenv(var):
                os.environ[var] = slurm_cpus

    job_id = os.getenv("SLURM_JOB_ID", "-")
    array_id = os.getenv("SLURM_ARRAY_JOB_ID", job_id)
    task_id = os.getenv("SLURM_ARRAY_TASK_ID", "-")
    node = os.getenv("SLURMD_NODENAME", os.uname().nodename)
    cpus = slurm_cpus or "-"
    print(f"[slurm] job_id={job_id} array_id={array_id} task_id={task_id} node={node} cpus={cpus}")


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
    return (
        f"{total_seconds // 3600:02d}:{(total_seconds % 3600) // 60:02d}:{total_seconds % 60:02d}"
    )


def run_case(index: int = 0):
    """
    Run a single case given the index.
    """
    t0 = time.time()

    # Fixed parameters
    case_type = "2x2_4node"
    element_type = "4node"
    dx, dz = 2.0, 2.0  # 2x2 discretization
    aHV = 10.0  # anisotropy ratio for the intralayer variability
    interlayer_seed = 14  # seed for the interlayer variability
    Vs2 = 1500.0  # Layer 2 (bedrock) velocity
    dz_1D = 2.0  # Vertical spacing for 1D profile
    motion_freq = 3.0  # Fixed at 3.0 Hz (Ricker wave central frequency)
    # Duration will be set based on f0: 50 sec if f0 < 1 Hz, else 30 sec
    # damp_freq = (0.75, 11.25)  # Damping frequencies
    Lx_variability = 500.0
    BC_width = 500.0
    Lx = Lx_variability + 2 * BC_width
    interlayer_amplitude = 2.0 / 2
    damping_method_idx = "global_avg"  # "global_avg" or "uniform_soil_only"

    # Parameter variations
    Vs1_list = [100, 360]
    thickness_list = [14, 100]  # Everything is a multiple of 2 meters
    rH_list = [10, 50]
    CV_list = [0.1, 0.3]
    seed_values = [10, 20, 30, 40, 50]  # 5 different seeds

    total_combinations = (
        len(Vs1_list) * len(thickness_list) * len(rH_list) * len(CV_list) * len(seed_values)
    )

    if index < 0 or index >= total_combinations:
        raise IndexError(
            f"Index {index} is out of range for {total_combinations} tasks "
            f"(valid 0..{total_combinations - 1})."
        )

    # Map index to parameter combination
    # Index structure: index = Vs1_idx * (2*2*2*5) + thickness_idx * (2*2*5) + rH_idx * (2*5) + CV_idx * (5) + seed_idx
    # Order: Vs1 -> thickness -> rH -> CV -> seed
    Vs1_idx = index // (len(thickness_list) * len(rH_list) * len(CV_list) * len(seed_values))
    remainder = index % (len(thickness_list) * len(rH_list) * len(CV_list) * len(seed_values))
    thickness_idx = remainder // (len(rH_list) * len(CV_list) * len(seed_values))
    remainder = remainder % (len(rH_list) * len(CV_list) * len(seed_values))
    rH_idx = remainder // (len(CV_list) * len(seed_values))
    remainder = remainder % (len(CV_list) * len(seed_values))
    CV_idx = remainder // len(seed_values)
    seed_idx = remainder % len(seed_values)

    Vs1 = Vs1_list[Vs1_idx]
    thickness = thickness_list[thickness_idx]
    rH = rH_list[rH_idx]
    CV = CV_list[CV_idx]
    seed = seed_values[seed_idx]

    # Create Vs_profile_1D
    layer_1_count = int(thickness / dz_1D)
    layer_2_count = int(10 / dz_1D)
    Vs_profile_1D = np.array(
        [Vs1] * layer_1_count + [Vs2] * layer_2_count
    )  # Adding 5 elements (10 meters) of bedrock
    Lz = layer_1_count * dz_1D + layer_2_count * dz_1D

    # Compute f0
    f0 = Vs1 / (4 * thickness)
    # Duration: 50 sec if f0 < 1 Hz, else 30 sec
    duration = 50.0 if f0 < 1.0 else 30.0
    # Damping frequencies: use min(f0, 3.0) or 3Hz (Ricker wave central freq)
    damping_freq_first = min(f0, motion_freq)  # Use min(f0, 3.0)
    damping_freqs = (damping_freq_first, 10.0)

    # Create output directory (simplified: only keep varying parameters, remove fixed damping_method)
    output_dir = f"results/Vs1_{Vs1:.0f}_thickness_{thickness:.0f}_rH_{rH:.0f}_CV_{CV:.3f}_s{seed}"
    os.makedirs(output_dir, exist_ok=True)

    # Save Vs_profile_1D to file
    np.savetxt(f"{output_dir}/Vs_profile_1D.txt", Vs_profile_1D)

    # Create directories with retry logic
    max_retries = 5
    for attempt in range(max_retries):
        try:
            os.makedirs(output_dir, exist_ok=True)
            break
        except (OSError, IOError):
            if attempt == max_retries - 1:
                raise
            time.sleep(0.1 * (attempt + 1))

    # Format realization string and task ID (simplified: remove fixed damping_method)
    realization_idx = seed_idx + 1  # 1-based for s01, s02, etc.
    realization_str = f"s{realization_idx:02d}"
    task_id = f"{case_type}_Vs1{Vs1:.0f}_th{thickness:.0f}_rH{rH:.0f}_CV{CV:.3f}_s{seed}"

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
    print(f"  Lx_variability = {Lx_variability} m, BC_width = {BC_width} m, Total Lx = {Lx} m")

    # Generate VS field
    t_field_start = time.time()
    print(f"[{case_type}] Generating VS field with seed={seed}")
    np.random.seed(seed)
    Vs_realization, x_coords, z_coords, h_mean, bedrock_mask_var = _generate_vs_variability_field(
        Vs_profile_1D,
        Lx_variability,
        Lz,
        dx,
        dz,
        rH,
        aHV,
        CV,
        seed=seed,
        dz_1D=dz_1D,
        interlayer_seed=interlayer_seed,
        interlayer_amplitude=interlayer_amplitude,
    )
    field_generation_time = time.time() - t_field_start

    # Extend the profile with BC zones on each side
    Vs_extended, x_total = _extend_profile(
        Vs_realization,
        Lx=Lx,
        dx=dx,
    )

    # Extend bedrock mask similarly
    bedrock_mask_extended, _ = _extend_profile(
        bedrock_mask_var.astype(float),  # Convert boolean to float for _extend_profile
        Lx=Lx,
        dx=dx,
    )
    bedrock_mask_extended = bedrock_mask_extended.astype(bool)  # Convert back to boolean

    # Save 2D Vs array (needed for results plots)
    np.savetxt(f"{output_dir}/Vs_array_2D.txt", Vs_extended)

    # Save  Vs realization plot
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

    # Build analysis config first (needed for plot)
    config = AnalysisConfig(
        Ly=Lz,
        Lx=Lx,
        hx=dx,  # Element size
        dt=0.01,
        duration=duration,
        motion_freq=motion_freq,  # Fixed at 3.0 Hz
        motion_t_shift=0.5,
        damping_freqs=damping_freqs,  # (min(f0, 3.0), 10.0) Hz
        damping_zeta=0.025,  # 2.5% damping for uniform_soil_only method
        damping_method=damping_method_idx,  # "global_avg" or "uniform_soil_only"
        boundary_condition_type="2D",
        record_center_nodes=True,
        center_node_y_positions=[2.0, Lz],  # Record at 2m from bottom and at surface
        record_all_surface_nodes=False,
        element_type=element_type,
        solver_type="Mumps",
    )

    # Save Damping realization plot (using config values)
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

    # Compute and save damping zeta grid array (same logic as plot_damping_realization)
    nz, nx = Vs_extended.shape
    zeta_grid = np.zeros_like(Vs_extended)

    if damping_method_idx == "global_avg":
        # Use bedrock mask to identify soil vs bedrock
        soil_mask = ~bedrock_mask_extended
        soil_Vs = Vs_extended[soil_mask]

        if len(soil_Vs) > 0:
            Q_values_soil = [compute_quality_factor(vs) for vs in soil_Vs]
            avg_damping_soil = compute_average_damping_harmonic(Q_values_soil)
            # Apply to all soil elements
            zeta_grid[soil_mask] = avg_damping_soil

        # Bedrock elements get bedrock damping
        if np.any(bedrock_mask_extended):
            bedrock_Vs = 1500.0
            Q_bedrock = compute_quality_factor(bedrock_Vs)
            xi_bedrock = compute_damping_from_Q(Q_bedrock)
            zeta_grid[bedrock_mask_extended] = xi_bedrock
    elif damping_method_idx == "uniform_soil_only":
        # Use bedrock mask to identify soil vs bedrock
        soil_mask = ~bedrock_mask_extended
        # Apply specified damping to soil elements
        zeta_grid[soil_mask] = config.damping_zeta
        # Apply fixed 0.75% damping to bedrock elements
        if np.any(bedrock_mask_extended):
            zeta_grid[bedrock_mask_extended] = 0.0075
    else:
        # For other methods, use uniform damping
        zeta_grid.fill(config.damping_zeta)

    # Save damping zeta grid array
    np.savetxt(f"{output_dir}/damping_zeta_grid.txt", zeta_grid)

    # Get solver information and print it
    solver_info = get_solver_info(config)
    solver_type_str = solver_info["solver_type"]
    if solver_info["mumps_parallel_procs"]:
        solver_type_str += f" (parallel, {solver_info['mumps_parallel_procs']} procs)"
    print(f"  Solver: {solver_type_str}")
    if solver_info["mumps_icntl"]:
        print(f"  MUMPS ICNTL parameters: {solver_info['mumps_icntl']}")
    print(f"  Domain: {Lx}m x {Lz}m")
    ndivx = int(Lx / dx)
    ndivy = int(Lz / dx)  # Using dx since hx = hy in config
    print(f"  Elements: {ndivx} x {ndivy} = {ndivx * ndivy}")
    print(f"  Time step: {config.dt}s, Duration: {config.duration}s")
    print(f"  Motion frequency: {config.motion_freq} Hz")
    print(f"  Damping method: {config.damping_method}")
    print(f"  Recording center nodes at Y positions: {config.center_node_y_positions}")

    rho = np.ones_like(Vs_extended) * 2000.0
    nu = np.ones_like(Vs_extended) * 0.3

    # Build model
    t_model_start = time.time()
    model_data = build_model_data(config, Vs_extended, rho, nu)
    model_build_time = time.time() - t_model_start

    # Run analysis
    t_analysis_start = time.time()
    print(f"[{case_type}] Running OpenSees for {task_id} -> {output_dir}")
    result = run_opensees_analysis(config, model_data, task_id, output_dir)
    analysis_time = time.time() - t_analysis_start

    # Calculate total time and breakdown
    total_time = time.time() - t0

    # Save timing information to task-specific CSV file
    timing_file = Path(f"results/timing_data_task_{index}.csv")
    timing_file.parent.mkdir(parents=True, exist_ok=True)

    # Each task writes its own file with header and data
    with open(timing_file, "w", newline="") as f:
        writer = csv.writer(f)

        # Write header
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
                "model_build_time_sec",
                "analysis_time_sec",
                "status",
            ]
        )

        # Write timing data
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
    p = argparse.ArgumentParser(description="Run Sensitivity Analysis Experiment.")
    p.add_argument(
        "--index",
        type=int,
        help="Parameter combination index (0-79). If not provided, uses SLURM_ARRAY_TASK_ID",
    )
    return p.parse_args()


if __name__ == "__main__":
    # Change to the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    program_start = time.time()

    # SLURM-aware setup
    _configure_slurm_environment()
    _install_sigterm_handler()

    args = _parse_args()

    # Get index from args or environment
    idx = args.index
    if idx is None:
        env_idx = os.getenv("SLURM_ARRAY_TASK_ID")
        if env_idx is not None and env_idx != "":
            try:
                idx = int(env_idx)
            except ValueError:
                print(f"Invalid SLURM_ARRAY_TASK_ID={env_idx!r}", file=sys.stderr)
                sys.exit(2)

    if idx is None:
        print("Error: --index or SLURM_ARRAY_TASK_ID must be provided", file=sys.stderr)
        sys.exit(1)

    run_case(idx)
    print(f"[program] Total wall time: {_fmt_hms(time.time() - program_start)}")
    sys.exit(0)
