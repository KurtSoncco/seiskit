"""
Sensitivity analysis for spatially variable soil properties (Local version).
In this file, we will only create the 1D profile cases, no variability.

In this case, we will try:
- 2 different values of Vs1: 100, 360 m/s
- 2 different values of thickness: 14, 100 m
- 1 value of Vs2 = 1500 m/s, only extend it 10 m from the Vs1 profile
- 1 value of motion_freq = 3.0 Hz (Ricker wave central frequency)
- Duration: 50.0 seconds if f0 < 1 Hz, else 30.0 seconds
- Damping frequencies = (min(f0, 3.0), 10.0) Hz (uses f0 or 3Hz, whichever is smaller)
- 1 value of Damping method: "global_avg"
- Boundary condition type = "2D"
- Record center nodes at surface and base after 2 meters of the bottom (2 m from the bottom)
- Element type = "4node"
- Solver type = "Mumps"

Total number of combinations: 2 * 2 = 4
Results saved in 1D_results/

This is a local version that can run without SLURM.
By default, runs all cases sequentially. Use --index N to run a specific case, or --single to run only one case.
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
from seiskit.plot_results import plot_damping_realization, plot_realization
from seiskit.solver_utils import get_solver_info


def _configure_slurm_environment() -> None:
    """Configure threading and report SLURM context when running under SLURM."""
    # Set threading environment variables to 1 (help desk recommendation)
    # This prevents threading issues and ensures single-threaded execution
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if not os.getenv(var):
            os.environ[var] = "1"

    job_id = os.getenv("SLURM_JOB_ID", "-")
    array_id = os.getenv("SLURM_ARRAY_JOB_ID", job_id)
    task_id = os.getenv("SLURM_ARRAY_TASK_ID", "-")
    node = os.getenv("SLURMD_NODENAME", os.uname().nodename)
    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK", "-")
    print(
        f"[slurm] job_id={job_id} array_id={array_id} task_id={task_id} node={node} cpus={slurm_cpus}"
    )


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


def create_1D_profile(
    Vs1: float, Vs2: float, thickness: float, dz_1D: float, bedrock_depth: float = 10.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a 1D profile for 1D site response analysis.

    Args:
        Vs1: shear wave velocity of the soil layer
        Vs2: shear wave velocity of the bedrock layer
        thickness: thickness of the soil layer
        dz_1D: vertical spacing for 1D profile
        bedrock_depth: depth of bedrock extension (default: 10.0 m)

    Returns:
        Vs_realization: 2D array of shear wave velocities. Shape: (layer_1_count + layer_2_count, 1)
        bedrock_mask: 2D array of bedrock mask. Shape: (layer_1_count + layer_2_count, 1)
    """
    layer_1_count = int(thickness / dz_1D)
    layer_2_count = int(bedrock_depth / dz_1D)
    Vs_realization = np.array([Vs1] * layer_1_count + [Vs2] * layer_2_count)
    bedrock_mask = np.array([False] * layer_1_count + [True] * layer_2_count)

    # Vs_realization is a 2D array of shape (layer_1_count + layer_2_count, 1)
    Vs_realization = Vs_realization.reshape(-1, 1)
    bedrock_mask = bedrock_mask.reshape(-1, 1)

    assert Vs_realization.shape == (layer_1_count + layer_2_count, 1)
    assert bedrock_mask.shape == (layer_1_count + layer_2_count, 1)

    return Vs_realization, bedrock_mask


def run_case(index: int = 0):
    """
    Run a single case given the index.
    """
    t0 = time.time()

    # Fixed parameters
    case_type = "2x2_4node"
    element_type = "4node"
    dx, dz = 2.0, 2.0  # 2x2 discretization
    Vs2 = 1500.0  # Layer 2 (bedrock) velocity
    dz_1D = 2.0  # Vertical spacing for 1D profile
    bedrock_depth = 10.0  # Bedrock extension depth in meters
    motion_freq = 3.0  # Fixed at 3.0 Hz (Ricker wave central frequency)
    damping_method_idx = "global_avg"  # "global_avg" or "uniform_soil_only"
    Lx = dz_1D

    # Parameter variations
    Vs1_list = [100, 360]
    thickness_list = [14, 100]  # Everything is a multiple of 2 meters

    total_combinations = len(Vs1_list) * len(thickness_list)

    if index < 0 or index >= total_combinations:
        raise IndexError(
            f"Index {index} is out of range for {total_combinations} tasks "
            f"(valid 0..{total_combinations - 1})."
        )

    # Map index to parameter combination
    # Index structure: index = Vs1_idx * len(thickness_list) + thickness_idx
    # Order: Vs1 -> thickness
    # Each Vs1 value has 2 thicknesses, so total = 2 * 2 = 4 cases
    Vs1_idx = index // len(thickness_list)
    remainder = index % len(thickness_list)
    thickness_idx = remainder
    Vs1 = Vs1_list[Vs1_idx]
    thickness = thickness_list[thickness_idx]

    # Create Vs_profile_1D
    layer_1_count = int(thickness / dz_1D)
    layer_2_count = int(bedrock_depth / dz_1D)
    Vs_profile_1D = np.array([Vs1] * layer_1_count + [Vs2] * layer_2_count)
    Lz = (layer_1_count + layer_2_count) * dz_1D

    # Compute f0
    f0 = Vs1 / (4 * thickness)
    # Duration: 50 sec if f0 < 1 Hz, else 30 sec
    duration = 50.0 if f0 < 1.0 else 30.0
    # Damping frequencies: use min(f0, 3.0) or 3Hz (Ricker wave central freq)
    damping_freq_first = min(f0, motion_freq)  # Use min(f0, 3.0)
    damping_freqs = (damping_freq_first, 10.0)

    # Create output directory with retry logic
    output_dir = (
        f"1D_results/Vs1_{Vs1:.0f}_thickness_{thickness:.0f}_damping_method_{damping_method_idx}"
    )
    max_retries = 5
    for attempt in range(max_retries):
        try:
            os.makedirs(output_dir, exist_ok=True)
            break
        except (OSError, IOError):
            if attempt == max_retries - 1:
                raise
            time.sleep(0.1 * (attempt + 1))

    # Save Vs_profile_1D to file
    np.savetxt(f"{output_dir}/Vs_profile_1D.txt", Vs_profile_1D)

    # Format realization string and task ID
    task_id = f"{case_type}_Vs1{Vs1:.0f}_th{thickness:.0f}_damping_method_{damping_method_idx}"

    print(f"[{case_type}] Starting task {task_id} (index={index})")
    print(f"  Case: {case_type}, Element type: {element_type}")
    print(f"  Layer 1 height: {thickness:.0f} m")
    print(f"  Vs1 = {Vs1} m/s, Vs2 = {Vs2} m/s, Total depth Lz = {Lz} m")
    print(f"  dx={dx} m, dz={dz} m")
    print(f"  Damping method: {damping_method_idx}")
    print(f"  Motion frequency = {motion_freq} Hz")
    print(f"  f0 = {f0:.4f} Hz")
    print(f"  Duration = {duration:.1f} seconds")
    print(f"  Damping frequencies = {damping_freqs} Hz")

    # Generate VS field
    t_field_start = time.time()
    Vs_realization, bedrock_mask = create_1D_profile(Vs1, Vs2, thickness, dz_1D, bedrock_depth)
    field_generation_time = time.time() - t_field_start

    # Save  Vs realization plot
    plot_realization(
        Vs_profile_1D,
        Vs_realization,
        Lx,
        Lz,
        dx,
        dz,
        save_path=f"{output_dir}/Vs_realization.png",
        title=f"Vs Realization (Vs1={Vs1:.0f} m/s, thickness={thickness:.0f} m, damping_method={damping_method_idx})",
        bedrock_mask=bedrock_mask,
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
        Vs_realization,
        damping_method_idx,
        Lx,
        Lz,
        dx,
        dz,
        save_path=f"{output_dir}/Damping_realization.png",
        title=f"Damping Realization (damping_method={damping_method_idx}, Vs1={Vs1:.0f} m/s, thickness={thickness:.0f} m)",
        bedrock_mask=bedrock_mask,
        config=config,
    )

    # Compute and save damping zeta grid array (same logic as plot_damping_realization)
    zeta_grid = np.zeros_like(Vs_realization)
    soil_mask = ~bedrock_mask  # Compute once and reuse

    if damping_method_idx == "global_avg":
        soil_Vs = Vs_realization[soil_mask]
        if len(soil_Vs) > 0:
            Q_values_soil = [compute_quality_factor(vs) for vs in soil_Vs]
            avg_damping_soil = compute_average_damping_harmonic(Q_values_soil)
            zeta_grid[soil_mask] = avg_damping_soil

        if np.any(bedrock_mask):
            Q_bedrock = compute_quality_factor(Vs2)
            zeta_grid[bedrock_mask] = compute_damping_from_Q(Q_bedrock)
    elif damping_method_idx == "uniform_soil_only":
        zeta_grid[soil_mask] = config.damping_zeta
        if np.any(bedrock_mask):
            zeta_grid[bedrock_mask] = 0.0075
    else:
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

    rho = np.ones_like(Vs_realization) * 2000.0
    nu = np.ones_like(Vs_realization) * 0.3

    # Build model
    t_model_start = time.time()
    model_data = build_model_data(config, Vs_realization, rho, nu)
    model_build_time = time.time() - t_model_start

    # Run analysis
    t_analysis_start = time.time()
    print(f"[{case_type}] Running OpenSees for {task_id} -> {output_dir}")
    result = run_opensees_analysis(config, model_data, task_id, output_dir)
    analysis_time = time.time() - t_analysis_start

    # Calculate total time and breakdown
    total_time = time.time() - t0

    # Save timing information to task-specific CSV file
    timing_file = Path(f"1D_results/timing_data_task_{index}.csv")
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
    p = argparse.ArgumentParser(description="Run Sensitivity Analysis Experiment (Local version).")
    p.add_argument(
        "--index",
        type=int,
        default=None,
        help="Run only a specific case by index (0-3). If not provided, run all cases sequentially.",
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

    # Helper function to run a single case with error handling
    def _run_single_case(index: int) -> tuple[bool, str]:
        """Run a single case and return (success, status)."""
        try:
            result = run_case(index)
            success = result == "success" or result.startswith("Finished")
            return success, result
        except Exception as e:
            import traceback

            print(f"[local] Case {index} failed: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return False, str(e)

    if args.index is not None:
        # Run single case
        print(f"[local] Running single case with index={args.index}")
        run_case(args.index)
    else:
        # Default: Run all cases sequentially
        total_combinations = 2 * 2  # 4
        print(f"[local] Running all {total_combinations} cases sequentially...")
        print("[local] This may take a very long time!")

        successful = failed = 0
        for idx in range(total_combinations):
            print(f"\n{'=' * 80}")
            print(f"[local] Running case {idx + 1}/{total_combinations} (index={idx})")
            print(f"{'=' * 80}")
            success, result = _run_single_case(idx)
            if success:
                successful += 1
            else:
                failed += 1
                print(f"[local] Case {idx} completed with status: {result}")

        print(f"\n{'=' * 80}")
        print("[local] All cases completed!")
        print(f"  Successful: {successful}/{total_combinations}")
        print(f"  Failed: {failed}/{total_combinations}")
        print(f"{'=' * 80}")

    print(f"\n[program] Total wall time: {_fmt_hms(time.time() - program_start)}")
    sys.exit(0)
