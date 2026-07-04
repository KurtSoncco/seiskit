"""
Sensitivity analysis for spatially variable soil properties on Mesh Refinement Study.

In this case, we will try:
- 1 different values of Vs1: 100 m/s
- 2 different values of thickness: 14, 100 m
- 1 different values of rH: 10 m
- 1 different values of CV: 0.3
- 6 different seeds: 10, 20, 30, 40, 50, 60
- 1 value of aHV = 1
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

- Realization in 2x2, and then re-discretized to 1x1 and 0.5x0.5 (3 cases)

Total number of combinations: 1 * 2 * 1 * 1 * 6 * 3 = 36
"""

import argparse
import csv
import os
import signal
import sys
import time
from pathlib import Path
from typing import Literal, Optional

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


def re_discretization(
    Vs_array: np.ndarray,
    dx_old: float,
    dz_old: float,
    dz_new: float,
    dx_new: Optional[float] = None,
) -> np.ndarray:
    """
    Re-discretize a 2D array of Vs values from old discretization (dx_old, dz_old)
    to finer discretization (dx_new, dz_new) by expanding each element.

    For example, converts 2x2 elements to 1x1 elements by expanding each cell
    into 4 smaller cells (2x2 expansion when going from 2m to 1m spacing).

    Parameters
    ----------
    Vs_array : np.ndarray
        Array of shape (nz_old, nx_old) with Vs values at the old discretization.
    dx_old : float
        Old horizontal discretization size (> 0).
    dz_old : float
        Old vertical discretization size (> 0).
    dz_new : float
        New vertical discretization size (> 0).
    dx_new : float, optional
        New horizontal discretization size (> 0). If None, uses dz_new.

    Returns
    -------
    np.ndarray
        Array of shape (nz_new, nx_new) where:
        - nz_new = nz_old * (dz_old / dz_new)
        - nx_new = nx_old * (dx_old / dx_new)
    """
    if not isinstance(Vs_array, np.ndarray):
        raise TypeError(f"Vs_array must be a numpy array, got {type(Vs_array)}")

    if Vs_array.ndim != 2:
        raise ValueError(f"Vs_array must be 2D, got shape {Vs_array.shape} ({Vs_array.ndim}D)")

    if dx_new is None:
        dx_new = dz_new

    if dx_old <= 0 or dx_new <= 0 or dz_old <= 0 or dz_new <= 0:
        raise ValueError(
            f"All discretization sizes must be positive. "
            f"Got dx_old={dx_old}, dx_new={dx_new}, dz_old={dz_old}, dz_new={dz_new}"
        )

    scale_x = dx_old / dx_new
    scale_z = dz_old / dz_new

    tolerance = 1e-9
    scale_x_int = int(round(scale_x))
    scale_z_int = int(round(scale_z))

    if abs(scale_x - scale_x_int) > tolerance:
        raise ValueError(
            f"dx_old/dx_new must be an integer. "
            f"Got dx_old={dx_old}, dx_new={dx_new}, ratio={scale_x:.10f}"
        )
    if abs(scale_z - scale_z_int) > tolerance:
        raise ValueError(
            f"dz_old/dz_new must be an integer. "
            f"Got dz_old={dz_old}, dz_new={dz_new}, ratio={scale_z:.10f}"
        )

    if scale_x_int < 1 or scale_z_int < 1:
        raise ValueError(
            f"Refinement only: dx_new ({dx_new}) and dz_new ({dz_new}) "
            f"must be <= dx_old ({dx_old}) and dz_old ({dz_old})"
        )

    if scale_x_int == 1 and scale_z_int == 1:
        return Vs_array.copy()

    if scale_z_int == 1:
        return np.repeat(Vs_array, repeats=scale_x_int, axis=1)

    if scale_x_int == 1:
        return np.repeat(Vs_array, repeats=scale_z_int, axis=0)

    result = np.repeat(Vs_array, repeats=scale_z_int, axis=0)
    result = np.repeat(result, repeats=scale_x_int, axis=1)

    return result


def run_case(
    case_type: Literal["2x2_4node", "1x1_4node", "0.5x0.5_4node"],
    index: int = 0,
):
    """
    Run a single case given the case type and index.

    Args:
        case_type: Type of discretization case to run:
            - "2x2_4node": 2x2m elements with 4-node elements (baseline)
            - "1x1_4node": 1x1m elements with 4-node elements (re-discretized)
            - "0.5x0.5_4node": 0.5x0.5m elements with 4-node elements (re-discretized)
        index: Parameter combination index (0-based) for:
            - 1 Vs1 value (100 m/s)
            - 2 thickness values (14, 100 m)
            - 1 rH value (10 m)
            - 1 CV value (0.3)
            - 6 seed values
            Total: 1 * 2 * 1 * 1 * 6 = 12 combinations per discretization case

    Returns:
        Result status message
    """
    t0 = time.time()

    # Fixed parameters
    element_type = "4node"
    dx_base, dz_base = 2.0, 2.0  # Base discretization (2x2)
    aHV = 1.0  # anisotropy ratio for the intralayer variability (fixed at 1 per docstring)
    interlayer_seed = 14  # seed for the interlayer variability
    Vs2 = 1500.0  # Layer 2 (bedrock) velocity
    dz_1D = 2.0  # Vertical spacing for 1D profile
    motion_freq = 3.0  # Fixed at 3.0 Hz (Ricker wave central frequency)
    Lx_variability = 500.0
    BC_width = 500.0
    Lx = Lx_variability + 2 * BC_width
    interlayer_amplitude = 2.0 / 2
    damping_method_idx = "global_avg"  # "global_avg" or "uniform_soil_only"

    # Parameter variations (matching docstring)
    Vs1_list = [100]  # 1 value
    thickness_list = [14, 100]  # 2 values
    rH_list = [10]  # 1 value
    CV_list = [0.3]  # 1 value
    seed_values = [10, 20, 30, 40, 50, 60]  # 6 different seeds

    total_combinations = (
        len(Vs1_list) * len(thickness_list) * len(rH_list) * len(CV_list) * len(seed_values)
    )

    if index < 0 or index >= total_combinations:
        raise IndexError(
            f"Index {index} is out of range for {total_combinations} tasks "
            f"(valid 0..{total_combinations - 1})."
        )

    # Map index to parameter combination
    # Index structure: index = Vs1_idx * (2*1*1*6) + thickness_idx * (1*1*6) + rH_idx * (1*6) + CV_idx * (6) + seed_idx
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

    # Determine discretization based on case type
    case_name = case_type  # Use case_type directly as case_name
    if case_type == "2x2_4node":
        dx, dz = dx_base, dz_base
    elif case_type == "1x1_4node":
        dx, dz = dx_base / 2, dz_base / 2  # 1x1 discretization
    elif case_type == "0.5x0.5_4node":
        dx, dz = dx_base / 4, dz_base / 4  # 0.5x0.5 discretization
    else:
        raise ValueError(f"Unknown case type: {case_type}")

    # Create Vs_profile_1D
    bedrock_depth = 10.0  # Bedrock extension depth in meters
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

    # Create output directory
    output_dir = f"results/{case_name}/Vs1_{Vs1:.0f}_thickness_{thickness:.0f}_rH_{rH:.0f}_CV_{CV:.3f}_s{seed}"

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

    # Save Vs_profile_1D to file
    np.savetxt(f"{output_dir}/Vs_profile_1D.txt", Vs_profile_1D)

    # Format realization string and task ID
    realization_idx = seed_idx + 1  # 1-based for s01, s02, etc.
    realization_str = f"s{realization_idx:02d}"
    task_id = f"{case_name}_Vs1{Vs1:.0f}_th{thickness:.0f}_rH{rH:.0f}_CV{CV:.3f}_s{seed}"

    print(f"[{case_name}] Starting task {task_id} (index={index})")
    print(f"  Case: {case_name}, Element type: {element_type}")
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

    # Generate VS field with base discretization (2x2)
    t_field_start = time.time()
    print(f"[{case_name}] Generating VS field with seed={seed}")
    np.random.seed(seed)
    Vs_realization, _, _, _, bedrock_mask_var = _generate_vs_variability_field(
        Vs_profile_1D,
        Lx_variability,
        Lz,
        dx_base,  # Always generate with base discretization
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

    # Apply re-discretization if needed
    t_rediscret_start = time.time()
    rediscretization_time = 0.0
    if case_type != "2x2_4node":
        print(f"[{case_name}] Re-discretizing from {dx_base}x{dz_base} to {dx}x{dz}")
        Vs_realization = re_discretization(Vs_realization, dx_base, dz_base, dz, dx)
        bedrock_mask_var = re_discretization(
            bedrock_mask_var.astype(float), dx_base, dz_base, dz, dx
        ).astype(bool)
        print(
            f"  Shape after re-discretization: {Vs_realization.shape} "
            f"(was {Vs_realization.shape[0] // int(dx_base / dx)}x{Vs_realization.shape[1] // int(dx_base / dx)})"
        )
    rediscretization_time = time.time() - t_rediscret_start

    # Extend the profile with BC zones on each side
    Vs_extended, _ = _extend_profile(Vs_realization, Lx=Lx, dx=dx)

    # Extend bedrock mask similarly
    bedrock_mask_extended, _ = _extend_profile(bedrock_mask_var.astype(float), Lx=Lx, dx=dx)
    bedrock_mask_extended = bedrock_mask_extended.astype(bool)

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
    zeta_grid = np.zeros_like(Vs_extended)
    soil_mask = ~bedrock_mask_extended  # Compute once and reuse

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
    print(f"[{case_name}] Running OpenSees for {task_id} -> {output_dir}")
    result = run_opensees_analysis(config, model_data, task_id, output_dir)
    analysis_time = time.time() - t_analysis_start

    # Calculate total time and breakdown
    total_time = time.time() - t0

    # Save timing information to task-specific CSV file
    timing_file = Path(f"results/timing_data_task_{case_name}_{index}.csv")
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
                "rediscretization_time_sec",
                "model_build_time_sec",
                "analysis_time_sec",
                "status",
            ]
        )

        # Write timing data
        writer.writerow(
            [
                case_name,
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

    # Print timing summary
    REDISCRET_THRESHOLD = 0.001
    print(f"[{case_name}] Done: {result} | Wall time: {_fmt_hms(total_time)}")
    print("  Timing breakdown:")
    print(f"    Field generation: {_fmt_hms(field_generation_time)}")
    if rediscretization_time > REDISCRET_THRESHOLD:
        print(f"    Re-discretization: {_fmt_hms(rediscretization_time)}")
    print(f"    Model building: {_fmt_hms(model_build_time)}")
    print(f"    OpenSees analysis: {_fmt_hms(analysis_time)}")
    print(f"    Total: {_fmt_hms(total_time)}")
    print(f"  Timing data saved to: {timing_file}")

    return result


def _parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Run Discretization Study Experiment.")
    p.add_argument(
        "--case",
        type=str,
        choices=["2x2_4node", "1x1_4node", "0.5x0.5_4node"],
        default=None,
        help="Case type to run. If not provided, runs all cases.",
    )
    p.add_argument(
        "--index",
        type=int,
        default=None,
        help="Parameter combination index (0-11). If not provided, runs all combinations for the specified case(s).",
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

    # Parameter combinations per case: 1 * 2 * 1 * 1 * 6 = 12
    combinations_per_case = 12
    case_types: list[Literal["2x2_4node", "1x1_4node", "0.5x0.5_4node"]] = [
        "2x2_4node",
        "1x1_4node",
        "0.5x0.5_4node",
    ]

    # Get index from args or environment (for SLURM)
    idx = args.index
    if idx is None:
        env_idx = os.getenv("SLURM_ARRAY_TASK_ID")
        if env_idx is not None and env_idx != "":
            try:
                idx = int(env_idx)
            except ValueError:
                print(f"Invalid SLURM_ARRAY_TASK_ID={env_idx!r}", file=sys.stderr)
                sys.exit(2)

    # Helper function to run a single case with error handling
    def _run_single_case(
        case: Literal["2x2_4node", "1x1_4node", "0.5x0.5_4node"], index: int
    ) -> tuple[bool, str]:
        """Run a single case and return (success, status)."""
        try:
            result = run_case(case, index)
            success = result == "success" or result.startswith("Finished")
            return success, result
        except Exception as e:
            import traceback

            print(f"[local] Case {case} index {index} failed: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return False, str(e)

    # Determine which cases and indices to run
    if args.case is not None and idx is not None:
        # Run specific case and index
        if idx < 0 or idx >= combinations_per_case:
            print(
                f"Error: Index {idx} is out of range for {combinations_per_case} combinations "
                f"(valid 0..{combinations_per_case - 1})",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"[local] Running single case: {args.case}, index={idx}")
        run_case(args.case, idx)  # type: ignore[arg-type] - argparse validates choices
    elif args.case is not None:
        # Run all indices for specific case
        print(f"[local] Running all {combinations_per_case} combinations for case: {args.case}")
        print("[local] This may take a very long time!")

        successful = failed = 0
        for i in range(combinations_per_case):
            print(f"\n{'=' * 80}")
            print(f"[local] Running {args.case} case {i + 1}/{combinations_per_case} (index={i})")
            print(f"{'=' * 80}")
            success, result = _run_single_case(args.case, i)  # type: ignore[arg-type] - argparse validates choices
            if success:
                successful += 1
            else:
                failed += 1
                print(f"[local] Case {i} completed with status: {result}")

        print(f"\n{'=' * 80}")
        print(f"[local] All combinations for {args.case} completed!")
        print(f"  Successful: {successful}/{combinations_per_case}")
        print(f"  Failed: {failed}/{combinations_per_case}")
        print(f"{'=' * 80}")
    elif idx is not None:
        # SLURM mode: map global index to case and local index
        total_combinations = len(case_types) * combinations_per_case
        if idx < 0 or idx >= total_combinations:
            print(
                f"Error: Index {idx} is out of range for {total_combinations} total combinations "
                f"(valid 0..{total_combinations - 1})",
                file=sys.stderr,
            )
            sys.exit(1)

        case_idx = idx // combinations_per_case
        local_idx = idx % combinations_per_case
        case_type = case_types[case_idx]
        print(f"[slurm] Global index {idx} -> case: {case_type}, local index: {local_idx}")
        run_case(case_type, local_idx)
    else:
        # Default: Run all cases and all combinations
        total_combinations = len(case_types) * combinations_per_case
        print(
            f"[local] Running all {total_combinations} combinations across {len(case_types)} cases..."
        )
        print("[local] This may take a very long time!")

        successful = failed = 0
        for case_type in case_types:
            for i in range(combinations_per_case):
                print(f"\n{'=' * 80}")
                print(
                    f"[local] Running {case_type} case {i + 1}/{combinations_per_case} "
                    f"(overall {successful + failed + 1}/{total_combinations})"
                )
                print(f"{'=' * 80}")
                success, result = _run_single_case(case_type, i)
                if success:
                    successful += 1
                else:
                    failed += 1
                    print(f"[local] Case {case_type} index {i} completed with status: {result}")

        print(f"\n{'=' * 80}")
        print("[local] All cases completed!")
        print(f"  Successful: {successful}/{total_combinations}")
        print(f"  Failed: {failed}/{total_combinations}")
        print(f"{'=' * 80}")

    print(f"\n[program] Total wall time: {_fmt_hms(time.time() - program_start)}")
    sys.exit(0)
