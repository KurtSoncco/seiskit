"""Damping study experiment.

This experiment compares four damping methods:
- Model A: Global Average Damping (harmonic mean of Q values from soil layer only)
- Model B: Elemental Varying Damping (each element gets damping based on its Vs and Q)
- Model C: Elemental Mass-Only Damping (mass-proportional only, per element)
- Model D: Uniform Damping (same damping for all elements: zeta=0.0075, freqs=(0.75, 8.25))

Base case parameters:
- Fixed discretization: 2x2_4node only
- Fixed height: 75m (h75m)
- rH/CV combinations: (rH=10, CV=0.3) and (rH=50, CV=0.1)
- 5 realizations (seeds: 10, 20, 30, 40, 50)
- Fixed frequency: 3.0 Hz

Total combinations: 4 damping methods × 2 rH/CV combinations × 5 seeds = 40 combinations
"""

import argparse
import csv
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

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
    plot_realization,
)
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


def plot_damping_realization(
    Vs_extended: np.ndarray,
    damping_method: str,
    Lx: float,
    Lz: float,
    dx: float,
    dz: float,
    save_path: str,
    title: Optional[str] = None,
):
    """
    Plot damping zeta values for each element based on the damping method.

    Args:
        Vs_extended: 2D array of Vs values (nz, nx)
        damping_method: Damping method ("global_avg", "elemental_varying", "elemental_mass_only", "uniform")
        Lx: Domain width [m]
        Lz: Domain height [m]
        dx: Horizontal grid spacing [m]
        dz: Vertical grid spacing [m]
        save_path: Path to save the plot
    """
    import matplotlib.pyplot as plt

    SOIL_VS_THRESHOLD = 500.0
    nz, nx = Vs_extended.shape

    # Calculate zeta for each element based on damping method
    zeta_grid = np.zeros_like(Vs_extended)

    if damping_method == "global_avg":
        # Calculate harmonic mean Q from soil layer only
        soil_mask = Vs_extended < SOIL_VS_THRESHOLD
        soil_Vs = Vs_extended[soil_mask]

        if len(soil_Vs) > 0:
            Q_values_soil = [compute_quality_factor(vs) for vs in soil_Vs]
            avg_damping_soil = compute_average_damping_harmonic(Q_values_soil)
            # Apply to all soil elements
            zeta_grid[soil_mask] = avg_damping_soil

        # Bedrock elements get bedrock damping
        bedrock_mask = Vs_extended >= SOIL_VS_THRESHOLD
        if np.any(bedrock_mask):
            bedrock_Vs = 1500.0
            Q_bedrock = compute_quality_factor(bedrock_Vs)
            xi_bedrock = compute_damping_from_Q(Q_bedrock)
            zeta_grid[bedrock_mask] = xi_bedrock

    elif damping_method in ["elemental_varying", "elemental_mass_only"]:
        # Each element gets damping based on its Vs
        for i in range(nz):
            for j in range(nx):
                vs = Vs_extended[i, j]
                Q = compute_quality_factor(vs)
                xi = compute_damping_from_Q(Q)
                zeta_grid[i, j] = xi
    elif damping_method == "uniform":
        # Uniform damping: same damping for all elements
        uniform_zeta = 0.0075
        zeta_grid.fill(uniform_zeta)
    else:
        raise ValueError(f"Unknown damping method: {damping_method}")

    # Plot the damping realization
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))

    # Compute physical extents
    computed_Lx = nx * dx
    computed_Lz = nz * dz
    extent = (0, computed_Lx, computed_Lz, 0)

    # Determine color scale limits
    vmin = zeta_grid.min()
    vmax = zeta_grid.max()

    # Use a colormap suitable for damping values
    cmap = plt.colormaps.get_cmap("viridis")

    im = ax.imshow(
        zeta_grid,
        extent=extent,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Damping Ratio $\\zeta$", fontsize=12)

    ax.set_xlabel("Distance (m)", fontsize=12)
    ax.set_ylabel("Depth (m)", fontsize=12)
    ax.set_title(
        title or f"Damping Realization ({damping_method.replace('_', ' ').title()})",
        fontsize=14,
    )
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def run_damping_case(index: int = 0):
    """
    Run a single damping study case.

    Args:
        index: Parameter combination index (0-based) for:
            - 4 damping methods (global_avg, elemental_varying, elemental_mass_only, uniform)
            - 2 rH/CV combinations (rH=10/CV=0.3, rH=50/CV=0.1)
            - 5 seed values (10, 20, 30, 40, 50)
            Total: 40 combinations (0-39)

    Returns:
        Result status message
    """
    t0 = time.time()

    # Fixed parameters
    case_type = "2x2_4node"
    element_type = "4node"
    Lz = 150.0  # Total depth: 150 meters
    dx, dz = 2.0, 2.0  # 2x2 discretization
    aHV = 10.0
    interlayer_seed = 42
    Vs1 = 80.0  # Layer 1 velocity
    Vs2 = 1500.0  # Layer 2 (bedrock) velocity
    dz_1D = 5.0  # Vertical spacing for 1D profile
    layer1_height = 75.0  # Fixed at 75m
    layer1_count = 15  # 75m / 5m = 15 layers
    layer2_count = 15  # 75m / 5m = 15 layers
    motion_freq = 3.0  # Fixed at 3.0 Hz

    # Parameter variations
    damping_methods = [
        "global_avg",
        "elemental_varying",
        "elemental_mass_only",
        "uniform",
    ]
    rH_CV_combinations = [
        (10.0, 0.3),  # rH=10, CV=0.3
        (50.0, 0.1),  # rH=50, CV=0.1
    ]
    seed_values = [10, 20, 30, 40, 50]  # 5 different seeds

    # Fixed spatial dimensions
    Lx_variability = 500.0
    BC_width = 500.0
    Lx = Lx_variability + 2 * BC_width  # 1500m total

    # Total combinations: 4 damping methods × 2 rH/CV combinations × 5 seeds = 40
    total_combinations = len(damping_methods) * len(rH_CV_combinations) * len(seed_values)

    if index < 0 or index >= total_combinations:
        raise IndexError(
            f"Index {index} is out of range for {total_combinations} tasks "
            f"(valid 0..{total_combinations - 1})."
        )

    # Map index to parameter combination
    # index = damping_idx × (2 × 5) + combo_idx × 5 + seed_idx
    damping_idx = index // (len(rH_CV_combinations) * len(seed_values))
    remainder = index % (len(rH_CV_combinations) * len(seed_values))
    combo_idx = remainder // len(seed_values)
    seed_idx = remainder % len(seed_values)

    damping_method = damping_methods[damping_idx]
    rH, CV = rH_CV_combinations[combo_idx]
    seed = seed_values[seed_idx]

    # Create Vs_profile_1D (fixed at 75m height)
    Vs_profile_1D = np.array([Vs1] * layer1_count + [Vs2] * layer2_count)

    # Format realization as s01, s02, etc. with zero-padding
    realization_idx = seed_idx + 1  # 1-based for s01, s02, etc.
    realization_str = f"s{realization_idx:02d}"
    task_id = f"{case_type}_{damping_method}_rH{rH:.0f}_CV{CV}_s{seed}"
    output_dir = f"results/{damping_method}/rH_{rH:.0f}/CV_{CV}/{task_id}"

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

    print(f"[{case_type}] Starting task {task_id} (index={index})")
    print(f"  Case: {case_type}, Element type: {element_type}")
    print(f"  Layer 1 height: {layer1_height:.0f} m (fixed)")
    print(f"  Vs1 = {Vs1} m/s, Vs2 = {Vs2} m/s, Total depth Lz = {Lz} m")
    print(f"  dx={dx} m, dz={dz} m")
    print(f"  Damping method: {damping_method}")
    print(f"  rH = {rH} m, CV = {CV}, seed = {seed}, realization = {realization_str}")
    print(f"  Motion frequency = {motion_freq} Hz")
    print(f"  Lx_variability = {Lx_variability} m, BC_width = {BC_width} m, Total Lx = {Lx} m")

    # Generate VS field
    t_field_start = time.time()
    print(f"[{case_type}] Generating VS field with seed={seed}")
    np.random.seed(seed)
    Vs_realization, x_coords, z_coords, h_mean = _generate_vs_variability_field(
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
    )
    field_generation_time = time.time() - t_field_start

    # Extend the profile with BC zones on each side
    Vs_extended, x_total = _extend_profile(
        Vs_realization,
        Lx=Lx,
        dx=dx,
    )

    # Save realization plot
    plot_realization(
        Vs_profile_1D,
        Vs_extended,
        Lx,
        Lz,
        dx,
        dz,
        save_path=f"{output_dir}/Vs_realization.png",
    )

    # Build analysis config
    config = AnalysisConfig(
        Ly=Lz,
        Lx=Lx,
        hx=dx,  # Element size
        dt=0.01,
        duration=25.0,
        motion_freq=motion_freq,  # Fixed at 3.0 Hz
        motion_t_shift=0.5,
        damping_zeta=0.0075,
        damping_freqs=(0.75, 8.25),
        damping_method=damping_method,  # Set damping method
        damping_f_target=motion_freq,  # For mass-only damping
        boundary_condition_type="2D",
        record_all_surface_nodes=True,
        element_type=element_type,
        solver_type="Mumps",
    )

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

    rho = np.ones_like(Vs_extended) * 2000.0
    nu = np.ones_like(Vs_extended) * 0.3

    # Build model
    t_model_start = time.time()
    model_data = build_model_data(config, Vs_extended, rho, nu)
    model_build_time = time.time() - t_model_start

    # Plot damping realization
    damping_plot_path = f"{output_dir}/damping_realization.png"
    plot_damping_realization(
        Vs_extended,
        damping_method,
        Lx,
        Lz,
        dx,
        dz,
        damping_plot_path,
        title=f"Damping Realization ({damping_method.replace('_', ' ').title()}) rH={rH:.0f} CV={CV:.3f} seed={seed}",
    )

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
                "damping_method",
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
                damping_method,
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
    p = argparse.ArgumentParser(description="Run Damping Study Experiment.")
    p.add_argument(
        "--index",
        type=int,
        help="Parameter combination index (0-39). If not provided, uses SLURM_ARRAY_TASK_ID",
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

    run_damping_case(idx)
    print(f"[program] Total wall time: {_fmt_hms(time.time() - program_start)}")
    sys.exit(0)
