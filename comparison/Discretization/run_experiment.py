"""Discretization comparison experiment.

This experiment compares two discretization cases:
1. 2x2 discretization with 4-node elements (baseline)
2. 1x1 discretization with 4-node elements (using re_discretization)

For each discretization case, three base cases are tested with different layer 1 heights:
- h5m: Layer 1 = 5m (Vs=100 m/s), Layer 2 = 145m (Vs=1500 m/s)
- h75m: Layer 1 = 75m (Vs=100 m/s), Layer 2 = 75m (Vs=1500 m/s)
- h145m: Layer 1 = 145m (Vs=100 m/s), Layer 2 = 5m (Vs=1500 m/s)
Total depth is 150m for all base cases.
"""

import argparse
import csv
import os
import signal
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np

from seiskit.analysis import run_opensees_analysis
from seiskit.builder import build_model_data
from seiskit.config import AnalysisConfig
from seiskit.gaussian_field import (
    _extend_profile,
    _generate_vs_variability_field,
    plot_realization,
)

# Import re_discretization function

# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))
from re_discretization import re_discretization


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
    print(
        f"[slurm] job_id={job_id} array_id={array_id} task_id={task_id} node={node} cpus={cpus}"
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
    return f"{total_seconds // 3600:02d}:{(total_seconds % 3600) // 60:02d}:{total_seconds % 60:02d}"


def run_discretization_case(
    case_type: Literal["2x2_4node", "1x1_4node"],
    index: int = 0,
):
    """
    Run a single discretization comparison case.

    Args:
        case_type: Type of discretization case to run:
            - "2x2_4node": 2x2m elements with 4-node elements (baseline)
            - "1x1_4node": 1x1m elements with 4-node elements (re-discretized)
        index: Parameter combination index (0-based) for:
            - 3 base cases (layer 1 heights: 5m, 75m, 145m)
            - 2 rH_CV combinations (rH=10/CV=0.3, rH=50/CV=0.1)
            - 5 seed values
            Total: 30 combinations (0-29)

    Returns:
        Result status message
    """
    t0 = time.time()

    # Base case parameters - three different layer 1 heights
    # Each base case: Vs1=100 m/s, Vs2=1500 m/s, total depth=150 m
    # Using dz_1D = 5.0 m (default) for profile discretization
    base_cases = [
        {
            "name": "h5m",
            "layer1_height": 5.0,
            "layer1_count": 1,
            "layer2_count": 29,
        },  # 5m layer1, 145m layer2
        {
            "name": "h75m",
            "layer1_height": 75.0,
            "layer1_count": 15,
            "layer2_count": 15,
        },  # 75m layer1, 75m layer2
        {
            "name": "h145m",
            "layer1_height": 145.0,
            "layer1_count": 29,
            "layer2_count": 1,
        },  # 145m layer1, 5m layer2
    ]

    Lz = 150.0  # Total depth: 150 meters
    dx_base, dz_base = 2.0, 2.0  # Base discretization (2x2)
    aHV = 10.0
    interlayer_seed = 42
    Vs1 = 100.0  # Layer 1 velocity
    Vs2 = 1500.0  # Layer 2 (bedrock) velocity
    dz_1D = 5.0  # Vertical spacing for 1D profile

    # Parameter variations for spatial field
    # Only two combinations: (rH=10, CV=0.3) and (rH=50, CV=0.1)
    # 5 realizations (seeds) for each combination
    rH_CV_combinations = [
        (10.0, 0.3),  # rH=10, CV=0.3
        (50.0, 0.1),  # rH=50, CV=0.1
    ]
    seed_values = [10, 20, 30, 40, 50]  # 5 different seeds

    # Fixed spatial dimensions
    Lx_variability = 500.0
    BC_width = 500.0
    Lx = Lx_variability + 2 * BC_width  # 1500m total

    # Total combinations: 3 base cases × 2 rH_CV combinations × 5 seeds = 30
    total_combinations = len(base_cases) * len(rH_CV_combinations) * len(seed_values)

    if index < 0 or index >= total_combinations:
        raise IndexError(
            f"Index {index} is out of range for {total_combinations} tasks "
            f"(valid 0..{total_combinations - 1})."
        )

    # Map index to parameter combination
    # index = base_case_idx × (2 × 5) + combo_idx × 5 + seed_idx
    # index = base_case_idx × 10 + combo_idx × 5 + seed_idx
    base_case_idx = index // (len(rH_CV_combinations) * len(seed_values))
    remainder = index % (len(rH_CV_combinations) * len(seed_values))
    combo_idx = remainder // len(seed_values)
    seed_idx = remainder % len(seed_values)

    base_case = base_cases[base_case_idx]
    rH, CV = rH_CV_combinations[combo_idx]
    seed = seed_values[seed_idx]

    # Create Vs_profile_1D based on selected base case
    Vs_profile_1D = np.array(
        [Vs1] * base_case["layer1_count"] + [Vs2] * base_case["layer2_count"]
    )

    # Determine discretization based on case type
    if case_type == "2x2_4node":
        dx, dz = dx_base, dz_base
        element_type = "4node"
        case_name = "2x2_4node"
    elif case_type == "1x1_4node":
        dx, dz = dx_base / 2, dz_base / 2  # 1x1 discretization
        element_type = "4node"
        case_name = "1x1_4node"
    else:
        raise ValueError(f"Unknown case type: {case_type}")

    task_id = f"{case_name}_{base_case['name']}_rH{rH:.0f}_CV{CV}_s{seed}"
    output_dir = (
        f"results/{case_name}/{base_case['name']}/rH_{rH:.0f}/CV_{CV}/{task_id}"
    )

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

    print(f"[{case_name}] Starting task {task_id} (index={index})")
    print(f"  Case: {case_name}, Element type: {element_type}")
    print(
        f"  Base case: {base_case['name']} (Layer 1 height: {base_case['layer1_height']:.0f} m)"
    )
    print(f"  Vs1 = {Vs1} m/s, Vs2 = {Vs2} m/s, Total depth Lz = {Lz} m")
    print(f"  dx={dx} m, dz={dz} m")
    print(f"  rH = {rH} m, CV = {CV}, seed = {seed}")
    print(
        f"  Lx_variability = {Lx_variability} m, BC_width = {BC_width} m, Total Lx = {Lx} m"
    )

    # Generate VS field with base discretization (2x2)
    t_field_start = time.time()
    print(f"[{case_name}] Generating VS field with seed={seed}")
    np.random.seed(seed)
    Vs_realization, x_coords, z_coords, h_mean = _generate_vs_variability_field(
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
    )
    field_generation_time = time.time() - t_field_start

    # Apply re-discretization if needed (for 1x1 case)
    t_rediscret_start = time.time()
    rediscretization_time = 0.0
    if case_type == "1x1_4node":
        print(f"[{case_name}] Re-discretizing from {dx_base}x{dz_base} to {dx}x{dz}")
        Vs_realization = re_discretization(Vs_realization, dx_base, dz_base, dz)
        print(
            f"  Shape after re-discretization: {Vs_realization.shape} "
            f"(was {Vs_realization.shape[0] * 2}x{Vs_realization.shape[1] * 2})"
        )
    rediscretization_time = time.time() - t_rediscret_start

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
        duration=15.0,
        motion_freq=1.75,
        motion_t_shift=1.4,
        damping_zeta=0.0075,
        damping_freqs=(0.75, 2.25),
        boundary_condition_type="2D",
        record_all_surface_nodes=True,
        element_type=element_type,  # Pass element type to config
    )

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

    # Save timing information to CSV
    timing_file = Path("results/timing_data.csv")
    timing_file.parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists to determine if we need to write header
    file_exists = timing_file.exists()

    with open(timing_file, "a", newline="") as f:
        writer = csv.writer(f)

        # Write header if file is new
        if not file_exists:
            writer.writerow(
                [
                    "case_type",
                    "base_case",
                    "layer1_height_m",
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
                base_case["name"],
                f"{base_case['layer1_height']:.1f}",
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

    print(f"[{case_name}] Done: {result} | Wall time: {_fmt_hms(total_time)}")
    print("  Timing breakdown:")
    print(f"    Field generation: {_fmt_hms(field_generation_time)}")
    if rediscretization_time > 0.001:
        print(f"    Re-discretization: {_fmt_hms(rediscretization_time)}")
    print(f"    Model building: {_fmt_hms(model_build_time)}")
    print(f"    OpenSees analysis: {_fmt_hms(analysis_time)}")
    print(f"    Total: {_fmt_hms(total_time)}")
    print(f"  Timing data saved to: {timing_file}")

    return result


def _parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Run Discretization Comparison Experiment.")
    p.add_argument(
        "--case",
        type=str,
        choices=["2x2_4node", "1x1_4node"],
        required=True,
        help="Case type to run",
    )
    p.add_argument(
        "--index",
        type=int,
        help="Parameter combination index (0-29). If not provided, uses SLURM_ARRAY_TASK_ID",
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

    run_discretization_case(args.case, idx)
    print(f"[program] Total wall time: {_fmt_hms(time.time() - program_start)}")
    sys.exit(0)
