import argparse
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from seiskit.analysis import run_opensees_analysis
from seiskit.builder import build_model_data
from seiskit.config import AnalysisConfig
from seiskit.gaussian_field import (
    _extend_profile,
    _generate_vs_variability_field,
    plot_realization,
)
from seiskit.plot_results import (
    FALLBACK_COLORS,
    MODEL_COLORS,
    load_datasets,
    plot_acceleration_comparison,
    plot_stacked_acceleration,
)
from seiskit.ttf.TTF import TTF


def _configure_slurm_environment() -> None:
    """Configure threading and report SLURM context when running under SLURM.

    - Pins BLAS/OpenMP thread counts to SLURM_CPUS_PER_TASK to avoid oversubscription.
    - Prints a short header with SLURM job metadata for easier debugging.
    """
    slurm_cpus = os.getenv("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        for var in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            # Only set if not already explicitly provided
            if not os.getenv(var):
                os.environ[var] = slurm_cpus

    # Minimal context log
    job_id = os.getenv("SLURM_JOB_ID", "-")
    array_id = os.getenv("SLURM_ARRAY_JOB_ID", job_id)
    task_id = os.getenv("SLURM_ARRAY_TASK_ID", "-")
    node = os.getenv("SLURMD_NODENAME", os.uname().nodename)
    cpus = slurm_cpus or "-"
    print(
        f"[slurm] job_id={job_id} array_id={array_id} task_id={task_id} node={node} cpus={cpus}"
    )


def _install_sigterm_handler():
    """Install a SIGTERM handler to flush logs and exit cleanly on preemption/time limit."""

    def _handler(signum, frame):  # noqa: ARG001
        try:
            print("[signal] Received SIGTERM. Attempting graceful shutdown...")
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            # Use code 143 (128+15) which indicates SIGTERM
            os._exit(143)

    try:
        signal.signal(signal.SIGTERM, _handler)
    except Exception:
        # Not fatal if signals are unavailable on the platform
        pass


def _fmt_hms(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    total_seconds = int(seconds)
    return f"{total_seconds // 3600:02d}:{(total_seconds % 3600) // 60:02d}:{total_seconds % 60:02d}"


def run_array_index(index: int):
    """
    Run a single parametric study case determined by the given array index.

    This experiment varies VS spatial variability parameters:
    - rH: 10, 30, 50 (3 values) - correlation length
    - aHV: 1.0 (fixed) - aspect ratio
    - CV: 0.1, 0.2, 0.3 (3 values) - coefficient of variation
    - 5 different seed values for spatial field generation
    - interlayer_seed: 42 (fixed) - seed for wavy boundary (interlayer variability)

    Total: 3 x 3 x 5 = 45 combinations.

    Fixed parameters:
    - Lx_variability = 500m
    - BC_width = 500m
    - Total Lx = 1500m (500 + 2x500)

    Index mapping:
    - Iterates through rH (0-2) → CV (0-2) → seed (0-4)
    - index = rH_idx x (3x5) + CV_idx x 5 + seed_idx
    """
    t0 = time.time()

    # Base case parameters
    Vs_profile_1D = np.array([180.0] * 8 + [1300.0] * 1)
    Lz = 50.0
    dx, dz = 2.5, 2.5
    aHV = 1.0  # Fixed horizontal-to-vertical aspect ratio
    interlayer_seed = 42  # Fixed seed for interlayer (wavy boundary) variability

    # Parameter variations
    rH_values = [10.0, 30.0, 50.0]
    CV_values = [0.1, 0.2, 0.3]
    seed_values = [10, 20, 30, 40, 50]  # 5 different seeds for spatial field

    # Fixed spatial dimensions
    Lx_variability = 500.0
    BC_width = 500.0
    Lx = Lx_variability + 2 * BC_width  # 1500m total

    total_combinations = len(rH_values) * len(CV_values) * len(seed_values)

    if index < 0 or index >= total_combinations:
        raise IndexError(
            f"Index {index} is out of range for {total_combinations} tasks "
            f"(valid 0..{total_combinations - 1})."
        )

    # Map index to parameter combination
    # index = rH_idx × (3×5) + CV_idx × 5 + seed_idx
    rH_idx = index // (len(CV_values) * len(seed_values))
    remaining = index % (len(CV_values) * len(seed_values))
    CV_idx = remaining // len(seed_values)
    seed_idx = remaining % len(seed_values)

    rH = rH_values[rH_idx]
    CV = CV_values[CV_idx]
    seed = seed_values[seed_idx]

    task_id = f"rH{rH:.0f}_CV{CV}_s{seed}"
    output_dir = f"results/rH_{rH:.0f}/CV_{CV}/{task_id}"
    # Create directories with retry logic for file system contention
    max_retries = 5
    for attempt in range(max_retries):
        try:
            os.makedirs(output_dir, exist_ok=True)
            break
        except (OSError, IOError):
            if attempt == max_retries - 1:
                raise
            time.sleep(
                0.1 * (attempt + 1)
            )  # Exponential backoff: 0.1s, 0.2s, 0.3s, 0.4s

    print(f"[run_array_index] Starting task {task_id} (index={index})")
    print(f"  rH = {rH} m, CV = {CV}, seed = {seed}")
    print(
        f"  Lx_variability = {Lx_variability} m, BC_width = {BC_width} m, Total Lx = {Lx} m"
    )

    # Generate VS field with the specified parameters
    print(f"[run_array_index] Generating VS field with seed={seed}")
    print(
        f"[run_array_index] Using interlayer_seed={interlayer_seed} for wavy boundary"
    )
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
        interlayer_seed=interlayer_seed,
    )

    # Extend the profile with BC zones on each side
    # Total width = Lx_variability + 2*BC_width (BC zone on left and right)
    Vs_extended, x_total = _extend_profile(
        Vs_realization,
        Lx=Lx,  # Extended width with BC zones
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
        Ly=Lz,  # Domain height
        Lx=Lx,  # Extended domain width
        hx=dx,  # Element size
        dt=0.01,
        duration=15.0,
        motion_freq=0.75,
        motion_t_shift=1.4,
        damping_zeta=0.0075,
        damping_freqs=(0.75, 2.25),
        boundary_condition_type="2D",
        record_all_surface_nodes=True,
    )

    rho = np.ones_like(Vs_extended) * 2000.0
    nu = np.ones_like(Vs_extended) * 0.3

    # Build model and run analysis
    model_data = build_model_data(config, Vs_extended, rho, nu)

    print(f"[run_array_index] Running OpenSees for {task_id} -> {output_dir}")
    result = run_opensees_analysis(config, model_data, task_id, output_dir)

    elapsed = time.time() - t0
    print(f"[run_array_index] Done: {result} | Wall time: {_fmt_hms(elapsed)}")
    return result


def plot_transfer_functions_with_vsmin(
    datasets: dict,
    output_path: Path,
    Vs_min: float,
    dz: float = 2.5,
    show_fig: bool = False,
) -> None:
    """
    Plot transfer functions with custom Vs_min parameter.

    Args:
        datasets: The dictionary of loaded model data.
        output_path: Path to save the HTML file.
        Vs_min: Minimum Vs value from the field realization.
        dz: Depth increment (default 2.5m).
        show_fig: If True, displays the figure interactively.
    """
    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=("Transfer Functions (Top/Base)",),
    )

    for model_name, model_data in datasets.items():
        if "base" not in model_data or "top" not in model_data:
            continue

        base_time, base_accel = model_data["base"]
        _, top_accel = model_data["top"]

        # Compute transfer function with Vs_min
        freq_model, tf_model = TTF(
            top_accel, base_accel, dt=base_time[1] - base_time[0], dz=dz, Vsmin=Vs_min
        )

        color = MODEL_COLORS.get(model_name, next(FALLBACK_COLORS))
        fig.add_trace(
            go.Scatter(  # type: ignore[call-arg]
                x=freq_model,
                y=tf_model,
                mode="lines",
                name=f"{model_name}",
                line=dict(color=color),
            ),
            row=1,
            col=1,
        )

    fig.update_xaxes(title_text="Frequency (Hz)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Transfer Function Magnitude", type="log", row=1, col=1)
    fig.update_layout(
        height=600,
        width=1200,
        title_text="Transfer Functions Comparison - Parametric Study",
        showlegend=True,
    )

    fig.write_html(str(output_path))
    print(f"Transfer function plot saved to {output_path}")

    if show_fig:
        fig.show()


def analysis_results():
    """Plot analysis results after all tasks are completed.

    Generates:
    1. Transfer function comparison plots grouped by rH, CV, and seed
    2. Surface acceleration comparison plots
    3. Stacked acceleration plots

    Uses a representative Vs_min for transfer function calculation.
    """

    # Generate a sample realization to extract Vs_min
    print("Generating sample realization to extract Vs_min...")
    Vs_profile_1D = np.array([180.0] * 8 + [1300.0] * 1)
    Lz = 50.0
    dx, dz = 2.5, 2.5
    rH, aHV, CV = 30.0, 1.0, 0.2  # Middle values
    seed = 30  # Middle seed value
    interlayer_seed = 42  # Fixed seed for interlayer variability

    np.random.seed(seed)
    Vs_sample, _, _, _ = _generate_vs_variability_field(
        Vs_profile_1D,
        500,
        Lz,
        dx,
        dz,
        rH,
        aHV,
        CV,
        seed=seed,
        interlayer_seed=interlayer_seed,
    )

    # Calculate Vs_min from the sample realization
    Vs_min = np.min(Vs_sample)
    print(f"Using Vs_min = {Vs_min:.2f} m/s for transfer function calculation")

    RESULTS_DIR = Path("./results")

    # Build data configuration for all combinations
    rH_values = [10.0, 30.0, 50.0]
    CV_values = [0.1, 0.2, 0.3]
    seed_values = [10, 20, 30, 40, 50]

    DATA_CONFIG = {}
    for rH in rH_values:
        for CV in CV_values:
            for seed in seed_values:
                model_name = f"rH{rH:.0f}_CV{CV}_s{seed}"
                task_id = f"rH{rH:.0f}_CV{CV}_s{seed}"
                DATA_CONFIG[model_name] = {
                    "base": RESULTS_DIR
                    / f"rH_{rH:.0f}/CV_{CV}/{task_id}/{task_id}/soil_base_dof1_accel.txt",
                    "top": RESULTS_DIR
                    / f"rH_{rH:.0f}/CV_{CV}/{task_id}/{task_id}/soil_top_dof1_accel.txt",
                    "surface": RESULTS_DIR
                    / f"rH_{rH:.0f}/CV_{CV}/{task_id}/{task_id}/surface_nodes_dof1_accel.txt",
                }

    print(f"Loading data from {len(DATA_CONFIG)} configurations...")
    try:
        data = load_datasets(DATA_CONFIG)
    except FileNotFoundError as e:
        print(f"Error: Some result files not found. {e}")
        print("Make sure all array jobs have completed successfully.")
        sys.exit(1)

    # Plot transfer functions with Vs_min
    print("Plotting transfer functions...")
    plot_transfer_functions_with_vsmin(
        datasets=data,
        output_path=Path("transfer_functions_comparison_all.html"),
        Vs_min=Vs_min,
        dz=dz,
    )

    # Plot transfer functions grouped by rH
    for rH in rH_values:
        data_rH = {k: v for k, v in data.items() if f"rH{rH:.0f}" in k}
        if data_rH:
            print(f"Plotting transfer functions for rH={rH}...")
            plot_transfer_functions_with_vsmin(
                datasets=data_rH,
                output_path=Path(f"transfer_functions_rH{rH:.0f}.html"),
                dz=dz,
                Vs_min=Vs_min,
            )

    # Plot transfer functions grouped by CV
    for CV in CV_values:
        data_CV = {k: v for k, v in data.items() if f"CV{CV}" in k}
        if data_CV:
            print(f"Plotting transfer functions for CV={CV}...")
            plot_transfer_functions_with_vsmin(
                datasets=data_CV,
                output_path=Path(f"transfer_functions_CV{CV}.html"),
                dz=dz,
                Vs_min=Vs_min,
            )

    # Plot transfer functions grouped by seed
    for seed in seed_values:
        data_seed = {k: v for k, v in data.items() if f"s{seed}" in k}
        if data_seed:
            print(f"Plotting transfer functions for seed={seed}...")
            plot_transfer_functions_with_vsmin(
                datasets=data_seed,
                output_path=Path(f"transfer_functions_seed{seed}.html"),
                dz=dz,
                Vs_min=Vs_min,
            )

    # Plot acceleration comparison
    print("Plotting acceleration comparisons...")
    plot_acceleration_comparison(
        datasets=data,
        reference_name="",
        output_path=Path("acceleration_time_histories_comparison.html"),
    )

    # Plot stacked acceleration
    print("Plotting stacked accelerations...")
    plot_stacked_acceleration(
        datasets=data,
        data_config=DATA_CONFIG,
        vertical_spacing=2.5,
        scale_factor=4.0,
    )

    print("\nAll plots generated successfully!")


def _parse_args():
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Run Parametric Study. Use --index or SLURM_ARRAY_TASK_ID for array mode."
    )
    p.add_argument("--index", type=int, help="Array index (0-44)")
    p.add_argument(
        "--full", action="store_true", help="Run the full parallel experiment"
    )
    p.add_argument("--plot", action="store_true", help="Generate plots from results")
    return p.parse_args()


if __name__ == "__main__":
    # Change to the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    program_start = time.time()

    # SLURM-aware setup
    _configure_slurm_environment()
    _install_sigterm_handler()

    args = _parse_args()

    # Prefer explicit --index; otherwise check SLURM_ARRAY_TASK_ID
    idx = args.index
    if idx is None:
        env_idx = os.getenv("SLURM_ARRAY_TASK_ID")
        if env_idx is not None and env_idx != "":
            try:
                idx = int(env_idx)
            except ValueError:
                print(f"Invalid SLURM_ARRAY_TASK_ID={env_idx!r}", file=sys.stderr)
                sys.exit(2)

    if idx is not None:
        run_array_index(idx)
        print(f"[program] Total wall time: {_fmt_hms(time.time() - program_start)}")
        sys.exit(0)

    # Non-array modes
    if args.full:
        print("[program] Running full experiment (array mode recommended)...")
        print(
            "For full experiment with 45 tasks, use SLURM array job or run each index manually."
        )
        sys.exit(1)

    elif args.plot:
        print("[program] Generating plots from results...")
        t1 = time.time()
        analysis_results()
        print(f"[program] Plotting wall time: {_fmt_hms(time.time() - t1)}")

    else:
        # Default to help if nothing specified
        print("No action specified. Use one of:")
        print(
            "  --index N        # run one parametric case by array index (or set SLURM_ARRAY_TASK_ID)"
        )
        print("  --full           # info about running full experiment")
        print("  --plot           # plot results")
        sys.exit(1)

    print(f"[program] Total wall time: {_fmt_hms(time.time() - program_start)}")
