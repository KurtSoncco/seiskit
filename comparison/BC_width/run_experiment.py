import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from seiskit.analysis import run_opensees_analysis
from seiskit.builder import build_model_data
from seiskit.config import AnalysisConfig
from seiskit.gaussian_field import (  # [seiskit/gaussian_field.py](seiskit/gaussian_field.py)
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


def _fmt_hms(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    total_seconds = int(seconds)
    return f"{total_seconds // 3600:02d}:{(total_seconds % 3600) // 60:02d}:{total_seconds % 60:02d}"


def run_array_index(index: int):
    """
    Run a single BC_width case determined by the given array index.

    This experiment tests the effect of boundary condition width on results.
    We have 2 Lx variability values (800, 100) and 7 BC width values (0, 100, 200, 300, 400, 500, 1000).
    Total: 2 x 7 = 14 combinations.

    Index mapping:
    - index 0-6: Lx_var=800 with BC_width values [0, 100, 200, 300, 400, 500, 1000]
    - index 7-13: Lx_var=100 with BC_width values [0, 100, 200, 300, 400, 500, 1000]
    """
    t0 = time.time()

    # Base case parameters
    Vs_profile_1D = np.array([180.0] * 8 + [1300.0] * 1)
    Lz = 50.0
    dx, dz = 2.5, 2.5
    rH, aHV, CV = 10.0, 2.0, 0.3
    seed = 42

    Lx_variability_values = [800, 100]
    BC_width_values = [0, 100, 200, 300, 400, 500, 1000]

    total_combinations = len(Lx_variability_values) * len(BC_width_values)

    if index < 0 or index >= total_combinations:
        raise IndexError(
            f"Index {index} is out of range for {total_combinations} tasks "
            f"(valid 0..{total_combinations - 1})."
        )

    # Map index to combination
    lx_var_index = index // len(BC_width_values)
    bc_width_index = index % len(BC_width_values)

    Lx_var = Lx_variability_values[lx_var_index]
    BC_width = BC_width_values[bc_width_index]
    Lx = Lx_var + 2 * BC_width

    task_id = f"{Lx_var}_BC_width_{BC_width}"
    output_dir = f"results/Lx_{Lx_var}/BC_width_{BC_width}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"[run_array_index] Starting task {task_id} (index={index})")
    print(f"  Lx_variability = {Lx_var}m, BC_width = {BC_width}m")

    # Always generate an 800m base field for consistency (both cases will use it)
    np.random.seed(seed)
    print(f"[run_array_index] Generating base Vs field with width 800 m (seed={seed})")
    Vs_base, x_coords, z_coords, h_mean = _generate_vs_variability_field(
        Vs_profile_1D,
        800,  # Always generate 800m field for consistency
        Lz,
        dx,
        dz,
        rH,
        aHV,
        CV,
        seed=seed,
    )

    # Extract the desired Lx_var width from the base field
    if Lx_var == 800:
        # Keep the full 800m field
        Vs_realization = Vs_base
    else:  # Lx_var == 100
        # Extract the central 100m region from the 800m field
        center_x = 800 / 2.0
        half_Lx = 100 / 2.0  # Extract 100m central region
        x_start = center_x - half_Lx
        x_end = center_x + half_Lx
        start_idx = int(x_start / dx)
        end_idx = int(x_end / dx)
        Vs_realization = Vs_base[:, start_idx:end_idx]

    # Extend the profile with BC zones on each side
    # Total width = Lx_var + 2*BC_width (BC zone on left and right)
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
        Vs_min: Minimum Vs value from the 800m realization.
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
        title_text="Transfer Functions Comparison with Vs_min from 800m realization",
        showlegend=True,
    )

    fig.write_html(str(output_path))
    print(f"Transfer function plot saved to {output_path}")

    if show_fig:
        fig.show()


def analysis_results():
    """Plot analysis results after all tasks are completed.

    Generates:
    1. Transfer function comparison plots
    2. Surface acceleration comparison plots
    3. Stacked acceleration plots

    Uses Vs_min from the 800m realization for transfer function calculation.
    """

    # Generate the 800m realization to extract Vs_min
    print("Generating 800m realization to extract Vs_min...")
    Vs_profile_1D = np.array([180.0] * 8 + [1300.0] * 1)
    Lz = 50.0
    dx, dz = 2.5, 2.5
    rH, aHV, CV = 10.0, 2.0, 0.3
    seed = 42

    np.random.seed(seed)
    Vs_800_realization, _, _, _ = _generate_vs_variability_field(
        Vs_profile_1D,
        800,
        Lz,
        dx,
        dz,
        rH,
        aHV,
        CV,
        seed=seed,
    )

    # Calculate Vs_min from the full 800m realization
    Vs_min = np.min(Vs_800_realization)
    print(f"Using Vs_min = {Vs_min:.2f} m/s for transfer function calculation")

    RESULTS_DIR = Path("./results")

    # Build data configuration for all combinations
    Lx_variability_values = [800, 100]
    BC_width_values = [0, 100, 200, 300, 400, 500, 1000]

    DATA_CONFIG = {}
    for Lx_var in Lx_variability_values:
        for BC_width in BC_width_values:
            model_name = f"Lx{Lx_var}_BC{BC_width}"
            task_id = f"{Lx_var}_BC_width_{BC_width}"
            DATA_CONFIG[model_name] = {
                "base": RESULTS_DIR
                / f"Lx_{Lx_var}/BC_width_{BC_width}/{task_id}/soil_base_dof1_accel.txt",
                "top": RESULTS_DIR
                / f"Lx_{Lx_var}/BC_width_{BC_width}/{task_id}/soil_top_dof1_accel.txt",
                "surface": RESULTS_DIR
                / f"Lx_{Lx_var}/BC_width_{BC_width}/{task_id}/surface_nodes_dof1_accel.txt",
            }

    print(f"Loading data from {len(DATA_CONFIG)} configurations...")
    try:
        data = load_datasets(DATA_CONFIG)
    except FileNotFoundError as e:
        print(f"Error: Some result files not found. {e}")
        print("Make sure all array jobs have completed successfully.")
        sys.exit(1)

    # Plot transfer functions with Vs_min
    print("Plotting transfer functions with Vs_min...")
    plot_transfer_functions_with_vsmin(
        datasets=data,
        output_path=Path("transfer_functions_comparison.html"),
        Vs_min=Vs_min,
        dz=dz,
    )

    ## Plot trasnfer functions only for 800m case
    data_800 = {k: v for k, v in data.items() if "Lx800" in k}
    print("Plotting transfer functions only for 800m case...")
    plot_transfer_functions_with_vsmin(
        datasets=data_800,
        output_path=Path("transfer_functions_comparison_800m.html"),
        dz=dz,
        Vs_min=Vs_min,
    )
    ## Plot trasnfer functions only for 100m case
    print("Plotting transfer functions only for 100m case...")
    data_100 = {k: v for k, v in data.items() if "Lx100" in k}
    plot_transfer_functions_with_vsmin(
        datasets=data_100,
        output_path=Path("transfer_functions_comparison_100m.html"),
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
        description="Run BC Width experiment. Use --index or SLURM_ARRAY_TASK_ID for array mode."
    )
    p.add_argument("--index", type=int, help="Array index (0-13)")
    p.add_argument(
        "--full", action="store_true", help="Run the full parallel experiment"
    )
    p.add_argument("--plot", action="store_true", help="Generate plots from results")
    return p.parse_args()


if __name__ == "__main__":
    # Change to the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    program_start = time.time()

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
            "For full experiment with 14 tasks, use SLURM array job or run each index manually."
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
            "  --index N        # run one BC_width case by array index (or set SLURM_ARRAY_TASK_ID)"
        )
        print("  --full           # info about running full experiment")
        print("  --plot           # plot results")
        sys.exit(1)

    print(f"[program] Total wall time: {_fmt_hms(time.time() - program_start)}")
