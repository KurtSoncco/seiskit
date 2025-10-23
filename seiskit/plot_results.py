"""
Lightweight plotting helpers that wrap Plotly for generating seismic analysis
comparison plots.

These functions accept structured data and paths to generate comparison
and difference plots in HTML format.
"""

import itertools
import sys
from os import PathLike
from pathlib import Path
from typing import Dict, Mapping, Tuple, Union

import numpy as np
from scipy.interpolate import interp1d

from seiskit.ttf.TTF import TTF

# It's good practice to guard optional imports.
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    go = None

    def _plotly_subplots_stub(*args, **kwargs):
        raise ImportError("plotly is not available; install plotly to enable plotting")

    make_subplots = _plotly_subplots_stub
    PLOTLY_AVAILABLE = False

# --- Data Structures ---
# [IMPROVEMENT] Using more specific and cleaner type hints.
TimeSeries = Tuple[np.ndarray, np.ndarray]  # (time, acceleration)
ModelResult = Dict[str, TimeSeries]  # e.g., {"base": TimeSeries, "top": TimeSeries}
DataSet = Dict[
    str, ModelResult
]  # e.g., {"SPECFEM": ModelResult, "PLAXIS": ModelResult}
PathType = Union[str, Path, PathLike]

# [IMPROVEMENT] Centralized styling for consistent plots.
MODEL_COLORS = {
    "My_New_Run": "#1f77b4",  # Muted blue
    "PLAXIS": "#ff7f0e",  # Safety orange
    "OpenSeesPy_Prev": "#2ca02c",  # Cooked asparagus green
    # Add more models here for consistent colors
}
# A cycle for any models not in the map above
FALLBACK_COLORS = itertools.cycle(
    ["#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
)


def load_datasets(config: Mapping[str, Mapping[str, PathType]]) -> DataSet:
    """Loads multiple datasets from file paths defined in a config dictionary.

    Args:
        config: A mapping where keys are model names (e.g., "PLAXIS")
                and values are mappings from location ("base", "top") to a
                filesystem path.

    Returns:
        A dictionary containing the loaded and structured data.

    Raises:
        FileNotFoundError: If a file specified in the config does not exist.
    """
    # [IMPROVEMENT] Using a dictionary comprehension for a more concise and
    # Pythonic implementation.
    datasets: DataSet = {}
    for model_name, paths in config.items():
        datasets[model_name] = {}
        for location, file_path in paths.items():
            if location in ("base", "top"):
                data = np.loadtxt(file_path)
                datasets[model_name][location] = (data[:, 0], data[:, 1])
    return datasets


def _resample_and_calculate_difference(
    reference: TimeSeries, comparison: TimeSeries
) -> TimeSeries:
    """Resamples a comparison series and computes the difference with a reference.

    Uses linear interpolation to align the comparison time series with the
    reference time series before calculating the element-wise difference.

    Args:
        reference: The (time, acceleration) tuple for the reference signal.
        comparison: The (time, acceleration) tuple for the signal to compare.

    Returns:
        A (time, acceleration_difference) tuple aligned with the reference time.
    """
    ref_time, ref_accel = reference
    comp_time, comp_accel = comparison

    interp_func = interp1d(comp_time, comp_accel, bounds_error=False, fill_value=0.0)
    resampled_comp_accel = interp_func(ref_time)
    difference = ref_accel - resampled_comp_accel
    return (ref_time, difference)


def _setup_figure(title: str, subplot_titles: Tuple[str, str], y_axis_title: str):
    """Creates a standard 2-row Plotly figure with shared axes."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.1,
    )
    fig.update_yaxes(title_text=y_axis_title)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_layout(
        height=700,
        width=1000,
        title_text=title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_acceleration_comparison(
    datasets: DataSet, reference_name: str, output_path: Path, show_fig: bool = False
) -> None:
    """Creates and saves a Plotly chart comparing acceleration time histories.

    Args:
        datasets: The dictionary of loaded model data.
        reference_name: Key from `datasets` to use as the reference.
        output_path: The Path object for the output HTML file.
        show_fig: If True, displays the figure interactively.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly is not installed. Skipping plot generation.")
        return

    fig = _setup_figure(
        title="Comparison of Acceleration Time Histories",
        subplot_titles=("Base Acceleration", "Top (Surface) Acceleration"),
        y_axis_title="Acceleration ($m/s^2$)",
    )

    # Create consistent color and line style mapping for each model name
    line_styles = ["dash", "dot", "dashdot"]
    fallback_colors = ["#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#17becf", "#bcbd22"]
    
    # Get all model names (including reference)
    all_model_names = list(datasets.keys())
    
    # Create consistent mappings
    model_styles = {}
    for i, model_name in enumerate(all_model_names):
        color = MODEL_COLORS.get(model_name, fallback_colors[i % len(fallback_colors)])
        line_style = line_styles[i % len(line_styles)] if model_name != reference_name else "solid"
        model_styles[model_name] = {
            "color": color,
            "line_style": line_style
        }

    for i, location in enumerate(["base", "top"], 1):
        # Plot other models first to ensure reference is plotted on top
        for model_name, model_data in datasets.items():
            if model_name == reference_name or location not in model_data:
                continue

            time, accel = model_data[location]
            style = model_styles[model_name]
            
            fig.add_trace(
                go.Scatter(  # type: ignore[call-arg]
                    x=time,
                    y=accel,
                    mode="lines",
                    name=f"{model_name} ({location})",  # Include location for clarity
                    line=dict(dash=style["line_style"], color=style["color"]),
                    showlegend=True,  # Show legend for all traces
                ),
                row=i,
                col=1,
            )

        # Plot the reference model last with a solid line
        if location in datasets.get(reference_name, {}):
            time, accel = datasets[reference_name][location]
            style = model_styles[reference_name]
            
            fig.add_trace(
                go.Scatter(  # type: ignore[call-arg]
                    x=time,
                    y=accel,
                    mode="lines",
                    name=f"{reference_name} ({location})",  # Include location for clarity
                    line=dict(color=style["color"], width=2.5),  # Thicker line for reference
                    showlegend=True,  # Show legend for all traces
                ),
                row=i,
                col=1,
            )

    # Move legend to bottom
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5
        )
    )

    fig.write_html(str(output_path))
    print(f"Comparison plot saved to {output_path}")
    if show_fig:
        fig.show()


def plot_acceleration_differences(
    datasets: DataSet, reference_name: str, output_path: Path, show_fig: bool = False
) -> None:
    """Computes and plots the difference between a reference and other datasets.

    Args:
        datasets: The dictionary of loaded model data.
        reference_name: The key for the dataset to use as the reference.
        output_path: The Path object for the output HTML file.
        show_fig: If True, displays the figure interactively.
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly is not installed. Skipping plot generation.")
        return

    if reference_name not in datasets:
        raise ValueError(f"Reference '{reference_name}' not found in datasets.")

    fig = _setup_figure(
        title=f"Acceleration Difference relative to {reference_name}",
        subplot_titles=("Difference at Base", "Difference at Top (Surface)"),
        y_axis_title="Acceleration Difference ($m/s^2$)",
    )

    reference_data = datasets[reference_name]

    for i, location in enumerate(["base", "top"], 1):
        if location not in reference_data:
            continue

        for model_name, model_data in datasets.items():
            if model_name == reference_name or location not in model_data:
                continue

            diff_time, diff_accel = _resample_and_calculate_difference(
                reference_data[location], model_data[location]
            )
            color = MODEL_COLORS.get(model_name, next(FALLBACK_COLORS))
            fig.add_trace(
                go.Scatter(  # type: ignore[call-arg]
                    x=diff_time,
                    y=diff_accel,
                    mode="lines",
                    name=f"{reference_name} - {model_name} ({location})",
                    line=dict(color=color),
                ),
                row=i,
                col=1,
            )

    fig.write_html(str(output_path))
    print(f"Difference plot saved to {output_path}")
    if show_fig:
        fig.show()


def plot_transfer_functions(
    datasets: DataSet,
    reference_name: str,
    output_path: Path,
    show_fig: bool = False,
    dz: float = 5.0,
) -> None:
    """
    Plot transfer functions between base and top accelerations for each model.
    This functions computes the transfer function from base to top for each model
    and plots them on a log-log scale.

    Args:
        datasets: The dictionary of loaded model data.
        reference_name: The key for the dataset to use as the reference.
        output_path: The Path object for the output HTML file.
        show_fig: If True, displays the figure interactively.
        dz: Depth increment used in the model (default is 5.0).

    Returns:
        None
    """
    if not PLOTLY_AVAILABLE:
        print("Plotly is not installed. Skipping plot generation.")
        return

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

        # Compute transfer function
        freq_model, tf_model = TTF(
            top_accel, base_accel, dt=base_time[1] - base_time[0], dz=dz
        )

        color = MODEL_COLORS.get(model_name, next(FALLBACK_COLORS))
        fig.add_trace(
            go.Scatter(  # type: ignore[call-arg]
                x=freq_model,
                y=tf_model,
                mode="lines",
                name=f"{model_name} TF",
                line=dict(color=color),
            ),
            row=1,
            col=1,
        )

    fig.update_xaxes(title_text="Frequency (Hz)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="Transfer Function Magnitude", type="log", row=1, col=1)
    fig.update_layout(
        height=600,
        width=800,
        title_text="Transfer Functions from Base to Top",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.1,
            xanchor="center",
            x=0.5
        ),
    )

    fig.write_html(str(output_path))
    print(f"Transfer function plot saved to {output_path}")
    if show_fig:
        fig.show()


# --- Main Execution ---
if __name__ == "__main__":
    RESULTS_DIR = Path("./results/Case_1a")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)  # Ensure dir exists

    # 1. Define the configuration for all data sources
    DATA_CONFIG = {
        "My_New_Run": {
            "base": RESULTS_DIR / "soil_base.txt",
            "top": RESULTS_DIR / "soil_top.txt",
        },
        "PLAXIS": {
            "base": RESULTS_DIR / "BaseRight.txt",
            "top": RESULTS_DIR / "SurfaceRight.txt",
        },
        "OpenSeesPy_Prev": {
            "base": RESULTS_DIR / "acc_base_left_hor.out",
            "top": RESULTS_DIR / "acc_surf_left_hor.out",
        },
    }

    # 2. Load all the data into a standardized structure
    try:
        all_data = load_datasets(DATA_CONFIG)

        # Preprocessing can be done here. Example: normalize time to start at 0
        model_to_process = "OpenSeesPy_Prev"
        if model_to_process in all_data:
            for loc in ("base", "top"):
                if loc in all_data[model_to_process]:
                    time, accel = all_data[model_to_process][loc]
                    all_data[model_to_process][loc] = (time - time[0], accel)

    except FileNotFoundError as e:
        print(f"Error: Data file not found. {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Generate the plots
    REFERENCE_MODEL = "My_New_Run"

    plot_acceleration_comparison(
        datasets=all_data,
        reference_name=REFERENCE_MODEL,
        output_path=Path("acceleration_time_histories_comparison.html"),
    )

    plot_acceleration_differences(
        datasets=all_data,
        reference_name=REFERENCE_MODEL,
        output_path=Path("acceleration_time_histories_differences.html"),
    )

    plot_transfer_functions(
        datasets=all_data,
        reference_name=REFERENCE_MODEL,
        output_path=Path("transfer_functions.html"),
    )
