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
from typing import TYPE_CHECKING, Dict, Mapping, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import interp1d

if TYPE_CHECKING:
    from seiskit.config import AnalysisConfig

try:
    import seaborn as sns

    SEABORN_AVAILABLE = True
except ImportError:
    sns = None  # type: ignore[assignment]
    SEABORN_AVAILABLE = False

from seiskit.damping import (
    compute_average_damping_harmonic,
    compute_damping_from_Q,
    compute_quality_factor,
)
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
TimeSeriesMulti = Tuple[np.ndarray, np.ndarray]  # (time, acceleration_columns) for surface data
ModelResult = Dict[
    str, Union[TimeSeries, TimeSeriesMulti]
]  # e.g., {"base": TimeSeries, "top": TimeSeries, "surface": TimeSeriesMulti}
DataSet = Dict[str, ModelResult]  # e.g., {"SPECFEM": ModelResult, "PLAXIS": ModelResult}
PathType = Union[str, Path, PathLike]

# [IMPROVEMENT] Centralized styling for consistent plots using colorblind-friendly palette.
if SEABORN_AVAILABLE and sns is not None:
    # Use seaborn's colorblind-friendly palette
    COLORBLIND_PALETTE = sns.color_palette("colorblind", as_cmap=False)
    COLORBLIND_COLORS = [
        f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}" for r, g, b in COLORBLIND_PALETTE
    ]
    MODEL_COLORS = {
        "My_New_Run": COLORBLIND_COLORS[0],
        "PLAXIS": COLORBLIND_COLORS[1],
        "OpenSeesPy_Prev": COLORBLIND_COLORS[2],
        # Add more models here for consistent colors
    }
    # A cycle for any models not in the map above
    FALLBACK_COLORS = itertools.cycle(COLORBLIND_COLORS)
else:
    # Fallback to colorblind-friendly colors if seaborn not available
    # These are approximate colorblind-friendly colors
    COLORBLIND_COLORS = [
        "#0173b2",
        "#de8f05",
        "#029e73",
        "#cc78bc",
        "#56b4e9",
        "#ece133",
        "#0072b2",
        "#d55e00",
    ]
    MODEL_COLORS = {
        "My_New_Run": COLORBLIND_COLORS[0],
        "PLAXIS": COLORBLIND_COLORS[1],
        "OpenSeesPy_Prev": COLORBLIND_COLORS[2],
    }
    FALLBACK_COLORS = itertools.cycle(COLORBLIND_COLORS)


def load_datasets(config: Mapping[str, Mapping[str, PathType]]) -> DataSet:
    """Loads multiple datasets from file paths defined in a config dictionary.

    Args:
        config: A mapping where keys are model names (e.g., "PLAXIS")
                and values are mappings from location ("base", "top", "surface") to a
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
            if location in ("base", "top", "surface"):
                try:
                    data = np.loadtxt(file_path)
                    # Handle empty files or 1D arrays
                    if data.size == 0:
                        raise ValueError(f"File {file_path} is empty")
                    if data.ndim == 1:
                        raise ValueError(
                            f"File {file_path} contains 1D data (expected 2D with time and values)"
                        )
                    if location == "surface" and data.shape[1] > 2:
                        # For surface data with multiple nodes, return (time, all_accel_columns)
                        datasets[model_name][location] = (data[:, 0], data[:, 1:])
                    else:
                        # For base/top data or single-column surface data
                        datasets[model_name][location] = (data[:, 0], data[:, 1])
                except ValueError as e:
                    raise ValueError(
                        f"Error loading {file_path}: {e}. File may be empty or malformed."
                    ) from e
    return datasets


def _resample_and_calculate_difference(reference: TimeSeries, comparison: TimeSeries) -> TimeSeries:
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
        subplot_titles=("Top (Surface) Acceleration", "Base Acceleration"),
        y_axis_title="Acceleration ($m/s^2$)",
    )

    # Create consistent color and line style mapping for each model name
    line_styles = ["dash", "dot", "dashdot"]
    fallback_colors = COLORBLIND_COLORS

    # Get all model names (including reference)
    all_model_names = list(datasets.keys())

    # Create consistent mappings
    model_styles = {}
    for i, model_name in enumerate(all_model_names):
        color = MODEL_COLORS.get(model_name, fallback_colors[i % len(fallback_colors)])
        line_style = line_styles[i % len(line_styles)] if model_name != reference_name else "solid"
        model_styles[model_name] = {"color": color, "line_style": line_style}

    # Plot top (surface) at row 1, base at row 2
    location_to_row = {"top": 1, "base": 2}
    for location in ["top", "base"]:
        i = location_to_row[location]
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
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5))

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


def plot_stacked_acceleration(
    datasets: DataSet,
    data_config: Mapping[str, Mapping[str, PathType]],
    scale_factor: float = 5.0,
    vertical_spacing: float = 2.5,
    alpha: float = 0.3,
) -> None:
    """
    Plot stacked acceleration time histories for multiple surface nodes using matplotlib.

    This function creates individual matplotlib plots for each model, saved in the same
    folder as the data. Each plot shows surface node accelerations stacked vertically
    with gray color and small alpha.

    Args:
        datasets: The dictionary of loaded model data.
        data_config: The original data configuration mapping model names to file paths.
        scale_factor: Scaling factor for acceleration amplitudes (default: 5.0).
        vertical_spacing: Vertical spacing between stacked traces (default: 2.5).
        alpha: Transparency level for the lines (default: 0.3).

    Returns:
        None
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is not installed. Skipping plot generation.")
        return

    # Plot each model's surface acceleration
    for model_name, model_data in datasets.items():
        if "surface" not in model_data:
            continue

        time, accel = model_data["surface"]

        # Get the output directory from the data config
        if model_name not in data_config or "surface" not in data_config[model_name]:
            continue

        surface_file_path = Path(data_config[model_name]["surface"])
        output_dir = surface_file_path.parent
        output_file = output_dir / f"{model_name}_surface_nodes_acceleration_stacked.png"

        # Create the plot
        plt.figure(figsize=(12, 8))

        # Check if accel is 2D (multiple nodes) or 1D (single node)
        if accel.ndim == 2 and accel.shape[1] > 1:
            # Multiple surface nodes - plot each one
            num_nodes = accel.shape[1]
            vertical_offset = 0.0

            for i in range(num_nodes):
                # Scale and offset the acceleration: accel * scale + offset
                scaled_accel = accel[:, i] * scale_factor + vertical_offset

                plt.plot(
                    scaled_accel,  # X-axis: acceleration + offset
                    time,  # Y-axis: time
                    color="gray",
                    alpha=alpha,
                    linewidth=0.8,
                )

                # Update vertical offset for next node
                vertical_offset += vertical_spacing
        else:
            # Single surface node
            if accel.ndim == 2:
                accel = accel[:, 0]  # Take first column if 2D

            # Scale the acceleration
            scaled_accel = accel * scale_factor

            plt.plot(
                scaled_accel,  # X-axis: acceleration
                time,  # Y-axis: time
                color="gray",
                alpha=alpha,
                linewidth=0.8,
            )

        # Set up the plot
        plt.title(f"{model_name} - Surface Nodes Acceleration X (Stacked and Scaled)")
        plt.xlabel("Acceleration + Offset (m/s²)")
        plt.ylabel("Time (s)")
        plt.grid(True, alpha=0.3)

        # Set time axis to go from 0 (bottom) to 15 (top)
        plt.ylim(0, 15)

        # Save the plot
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Stacked acceleration plot saved to {output_file}")


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
        freq_model, tf_model = TTF(top_accel, base_accel, dt=base_time[1] - base_time[0], dz=dz)

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
        legend=dict(orientation="h", yanchor="top", y=-0.1, xanchor="center", x=0.5),
    )

    fig.write_html(str(output_path))
    print(f"Transfer function plot saved to {output_path}")
    if show_fig:
        fig.show()


def get_damping_zeta_grid(
    Vs_extended: np.ndarray,
    damping_method: str,
    Lx: float,
    Lz: float,
    dx: float,
    dz: float,
    bedrock_mask: np.ndarray | None = None,
    damping_zeta: float | None = None,
    damping_freqs: tuple[float, float] | None = None,
    damping_f_target: float | None = None,
    config: "AnalysisConfig | None" = None,
) -> np.ndarray:
    """
    Compute damping ratio (zeta) grid for each element. Same logic as plot_damping_realization.

    Returns:
        zeta_grid: 2D array (nz, nx) of damping ratios.
    """
    if config is not None:
        damping_zeta = damping_zeta if damping_zeta is not None else config.damping_zeta
        damping_freqs = damping_freqs if damping_freqs is not None else config.damping_freqs
        damping_f_target = (
            damping_f_target if damping_f_target is not None else config.damping_f_target
        )
    else:
        damping_zeta = damping_zeta if damping_zeta is not None else 0.02
        damping_f_target = damping_f_target if damping_f_target is not None else 0.75

    nz, nx = Vs_extended.shape
    zeta_grid = np.zeros_like(Vs_extended)

    if damping_method == "global_avg":
        if bedrock_mask is None:
            raise ValueError(
                "bedrock_mask is required for 'global_avg' damping method. "
                "Provide bedrock_mask from create_vs_realization() for accurate bedrock/soil identification."
            )
        if bedrock_mask.shape != (nz, nx):
            raise ValueError(
                f"bedrock_mask shape {bedrock_mask.shape} != Vs_extended shape {(nz, nx)}"
            )
        soil_mask = ~bedrock_mask
        soil_Vs = Vs_extended[soil_mask]
        if len(soil_Vs) > 0:
            Q_values_soil = [compute_quality_factor(vs) for vs in soil_Vs]
            avg_damping_soil = compute_average_damping_harmonic(Q_values_soil)
            zeta_grid[soil_mask] = avg_damping_soil
        if np.any(bedrock_mask):
            bedrock_Vs = 1500.0
            Q_bedrock = compute_quality_factor(bedrock_Vs)
            xi_bedrock = compute_damping_from_Q(Q_bedrock)
            zeta_grid[bedrock_mask] = xi_bedrock

    elif damping_method == "elemental_varying":
        for i in range(nz):
            for j in range(nx):
                vs = Vs_extended[i, j]
                Q = compute_quality_factor(vs)
                xi = compute_damping_from_Q(Q)
                zeta_grid[i, j] = xi
    elif damping_method == "elemental_mass_only":
        for i in range(nz):
            for j in range(nx):
                vs = Vs_extended[i, j]
                Q = compute_quality_factor(vs)
                xi = compute_damping_from_Q(Q)
                zeta_grid[i, j] = xi
    elif damping_method == "uniform":
        zeta_grid.fill(damping_zeta)
    elif damping_method == "uniform_soil_only":
        if bedrock_mask is None:
            raise ValueError("bedrock_mask is required for 'uniform_soil_only' damping method.")
        if bedrock_mask.shape != (nz, nx):
            raise ValueError(
                f"bedrock_mask shape {bedrock_mask.shape} != Vs_extended shape {(nz, nx)}"
            )
        soil_mask = ~bedrock_mask
        zeta_grid[soil_mask] = damping_zeta
        if np.any(bedrock_mask):
            zeta_grid[bedrock_mask] = 0.0075
    else:
        raise ValueError(f"Unknown damping method: {damping_method}")

    return zeta_grid


def plot_damping_realization(
    Vs_extended: np.ndarray,
    damping_method: str,
    Lx: float,
    Lz: float,
    dx: float,
    dz: float,
    save_path: PathType,
    title: str | None = None,
    bedrock_mask: np.ndarray | None = None,
    damping_zeta: float | None = None,
    damping_freqs: tuple[float, float] | None = None,
    damping_f_target: float | None = None,
    config: "AnalysisConfig | None" = None,
) -> None:
    """
    Plot damping zeta values for each element based on the damping method.

    This function matches the damping logic used in _apply_damping() to ensure
    the plot accurately represents what damping is actually applied in the analysis.

    Args:
        Vs_extended: 2D array of Vs values (nz, nx)
        damping_method: Damping method ("global_avg", "elemental_varying", "elemental_mass_only", "uniform", "uniform_soil_only")
        Lx: Domain width [m]
        Lz: Domain height [m]
        dx: Horizontal grid spacing [m]
        dz: Vertical grid spacing [m]
        save_path: Path to save the plot
        title: Optional title for the plot
        bedrock_mask: Boolean mask of shape (nz, nx) where True indicates bedrock.
                     Required for "global_avg" and "uniform_soil_only" damping methods to accurately identify bedrock vs soil.
        damping_zeta: Damping ratio for uniform and uniform_soil_only damping methods.
                     If None and config is provided, uses config.damping_zeta. Otherwise defaults to 0.02.
        damping_freqs: Tuple of (f1, f2) frequencies for Rayleigh damping.
                      If None and config is provided, uses config.damping_freqs.
        damping_f_target: Target frequency for mass-only damping.
                         If None and config is provided, uses config.damping_f_target. Otherwise defaults to 0.75.
        config: Optional AnalysisConfig object. If provided, its damping parameters will be used
                instead of individual damping_zeta, damping_freqs, and damping_f_target parameters.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is not installed. Skipping plot generation.")
        return

    zeta_grid = get_damping_zeta_grid(
        Vs_extended,
        damping_method,
        Lx,
        Lz,
        dx,
        dz,
        bedrock_mask=bedrock_mask,
        damping_zeta=damping_zeta,
        damping_freqs=damping_freqs,
        damping_f_target=damping_f_target,
        config=config,
    )

    # Plot the damping realization
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))

    # Use provided domain dimensions for plot extent
    # Note: These should match nx*dx and nz*dz, but we use the provided values
    # to ensure consistency with the caller's expectations
    extent = (0, Lx, Lz, 0)

    # Determine color scale limits
    vmin = zeta_grid.min()
    vmax = zeta_grid.max()

    # Use a colorblind-friendly colormap suitable for damping values
    if SEABORN_AVAILABLE and sns is not None:
        # Use seaborn's colorblind-friendly colormap
        cmap = sns.color_palette("rocket", as_cmap=True)
    else:
        # Fallback to a colorblind-friendly matplotlib colormap
        # Use modern matplotlib API: plt.colormaps["viridis"]
        cmap = plt.colormaps["viridis"]

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


def plot_realization(
    Vs_1D_profile: np.ndarray,
    Vs_realization: np.ndarray,
    Lx: float,
    Lz: float,
    dx: float,
    dz: float,
    save_path: Optional[PathType] = None,
    title: Optional[str] = None,
    bedrock_mask: Optional[np.ndarray] = None,
) -> None:
    """
    Plot the Vs realization with a color scale focused on the soil layer.
    Bedrock values are colored distinctly to highlight the soil variability.

    Args:
        Vs_1D_profile: 1D array of shear wave velocities defining the initial layers.
        Vs_realization: 2D array of Vs values (nz, nx) representing the realization.
        Lx: Domain width [m]
        Lz: Domain height [m]
        dx: Horizontal grid spacing [m]
        dz: Vertical grid spacing [m]
        save_path: Optional path to save the plot. If None, displays the plot.
        title: Optional title for the plot.
        bedrock_mask: Optional boolean mask of shape (nz, nx) where True indicates bedrock.
                     If provided, uses this mask to accurately identify soil vs bedrock for
                     color scale calculation. If None, uses Vs < Vs2 heuristic.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is not installed. Skipping plot generation.")
        return

    nz, nx = Vs_realization.shape

    # Validate bedrock_mask if provided
    if bedrock_mask is not None:
        if bedrock_mask.shape != (nz, nx):
            raise ValueError(
                f"bedrock_mask shape {bedrock_mask.shape} != Vs_realization shape {(nz, nx)}"
            )
        if bedrock_mask.dtype != bool:
            bedrock_mask = bedrock_mask.astype(bool)

    # Determine Vs1 and Vs2 from the profile
    Vs_unique = np.unique(Vs_1D_profile)
    _, Vs2 = Vs_unique[0], Vs_unique[1]

    # Isolate the Vs values of the soil layer
    if bedrock_mask is not None:
        # Use bedrock_mask for accurate soil identification
        soil_mask = ~bedrock_mask
        soil_vs_values = Vs_realization[soil_mask]
    else:
        # Fallback to Vs < Vs2 heuristic
        soil_vs_values = Vs_realization[Vs_realization < Vs2]

    # Determine the min and max for the color bar
    vmin = soil_vs_values.min()
    vmax = soil_vs_values.max()

    # Get a colormap and set a specific color for values > vmax (the bedrock)
    cmap = plt.colormaps.get_cmap("viridis_r").copy()
    cmap.set_over("gray")  # Bedrock will be colored gray

    # Plotting the result
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))

    # Compute physical extents from grid spacing to ensure dx/dz are honored
    nz, nx = Vs_realization.shape
    computed_Lx = nx * dx
    computed_Lz = nz * dz
    assert np.isclose(computed_Lx, Lx) and np.isclose(computed_Lz, Lz), (
        f"Computed Lx/Lz do not match provided Lx/Lz: {computed_Lx} m != {Lx} m, {computed_Lz} m != {Lz} m"
    )

    # If provided Lx/Lz differ noticeably, prefer computed values for accurate scaling
    extent = (0, computed_Lx, computed_Lz, 0)

    im = ax.imshow(
        Vs_realization,
        extent=extent,
        aspect="auto",
        cmap=cmap,  # Use the modified colormap
        interpolation="nearest",
        vmin=vmin,  # Set the minimum for the color scale
        vmax=vmax,  # Set the maximum for the color scale
    )

    # Add 'extend' to the colorbar to show there are values beyond its max
    cbar = fig.colorbar(im, ax=ax, extend="max")
    cbar.set_label("Soil $V_s$ (m/s)", fontsize=12)
    # Add label for the extension showing Vs2 value
    cbar.ax.text(
        0.5,
        1.15,
        f"$V_{{s2}}$ = {Vs2:.0f} m/s",
        transform=cbar.ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
    )

    ax.set_xlabel("Distance (m)", fontsize=12)
    ax.set_ylabel("Depth (m)", fontsize=12)
    ax.set_title(
        title if title is not None else "Optimized 2D $V_s$ Realization (Soil-Focused Color Scale)",
        fontsize=14,
    )
    ax.grid(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    plt.close()


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

    plot_stacked_acceleration(
        datasets=all_data,
        data_config=DATA_CONFIG,
        vertical_spacing=5.0,
    )
