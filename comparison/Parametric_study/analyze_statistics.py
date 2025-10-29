import sys
from itertools import product
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from seiskit.plot_results import load_datasets
from seiskit.ttf.TTF import TTF


def compute_tf_statistics(datasets, dz=2.5, Vs_min=None):
    """
    Compute mean and standard deviation of transfer functions for each rH-CV combination.

    Args:
        datasets: Dictionary of loaded model data
        dz: Depth increment
        Vs_min: Minimum Vs for TTF calculation

    Returns:
        Dictionary with keys like (rH, CV) mapping to statistics dict
    """
    # Group data by rH and CV
    grouped_data = {}

    skipped_realizations = []

    for model_name, model_data in datasets.items():
        if "base" not in model_data or "top" not in model_data:
            skipped_realizations.append((model_name, "missing base/top"))
            continue

        # Extract rH, CV, seed from model name
        # Format: "rH{rH:.0f}_CV{CV}_s{seed}"
        parts = model_name.split("_")
        if len(parts) >= 3:
            rH_str = parts[0].replace("rH", "")
            CV_str = parts[1].replace("CV", "")
            seed_str = parts[2].replace("s", "")

            try:
                rH = float(rH_str)
                CV = float(CV_str)
                seed = int(seed_str)

                # Create key for (rH, CV) combination
                key = (rH, CV)

                if key not in grouped_data:
                    grouped_data[key] = []

                # Validate data shapes and emptiness
                try:
                    base_time, base_accel = model_data["base"]
                    top_time, top_accel = model_data["top"]
                except Exception:
                    skipped_realizations.append((model_name, "invalid base/top tuple"))
                    continue

                if (
                    base_time is None
                    or base_accel is None
                    or top_accel is None
                    or np.size(base_time) == 0
                    or np.size(base_accel) == 0
                    or np.size(top_accel) == 0
                ):
                    skipped_realizations.append((model_name, "empty arrays"))
                    continue

                if len(base_time) < 2:
                    skipped_realizations.append(
                        (model_name, "insufficient base time steps")
                    )
                    continue

                grouped_data[key].append((model_data, seed))

            except (ValueError, IndexError):
                print(f"Warning: Could not parse model name: {model_name}")
                continue

    # Compute statistics for each rH-CV combination
    stats = {}

    for (rH, CV), data_list in grouped_data.items():
        freq_arrays = []
        tf_arrays = []

        for model_data, seed in data_list:
            base_time, base_accel = model_data["base"]
            _, top_accel = model_data["top"]

            # Compute transfer function
            freq, tf = TTF(
                top_accel,
                base_accel,
                dt=base_time[1] - base_time[0],
                dz=dz,
                Vsmin=Vs_min,
            )

            freq_arrays.append(freq)
            tf_arrays.append(tf)

        # Align all arrays to a common frequency grid
        # Use the finest resolution (smallest frequency step)
        min_freq = min(f.min() for f in freq_arrays)
        max_freq = max(f.max() for f in freq_arrays)

        # Find the finest step size
        step_sizes = [np.diff(f).min() for f in freq_arrays if len(f) > 1]
        if len(step_sizes) == 0:
            # Fallback if all arrays are too short
            finest_step = (max_freq - min_freq) / 100
        else:
            finest_step = min(step_sizes)

        # Create common frequency grid
        common_freq = np.arange(min_freq, max_freq + finest_step, finest_step)

        # Interpolate all TFs to common grid
        tf_interpolated = []
        for freq, tf in zip(freq_arrays, tf_arrays):
            # Only interpolate within the frequency range of each realization
            tf_interp = np.interp(
                common_freq,
                freq,
                tf,
                left=np.nan,
                right=np.nan,  # Use NaN outside the range
            )
            tf_interpolated.append(tf_interp)

        # Compute statistics, ignoring NaN values
        tf_matrix = np.array(tf_interpolated)
        mean_tf = np.nanmean(tf_matrix, axis=0)
        std_tf = np.nanstd(tf_matrix, axis=0)

        # Filter out NaN values from the output
        valid_idx = ~np.isnan(mean_tf)
        common_freq = common_freq[valid_idx]
        mean_tf = mean_tf[valid_idx]
        std_tf = std_tf[valid_idx]

        stats[(rH, CV)] = {
            "freq": common_freq,
            "mean": mean_tf,
            "std": std_tf,
            "n_realizations": len(data_list),
        }

    if skipped_realizations:
        print("Warning: skipped some realizations in TF stats due to data issues:")
        for name, reason in skipped_realizations:
            print(f"  - {name}: {reason}")

    return stats


def compute_time_history_statistics(datasets):
    """
    Compute mean and standard deviation of time histories for each rH-CV combination.
    Focuses on central point (surface and base accelerations).

    Args:
        datasets: Dictionary of loaded model data

    Returns:
        Dictionary with keys like (rH, CV) mapping to statistics dict
    """
    # Group data by rH and CV
    grouped_data = {}

    skipped_realizations = []

    for model_name, model_data in datasets.items():
        if "base" not in model_data or "surface" not in model_data:
            skipped_realizations.append((model_name, "missing base/surface"))
            continue

        # Extract rH, CV, seed from model name
        parts = model_name.split("_")
        if len(parts) >= 3:
            rH_str = parts[0].replace("rH", "")
            CV_str = parts[1].replace("CV", "")
            seed_str = parts[2].replace("s", "")

            try:
                rH = float(rH_str)
                CV = float(CV_str)
                seed = int(seed_str)

                key = (rH, CV)

                if key not in grouped_data:
                    grouped_data[key] = []

                # Validate tuple structure and non-emptiness
                try:
                    base_time, base_accel = model_data["base"]
                    surface_time, surface_accel_all = model_data["surface"]
                except Exception:
                    skipped_realizations.append((model_name, "invalid tuples"))
                    continue

                if (
                    np.size(base_time) == 0
                    or np.size(base_accel) == 0
                    or np.size(surface_time) == 0
                    or np.size(surface_accel_all) == 0
                ):
                    skipped_realizations.append((model_name, "empty arrays"))
                    continue

                if len(surface_time) < 2:
                    skipped_realizations.append(
                        (model_name, "insufficient surface time steps")
                    )
                    continue

                grouped_data[key].append((model_data, seed))

            except (ValueError, IndexError):
                print(f"Warning: Could not parse model name: {model_name}")
                continue

    # Compute statistics for each rH-CV combination
    stats = {}

    for (rH, CV), data_list in grouped_data.items():
        surface_time_arrays = []
        surface_accel_arrays = []
        base_time_arrays = []
        base_accel_arrays = []

        for model_data, seed in data_list:
            # Extract central point from surface nodes (middle column)
            if "surface" in model_data:
                surface_time, surface_accel_all = model_data["surface"]

                # Determine central node/column if 2D, otherwise use 1D series
                if hasattr(surface_accel_all, "ndim") and surface_accel_all.ndim == 2:
                    n_nodes = surface_accel_all.shape[1]
                    central_idx = n_nodes // 2 if n_nodes > 1 else 0
                    surface_accel_central = surface_accel_all[:, central_idx]
                else:
                    surface_accel_central = surface_accel_all

                surface_time_arrays.append(surface_time)
                surface_accel_arrays.append(surface_accel_central)

            if "base" in model_data:
                base_time, base_accel = model_data["base"]
                base_time_arrays.append(base_time)
                base_accel_arrays.append(base_accel)

        # Align all arrays to a common time grid
        common_time = None
        surface_mean = None
        surface_std = None

        if surface_time_arrays:
            # Determine common dt as median of all dts to reduce aliasing
            dts = []
            for t in surface_time_arrays:
                if len(t) > 1:
                    dts.append(np.median(np.diff(t)))
            common_timestep = np.median(dts) if dts else 0.01

            min_time = min(t.min() for t in surface_time_arrays)
            max_time = min(
                t.max() for t in surface_time_arrays
            )  # min to handle lengths
            common_time = np.arange(min_time, max_time, common_timestep)

            # Interpolate all accelerations to common time grid
            surface_accel_interpolated = []
            for time, accel in zip(surface_time_arrays, surface_accel_arrays):
                accel_interp = np.interp(common_time, time, accel, left=0, right=0)
                surface_accel_interpolated.append(accel_interp)

            # Compute statistics for surface
            surface_matrix = np.array(surface_accel_interpolated)
            surface_mean = np.mean(surface_matrix, axis=0)
            surface_std = np.std(surface_matrix, axis=0)

        # If no surface data, use base data for time grid
        elif base_time_arrays:
            dts = []
            for t in base_time_arrays:
                if len(t) > 1:
                    dts.append(np.median(np.diff(t)))
            common_timestep = np.median(dts) if dts else 0.01
            min_time = min(t.min() for t in base_time_arrays)
            max_time = min(t.max() for t in base_time_arrays)
            common_time = np.arange(min_time, max_time, common_timestep)

        # Store stats
        if common_time is not None:
            stats[(rH, CV)] = {
                "time": common_time,
                "surface_mean": surface_mean,
                "surface_std": surface_std,
                "base_mean": None,  # Will add base if available
                "base_std": None,
                "n_realizations": len(data_list),
            }

        # Handle base acceleration
        if base_time_arrays and common_time is not None:
            base_accel_interpolated = []
            for time, accel in zip(base_time_arrays, base_accel_arrays):
                accel_interp = np.interp(common_time, time, accel, left=0, right=0)
                base_accel_interpolated.append(accel_interp)

            base_matrix = np.array(base_accel_interpolated)
            base_mean = np.mean(base_matrix, axis=0)
            base_std = np.std(base_matrix, axis=0)

            stats[(rH, CV)]["base_mean"] = base_mean
            stats[(rH, CV)]["base_std"] = base_std

    if skipped_realizations:
        print(
            "Warning: skipped some realizations in time-history stats due to data issues:"
        )
        for name, reason in skipped_realizations:
            print(f"  - {name}: {reason}")

    return stats


def plot_tf_statistics(stats, output_path: Path, show_fig: bool = False):
    """
    Plot transfer function statistics with mean lines and ±1 std regions.

    Args:
        stats: Dictionary of statistics computed by compute_tf_statistics
        output_path: Path to save the HTML file
        show_fig: If True, displays the figure interactively
    """
    # Generate distinct color and linestyle combinations
    colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
        "#17becf",  # cyan
    ]

    linestyles = ["solid", "dash", "dot", "dashdot"]

    fig = go.Figure()

    # Generate all unique combinations of color and linestyle
    style_combinations = list(product(range(len(colors)), range(len(linestyles))))

    # Sort by (rH, CV) for consistent ordering
    sorted_keys = sorted(stats.keys())

    for idx, (rH, CV) in enumerate(sorted_keys):
        data = stats[(rH, CV)]

        # Get unique color/linestyle for this combination
        color_idx, linestyle_idx = style_combinations[idx % len(style_combinations)]
        color = colors[color_idx]
        linestyle = linestyles[linestyle_idx]

        label = f"rH={rH:.0f}, CV={CV}"

        # Plot mean line
        fig.add_trace(
            go.Scatter(
                x=data["freq"],
                y=data["mean"],
                mode="lines",
                name=label,
                line=dict(color=color, dash=linestyle, width=2),
                legendgroup=label,
            )
        )

        # Plot ±1 std region (filled)
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([data["freq"], data["freq"][::-1]]),
                y=np.concatenate(
                    [data["mean"] + data["std"], (data["mean"] - data["std"])[::-1]]
                ),
                fill="toself",
                fillcolor=color,
                opacity=0.1,
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
                hoverinfo="skip",
                legendgroup=label,
            )
        )

    # Update layout
    fig.update_xaxes(title_text="Frequency (Hz)", type="log")
    fig.update_yaxes(title_text="Transfer Function Magnitude", type="log")
    fig.update_layout(
        height=700,
        width=1200,
        title_text="Transfer Function Statistics (±1 Std Dev across 5 realizations)",
        showlegend=True,
        hovermode="closest",
    )

    fig.write_html(str(output_path))
    print(f"Transfer function statistics plot saved to {output_path}")

    if show_fig:
        fig.show()


def plot_time_history_statistics(stats, output_path: Path, show_fig: bool = False):
    """
    Plot time history statistics with mean lines and ±1 std regions.
    Uses subplots: surface acceleration on top, base acceleration on bottom.

    Args:
        stats: Dictionary of statistics computed by compute_time_history_statistics
        output_path: Path to save the HTML file
        show_fig: If True, displays the figure interactively
    """
    # Generate distinct color and linestyle combinations
    colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # olive
        "#17becf",  # cyan
    ]

    linestyles = ["solid", "dash", "dot", "dashdot"]

    # Create subplots: surface on top (row 1), base on bottom (row 2)
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Surface Acceleration", "Base Acceleration"),
        vertical_spacing=0.08,
    )

    # Generate all unique combinations of color and linestyle
    style_combinations = list(product(range(len(colors)), range(len(linestyles))))

    # Sort by (rH, CV) for consistent ordering
    sorted_keys = sorted(stats.keys())

    for idx, (rH, CV) in enumerate(sorted_keys):
        data = stats[(rH, CV)]

        # Get unique color/linestyle for this combination
        color_idx, linestyle_idx = style_combinations[idx % len(style_combinations)]
        color = colors[color_idx]
        linestyle = linestyles[linestyle_idx]

        label = f"rH={rH:.0f}, CV={CV}"

        # Plot surface acceleration (top subplot)
        fig.add_trace(
            go.Scatter(
                x=data["time"],
                y=data["surface_mean"],
                mode="lines",
                name=label,
                line=dict(color=color, dash=linestyle, width=2),
                legendgroup=label,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        # Plot ±1 std region for surface
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([data["time"], data["time"][::-1]]),
                y=np.concatenate(
                    [
                        data["surface_mean"] + data["surface_std"],
                        (data["surface_mean"] - data["surface_std"])[::-1],
                    ]
                ),
                fill="toself",
                fillcolor=color,
                opacity=0.1,
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
                hoverinfo="skip",
                legendgroup=label,
            ),
            row=1,
            col=1,
        )

        # Plot base acceleration if available
        if data["base_mean"] is not None:
            fig.add_trace(
                go.Scatter(
                    x=data["time"],
                    y=data["base_mean"],
                    mode="lines",
                    name=label,
                    line=dict(color=color, dash=linestyle, width=2),
                    legendgroup=label,
                    showlegend=False,  # Only show in first subplot
                ),
                row=2,
                col=1,
            )

            # Plot ±1 std region for base
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([data["time"], data["time"][::-1]]),
                    y=np.concatenate(
                        [
                            data["base_mean"] + data["base_std"],
                            (data["base_mean"] - data["base_std"])[::-1],
                        ]
                    ),
                    fill="toself",
                    fillcolor=color,
                    opacity=0.1,
                    line=dict(color="rgba(255,255,255,0)"),
                    showlegend=False,
                    hoverinfo="skip",
                    legendgroup=label,
                ),
                row=2,
                col=1,
            )

    # Update axes for both subplots
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_yaxes(title_text="Acceleration", row=1, col=1)
    fig.update_yaxes(title_text="Acceleration", row=2, col=1)

    fig.update_layout(
        height=900,
        width=1200,
        title_text="Time History Statistics (±1 Std Dev across 5 realizations)",
        showlegend=True,
        hovermode="closest",
    )

    fig.write_html(str(output_path))
    print(f"Time history statistics plot saved to {output_path}")

    if show_fig:
        fig.show()


def main():
    """Main function to compute and plot statistics."""
    # Get Vs_min from a sample realization
    print("Generating sample realization to extract Vs_min...")
    Vs_profile_1D = np.array([180.0] * 8 + [1300.0] * 1)
    Lz = 50.0
    dx, dz = 2.5, 2.5
    rH, aHV, CV = 30.0, 1.0, 0.2  # Middle values
    seed = 30  # Middle seed value
    interlayer_seed = 42

    np.random.seed(seed)
    from seiskit.gaussian_field import _generate_vs_variability_field

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

    Vs_min = np.min(Vs_sample)
    print(f"Using Vs_min = {Vs_min:.2f} m/s for transfer function calculation")

    # Load data
    RESULTS_DIR = Path("./results")
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

    # Compute statistics
    print("Computing transfer function statistics...")
    stats = compute_tf_statistics(data, dz=dz, Vs_min=Vs_min)

    print(f"Computed statistics for {len(stats)} rH-CV combinations:")
    for (rH, CV), entry in stats.items():
        print(f"  rH={rH:.0f}, CV={CV}: {entry['n_realizations']} realizations")

    # Plot statistics
    print("Plotting transfer function statistics...")
    plot_tf_statistics(stats, Path("transfer_functions_statistics.html"))

    # Compute time history statistics
    print("\nComputing time history statistics...")
    time_stats = compute_time_history_statistics(data)

    print(f"Computed time history statistics for {len(time_stats)} rH-CV combinations:")
    for (rH, CV), entry in time_stats.items():
        print(f"  rH={rH:.0f}, CV={CV}: {entry['n_realizations']} realizations")

    # Plot time history statistics
    print("Plotting time history statistics...")
    plot_time_history_statistics(time_stats, Path("time_histories_statistics.html"))

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
