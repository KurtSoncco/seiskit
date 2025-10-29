import argparse
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

        # Decide if interpolation is needed: if all frequency arrays are identical
        # within a tight tolerance, we can stack directly to avoid interpolation noise.
        use_direct_stack = True
        ref_freq = freq_arrays[0]
        for f in freq_arrays[1:]:
            if f.shape != ref_freq.shape or not np.allclose(
                f, ref_freq, rtol=1e-9, atol=1e-12
            ):
                use_direct_stack = False
                break

        if use_direct_stack:
            common_freq = ref_freq
            tf_interpolated = tf_arrays  # already aligned
            print(f"TF alignment rH={rH:.0f}, CV={CV}: direct stack (no interpolation)")
        else:
            # Align all arrays to a common frequency grid using finest resolution
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
                    right=np.nan,
                )
                tf_interpolated.append(tf_interp)
            print(
                f"TF alignment rH={rH:.0f}, CV={CV}: interpolated to common grid (step={finest_step:.3e})"
            )

        # Compute statistics, ignoring NaN values
        tf_matrix = np.array(tf_interpolated)
        valid_counts = np.sum(~np.isnan(tf_matrix), axis=0)
        mean_tf = np.nanmean(tf_matrix, axis=0)
        std_tf = np.nanstd(tf_matrix, axis=0, ddof=1)

        # Filter out NaN values from the output
        valid_idx = ~np.isnan(mean_tf)
        common_freq = common_freq[valid_idx]
        mean_tf = mean_tf[valid_idx]
        std_tf = std_tf[valid_idx]
        valid_counts = valid_counts[valid_idx]

        # Diagnostics for small std values and contributor counts
        finite_std = std_tf[np.isfinite(std_tf)]
        if finite_std.size > 0:
            std_median = float(np.median(finite_std))
            std_min = float(np.min(finite_std))
            std_max = float(np.max(finite_std))
            counts_min = int(np.min(valid_counts)) if valid_counts.size else 0
            counts_median = int(np.median(valid_counts)) if valid_counts.size else 0
            counts_max = int(np.max(valid_counts)) if valid_counts.size else 0
            print(
                f"TF stats rH={rH:.0f}, CV={CV}: std(med/min/max)="
                f"{std_median:.3e}/{std_min:.3e}/{std_max:.3e}; "
                f"contributors per freq (min/med/max)="
                f"{counts_min}/{counts_median}/{counts_max}"
            )

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
            surface_std = np.std(surface_matrix, axis=0, ddof=1)

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
            base_std = np.std(base_matrix, axis=0, ddof=1)

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
    # Define color shades by rH group and linestyles by CV
    # rH 10 -> blue shades, rH 30 -> green shades, rH 50 -> red shades
    rh_colors = {
        10.0: {0.1: "#0b3d91", 0.2: "#1f77b4", 0.3: "#85b6e2"},
        30.0: {0.1: "#145a32", 0.2: "#2ca02c", 0.3: "#98df8a"},
        50.0: {0.1: "#7f0000", 0.2: "#d62728", 0.3: "#f28e8e"},
    }
    cv_linestyles = {0.1: "solid", 0.2: "dash", 0.3: "dot"}

    fig = go.Figure()

    # Sort by (rH, CV) for consistent ordering
    sorted_keys = sorted(stats.keys())

    for idx, (rH, CV) in enumerate(sorted_keys):
        data = stats[(rH, CV)]

        # Choose color shades by rH and CV, and linestyles by CV
        color = rh_colors.get(rH, {}).get(CV, "#7f7f7f")
        linestyle = cv_linestyles.get(CV, "solid")

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


def plot_tf_std(stats, output_path: Path, show_fig: bool = False):
    """
    Plot transfer function standard deviation as lines across frequency
    for each (rH, CV) combination.

    Args:
        stats: Dictionary from compute_tf_statistics containing 'freq' and 'std'
        output_path: Path to save the HTML file
        show_fig: If True, displays the figure interactively
    """
    # Same color/style mapping as mean plot
    rh_colors = {
        10.0: {0.1: "#0b3d91", 0.2: "#1f77b4", 0.3: "#85b6e2"},
        30.0: {0.1: "#145a32", 0.2: "#2ca02c", 0.3: "#98df8a"},
        50.0: {0.1: "#7f0000", 0.2: "#d62728", 0.3: "#f28e8e"},
    }
    cv_linestyles = {0.1: "solid", 0.2: "dash", 0.3: "dot"}

    fig = go.Figure()

    sorted_keys = sorted(stats.keys())

    for rH, CV in sorted_keys:
        data = stats[(rH, CV)]
        color = rh_colors.get(rH, {}).get(CV, "#7f7f7f")
        linestyle = cv_linestyles.get(CV, "solid")
        label = f"rH={rH:.0f}, CV={CV}"

        fig.add_trace(
            go.Scatter(
                x=data["freq"],
                y=data["std"],
                mode="lines",
                name=label,
                line=dict(color=color, dash=linestyle, width=2),
            )
        )

    fig.update_xaxes(title_text="Frequency (Hz)", type="log")
    fig.update_yaxes(title_text="TF Standard Deviation", type="log")
    fig.update_layout(
        height=700,
        width=1200,
        title_text="Transfer Function Standard Deviation across Realizations",
        showlegend=True,
        hovermode="closest",
    )

    fig.write_html(str(output_path))
    print(f"Transfer function std plot saved to {output_path}")

    if show_fig:
        fig.show()


def plot_tf_cov(stats, output_path: Path, show_fig: bool = False):
    """
    Plot coefficient of variation (std/mean) of transfer functions as lines
    across frequency for each (rH, CV) combination.

    Args:
        stats: Dictionary from compute_tf_statistics containing 'freq', 'mean', 'std'
        output_path: Path to save the HTML file
        show_fig: If True, displays the figure interactively
    """
    rh_colors = {
        10.0: {0.1: "#0b3d91", 0.2: "#1f77b4", 0.3: "#85b6e2"},
        30.0: {0.1: "#145a32", 0.2: "#2ca02c", 0.3: "#98df8a"},
        50.0: {0.1: "#7f0000", 0.2: "#d62728", 0.3: "#f28e8e"},
    }
    cv_linestyles = {0.1: "solid", 0.2: "dash", 0.3: "dot"}

    fig = go.Figure()

    sorted_keys = sorted(stats.keys())

    for rH, CV in sorted_keys:
        data = stats[(rH, CV)]

        mean_vals = data["mean"]
        std_vals = data["std"]
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = np.where(np.abs(mean_vals) > 1e-12, mean_vals, np.nan)
            cov = std_vals / denom

        color = rh_colors.get(rH, {}).get(CV, "#7f7f7f")
        linestyle = cv_linestyles.get(CV, "solid")
        label = f"rH={rH:.0f}, CV={CV}"

        fig.add_trace(
            go.Scatter(
                x=data["freq"],
                y=cov,
                mode="lines",
                name=label,
                line=dict(color=color, dash=linestyle, width=2),
            )
        )

    fig.update_xaxes(title_text="Frequency (Hz)", type="log")
    fig.update_yaxes(title_text="TF Coefficient of Variation (std/mean)")
    fig.update_layout(
        height=700,
        width=1200,
        title_text="Transfer Function Coefficient of Variation across Realizations",
        showlegend=True,
        hovermode="closest",
    )

    fig.write_html(str(output_path))
    print(f"Transfer function CoV plot saved to {output_path}")

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
    # CLI arguments
    parser = argparse.ArgumentParser(
        description="Compute and plot parametric study statistics"
    )
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=["tf", "tfstd", "tfcov", "time", "surface", "all"],
        default=["all"],
        help="Which plots to generate: tf, tfstd, tfcov, time, surface, or all",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive figures in addition to writing HTML",
    )
    args = parser.parse_args()

    do_all = "all" in args.plots
    do_tf = do_all or ("tf" in args.plots)
    do_tfstd = do_all or ("tfstd" in args.plots)
    do_tfcov = do_all or ("tfcov" in args.plots)
    do_time = do_all or ("time" in args.plots)
    do_surface = do_all or ("surface" in args.plots)

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

    # Transfer functions: compute Vs_min and TF stats only if requested (tf, tfstd, or tfcov)
    if do_tf or do_tfstd or do_tfcov:
        print("Generating sample realization to extract Vs_min...")
        Vs_profile_1D = np.array([180.0] * 8 + [1300.0] * 1)
        Lz = 50.0
        dx, dz = 2.5, 2.5
        rH, aHV, CV = 30.0, 1.0, 0.2
        seed = 30
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

        print("Computing transfer function statistics...")
        stats = compute_tf_statistics(data, dz=dz, Vs_min=Vs_min)

        print(f"Computed statistics for {len(stats)} rH-CV combinations:")
        for (rH, CV), entry in stats.items():
            print(f"  rH={rH:.0f}, CV={CV}: {entry['n_realizations']} realizations")

        if do_tf:
            print("Plotting transfer function statistics...")
            plot_tf_statistics(
                stats, Path("transfer_functions_statistics.html"), show_fig=args.show
            )

        if do_tfstd:
            print("Plotting transfer function std...")
            plot_tf_std(stats, Path("transfer_functions_std.html"), show_fig=args.show)

        if do_tfcov:
            print("Plotting transfer function coefficient of variation...")
            plot_tf_cov(stats, Path("transfer_functions_cov.html"), show_fig=args.show)

    # Time history statistics
    if do_time:
        print("\nComputing time history statistics...")
        time_stats = compute_time_history_statistics(data)

        print(
            f"Computed time history statistics for {len(time_stats)} rH-CV combinations:"
        )
        for (rH, CV), entry in time_stats.items():
            print(f"  rH={rH:.0f}, CV={CV}: {entry['n_realizations']} realizations")

        print("Plotting time history statistics...")
        plot_time_history_statistics(
            time_stats, Path("time_histories_statistics.html"), show_fig=args.show
        )

    # Surface accelerations (stacked)
    if do_surface:
        try:
            from seiskit.plot_results import plot_stacked_acceleration

            print("Plotting stacked surface accelerations...")
            plot_stacked_acceleration(
                datasets=data, data_config=DATA_CONFIG, vertical_spacing=3.0
            )
        except Exception as e:
            print(
                f"Warning: Could not generate stacked surface acceleration plots: {e}"
            )

    print("\nAnalysis complete!")


if __name__ == "__main__":
    ## Usage: python analyze_statistics.py --plots tf tfstd tfcov time surface --show
    main()
