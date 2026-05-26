"""
Script to create transfer function plots with geomean for each rH_**/CV_** folder.

For each folder in results/rH_**/CV_**, this script:
1. Finds all 5 realization folders
2. Loads base and top acceleration data from each realization
3. Computes transfer functions for each realization
4. Computes the geometric mean across all realizations
5. Creates a plotly figure with all individual TFs and the geomean
6. Saves the figure in the rH_**/CV_** folder
"""

import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from seiskit.ttf.TTF import TTF


def compute_vs_min():
    """Compute Vs_min for transfer function calculation."""
    Vs_profile_1D = np.array([180.0] * 8 + [1300.0] * 1)
    Lz = 50.0
    dx, dz = 2.5, 2.5
    rH, aHV, CV = 30.0, 1.0, 0.2
    seed = 30
    interlayer_seed = 42

    np.random.seed(seed)
    from seiskit.gaussian_field import generate_vs_variability_field

    Vs_sample, _, _, _ = generate_vs_variability_field(
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
    return Vs_min, dz


def load_realization_data(results_dir, rH, CV, seed):
    """
    Load base and top acceleration data for a specific realization.

    Args:
        results_dir: Base results directory
        rH: rH value
        CV: CV value
        seed: Seed value

    Returns:
        Tuple of (base_time, base_accel, top_time, top_accel) or None if files not found
    """
    # Construct paths - note the double folder structure
    task_id = f"rH{rH:.0f}_CV{CV}_s{seed}"
    base_path = results_dir / f"rH_{rH:.0f}/CV_{CV}/{task_id}/{task_id}/soil_base_dof1_accel.txt"
    top_path = results_dir / f"rH_{rH:.0f}/CV_{CV}/{task_id}/{task_id}/soil_top_dof1_accel.txt"

    if not base_path.exists() or not top_path.exists():
        return None

    try:
        base_data = np.loadtxt(base_path)
        top_data = np.loadtxt(top_path)

        base_time = base_data[:, 0]
        base_accel = base_data[:, 1]
        top_time = top_data[:, 0]
        top_accel = top_data[:, 1]

        return (base_time, base_accel, top_time, top_accel)
    except Exception as e:
        print(f"  Warning: Could not load data for {task_id}: {e}")
        return None


def compute_transfer_functions_for_folder(results_dir, rH, CV, dz=2.5, Vs_min=None):
    """
    Compute transfer functions for all realizations in a given rH/CV folder.

    Args:
        results_dir: Base results directory
        rH: rH value
        CV: CV value
        dz: Depth increment for TTF calculation
        Vs_min: Minimum Vs for TTF calculation

    Returns:
        Dictionary with 'freq', 'individual_tfs', 'geomean', 'n_realizations', 'seeds'
    """
    seed_values = [10, 20, 30, 40, 50]
    freq_arrays = []
    tf_arrays = []
    successful_seeds = []  # Track which seeds were successfully processed

    for seed in seed_values:
        data = load_realization_data(results_dir, rH, CV, seed)
        if data is None:
            continue

        base_time, base_accel, top_time, top_accel = data

        # Validate data
        if (
            base_time is None
            or base_accel is None
            or top_accel is None
            or np.size(base_time) == 0
            or np.size(base_accel) == 0
            or np.size(top_accel) == 0
            or len(base_time) < 2
        ):
            continue

        # Compute transfer function
        try:
            dt = base_time[1] - base_time[0]
            freq, tf = TTF(
                top_accel,
                base_accel,
                dt=dt,
                dz=dz,
                Vsmin=Vs_min,
            )
            freq_arrays.append(freq)
            tf_arrays.append(tf)
            successful_seeds.append(seed)  # Track successful seed
        except Exception as e:
            print(f"  Warning: Could not compute TF for seed {seed}: {e}")
            continue

    if len(tf_arrays) == 0:
        return None

    # Align all arrays to a common frequency grid
    use_direct_stack = True
    ref_freq = freq_arrays[0]
    for f in freq_arrays[1:]:
        if f.shape != ref_freq.shape or not np.allclose(f, ref_freq, rtol=1e-9, atol=1e-12):
            use_direct_stack = False
            break

    if use_direct_stack:
        common_freq = ref_freq
        tf_interpolated = tf_arrays
    else:
        # Interpolate to common frequency grid
        min_freq = min(f.min() for f in freq_arrays)
        max_freq = max(f.max() for f in freq_arrays)

        step_sizes = [np.diff(f).min() for f in freq_arrays if len(f) > 1]
        if len(step_sizes) == 0:
            finest_step = (max_freq - min_freq) / 100
        else:
            finest_step = min(step_sizes)

        common_freq = np.arange(min_freq, max_freq + finest_step, finest_step)

        tf_interpolated = []
        for freq, tf in zip(freq_arrays, tf_arrays):
            tf_interp = np.interp(
                common_freq,
                freq,
                tf,
                left=np.nan,
                right=np.nan,
            )
            tf_interpolated.append(tf_interp)

    # Compute geometric mean
    tf_matrix = np.array(tf_interpolated)
    geomean_tf = np.exp(np.nanmean(np.log(tf_matrix), axis=0))

    # Filter out NaN values
    valid_idx = ~np.isnan(geomean_tf)
    common_freq = common_freq[valid_idx]
    geomean_tf = geomean_tf[valid_idx]
    tf_interpolated_filtered = [tf[valid_idx] for tf in tf_interpolated]

    return {
        "freq": common_freq,
        "individual_tfs": tf_interpolated_filtered,
        "geomean": geomean_tf,
        "n_realizations": len(tf_interpolated_filtered),
        "seeds": successful_seeds,  # Include seed values for labeling
    }


def plot_transfer_functions_plotly(tf_data, output_path, rH, CV):
    """
    Create a plotly figure with all individual transfer functions and the geomean.

    Args:
        tf_data: Dictionary with 'freq', 'individual_tfs', 'geomean', 'n_realizations'
        output_path: Path to save the HTML file
        rH: rH value for title
        CV: CV value for title
    """
    fig = go.Figure()

    # Define distinct colors for each realization (5 colors)
    realization_colors = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
    ]

    # Define line widths for realizations (varying for distinction)
    realization_linewidths = [1.8, 1.6, 1.7, 1.5, 1.9]

    # Plot individual transfer functions with distinct colors and labels
    seeds = tf_data.get("seeds", [])
    for idx, tf in enumerate(tf_data["individual_tfs"]):
        valid_mask = ~np.isnan(tf)
        if np.any(valid_mask):
            color = realization_colors[idx % len(realization_colors)]
            linewidth = realization_linewidths[idx % len(realization_linewidths)]
            # Use seed value in label if available, otherwise use index
            if idx < len(seeds):
                label = f"Realization s{seeds[idx]}"
            else:
                label = f"Realization {idx + 1}"
            fig.add_trace(
                go.Scatter(
                    x=tf_data["freq"][valid_mask],
                    y=tf[valid_mask],
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=linewidth),
                    showlegend=True,
                    opacity=0.8,
                )
            )

    # Plot geometric mean in a highlighted color (thicker, darker)
    geomean_color = "#000000"  # Black for strong contrast
    valid_geomean_mask = ~np.isnan(tf_data["geomean"])
    if np.any(valid_geomean_mask):
        fig.add_trace(
            go.Scatter(
                x=tf_data["freq"][valid_geomean_mask],
                y=tf_data["geomean"][valid_geomean_mask],
                mode="lines",
                name="Geometric Mean",
                line=dict(color=geomean_color, width=3.5),
                showlegend=True,
            )
        )

    # Update layout
    fig.update_xaxes(title_text="Frequency (Hz)", type="log")
    fig.update_yaxes(title_text="Transfer Function Magnitude", type="log")
    fig.update_layout(
        height=700,
        width=1200,
        title_text=f"Transfer Functions - rH={rH:.0f}, CV={CV} ({tf_data['n_realizations']} realizations)",
        showlegend=True,
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    fig.write_html(str(output_path))
    print(f"  Saved transfer function plot to {output_path}")


def main():
    """Main function to process all rH_**/CV_** folders."""
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    results_dir = Path("./results")
    if not results_dir.exists():
        print(f"Error: Results directory {results_dir} not found")
        return

    # Compute Vs_min (same for all folders)
    print("Computing Vs_min for transfer function calculation...")
    Vs_min, dz = compute_vs_min()
    print(f"Using Vs_min = {Vs_min:.2f} m/s, dz = {dz:.2f} m")

    # Find all rH folders
    rH_folders = sorted(
        [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("rH_")]
    )

    if not rH_folders:
        print(f"No rH folders found in {results_dir}")
        return

    # Process each rH/CV combination
    total_processed = 0
    total_failed = 0

    for rH_folder in rH_folders:
        # Extract rH value
        try:
            rH_str = rH_folder.name.replace("rH_", "")
            rH = float(rH_str)
        except ValueError:
            print(f"Warning: Could not parse rH from folder name {rH_folder.name}")
            continue

        # Find all CV folders
        CV_folders = sorted(
            [d for d in rH_folder.iterdir() if d.is_dir() and d.name.startswith("CV_")]
        )

        for CV_folder in CV_folders:
            # Extract CV value
            try:
                CV_str = CV_folder.name.replace("CV_", "")
                CV = float(CV_str)
            except ValueError:
                print(f"Warning: Could not parse CV from folder name {CV_folder.name}")
                continue

            print(f"\nProcessing rH={rH:.0f}, CV={CV}...")

            # Compute transfer functions for this folder
            tf_data = compute_transfer_functions_for_folder(
                results_dir, rH, CV, dz=dz, Vs_min=Vs_min
            )

            if tf_data is None:
                print("  Failed: No valid transfer function data found")
                total_failed += 1
                continue

            if tf_data["n_realizations"] == 0:
                print("  Failed: No realizations processed")
                total_failed += 1
                continue

            print(f"  Found {tf_data['n_realizations']} valid realizations")

            # Create plot
            output_path = CV_folder / "transfer_functions_plotly.html"
            plot_transfer_functions_plotly(tf_data, output_path, rH, CV)
            total_processed += 1

    print(f"\n{'=' * 60}")
    print("Processing complete!")
    print(f"  Successfully processed: {total_processed} folders")
    print(f"  Failed: {total_failed} folders")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
