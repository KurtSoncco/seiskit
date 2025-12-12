"""
Create validation plots comparing uniform 1D OpenSees results with variability results.

This script generates validation plots to demonstrate consistency with Huang et al.'s findings:
1. Uniform vs theoretical resonance frequencies
2. CV effect at fixed rH
3. rH effect at fixed CV
4. Velocity-dependent sensitivity
5. Amplification vs CV summary
"""

import csv
import os
import re
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Import functions from compute_transfer_functions.py
from compute_transfer_functions import (
    compute_coefficient_of_variation,
    compute_geometric_mean,
    compute_transfer_function,
    load_acceleration_files,
    load_all_results,
    load_tf_dict_from_pickle,
)


def parse_1D_folder(folder_name: str) -> Dict[str, float | str] | None:
    """
    Parse 1D result folder name to extract parameters.

    Format: Vs1_{Vs1}_thickness_{thickness}_damping_method_{damping_method}

    Returns:
        Dictionary with keys: Vs1, thickness, damping_method
        None if parsing fails
    """
    pattern = r"Vs1_(\d+\.?\d*)_thickness_(\d+\.?\d*)_damping_method_([a-zA-Z_]+)"
    match = re.match(pattern, folder_name)
    if match:
        return {
            "Vs1": float(match.group(1)),
            "thickness": float(match.group(2)),
            "damping_method": match.group(3),
        }
    return None


def load_1D_results(
    results_1D_dir: Path, Vsmin: float = 1000.0
) -> Dict[Tuple[float, float, str], Tuple[np.ndarray, np.ndarray]]:
    """
    Load transfer functions from 1D_results folders.

    Args:
        results_1D_dir: Path to 1D_results directory
        Vsmin: Minimum shear wave velocity for fmax calculation

    Returns:
        Dictionary keyed by (Vs1, thickness, damping_method) -> (freq, tf)
    """
    tf_dict = {}
    results_path = Path(results_1D_dir)

    if not results_path.exists():
        raise FileNotFoundError(f"1D results directory not found: {results_path}")

    print(f"Loading 1D results from: {results_path}")
    folders = sorted([f for f in results_path.iterdir() if f.is_dir()])

    loaded_count = 0
    skipped_count = 0

    for folder in folders:
        params = parse_1D_folder(folder.name)
        if params is None:
            print(f"Skipping folder (could not parse): {folder.name}")
            skipped_count += 1
            continue

        # Load acceleration files
        thickness_val = float(params["thickness"])
        accel_data = load_acceleration_files(folder, thickness_val)
        if accel_data is None:
            print(f"Skipping folder (no acceleration data): {folder.name}")
            skipped_count += 1
            continue

        base_data, top_data = accel_data

        # Compute transfer function
        try:
            freq, tf = compute_transfer_function(base_data, top_data, Vsmin=Vsmin)

            # Store in dictionary
            key = (params["Vs1"], params["thickness"], params["damping_method"])
            tf_dict[key] = (freq, tf)
            loaded_count += 1

        except Exception as e:
            print(f"Error computing TF for {folder.name}: {e}")
            skipped_count += 1
            continue

    print(f"Loaded {loaded_count} 1D transfer functions")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} folders")

    return tf_dict


def compute_theoretical_resonance_frequencies(
    Vs1: float, thickness: float, n_modes: int = 5
) -> np.ndarray:
    """
    Compute theoretical resonance frequencies using Kramer's formula.

    Formula: fn = (2n - 1) * Vs / (4 * H)
    where n = 1, 2, 3, ... is the mode number

    Args:
        Vs1: Shear wave velocity of the layer (m/s)
        thickness: Thickness of the layer (m)
        n_modes: Number of modes to compute (default 5)

    Returns:
        Array of resonance frequencies (Hz)
    """
    frequencies = []
    for n in range(1, n_modes + 1):
        fn = (2 * n - 1) * Vs1 / (4 * thickness)
        frequencies.append(fn)
    return np.array(frequencies)


def plot_uniform_vs_theoretical(
    tf_dict: Dict,
    tf_dict_1D: Dict,
    output_dir: Path,
):
    """
    Plot 1: Uniform vs theoretical resonance frequencies.

    One figure per Vs1, for each thickness:
    - Overlay geomean TF from lowest variability case (rH=50, CV=0.1) with 1D uniform TF
    - Mark theoretical resonance frequencies as vertical lines

    Args:
        tf_dict: Dictionary of variability TFs keyed by (rH, CV, Vs1, thickness, seed, damping_method)
        tf_dict_1D: Dictionary of 1D uniform TFs keyed by (Vs1, thickness, damping_method)
        output_dir: Output directory for plots
    """
    Vs1_values = sorted(set(k[2] for k in tf_dict.keys()))
    thickness_values = sorted(set(k[3] for k in tf_dict.keys()))
    damping_methods = sorted(set(k[5] for k in tf_dict.keys()))

    for damping_method in damping_methods:
        for Vs1 in Vs1_values:
            fig, axes = plt.subplots(
                1, len(thickness_values), figsize=(6 * len(thickness_values), 6)
            )
            if len(thickness_values) == 1:
                axes = [axes]

            for idx, thickness in enumerate(thickness_values):
                ax = axes[idx]

                # Get 1D uniform TF
                uniform_key = (Vs1, thickness, damping_method)
                if uniform_key in tf_dict_1D:
                    uniform_freq, uniform_tf = tf_dict_1D[uniform_key]
                    ax.loglog(
                        uniform_freq,
                        uniform_tf,
                        color="black",
                        linewidth=2.5,
                        linestyle="-",
                        label="Uniform (1D)",
                    )

                # Get lowest variability case (rH=50, CV=0.1) for this damping method
                filtered_tfs = {
                    key: value
                    for key, value in tf_dict.items()
                    if key[0] == 50.0
                    and key[1] == 0.1
                    and key[2] == Vs1
                    and key[3] == thickness
                    and key[5] == damping_method
                }

                if len(filtered_tfs) > 0:
                    # Collect all TFs for this case
                    tf_arrays = []
                    common_freq = None
                    for freq, tf in filtered_tfs.values():
                        if common_freq is None:
                            common_freq = freq
                        tf_arrays.append(tf)

                    # Compute geometric mean
                    geomean = compute_geometric_mean(tf_arrays)
                    ax.loglog(
                        common_freq,
                        geomean,
                        color="red",
                        linewidth=2.5,
                        linestyle="--",
                        label="rH=50, CV=0.1 (geomean)",
                    )

                # Mark theoretical resonance frequencies
                theoretical_freqs = compute_theoretical_resonance_frequencies(
                    Vs1, thickness, n_modes=5
                )
                for fn in theoretical_freqs:
                    ax.axvline(
                        fn, color="blue", linestyle=":", linewidth=1.5, alpha=0.7
                    )

                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Transfer Function Magnitude")
                ax.set_title(f"h={thickness:.0f} m")
                ax.set_xlim(1e-1, 5e1)
                ax.set_ylim(1e-2, 1e2)
                ax.grid(True, alpha=0.3)
                ax.legend(loc="best", fontsize=8)

            plt.suptitle(
                f"Vs1={Vs1:.0f} m/s, damping={damping_method}", fontsize=16, y=1.02
            )
            plt.tight_layout()
            output_file = (
                output_dir
                / f"validation_uniform_vs_theoretical_Vs1_{Vs1:.0f}_damping_{damping_method}.png"
            )
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"Saved plot: {output_file}")
            plt.close()


def plot_CV_effect_at_fixed_rH(
    tf_dict: Dict,
    tf_dict_1D: Dict,
    output_dir: Path,
    rH: float = 50.0,
    thickness: float = 76.0,
):
    """
    Plot 2: CV effect at fixed rH.

    For each Vs1, at fixed depth (h=76 m) and rH:
    - Plot geomean TF for CV = 0.10, 0.20, 0.30
    - Overlay uniform TF from 1D_results as reference
    - Optionally: second panel with CV (coefficient of variation) vs frequency

    Args:
        tf_dict: Dictionary of variability TFs keyed by (rH, CV, Vs1, thickness, seed, damping_method)
        tf_dict_1D: Dictionary of 1D uniform TFs keyed by (Vs1, thickness, damping_method)
        output_dir: Output directory for plots
        rH: Fixed rH value (default 50.0 m)
        thickness: Fixed thickness value (default 76.0 m)
    """
    Vs1_values = sorted(set(k[2] for k in tf_dict.keys()))
    CV_values = sorted(
        set(k[1] for k in tf_dict.keys())
    )  # Use available CV values from data
    damping_methods = sorted(set(k[5] for k in tf_dict.keys()))

    for damping_method in damping_methods:
        for Vs1 in Vs1_values:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            # Panel 1: Transfer functions
            ax1 = axes[0]

            # Get uniform TF
            uniform_key = (Vs1, thickness, damping_method)
            if uniform_key in tf_dict_1D:
                uniform_freq, uniform_tf = tf_dict_1D[uniform_key]
                ax1.loglog(
                    uniform_freq,
                    uniform_tf,
                    color="black",
                    linewidth=2.5,
                    linestyle="-",
                    label="Uniform (1D)",
                )

            # Plot geomean for each CV
            colors = plt.colormaps["viridis"](np.linspace(0, 1, len(CV_values)))
            for cv_val, color in zip(CV_values, colors):
                filtered_tfs = {
                    key: value
                    for key, value in tf_dict.items()
                    if key[0] == rH
                    and key[1] == cv_val
                    and key[2] == Vs1
                    and key[3] == thickness
                    and key[5] == damping_method
                }

                if len(filtered_tfs) > 0:
                    tf_arrays = []
                    common_freq = None
                    for freq, tf in filtered_tfs.values():
                        if common_freq is None:
                            common_freq = freq
                        tf_arrays.append(tf)

                    geomean = compute_geometric_mean(tf_arrays)
                    ax1.loglog(
                        common_freq,
                        geomean,
                        color=color,
                        linewidth=2.5,
                        linestyle="--",
                        label=f"CV={cv_val:.2f} (geomean)",
                    )

            ax1.set_xlabel("Frequency (Hz)")
            ax1.set_ylabel("Transfer Function Magnitude")
            ax1.set_title(
                f"Geomean TF: Vs1={Vs1:.0f} m/s, h={thickness:.0f} m, rH={rH:.0f} m, damping={damping_method}"
            )
            ax1.set_xlim(1e-1, 5e1)
            ax1.set_ylim(1e-2, 1e2)
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc="best", fontsize=8)

            # Panel 2: Coefficient of variation vs frequency
            ax2 = axes[1]

            for cv_val, color in zip(CV_values, colors):
                filtered_tfs = {
                    key: value
                    for key, value in tf_dict.items()
                    if key[0] == rH
                    and key[1] == cv_val
                    and key[2] == Vs1
                    and key[3] == thickness
                    and key[5] == damping_method
                }

                if len(filtered_tfs) > 0:
                    tf_arrays = []
                    common_freq = None
                    for freq, tf in filtered_tfs.values():
                        if common_freq is None:
                            common_freq = freq
                        tf_arrays.append(tf)

                    cv_array = compute_coefficient_of_variation(tf_arrays)
                    ax2.semilogx(
                        common_freq,
                        cv_array,
                        color=color,
                        linewidth=2.5,
                        label=f"CV={cv_val:.2f}",
                    )

            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("Coefficient of Variation")
            ax2.set_title(
                f"CV vs Frequency: Vs1={Vs1:.0f} m/s, h={thickness:.0f} m, rH={rH:.0f} m, damping={damping_method}"
            )
            ax2.set_xlim(1e-1, 5e1)
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="best", fontsize=8)

            plt.tight_layout()
            output_file = (
                output_dir
                / f"validation_CV_effect_Vs1_{Vs1:.0f}_rH_{rH:.0f}_damping_{damping_method}.png"
            )
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"Saved plot: {output_file}")
            plt.close()


def plot_rH_effect_at_fixed_CV(
    tf_dict: Dict,
    tf_dict_1D: Dict,
    output_dir: Path,
    CV: float = 0.3,
    thickness: float = 76.0,
):
    """
    Plot 3: rH effect at fixed CV.

    For each Vs1 (or at least Vs1=100 m/s):
    - Fix CV (e.g., CV=0.30) and depth (h=76 m)
    - Plot geomean TFs for rH=10 m and rH=50 m
    - Optionally: band-averaged TF (5-15 Hz) as bar plots

    Args:
        tf_dict: Dictionary of variability TFs keyed by (rH, CV, Vs1, thickness, seed, damping_method)
        tf_dict_1D: Dictionary of 1D uniform TFs keyed by (Vs1, thickness, damping_method)
        output_dir: Output directory for plots
        CV: Fixed CV value (default 0.3)
        thickness: Fixed thickness value (default 76.0 m)
    """
    Vs1_values = sorted(set(k[2] for k in tf_dict.keys()))
    rH_values = sorted(
        set(k[0] for k in tf_dict.keys())
    )  # Use available rH values from data
    damping_methods = sorted(set(k[5] for k in tf_dict.keys()))

    for damping_method in damping_methods:
        for Vs1 in Vs1_values:
            fig, axes = plt.subplots(1, 3, figsize=(24, 6))

            # Panel 1: Transfer functions
            ax1 = axes[0]

            # Compute theoretical resonance frequencies (used in multiple panels)
            theoretical_freqs = compute_theoretical_resonance_frequencies(
                Vs1, thickness, n_modes=5
            )
            window_fraction = 0.1

            colors = ["blue", "red"]
            for rH, color in zip(rH_values, colors):
                filtered_tfs = {
                    key: value
                    for key, value in tf_dict.items()
                    if key[0] == rH
                    and key[1] == CV
                    and key[2] == Vs1
                    and key[3] == thickness
                    and key[5] == damping_method
                }

                if len(filtered_tfs) > 0:
                    tf_arrays = []
                    common_freq = None
                    for freq, tf in filtered_tfs.values():
                        if common_freq is None:
                            common_freq = freq
                        tf_arrays.append(tf)

                    geomean = compute_geometric_mean(tf_arrays)
                    ax1.loglog(
                        common_freq,
                        geomean,
                        color=color,
                        linewidth=2.5,
                        linestyle="-",
                        label=f"rH={rH:.0f} m",
                    )

            # Add vertical lines at theoretical resonance frequencies and ±10% range
            for fn in theoretical_freqs:
                # Vertical line at resonance frequency (dotted blue)
                ax1.axvline(fn, color="blue", linestyle=":", linewidth=1.5, alpha=0.7)
                # Dashed lines at ±10% of fn (gray dashed)
                window_width = window_fraction * fn
                freq_low = fn - window_width
                freq_high = fn + window_width
                ax1.axvline(
                    freq_low, color="gray", linestyle="--", linewidth=1.0, alpha=0.5
                )
                ax1.axvline(
                    freq_high, color="gray", linestyle="--", linewidth=1.0, alpha=0.5
                )

            ax1.set_xlabel("Frequency (Hz)")
            ax1.set_ylabel("Transfer Function Magnitude")
            ax1.set_title(
                f"Geomean TF: Vs1={Vs1:.0f} m/s, h={thickness:.0f} m, CV={CV:.2f}, damping={damping_method}"
            )
            ax1.set_xlim(1e-1, 5e1)
            ax1.set_ylim(1e-2, 1e2)
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc="best", fontsize=8)

            # Panel 2: Local band-averaged TF around each mode - box plot for individual realizations
            ax2 = axes[1]

            # Window width is ±0.1*fn (10% of the resonance frequency)
            # (theoretical_freqs already computed above)

            # Prepare data structure: {mode_number: {rH: [band_avg_values]}}
            mode_data = {}
            for mode_num in range(1, len(theoretical_freqs) + 1):
                mode_data[mode_num] = {}
                for rH in rH_values:
                    mode_data[mode_num][rH] = []

            # Compute local band averages for each realization
            for rH in rH_values:
                filtered_tfs = {
                    key: value
                    for key, value in tf_dict.items()
                    if key[0] == rH
                    and key[1] == CV
                    and key[2] == Vs1
                    and key[3] == thickness
                    and key[5] == damping_method
                }

                if len(filtered_tfs) > 0:
                    for freq, tf in filtered_tfs.values():
                        # For each mode, compute local band average
                        for mode_num, fn in enumerate(theoretical_freqs, start=1):
                            window_width = window_fraction * fn  # ±0.1*fn
                            freq_low = fn - window_width
                            freq_high = fn + window_width

                            # Compute band-averaged TF in this window for this realization
                            freq_mask = (freq >= freq_low) & (freq <= freq_high)
                            if np.any(freq_mask):
                                band_avg = np.mean(tf[freq_mask])
                                mode_data[mode_num][rH].append(band_avg)

            # Create box plot for each mode
            if len(mode_data) > 0:
                # Prepare data for boxplot: group by mode, then by rH
                plot_data = []
                plot_labels = []
                plot_positions = []
                pos = 0

                for mode_num in sorted(mode_data.keys()):
                    for rH in rH_values:
                        if len(mode_data[mode_num][rH]) > 0:
                            plot_data.append(mode_data[mode_num][rH])
                            plot_labels.append(f"Mode {mode_num}\nrH={rH:.0f}m")
                            plot_positions.append(pos)
                            pos += 1
                    # Add spacing between modes
                    pos += 0.5

                if len(plot_data) > 0:
                    bp = ax2.boxplot(
                        plot_data,
                        positions=plot_positions,
                        tick_labels=plot_labels,
                        patch_artist=True,
                        widths=0.4,
                    )

                    # Color the boxes: blue for rH=10, red for rH=50
                    colors_box = []
                    for i, label in enumerate(plot_labels):
                        if "rH=10" in label:
                            colors_box.append("blue")
                        else:
                            colors_box.append("red")

                    for patch, color in zip(bp["boxes"], colors_box):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)

                    ax2.set_ylabel("Local Band-Averaged TF")
                    ax2.set_title(
                        f"Local Band-Averaged TF Around Each Mode: Vs1={Vs1:.0f} m/s, h={thickness:.0f} m, CV={CV:.2f}, damping={damping_method}"
                    )
                    ax2.grid(True, alpha=0.3, axis="y")

                    # Add vertical lines to separate modes
                    mode_separators = []
                    current_pos = 0
                    for mode_num in sorted(mode_data.keys())[:-1]:  # All but last
                        # Count positions for this mode
                        for rH in rH_values:
                            if len(mode_data[mode_num][rH]) > 0:
                                current_pos += 1
                        current_pos += 0.5  # Spacing
                        mode_separators.append(current_pos)

                    for sep_pos in mode_separators:
                        ax2.axvline(
                            sep_pos,
                            color="gray",
                            linestyle="--",
                            linewidth=0.5,
                            alpha=0.5,
                        )

            # Panel 3: Normalized local band-averaged TF (relative to uniform 1D)
            ax3 = axes[2]

            # Get uniform TF for normalization
            uniform_key = (Vs1, thickness, damping_method)
            uniform_peaks = {}
            if uniform_key in tf_dict_1D:
                uniform_freq, uniform_tf = tf_dict_1D[uniform_key]

                # Compute uniform peak for each mode
                for mode_num, fn in enumerate(theoretical_freqs, start=1):
                    window_width = window_fraction * fn  # ±0.1*fn
                    freq_low = fn - window_width
                    freq_high = fn + window_width

                    # Compute peak TF in this window for uniform case
                    freq_mask = (uniform_freq >= freq_low) & (uniform_freq <= freq_high)
                    if np.any(freq_mask):
                        uniform_peak = np.mean(uniform_tf[freq_mask])
                        uniform_peaks[mode_num] = uniform_peak

            # Compute normalized band averages
            normalized_mode_data = {}
            for mode_num in range(1, len(theoretical_freqs) + 1):
                normalized_mode_data[mode_num] = {}
                for rH in rH_values:
                    normalized_mode_data[mode_num][rH] = []

                    # Normalize each realization's band average by uniform peak
                    if mode_num in uniform_peaks and mode_num in mode_data:
                        uniform_peak = uniform_peaks[mode_num]
                        for band_avg in mode_data[mode_num][rH]:
                            normalized_value = band_avg / uniform_peak
                            normalized_mode_data[mode_num][rH].append(normalized_value)

            # Create box plot for normalized values
            # Check if we have any normalized data to plot
            has_normalized_data = False
            for mode_num in normalized_mode_data.keys():
                for rH in rH_values:
                    if len(normalized_mode_data[mode_num][rH]) > 0:
                        has_normalized_data = True
                        break
                if has_normalized_data:
                    break

            if has_normalized_data:
                # Prepare data for boxplot: group by mode, then by rH
                plot_data_norm = []
                plot_labels_norm = []
                plot_positions_norm = []
                pos = 0

                for mode_num in sorted(normalized_mode_data.keys()):
                    for rH in rH_values:
                        if len(normalized_mode_data[mode_num][rH]) > 0:
                            plot_data_norm.append(normalized_mode_data[mode_num][rH])
                            plot_labels_norm.append(f"Mode {mode_num}\nrH={rH:.0f}m")
                            plot_positions_norm.append(pos)
                            pos += 1
                    # Add spacing between modes
                    pos += 0.5

                if len(plot_data_norm) > 0:
                    bp_norm = ax3.boxplot(
                        plot_data_norm,
                        positions=plot_positions_norm,
                        tick_labels=plot_labels_norm,
                        patch_artist=True,
                        widths=0.4,
                    )

                    # Color the boxes: blue for rH=10, red for rH=50
                    colors_box_norm = []
                    for i, label in enumerate(plot_labels_norm):
                        if "rH=10" in label:
                            colors_box_norm.append("blue")
                        else:
                            colors_box_norm.append("red")

                    for patch, color in zip(bp_norm["boxes"], colors_box_norm):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)

                    ax3.set_ylabel(
                        "Normalized Band-Averaged TF\n(Relative to Uniform 1D)"
                    )
                    ax3.set_title(
                        f"Normalized Band-Averaged TF: Vs1={Vs1:.0f} m/s, h={thickness:.0f} m, CV={CV:.2f}, damping={damping_method}"
                    )
                    ax3.grid(True, alpha=0.3, axis="y")

                    # Add horizontal line at 1.0 (uniform reference)
                    ax3.axhline(
                        1.0,
                        color="black",
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.7,
                        label="Uniform (1D)",
                    )

                    # Add vertical lines to separate modes
                    mode_separators_norm = []
                    current_pos = 0
                    for mode_num in sorted(normalized_mode_data.keys())[
                        :-1
                    ]:  # All but last
                        # Count positions for this mode
                        for rH in rH_values:
                            if len(normalized_mode_data[mode_num][rH]) > 0:
                                current_pos += 1
                        current_pos += 0.5  # Spacing
                        mode_separators_norm.append(current_pos)

                    for sep_pos in mode_separators_norm:
                        ax3.axvline(
                            sep_pos,
                            color="gray",
                            linestyle="--",
                            linewidth=0.5,
                            alpha=0.5,
                        )

                    ax3.legend(loc="best", fontsize=8)
                else:
                    # No normalized data to plot
                    ax3.text(
                        0.5,
                        0.5,
                        "No normalized data\n(Uniform TF not available)",
                        ha="center",
                        va="center",
                        transform=ax3.transAxes,
                        fontsize=12,
                        color="gray",
                    )
                    ax3.set_ylabel(
                        "Normalized Band-Averaged TF\n(Relative to Uniform 1D)"
                    )
                    ax3.set_title(
                        f"Normalized Band-Averaged TF: Vs1={Vs1:.0f} m/s, h={thickness:.0f} m, CV={CV:.2f}, damping={damping_method}"
                    )
            else:
                # No normalized data structure created
                ax3.text(
                    0.5,
                    0.5,
                    "No normalized data\n(Uniform TF not available)",
                    ha="center",
                    va="center",
                    transform=ax3.transAxes,
                    fontsize=12,
                    color="gray",
                )
                ax3.set_ylabel("Normalized Band-Averaged TF\n(Relative to Uniform 1D)")
                ax3.set_title(
                    f"Normalized Band-Averaged TF: Vs1={Vs1:.0f} m/s, h={thickness:.0f} m, CV={CV:.2f}, damping={damping_method}"
                )

            plt.tight_layout()
            output_file = (
                output_dir
                / f"validation_rH_effect_Vs1_{Vs1:.0f}_CV_{CV:.2f}_damping_{damping_method}.png"
            )
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            print(f"Saved plot: {output_file}")
            plt.close()


def plot_velocity_dependent_sensitivity(
    tf_dict: Dict,
    output_dir: Path,
    thickness: float = 76.0,
):
    """
    Plot 4: Velocity-dependent sensitivity.

    Single depth (h=76 m), 3×1 panel:
    - Left: Vs1=750 m/s, all six curves (rH=10/50, CV=0.1/0.2/0.3)
    - Middle: Vs1=350 m/s, same
    - Right: Vs1=100 m/s, same
    - Identical axes and color coding

    Args:
        tf_dict: Dictionary of variability TFs keyed by (rH, CV, Vs1, thickness, seed, damping_method)
        output_dir: Output directory for plots
        thickness: Fixed thickness value (default 76.0 m)
    """
    Vs1_values = sorted(set(k[2] for k in tf_dict.keys()))[
        :3
    ]  # Use available Vs1 values, limit to 3
    rH_values = sorted(
        set(k[0] for k in tf_dict.keys())
    )  # Use available rH values from data
    CV_values = sorted(
        set(k[1] for k in tf_dict.keys())
    )  # Use available CV values from data
    damping_methods = sorted(set(k[5] for k in tf_dict.keys()))

    for damping_method in damping_methods:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Create color map for combinations
        combinations = []
        for rH in rH_values:
            for cv in CV_values:
                combinations.append((rH, cv))

        cmap = plt.colormaps["tab10"]
        colors_map = {combo: cmap(i) for i, combo in enumerate(combinations)}

        for idx, Vs1 in enumerate(Vs1_values):
            ax = axes[idx]

            for rH, cv in combinations:
                filtered_tfs = {
                    key: value
                    for key, value in tf_dict.items()
                    if key[0] == rH
                    and key[1] == cv
                    and key[2] == Vs1
                    and key[3] == thickness
                    and key[5] == damping_method
                }

                if len(filtered_tfs) > 0:
                    tf_arrays = []
                    common_freq = None
                    for freq, tf in filtered_tfs.values():
                        if common_freq is None:
                            common_freq = freq
                        tf_arrays.append(tf)

                    geomean = compute_geometric_mean(tf_arrays)
                    label = f"rH={rH:.0f}, CV={cv:.1f}"
                    ax.loglog(
                        common_freq,
                        geomean,
                        color=colors_map[(rH, cv)],
                        linewidth=2.0,
                        linestyle="-",
                        label=label,
                    )

            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Transfer Function Magnitude")
            ax.set_title(f"Vs1={Vs1:.0f} m/s")
            ax.set_xlim(1e-1, 5e1)
            ax.set_ylim(1e-2, 1e2)
            ax.grid(True, alpha=0.3)
            if idx == 0:  # Only show legend on first subplot
                ax.legend(loc="best", fontsize=7, ncol=2)

        plt.suptitle(
            f"Velocity-Dependent Sensitivity (h={thickness:.0f} m, damping={damping_method})",
            fontsize=16,
            y=1.02,
        )
        plt.tight_layout()
        output_file = (
            output_dir
            / f"validation_velocity_dependent_sensitivity_damping_{damping_method}.png"
        )
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {output_file}")
        plt.close()


def find_peak_amplification(
    freq: np.ndarray,
    tf: np.ndarray,
    mode_number: int = 1,
    Vs1: float | None = None,
    thickness: float | None = None,
) -> float:
    """
    Find peak amplification at a specific mode.

    Args:
        freq: Frequency array
        tf: Transfer function array
        mode_number: Mode number (1 for first mode, 3 for third mode, etc.)
        Vs1: Shear wave velocity (optional, for theoretical frequency calculation)
        thickness: Layer thickness (optional, for theoretical frequency calculation)

    Returns:
        Peak amplification value
    """
    # If we have Vs1 and thickness, find peak near theoretical frequency
    if Vs1 is not None and thickness is not None:
        theoretical_fn = (2 * mode_number - 1) * Vs1 / (4 * thickness)
        # Look for peak within ±50% of theoretical frequency
        freq_range = theoretical_fn * np.array([0.5, 1.5])
        mask = (freq >= freq_range[0]) & (freq <= freq_range[1])
        if np.any(mask):
            return np.max(tf[mask])

    # Fallback: use frequency ranges
    if mode_number == 1:
        # First mode: look in lower frequency range
        mask = freq <= 10.0
        if np.any(mask):
            return np.max(tf[mask])
    elif mode_number == 3:
        # Third mode: look in mid frequency range
        mask = (freq >= 5.0) & (freq <= 20.0)
        if np.any(mask):
            return np.max(tf[mask])

    # Fallback: return overall maximum
    return np.max(tf)


def plot_amplification_vs_CV(
    tf_dict: Dict,
    tf_dict_1D: Dict,
    output_dir: Path,
    thickness: float = 76.0,
):
    """
    Plot 5: Amplification vs CV summary.

    For each Vs1 and depth:
    - Peak amplification at first mode
    - Peak amplification at higher mode (3rd or 4th peak)
    - Plot amplification vs CV for rH=50 m and rH=10 m as two lines
    - Include uniform 1D peaks as reference lines

    Args:
        tf_dict: Dictionary of variability TFs keyed by (rH, CV, Vs1, thickness, seed, damping_method)
        tf_dict_1D: Dictionary of 1D uniform TFs keyed by (Vs1, thickness, damping_method)
        output_dir: Output directory for plots
        thickness: Fixed thickness value (default 76.0 m)
    """
    Vs1_values = sorted(set(k[2] for k in tf_dict.keys()))
    CV_values = sorted(
        set(k[1] for k in tf_dict.keys())
    )  # Use available CV values from data
    rH_values = sorted(
        set(k[0] for k in tf_dict.keys())
    )  # Use available rH values from data
    damping_methods = sorted(set(k[5] for k in tf_dict.keys()))

    for damping_method in damping_methods:
        n_Vs1 = len(Vs1_values)
        fig, axes = plt.subplots(2, n_Vs1, figsize=(6 * n_Vs1, 12))

        if n_Vs1 == 1:
            axes = axes.reshape(-1, 1)

        for col_idx, Vs1 in enumerate(Vs1_values):
            # Compute uniform 1D peaks for reference
            uniform_key = (Vs1, thickness, damping_method)
            uniform_peak_1st = None
            uniform_peak_3rd = None

            if uniform_key in tf_dict_1D:
                uniform_freq, uniform_tf = tf_dict_1D[uniform_key]
                uniform_peak_1st = find_peak_amplification(
                    uniform_freq,
                    uniform_tf,
                    mode_number=1,
                    Vs1=Vs1,
                    thickness=thickness,
                )
                uniform_peak_3rd = find_peak_amplification(
                    uniform_freq,
                    uniform_tf,
                    mode_number=3,
                    Vs1=Vs1,
                    thickness=thickness,
                )

            # Top row: First mode peak amplification
            ax1 = axes[0, col_idx]

            for rH in rH_values:
                amplifications = []
                for cv in CV_values:
                    filtered_tfs = {
                        key: value
                        for key, value in tf_dict.items()
                        if key[0] == rH
                        and key[1] == cv
                        and key[2] == Vs1
                        and key[3] == thickness
                        and key[5] == damping_method
                    }

                    if len(filtered_tfs) > 0:
                        tf_arrays = []
                        common_freq = None
                        for freq, tf in filtered_tfs.values():
                            if common_freq is None:
                                common_freq = freq
                            tf_arrays.append(tf)

                        geomean = compute_geometric_mean(tf_arrays)
                        peak_amp = find_peak_amplification(
                            common_freq if common_freq is not None else np.array([]),
                            geomean,
                            mode_number=1,
                            Vs1=Vs1,
                            thickness=thickness,
                        )
                        amplifications.append(peak_amp)
                    else:
                        amplifications.append(np.nan)

                color = "blue" if rH == 10.0 else "red"
                ax1.plot(
                    CV_values,
                    amplifications,
                    marker="o",
                    color=color,
                    linewidth=2.0,
                    label=f"rH={rH:.0f} m",
                )

            # Add uniform 1D reference line
            if uniform_peak_1st is not None:
                ax1.axhline(
                    uniform_peak_1st,
                    color="black",
                    linestyle="--",
                    linewidth=2.0,
                    alpha=0.7,
                    label="Uniform (1D)",
                )

            ax1.set_xlabel("CV")
            ax1.set_ylabel("Peak Amplification (1st mode)")
            ax1.set_title(
                f"Vs1={Vs1:.0f} m/s, h={thickness:.0f} m, damping={damping_method}"
            )
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc="best")
            ax1.set_xticks(CV_values)

            # Bottom row: Higher mode peak amplification
            ax2 = axes[1, col_idx]

            for rH in rH_values:
                amplifications = []
                for cv in CV_values:
                    filtered_tfs = {
                        key: value
                        for key, value in tf_dict.items()
                        if key[0] == rH
                        and key[1] == cv
                        and key[2] == Vs1
                        and key[3] == thickness
                        and key[5] == damping_method
                    }

                    if len(filtered_tfs) > 0:
                        tf_arrays = []
                        common_freq = None
                        for freq, tf in filtered_tfs.values():
                            if common_freq is None:
                                common_freq = freq
                            tf_arrays.append(tf)

                        geomean = compute_geometric_mean(tf_arrays)
                        peak_amp = find_peak_amplification(
                            common_freq if common_freq is not None else np.array([]),
                            geomean,
                            mode_number=3,
                            Vs1=Vs1,
                            thickness=thickness,
                        )
                        amplifications.append(peak_amp)
                    else:
                        amplifications.append(np.nan)

                color = "blue" if rH == 10.0 else "red"
                ax2.plot(
                    CV_values,
                    amplifications,
                    marker="s",
                    color=color,
                    linewidth=2.0,
                    label=f"rH={rH:.0f} m",
                )

            # Add uniform 1D reference line
            if uniform_peak_3rd is not None:
                ax2.axhline(
                    uniform_peak_3rd,
                    color="black",
                    linestyle="--",
                    linewidth=2.0,
                    alpha=0.7,
                    label="Uniform (1D)",
                )

            ax2.set_xlabel("CV")
            ax2.set_ylabel("Peak Amplification (3rd mode)")
            ax2.set_title(
                f"Vs1={Vs1:.0f} m/s, h={thickness:.0f} m, damping={damping_method}"
            )
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc="best")
            ax2.set_xticks(CV_values)

        plt.tight_layout()
        output_file = (
            output_dir / f"validation_amplification_vs_CV_damping_{damping_method}.png"
        )
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {output_file}")
        plt.close()


def plot_damping_method_comparison(
    tf_dict: Dict,
    tf_dict_1D: Dict,
    output_dir: Path,
    rH: float = 50.0,
    CV: float = 0.3,
    thickness: float = 76.0,
):
    """
    Compare damping methods side-by-side in the same plot.

    Args:
        tf_dict: Dictionary of variability TFs keyed by (rH, CV, Vs1, thickness, seed, damping_method)
        tf_dict_1D: Dictionary of 1D uniform TFs keyed by (Vs1, thickness, damping_method)
        output_dir: Output directory for plots
        rH: Fixed rH value
        CV: Fixed CV value
        thickness: Fixed thickness value
    """
    Vs1_values = sorted(set(k[2] for k in tf_dict.keys()))
    damping_methods = sorted(set(k[5] for k in tf_dict.keys()))

    for Vs1 in Vs1_values:
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.colormaps["tab10"](np.linspace(0, 1, len(damping_methods)))

        for damping_method, color in zip(damping_methods, colors):
            # Get uniform TF
            uniform_key = (Vs1, thickness, damping_method)
            if uniform_key in tf_dict_1D:
                uniform_freq, uniform_tf = tf_dict_1D[uniform_key]
                ax.loglog(
                    uniform_freq,
                    uniform_tf,
                    color=color,
                    linewidth=2.0,
                    linestyle="-",
                    alpha=0.7,
                    label=f"Uniform (1D), {damping_method}",
                )

            # Get variability geomean
            filtered_tfs = {
                key: value
                for key, value in tf_dict.items()
                if key[0] == rH
                and key[1] == CV
                and key[2] == Vs1
                and key[3] == thickness
                and key[5] == damping_method
            }

            if len(filtered_tfs) > 0:
                tf_arrays = []
                common_freq = None
                for freq, tf in filtered_tfs.values():
                    if common_freq is None:
                        common_freq = freq
                    tf_arrays.append(tf)

                geomean = compute_geometric_mean(tf_arrays)
                ax.loglog(
                    common_freq,
                    geomean,
                    color=color,
                    linewidth=2.5,
                    linestyle="--",
                    label=f"rH={rH:.0f}, CV={CV:.2f}, {damping_method}",
                )

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Transfer Function Magnitude")
        ax.set_title(
            f"Damping Method Comparison: Vs1={Vs1:.0f} m/s, h={thickness:.0f} m, rH={rH:.0f} m, CV={CV:.2f}"
        )
        ax.set_xlim(1e-1, 5e1)
        ax.set_ylim(1e-2, 1e2)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

        plt.tight_layout()
        output_file = (
            output_dir / f"damping_comparison_Vs1_{Vs1:.0f}_rH_{rH:.0f}_CV_{CV:.2f}.png"
        )
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {output_file}")
        plt.close()


def plot_damping_method_subplots(
    tf_dict: Dict,
    tf_dict_1D: Dict,
    output_dir: Path,
    rH: float = 50.0,
    CV: float = 0.3,
    thickness: float = 76.0,
):
    """
    Compare damping methods in separate subplots.

    Args:
        tf_dict: Dictionary of variability TFs keyed by (rH, CV, Vs1, thickness, seed, damping_method)
        tf_dict_1D: Dictionary of 1D uniform TFs keyed by (Vs1, thickness, damping_method)
        output_dir: Output directory for plots
        rH: Fixed rH value
        CV: Fixed CV value
        thickness: Fixed thickness value
    """
    Vs1_values = sorted(set(k[2] for k in tf_dict.keys()))
    damping_methods = sorted(set(k[5] for k in tf_dict.keys()))

    for Vs1 in Vs1_values:
        n_methods = len(damping_methods)
        fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 6))
        if n_methods == 1:
            axes = [axes]

        for idx, damping_method in enumerate(damping_methods):
            ax = axes[idx]

            # Get uniform TF
            uniform_key = (Vs1, thickness, damping_method)
            if uniform_key in tf_dict_1D:
                uniform_freq, uniform_tf = tf_dict_1D[uniform_key]
                ax.loglog(
                    uniform_freq,
                    uniform_tf,
                    color="black",
                    linewidth=2.5,
                    linestyle="-",
                    label="Uniform (1D)",
                )

            # Get variability geomean
            filtered_tfs = {
                key: value
                for key, value in tf_dict.items()
                if key[0] == rH
                and key[1] == CV
                and key[2] == Vs1
                and key[3] == thickness
                and key[5] == damping_method
            }

            if len(filtered_tfs) > 0:
                tf_arrays = []
                common_freq = None
                for freq, tf in filtered_tfs.values():
                    if common_freq is None:
                        common_freq = freq
                    tf_arrays.append(tf)

                geomean = compute_geometric_mean(tf_arrays)
                ax.loglog(
                    common_freq,
                    geomean,
                    color="red",
                    linewidth=2.5,
                    linestyle="--",
                    label=f"rH={rH:.0f}, CV={CV:.2f} (geomean)",
                )

            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("Transfer Function Magnitude")
            ax.set_title(f"{damping_method}")
            ax.set_xlim(1e-1, 5e1)
            ax.set_ylim(1e-2, 1e2)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)

        plt.suptitle(
            f"Damping Method Comparison: Vs1={Vs1:.0f} m/s, h={thickness:.0f} m, rH={rH:.0f} m, CV={CV:.2f}",
            fontsize=14,
            y=1.02,
        )
        plt.tight_layout()
        output_file = (
            output_dir
            / f"damping_comparison_subplots_Vs1_{Vs1:.0f}_rH_{rH:.0f}_CV_{CV:.2f}.png"
        )
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {output_file}")
        plt.close()


def merge_timing_data(script_dir: Path) -> Path:
    """
    Merge all timing_data_task*.csv files from 1D_results/ and results/ directories.

    Args:
        script_dir: Script directory path

    Returns:
        Path to merged CSV file
    """
    merged_file = script_dir / "timing_data_merged.csv"

    timing_files = []

    # Collect timing files from 1D_results/
    results_1D_dir = script_dir / "1D_results"
    if results_1D_dir.exists():
        timing_files.extend(results_1D_dir.glob("timing_data_task_*.csv"))

    # Collect timing files from results/ directory
    results_dir = script_dir / "results"
    if results_dir.exists():
        timing_files.extend(results_dir.glob("timing_data_task_*.csv"))

    if not timing_files:
        print("Warning: No timing files found")
        return merged_file

    print(f"Found {len(timing_files)} timing files to merge")

    all_rows = []
    header_written = False

    for timing_file in sorted(timing_files):
        try:
            with open(timing_file, "r") as f:
                reader = csv.reader(f)
                header = next(reader)
                if not header_written:
                    all_rows.append(header)
                    header_written = True
                for row in reader:
                    all_rows.append(row)
        except Exception as e:
            print(f"Warning: Could not read {timing_file}: {e}")

    # Write merged file
    with open(merged_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(all_rows)

    print(f"Merged {len(all_rows) - 1} timing records to {merged_file}")
    return merged_file


def extract_damping_method_from_task_id(task_id: str) -> str | None:
    """Extract damping method from task_id string."""
    if "_damping_method_" in task_id:
        parts = task_id.split("_damping_method_")
        if len(parts) > 1:
            return parts[1]
    return None


def main():
    """Main function to create all validation plots."""
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Set up paths
    results_dir = script_dir / "results"
    results_1D_dir = script_dir / "1D_results"
    output_dir = script_dir / "transfer_function_results" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Creating Validation Plots")
    print("=" * 60)

    # Set Vsmin for consistent frequency range
    Vsmin = 1000.0  # fmax = 50 Hz for dz=2.0

    # Merge timing data
    print("\nMerging timing data...")
    merge_timing_data(script_dir)

    # Try to load from pickle first
    pickle_file = (
        script_dir / "transfer_function_results" / "transfer_functions_dict.pkl"
    )
    tf_dict = None
    if pickle_file.exists():
        print(f"\nLoading transfer functions from pickle: {pickle_file}")
        tf_dict = load_tf_dict_from_pickle(pickle_file)

    # If pickle doesn't exist or failed, compute from scratch
    if tf_dict is None:
        print("\nComputing transfer functions from scratch...")
        tf_dict, skipped_folders = load_all_results(results_dir, Vsmin=Vsmin)

        if len(tf_dict) == 0:
            print("Error: No variability results loaded. Exiting.")
            return

        # Save to pickle for future use
        pickle_file.parent.mkdir(parents=True, exist_ok=True)
        import pickle

        with open(pickle_file, "wb") as f:
            pickle.dump(tf_dict, f)
        print(f"Saved transfer functions to pickle: {pickle_file}")

    # Load 1D uniform results
    print("\nLoading 1D uniform results...")
    tf_dict_1D = load_1D_results(results_1D_dir, Vsmin=Vsmin)

    if len(tf_dict_1D) == 0:
        print("Warning: No 1D results loaded. Some plots may be incomplete.")

    print(f"\nLoaded {len(tf_dict)} variability transfer functions")
    print(f"Loaded {len(tf_dict_1D)} 1D uniform transfer functions")

    # Create all validation plots
    print("\n" + "=" * 60)
    print("Generating validation plots...")
    print("=" * 60)

    # Plot 1: Uniform vs theoretical
    print("\n1. Creating uniform vs theoretical plots...")
    plot_uniform_vs_theoretical(tf_dict, tf_dict_1D, output_dir)

    # Plot 2: CV effect at fixed rH
    print("\n2. Creating CV effect plots...")
    print("  Creating CV effect plots for rH=50 m...")
    plot_CV_effect_at_fixed_rH(tf_dict, tf_dict_1D, output_dir, rH=50.0, thickness=50.0)
    # Note: rH=10.0 not available in data, skipping
    # plot_CV_effect_at_fixed_rH(tf_dict, tf_dict_1D, output_dir, rH=10.0, thickness=50.0)

    # Plot 3: rH effect at fixed CV
    print("\n3. Creating rH effect plots...")
    plot_rH_effect_at_fixed_CV(tf_dict, tf_dict_1D, output_dir, CV=0.3, thickness=50.0)

    # Plot 4: Velocity-dependent sensitivity
    print("\n4. Creating velocity-dependent sensitivity plot...")
    plot_velocity_dependent_sensitivity(tf_dict, output_dir, thickness=50.0)

    # Plot 5: Amplification vs CV
    print("\n5. Creating amplification vs CV plot...")
    plot_amplification_vs_CV(tf_dict, tf_dict_1D, output_dir, thickness=50.0)

    # Plot 6: Damping method comparison
    print("\n6. Creating damping method comparison plots...")
    print("  Creating side-by-side comparison plots...")
    plot_damping_method_comparison(
        tf_dict, tf_dict_1D, output_dir, rH=50.0, CV=0.3, thickness=50.0
    )
    print("  Creating subplot comparison plots...")
    plot_damping_method_subplots(
        tf_dict, tf_dict_1D, output_dir, rH=50.0, CV=0.3, thickness=50.0
    )

    print("\n" + "=" * 60)
    print("Validation plots complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
