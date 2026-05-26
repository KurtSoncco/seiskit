"""
Compare transfer functions across different Ricker wave frequencies.

This script:
1. Loads all realization results from Ricker wave analysis
2. Computes transfer functions for each realization
3. Groups results by Ricker frequency (motion_freq) and damping method
4. Creates three types of comparisons:
   - Complete: All individual realizations + geometric mean
   - Geomean only: Just the geometric mean for each frequency
   - CoV: Coefficient of Variation for each frequency
"""

import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from seiskit.ttf.TTF import TTF


def parse_result_folder(folder_name: str) -> Optional[Dict]:
    """
    Parse folder name to extract parameters.

    Format: motion_freq_{freq}_s{seed}_damping_method_{method}

    Returns:
        Dictionary with 'motion_freq', 'seed', 'damping_method', or None if parsing fails
    """
    pattern = r"motion_freq_([\d.]+)_s(\d+)_damping_method_(.+)"
    match = re.match(pattern, folder_name)
    if match:
        return {
            "motion_freq": float(match.group(1)),
            "seed": int(match.group(2)),
            "damping_method": match.group(3),
        }
    return None


def find_task_id_dir(folder: Path) -> Optional[Path]:
    """Find the task ID directory inside a result folder."""
    # Look for directories that start with the case type pattern
    for subdir in folder.iterdir():
        if subdir.is_dir() and "motion_freq" in subdir.name:
            return subdir
    return None


def load_acceleration_files(
    folder: Path,
) -> Optional[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
    """
    Load base and top acceleration files from a result folder.

    Returns:
        ((base_time, base_accel), (top_time, top_accel)) or None if files not found
    """
    task_dir = find_task_id_dir(folder)
    if task_dir is None:
        return None

    # Base is at y=2.00 m
    base_file = task_dir / "center_node_y2.00_dof1_accel.txt"
    # Top is at y=110.00 m (thickness=100m + 10m bedrock, so surface is at 110m)
    top_file = task_dir / "center_node_y110.00_dof1_accel.txt"

    if not base_file.exists() or not top_file.exists():
        return None

    try:
        # Load data: first column is time, second is acceleration
        base_data = np.loadtxt(base_file)
        top_data = np.loadtxt(top_file)

        base_time = base_data[:, 0]
        base_accel = base_data[:, 1]
        top_time = top_data[:, 0]
        top_accel = top_data[:, 1]

        return (base_time, base_accel), (top_time, top_accel)
    except Exception as e:
        print(f"Error loading files from {folder.name}: {e}")
        return None


def compute_transfer_function(
    base_data: Tuple[np.ndarray, np.ndarray],
    top_data: Tuple[np.ndarray, np.ndarray],
    dz: float = 2.0,
    dt: float = 0.01,
    Vsmin: float = 1000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute transfer function from base and top acceleration data.

    Args:
        base_data: (time, acceleration) tuple for base
        top_data: (time, acceleration) tuple for top
        dz: Depth increment (default 2.0 m)
        dt: Time step (default 0.01 s)
        Vsmin: Minimum shear wave velocity for fmax calculation (default 1000 m/s)

    Returns:
        (freq, tf) tuple
    """
    base_time, base_accel = base_data
    top_time, top_accel = top_data

    # Interpolate the data to dt/10 for more accurate transfer function
    base_time_interpolated = np.arange(base_time[0], base_time[-1], dt / 10)
    base_accel_interpolated = interp1d(base_time, base_accel)(base_time_interpolated)
    top_time_interpolated = np.arange(top_time[0], top_time[-1], dt / 10)
    top_accel_interpolated = interp1d(top_time, top_accel)(top_time_interpolated)

    # Calculate Nyquist frequency to avoid interpolation errors
    nyquist_freq = 1.0 / (2.0 * dt / 10)

    # Calculate desired fmax from Vsmin
    desired_fmax = Vsmin / (10 * dz)

    # Ensure fmax doesn't exceed Nyquist (with safety margin)
    max_allowed_fmax = nyquist_freq * 0.95
    if desired_fmax > max_allowed_fmax:
        print(
            f"Warning: Requested fmax={desired_fmax:.1f} Hz exceeds safe limit ({max_allowed_fmax:.1f} Hz). "
            f"Limiting to {max_allowed_fmax:.1f} Hz."
        )
        safe_vsmin = max_allowed_fmax * 10 * dz
        Vsmin = safe_vsmin

    # Compute transfer function
    actual_fmax = Vsmin / (10 * dz)
    try:
        freq, tf = TTF(
            top_accel_interpolated,
            base_accel_interpolated,
            dt=dt / 10,
            dz=dz,
            Vsmin=Vsmin,
        )
    except (ValueError, Exception) as e:
        # If TTF fails, try with a safe Vsmin based on Nyquist
        print(
            f"Warning: TTF failed with Vsmin={Vsmin:.0f} (fmax={desired_fmax:.1f} Hz). "
            f"Error: {e}. Trying with safe Vsmin."
        )
        safe_vsmin = max_allowed_fmax * 10 * dz
        freq, tf = TTF(
            top_accel_interpolated,
            base_accel_interpolated,
            dt=dt / 10,
            dz=dz,
            Vsmin=safe_vsmin,
        )

    return freq, tf


def compute_geometric_mean(tf_list: List[np.ndarray]) -> np.ndarray:
    """
    Compute geometric mean across transfer functions.

    Args:
        tf_list: List of transfer function arrays (all same length)

    Returns:
        Geometric mean array
    """
    # Stack arrays
    tf_array = np.array(tf_list)

    # Compute geometric mean: exp(mean(log(x)))
    log_tf = np.log(np.abs(tf_array) + 1e-12)  # Add small epsilon to avoid log(0)
    geomean = np.exp(np.mean(log_tf, axis=0))

    return geomean


def compute_coefficient_of_variation(tf_list: List[np.ndarray]) -> np.ndarray:
    """
    Compute coefficient of variation (CV) across transfer functions.
    CV = std / mean

    Args:
        tf_list: List of transfer function arrays (all same length)

    Returns:
        Coefficient of variation array
    """
    # Stack arrays
    tf_array = np.array(tf_list)

    # Compute mean and standard deviation
    mean_tf = np.mean(tf_array, axis=0)
    std_tf = np.std(tf_array, axis=0)

    # Compute CV = std / mean
    # Add small epsilon to avoid division by zero
    cv = std_tf / (np.abs(mean_tf) + 1e-12)

    return cv


def interpolate_to_common_frequency(
    freq_list: List[np.ndarray], tf_list: List[np.ndarray], common_freq: np.ndarray
) -> List[np.ndarray]:
    """
    Interpolate all transfer functions to a common frequency grid.

    Args:
        freq_list: List of frequency arrays
        tf_list: List of transfer function arrays
        common_freq: Common frequency grid to interpolate to

    Returns:
        List of interpolated transfer function arrays
    """
    interpolated_tfs = []
    for freq, tf in zip(freq_list, tf_list):
        # Use log interpolation for better accuracy
        log_freq = np.log10(freq + 1e-12)
        log_tf = np.log10(np.abs(tf) + 1e-12)
        log_common_freq = np.log10(common_freq + 1e-12)

        # Interpolate in log space
        log_tf_interp = np.interp(log_common_freq, log_freq, log_tf)
        tf_interp = 10**log_tf_interp

        interpolated_tfs.append(tf_interp)

    return interpolated_tfs


def load_all_results(
    results_dir: Path,
    Vsmin: float = 1000.0,
) -> Dict[Tuple[float, str], List[Tuple[np.ndarray, np.ndarray, int]]]:
    """
    Load all transfer functions from result directories.

    Returns:
        Dictionary keyed by (motion_freq, damping_method) -> List of (freq, tf, seed)
    """
    tf_dict = defaultdict(list)
    results_path = Path(results_dir)

    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_path}")

    print(f"Scanning results directory: {results_path}")
    folders = sorted([f for f in results_path.iterdir() if f.is_dir()])

    loaded_count = 0
    skipped_count = 0

    for folder in folders:
        params = parse_result_folder(folder.name)
        if params is None:
            skipped_count += 1
            continue

        # Load acceleration files
        accel_data = load_acceleration_files(folder)
        if accel_data is None:
            skipped_count += 1
            continue

        base_data, top_data = accel_data

        # Compute transfer function
        try:
            freq, tf = compute_transfer_function(base_data, top_data, Vsmin=Vsmin)

            # Store in dictionary grouped by (motion_freq, damping_method)
            key = (params["motion_freq"], params["damping_method"])
            tf_dict[key].append((freq, tf, params["seed"]))
            loaded_count += 1

        except Exception as e:
            print(f"Error computing TF for {folder.name}: {e}")
            skipped_count += 1
            continue

    print(f"Loaded {loaded_count} transfer functions")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} folders")

    return tf_dict


def create_comparison_plots(
    tf_dict: Dict[Tuple[float, str], List[Tuple[np.ndarray, np.ndarray, int]]],
    output_dir: Path,
):
    """
    Create three types of comparison plots:
    1. Complete: All realizations + geomean
    2. Geomean only: Just geomean for each frequency
    3. CoV: Coefficient of variation for each frequency
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by damping method
    damping_methods = sorted(set(k[1] for k in tf_dict.keys()))

    for damping_method in damping_methods:
        # Get all frequencies for this damping method
        freq_keys = [k for k in tf_dict.keys() if k[1] == damping_method]
        motion_freqs = sorted(set(k[0] for k in freq_keys))

        if len(motion_freqs) == 0:
            continue

        # Find common frequency grid (use intersection of all frequency ranges)
        all_freqs = []
        for key in freq_keys:
            for freq, _, _ in tf_dict[key]:
                all_freqs.append(freq)

        if len(all_freqs) == 0:
            continue

        # Create common frequency grid from min to max of all frequencies
        min_freq = min(f.min() for f in all_freqs)
        max_freq = min(f.max() for f in all_freqs)  # Use min of maxes to ensure all cover it
        common_freq = np.logspace(np.log10(min_freq), np.log10(max_freq), 1000)

        # Prepare data for plotting
        plot_data = {}
        for motion_freq in motion_freqs:
            key = (motion_freq, damping_method)
            if key not in tf_dict:
                continue

            freq_list = []
            tf_list = []
            seeds = []

            for freq, tf, seed in tf_dict[key]:
                freq_list.append(freq)
                tf_list.append(tf)
                seeds.append(seed)

            # Interpolate to common frequency
            tf_interp_list = interpolate_to_common_frequency(freq_list, tf_list, common_freq)

            # Compute geomean
            geomean = compute_geometric_mean(tf_interp_list)

            # Compute CoV
            cv = compute_coefficient_of_variation(tf_interp_list)

            plot_data[motion_freq] = {
                "freq": common_freq,
                "individual_tfs": tf_interp_list,
                "seeds": seeds,
                "geomean": geomean,
                "cv": cv,
            }

        # Plot 1: Complete (all realizations + geomean)
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(motion_freqs)))

        for idx, motion_freq in enumerate(motion_freqs):
            data = plot_data[motion_freq]
            color = colors[idx]

            # Plot individual realizations (light, thin lines)
            for tf, seed in zip(data["individual_tfs"], data["seeds"]):
                ax.loglog(
                    data["freq"],
                    tf,
                    color=color,
                    alpha=0.3,
                    linewidth=0.8,
                    linestyle="--",
                )

            # Plot geomean (thick, solid line)
            ax.loglog(
                data["freq"],
                data["geomean"],
                color=color,
                linewidth=2.5,
                label=f"Ricker freq = {motion_freq:.1f} Hz (geomean, n={len(data['individual_tfs'])})",
            )

        ax.set_xlabel("Frequency (Hz)", fontsize=12)
        ax.set_ylabel("Transfer Function", fontsize=12)
        ax.set_title(
            f"Transfer Functions: Complete Comparison\n"
            f"Damping Method: {damping_method}\n"
            f"(Individual realizations shown as thin dashed lines, geomean as thick solid lines)",
            fontsize=11,
        )
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlim([min_freq, max_freq])

        plt.tight_layout()
        output_file = output_dir / f"transfer_function_complete_{damping_method}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved complete comparison plot: {output_file}")

        # Plot 2: Geomean only
        fig, ax = plt.subplots(figsize=(10, 6))

        for idx, motion_freq in enumerate(motion_freqs):
            data = plot_data[motion_freq]
            color = colors[idx]

            ax.loglog(
                data["freq"],
                data["geomean"],
                color=color,
                linewidth=2.5,
                label=f"Ricker freq = {motion_freq:.1f} Hz (n={len(data['individual_tfs'])})",
            )

        ax.set_xlabel("Frequency (Hz)", fontsize=12)
        ax.set_ylabel("Transfer Function (Geometric Mean)", fontsize=12)
        ax.set_title(
            f"Transfer Functions: Geometric Mean Comparison\nDamping Method: {damping_method}",
            fontsize=11,
        )
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlim([min_freq, max_freq])

        plt.tight_layout()
        output_file = output_dir / f"transfer_function_geomean_only_{damping_method}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved geomean-only comparison plot: {output_file}")

        # Plot 3: Coefficient of Variation
        fig, ax = plt.subplots(figsize=(10, 6))

        for idx, motion_freq in enumerate(motion_freqs):
            data = plot_data[motion_freq]
            color = colors[idx]

            ax.semilogx(
                data["freq"],
                data["cv"],
                color=color,
                linewidth=2.5,
                label=f"Ricker freq = {motion_freq:.1f} Hz (n={len(data['individual_tfs'])})",
            )

        ax.set_xlabel("Frequency (Hz)", fontsize=12)
        ax.set_ylabel("Coefficient of Variation (CoV)", fontsize=12)
        ax.set_title(
            f"Transfer Functions: Coefficient of Variation Comparison\n"
            f"Damping Method: {damping_method}",
            fontsize=11,
        )
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlim([min_freq, max_freq])

        plt.tight_layout()
        output_file = output_dir / f"transfer_function_cov_{damping_method}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved CoV comparison plot: {output_file}")


def main():
    """Main function to run the comparison."""
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    results_dir = script_dir / "results"
    output_dir = script_dir / "transfer_function_comparison"

    print("=" * 80)
    print("Ricker Wave Frequency Comparison")
    print("=" * 80)

    # Load all results
    print("\nLoading transfer functions...")
    tf_dict = load_all_results(results_dir, Vsmin=1000.0)

    if len(tf_dict) == 0:
        print("No transfer functions loaded. Exiting.")
        return

    print(f"\nFound {len(tf_dict)} unique (motion_freq, damping_method) combinations")
    for key, tfs in tf_dict.items():
        print(f"  {key}: {len(tfs)} realizations")

    # Create comparison plots
    print("\nCreating comparison plots...")
    create_comparison_plots(tf_dict, output_dir)

    print("\n" + "=" * 80)
    print("Comparison complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
