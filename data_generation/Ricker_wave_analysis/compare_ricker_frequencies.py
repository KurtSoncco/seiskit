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

All figures follow the centralized publication-quality style from
:mod:`seiskit.plot_config`.
"""

import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from seiskit.plot_config import (
    COLORBLIND_COLORS,
    add_subfigure_label,
    apply_style,
    format_title,
    get_crameri_cmap,
    place_legend,
    to_title_case,
)
from seiskit.plot_config.labels import format_label
from seiskit.ttf.TTF import TTF


def parse_result_folder(folder_name: str) -> Optional[Dict]:
    """Parse folder name to extract parameters."""
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
    for subdir in folder.iterdir():
        if subdir.is_dir() and "motion_freq" in subdir.name:
            return subdir
    return None


def load_acceleration_files(
    folder: Path,
) -> Optional[Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
    """Load base and top acceleration files from a result folder."""
    task_dir = find_task_id_dir(folder)
    if task_dir is None:
        return None

    base_file = task_dir / "center_node_y2.00_dof1_accel.txt"
    top_file = task_dir / "center_node_y110.00_dof1_accel.txt"

    if not base_file.exists() or not top_file.exists():
        return None

    try:
        base_data = np.loadtxt(base_file)
        top_data = np.loadtxt(top_file)
        return (base_data[:, 0], base_data[:, 1]), (top_data[:, 0], top_data[:, 1])
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
    """Compute transfer function from base and top acceleration data."""
    base_time, base_accel = base_data
    top_time, top_accel = top_data

    base_time_interpolated = np.arange(base_time[0], base_time[-1], dt / 10)
    base_accel_interpolated = interp1d(base_time, base_accel)(base_time_interpolated)
    top_time_interpolated = np.arange(top_time[0], top_time[-1], dt / 10)
    top_accel_interpolated = interp1d(top_time, top_accel)(top_time_interpolated)

    nyquist_freq = 1.0 / (2.0 * dt / 10)
    desired_fmax = Vsmin / (10 * dz)
    max_allowed_fmax = nyquist_freq * 0.95

    if desired_fmax > max_allowed_fmax:
        print(
            f"Warning: Requested fmax={desired_fmax:.1f} Hz exceeds safe limit ({max_allowed_fmax:.1f} Hz). "
            f"Limiting to {max_allowed_fmax:.1f} Hz."
        )
        Vsmin = max_allowed_fmax * 10 * dz

    try:
        freq, tf = TTF(
            top_accel_interpolated, base_accel_interpolated,
            dt=dt / 10, dz=dz, Vsmin=Vsmin,
        )
    except (ValueError, Exception) as e:
        print(
            f"Warning: TTF failed with Vsmin={Vsmin:.0f} (fmax={desired_fmax:.1f} Hz). "
            f"Error: {e}. Trying with safe Vsmin."
        )
        safe_vsmin = max_allowed_fmax * 10 * dz
        freq, tf = TTF(
            top_accel_interpolated, base_accel_interpolated,
            dt=dt / 10, dz=dz, Vsmin=safe_vsmin,
        )

    return freq, tf


def compute_geometric_mean(tf_list: List[np.ndarray]) -> np.ndarray:
    """Compute geometric mean across transfer functions."""
    tf_array = np.array(tf_list)
    log_tf = np.log(np.abs(tf_array) + 1e-12)
    return np.exp(np.mean(log_tf, axis=0))


def compute_coefficient_of_variation(tf_list: List[np.ndarray]) -> np.ndarray:
    """Compute coefficient of variation (CV) across transfer functions."""
    tf_array = np.array(tf_list)
    mean_tf = np.mean(tf_array, axis=0)
    std_tf = np.std(tf_array, axis=0)
    return std_tf / (np.abs(mean_tf) + 1e-12)


def interpolate_to_common_frequency(
    freq_list: List[np.ndarray], tf_list: List[np.ndarray], common_freq: np.ndarray
) -> List[np.ndarray]:
    """Interpolate all transfer functions to a common frequency grid."""
    interpolated_tfs = []
    for freq, tf in zip(freq_list, tf_list):
        log_freq = np.log10(freq + 1e-12)
        log_tf = np.log10(np.abs(tf) + 1e-12)
        log_common_freq = np.log10(common_freq + 1e-12)
        log_tf_interp = np.interp(log_common_freq, log_freq, log_tf)
        interpolated_tfs.append(10**log_tf_interp)
    return interpolated_tfs


def load_all_results(
    results_dir: Path, Vsmin: float = 1000.0,
) -> Dict[Tuple[float, str], List[Tuple[np.ndarray, np.ndarray, int]]]:
    """Load all transfer functions from result directories."""
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

        accel_data = load_acceleration_files(folder)
        if accel_data is None:
            skipped_count += 1
            continue

        base_data, top_data = accel_data

        try:
            freq, tf = compute_transfer_function(base_data, top_data, Vsmin=Vsmin)
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
    """Create three types of comparison plots."""
    apply_style()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    damping_methods = sorted(set(k[1] for k in tf_dict.keys()))

    for damping_method in damping_methods:
        freq_keys = [k for k in tf_dict.keys() if k[1] == damping_method]
        motion_freqs = sorted(set(k[0] for k in freq_keys))

        if len(motion_freqs) == 0:
            continue

        all_freqs = []
        for key in freq_keys:
            for freq, _, _ in tf_dict[key]:
                all_freqs.append(freq)

        if len(all_freqs) == 0:
            continue

        min_freq = min(f.min() for f in all_freqs)
        max_freq = min(f.max() for f in all_freqs)
        common_freq = np.logspace(np.log10(min_freq), np.log10(max_freq), 1000)

        plot_data = {}
        for motion_freq in motion_freqs:
            key = (motion_freq, damping_method)
            if key not in tf_dict:
                continue

            freq_list, tf_list, seeds = [], [], []
            for freq, tf, seed in tf_dict[key]:
                freq_list.append(freq)
                tf_list.append(tf)
                seeds.append(seed)

            tf_interp_list = interpolate_to_common_frequency(freq_list, tf_list, common_freq)
            geomean = compute_geometric_mean(tf_interp_list)
            cv = compute_coefficient_of_variation(tf_interp_list)

            plot_data[motion_freq] = {
                "freq": common_freq,
                "individual_tfs": tf_interp_list,
                "seeds": seeds,
                "geomean": geomean,
                "cv": cv,
            }

        n_freqs = len(motion_freqs)
        colors = COLORBLIND_COLORS[:n_freqs] if n_freqs <= len(COLORBLIND_COLORS) else [
            COLORBLIND_COLORS[i % len(COLORBLIND_COLORS)] for i in range(n_freqs)
        ]

        # Plot 1: Complete (all realizations + geomean)
        fig, ax = plt.subplots(figsize=(10, 6))

        for idx, motion_freq in enumerate(motion_freqs):
            data = plot_data[motion_freq]
            color = colors[idx]

            for tf, seed in zip(data["individual_tfs"], data["seeds"]):
                ax.loglog(
                    data["freq"], tf,
                    color=color, alpha=0.3, linewidth=0.8, linestyle="--",
                )

            ax.loglog(
                data["freq"], data["geomean"],
                color=color, linewidth=2.5,
                label=f"Ricker freq = {motion_freq:.1f} Hz (geomean, n={len(data['individual_tfs'])})",
            )

        ax.set_xlabel(to_title_case("Frequency (Hz)"))
        ax.set_ylabel(to_title_case("Transfer Function"))
        format_title(
            "Transfer Functions: Complete Comparison",
            subtitle=(
                f"Damping Method: {damping_method}\n"
                "(Individual realizations shown as thin dashed lines, geomean as thick solid lines)"
            ),
            ax=ax,
        )
        place_legend(ax, position="bottom")
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlim([min_freq, max_freq])
        add_subfigure_label(ax, 0)

        plt.tight_layout()
        output_file = output_dir / f"transfer_function_complete_{damping_method}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved complete comparison plot: {output_file}")

        # Plot 2: Geomean only
        fig, ax = plt.subplots(figsize=(10, 6))

        for idx, motion_freq in enumerate(motion_freqs):
            data = plot_data[motion_freq]
            ax.loglog(
                data["freq"], data["geomean"],
                color=colors[idx], linewidth=2.5,
                label=f"Ricker freq = {motion_freq:.1f} Hz (n={len(data['individual_tfs'])})",
            )

        ax.set_xlabel(to_title_case("Frequency (Hz)"))
        ax.set_ylabel(to_title_case("Transfer Function (Geometric Mean)"))
        format_title(
            "Transfer Functions: Geometric Mean Comparison",
            subtitle=f"Damping Method: {damping_method}",
            ax=ax,
        )
        place_legend(ax, position="bottom")
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlim([min_freq, max_freq])
        add_subfigure_label(ax, 0)

        plt.tight_layout()
        output_file = output_dir / f"transfer_function_geomean_only_{damping_method}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved geomean-only comparison plot: {output_file}")

        # Plot 3: Coefficient of Variation
        fig, ax = plt.subplots(figsize=(10, 6))

        for idx, motion_freq in enumerate(motion_freqs):
            data = plot_data[motion_freq]
            ax.semilogx(
                data["freq"], data["cv"],
                color=colors[idx], linewidth=2.5,
                label=f"Ricker freq = {motion_freq:.1f} Hz (n={len(data['individual_tfs'])})",
            )

        ax.set_xlabel(to_title_case("Frequency (Hz)"))
        ax.set_ylabel(to_title_case(format_label("CoV")))
        format_title(
            "Transfer Functions: " + format_label("CoV") + " Comparison",
            subtitle=f"Damping Method: {damping_method}",
            ax=ax,
        )
        place_legend(ax, position="bottom")
        ax.grid(True, alpha=0.3, which="both")
        ax.set_xlim([min_freq, max_freq])
        add_subfigure_label(ax, 0)

        plt.tight_layout()
        output_file = output_dir / f"transfer_function_cov_{damping_method}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved CoV comparison plot: {output_file}")


def main():
    """Main function to run the comparison."""
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    results_dir = script_dir / "results"
    output_dir = script_dir / "transfer_function_comparison"

    print("=" * 80)
    print("Ricker Wave Frequency Comparison")
    print("=" * 80)

    print("\nLoading transfer functions...")
    tf_dict = load_all_results(results_dir, Vsmin=1000.0)

    if len(tf_dict) == 0:
        print("No transfer functions loaded. Exiting.")
        return

    print(f"\nFound {len(tf_dict)} unique (motion_freq, damping_method) combinations")
    for key, tfs in tf_dict.items():
        print(f"  {key}: {len(tfs)} realizations")

    print("\nCreating comparison plots...")
    create_comparison_plots(tf_dict, output_dir)

    print("\n" + "=" * 80)
    print("Comparison complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
