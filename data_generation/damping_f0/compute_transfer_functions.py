"""
Compute transfer functions for all sensitivity analysis realizations.

This script:
1. Loads all realization results from the sensitivity analysis
2. Computes transfer functions for each realization
3. Groups results by (rH, CV) and (VS1, height)
4. Creates subplots showing all realizations per group
5. Computes and plots geometric means
6. Stores all transfer functions in a dictionary
"""

import os
import pickle
import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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

# Apply centralized publication-quality style
apply_style()


def activate_venv():
    """Activate virtual environment if not already activated."""
    venv_path = Path(__file__).parent.parent.parent / ".venv" / "bin" / "activate"
    if venv_path.exists():
        # Note: This is informational - actual activation should be done in shell
        print(f"Note: Virtual environment available at {venv_path}")
        print(
            "Please ensure virtual environment is activated: source .venv/bin/activate"
        )


def parse_result_folder(folder_name: str) -> Dict[str, float | str] | None:
    """
    Parse result folder name to extract parameters.

    Format: Vs1_{Vs1}_thickness_{thickness}_rH_{rH}_CV_{CV}_s{seed}_damping_method_{damping_method}

    Returns:
        Dictionary with keys: Vs1, thickness, rH, CV, seed, damping_method
        None if parsing fails
    """
    pattern = r"Vs1_(\d+\.?\d*)_thickness_(\d+\.?\d*)_rH_(\d+\.?\d*)_CV_(\d+\.\d+)_s(\d+)_damping_method_([a-zA-Z_]+)"
    match = re.match(pattern, folder_name)
    if match:
        return {
            "Vs1": float(match.group(1)),
            "thickness": float(match.group(2)),
            "rH": float(match.group(3)),
            "CV": float(match.group(4)),
            "seed": int(match.group(5)),
            "damping_method": match.group(6),
        }
    return None


def _extract_numeric_params(params: Dict[str, float | str]) -> Dict[str, float]:
    """
    Extract only numeric parameters from params dict (excludes damping_method).

    Args:
        params: Dictionary that may contain string values (like damping_method)

    Returns:
        Dictionary with only float values
    """
    return {
        "Vs1": float(params["Vs1"]),
        "thickness": float(params["thickness"]),
        "rH": float(params["rH"]),
        "CV": float(params["CV"]),
        "seed": float(params["seed"]),
    }


def compute_index_from_params(params: Dict[str, float]) -> int:
    """
    Compute the index from parameters (reverse of run_experiment.py indexing).

    Index structure: index = Vs1_idx * (2*2*5) + thickness_idx * (2*5) + rH_idx * (5) + CV_idx * (5) + seed_idx
    Order: Vs1 -> thickness -> rH -> CV -> seed
    """
    # Parameter lists (must match run_experiment.py)
    Vs1_list = [100, 360]
    thickness_list = [14, 100]
    rH_list = [10, 50]
    CV_list = [0.1, 0.3]
    seed_values = [10, 20, 30, 40, 50]

    try:
        Vs1_idx = Vs1_list.index(int(params["Vs1"]))
        thickness_idx = thickness_list.index(int(params["thickness"]))
        rH_idx = rH_list.index(int(params["rH"]))
        CV_idx = CV_list.index(float(params["CV"]))
        seed_idx = seed_values.index(int(params["seed"]))

        # Compute index
        index = (
            Vs1_idx
            * (len(thickness_list) * len(rH_list) * len(CV_list) * len(seed_values))
            + thickness_idx * (len(rH_list) * len(CV_list) * len(seed_values))
            + rH_idx * (len(CV_list) * len(seed_values))
            + CV_idx * len(seed_values)
            + seed_idx
        )
        return index
    except ValueError:
        return -1  # Return -1 if parameter not found in lists


def find_task_id_dir(result_dir: Path) -> Path | None:
    """Find the task_id subdirectory containing acceleration files."""
    # Task ID format: 2x2_4node_Vs1{Vs1}_th{thickness}_rH{rH}_CV{CV}_s{seed}
    for subdir in result_dir.iterdir():
        if subdir.is_dir() and subdir.name.startswith("2x2_4node_"):
            return subdir
    return None


def load_acceleration_files(
    result_dir: Path, thickness: float
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]] | None:
    """
    Load base and top acceleration files from a result directory.

    Args:
        result_dir: Path to result directory
        thickness: Thickness value to compute Lz (needed for top file name)

    Returns:
        Tuple of (base_accel, top_accel) as (time, acceleration) arrays
        Returns None if files not found
    """
    task_dir = find_task_id_dir(result_dir)
    if task_dir is None:
        return None

    # Base is at y=2.0, Top is at y=Lz
    # Lz = thickness + 10 (bedrock layer)
    Lz = thickness + 10.0

    base_file = task_dir / "center_node_y2.00_dof1_accel.txt"
    top_file = task_dir / f"center_node_y{Lz:.2f}_dof1_accel.txt"

    if not base_file.exists() or not top_file.exists():
        return None

    # Check if files are empty before attempting to load
    if base_file.stat().st_size == 0 or top_file.stat().st_size == 0:
        return None

    try:
        base_data = np.loadtxt(base_file)
        top_data = np.loadtxt(top_file)

        # Check if loaded data is empty
        if base_data.size == 0 or top_data.size == 0:
            return None

        # Handle both 1D and 2D arrays
        if base_data.ndim == 2:
            if base_data.shape[1] < 2:
                return None
            base_time, base_accel = base_data[:, 0], base_data[:, 1]
        elif base_data.ndim == 1:
            # If 1D, assume it's just acceleration values, create time array
            base_accel = base_data
            dt = 0.01  # Default time step
            base_time = np.arange(len(base_accel), dtype=float) * dt
        else:
            return None

        if top_data.ndim == 2:
            if top_data.shape[1] < 2:
                return None
            top_time, top_accel = top_data[:, 0], top_data[:, 1]
        elif top_data.ndim == 1:
            # If 1D, assume it's just acceleration values, create time array
            top_accel = top_data
            dt = 0.01  # Default time step
            top_time = np.arange(len(top_accel), dtype=float) * dt
        else:
            return None

        # Check if arrays have valid length
        if len(base_time) == 0 or len(top_time) == 0:
            return None

        return (base_time, base_accel), (top_time, top_accel)
    except (ValueError, OSError, IOError):
        # Silently skip files that can't be loaded
        return None
    except (ValueError, OSError, IOError, Exception):
        # Silently skip files with any errors
        return None


def compute_transfer_function(
    base_data: Tuple[np.ndarray, np.ndarray],
    top_data: Tuple[np.ndarray, np.ndarray],
    dz: float = 2.0,
    dt: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute transfer function from base and top acceleration data.

    Args:
        base_data: (time, acceleration) tuple for base
        top_data: (time, acceleration) tuple for top
        dz: Depth increment (default 2.0 m)
        dt: Time step (default 0.01 s)

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

    # Compute transfer function (TTF uses default fmax=10 Hz if Vsmin=None)
    freq, tf = TTF(
        top_accel_interpolated,
        base_accel_interpolated,
        dt=dt / 10,
        dz=dz,
        Vsmin=None,  # Use default fmax=10 Hz
    )

    return freq, tf


def load_all_results(
    results_dir: Path,
) -> Tuple[
    Dict[Tuple[float, float, float, float, int, str], Tuple[np.ndarray, np.ndarray]],
    List[Dict],
]:
    """
    Load all transfer functions from result directories.

    Returns:
        Tuple of (tf_dict, skipped_folders)
        - tf_dict: Dictionary keyed by (rH, CV, Vs1, thickness, seed, damping_method) -> (freq, tf)
        - skipped_folders: List of dicts with 'folder', 'params', 'index', 'reason'
    """
    tf_dict = {}
    skipped_folders = []
    results_path = Path(results_dir)

    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_path}")

    print(f"Scanning results directory: {results_path}")
    folders = sorted([f for f in results_path.iterdir() if f.is_dir()])

    loaded_count = 0
    skipped_count = 0

    # Debug: Print first few folders to check parsing
    print(f"Total folders found: {len(folders)}")
    if len(folders) > 0:
        print(f"First folder: {folders[0].name}")
        test_params = parse_result_folder(folders[0].name)
        print(f"First folder parsed: {test_params}")

    for folder in folders:
        params = parse_result_folder(folder.name)
        if params is None:
            skipped_folders.append(
                {
                    "folder": folder.name,
                    "params": None,
                    "index": -1,
                    "reason": "Could not parse folder name",
                }
            )
            skipped_count += 1
            continue

        # Load acceleration files
        thickness_val = float(params["thickness"])
        accel_data = load_acceleration_files(folder, thickness_val)
        if accel_data is None:
            # Extract only numeric parameters for compute_index_from_params
            index = compute_index_from_params(_extract_numeric_params(params))
            # Check what went wrong with detailed debugging
            task_dir = find_task_id_dir(folder)
            reason = "Unknown error"
            if task_dir is None:
                reason = "Task directory not found"
                # Debug: list what directories exist
                subdirs = [d.name for d in folder.iterdir() if d.is_dir()]
                reason += f" (found subdirs: {subdirs[:3]})"
            else:
                Lz = float(params["thickness"]) + 10.0
                base_file = task_dir / "center_node_y2.00_dof1_accel.txt"
                top_file = task_dir / f"center_node_y{Lz:.2f}_dof1_accel.txt"
                if not base_file.exists():
                    reason = f"Base file not found: {base_file.name}"
                    # Debug: list what files exist
                    files = [f.name for f in task_dir.iterdir() if f.is_file()]
                    reason += f" (found files: {files[:5]})"
                elif not top_file.exists():
                    reason = f"Top file not found: {top_file.name} (Lz={Lz:.2f})"
                    # Debug: list what files exist
                    files = [f.name for f in task_dir.iterdir() if f.is_file()]
                    reason += f" (found files: {files[:5]})"
                else:
                    # Files exist but loading failed - check file sizes
                    try:
                        base_size = base_file.stat().st_size
                        top_size = top_file.stat().st_size
                        reason = f"Files exist but loading failed (base_size={base_size}, top_size={top_size})"
                    except OSError:
                        reason = "Files exist but stat() failed"
            skipped_folders.append(
                {
                    "folder": folder.name,
                    "params": params,
                    "index": index,
                    "reason": reason,
                }
            )
            skipped_count += 1
            continue

        base_data, top_data = accel_data

        # Compute transfer function
        try:
            freq, tf = compute_transfer_function(base_data, top_data)

            # Store in dictionary
            key = (
                params["rH"],
                params["CV"],
                params["Vs1"],
                params["thickness"],
                params["seed"],
                params["damping_method"],
            )
            tf_dict[key] = (freq, tf)
            loaded_count += 1

        except Exception as e:
            # Extract only numeric parameters for compute_index_from_params
            index = compute_index_from_params(_extract_numeric_params(params))
            reason = f"Error computing TF: {e}"
            skipped_folders.append(
                {
                    "folder": folder.name,
                    "params": params,
                    "index": index,
                    "reason": reason,
                }
            )
            skipped_count += 1
            continue

    print(f"Loaded {loaded_count} transfer functions")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} folders")

    return tf_dict, skipped_folders


def load_tf_dict_from_pickle(pickle_file: Path) -> Dict | None:
    """
    Load transfer function dictionary from pickle file.

    Args:
        pickle_file: Path to pickle file

    Returns:
        Dictionary of transfer functions or None if file doesn't exist
    """
    if not pickle_file.exists():
        return None

    try:
        with open(pickle_file, "rb") as f:
            tf_dict = pickle.load(f)
        print(f"Loaded transfer functions from pickle: {pickle_file}")
        print(f"  Total transfer functions: {len(tf_dict)}")
        return tf_dict
    except Exception as e:
        print(f"Error loading pickle file {pickle_file}: {e}")
        return None


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
    # Handle zeros and negatives by using absolute values and sign
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


def group_by_rH_CV(tf_dict: Dict) -> Dict[Tuple[float, float], List[Tuple]]:
    """Group transfer functions by (rH, CV)."""
    grouped = defaultdict(list)
    for key, value in tf_dict.items():
        rH, CV = key[0], key[1]
        grouped[(rH, CV)].append((key, value))
    return dict(grouped)


def group_by_Vs1_thickness(
    tf_dict: Dict,
) -> Dict[Tuple[float, float], List[Tuple]]:
    """Group transfer functions by (Vs1, thickness)."""
    grouped = defaultdict(list)
    for key, value in tf_dict.items():
        Vs1, thickness = key[2], key[3]
        grouped[(Vs1, thickness)].append((key, value))
    return dict(grouped)


def plot_grouped_by_rH_CV(
    tf_dict: Dict,
    output_dir: Path,
    dz: float = 2.0,
):
    """
    Create plots grouped by (rH, CV).
    Each subplot shows all (Vs1, thickness) combinations for that (rH, CV).
    Creates two versions: geomean-only and combined (individual + geomean).
    """
    grouped = group_by_rH_CV(tf_dict)

    # Get unique (rH, CV) combinations
    rH_CV_combos = sorted(grouped.keys())
    n_subplots = len(rH_CV_combos)

    # Create figure with subplots in 3x3 grid
    n_cols = 3
    n_rows = 3
    max_subplots = n_rows * n_cols

    # Warn if there are more combinations than subplots
    if n_subplots > max_subplots:
        print(
            f"Warning: {n_subplots} (rH, CV) combinations found, but only {max_subplots} subplots available. "
            f"Only the first {max_subplots} combinations will be plotted."
        )

    # Get unique (Vs1, thickness) combinations for color coding
    Vs1_thickness_combos = sorted(set((k[2], k[3]) for k in tf_dict.keys()))
    cmap = plt.colormaps["tab10"]
    colors = cmap(np.linspace(0, 1, len(Vs1_thickness_combos)))
    color_map = {combo: colors[i] for i, combo in enumerate(Vs1_thickness_combos)}

    # Helper function to plot a single subplot
    def plot_subplot(ax, rH, CV, show_individuals=True):
        """Plot a single subplot for given (rH, CV) combination."""
        # Get all realizations for this (rH, CV)
        realizations = grouped[(rH, CV)]

        # Group by (Vs1, thickness) within this (rH, CV, damping_method)
        by_Vs1_th = defaultdict(list)
        for key, value in realizations:
            Vs1, thickness = key[2], key[3]
            by_Vs1_th[(Vs1, thickness)].append((key, value))

        # Plot each (Vs1, thickness) combination
        for (Vs1, thickness), tf_list in sorted(by_Vs1_th.items()):
            color = color_map[(Vs1, thickness)]
            label_base = f"Vs1={Vs1:.0f}, h={thickness:.0f}m"

            # Collect all TFs for this (Vs1, thickness)
            # All TFs use the same TTF parameters, so they have the same frequency grid
            tf_arrays = []
            common_freq = None
            for key, (freq, tf) in tf_list:
                if common_freq is None:
                    common_freq = (
                        freq  # Use frequency from first TF (all should be the same)
                    )
                tf_arrays.append(tf)

            # Plot individual realizations (light, thin lines) if requested
            if show_individuals:
                for i, tf in enumerate(tf_arrays):
                    ax.loglog(
                        common_freq,
                        tf,
                        color=color,
                        alpha=0.3,
                        linewidth=0.5,
                        label=None if i > 0 else label_base,
                    )

            # Compute and plot geometric mean (thick line)
            geomean = compute_geometric_mean(tf_arrays)
            ax.loglog(
                common_freq,
                geomean,
                color=color,
                linewidth=2.5,
                linestyle="--" if show_individuals else "-",
                label=f"{label_base} (geomean)" if show_individuals else label_base,
            )

        ax.set_xlabel(to_title_case("Frequency (Hz)"))
        ax.set_ylabel(to_title_case("Transfer Function Magnitude"))
        ax.set_title(f"rH={rH:.0f}, CV={CV:.3f}")
        ax.set_xlim(1e-1, 10)  # 0.1 to 10 Hz
        ax.set_ylim(1e-2, 1e2)  # 0.01 to 100
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)

    # Create COMBINED plot (individuals + geomean)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 18))
    axes_flat = axes.flatten()
    for idx, (rH, CV) in enumerate(rH_CV_combos):
        if idx >= max_subplots:
            break
        plot_subplot(axes_flat[idx], rH, CV, show_individuals=True)
    # Hide unused subplots
    for idx in range(len(rH_CV_combos), n_rows * n_cols):
        axes_flat[idx].set_visible(False)
    plt.tight_layout()
    output_file = output_dir / "transfer_functions_grouped_by_rH_CV_combined.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_file}")
    plt.close()

    # Create GEOMEAN-ONLY plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 18))
    axes_flat = axes.flatten()
    for idx, (rH, CV) in enumerate(rH_CV_combos):
        if idx >= max_subplots:
            break
        plot_subplot(axes_flat[idx], rH, CV, show_individuals=False)
    # Hide unused subplots
    for idx in range(len(rH_CV_combos), n_rows * n_cols):
        axes_flat[idx].set_visible(False)
    plt.tight_layout()
    output_file = output_dir / "transfer_functions_grouped_by_rH_CV_geomean_only.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_file}")
    plt.close()


def plot_grouped_by_Vs1_thickness(
    tf_dict: Dict,
    output_dir: Path,
    dz: float = 2.0,
):
    """
    Create plots grouped by (Vs1, thickness).
    Each subplot shows all (rH, CV) combinations for that (Vs1, thickness).
    Creates two versions: geomean-only and combined (individual + geomean).
    """
    grouped = group_by_Vs1_thickness(tf_dict)

    # Get unique (Vs1, thickness) combinations
    Vs1_th_combos = sorted(grouped.keys())
    n_subplots = len(Vs1_th_combos)

    # Create figure with subplots in 3x3 grid
    n_cols = 3
    n_rows = 3
    max_subplots = n_rows * n_cols

    # Warn if there are more combinations than subplots
    if n_subplots > max_subplots:
        print(
            f"Warning: {n_subplots} (Vs1, thickness) combinations found, but only {max_subplots} subplots available. "
            f"Only the first {max_subplots} combinations will be plotted."
        )

    # Get unique (rH, CV) combinations for color coding
    rH_CV_combos = sorted(set((k[0], k[1]) for k in tf_dict.keys()))
    cmap = plt.colormaps["tab10"]
    colors = cmap(np.linspace(0, 1, len(rH_CV_combos)))
    color_map = {combo: colors[i] for i, combo in enumerate(rH_CV_combos)}

    # Helper function to plot a single subplot
    def plot_subplot(ax, Vs1, thickness, show_individuals=True):
        """Plot a single subplot for given (Vs1, thickness) combination."""
        # Get all realizations for this (Vs1, thickness)
        realizations = grouped[(Vs1, thickness)]

        # Group by (rH, CV) within this (Vs1, thickness, damping_method)
        by_rH_CV = defaultdict(list)
        for key, value in realizations:
            rH, CV = key[0], key[1]
            by_rH_CV[(rH, CV)].append((key, value))

        # Plot each (rH, CV) combination
        for (rH, CV), tf_list in sorted(by_rH_CV.items()):
            color = color_map[(rH, CV)]
            label_base = f"rH={rH:.0f}, CV={CV:.3f}"

            # Collect all TFs for this (rH, CV)
            # All TFs use the same TTF parameters, so they have the same frequency grid
            tf_arrays = []
            common_freq = None
            for key, (freq, tf) in tf_list:
                if common_freq is None:
                    common_freq = (
                        freq  # Use frequency from first TF (all should be the same)
                    )
                tf_arrays.append(tf)

            # Plot individual realizations (light, thin lines) if requested
            if show_individuals:
                for i, tf in enumerate(tf_arrays):
                    ax.loglog(
                        common_freq,
                        tf,
                        color=color,
                        alpha=0.3,
                        linewidth=0.5,
                        label=None if i > 0 else label_base,
                    )

            # Compute and plot geometric mean (thick line)
            geomean = compute_geometric_mean(tf_arrays)
            ax.loglog(
                common_freq,
                geomean,
                color=color,
                linewidth=2.5,
                linestyle="--" if show_individuals else "-",
                label=f"{label_base} (geomean)" if show_individuals else label_base,
            )

        ax.set_xlabel(to_title_case("Frequency (Hz)"))
        ax.set_ylabel(to_title_case("Transfer Function Magnitude"))
        ax.set_title(f"Vs1={Vs1:.0f} m/s, thickness={thickness:.0f} m")
        ax.set_xlim(1e-1, 10)  # 0.1 to 10 Hz
        ax.set_ylim(1e-2, 1e2)  # 0.01 to 100
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)

    # Create COMBINED plot (individuals + geomean)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 18))
    axes_flat = axes.flatten()
    for idx, (Vs1, thickness) in enumerate(Vs1_th_combos):
        if idx >= max_subplots:
            break
        plot_subplot(axes_flat[idx], Vs1, thickness, show_individuals=True)
    # Hide unused subplots
    for idx in range(len(Vs1_th_combos), n_rows * n_cols):
        axes_flat[idx].set_visible(False)
    plt.tight_layout()
    output_file = (
        output_dir / "transfer_functions_grouped_by_Vs1_thickness_combined.png"
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_file}")
    plt.close()

    # Create GEOMEAN-ONLY plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 18))
    axes_flat = axes.flatten()
    for idx, (Vs1, thickness) in enumerate(Vs1_th_combos):
        if idx >= max_subplots:
            break
        plot_subplot(axes_flat[idx], Vs1, thickness, show_individuals=False)
    # Hide unused subplots
    for idx in range(len(Vs1_th_combos), n_rows * n_cols):
        axes_flat[idx].set_visible(False)
    plt.tight_layout()
    output_file = (
        output_dir / "transfer_functions_grouped_by_Vs1_thickness_geomean_only.png"
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_file}")
    plt.close()


def plot_grouped_by_Vs1_and_thickness(
    tf_dict: Dict,
    output_dir: Path,
    dz: float = 2.0,
):
    """
    Create plots grouped by Vs1, then by thickness.
    For each Vs1 value, creates a separate figure.
    Within each figure, creates subplots for each thickness.
    Each subplot shows TFs grouped by (rH, CV) combinations.
    Creates two versions: geomean-only and combined (individual + geomean).
    """
    # Get unique Vs1 and thickness values
    Vs1_values = sorted(set(k[2] for k in tf_dict.keys()))
    thickness_values = sorted(set(k[3] for k in tf_dict.keys()))

    # Get unique (rH, CV) combinations for color coding
    rH_CV_combos = sorted(set((k[0], k[1]) for k in tf_dict.keys()))
    cmap = plt.colormaps["tab10"]
    colors = cmap(np.linspace(0, 1, len(rH_CV_combos)))
    color_map = {combo: colors[i] for i, combo in enumerate(rH_CV_combos)}

    n_cols = 3  # One column per thickness
    n_rows = 1  # One row (or adjust if needed)

    # Helper function to plot a single subplot
    def plot_subplot(ax, Vs1, thickness, show_individuals=True):
        """Plot a single subplot for given (Vs1, thickness) combination."""
        # Filter TFs for this (Vs1, thickness)
        filtered_tfs = {
            key: value
            for key, value in tf_dict.items()
            if key[2] == Vs1 and key[3] == thickness
        }

        if len(filtered_tfs) == 0:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(f"h={thickness:.0f} m")
            ax.set_xlim(1e-1, 10)  # 0.1 to 10 Hz
            ax.set_ylim(1e-2, 1e2)  # 0.01 to 100
            return

        # Group by (rH, CV) within this (Vs1, thickness, damping_method)
        by_rH_CV = defaultdict(list)
        for key, value in filtered_tfs.items():
            rH, CV = key[0], key[1]
            by_rH_CV[(rH, CV)].append((key, value))

        # Plot each (rH, CV) combination
        for (rH, CV), tf_list in sorted(by_rH_CV.items()):
            color = color_map[(rH, CV)]
            label_base = f"rH={rH:.0f}, CV={CV:.3f}"

            # Collect all TFs for this (rH, CV)
            # All TFs use the same TTF parameters, so they have the same frequency grid
            tf_arrays = []
            common_freq = None
            for key, (freq, tf) in tf_list:
                if common_freq is None:
                    common_freq = (
                        freq  # Use frequency from first TF (all should be the same)
                    )
                tf_arrays.append(tf)

            # Plot individual realizations (light, thin lines) if requested
            if show_individuals:
                for i, tf in enumerate(tf_arrays):
                    ax.loglog(
                        common_freq,
                        tf,
                        color=color,
                        alpha=0.3,
                        linewidth=0.5,
                        label=None if i > 0 else label_base,
                    )

            # Compute and plot geometric mean (thick line)
            geomean = compute_geometric_mean(tf_arrays)
            ax.loglog(
                common_freq,
                geomean,
                color=color,
                linewidth=2.5,
                linestyle="--" if show_individuals else "-",
                label=f"{label_base} (geomean)" if show_individuals else label_base,
            )

        ax.set_xlabel(to_title_case("Frequency (Hz)"))
        ax.set_ylabel(to_title_case("Transfer Function Magnitude"))
        ax.set_title(f"h={thickness:.0f} m")
        ax.set_xlim(1e-1, 10)  # 0.1 to 10 Hz
        ax.set_ylim(1e-2, 1e2)  # 0.01 to 100
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)

    # Create plots for each Vs1 value
    for Vs1 in Vs1_values:
        # Create COMBINED plot (individuals + geomean)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6))
        # Convert axes to list for consistent handling
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = list(axes) if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()

        for idx, thickness in enumerate(thickness_values):
            if idx >= len(axes):
                break
            plot_subplot(axes[idx], Vs1, thickness, show_individuals=True)

        # Hide unused subplots
        for idx in range(len(thickness_values), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f"Vs1={Vs1:.0f} m/s", fontsize=16, y=1.02)
        plt.tight_layout()
        output_file = output_dir / f"transfer_functions_Vs1_{Vs1:.0f}_combined.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {output_file}")
        plt.close()

        # Create GEOMEAN-ONLY plot
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6))
        # Convert axes to list for consistent handling
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = list(axes) if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()

        for idx, thickness in enumerate(thickness_values):
            if idx >= len(axes):
                break
            plot_subplot(axes[idx], Vs1, thickness, show_individuals=False)

        # Hide unused subplots
        for idx in range(len(thickness_values), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f"Vs1={Vs1:.0f} m/s", fontsize=16, y=1.02)
        plt.tight_layout()
        output_file = output_dir / f"transfer_functions_Vs1_{Vs1:.0f}_geomean_only.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {output_file}")
        plt.close()


def plot_grouped_by_rH_CV_CV(
    tf_dict: Dict,
    output_dir: Path,
    dz: float = 2.0,
):
    """
    Create CV plots grouped by (rH, CV).
    Each subplot shows CV for all (Vs1, thickness) combinations for that (rH, CV).
    """
    grouped = group_by_rH_CV(tf_dict)

    # Get unique (rH, CV) combinations
    rH_CV_combos = sorted(grouped.keys())
    n_subplots = len(rH_CV_combos)

    # Create figure with subplots in 3x3 grid
    n_cols = 3
    n_rows = 3
    max_subplots = n_rows * n_cols

    # Warn if there are more combinations than subplots
    if n_subplots > max_subplots:
        print(
            f"Warning: {n_subplots} (rH, CV) combinations found, but only {max_subplots} subplots available. "
            f"Only the first {max_subplots} combinations will be plotted."
        )

    # Get unique (Vs1, thickness) combinations for color coding
    Vs1_thickness_combos = sorted(set((k[2], k[3]) for k in tf_dict.keys()))
    cmap = plt.colormaps["tab10"]
    colors = cmap(np.linspace(0, 1, len(Vs1_thickness_combos)))
    color_map = {combo: colors[i] for i, combo in enumerate(Vs1_thickness_combos)}

    # Helper function to plot a single subplot
    def plot_subplot(ax, rH, CV):
        """Plot CV for a single subplot for given (rH, CV) combination."""
        # Get all realizations for this (rH, CV)
        realizations = grouped[(rH, CV)]

        # Group by (Vs1, thickness) within this (rH, CV, damping_method)
        by_Vs1_th = defaultdict(list)
        for key, value in realizations:
            Vs1, thickness = key[2], key[3]
            by_Vs1_th[(Vs1, thickness)].append((key, value))

        # Plot CV for each (Vs1, thickness) combination
        for (Vs1, thickness), tf_list in sorted(by_Vs1_th.items()):
            color = color_map[(Vs1, thickness)]
            label_base = f"Vs1={Vs1:.0f}, h={thickness:.0f}m"

            # Collect all TFs for this (Vs1, thickness)
            tf_arrays = []
            common_freq = None
            for key, (freq, tf) in tf_list:
                if common_freq is None:
                    common_freq = freq
                tf_arrays.append(tf)

            # Compute and plot coefficient of variation
            cv = compute_coefficient_of_variation(tf_arrays)
            ax.semilogx(
                common_freq,
                cv,
                color=color,
                linewidth=2.5,
                label=label_base,
            )

        ax.set_xlabel(to_title_case("Frequency (Hz)"))
        ax.set_ylabel("Coefficient of Variation")
        ax.set_title(f"rH={rH:.0f}, CV={CV:.3f}")
        ax.set_xlim(1e-1, 10)  # 0.1 to 10 Hz
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)

    # Create CV plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 18))
    axes_flat = axes.flatten()
    for idx, (rH, CV) in enumerate(rH_CV_combos):
        if idx >= max_subplots:
            break
        plot_subplot(axes_flat[idx], rH, CV)
    # Hide unused subplots
    for idx in range(len(rH_CV_combos), n_rows * n_cols):
        axes_flat[idx].set_visible(False)
    plt.tight_layout()
    output_file = output_dir / "transfer_functions_grouped_by_rH_CV_CV.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_file}")
    plt.close()


def plot_grouped_by_Vs1_thickness_CV(
    tf_dict: Dict,
    output_dir: Path,
    dz: float = 2.0,
):
    """
    Create CV plots grouped by (Vs1, thickness).
    Each subplot shows CV for all (rH, CV) combinations for that (Vs1, thickness).
    """
    grouped = group_by_Vs1_thickness(tf_dict)

    # Get unique (Vs1, thickness) combinations
    Vs1_th_combos = sorted(grouped.keys())
    n_subplots = len(Vs1_th_combos)

    # Create figure with subplots in 3x3 grid
    n_cols = 3
    n_rows = 3
    max_subplots = n_rows * n_cols

    # Warn if there are more combinations than subplots
    if n_subplots > max_subplots:
        print(
            f"Warning: {n_subplots} (Vs1, thickness) combinations found, but only {max_subplots} subplots available. "
            f"Only the first {max_subplots} combinations will be plotted."
        )

    # Get unique (rH, CV) combinations for color coding
    rH_CV_combos = sorted(set((k[0], k[1]) for k in tf_dict.keys()))
    cmap = plt.colormaps["tab10"]
    colors = cmap(np.linspace(0, 1, len(rH_CV_combos)))
    color_map = {combo: colors[i] for i, combo in enumerate(rH_CV_combos)}

    # Helper function to plot a single subplot
    def plot_subplot(ax, Vs1, thickness):
        """Plot CV for a single subplot for given (Vs1, thickness) combination."""
        # Get all realizations for this (Vs1, thickness)
        realizations = grouped[(Vs1, thickness)]

        # Group by (rH, CV) within this (Vs1, thickness, damping_method)
        by_rH_CV = defaultdict(list)
        for key, value in realizations:
            rH, CV = key[0], key[1]
            by_rH_CV[(rH, CV)].append((key, value))

        # Plot CV for each (rH, CV) combination
        for (rH, CV), tf_list in sorted(by_rH_CV.items()):
            color = color_map[(rH, CV)]
            label_base = f"rH={rH:.0f}, CV={CV:.3f}"

            # Collect all TFs for this (rH, CV)
            tf_arrays = []
            common_freq = None
            for key, (freq, tf) in tf_list:
                if common_freq is None:
                    common_freq = freq
                tf_arrays.append(tf)

            # Compute and plot coefficient of variation
            cv = compute_coefficient_of_variation(tf_arrays)
            ax.semilogx(
                common_freq,
                cv,
                color=color,
                linewidth=2.5,
                label=label_base,
            )

        ax.set_xlabel(to_title_case("Frequency (Hz)"))
        ax.set_ylabel("Coefficient of Variation")
        ax.set_title(f"Vs1={Vs1:.0f} m/s, thickness={thickness:.0f} m")
        ax.set_xlim(1e-1, 10)  # 0.1 to 10 Hz
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)

    # Create CV plot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 18))
    axes_flat = axes.flatten()
    for idx, (Vs1, thickness) in enumerate(Vs1_th_combos):
        if idx >= max_subplots:
            break
        plot_subplot(axes_flat[idx], Vs1, thickness)
    # Hide unused subplots
    for idx in range(len(Vs1_th_combos), n_rows * n_cols):
        axes_flat[idx].set_visible(False)
    plt.tight_layout()
    output_file = output_dir / "transfer_functions_grouped_by_Vs1_thickness_CV.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {output_file}")
    plt.close()


def plot_grouped_by_Vs1_and_thickness_CV(
    tf_dict: Dict,
    output_dir: Path,
    dz: float = 2.0,
):
    """
    Create CV plots grouped by Vs1, then by thickness.
    For each Vs1 value, creates a separate figure.
    Within each figure, creates subplots for each thickness.
    Each subplot shows CV for TFs grouped by (rH, CV) combinations.
    """
    # Get unique Vs1 and thickness values
    Vs1_values = sorted(set(k[2] for k in tf_dict.keys()))
    thickness_values = sorted(set(k[3] for k in tf_dict.keys()))

    # Get unique (rH, CV) combinations for color coding
    rH_CV_combos = sorted(set((k[0], k[1]) for k in tf_dict.keys()))
    cmap = plt.colormaps["tab10"]
    colors = cmap(np.linspace(0, 1, len(rH_CV_combos)))
    color_map = {combo: colors[i] for i, combo in enumerate(rH_CV_combos)}

    n_cols = 3  # One column per thickness
    n_rows = 1  # One row

    # Helper function to plot a single subplot
    def plot_subplot(ax, Vs1, thickness):
        """Plot CV for a single subplot for given (Vs1, thickness) combination."""
        # Filter TFs for this (Vs1, thickness)
        filtered_tfs = {
            key: value
            for key, value in tf_dict.items()
            if key[2] == Vs1 and key[3] == thickness
        }

        if len(filtered_tfs) == 0:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(f"h={thickness:.0f} m")
            ax.set_xlim(1e-1, 10)  # 0.1 to 10 Hz
            return

        # Group by (rH, CV) within this (Vs1, thickness, damping_method)
        by_rH_CV = defaultdict(list)
        for key, value in filtered_tfs.items():
            rH, CV = key[0], key[1]
            by_rH_CV[(rH, CV)].append((key, value))

        # Plot CV for each (rH, CV) combination
        for (rH, CV), tf_list in sorted(by_rH_CV.items()):
            color = color_map[(rH, CV)]
            label_base = f"rH={rH:.0f}, CV={CV:.3f}"

            # Collect all TFs for this (rH, CV)
            tf_arrays = []
            common_freq = None
            for key, (freq, tf) in tf_list:
                if common_freq is None:
                    common_freq = freq
                tf_arrays.append(tf)

            # Compute and plot coefficient of variation
            cv = compute_coefficient_of_variation(tf_arrays)
            ax.semilogx(
                common_freq,
                cv,
                color=color,
                linewidth=2.5,
                label=label_base,
            )

        ax.set_xlabel(to_title_case("Frequency (Hz)"))
        ax.set_ylabel("Coefficient of Variation")
        ax.set_title(f"h={thickness:.0f} m")
        ax.set_xlim(1e-1, 10)  # 0.1 to 10 Hz
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)

    # Create plots for each Vs1 value
    for Vs1 in Vs1_values:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6))
        # Convert axes to list for consistent handling
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = list(axes) if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()

        for idx, thickness in enumerate(thickness_values):
            if idx >= len(axes):
                break
            plot_subplot(axes[idx], Vs1, thickness)

        # Hide unused subplots
        for idx in range(len(thickness_values), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f"Vs1={Vs1:.0f} m/s", fontsize=16, y=1.02)
        plt.tight_layout()
        output_file = output_dir / f"transfer_functions_Vs1_{Vs1:.0f}_CV.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {output_file}")
        plt.close()


def plot_individual_cases(
    tf_dict: Dict,
    output_dir: Path,
    dz: float = 2.0,
):
    """
    Create individual plots for each (thickness, rH, Vs1, CV) combination.
    Each plot shows the 5 realizations (seeds) and their geometric mean.

    Creates 3x3x2x3 = 54 individual plots in a separate folder.
    """
    # Create output directory for individual cases
    individual_dir = output_dir / "individual_cases"
    individual_dir.mkdir(exist_ok=True)

    # Get unique parameter values
    thickness_values = sorted(set(k[3] for k in tf_dict.keys()))
    rH_values = sorted(set(k[0] for k in tf_dict.keys()))
    Vs1_values = sorted(set(k[2] for k in tf_dict.keys()))
    CV_values = sorted(set(k[1] for k in tf_dict.keys()))
    damping_method = "global_avg"  # Only one damping method

    print(
        f"Creating individual plots for {len(thickness_values)}x{len(rH_values)}x{len(Vs1_values)}x{len(CV_values)} = "
        f"{len(thickness_values) * len(rH_values) * len(Vs1_values) * len(CV_values)} combinations..."
    )

    plot_count = 0
    skipped_count = 0

    # Iterate through all combinations
    for thickness in thickness_values:
        for rH in rH_values:
            for Vs1 in Vs1_values:
                for CV in CV_values:
                    # Filter TFs for this combination (all seeds, same damping_method)
                    filtered_tfs = {
                        key: value
                        for key, value in tf_dict.items()
                        if key[0] == rH
                        and key[1] == CV
                        and key[2] == Vs1
                        and key[3] == thickness
                        and key[5] == damping_method
                    }

                    if len(filtered_tfs) == 0:
                        skipped_count += 1
                        continue

                    # Collect all TFs for this combination
                    tf_arrays = []
                    common_freq = None
                    seed_list = []

                    for key, (freq, tf) in filtered_tfs.items():
                        seed = key[4]
                        seed_list.append(seed)
                        if common_freq is None:
                            common_freq = freq
                        tf_arrays.append(tf)

                    # Sort by seed to ensure consistent ordering
                    sorted_indices = sorted(
                        range(len(seed_list)), key=lambda i: seed_list[i]
                    )
                    tf_arrays = [tf_arrays[i] for i in sorted_indices]
                    seed_list = [seed_list[i] for i in sorted_indices]

                    # Create figure
                    fig, ax = plt.subplots(figsize=(10, 8))

                    # Plot individual realizations (light, thin lines)
                    for tf, seed in zip(tf_arrays, seed_list):
                        ax.loglog(
                            common_freq,
                            tf,
                            color="gray",
                            alpha=0.4,
                            linewidth=0.8,
                            label=f"Seed {seed}",
                        )

                    # Compute and plot geometric mean (thick line)
                    if len(tf_arrays) > 0:
                        geomean = compute_geometric_mean(tf_arrays)
                        ax.loglog(
                            common_freq,
                            geomean,
                            color="red",
                            linewidth=2.5,
                            linestyle="--",
                            label="Geometric Mean",
                        )

                    # Set labels and title
                    ax.set_xlabel(to_title_case("Frequency (Hz)"))
                    ax.set_ylabel(to_title_case("Transfer Function Magnitude"))
                    ax.set_title(
                        f"thickness={thickness:.0f}m, rH={rH:.0f}, Vs1={Vs1:.0f}m/s, CV={CV:.3f}",
                        fontsize=11,
                    )
                    ax.set_xlim(1e-1, 10)  # 0.1 to 10 Hz
                    ax.set_ylim(1e-2, 1e2)  # 0.01 to 100
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc="best", fontsize=9)

                    # Create filename
                    filename = f"thickness_{thickness:.0f}_rH_{rH:.0f}_Vs1_{Vs1:.0f}_CV_{CV:.3f}.png"
                    output_file = individual_dir / filename

                    plt.tight_layout()
                    plt.savefig(output_file, dpi=300, bbox_inches="tight")
                    plt.close()

                    plot_count += 1

                    if plot_count % 10 == 0:
                        print(f"  Created {plot_count} individual plots...")

    print(f"Created {plot_count} individual plots in {individual_dir}")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} combinations (no data available)")


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
    results_1D_dir: Path,
) -> Dict[Tuple[float, float, str], Tuple[np.ndarray, np.ndarray]]:
    """
    Load transfer functions from 1D_results folders.

    Args:
        results_1D_dir: Path to 1D_results directory

    Returns:
        Dictionary keyed by (Vs1, thickness, damping_method) -> (freq, tf)
    """
    tf_dict = {}
    results_path = Path(results_1D_dir)

    if not results_path.exists():
        print(f"Warning: 1D results directory not found: {results_path}")
        return tf_dict

    print(f"Loading 1D results from: {results_path}")
    folders = sorted([f for f in results_path.iterdir() if f.is_dir()])

    loaded_count = 0
    skipped_count = 0

    for folder in folders:
        params = parse_1D_folder(folder.name)
        if params is None:
            skipped_count += 1
            continue

        # Load acceleration files
        thickness_val = float(params["thickness"])
        accel_data = load_acceleration_files(folder, thickness_val)
        if accel_data is None:
            skipped_count += 1
            continue

        base_data, top_data = accel_data

        # Compute transfer function
        try:
            freq, tf = compute_transfer_function(base_data, top_data)

            # Store in dictionary
            key = (params["Vs1"], params["thickness"], params["damping_method"])
            tf_dict[key] = (freq, tf)
            loaded_count += 1

        except Exception:
            skipped_count += 1
            continue

    print(f"Loaded {loaded_count} 1D transfer functions")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} folders")

    return tf_dict


def plot_individual_seed_vs_uniform(
    tf_dict: Dict,
    tf_dict_1D: Dict,
    output_dir: Path,
    seed: int = 10,
):
    """
    Plot individual seed cases (no geomean) comparing heterogeneous cases with uniform 1D cases.

    For each (Vs1, thickness, CV, rH) combination:
    - Plot the individual seed TF (heterogeneous case) for all damping methods
    - Overlay the uniform 1D TF for all damping methods
    - This shows the effect of variability in each realization across all damping methods

    Args:
        tf_dict: Dictionary of variability TFs keyed by (rH, CV, Vs1, thickness, seed, damping_method)
        tf_dict_1D: Dictionary of 1D uniform TFs keyed by (Vs1, thickness, damping_method)
        output_dir: Output directory for plots
        seed: Seed value to plot (default 10)
    """
    # Create output directory for individual seed plots
    seed_dir = output_dir / "individual_seed_vs_uniform"
    seed_dir.mkdir(exist_ok=True)

    # Get unique parameter values
    Vs1_values = sorted(set(k[2] for k in tf_dict.keys()))
    thickness_values = sorted(set(k[3] for k in tf_dict.keys()))
    CV_values = sorted(set(k[1] for k in tf_dict.keys()))
    rH_values = sorted(set(k[0] for k in tf_dict.keys()))
    damping_method = "global_avg"  # Only one damping method

    print(
        f"Creating individual seed plots for seed={seed} comparing with uniform 1D cases..."
    )

    plot_count = 0
    skipped_count = 0

    # Iterate through all combinations
    for Vs1 in Vs1_values:
        for thickness in thickness_values:
            for CV in CV_values:
                for rH in rH_values:
                    # Create figure for this combination
                    fig, ax = plt.subplots(figsize=(10, 8))

                    has_data = False

                    # Get uniform 1D TF
                    uniform_key = (Vs1, thickness, damping_method)
                    if uniform_key not in tf_dict_1D:
                        skipped_count += 1
                        plt.close()
                        continue

                    uniform_freq, uniform_tf = tf_dict_1D[uniform_key]

                    # Get heterogeneous TF for this seed
                    heterogeneous_key = (
                        rH,
                        CV,
                        Vs1,
                        thickness,
                        seed,
                        damping_method,
                    )
                    if heterogeneous_key not in tf_dict:
                        skipped_count += 1
                        plt.close()
                        continue

                    hetero_freq, hetero_tf = tf_dict[heterogeneous_key]

                    # Plot uniform 1D TF
                    ax.loglog(
                        uniform_freq,
                        uniform_tf,
                        color="black",
                        linewidth=2.5,
                        linestyle="-",
                        label="Uniform (1D)",
                    )

                    # Plot heterogeneous TF for this seed
                    ax.loglog(
                        hetero_freq,
                        hetero_tf,
                        color="red",
                        linewidth=2.0,
                        linestyle="--",
                        alpha=0.8,
                        label=f"Heterogeneous (seed={seed})",
                    )

                    has_data = True

                    if not has_data:
                        plt.close()
                        continue

                    # Set labels and title
                    ax.set_xlabel(to_title_case("Frequency (Hz)"))
                    ax.set_ylabel(to_title_case("Transfer Function Magnitude"))
                    ax.set_title(
                        f"Vs1={Vs1:.0f} m/s, h={thickness:.0f} m, rH={rH:.0f}, CV={CV:.3f}",
                        fontsize=11,
                    )
                    ax.set_xlim(1e-1, 10)  # 0.1 to 10 Hz
                    ax.set_ylim(1e-2, 1e2)  # 0.01 to 100
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc="best", fontsize=8)

                    # Create filename (no damping_method in name)
                    filename = (
                        f"seed_{seed}_Vs1_{Vs1:.0f}_thickness_{thickness:.0f}_"
                        f"rH_{rH:.0f}_CV_{CV:.3f}.png"
                    )
                    output_file = seed_dir / filename

                    plt.tight_layout()
                    plt.savefig(output_file, dpi=300, bbox_inches="tight")
                    plt.close()

                    plot_count += 1

                    if plot_count % 10 == 0:
                        print(f"  Created {plot_count} individual seed plots...")

    print(f"Created {plot_count} individual seed plots in {seed_dir}")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} combinations (no data available)")


def main():
    """Main function to compute and plot transfer functions."""
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Activate venv (informational)
    activate_venv()

    # Set up paths
    results_dir = script_dir / "results"
    output_dir = script_dir / "transfer_function_results"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Computing Transfer Functions for Sensitivity Analysis")
    print("=" * 60)

    # Load all results
    tf_dict, skipped_folders = load_all_results(results_dir)

    if len(tf_dict) == 0:
        print("No transfer functions loaded. Exiting.")
        return

    # Save transfer functions dictionary
    dict_file = output_dir / "transfer_functions_dict.pkl"
    with open(dict_file, "wb") as f:
        pickle.dump(tf_dict, f)
    print(f"Saved transfer functions dictionary: {dict_file}")

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"  Total transfer functions: {len(tf_dict)}")

    # Count by parameter
    rH_values = sorted(set(k[0] for k in tf_dict.keys()))
    CV_values = sorted(set(k[1] for k in tf_dict.keys()))
    Vs1_values = sorted(set(k[2] for k in tf_dict.keys()))
    thickness_values = sorted(set(k[3] for k in tf_dict.keys()))
    seed_values = sorted(set(k[4] for k in tf_dict.keys()))

    print(f"  rH values: {rH_values}")
    print(f"  CV values: {CV_values}")
    print(f"  Vs1 values: {Vs1_values}")
    print(f"  thickness values: {thickness_values}")
    print(f"  seed values: {seed_values}")

    # Print skipped folders with indexes
    if skipped_folders:
        print("\n" + "=" * 60)
        print("SKIPPED FOLDERS (for rerunning):")
        print("=" * 60)

        # Sort by index (valid indexes first, then invalid)
        skipped_with_index = [s for s in skipped_folders if s["index"] >= 0]
        skipped_without_index = [s for s in skipped_folders if s["index"] < 0]

        skipped_with_index.sort(key=lambda x: x["index"])

        print("\nFolders with valid indexes (can be rerun):")
        print("-" * 60)
        for skip in skipped_with_index:
            params = skip["params"]
            print(
                f"Index {skip['index']:3d}: {skip['folder']} | "
                f"Vs1={params['Vs1']:.0f}, thickness={params['thickness']:.0f}, "
                f"rH={params['rH']:.0f}, CV={params['CV']:.3f}, seed={params['seed']} | "
                f"Reason: {skip['reason']}"
            )

        if skipped_without_index:
            print("\nFolders without valid indexes (parsing failed):")
            print("-" * 60)
            for skip in skipped_without_index:
                print(f"  {skip['folder']} | Reason: {skip['reason']}")

        # Print indexes for easy copy-paste
        valid_indexes = [s["index"] for s in skipped_with_index]
        if valid_indexes:
            print("\n" + "=" * 60)
            print("Indexes to rerun (comma-separated):")
            print(",".join(map(str, valid_indexes)))
            print("=" * 60)

    # Create plots
    print("\nGenerating plots...")

    # Plot 1: Grouped by (rH, CV)
    print("  Creating plots grouped by (rH, CV)...")
    plot_grouped_by_rH_CV(tf_dict, output_dir)

    # Plot 2: Grouped by (Vs1, thickness)
    print("  Creating plots grouped by (Vs1, thickness)...")
    plot_grouped_by_Vs1_thickness(tf_dict, output_dir)

    # Plot 3: Grouped by Vs1, then by thickness (showing rH, CV variations)
    print("  Creating plots grouped by Vs1 and thickness...")
    plot_grouped_by_Vs1_and_thickness(tf_dict, output_dir)

    # Create CV plots
    print("\nGenerating CV plots...")

    # CV Plot 1: Grouped by (rH, CV)
    print("  Creating CV plots grouped by (rH, CV)...")
    plot_grouped_by_rH_CV_CV(tf_dict, output_dir)

    # CV Plot 2: Grouped by (Vs1, thickness)
    print("  Creating CV plots grouped by (Vs1, thickness)...")
    plot_grouped_by_Vs1_thickness_CV(tf_dict, output_dir)

    # CV Plot 3: Grouped by Vs1, then by thickness (showing rH, CV variations)
    print("  Creating CV plots grouped by Vs1 and thickness...")
    plot_grouped_by_Vs1_and_thickness_CV(tf_dict, output_dir)

    # Create individual case plots
    print("\nGenerating individual case plots...")
    print(
        "  Creating individual plots for each (thickness, rH, Vs1, CV) combination..."
    )
    plot_individual_cases(tf_dict, output_dir)

    # Create individual seed vs uniform plots
    print("\nGenerating individual seed vs uniform plots...")
    # Load 1D uniform results
    results_1D_dir = script_dir / "1D_results"
    tf_dict_1D = load_1D_results(results_1D_dir)

    if len(tf_dict_1D) > 0:
        # Use first available seed (or default to 10)
        available_seeds = sorted(set(k[4] for k in tf_dict.keys()))
        if available_seeds:
            seed_to_plot = available_seeds[0]  # Use first seed
            print(
                f"  Creating individual seed plots for seed={seed_to_plot} comparing with uniform 1D cases..."
            )
            plot_individual_seed_vs_uniform(
                tf_dict, tf_dict_1D, output_dir, seed=seed_to_plot
            )
        else:
            print("  No seeds available in tf_dict")
    else:
        print(
            "  Skipping individual seed vs uniform plots (no 1D uniform results found)"
        )

    print("\n" + "=" * 60)
    print("Transfer function computation complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)

    # Compute and plot all spectral metrics
    print("\nComputing and plotting all spectral metrics...")
    metrics_results = compute_spectral_metrics(tf_dict, tf_dict_1D)
    plot_all_spectral_metrics(metrics_results, output_dir)
    print("Spectral metrics computation and plotting complete!")
    print("Results saved to: {output_dir}")
    print("=" * 60)


# ============================================================================
# Spectral Metrics Analysis Functions
# ============================================================================


def _find_peak_frequency(freq: np.ndarray, tf: np.ndarray) -> float:
    """
    Find the frequency at which the transfer function magnitude is maximum.

    Args:
        freq: Frequency array (Hz)
        tf: Transfer function magnitude array

    Returns:
        Peak frequency (Hz)
    """
    peak_idx = np.argmax(tf)
    return freq[peak_idx]


def _find_peak_amplification(tf: np.ndarray) -> float:
    """
    Find the maximum transfer function magnitude (peak amplification).

    Args:
        tf: Transfer function magnitude array

    Returns:
        Peak amplification value
    """
    return float(np.max(tf))


def _compute_bandwidth_3db(
    freq: np.ndarray, tf: np.ndarray, f_peak: float, A_peak: float
) -> float:
    """
    Compute frequency bandwidth using -3dB (half-power) method.

    Finds frequencies where |H(f)| = A_peak / sqrt(2) around the peak frequency.

    Args:
        freq: Frequency array (Hz)
        tf: Transfer function magnitude array
        f_peak: Peak frequency (Hz)
        A_peak: Peak amplification value

    Returns:
        Bandwidth Δf = f_upper - f_lower (Hz), or NaN if crossing points not found
    """
    # Half-power level
    half_power_level = A_peak / np.sqrt(2.0)

    # Find lower crossing point (below peak)
    lower_mask = freq <= f_peak
    if np.any(lower_mask):
        lower_tf = tf[lower_mask]
        lower_freq = freq[lower_mask]
        # Find where TF crosses half-power level going upward
        below_threshold = lower_tf < half_power_level
        if np.any(below_threshold) and np.any(~below_threshold):
            # Find crossing point by interpolation
            crossing_idx = np.where(np.diff(np.sign(lower_tf - half_power_level)))[0]
            if len(crossing_idx) > 0:
                # Use last crossing before peak
                idx = crossing_idx[-1]
                if idx < len(lower_freq) - 1:
                    # Linear interpolation
                    f_lower = np.interp(
                        half_power_level,
                        lower_tf[idx : idx + 2],
                        lower_freq[idx : idx + 2],
                    )
                else:
                    f_lower = lower_freq[idx]
            else:
                # No crossing found, use first point below threshold or first frequency
                f_lower = lower_freq[0] if len(lower_freq) > 0 else freq[0]
        else:
            # All above or all below threshold
            if np.all(lower_tf >= half_power_level):
                # All above threshold, use first frequency
                f_lower = lower_freq[0] if len(lower_freq) > 0 else freq[0]
            else:
                # All below threshold, need to extrapolate or use first frequency
                f_lower = lower_freq[0] if len(lower_freq) > 0 else freq[0]
    else:
        f_lower = freq[0]

    # Find upper crossing point (above peak)
    upper_mask = freq >= f_peak
    if np.any(upper_mask):
        upper_tf = tf[upper_mask]
        upper_freq = freq[upper_mask]
        # Find where TF crosses half-power level going downward
        above_threshold = upper_tf > half_power_level
        if np.any(above_threshold) and np.any(~above_threshold):
            # Find crossing point by interpolation
            crossing_idx = np.where(np.diff(np.sign(upper_tf - half_power_level)))[0]
            if len(crossing_idx) > 0:
                # Use first crossing after peak
                idx = crossing_idx[0]
                if idx < len(upper_freq) - 1:
                    # Linear interpolation
                    f_upper = float(
                        np.interp(
                            half_power_level,
                            upper_tf[idx : idx + 2],
                            upper_freq[idx : idx + 2],
                        )
                    )
                else:
                    f_upper = float(upper_freq[idx])
            else:
                # No crossing found, use last point above threshold or last frequency
                f_upper = (
                    float(upper_freq[-1]) if len(upper_freq) > 0 else float(freq[-1])
                )
        else:
            # All above or all below threshold
            if np.all(upper_tf >= half_power_level):
                # All above threshold, use last frequency
                f_upper = (
                    float(upper_freq[-1]) if len(upper_freq) > 0 else float(freq[-1])
                )
            else:
                # All below threshold, need to extrapolate or use last frequency
                f_upper = (
                    float(upper_freq[-1]) if len(upper_freq) > 0 else float(freq[-1])
                )
    else:
        f_upper = float(freq[-1])

    bandwidth = float(f_upper - f_lower)
    if bandwidth > 0:
        return bandwidth
    else:
        return float(np.nan)


def _compute_bandwidth_spectral_moment(freq: np.ndarray, tf: np.ndarray) -> float:
    """
    Compute frequency bandwidth using spectral moment definition.

    Δf = sqrt(m2 / m0)
    where m0 = sum(|H(f)|^2 * df) and m2 = sum(f^2 * |H(f)|^2 * df)

    Args:
        freq: Frequency array (Hz)
        tf: Transfer function magnitude array

    Returns:
        Bandwidth Δf (Hz)
    """
    # Compute frequency spacing (assuming uniform spacing)
    df = freq[1] - freq[0] if len(freq) > 1 else 1.0

    # Compute spectral moments
    m0 = np.sum(tf**2 * df)
    m2 = np.sum(freq**2 * tf**2 * df)

    if m0 <= 0:
        return np.nan

    bandwidth = np.sqrt(m2 / m0)
    return bandwidth


def _compute_coherency(
    freq1: np.ndarray,
    tf1: np.ndarray,
    freq2: np.ndarray,
    tf2: np.ndarray,
    f_peak: float,
) -> float:
    """
    Compute coherency at peak frequency between two transfer functions.

    Coherency = |Re(H1* * H2)|^2 / (|H1|^2 * |H2|^2)
    where H1* is the complex conjugate of H1.

    Note: This function assumes tf1 and tf2 are magnitudes. For proper coherency
    computation, we would need complex transfer functions. This implementation
    approximates coherency using magnitude-only data.

    Args:
        freq1: Frequency array for first TF (Hz)
        tf1: Transfer function magnitude array for first TF
        freq2: Frequency array for second TF (Hz)
        tf2: Transfer function magnitude array for second TF
        f_peak: Peak frequency at which to evaluate coherency (Hz)

    Returns:
        Coherency value (0-1), or NaN if evaluation fails
    """
    # Interpolate both TFs to peak frequency
    try:
        H1_at_peak = np.interp(f_peak, freq1, tf1)
        H2_at_peak = np.interp(f_peak, freq2, tf2)
    except Exception:
        return np.nan

    # Since we only have magnitudes, approximate coherency
    # For magnitude-only data, we use: coherency ≈ (H1 * H2)^2 / (H1^2 * H2^2) = 1
    # This is a simplified approximation. For true coherency, we need complex TFs.
    # A better approximation: assume phase difference is small near peak
    if H1_at_peak <= 0 or H2_at_peak <= 0:
        return np.nan

    # Simplified coherency: normalized product
    # This assumes the TFs are in phase at the peak (reasonable for resonance)
    coherency = (H1_at_peak * H2_at_peak) ** 2 / ((H1_at_peak**2) * (H2_at_peak**2))

    return coherency


def _fit_regression(
    x: np.ndarray, y: np.ndarray, model_type: str = "linear"
) -> Dict[str, np.ndarray | float]:
    """
    Fit regression model to data.

    Args:
        x: Independent variable array
        y: Dependent variable array
        model_type: Type of model ("linear" or "polynomial")

    Returns:
        Dictionary with keys:
        - 'coeffs': Array of coefficients
        - 'R2': R-squared value
        - 'y_pred': Predicted y values
    """
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 2:
        return {"coeffs": np.array([]), "R2": np.nan, "y_pred": np.array([])}

    if model_type == "linear":
        # Linear fit: y = a0 + a1*x
        coeffs = np.polyfit(x_clean, y_clean, 1)
        y_pred = np.polyval(coeffs, x_clean)
    elif model_type == "polynomial":
        # Polynomial fit: y = a0 + a1*x + a2*x^2
        # Only fit polynomial if we have at least 3 points
        if len(x_clean) >= 3:
            # Suppress RankWarning for polynomial fit with few points
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Polyfit may be poorly conditioned"
                )
                coeffs = np.polyfit(x_clean, y_clean, 2)
            y_pred = np.polyval(coeffs, x_clean)
        else:
            # Fall back to linear if not enough points
            coeffs = np.polyfit(x_clean, y_clean, 1)
            y_pred = np.polyval(coeffs, x_clean)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Compute R-squared
    ss_res = np.sum((y_clean - y_pred) ** 2)
    ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
    R2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else float(np.nan)

    return {"coeffs": coeffs, "R2": R2, "y_pred": y_pred}


def compute_spectral_metrics(
    tf_dict: Dict,
    tf_dict_1D: Dict | None = None,
    bandwidth_method: str = "-3dB",
    coherency_pairs: List[Tuple[int, int]] | None = None,
) -> Dict:
    """
    Compute comprehensive spectral metrics from transfer function data.

    This function implements a four-level analysis:
    - Level 1: Per-realization spectral metrics (peak frequency, amplification, bandwidth, Q)
    - Level 2: Normalized shift metrics relative to homogeneous case
    - Level 3: Coherency-based metrics at peak frequency
    - Level 4: Statistical relationships (regression models)

    Args:
        tf_dict: Dictionary keyed by (rH, CV, Vs1, thickness, seed, damping_method) -> (freq, tf)
        tf_dict_1D: Dictionary of homogeneous TFs keyed by (Vs1, thickness, damping_method) -> (freq, tf)
                   Used as reference for Level 2 normalized metrics. If None, Level 2 metrics are skipped.
        bandwidth_method: Method for computing bandwidth ("-3dB" or "spectral_moment")
        coherency_pairs: List of (seed1, seed2) tuples for coherency computation.
                        If None, uses all pairs from available seeds.

    Returns:
        Dictionary with keys:
        - 'level1': Per-realization metrics and ensemble statistics
        - 'level2': Normalized shift metrics (if tf_dict_1D provided)
        - 'level3': Coherency metrics
        - 'level4': Statistical relationships
    """
    results = {
        "level1": {"per_realization": {}, "ensemble_stats": {}},
        "level2": {},
        "level3": {},
        "level4": {},
    }

    # Level 1: Per-Realization Spectral Metrics
    print("Computing Level 1: Per-realization spectral metrics...")
    for key, (freq, tf) in tf_dict.items():
        rH, CV, Vs1, thickness, seed, damping_method = key

        # Compute metrics for this realization
        f_peak = _find_peak_frequency(freq, tf)
        A_peak = _find_peak_amplification(tf)

        if bandwidth_method == "-3dB":
            bandwidth = _compute_bandwidth_3db(freq, tf, f_peak, A_peak)
        elif bandwidth_method == "spectral_moment":
            bandwidth = _compute_bandwidth_spectral_moment(freq, tf)
        else:
            raise ValueError(f"Unknown bandwidth_method: {bandwidth_method}")

        Q = f_peak / bandwidth if bandwidth > 0 and not np.isnan(bandwidth) else np.nan

        # Store per-realization metrics
        results["level1"]["per_realization"][key] = {
            "f_peak": f_peak,
            "A_peak": A_peak,
            "bandwidth": bandwidth,
            "Q": Q,
        }

    # Compute ensemble statistics
    print("Computing ensemble statistics...")
    # Group by (rH, CV, Vs1, thickness) - excluding seed and damping_method
    grouped_realizations = defaultdict(list)
    for key in results["level1"]["per_realization"]:
        rH, CV, Vs1, thickness, seed, damping_method = key
        group_key = (rH, CV, Vs1, thickness)
        grouped_realizations[group_key].append(key)

    for group_key, realization_keys in grouped_realizations.items():
        # Collect metrics for all realizations in this group
        f_peaks = [
            results["level1"]["per_realization"][k]["f_peak"]
            for k in realization_keys
            if not np.isnan(results["level1"]["per_realization"][k]["f_peak"])
        ]
        A_peaks = [
            results["level1"]["per_realization"][k]["A_peak"]
            for k in realization_keys
            if not np.isnan(results["level1"]["per_realization"][k]["A_peak"])
        ]
        bandwidths = [
            results["level1"]["per_realization"][k]["bandwidth"]
            for k in realization_keys
            if not np.isnan(results["level1"]["per_realization"][k]["bandwidth"])
        ]
        Qs = [
            results["level1"]["per_realization"][k]["Q"]
            for k in realization_keys
            if not np.isnan(results["level1"]["per_realization"][k]["Q"])
        ]

        # Compute statistics
        stats = {}
        for metric_name, values in [
            ("f_peak", f_peaks),
            ("A_peak", A_peaks),
            ("bandwidth", bandwidths),
            ("Q", Qs),
        ]:
            if len(values) > 0:
                mean_val = np.mean(values)
                std_val = np.std(values)
                cov_val = std_val / mean_val if mean_val > 0 else np.nan
                stats[metric_name] = {
                    "mean": mean_val,
                    "std": std_val,
                    "cov": cov_val,
                }
            else:
                stats[metric_name] = {
                    "mean": np.nan,
                    "std": np.nan,
                    "cov": np.nan,
                }

        results["level1"]["ensemble_stats"][group_key] = stats

    # Level 2: Normalized Shift Metrics
    if tf_dict_1D is not None:
        print("Computing Level 2: Normalized shift metrics...")
        # For each heterogeneous case, compare to homogeneous reference
        for group_key in results["level1"]["ensemble_stats"]:
            rH, CV, Vs1, thickness = group_key

            # Get homogeneous reference (CV=0 equivalent)
            uniform_key = (Vs1, thickness, "global_avg")
            if uniform_key not in tf_dict_1D:
                continue

            uniform_freq, uniform_tf = tf_dict_1D[uniform_key]

            # Compute homogeneous metrics
            f_peak_homogeneous = _find_peak_frequency(uniform_freq, uniform_tf)
            A_peak_homogeneous = _find_peak_amplification(uniform_tf)

            if bandwidth_method == "-3dB":
                bandwidth_homogeneous = _compute_bandwidth_3db(
                    uniform_freq, uniform_tf, f_peak_homogeneous, A_peak_homogeneous
                )
            else:
                bandwidth_homogeneous = _compute_bandwidth_spectral_moment(
                    uniform_freq, uniform_tf
                )

            # Get heterogeneous ensemble statistics
            het_stats = results["level1"]["ensemble_stats"][group_key]

            # Compute normalized shifts
            if (
                f_peak_homogeneous > 0
                and not np.isnan(het_stats["f_peak"]["mean"])
                and not np.isnan(f_peak_homogeneous)
            ):
                delta_f_norm = (
                    het_stats["f_peak"]["mean"] - f_peak_homogeneous
                ) / f_peak_homogeneous
            else:
                delta_f_norm = np.nan

            if (
                A_peak_homogeneous > 0
                and not np.isnan(het_stats["A_peak"]["mean"])
                and not np.isnan(A_peak_homogeneous)
            ):
                delta_A_factor = (
                    het_stats["A_peak"]["mean"] - A_peak_homogeneous
                ) / A_peak_homogeneous
            else:
                delta_A_factor = np.nan

            if (
                bandwidth_homogeneous > 0
                and not np.isnan(het_stats["bandwidth"]["mean"])
                and not np.isnan(bandwidth_homogeneous)
            ):
                delta_B_factor = (
                    het_stats["bandwidth"]["mean"] - bandwidth_homogeneous
                ) / bandwidth_homogeneous
            else:
                delta_B_factor = np.nan

            results["level2"][group_key] = {
                "delta_f_norm": delta_f_norm,
                "delta_A_factor": delta_A_factor,
                "delta_B_factor": delta_B_factor,
            }

    # Level 3: Coherency-Based Metrics
    print("Computing Level 3: Coherency metrics...")
    # Determine coherency pairs
    if coherency_pairs is None:
        # Get all unique seeds
        all_seeds = sorted(set(k[4] for k in tf_dict.keys()))
        # Generate all pairs
        coherency_pairs = [
            (s1, s2) for i, s1 in enumerate(all_seeds) for s2 in all_seeds[i + 1 :]
        ]

    # Compute coherency for each group
    for group_key in results["level1"]["ensemble_stats"]:
        rH, CV, Vs1, thickness = group_key

        coherency_values = []
        for seed1, seed2 in coherency_pairs:
            # Get TFs for this pair
            key1 = (rH, CV, Vs1, thickness, seed1, "global_avg")
            key2 = (rH, CV, Vs1, thickness, seed2, "global_avg")

            if key1 not in tf_dict or key2 not in tf_dict:
                continue

            freq1, tf1 = tf_dict[key1]
            freq2, tf2 = tf_dict[key2]

            # Use average peak frequency for coherency evaluation
            f_peak1 = results["level1"]["per_realization"][key1]["f_peak"]
            f_peak2 = results["level1"]["per_realization"][key2]["f_peak"]
            f_peak_avg = (f_peak1 + f_peak2) / 2.0

            coherency = _compute_coherency(freq1, tf1, freq2, tf2, f_peak_avg)
            if not np.isnan(coherency):
                coherency_values.append(coherency)

        if len(coherency_values) > 0:
            results["level3"][group_key] = {
                "coherency_peak_mean": np.mean(coherency_values),
                "coherency_peak_std": np.std(coherency_values),
            }
        else:
            results["level3"][group_key] = {
                "coherency_peak_mean": np.nan,
                "coherency_peak_std": np.nan,
            }

    # Level 4: Statistical Relationships
    print("Computing Level 4: Statistical relationships...")
    # Group by (rH, Vs1, thickness) and analyze CV dependence
    cv_groups = defaultdict(list)
    for group_key in results["level1"]["ensemble_stats"]:
        rH, CV, Vs1, thickness = group_key
        cv_groups[(rH, Vs1, thickness)].append((CV, group_key))

    # Fit regression models for each (rH, Vs1, thickness) combination
    for (rH, Vs1, thickness), cv_data in cv_groups.items():
        cv_data_sorted = sorted(cv_data, key=lambda x: x[0])
        CV_values = np.array([cv for cv, _ in cv_data_sorted])
        group_keys = [gk for _, gk in cv_data_sorted]

        if len(CV_values) < 2:
            continue

        # Extract metrics for regression
        f_peak_means = [
            results["level1"]["ensemble_stats"][gk]["f_peak"]["mean"]
            for gk in group_keys
        ]
        A_peak_covs = [
            results["level1"]["ensemble_stats"][gk]["A_peak"]["cov"]
            for gk in group_keys
        ]
        Q_means = [
            results["level1"]["ensemble_stats"][gk]["Q"]["mean"] for gk in group_keys
        ]

        # Get Q(0) for normalization (use lowest CV case)
        Q_0 = Q_means[0] if len(Q_means) > 0 and not np.isnan(Q_means[0]) else None

        # Fit f_peak vs CV
        f_peak_array = np.array(f_peak_means)
        valid_mask = ~np.isnan(f_peak_array)
        if np.sum(valid_mask) >= 2:
            f_peak_linear = _fit_regression(
                CV_values[valid_mask], f_peak_array[valid_mask], "linear"
            )
            f_peak_poly = _fit_regression(
                CV_values[valid_mask], f_peak_array[valid_mask], "polynomial"
            )
        else:
            f_peak_linear = {
                "coeffs": np.array([]),
                "R2": np.nan,
                "y_pred": np.array([]),
            }
            f_peak_poly = {"coeffs": np.array([]), "R2": np.nan, "y_pred": np.array([])}

        # Fit CoV_A vs CV
        cov_A_array = np.array(A_peak_covs)
        valid_mask_cov = ~np.isnan(cov_A_array)
        if np.sum(valid_mask_cov) >= 2:
            cov_A_linear = _fit_regression(
                CV_values[valid_mask_cov], cov_A_array[valid_mask_cov], "linear"
            )
        else:
            cov_A_linear = {
                "coeffs": np.array([]),
                "R2": np.nan,
                "y_pred": np.array([]),
            }

        # Compute Q degradation
        Q_array = np.array(Q_means)
        valid_mask_Q = ~np.isnan(Q_array) & (Q_array > 0)
        if Q_0 is not None and Q_0 > 0 and np.sum(valid_mask_Q) >= 1:
            Q_degradation = Q_array[valid_mask_Q] / Q_0
            Q_degradation_linear = _fit_regression(
                CV_values[valid_mask_Q], Q_degradation, "linear"
            )
        else:
            Q_degradation_linear = {
                "coeffs": np.array([]),
                "R2": np.nan,
                "y_pred": np.array([]),
            }

        # Store results
        group_id = f"rH_{rH}_Vs1_{Vs1}_thickness_{thickness}"
        results["level4"][group_id] = {
            "f_peak_vs_CV": {
                "linear": f_peak_linear,
                "polynomial": f_peak_poly,
            },
            "CoV_A_vs_CV": {
                "linear": cov_A_linear,
            },
            "Q_degradation": {
                "linear": Q_degradation_linear,
                "Q_0": Q_0,
            },
        }

    print("Spectral metrics computation complete!")
    return results


def plot_level1_metrics(
    metrics_results: Dict, output_dir: Path, group_by: str = "CV"
) -> None:
    """
    Plot Level 1 per-realization spectral metrics.

    Creates figures showing distributions of f_peak, A_peak, bandwidth, and Q
    across realizations, grouped by CV or rH.

    Args:
        metrics_results: Results dictionary from compute_spectral_metrics
        output_dir: Output directory for plots
        group_by: Grouping variable ("CV" or "rH")
    """
    level1 = metrics_results.get("level1", {})
    per_realization = level1.get("per_realization", {})
    ensemble_stats = level1.get("ensemble_stats", {})

    if not per_realization:
        print("No Level 1 data to plot")
        return

    # Extract data for plotting
    data = []
    for key, metrics in per_realization.items():
        rH, CV, Vs1, thickness, seed, damping_method = key
        data.append(
            {
                "rH": rH,
                "CV": CV,
                "Vs1": Vs1,
                "thickness": thickness,
                "seed": seed,
                "f_peak": metrics["f_peak"],
                "A_peak": metrics["A_peak"],
                "bandwidth": metrics["bandwidth"],
                "Q": metrics["Q"],
            }
        )

    import pandas as pd

    df = pd.DataFrame(data)

    # Create figure with subplots for each metric
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    metrics_to_plot = [
        ("f_peak", "Peak Frequency (Hz)"),
        ("A_peak", "Peak Amplification"),
        ("bandwidth", "Bandwidth (Hz)"),
        ("Q", "Quality Factor"),
    ]

    for idx, (metric_name, metric_label) in enumerate(metrics_to_plot):
        ax = axes[idx]

        if group_by == "CV":
            # Box plot grouped by CV
            unique_cvs = sorted(df["CV"].unique())
            box_data = []
            for cv in unique_cvs:
                subset = df[df["CV"] == cv][metric_name]
                # Filter out NaN values and convert to numpy array
                values = np.array(subset[~np.isnan(subset)])
                box_data.append(values)
            bp = ax.boxplot(
                box_data,
                tick_labels=[f"CV={cv:.2f}" for cv in unique_cvs],
                patch_artist=True,
            )

            # Color boxes using colorblind palette
            colors = sns.color_palette("colorblind", len(unique_cvs))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_xlabel("Coefficient of Variation (CV)")
            ax.set_title(f"{metric_label} Distribution by CV")

        elif group_by == "rH":
            # Box plot grouped by rH
            unique_rhs = sorted(df["rH"].unique())
            box_data = []
            for rh in unique_rhs:
                subset = df[df["rH"] == rh][metric_name]
                # Filter out NaN values and convert to numpy array
                values = np.array(subset[~np.isnan(subset)])
                box_data.append(values)
            bp = ax.boxplot(
                box_data,
                tick_labels=[f"rH={rh:.0f}" for rh in unique_rhs],
                patch_artist=True,
            )

            # Color boxes using colorblind palette
            colors = sns.color_palette("colorblind", len(unique_rhs))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_xlabel("Horizontal Correlation Length rH (m)")
            ax.set_title(f"{metric_label} Distribution by rH")

        ax.set_ylabel(metric_label)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / f"level1_metrics_grouped_by_{group_by}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved Level 1 plot: {output_file}")
    plt.close()

    # Create ensemble statistics plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (metric_name, metric_label) in enumerate(metrics_to_plot):
        ax = axes[idx]

        # Extract ensemble statistics
        cv_values = []
        mean_values = []
        std_values = []

        for group_key, stats in ensemble_stats.items():
            rH, CV, Vs1, thickness = group_key
            if metric_name in stats and not np.isnan(stats[metric_name]["mean"]):
                cv_values.append(CV)
                mean_values.append(stats[metric_name]["mean"])
                std_values.append(stats[metric_name]["std"])

        if cv_values:
            cv_values = np.array(cv_values)
            mean_values = np.array(mean_values)
            std_values = np.array(std_values)

            # Sort by CV
            sort_idx = np.argsort(cv_values)
            cv_values = cv_values[sort_idx]
            mean_values = mean_values[sort_idx]
            std_values = std_values[sort_idx]

            # Plot mean with error bars
            ax.errorbar(
                cv_values,
                mean_values,
                yerr=std_values,
                marker="o",
                linestyle="-",
                capsize=5,
                capthick=2,
                markersize=8,
                label="Mean ± Std",
            )

            ax.set_xlabel("Coefficient of Variation (CV)")
            ax.set_ylabel(metric_label)
            ax.set_title(f"{metric_label} Ensemble Statistics vs CV")
            ax.grid(True, alpha=0.3)
            ax.legend()

    plt.tight_layout()
    output_file = output_dir / "level1_ensemble_statistics.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved Level 1 ensemble statistics plot: {output_file}")
    plt.close()


def plot_level2_metrics(metrics_results: Dict, output_dir: Path) -> None:
    """
    Plot Level 2 normalized shift metrics.

    Creates figures showing normalized shifts in frequency, amplification, and bandwidth
    relative to homogeneous case.

    Args:
        metrics_results: Results dictionary from compute_spectral_metrics
        output_dir: Output directory for plots
    """
    level2 = metrics_results.get("level2", {})

    if not level2:
        print("No Level 2 data to plot (tf_dict_1D may not have been provided)")
        return

    # Extract data
    data = []
    for group_key, metrics in level2.items():
        rH, CV, Vs1, thickness = group_key
        data.append(
            {
                "rH": rH,
                "CV": CV,
                "Vs1": Vs1,
                "thickness": thickness,
                "delta_f_norm": metrics["delta_f_norm"],
                "delta_A_factor": metrics["delta_A_factor"],
                "delta_B_factor": metrics["delta_B_factor"],
            }
        )

    import pandas as pd

    df = pd.DataFrame(data)

    # Create figure with subplots for each metric
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics_to_plot = [
        ("delta_f_norm", "Normalized Frequency Shift", "Δf_norm"),
        ("delta_A_factor", "Amplification Factor Change", "ΔA_factor"),
        ("delta_B_factor", "Bandwidth Factor Change", "ΔB_factor"),
    ]

    # Get unique rH values for color coding
    unique_rhs = sorted(df["rH"].unique())
    colors = sns.color_palette("colorblind", len(unique_rhs))

    for idx, (metric_name, metric_title, metric_symbol) in enumerate(metrics_to_plot):
        ax = axes[idx]

        for rh_idx, rH in enumerate(unique_rhs):
            rh_data = df[df["rH"] == rH].sort_values(by="CV")  # type: ignore
            if len(rh_data) > 0:
                ax.plot(
                    rh_data["CV"],
                    rh_data[metric_name],
                    marker="o",
                    linestyle="-",
                    linewidth=2,
                    markersize=8,
                    label=f"rH={rH:.0f} m",
                    color=colors[rh_idx],
                )

        ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xlabel("Coefficient of Variation (CV)")
        ax.set_ylabel(metric_symbol)
        ax.set_title(metric_title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    output_file = output_dir / "level2_normalized_shifts.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved Level 2 plot: {output_file}")
    plt.close()


def plot_level3_metrics(metrics_results: Dict, output_dir: Path) -> None:
    """
    Plot Level 3 coherency metrics.

    Creates figures showing coherency at peak frequency vs CV.

    Args:
        metrics_results: Results dictionary from compute_spectral_metrics
        output_dir: Output directory for plots
    """
    level3 = metrics_results.get("level3", {})

    if not level3:
        print("No Level 3 data to plot")
        return

    # Extract data
    data = []
    for group_key, metrics in level3.items():
        rH, CV, Vs1, thickness = group_key
        if not np.isnan(metrics["coherency_peak_mean"]):
            data.append(
                {
                    "rH": rH,
                    "CV": CV,
                    "Vs1": Vs1,
                    "thickness": thickness,
                    "coherency_mean": metrics["coherency_peak_mean"],
                    "coherency_std": metrics["coherency_peak_std"],
                }
            )

    if not data:
        print("No valid Level 3 coherency data to plot")
        return

    import pandas as pd

    df = pd.DataFrame(data)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get unique rH values for color coding
    unique_rhs = sorted(df["rH"].unique())
    colors = sns.color_palette("colorblind", len(unique_rhs))

    for rh_idx, rH in enumerate(unique_rhs):
        rh_data = df[df["rH"] == rH].sort_values(by="CV")  # type: ignore
        if len(rh_data) > 0:
            ax.errorbar(
                rh_data["CV"],
                rh_data["coherency_mean"],
                yerr=rh_data["coherency_std"],
                marker="o",
                linestyle="-",
                linewidth=2,
                capsize=5,
                capthick=2,
                markersize=8,
                label=f"rH={rH:.0f} m",
                color=colors[rh_idx],
            )

    ax.set_xlabel("Coefficient of Variation (CV)")
    ax.set_ylabel("Coherency at Peak Frequency")
    ax.set_title("Coherency Degradation with Spatial Variability")
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    output_file = output_dir / "level3_coherency.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved Level 3 plot: {output_file}")
    plt.close()


def plot_level4_metrics(metrics_results: Dict, output_dir: Path) -> None:
    """
    Plot Level 4 statistical relationships.

    Creates figures showing regression models for f_peak vs CV, CoV_A vs CV, and Q degradation.

    Args:
        metrics_results: Results dictionary from compute_spectral_metrics
        output_dir: Output directory for plots
    """
    level1 = metrics_results.get("level1", {})
    level4 = metrics_results.get("level4", {})

    if not level4:
        print("No Level 4 data to plot")
        return

    ensemble_stats = level1.get("ensemble_stats", {})

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Get unique group IDs
    group_ids = list(level4.keys())
    colors = sns.color_palette("colorblind", len(group_ids))

    # Plot 1: f_peak vs CV
    ax1 = axes[0]
    for idx, group_id in enumerate(group_ids):
        group_data = level4[group_id]
        f_peak_data = group_data.get("f_peak_vs_CV", {})

        # Extract CV and f_peak values from ensemble stats
        cv_values = []
        f_peak_values = []
        for group_key, stats in ensemble_stats.items():
            rH, CV, Vs1, thickness = group_key
            if group_id == f"rH_{rH}_Vs1_{Vs1}_thickness_{thickness}":
                if not np.isnan(stats["f_peak"]["mean"]):
                    cv_values.append(CV)
                    f_peak_values.append(stats["f_peak"]["mean"])

        if len(cv_values) >= 2:
            cv_values = np.array(cv_values)
            f_peak_values = np.array(f_peak_values)

            # Sort by CV
            sort_idx = np.argsort(cv_values)
            cv_values = cv_values[sort_idx]
            f_peak_values = f_peak_values[sort_idx]

            # Plot data points
            ax1.scatter(
                cv_values,
                f_peak_values,
                color=colors[idx],
                s=100,
                alpha=0.7,
                label=group_id.replace("_", " "),
            )

            # Plot linear fit
            linear_fit = f_peak_data.get("linear", {})
            if linear_fit.get("coeffs") is not None and len(linear_fit["coeffs"]) > 0:
                coeffs = linear_fit["coeffs"]
                cv_fit = np.linspace(cv_values.min(), cv_values.max(), 100)
                f_peak_fit = np.polyval(coeffs, cv_fit)
                ax1.plot(
                    cv_fit,
                    f_peak_fit,
                    color=colors[idx],
                    linestyle="--",
                    linewidth=2,
                    alpha=0.7,
                    label=f"Linear (R²={linear_fit.get('R2', 0):.3f})",
                )

    ax1.set_xlabel("Coefficient of Variation (CV)")
    ax1.set_ylabel("Peak Frequency (Hz)")
    ax1.set_title("Peak Frequency vs CV")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    # Plot 2: CoV_A vs CV
    ax2 = axes[1]
    for idx, group_id in enumerate(group_ids):
        group_data = level4[group_id]
        cov_A_data = group_data.get("CoV_A_vs_CV", {})

        # Extract CV and CoV_A values
        cv_values = []
        cov_A_values = []
        for group_key, stats in ensemble_stats.items():
            rH, CV, Vs1, thickness = group_key
            if group_id == f"rH_{rH}_Vs1_{Vs1}_thickness_{thickness}":
                if not np.isnan(stats["A_peak"]["cov"]):
                    cv_values.append(CV)
                    cov_A_values.append(stats["A_peak"]["cov"])

        if len(cv_values) >= 2:
            cv_values = np.array(cv_values)
            cov_A_values = np.array(cov_A_values)

            # Sort by CV
            sort_idx = np.argsort(cv_values)
            cv_values = cv_values[sort_idx]
            cov_A_values = cov_A_values[sort_idx]

            # Plot data points
            ax2.scatter(
                cv_values,
                cov_A_values,
                color=colors[idx],
                s=100,
                alpha=0.7,
                label=group_id.replace("_", " "),
            )

            # Plot linear fit
            linear_fit = cov_A_data.get("linear", {})
            if linear_fit.get("coeffs") is not None and len(linear_fit["coeffs"]) > 0:
                coeffs = linear_fit["coeffs"]
                cv_fit = np.linspace(cv_values.min(), cv_values.max(), 100)
                cov_A_fit = np.polyval(coeffs, cv_fit)
                ax2.plot(
                    cv_fit,
                    cov_A_fit,
                    color=colors[idx],
                    linestyle="--",
                    linewidth=2,
                    alpha=0.7,
                    label=f"Linear (R²={linear_fit.get('R2', 0):.3f})",
                )

    ax2.set_xlabel("Coefficient of Variation (CV)")
    ax2.set_ylabel("CoV of Amplification")
    ax2.set_title("Amplification Scatter vs CV")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    # Plot 3: Q degradation
    ax3 = axes[2]
    for idx, group_id in enumerate(group_ids):
        group_data = level4[group_id]
        q_degradation_data = group_data.get("Q_degradation", {})
        Q_0 = q_degradation_data.get("Q_0")

        # Extract CV and Q values
        cv_values = []
        q_values = []
        for group_key, stats in ensemble_stats.items():
            rH, CV, Vs1, thickness = group_key
            if group_id == f"rH_{rH}_Vs1_{Vs1}_thickness_{thickness}":
                if not np.isnan(stats["Q"]["mean"]) and Q_0 is not None and Q_0 > 0:
                    cv_values.append(CV)
                    q_values.append(stats["Q"]["mean"] / Q_0)

        if len(cv_values) >= 1:
            cv_values = np.array(cv_values)
            q_values = np.array(q_values)

            # Sort by CV
            sort_idx = np.argsort(cv_values)
            cv_values = cv_values[sort_idx]
            q_values = q_values[sort_idx]

            # Plot data points
            ax3.scatter(
                cv_values,
                q_values,
                color=colors[idx],
                s=100,
                alpha=0.7,
                label=group_id.replace("_", " "),
            )

            # Plot linear fit
            linear_fit = q_degradation_data.get("linear", {})
            if linear_fit.get("coeffs") is not None and len(linear_fit["coeffs"]) > 0:
                coeffs = linear_fit["coeffs"]
                cv_fit = np.linspace(cv_values.min(), cv_values.max(), 100)
                q_fit = np.polyval(coeffs, cv_fit)
                ax3.plot(
                    cv_fit,
                    q_fit,
                    color=colors[idx],
                    linestyle="--",
                    linewidth=2,
                    alpha=0.7,
                    label=f"Linear (R²={linear_fit.get('R2', 0):.3f})",
                )

    ax3.axhline(
        1.0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Q(0)"
    )
    ax3.set_xlabel("Coefficient of Variation (CV)")
    ax3.set_ylabel("Q(CV) / Q(0)")
    ax3.set_title("Quality Factor Degradation")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)

    plt.tight_layout()
    output_file = output_dir / "level4_statistical_relationships.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved Level 4 plot: {output_file}")
    plt.close()


def plot_all_spectral_metrics(
    metrics_results: Dict, output_dir: Path, group_by: str = "CV"
) -> None:
    """
    Create all spectral metrics plots.

    Args:
        metrics_results: Results dictionary from compute_spectral_metrics
        output_dir: Output directory for plots
        group_by: Grouping variable for Level 1 plots ("CV" or "rH")
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creating Level 1 plots...")
    plot_level1_metrics(metrics_results, output_dir, group_by=group_by)

    print("Creating Level 2 plots...")
    plot_level2_metrics(metrics_results, output_dir)

    print("Creating Level 3 plots...")
    plot_level3_metrics(metrics_results, output_dir)

    print("Creating Level 4 plots...")
    plot_level4_metrics(metrics_results, output_dir)

    print("All spectral metrics plots complete!")


if __name__ == "__main__":
    main()
    # plot_all_spectral_metrics(metrics_results, output_dir)
