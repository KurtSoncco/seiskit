"""
Compare transfer function results between damping_f0, new_sensitivity, and final_generation directories.

This script:
1. Loads transfer function dictionaries from all three directories
2. Matches cases with identical parameters (Vs1, rH, CV, seed, damping_method='global_avg')
3. Creates comparison plots with two subplots:
   - Overlay of all three transfer functions
   - Peak differences between the models
4. Limits frequency range to 0.1-10 Hz

All figures follow the centralized publication-quality style from
:mod:`seiskit.plot_config`.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from seiskit.plot_config import (
    COLORBLIND_COLORS,
    add_subfigure_label,
    apply_style,
    format_title,
    place_legend,
    to_title_case,
)
from seiskit.plot_config.labels import format_label


def load_tf_dict_from_pickle(pickle_file: Path) -> Optional[Dict]:
    """Load transfer function dictionary from pickle file."""
    if not pickle_file.exists():
        print(f"Warning: Pickle file not found: {pickle_file}")
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


def load_tf_dicts(
    damping_f0_dir: Path, new_sensitivity_dir: Path, final_generation_dir: Path
) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
    """Load transfer function dictionaries from all three directories."""
    damping_f0_pickle = damping_f0_dir / "transfer_function_results" / "transfer_functions_dict.pkl"
    new_sensitivity_pickle = (
        new_sensitivity_dir / "transfer_function_results" / "transfer_functions_dict.pkl"
    )
    final_generation_pickle = (
        final_generation_dir / "transfer_function_results" / "transfer_functions_dict.pkl"
    )

    tf_dict_damping_f0 = load_tf_dict_from_pickle(damping_f0_pickle)
    tf_dict_new_sensitivity = load_tf_dict_from_pickle(new_sensitivity_pickle)
    tf_dict_final_generation = load_tf_dict_from_pickle(final_generation_pickle)

    return tf_dict_damping_f0, tf_dict_new_sensitivity, tf_dict_final_generation


def filter_global_avg(tf_dict: Dict) -> Dict:
    """Filter dictionary to only include entries with damping_method='global_avg'."""
    if len(tf_dict) == 0:
        return tf_dict

    sample_key = next(iter(tf_dict.keys()))
    if len(sample_key) == 5:
        print(
            f"Dictionary has 5-tuple keys (final_generation), all {len(tf_dict)} entries included"
        )
        return tf_dict
    else:
        filtered = {
            key: value
            for key, value in tf_dict.items()
            if key[5] == "global_avg"
        }
        print(f"Filtered to {len(filtered)} entries with damping_method='global_avg'")
        return filtered


def match_cases(
    tf_dict_damping_f0: Dict,
    tf_dict_new_sensitivity: Dict,
    tf_dict_final_generation: Dict,
) -> List[Tuple[Tuple, Tuple, Tuple]]:
    """Match cases between all three dictionaries based on (Vs1, rH, CV, thickness, seed)."""
    matched_cases = []

    lookup_new_sensitivity = {}
    for key, value in tf_dict_new_sensitivity.items():
        match_key = (key[2], key[0], key[1], key[3], key[4])
        lookup_new_sensitivity[match_key] = (key, value)

    lookup_final_generation = {}
    for key, value in tf_dict_final_generation.items():
        match_key = (key[2], key[0], key[1], key[3], key[4])
        lookup_final_generation[match_key] = (key, value)

    for key_damping_f0, value_damping_f0 in tf_dict_damping_f0.items():
        match_key = (
            key_damping_f0[2], key_damping_f0[0], key_damping_f0[1],
            key_damping_f0[3], key_damping_f0[4],
        )

        if match_key in lookup_new_sensitivity and match_key in lookup_final_generation:
            key_new_sensitivity, value_new_sensitivity = lookup_new_sensitivity[match_key]
            key_final_generation, value_final_generation = lookup_final_generation[match_key]
            matched_cases.append(
                (
                    (key_damping_f0, value_damping_f0),
                    (key_new_sensitivity, value_new_sensitivity),
                    (key_final_generation, value_final_generation),
                )
            )

    print(f"Found {len(matched_cases)} matched cases across all three datasets")
    return matched_cases


def interpolate_to_common_freq(
    freq1: np.ndarray, tf1: np.ndarray,
    freq2: np.ndarray, tf2: np.ndarray,
    freq3: np.ndarray, tf3: np.ndarray,
    freq_min: float = 0.1, freq_max: float = 10.0, n_points: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate all three transfer functions to a common frequency grid."""
    common_freq = np.logspace(np.log10(freq_min), np.log10(freq_max), n_points)

    mask1 = (freq1 >= freq_min) & (freq1 <= freq_max)
    mask2 = (freq2 >= freq_min) & (freq2 <= freq_max)
    mask3 = (freq3 >= freq_min) & (freq3 <= freq_max)

    def _interp(freq_filt, tf_filt):
        if len(freq_filt) > 1:
            return interp1d(freq_filt, tf_filt, kind="linear", bounds_error=False, fill_value=np.nan)(common_freq)
        return np.full_like(common_freq, np.nan)

    tf1_interp = _interp(freq1[mask1], tf1[mask1])
    tf2_interp = _interp(freq2[mask2], tf2[mask2])
    tf3_interp = _interp(freq3[mask3], tf3[mask3])

    return common_freq, tf1_interp, tf2_interp, tf3_interp


def compute_difference(tf1: np.ndarray, tf2: np.ndarray, use_ratio: bool = False) -> np.ndarray:
    """Compute difference or ratio between two transfer functions."""
    if use_ratio:
        return np.divide(tf2, tf1, out=np.full_like(tf2, np.nan), where=(tf1 != 0))
    return tf2 - tf1


def create_comparison_plot(
    matched_case: Tuple[Tuple, Tuple, Tuple],
    output_dir: Path,
    use_ratio: bool = False,
) -> None:
    """Create a comparison plot with two subplots for a matched case."""
    apply_style()

    (
        (key_damping_f0, (freq_damping_f0, tf_damping_f0)),
        (key_new_sensitivity, (freq_new_sensitivity, tf_new_sensitivity)),
        (key_final_generation, (freq_final_generation, tf_final_generation)),
    ) = matched_case

    Vs1 = key_damping_f0[2]
    rH = key_damping_f0[0]
    CV = key_damping_f0[1]
    seed = key_damping_f0[4]
    thickness_damping_f0 = key_damping_f0[3]
    thickness_new_sensitivity = key_new_sensitivity[3]
    thickness_final_generation = key_final_generation[3]

    (
        common_freq,
        tf_damping_f0_interp,
        tf_new_sensitivity_interp,
        tf_final_generation_interp,
    ) = interpolate_to_common_freq(
        freq_damping_f0, tf_damping_f0,
        freq_new_sensitivity, tf_new_sensitivity,
        freq_final_generation, tf_final_generation,
    )

    valid_mask = ~(
        np.isnan(tf_damping_f0_interp)
        | np.isnan(tf_new_sensitivity_interp)
        | np.isnan(tf_final_generation_interp)
    )
    common_freq_plot = common_freq[valid_mask]
    tf_damping_f0_plot = tf_damping_f0_interp[valid_mask]
    tf_new_sensitivity_plot = tf_new_sensitivity_interp[valid_mask]
    tf_final_generation_plot = tf_final_generation_interp[valid_mask]

    if len(common_freq_plot) == 0:
        print(
            f"Warning: No valid data points for Vs1={Vs1}, "
            f"{format_label(f'rH={rH}')}, {format_label(f'CV={CV}')}, seed={seed}. Skipping."
        )
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Overlay comparison
    ax1 = axes[0]
    ax1.loglog(
        common_freq_plot, tf_damping_f0_plot,
        color=COLORBLIND_COLORS[0], linewidth=2.5, linestyle="-",
        label="damping\\_f0",
    )
    ax1.loglog(
        common_freq_plot, tf_new_sensitivity_plot,
        color=COLORBLIND_COLORS[1], linewidth=2.5, linestyle="--",
        label="new\\_sensitivity",
    )
    ax1.loglog(
        common_freq_plot, tf_final_generation_plot,
        color=COLORBLIND_COLORS[2], linewidth=2.5, linestyle="-.",
        label="final\\_generation",
    )
    ax1.set_xlabel(to_title_case("Frequency (Hz)"))
    ax1.set_ylabel(to_title_case("Transfer Function Magnitude"))
    format_title(
        "Overlay Comparison",
        subtitle=(
            f"Vs1={Vs1:.0f} m/s, " + format_label(f"rH={rH:.0f} m") + ", "
            + format_label(f"CV={CV:.2f}") + f", $H$={thickness_damping_f0:.0f} m, seed={seed}"
        ),
        ax=ax1,
    )
    ax1.set_xlim(0.1, 10)
    ax1.grid(True, alpha=0.3)
    place_legend(ax1, position="bottom")
    add_subfigure_label(ax1, 0)

    # Subplot 2: Differences
    ax2 = axes[1]
    diff_new = compute_difference(tf_damping_f0_plot, tf_new_sensitivity_plot, use_ratio=use_ratio)
    diff_final = compute_difference(tf_damping_f0_plot, tf_final_generation_plot, use_ratio=use_ratio)

    if use_ratio:
        ax2.semilogx(
            common_freq_plot, diff_new,
            color=COLORBLIND_COLORS[1], linewidth=2.0, linestyle="--",
            label="new\\_sensitivity / damping\\_f0",
        )
        ax2.semilogx(
            common_freq_plot, diff_final,
            color=COLORBLIND_COLORS[2], linewidth=2.0, linestyle="-.",
            label="final\\_generation / damping\\_f0",
        )
        ax2.axhline(1.0, color="black", linestyle=":", linewidth=1.5, alpha=0.7, label="Ratio = 1.0")
        ax2.set_ylabel(to_title_case("Ratio (Relative To Damping_F0)"))
    else:
        ax2.semilogx(
            common_freq_plot, diff_new,
            color=COLORBLIND_COLORS[1], linewidth=2.0, linestyle="--",
            label="new\\_sensitivity $-$ damping\\_f0",
        )
        ax2.semilogx(
            common_freq_plot, diff_final,
            color=COLORBLIND_COLORS[2], linewidth=2.0, linestyle="-.",
            label="final\\_generation $-$ damping\\_f0",
        )
        ax2.axhline(0.0, color="black", linestyle=":", linewidth=1.5, alpha=0.7, label="Zero Difference")
        ax2.set_ylabel(to_title_case("Difference (Relative To Damping_F0)"))

    ax2.set_xlabel(to_title_case("Frequency (Hz)"))
    format_title(
        "Differences",
        subtitle=(
            f"Vs1={Vs1:.0f} m/s, " + format_label(f"rH={rH:.0f} m") + ", "
            + format_label(f"CV={CV:.2f}") + f", $H$={thickness_damping_f0:.0f} m, seed={seed}"
        ),
        ax=ax2,
    )
    ax2.set_xlim(0.1, 10)
    ax2.grid(True, alpha=0.3)
    place_legend(ax2, position="bottom")
    add_subfigure_label(ax2, 1)

    thicknesses = [thickness_damping_f0, thickness_new_sensitivity, thickness_final_generation]
    if len(set(thicknesses)) > 1:
        fig.suptitle(
            f"Note: thickness differs (damping\\_f0={thickness_damping_f0:.0f} m, "
            f"new\\_sensitivity={thickness_new_sensitivity:.0f} m, "
            f"final\\_generation={thickness_final_generation:.0f} m)",
            fontsize=10,
            y=0.02,
            style="italic",
        )

    plt.tight_layout()

    filename = f"comparison_Vs1_{Vs1:.0f}_rH_{rH:.0f}_CV_{CV:.2f}_h_{thickness_damping_f0:.0f}_seed_{seed}.png"
    output_file = output_dir / filename
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved comparison plot: {output_file}")


def main():
    """Main function to create comparison plots."""
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    damping_f0_dir = script_dir / "damping_f0"
    new_sensitivity_dir = script_dir / "new_sensitivity"
    final_generation_dir = script_dir / "final_generation"
    output_dir = script_dir / "comparison_plots"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Comparison Plots: damping_f0 vs new_sensitivity vs final_generation")
    print("=" * 60)

    print("\nLoading transfer function dictionaries...")
    tf_dict_damping_f0, tf_dict_new_sensitivity, tf_dict_final_generation = load_tf_dicts(
        damping_f0_dir, new_sensitivity_dir, final_generation_dir
    )

    if (
        tf_dict_damping_f0 is None
        or tf_dict_new_sensitivity is None
        or tf_dict_final_generation is None
    ):
        print("Error: Could not load one or more dictionaries. Exiting.")
        return

    print("\nFiltering for damping_method='global_avg'...")
    tf_dict_damping_f0_filtered = filter_global_avg(tf_dict_damping_f0)
    tf_dict_new_sensitivity_filtered = filter_global_avg(tf_dict_new_sensitivity)
    tf_dict_final_generation_filtered = filter_global_avg(tf_dict_final_generation)

    print("\nMatching cases...")
    matched_cases = match_cases(
        tf_dict_damping_f0_filtered,
        tf_dict_new_sensitivity_filtered,
        tf_dict_final_generation_filtered,
    )

    if len(matched_cases) == 0:
        print("Warning: No matched cases found. Exiting.")
        return

    print("\n" + "=" * 60)
    print("Generating comparison plots...")
    print("=" * 60)

    plot_count = 0
    for matched_case in matched_cases:
        try:
            create_comparison_plot(matched_case, output_dir, use_ratio=False)
            plot_count += 1
        except Exception as e:
            (key_damping_f0, _), (key_new_sensitivity, _), (key_final_generation, _) = matched_case
            print(
                f"Error creating plot for Vs1={key_damping_f0[2]}, "
                f"{format_label(f'rH={key_damping_f0[0]}')}, "
                f"{format_label(f'CV={key_damping_f0[1]}')}, seed={key_damping_f0[4]}: {e}"
            )

    print("\n" + "=" * 60)
    print(f"Comparison plots complete! Created {plot_count} plots.")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
