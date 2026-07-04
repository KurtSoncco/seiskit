"""
Create validation plots comparing uniform 1D OpenSees results with variability results.

This script generates validation plots to demonstrate consistency with Huang et al.'s findings:
1. Uniform vs theoretical resonance frequencies
2. CV effect at fixed rH
3. rH effect at fixed CV
4. Velocity-dependent sensitivity
5. Amplification vs CV summary

All figures follow the centralized publication-quality style from
:mod:`seiskit.plot_config`.
"""

import csv
import os
import re
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from compute_transfer_functions import (
    compute_coefficient_of_variation,
    compute_geometric_mean,
    compute_transfer_function,
    load_acceleration_files,
    load_all_results,
    load_tf_dict_from_pickle,
)

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


def parse_1D_folder(folder_name: str) -> Dict[str, float | str] | None:
    """Parse 1D result folder name to extract parameters."""
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
    """Load transfer functions from 1D_results folders."""
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

        thickness_val = float(params["thickness"])
        accel_data = load_acceleration_files(folder, thickness_val)
        if accel_data is None:
            print(f"Skipping folder (no acceleration data): {folder.name}")
            skipped_count += 1
            continue

        base_data, top_data = accel_data

        try:
            freq, tf = compute_transfer_function(base_data, top_data, Vsmin=Vsmin)
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
    """Compute theoretical resonance frequencies using Kramer's formula."""
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
    """Plot 1: Uniform vs theoretical resonance frequencies."""
    apply_style()

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
                        color=COLORBLIND_COLORS[1],
                        linewidth=2.5,
                        linestyle="--",
                        label=format_label("rH=50, CV=0.1 (geomean)"),
                    )

                theoretical_freqs = compute_theoretical_resonance_frequencies(
                    Vs1, thickness, n_modes=5
                )
                for fn in theoretical_freqs:
                    ax.axvline(
                        fn, color=COLORBLIND_COLORS[0], linestyle=":", linewidth=1.5, alpha=0.7
                    )

                ax.set_xlabel(to_title_case("Frequency (Hz)"))
                ax.set_ylabel(to_title_case("Transfer Function Magnitude"))
                format_title(f"$H$={thickness:.0f} m", ax=ax)
                ax.set_xlim(1e-1, 5e1)
                ax.set_ylim(1e-2, 1e2)
                ax.grid(True, alpha=0.3)
                place_legend(ax, position="bottom")
                add_subfigure_label(ax, idx)

            fig.suptitle(
                f"$\\bf{{Vs1={Vs1:.0f}\\ m/s,\\ damping={damping_method}}}$",
                fontsize=14,
                y=1.02,
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
    """Plot 2: CV effect at fixed rH."""
    apply_style()

    Vs1_values = sorted(set(k[2] for k in tf_dict.keys()))
    CV_values = sorted(set(k[1] for k in tf_dict.keys()))
    damping_methods = sorted(set(k[5] for k in tf_dict.keys()))

    cmap = get_crameri_cmap("batlow")

    for damping_method in damping_methods:
        for Vs1 in Vs1_values:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            ax1 = axes[0]

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

            colors = [cmap(v) for v in np.linspace(0.2, 0.8, len(CV_values))]
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
                        label=format_label(f"CV={cv_val:.2f} (geomean)"),
                    )

            ax1.set_xlabel(to_title_case("Frequency (Hz)"))
            ax1.set_ylabel(to_title_case("Transfer Function Magnitude"))
            format_title(
                f"Geomean TF: Vs1={Vs1:.0f} m/s, $H$={thickness:.0f} m, "
                + format_label(f"rH={rH:.0f} m")
                + f", damping={damping_method}",
                ax=ax1,
            )
            ax1.set_xlim(1e-1, 5e1)
            ax1.set_ylim(1e-2, 1e2)
            ax1.grid(True, alpha=0.3)
            place_legend(ax1, position="bottom")
            add_subfigure_label(ax1, 0)

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
                        label=format_label(f"CV={cv_val:.2f}"),
                    )

            ax2.set_xlabel(to_title_case("Frequency (Hz)"))
            ax2.set_ylabel(to_title_case(format_label("CoV")))
            format_title(
                format_label("CoV")
                + f" vs Frequency: Vs1={Vs1:.0f} m/s, $H$={thickness:.0f} m, "
                + format_label(f"rH={rH:.0f} m")
                + f", damping={damping_method}",
                ax=ax2,
            )
            ax2.set_xlim(1e-1, 5e1)
            ax2.grid(True, alpha=0.3)
            place_legend(ax2, position="bottom")
            add_subfigure_label(ax2, 1)

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
    """Plot 3: rH effect at fixed CV."""
    apply_style()

    Vs1_values = sorted(set(k[2] for k in tf_dict.keys()))
    rH_values = sorted(set(k[0] for k in tf_dict.keys()))
    damping_methods = sorted(set(k[5] for k in tf_dict.keys()))

    for damping_method in damping_methods:
        for Vs1 in Vs1_values:
            fig, axes = plt.subplots(1, 3, figsize=(24, 6))

            # Panel 1: Transfer functions
            ax1 = axes[0]

            theoretical_freqs = compute_theoretical_resonance_frequencies(Vs1, thickness, n_modes=5)
            window_fraction = 0.1

            rh_colors = [COLORBLIND_COLORS[0], COLORBLIND_COLORS[1]]
            for rH, color in zip(rH_values, rh_colors):
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
                        label=format_label(f"rH={rH:.0f} m"),
                    )

            for fn in theoretical_freqs:
                ax1.axvline(fn, color=COLORBLIND_COLORS[0], linestyle=":", linewidth=1.5, alpha=0.7)
                window_width = window_fraction * fn
                ax1.axvline(
                    fn - window_width, color="gray", linestyle="--", linewidth=1.0, alpha=0.5
                )
                ax1.axvline(
                    fn + window_width, color="gray", linestyle="--", linewidth=1.0, alpha=0.5
                )

            ax1.set_xlabel(to_title_case("Frequency (Hz)"))
            ax1.set_ylabel(to_title_case("Transfer Function Magnitude"))
            format_title(
                f"Geomean TF: Vs1={Vs1:.0f} m/s, $H$={thickness:.0f} m, "
                + format_label(f"CV={CV:.2f}")
                + f", damping={damping_method}",
                ax=ax1,
            )
            ax1.set_xlim(1e-1, 5e1)
            ax1.set_ylim(1e-2, 1e2)
            ax1.grid(True, alpha=0.3)
            place_legend(ax1, position="bottom")
            add_subfigure_label(ax1, 0)

            # Panel 2: Local band-averaged TF box plot
            ax2 = axes[1]

            mode_data = {}
            for mode_num in range(1, len(theoretical_freqs) + 1):
                mode_data[mode_num] = {}
                for rH in rH_values:
                    mode_data[mode_num][rH] = []

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
                        for mode_num, fn in enumerate(theoretical_freqs, start=1):
                            window_width = window_fraction * fn
                            freq_low = fn - window_width
                            freq_high = fn + window_width
                            freq_mask = (freq >= freq_low) & (freq <= freq_high)
                            if np.any(freq_mask):
                                band_avg = np.mean(tf[freq_mask])
                                mode_data[mode_num][rH].append(band_avg)

            if len(mode_data) > 0:
                plot_data = []
                plot_labels = []
                plot_positions = []
                pos = 0
                for mode_num in sorted(mode_data.keys()):
                    for rH in rH_values:
                        if len(mode_data[mode_num][rH]) > 0:
                            plot_data.append(mode_data[mode_num][rH])
                            plot_labels.append(f"Mode {mode_num}\n" + format_label(f"rH={rH:.0f}m"))
                            plot_positions.append(pos)
                            pos += 1
                    pos += 0.5

                if len(plot_data) > 0:
                    bp = ax2.boxplot(
                        plot_data,
                        positions=plot_positions,
                        tick_labels=plot_labels,
                        patch_artist=True,
                        widths=0.4,
                    )

                    colors_box = []
                    for label in plot_labels:
                        if "rH=10" in label or "r_{h}=10" in label:
                            colors_box.append(COLORBLIND_COLORS[0])
                        else:
                            colors_box.append(COLORBLIND_COLORS[1])

                    for patch, color in zip(bp["boxes"], colors_box):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)

                    ax2.set_ylabel(to_title_case("Local Band-Averaged TF"))
                    format_title(
                        "Local Band-Averaged TF Around Each Mode",
                        subtitle=f"Vs1={Vs1:.0f} m/s, $H$={thickness:.0f} m",
                        ax=ax2,
                    )
                    ax2.grid(True, alpha=0.3, axis="y")
                    add_subfigure_label(ax2, 1)

                    mode_separators = []
                    current_pos = 0
                    for mode_num in sorted(mode_data.keys())[:-1]:
                        for rH in rH_values:
                            if len(mode_data[mode_num][rH]) > 0:
                                current_pos += 1
                        current_pos += 0.5
                        mode_separators.append(current_pos)

                    for sep_pos in mode_separators:
                        ax2.axvline(
                            sep_pos,
                            color="gray",
                            linestyle="--",
                            linewidth=0.5,
                            alpha=0.5,
                        )

            # Panel 3: Normalized local band-averaged TF
            ax3 = axes[2]

            uniform_key = (Vs1, thickness, damping_method)
            uniform_peaks = {}
            if uniform_key in tf_dict_1D:
                uniform_freq, uniform_tf = tf_dict_1D[uniform_key]
                for mode_num, fn in enumerate(theoretical_freqs, start=1):
                    window_width = window_fraction * fn
                    freq_mask = (uniform_freq >= fn - window_width) & (
                        uniform_freq <= fn + window_width
                    )
                    if np.any(freq_mask):
                        uniform_peaks[mode_num] = np.mean(uniform_tf[freq_mask])

            normalized_mode_data = {}
            for mode_num in range(1, len(theoretical_freqs) + 1):
                normalized_mode_data[mode_num] = {}
                for rH in rH_values:
                    normalized_mode_data[mode_num][rH] = []
                    if mode_num in uniform_peaks and mode_num in mode_data:
                        uniform_peak = uniform_peaks[mode_num]
                        for band_avg in mode_data[mode_num][rH]:
                            normalized_mode_data[mode_num][rH].append(band_avg / uniform_peak)

            has_normalized_data = any(
                len(normalized_mode_data[mn][rh]) > 0
                for mn in normalized_mode_data
                for rh in rH_values
            )

            if has_normalized_data:
                plot_data_norm = []
                plot_labels_norm = []
                plot_positions_norm = []
                pos = 0
                for mode_num in sorted(normalized_mode_data.keys()):
                    for rH in rH_values:
                        if len(normalized_mode_data[mode_num][rH]) > 0:
                            plot_data_norm.append(normalized_mode_data[mode_num][rH])
                            plot_labels_norm.append(
                                f"Mode {mode_num}\n" + format_label(f"rH={rH:.0f}m")
                            )
                            plot_positions_norm.append(pos)
                            pos += 1
                    pos += 0.5

                if len(plot_data_norm) > 0:
                    bp_norm = ax3.boxplot(
                        plot_data_norm,
                        positions=plot_positions_norm,
                        tick_labels=plot_labels_norm,
                        patch_artist=True,
                        widths=0.4,
                    )

                    colors_box_norm = []
                    for label in plot_labels_norm:
                        if "rH=10" in label or "r_{h}=10" in label:
                            colors_box_norm.append(COLORBLIND_COLORS[0])
                        else:
                            colors_box_norm.append(COLORBLIND_COLORS[1])

                    for patch, color in zip(bp_norm["boxes"], colors_box_norm):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)

                    ax3.set_ylabel(
                        to_title_case("Normalized Band-Averaged TF\n(Relative To Uniform 1D)")
                    )
                    format_title(
                        "Normalized Band-Averaged TF",
                        subtitle=f"Vs1={Vs1:.0f} m/s, $H$={thickness:.0f} m",
                        ax=ax3,
                    )
                    ax3.grid(True, alpha=0.3, axis="y")
                    ax3.axhline(
                        1.0,
                        color="black",
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.7,
                        label="Uniform (1D)",
                    )
                    place_legend(ax3, position="bottom")
                    add_subfigure_label(ax3, 2)

                    mode_separators_norm = []
                    current_pos = 0
                    for mode_num in sorted(normalized_mode_data.keys())[:-1]:
                        for rH in rH_values:
                            if len(normalized_mode_data[mode_num][rH]) > 0:
                                current_pos += 1
                        current_pos += 0.5
                        mode_separators_norm.append(current_pos)

                    for sep_pos in mode_separators_norm:
                        ax3.axvline(
                            sep_pos,
                            color="gray",
                            linestyle="--",
                            linewidth=0.5,
                            alpha=0.5,
                        )
            else:
                ax3.text(
                    0.5,
                    0.5,
                    "No normalized data\n(Uniform TF not available)",
                    ha="center",
                    va="center",
                    transform=ax3.transAxes,
                    color="gray",
                )
                ax3.set_ylabel(
                    to_title_case("Normalized Band-Averaged TF\n(Relative To Uniform 1D)")
                )
                add_subfigure_label(ax3, 2)

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
    """Plot 4: Velocity-dependent sensitivity."""
    apply_style()

    Vs1_values = sorted(set(k[2] for k in tf_dict.keys()))[:3]
    rH_values = sorted(set(k[0] for k in tf_dict.keys()))
    CV_values = sorted(set(k[1] for k in tf_dict.keys()))
    damping_methods = sorted(set(k[5] for k in tf_dict.keys()))

    for damping_method in damping_methods:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        combinations = [(rH, cv) for rH in rH_values for cv in CV_values]
        combo_colors = COLORBLIND_COLORS[: len(combinations)]
        colors_map = {
            combo: combo_colors[i % len(combo_colors)] for i, combo in enumerate(combinations)
        }

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
                    label = format_label(f"rH={rH:.0f}") + ", " + format_label(f"CV={cv:.1f}")
                    ax.loglog(
                        common_freq,
                        geomean,
                        color=colors_map[(rH, cv)],
                        linewidth=2.0,
                        linestyle="-",
                        label=label,
                    )

            ax.set_xlabel(to_title_case("Frequency (Hz)"))
            ax.set_ylabel(to_title_case("Transfer Function Magnitude"))
            format_title(f"Vs1={Vs1:.0f} m/s", ax=ax)
            ax.set_xlim(1e-1, 5e1)
            ax.set_ylim(1e-2, 1e2)
            ax.grid(True, alpha=0.3)
            add_subfigure_label(ax, idx)
            if idx == 0:
                place_legend(ax, position="bottom", ncol=2)

        fig.suptitle(
            f"$\\bf{{Velocity-Dependent\\ Sensitivity}}$\n"
            f"$H$={thickness:.0f} m, damping={damping_method}",
            fontsize=14,
            y=1.04,
        )
        plt.tight_layout()
        output_file = (
            output_dir / f"validation_velocity_dependent_sensitivity_damping_{damping_method}.png"
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
    """Find peak amplification at a specific mode."""
    if Vs1 is not None and thickness is not None:
        theoretical_fn = (2 * mode_number - 1) * Vs1 / (4 * thickness)
        freq_range = theoretical_fn * np.array([0.5, 1.5])
        mask = (freq >= freq_range[0]) & (freq <= freq_range[1])
        if np.any(mask):
            return np.max(tf[mask])

    if mode_number == 1:
        mask = freq <= 10.0
        if np.any(mask):
            return np.max(tf[mask])
    elif mode_number == 3:
        mask = (freq >= 5.0) & (freq <= 20.0)
        if np.any(mask):
            return np.max(tf[mask])

    return np.max(tf)


def plot_amplification_vs_CV(
    tf_dict: Dict,
    tf_dict_1D: Dict,
    output_dir: Path,
    thickness: float = 76.0,
):
    """Plot 5: Amplification vs CV summary."""
    apply_style()

    Vs1_values = sorted(set(k[2] for k in tf_dict.keys()))
    CV_values = sorted(set(k[1] for k in tf_dict.keys()))
    rH_values = sorted(set(k[0] for k in tf_dict.keys()))
    damping_methods = sorted(set(k[5] for k in tf_dict.keys()))

    for damping_method in damping_methods:
        n_Vs1 = len(Vs1_values)
        fig, axes = plt.subplots(2, n_Vs1, figsize=(6 * n_Vs1, 12))

        if n_Vs1 == 1:
            axes = axes.reshape(-1, 1)

        subfig_idx = 0
        for col_idx, Vs1 in enumerate(Vs1_values):
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

            ax1 = axes[0, col_idx]
            rh_colors = [COLORBLIND_COLORS[0], COLORBLIND_COLORS[1]]
            for rh_idx, rH in enumerate(rH_values):
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

                ax1.plot(
                    CV_values,
                    amplifications,
                    marker="o",
                    color=rh_colors[rh_idx % len(rh_colors)],
                    linewidth=2.0,
                    label=format_label(f"rH={rH:.0f} m"),
                )

            if uniform_peak_1st is not None:
                ax1.axhline(
                    uniform_peak_1st,
                    color="black",
                    linestyle="--",
                    linewidth=2.0,
                    alpha=0.7,
                    label="Uniform (1D)",
                )

            ax1.set_xlabel(to_title_case(format_label("CV")))
            ax1.set_ylabel(to_title_case("Peak Amplification (1st Mode)"))
            format_title(
                f"Vs1={Vs1:.0f} m/s, $H$={thickness:.0f} m",
                subtitle=f"damping={damping_method}",
                ax=ax1,
            )
            ax1.grid(True, alpha=0.3)
            place_legend(ax1, position="bottom")
            ax1.set_xticks(CV_values)
            add_subfigure_label(ax1, subfig_idx)
            subfig_idx += 1

            ax2 = axes[1, col_idx]
            for rh_idx, rH in enumerate(rH_values):
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

                ax2.plot(
                    CV_values,
                    amplifications,
                    marker="s",
                    color=rh_colors[rh_idx % len(rh_colors)],
                    linewidth=2.0,
                    label=format_label(f"rH={rH:.0f} m"),
                )

            if uniform_peak_3rd is not None:
                ax2.axhline(
                    uniform_peak_3rd,
                    color="black",
                    linestyle="--",
                    linewidth=2.0,
                    alpha=0.7,
                    label="Uniform (1D)",
                )

            ax2.set_xlabel(to_title_case(format_label("CV")))
            ax2.set_ylabel(to_title_case("Peak Amplification (3rd Mode)"))
            format_title(
                f"Vs1={Vs1:.0f} m/s, $H$={thickness:.0f} m",
                subtitle=f"damping={damping_method}",
                ax=ax2,
            )
            ax2.grid(True, alpha=0.3)
            place_legend(ax2, position="bottom")
            ax2.set_xticks(CV_values)
            add_subfigure_label(ax2, subfig_idx)
            subfig_idx += 1

        plt.tight_layout()
        output_file = output_dir / f"validation_amplification_vs_CV_damping_{damping_method}.png"
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
    """Compare damping methods side-by-side in the same plot."""
    apply_style()

    Vs1_values = sorted(set(k[2] for k in tf_dict.keys()))
    damping_methods = sorted(set(k[5] for k in tf_dict.keys()))

    for Vs1 in Vs1_values:
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = COLORBLIND_COLORS[: len(damping_methods)]
        for dm_idx, (damping_method, color) in enumerate(zip(damping_methods, colors)):
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
                    label=format_label(f"rH={rH:.0f}")
                    + ", "
                    + format_label(f"CV={CV:.2f}")
                    + f", {damping_method}",
                )

        ax.set_xlabel(to_title_case("Frequency (Hz)"))
        ax.set_ylabel(to_title_case("Transfer Function Magnitude"))
        format_title(
            "Damping Method Comparison",
            subtitle=f"Vs1={Vs1:.0f} m/s, $H$={thickness:.0f} m, "
            + format_label(f"rH={rH:.0f} m")
            + ", "
            + format_label(f"CV={CV:.2f}"),
            ax=ax,
        )
        ax.set_xlim(1e-1, 5e1)
        ax.set_ylim(1e-2, 1e2)
        ax.grid(True, alpha=0.3)
        place_legend(ax, position="bottom")
        add_subfigure_label(ax, 0)

        plt.tight_layout()
        output_file = output_dir / f"damping_comparison_Vs1_{Vs1:.0f}_rH_{rH:.0f}_CV_{CV:.2f}.png"
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
    """Compare damping methods in separate subplots."""
    apply_style()

    Vs1_values = sorted(set(k[2] for k in tf_dict.keys()))
    damping_methods = sorted(set(k[5] for k in tf_dict.keys()))

    for Vs1 in Vs1_values:
        n_methods = len(damping_methods)
        fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 6))
        if n_methods == 1:
            axes = [axes]

        for idx, damping_method in enumerate(damping_methods):
            ax = axes[idx]

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
                    color=COLORBLIND_COLORS[1],
                    linewidth=2.5,
                    linestyle="--",
                    label=format_label(f"rH={rH:.0f}")
                    + ", "
                    + format_label(f"CV={CV:.2f}")
                    + " (geomean)",
                )

            ax.set_xlabel(to_title_case("Frequency (Hz)"))
            ax.set_ylabel(to_title_case("Transfer Function Magnitude"))
            format_title(damping_method, ax=ax)
            ax.set_xlim(1e-1, 5e1)
            ax.set_ylim(1e-2, 1e2)
            ax.grid(True, alpha=0.3)
            place_legend(ax, position="bottom")
            add_subfigure_label(ax, idx)

        fig.suptitle(
            f"$\\bf{{Damping\\ Method\\ Comparison}}$\n"
            f"Vs1={Vs1:.0f} m/s, $H$={thickness:.0f} m, "
            + format_label(f"rH={rH:.0f} m")
            + ", "
            + format_label(f"CV={CV:.2f}"),
            fontsize=14,
            y=1.04,
        )
        plt.tight_layout()
        output_file = (
            output_dir / f"damping_comparison_subplots_Vs1_{Vs1:.0f}_rH_{rH:.0f}_CV_{CV:.2f}.png"
        )
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {output_file}")
        plt.close()


def merge_timing_data(script_dir: Path) -> Path:
    """Merge all timing_data_task*.csv files from 1D_results/ and results/ directories."""
    merged_file = script_dir / "timing_data_merged.csv"
    timing_files = []

    results_1D_dir = script_dir / "1D_results"
    if results_1D_dir.exists():
        timing_files.extend(results_1D_dir.glob("timing_data_task_*.csv"))

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
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    results_dir = script_dir / "results"
    results_1D_dir = script_dir / "1D_results"
    output_dir = script_dir / "transfer_function_results" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Creating Validation Plots")
    print("=" * 60)

    Vsmin = 1000.0

    print("\nMerging timing data...")
    merge_timing_data(script_dir)

    pickle_file = script_dir / "transfer_function_results" / "transfer_functions_dict.pkl"
    tf_dict = None
    if pickle_file.exists():
        print(f"\nLoading transfer functions from pickle: {pickle_file}")
        tf_dict = load_tf_dict_from_pickle(pickle_file)

    if tf_dict is None:
        print("\nComputing transfer functions from scratch...")
        tf_dict, skipped_folders = load_all_results(results_dir, Vsmin=Vsmin)

        if len(tf_dict) == 0:
            print("Error: No variability results loaded. Exiting.")
            return

        pickle_file.parent.mkdir(parents=True, exist_ok=True)
        import pickle

        with open(pickle_file, "wb") as f:
            pickle.dump(tf_dict, f)
        print(f"Saved transfer functions to pickle: {pickle_file}")

    print("\nLoading 1D uniform results...")
    tf_dict_1D = load_1D_results(results_1D_dir, Vsmin=Vsmin)

    if len(tf_dict_1D) == 0:
        print("Warning: No 1D results loaded. Some plots may be incomplete.")

    print(f"\nLoaded {len(tf_dict)} variability transfer functions")
    print(f"Loaded {len(tf_dict_1D)} 1D uniform transfer functions")

    print("\n" + "=" * 60)
    print("Generating validation plots...")
    print("=" * 60)

    print("\n1. Creating uniform vs theoretical plots...")
    plot_uniform_vs_theoretical(tf_dict, tf_dict_1D, output_dir)

    print("\n2. Creating CV effect plots...")
    print("  Creating CV effect plots for rH=50 m...")
    plot_CV_effect_at_fixed_rH(tf_dict, tf_dict_1D, output_dir, rH=50.0, thickness=50.0)

    print("\n3. Creating rH effect plots...")
    plot_rH_effect_at_fixed_CV(tf_dict, tf_dict_1D, output_dir, CV=0.3, thickness=50.0)

    print("\n4. Creating velocity-dependent sensitivity plot...")
    plot_velocity_dependent_sensitivity(tf_dict, output_dir, thickness=50.0)

    print("\n5. Creating amplification vs CV plot...")
    plot_amplification_vs_CV(tf_dict, tf_dict_1D, output_dir, thickness=50.0)

    print("\n6. Creating damping method comparison plots...")
    print("  Creating side-by-side comparison plots...")
    plot_damping_method_comparison(tf_dict, tf_dict_1D, output_dir, rH=50.0, CV=0.3, thickness=50.0)
    print("  Creating subplot comparison plots...")
    plot_damping_method_subplots(tf_dict, tf_dict_1D, output_dir, rH=50.0, CV=0.3, thickness=50.0)

    print("\n" + "=" * 60)
    print("Validation plots complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
