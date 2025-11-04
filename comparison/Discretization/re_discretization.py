# In this file, we take the 2x2 grid and discretize it into 1x1 cells by expanding (not average)

import os
import time
from typing import Optional

import numpy as np

from seiskit.gaussian_field import (
    _generate_vs_variability_field,
    plot_realization,
)


def re_discretization(
    Vs_array: np.ndarray,
    dx_old: float,
    dz_old: float,
    dz_new: float,
    dx_new: Optional[float] = None,
) -> np.ndarray:
    """
    Re-discretize a 2D array of Vs values from old discretization (dx_old, dz_old)
    to finer discretization (dx_new, dz_new) by expanding each element.

    For example, converts 2x2 elements to 1x1 elements by expanding each cell
    into 4 smaller cells (2x2 expansion when going from 2m to 1m spacing).

    Notes
    -----
    - Both dimensions are expanded: rows (z-axis) and columns (x-axis).
    - Requires that dx_old/dx_new and dz_old/dz_new are positive integers.
    - The scale factors can be different for x and z dimensions.
    - If dx_new is not provided, it defaults to dz_new.

    Parameters
    ----------
    Vs_array : np.ndarray
        Array of shape (nz_old, nx_old) with Vs values at the old discretization.
    dx_old : float
        Old horizontal discretization size (> 0).
    dz_old : float
        Old vertical discretization size (> 0).
    dz_new : float
        New vertical discretization size (> 0).
    dx_new : float, optional
        New horizontal discretization size (> 0). If None, uses dz_new.

    Returns
    -------
    np.ndarray
        Array of shape (nz_new, nx_new) where:
        - nz_new = nz_old * (dz_old / dz_new)
        - nx_new = nx_old * (dx_old / dx_new)
    """
    if dx_new is None:
        dx_new = dz_new

    if dx_old <= 0 or dx_new <= 0 or dz_old <= 0 or dz_new <= 0:
        raise ValueError("All discretization sizes must be positive.")

    # Calculate scale factors for both dimensions
    scale_x = dx_old / dx_new
    scale_z = dz_old / dz_new

    scale_x_rounded = int(round(scale_x))
    scale_z_rounded = int(round(scale_z))

    if not np.isclose(scale_x, scale_x_rounded):
        raise ValueError(f"dx_old/dx_new must be an integer (got {scale_x:.6f}).")
    if not np.isclose(scale_z, scale_z_rounded):
        raise ValueError(f"dz_old/dz_new must be an integer (got {scale_z:.6f}).")
    if scale_x_rounded < 1:
        raise ValueError(
            "dx_new must be less than or equal to dx_old (refinement only)."
        )
    if scale_z_rounded < 1:
        raise ValueError(
            "dz_new must be less than or equal to dz_old (refinement only)."
        )

    # Early return if no scaling needed
    if scale_x_rounded == 1 and scale_z_rounded == 1:
        return Vs_array.copy()

    # Expand along both axes: first rows (axis=0), then columns (axis=1)
    result = np.repeat(Vs_array, repeats=scale_z_rounded, axis=0)
    result = np.repeat(result, repeats=scale_x_rounded, axis=1)

    return result


if __name__ == "__main__":
    # Index
    index = 1

    # Base case parameters
    Vs_profile_1D = np.array([180.0] * 8 + [1300.0] * 1)
    Lz = 50.0
    dx, dz = 2, 2
    aHV = 1.0  # Fixed horizontal-to-vertical aspect ratio
    interlayer_seed = 42  # Fixed seed for interlayer (wavy boundary) variability

    # Parameter variations
    rH_values = [10.0, 30.0, 50.0]
    CV_values = [0.1, 0.2, 0.3]
    seed_values = [10, 20, 30, 40, 50]  # 5 different seeds for spatial field

    # Fixed spatial dimensions
    Lx_variability = 10.0
    BC_width = 500.0
    Lx = Lx_variability + 2 * BC_width  # 1500m total

    total_combinations = len(rH_values) * len(CV_values) * len(seed_values)

    if index < 0 or index >= total_combinations:
        raise IndexError(
            f"Index {index} is out of range for {total_combinations} tasks "
            f"(valid 0..{total_combinations - 1})."
        )

    # Map index to parameter combination
    # index = rH_idx × (3×5) + CV_idx × 5 + seed_idx
    rH_idx = index // (len(CV_values) * len(seed_values))
    remaining = index % (len(CV_values) * len(seed_values))
    CV_idx = remaining // len(seed_values)
    seed_idx = remaining % len(seed_values)

    rH = rH_values[rH_idx]
    CV = CV_values[CV_idx]
    seed = seed_values[seed_idx]

    task_id = f"rH{rH:.0f}_CV{CV}_s{seed}"
    output_dir = f"results/rH_{rH:.0f}/CV_{CV}/{task_id}"
    # Create directories with retry logic for file system contention
    max_retries = 5
    for attempt in range(max_retries):
        try:
            os.makedirs(output_dir, exist_ok=True)
            break
        except (OSError, IOError):
            if attempt == max_retries - 1:
                raise
            time.sleep(
                0.1 * (attempt + 1)
            )  # Exponential backoff: 0.1s, 0.2s, 0.3s, 0.4s

    print(f"[run_array_index] Starting task {task_id} (index={index})")
    print(f"  rH = {rH} m, CV = {CV}, seed = {seed}")
    print(
        f"  Lx_variability = {Lx_variability} m, BC_width = {BC_width} m, Total Lx = {Lx} m"
    )

    # Generate VS field with the specified parameters
    print(f"[run_array_index] Generating VS field with seed={seed}")
    print(
        f"[run_array_index] Using interlayer_seed={interlayer_seed} for wavy boundary"
    )
    np.random.seed(seed)
    Vs_realization, x_coords, z_coords, h_mean = _generate_vs_variability_field(
        Vs_profile_1D,
        Lx_variability,
        Lz,
        dx,
        dz,
        rH,
        aHV,
        CV,
        seed=seed,
        interlayer_seed=interlayer_seed,
    )

    plot_realization(
        Vs_1D_profile=Vs_profile_1D,
        Vs_realization=Vs_realization,
        Lx=Lx_variability,
        Lz=Lz,
        dx=dx,
        dz=dz,
        save_path=None,
    )

    # Let's do the re-discretization (from 2x2 elements to 1x1 elements)
    dz_new = dz / 2
    print(f"dz_new: {dz_new} m")
    # Call without dx_new; it will default to dz_new inside the function
    Vs_re_discretized = re_discretization(Vs_realization, dx, dz, dz_new)
    # Plot the re-discretized realization
    print(f"Shape of Vs_re_discretized: {Vs_re_discretized.shape}")
    plot_realization(
        Vs_1D_profile=Vs_profile_1D,
        Vs_realization=Vs_re_discretized,
        Lx=Lx_variability,
        Lz=Lz,
        dx=dz_new,
        dz=dz_new,
        save_path=None,
    )

    print(f"Shape of Vs_realization: {Vs_realization.shape}")
    print(f"Shape of Vs_re_discretized: {Vs_re_discretized.shape}")
