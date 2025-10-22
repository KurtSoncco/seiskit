from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# A modern, high-performance random number generator is recommended.
rng = np.random.default_rng(seed=42)


def generate_gaussian_field_fft(
    nx: int,
    nz: int,
    dx: float,
    dz: float,
    rH: float,
    aHV: float,
    rng: np.random.Generator = rng,
) -> np.ndarray:
    """
    Generates a 2D Gaussian random field using the FFT-based spectral method.
    This is significantly faster and more memory-efficient than direct simulation.

    Args:
        nx: Number of points in the x-direction.
        nz: Number of points in the z-direction.
        dx: Grid spacing in the x-direction.
        dz: Grid spacing in the z-direction.
        rH: Horizontal correlation length.
        aHV: Anisotropy ratio (rH / rV).

    Returns:
        A 2D numpy array representing the normalized Gaussian random field.
    """
    # Define spatial frequency grid
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    kz = 2 * np.pi * np.fft.fftfreq(nz, d=dz)
    Kx, Kz = np.meshgrid(kx, kz)

    # Vertical correlation length
    rV = rH / aHV

    # Power Spectral Density (PSD) for an exponential correlation function
    # This is the Fourier transform of the correlation function
    PSD = (2 * np.pi * rH * rV) / (1 + (rH * Kx) ** 2 + (rV * Kz) ** 2) ** 1.5

    # Generate white noise in the frequency domain
    noise_freq = rng.standard_normal((nz, nx)) + 1j * rng.standard_normal((nz, nx))

    # Color the noise with the PSD and transform back to spatial domain
    # The scaling factor is necessary to preserve variance
    field_freq = np.sqrt(PSD / (dx * dz)) * noise_freq
    field = np.fft.ifft2(field_freq).real

    # Normalize the field to have zero mean and unit variance
    field = (field - np.mean(field)) / np.std(field)

    return field


def _generate_vs_variability_field(
    Vs_profile: np.ndarray,
    Lx_variability: float,
    Lz: float,
    dx: float,
    dz: float,
    rH: float,
    aHV: float,
    CV: float,
    seed: Optional[int] = 42,
    dz_1D: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generates a 2D Vs variability field for the top layer using a lognormal
    transformation of a Gaussian random field.

    Args:
        Lx_variability: Horizontal length of the spatially variable region [m].
        Lz: Vertical depth of the domain [m].
        dx: Grid spacing in the horizontal direction [m].
        dz: Grid spacing in the vertical direction [m].
        rH: Correlation length (horizontal) for intralayer variability [m].
        aHV: Anisotropy ratio (rH / rV) for intralayer variability.
        CV: Coefficient of variation for the top layer's Vs.
        Vs1: Shear wave velocity of the top layer [m/s].
        rng: Random number generator for reproducibility.
    Returns:
        A tuple containing:
        - var_Vs_field (np.ndarray): The 2D Vs variability field of shape (nz, nx_var).
        - x_var (np.ndarray): The horizontal coordinates array for the variable region.
        - z (np.ndarray): The vertical coordinates array.
        - h (float): The average depth of the first layer interface [m].
    """

    # 0. Initialize random number generator for reproducibility
    rng = np.random.default_rng(seed)

    # 1. Profile and Grid Setup
    Vs_unique = np.unique(Vs_profile)
    if len(Vs_unique) < 2:
        raise ValueError("Vs_profile must contain at least two distinct layers.")
    Vs1, Vs2 = Vs_unique[0], Vs_unique[1]

    # Calculate the average height of the first layer from the grid spacing
    h = np.count_nonzero(Vs_profile == Vs1) * dz_1D

    # Define grid for the spatially variable region
    nx_var = int(Lx_variability / dx)
    nz = int(Lz / dz)
    x_var = np.linspace(0, Lx_variability - dx, nx_var) + dx / 2
    z = np.linspace(0, Lz - dz, nz) + dz / 2

    # 2. Intralayer Variability (Top Layer)
    # Generate the underlying correlated Gaussian random field
    gaussian_field = generate_gaussian_field_fft(nx_var, nz, dx, dz, rH, aHV, rng)

    # Apply lognormal transformation for a more realistic Vs distribution
    log_std = np.sqrt(np.log(1 + CV**2))
    log_mean = np.log(Vs1) - 0.5 * log_std**2
    Vs_field = np.exp(log_mean + log_std * gaussian_field)

    # Clip values to prevent extreme outliers (e.g., within 2 std devs)
    v_min, v_max = np.exp(log_mean - 2 * log_std), np.exp(log_mean + 2 * log_std)
    Vs_field = np.clip(Vs_field, v_min, v_max)

    # 3. Interlayer Variability (Wavy Boundary)
    # Generate a random wavy interface using a sum of sinusoids
    freq1 = rng.uniform(1 / 80, 1 / 40)
    freq2 = rng.uniform(1 / 40, 1 / 20)
    freq3 = rng.uniform(1 / 20, 1 / 10)
    phase_offset = rng.uniform(0, 2 * np.pi)
    wave_amplitude = (dz / 2) * (
        np.sin(2 * np.pi * freq1 * x_var + phase_offset)
        + np.sin(2 * np.pi * freq2 * x_var + phase_offset)
        + np.sin(2 * np.pi * freq3 * x_var + phase_offset)
    )
    interface_depth = h + wave_amplitude

    # 4. Combine Layers and Handle Boundary Cells
    # Find the row index of the cell containing the interface for each column
    interface_row_idx = np.floor(interface_depth / dz).astype(int)
    interface_row_idx = np.clip(interface_row_idx, 0, nz - 1)

    # For cells intersected by the boundary, calculate a harmonic average to get
    # an effective velocity for waves propagating vertically.
    h1_in_cell = interface_depth - (interface_row_idx * dz)
    h2_in_cell = dz - h1_in_cell

    cols = np.arange(nx_var)
    vs1_in_cell = Vs_field[interface_row_idx, cols]

    # Harmonic average formula: V_avg = H_total / sum(h_i / V_i)
    harmonic_avg = dz / (h1_in_cell / vs1_in_cell + h2_in_cell / Vs2)
    Vs_field[interface_row_idx, cols] = harmonic_avg

    # Use broadcasting to create a mask identifying the bottom layer
    mask_bottom_layer = z[:, np.newaxis] > interface_depth[np.newaxis, :]
    var_Vs_field = np.where(mask_bottom_layer, Vs2, Vs_field)

    return var_Vs_field, x_var, z, h


def _extend_profile(
    Vs_variability: np.ndarray,
    Lx: float,
    dx: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extends a Vs variability field to the full domain width by padding with
    repeated edge values.
    Args:
        Vs_variability: 2D array of Vs values for the variable region.
        Lx: Total horizontal length of the final domain [m].
        dx: Grid spacing in the horizontal direction [m].
    Returns:
        A tuple containing:
        - final_Vs (np.ndarray): The final 2D Vs field of shape (nz, nx_total).
        - x_total (np.ndarray): The final horizontal coordinates array.
    """
    # Read the dimensions of the variability field
    nz, nx_var = Vs_variability.shape
    x_var = np.linspace(0, Vs_variability.shape[1] * dx - dx, nx_var) + dx / 2

    # 5. Extend Profile to Full Domain Width (Lx)
    nx_total = int(Lx / dx)
    n_pad = nx_total - nx_var

    if n_pad > 0:
        # Pad by repeating the first and last columns of the variable field
        n_pad_left = n_pad // 2
        n_pad_right = n_pad - n_pad_left

        left_col = Vs_variability[:, 0:1]  # Keep as 2D array
        right_col = Vs_variability[:, -1:]  # Keep as 2D array

        left_padding = np.tile(left_col, (1, n_pad_left))
        right_padding = np.tile(right_col, (1, n_pad_right))

        final_Vs = np.hstack([left_padding, Vs_variability, right_padding])
        x_total = np.linspace(0, Lx - dx, nx_total) + dx / 2
    else:
        # If Lx_variability is the same as Lx, no padding is needed
        final_Vs = Vs_variability
        x_total = x_var

    return final_Vs, x_total


def create_vs_realization(
    Vs_profile: np.ndarray,
    Lx: float,
    Lx_variability: float,
    Lz: float,
    dx: float,
    dz: float,
    rH: float,
    aHV: float,
    CV: float,
    seed: Optional[int] = 42,
    dz_1D: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generates a 2D spatial variability realization of a Vs profile.

    This function creates a two-layer soil profile with a spatially variable
    interface and intralayer variability in the top layer. The top layer's Vs values
    are generated using a lognormal transformation of a Gaussian random field,
    while the bottom layer has a constant Vs. The interface between the two layers
    is modeled as a wavy boundary using a sum of sinusoids. The final Vs field is extended to the full domain width
    by padding with repeated edge values.

    Args:
        Vs_profile: 1D array of shear wave velocities defining the initial layers.
        Lx: Total horizontal length of the final domain [m].
        Lx_variability: Horizontal length of the spatially variable region [m].
        Lz: Total vertical depth of the domain [m].
        dx: Grid spacing in the horizontal direction [m].
        dz: Grid spacing in the vertical direction [m].
        rH: Correlation length (horizontal) for intralayer variability [m].
        aHV: Anisotropy ratio (rH / rV) for intralayer variability.
        CV: Coefficient of variation for the top layer's Vs.
        seed: Random seed for reproducibility.

    Returns:
        A tuple containing:
        - final_Vs (np.ndarray): The final 2D Vs field of shape (nz, nx_total).
        - x_total (np.ndarray): The final horizontal coordinates array.
        - z (np.ndarray): The vertical coordinates array.
        - h (float): The average depth of the first layer interface [m].
    """
    if Lx < Lx_variability:
        raise ValueError(
            "Total length Lx must be greater than or equal to Lx_variability."
        )

    # 0. Initialize random number generator for reproducibility
    rng = np.random.default_rng(seed)

    # 1. Profile and Grid Setup
    Vs_unique = np.unique(Vs_profile)
    if len(Vs_unique) < 2:
        raise ValueError("Vs_profile must contain at least two distinct layers.")
    Vs1, Vs2 = Vs_unique[0], Vs_unique[1]

    # Calculate the average height of the first layer from the grid spacing
    h = np.count_nonzero(Vs_profile == Vs1) * dz_1D

    # Define grid for the spatially variable region
    nx_var = int(Lx_variability / dx)
    nz = int(Lz / dz)
    x_var = np.linspace(0, Lx_variability - dx, nx_var) + dx / 2
    z = np.linspace(0, Lz - dz, nz) + dz / 2

    # 2. Intralayer Variability (Top Layer)
    # Generate the underlying correlated Gaussian random field
    gaussian_field = generate_gaussian_field_fft(nx_var, nz, dx, dz, rH, aHV, rng)

    # Apply lognormal transformation for a more realistic Vs distribution
    log_std = np.sqrt(np.log(1 + CV**2))
    log_mean = np.log(Vs1) - 0.5 * log_std**2
    Vs_field = np.exp(log_mean + log_std * gaussian_field)

    # Clip values to prevent extreme outliers (e.g., within 2 std devs)
    v_min, v_max = np.exp(log_mean - 2 * log_std), np.exp(log_mean + 2 * log_std)
    Vs_field = np.clip(Vs_field, v_min, v_max)

    # 3. Interlayer Variability (Wavy Boundary)
    # Generate a random wavy interface using a sum of sinusoids
    freq1 = rng.uniform(1 / 80, 1 / 40)
    freq2 = rng.uniform(1 / 40, 1 / 20)
    freq3 = rng.uniform(1 / 20, 1 / 10)
    phase_offset = rng.uniform(0, 2 * np.pi)
    wave_amplitude = (dz / 2) * (
        np.sin(2 * np.pi * freq1 * x_var + phase_offset)
        + np.sin(2 * np.pi * freq2 * x_var + phase_offset)
        + np.sin(2 * np.pi * freq3 * x_var + phase_offset)
    )
    interface_depth = h + wave_amplitude

    # 4. Combine Layers and Handle Boundary Cells
    # Find the row index of the cell containing the interface for each column
    interface_row_idx = np.floor(interface_depth / dz).astype(int)
    interface_row_idx = np.clip(interface_row_idx, 0, nz - 1)

    # For cells intersected by the boundary, calculate a harmonic average to get
    # an effective velocity for waves propagating vertically.
    h1_in_cell = interface_depth - (interface_row_idx * dz)
    h2_in_cell = dz - h1_in_cell

    cols = np.arange(nx_var)
    vs1_in_cell = Vs_field[interface_row_idx, cols]

    # Harmonic average formula: V_avg = H_total / sum(h_i / V_i)
    harmonic_avg = dz / (h1_in_cell / vs1_in_cell + h2_in_cell / Vs2)
    Vs_field[interface_row_idx, cols] = harmonic_avg

    # Use broadcasting to create a mask identifying the bottom layer
    mask_bottom_layer = z[:, np.newaxis] > interface_depth[np.newaxis, :]
    var_Vs_field = np.where(mask_bottom_layer, Vs2, Vs_field)

    # 5. Extend Profile to Full Domain Width (Lx)
    nx_total = int(Lx / dx)
    n_pad = nx_total - nx_var

    if n_pad > 0:
        # Pad by repeating the first and last columns of the variable field
        n_pad_left = n_pad // 2
        n_pad_right = n_pad - n_pad_left

        left_col = var_Vs_field[:, 0:1]  # Keep as 2D array
        right_col = var_Vs_field[:, -1:]  # Keep as 2D array

        left_padding = np.tile(left_col, (1, n_pad_left))
        right_padding = np.tile(right_col, (1, n_pad_right))

        final_Vs = np.hstack([left_padding, var_Vs_field, right_padding])
        x_total = np.linspace(0, Lx - dx, nx_total) + dx / 2
    else:
        # If Lx_variability is the same as Lx, no padding is needed
        final_Vs = var_Vs_field
        x_total = x_var

    return final_Vs, x_total, z, h


def plot_realization(
    Vs_1D_profile: np.ndarray,
    Vs_realization: np.ndarray,
    Lx: float,
    Lz: float,
    dx: float,
    dz: float,
    save_path: Optional[str] = None,
):
    """
    Plots the Vs realization with a color scale focused on the soil layer.
    Bedrock values are colored distinctly to highlight the soil variability.
    """
    # Determine Vs1 and Vs2 from the realization
    Vs_unique = np.unique(Vs_1D_profile)
    _, Vs2 = Vs_unique[0], Vs_unique[1]

    # Isolate the Vs values of the soil layer
    soil_vs_values = Vs_realization[Vs_realization < Vs2]

    # Determine the min and max for the color bar
    vmin = soil_vs_values.min()
    vmax = soil_vs_values.max()

    # Get a colormap and set a specific color for values > vmax (the bedrock)
    cmap = plt.colormaps.get_cmap("viridis_r").copy()
    cmap.set_over("gray")  # Bedrock will be colored gray

    # Plotting the result
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 5))

    im = ax.imshow(
        Vs_realization,
        extent=(0, Lx, Lz, 0),
        aspect="auto",
        cmap=cmap,  # Use the modified colormap
        interpolation="nearest",
        vmin=vmin,  # Set the minimum for the color scale
        vmax=vmax,  # Set the maximum for the color scale
    )

    # Add 'extend' to the colorbar to show there are values beyond its max
    cbar = fig.colorbar(im, ax=ax, extend="max")
    cbar.set_label("Soil $V_s$ (m/s)", fontsize=12)

    ax.set_xlabel("Distance (m)", fontsize=12)
    ax.set_ylabel("Depth (m)", fontsize=12)
    ax.set_title(
        "Optimized 2D $V_s$ Realization (Soil-Focused Color Scale)", fontsize=14
    )
    ax.grid(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    # Example usage
    Vs_profile_1D = np.array([180.0] * 8 + [1300.0] * 1)
    Lx, Lx_variability, Lz = 500.0, 300.0, 50.0
    dx, dz = 2.5, 2.5
    rH, aHV, CV = 10.0, 2.0, 0.3

    # --- Generation (same as before) ---
    print("Generating Vs realization...")
    Vs_realization, x_coords, z_coords, h_mean = create_vs_realization(
        Vs_profile=Vs_profile_1D,
        Lx=Lx,
        Lx_variability=Lx_variability,
        Lz=Lz,
        dx=dx,
        dz=dz,
        rH=rH,
        aHV=aHV,
        CV=CV,
    )
    print("Generation complete.")

    # --- Plotting with optimized color scale ---
    plot_realization(Vs_profile_1D, Vs_realization, Lx, Lz, dx, dz)

    ## Do two generation for Lx_var = 500 m and Lx_var = 100 m
    ## The Lx_var =100 is the central 100 m of the Lx_var = 500 m case

    Vs_profile_1D = np.array([180.0] * 8 + [1300.0] * 1)
    Lz = 50.0
    rH, aHV, CV = 10.0, 2.0, 0.3
    dx, dz = 2.5, 2.5
    seed = 42

    Lx_variability_values = [500.0, 100.0]
    Lx = 1000.0  # Total length for the larger variability case

    # Generate the larger variability case first
    print(
        f"Generating Vs realization for Lx_variability = {np.max(Lx_variability_values)} m..."
    )
    Vs_total_realization, x_coords, z_coords, h_mean = _generate_vs_variability_field(
        Vs_profile_1D,
        np.max(Lx_variability_values),
        Lz,
        dx,
        dz,
        rH,
        aHV,
        CV,
        seed=seed,
    )

    for Lx_variability in Lx_variability_values:
        print(f"Generating Vs realization for Lx_variability = {Lx_variability} m...")
        # Compute the central region given the desired Lx_variability
        center_x = np.max(Lx_variability_values) / 2.0
        half_Lx_var = Lx_variability / 2.0
        x_start = center_x - half_Lx_var
        x_end = center_x + half_Lx_var
        start_idx = int(x_start / dx)
        end_idx = int(x_end / dx)
        Vs_realization = Vs_total_realization[:, start_idx:end_idx]
        print(f"Generation complete for Lx_variability = {Lx_variability} m.\n")

        ## Extend to full Lx
        final_Vs, x_total = _extend_profile(Vs_realization, Lx, dx)

        plot_realization(
            Vs_profile_1D,
            final_Vs,
            Lx,
            Lz,
            dx,
            dz,
            save_path=f"Vs_realization_Lxvar_{int(Lx_variability)}.png",
        )
