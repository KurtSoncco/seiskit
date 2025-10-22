from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def Base_case_extraction(
    Vs_array: np.ndarray, dz: int = 5
) -> Tuple[float, float, float]:
    """
    Extract Vs1, Vs2, and h from the given Vs array.

    """
    assert dz > 0
    Vs_unique = np.unique(Vs_array)
    Vs1 = Vs_unique[0]
    Vs2 = Vs_unique[1]

    # Extract the height as the number of repetitions of Vs1
    Vs1_count = np.count_nonzero(Vs_array == Vs1)
    h = float(Vs1_count * dz)
    return Vs1, Vs2, h


def variability_realization(
    Vs_array,
    Lx,
    Lz,
    dx,
    dz,
    rH,
    aHV,
    CV,
    seed_intra: Optional[int] = None,
    seed_inter: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generate a 2D spatial variability realization of a Vs array in 1D given the parameters.

    """
    rng_intra = np.random.default_rng(seed_intra)
    rng_inter = np.random.default_rng(seed_inter)

    # Extract values
    Vs1, Vs2, h = Base_case_extraction(Vs_array)

    # COmpute the number of grid points in x and z directions
    nx = int(Lx / dx)
    nz = int(Lz / dz)

    # Define the grid
    x = np.linspace(0, nx * dx, nx) + dx / 2
    z = np.linspace(0, nz * dz, nz) + dz / 2

    ## Intralayer
    # Create a meshgrid and flatten it
    X, Z = np.meshgrid(x, z)
    X = X.flatten()
    Z = Z.flatten()

    # Vectorized computation of pairwise differences
    tau_x = np.abs(X[:, None] - X[None, :])
    tau_z = np.abs(Z[:, None] - Z[None, :])

    # Compute rho in one go
    rho = np.exp(-2 * (tau_x / rH + tau_z / (rH / aHV)))

    # Compute Cholesky decomposition and generate Gaussian random field
    L = np.linalg.cholesky(rho)
    G = L @ rng_intra.normal(0, 1, len(X))

    # Lognormal transformation
    psi = np.sqrt(np.log(1 + CV**2))
    lamb = np.log(Vs1) - 0.5 * psi**2

    # Compute the random field
    Z_Vs = np.exp(lamb + psi * G)

    ## Truncated the values
    z_max = np.exp(lamb + psi * 2)
    z_min = np.exp(lamb - psi * 2)
    Z_Vs = np.clip(Z_Vs, z_min, z_max)

    # Reshape the random field
    Z_Vs = Z_Vs.reshape(len(z), len(x))

    ## Interlayer variability
    y_index = np.arange(Z_Vs.shape[0])[:, np.newaxis] * dx

    # Interface model
    freq1 = rng_inter.uniform(1 / 60, 1 / 20)
    freq2 = rng_inter.uniform(1 / 20, 1 / 10)
    freq3 = rng_inter.uniform(1 / 10, 1 / 5)
    offset = rng_inter.uniform(0, 2 * dz)
    y_add = (
        np.sin(2 * np.pi * freq1 * x + offset)
        + np.sin(2 * np.pi * freq2 * x + offset)
        + np.sin(2 * np.pi * freq3 * x + offset)
    )

    # Normalize the signal to have a maximum amplitude of 1
    y_add_normalized = y_add / np.max(np.abs(y_add))
    y_prior = y_add_normalized * dz + h

    # Broadcast and compute the mask: True if row index is less than the y_prior
    mask = y_index < y_prior[np.newaxis, :]

    # Recalculate with the diff the last values
    mask_pos = np.floor(y_prior / dz).astype(int)
    diff = mask_pos * dz + dz / 2 - y_prior

    # Mapping function
    def mapping_func(x):
        return dz + x if x < 0 else x

    mapping = np.vectorize(mapping_func)

    Vs_A = Z_Vs[mask_pos, np.arange(len(x))]
    result_avg = 5 / (mapping(diff) / Vs_A + (dz - mapping(diff)) / Vs2)

    # Add this to Z_Vs
    Z_Vs[mask_pos, np.arange(len(x))] = result_avg

    # Now mask Z_Vs: assign a fill value (e.g. Vs2) outside the allowed region.
    Z_Vs_masked = np.where(mask, Z_Vs, Vs2)
    X = X.reshape(len(z), len(x))
    Z = Z.reshape(len(z), len(x))

    return Z_Vs_masked, x, z, h


if __name__ == "__main__":
    # Example usage
    Vs_profile_1D = np.array([180.0] * 8 + [1300.0] * 1)
    Lx, Lz = 500.0, 50.0
    dx, dz = 2.5, 2.5
    rH, aHV, CV = 10.0, 2.0, 0.3

    Vs_realization, x_coords, z_coords, h = variability_realization(
        Vs_profile_1D, Lx, Lz, dx, dz, rH, aHV, CV, seed_inter=42, seed_intra=42
    )

    # --- NEW: Set color limits based on soil layer ONLY ---
    Vs1, Vs2 = np.unique(Vs_profile_1D)[0], np.unique(Vs_profile_1D)[1]

    # Isolate the Vs values of the soil layer
    soil_vs_values = Vs_realization[Vs_realization < Vs2]

    # Determine the min and max for the color bar
    vmin = soil_vs_values.min()
    vmax = soil_vs_values.max()

    # Get a colormap and set a specific color for values > vmax (the bedrock)

    cmap = plt.colormaps.get_cmap("viridis_r").copy()
    cmap.set_over("gray")  # Bedrock will be colored gray

    # --- UPDATED: Plotting the result ---
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
    plt.show()
