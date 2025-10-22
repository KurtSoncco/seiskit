"""Pure utility functions for seismic analysis preparation.

This module contains stateless utility functions for material loading,
time series generation, and other data preparation tasks that don't
require OpenSees. These functions are designed to be parallelization-safe.
"""

import math
from typing import Dict, List, Optional, Sequence

import numpy as np


def load_material_properties(
    data: Dict[str, Optional[str]],
) -> Dict[str, np.ndarray]:
    """Load the data from the specified paths and return as numpy arrays.
    
    Args:
        data: Dictionary mapping property names to file paths
        
    Returns:
        Dictionary mapping property names to loaded numpy arrays
    """
    loaded_data = {}
    for key, path in data.items():
        if path is not None:
            loaded_data[key] = np.loadtxt(path, delimiter=",")
    return loaded_data


def compute_ricker(
    freq: float,
    t_shift: float,
    duration: float,
    dt: float,
) -> np.ndarray:
    """Generate a Ricker wavelet time series.
    
    Args:
        freq: Central frequency of the Ricker wavelet
        t_shift: Time shift parameter
        duration: Total duration of the time series
        dt: Time step
        
    Returns:
        Array of Ricker wavelet values
    """
    time_points = np.arange(0.0, duration+dt, dt)
    arg = (math.pi * freq * (time_points - t_shift)) ** 2
    ricker_values = (1.0 - 2.0 * arg) * np.exp(-arg)
    
    
    return ricker_values


def build_mesh_and_materials(
    Lx: float,
    Ly: float,
    hx: float,
    vs_grid: Optional[np.ndarray] = None,
    rho_grid: Optional[np.ndarray] = None,
    nu_grid: Optional[np.ndarray] = None,
    default_props: tuple[float, float, float] = (200.0, 0.3, 2100.0),
) -> tuple[List[int], Dict[tuple[float, float, float], int]]:
    """Create node and element topology and (optionally) materials.

    This function performs no OpenSees calls, making it testable and
    parallelization-safe.

    Args:
        Lx: Domain width
        Ly: Domain height  
        hx: Element size
        vs_grid: Optional shear wave velocity grid
        rho_grid: Optional density grid
        nu_grid: Optional Poisson's ratio grid
        default_props: Default (vs, nu, rho) for homogeneous case

    Returns:
        Tuple of (abs_elements, material_map)
    """
    ndivx = int(Lx / hx) + 2
    ndivy = int(Ly / hx) + 1

    abs_elements: List[int] = []
    material_map: Dict[tuple[float, float, float], int] = {}
    mat_tag_counter = 1

    # If grids are provided, validate shapes
    if vs_grid is not None:
        expect_shape = (int(Ly / hx), int(Lx / hx))
        if vs_grid.shape != expect_shape:
            raise ValueError(
                f"vs_grid shape {vs_grid.shape} != expected {expect_shape}"
            )

    # Iterate and create element stubs (no OpenSees calls)
    for j in range(ndivy):
        Yflag = "B" if j == 0 else ""
        for i in range(ndivx):
            Etag = j * ndivx + i + 1
            is_boundary = (i == 0) or (i == ndivx - 1) or (j == 0)
            if is_boundary:
                abs_elements.append(Etag)
            else:
                # Determine material properties
                if vs_grid is not None:
                    row_idx = int(Ly / hx) - (j - 1) - 1
                    col_idx = i - 1
                    vs = float(vs_grid[row_idx, col_idx])
                    rho = (
                        float(rho_grid[row_idx, col_idx])
                        if rho_grid is not None
                        else default_props[2]
                    )
                    nu = (
                        float(nu_grid[row_idx, col_idx])
                        if nu_grid is not None
                        else default_props[1]
                    )
                else:
                    vs, nu, rho = default_props
                    
                G = rho * vs**2
                E = G * 2.0 * (1.0 + nu)
                mat_props = (E, nu, rho)
                if mat_props not in material_map:
                    material_map[mat_props] = mat_tag_counter
                    mat_tag_counter += 1
                    
    return abs_elements, material_map
