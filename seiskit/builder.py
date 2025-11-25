"""Model builder for OpenSees 2D site response analysis.

This module provides functions to build model data structures (nodes, elements, materials)
without making OpenSees calls, making the code testable and parallelization-safe.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from seiskit.config import AnalysisConfig


@dataclass
class NodeData:
    """Data structure for a node in the model.

    Attributes:
        tag: Node tag (unique identifier)
        x: X coordinate
        y: Y coordinate
    """

    tag: int
    x: float
    y: float


@dataclass
class SoilElementData:
    """Data structure for an interior soil element.

    Attributes:
        tag: Element tag (unique identifier)
        nodes: Tuple of node tags (4 for 4-node, 8 for 8-node elements)
        mat_tag: Material tag reference
        gravity_load: Gravity load value
        vs_value: Shear wave velocity for this element (m/s)
    """

    tag: int
    nodes: Tuple[int, ...]  # 4 nodes for 4-node, 8 nodes for 8-node
    mat_tag: int
    gravity_load: float
    vs_value: float


@dataclass
class BoundaryElementData:
    """Data structure for a boundary/absorbing element.

    Attributes:
        tag: Element tag (unique identifier)
        nodes: Tuple of 4 node tags
        btype: Boundary type string (e.g., "L", "R", "B", "LB", "RB")
        G: Shear modulus
        poiss: Poisson's ratio
        rho: Density
    """

    tag: int
    nodes: Tuple[int, int, int, int]
    btype: str
    G: float
    poiss: float
    rho: float


@dataclass
class ModelData:
    """A structured representation of the entire model geometry and materials."""

    nodes: List[NodeData] = field(default_factory=list)
    soil_elements: List[SoilElementData] = field(default_factory=list)
    boundary_elements: List[BoundaryElementData] = field(default_factory=list)
    material_map: Dict[Tuple[float, float, float], int] = field(default_factory=dict)
    abs_element_tags: List[int] = field(default_factory=list)


def build_model_data(
    config: AnalysisConfig,
    vs_data: np.ndarray,
    rho_data: np.ndarray,
    nu_data: np.ndarray,
) -> ModelData:
    """
    Builds a structured representation of the model without making OpenSees calls.
    This function iterates through the mesh ONCE to generate all required data.
    """
    model = ModelData()
    mat_tag_counter = 1

    ndivx_total = int(config.Lx / config.hx) + 2
    ndivy_total = int(config.Ly / config.hy) + 1
    ndivx_soil = int(config.Lx / config.hx)
    ndivy_soil = int(config.Ly / config.hy)

    if vs_data.shape != (ndivy_soil, ndivx_soil):
        raise ValueError(
            f"Shape of Vs file {vs_data.shape} != expected ({ndivy_soil}, {ndivx_soil})"
        )

    # 1. Generate Node Data
    # For "8node" case, we use bbarQuad (B-bar formulation) which is higher-order
    # but still uses 4 nodes - no need for mid-nodes
    use_8node = config.element_type == "8node"

    abs_h = config.hx * 2.0

    # Generate corner nodes using vectorized operations
    # Create meshgrid for all node indices
    j_indices = np.arange(ndivy_total + 1)
    i_indices = np.arange(ndivx_total + 1)

    # Vectorize y coordinates
    y_coords = np.where(j_indices == 0, -abs_h, (j_indices - 1) * config.hy)

    # Vectorize x coordinates
    x_coords = np.where(
        i_indices == 0,
        -abs_h,
        np.where(
            i_indices == ndivx_total, config.Lx + abs_h, (i_indices - 1) * config.hx
        ),
    )

    # Create meshgrid for all node positions using 'ij' indexing for consistency
    # With 'ij' indexing: first arg varies along j (rows), second arg varies along i (columns)
    # We want: X to vary with i (columns), Y to vary with j (rows)
    # So: Y_grid first (varies with j), X_grid second (varies with i)
    Y_grid, X_grid = np.meshgrid(y_coords, x_coords - config.Lx / 2.0, indexing="ij")
    J_grid, I_grid = np.meshgrid(j_indices, i_indices, indexing="ij")

    # Flatten all arrays (row-major order: all columns for row 0, then row 1, etc.)
    node_tags = (J_grid * (ndivx_total + 1) + I_grid + 1).flatten()
    x_flat = X_grid.flatten()
    y_flat = Y_grid.flatten()

    # Create NodeData objects
    model.nodes = [
        NodeData(tag=int(tag), x=float(x), y=float(y))
        for tag, x, y in zip(node_tags, x_flat, y_flat)
    ]

    # Generate mid-nodes for 8-node elements (disabled - not supported in OpenSees 2D)
    if False and use_8node:
        # Horizontal mid-nodes (between corners horizontally)
        for j in range(ndivy_total + 1):
            y = -abs_h if j == 0 else (j - 1) * config.hy
            for i in range(ndivx_total):
                x1 = (
                    -abs_h
                    if i == 0
                    else (
                        config.Lx + abs_h
                        if i == ndivx_total - 1
                        else (i - 1) * config.hx
                    )
                )
                x2 = (
                    -abs_h
                    if i + 1 == 0
                    else (config.Lx + abs_h if i + 1 == ndivx_total else i * config.hx)
                )
                x_mid = (x1 + x2) / 2.0
                node_tag = (
                    (ndivy_total + 1) * (ndivx_total + 1) + j * ndivx_total + i + 1
                )
                model.nodes.append(
                    NodeData(tag=node_tag, x=x_mid - config.Lx / 2.0, y=y)
                )

        # Vertical mid-nodes (between corners vertically)
        for j in range(ndivy_total):
            for i in range(ndivx_total + 1):
                x = (
                    -abs_h
                    if i == 0
                    else (
                        config.Lx + abs_h if i == ndivx_total else (i - 1) * config.hx
                    )
                )
                y1 = -abs_h if j == 0 else (j - 1) * config.hy
                y2 = -abs_h if j + 1 == 0 else j * config.hy
                y_mid = (y1 + y2) / 2.0
                base_offset = (ndivy_total + 1) * (ndivx_total + 1) + (
                    ndivy_total + 1
                ) * ndivx_total
                node_tag = base_offset + j * (ndivx_total + 1) + i + 1
                model.nodes.append(
                    NodeData(tag=node_tag, x=x - config.Lx / 2.0, y=y_mid)
                )

    # 2. Generate Element and Material Data using vectorized operations
    # Create meshgrid for all element indices
    j_elem = np.arange(ndivy_total)
    i_elem = np.arange(ndivx_total)
    J_elem, I_elem = np.meshgrid(j_elem, i_elem, indexing="ij")

    # Flatten element indices
    j_flat = J_elem.flatten()
    i_flat = I_elem.flatten()
    n_elements = len(j_flat)

    # Compute element tags
    elem_tags = j_flat * ndivx_total + i_flat + 1

    # Compute node tags for all elements (vectorized)
    N1 = j_flat * (ndivx_total + 1) + i_flat + 1
    N2 = j_flat * (ndivx_total + 1) + i_flat + 2
    N4 = (j_flat + 1) * (ndivx_total + 1) + i_flat + 1
    N3 = (j_flat + 1) * (ndivx_total + 1) + i_flat + 2

    # Determine boundary flags (vectorized)
    # Note: Top surface (y=Ly) is a FREE SURFACE - no absorbing boundary elements
    # Only bottom, left, and right boundaries use ASDAbsorbingBoundary2D elements
    is_bottom = j_flat == 0
    is_top = (
        j_flat == ndivy_total - 1
    )  # Top surface (free surface - NO boundary elements)
    is_left = i_flat == 0
    is_right = i_flat == ndivx_total - 1

    # Determine boundary types based on boundary condition type
    if config.boundary_condition_type == "1D":
        # 1D Site Response: Only bottom boundary is absorbing
        is_boundary = is_bottom
        # For 1D: only bottom elements are boundaries
        btype_array = np.where(
            is_bottom, np.where(is_left, "LB", np.where(is_right, "RB", "B")), ""
        )
    else:  # 2D Free Field
        # 2D Free Field: Left, Right, and Bottom boundaries are absorbing
        # Top surface is FREE SURFACE (no absorbing boundaries) - exclude top row
        # Surface nodes at y=Ly are completely free to move (no constraints)
        is_boundary = (is_left | is_right | is_bottom) & ~is_top
        # Build boundary type strings (excluding top surface)
        btype_array = np.empty(n_elements, dtype=object)
        btype_array[is_left & is_bottom & ~is_top] = "LB"
        btype_array[is_right & is_bottom & ~is_top] = "RB"
        btype_array[is_left & ~is_bottom & ~is_top] = "L"
        btype_array[is_right & ~is_bottom & ~is_top] = "R"
        btype_array[is_bottom & ~is_left & ~is_right & ~is_top] = "B"
        btype_array[~is_boundary] = ""

    # Compute adjusted indices for material property lookup (vectorized)
    # For boundary elements, properties are taken from the adjacent interior columns:
    # - Left boundaries (i=0): use col_idx=0 (leftmost soil column)
    # - Right boundaries (i=ndivx_total-1): use col_idx=ndivx_soil-1 (rightmost soil column)
    # - Bottom boundaries: use properties from bottom row
    # This ensures boundary elements use material properties from the edge of the soil domain
    adj_i = np.clip(i_flat, 1, ndivx_total - 2)
    adj_j = np.clip(j_flat, 1, ndivy_total - 1)
    row_idx = ndivy_soil - (adj_j - 1) - 1
    col_idx = adj_i - 1

    # Extract material properties for all elements (vectorized)
    vs_all = vs_data[row_idx, col_idx]
    rho_all = rho_data[row_idx, col_idx]
    nu_all = nu_data[row_idx, col_idx]

    # Compute material properties for all elements (vectorized)
    G_all = rho_all * vs_all**2
    E_all = G_all * 2.0 * (1.0 + nu_all)

    # Process elements in original order (maintain compatibility with OpenSees)
    # First, find unique material properties for soil elements only (for material map)
    soil_mask = ~is_boundary
    if np.any(soil_mask):
        soil_indices_temp = np.where(soil_mask)[0]
        E_soil_temp = E_all[soil_indices_temp]
        nu_soil_temp = nu_all[soil_indices_temp]
        rho_soil_temp = rho_all[soil_indices_temp]

        # Create material properties tuples for unique identification
        mat_props_array = np.empty(
            len(E_soil_temp), dtype=[("E", "f8"), ("nu", "f8"), ("rho", "f8")]
        )
        mat_props_array["E"] = E_soil_temp
        mat_props_array["nu"] = nu_soil_temp
        mat_props_array["rho"] = rho_soil_temp

        # Find unique material properties
        _, unique_indices_temp, inverse_indices_temp = np.unique(
            mat_props_array, return_index=True, return_inverse=True
        )

        # Create material map from unique properties
        mat_tag_lookup_soil = np.empty(len(unique_indices_temp), dtype=np.int32)
        mat_tag_counter = 1
        for i, unique_idx in enumerate(unique_indices_temp):
            E_val = float(E_soil_temp[unique_idx])
            nu_val = float(nu_soil_temp[unique_idx])
            rho_val = float(rho_soil_temp[unique_idx])
            mat_props_tuple = (E_val, nu_val, rho_val)
            model.material_map[mat_props_tuple] = mat_tag_counter
            mat_tag_lookup_soil[i] = mat_tag_counter
            mat_tag_counter += 1

        # Create lookup for all soil elements
        mat_tags_all_soil = np.zeros(n_elements, dtype=np.int32)
        mat_tags_all_soil[soil_indices_temp] = mat_tag_lookup_soil[inverse_indices_temp]
    else:
        mat_tags_all_soil = np.zeros(n_elements, dtype=np.int32)

    # Process all elements in original order (boundary and soil interleaved)
    boundary_elements_list = []
    soil_elements_list = []

    # Process elements in the original nested loop order
    for idx in range(n_elements):
        elem_tag = int(elem_tags[idx])
        nodes_tuple = (int(N1[idx]), int(N2[idx]), int(N3[idx]), int(N4[idx]))

        if is_boundary[idx]:
            # Boundary element
            # Material properties (G, poiss, rho) are computed from vs_data at the adjacent
            # interior column/row. For lateral boundaries:
            # - Left (L): Uses vs_data[:, 0] (leftmost column, varies with depth)
            # - Right (R): Uses vs_data[:, ndivx_soil-1] (rightmost column, varies with depth)
            # - Bottom (B): Uses vs_data at corresponding depth
            # The Vs value used is: vs_all[idx] = vs_data[row_idx, col_idx]
            model.abs_element_tags.append(elem_tag)
            boundary_elements_list.append(
                BoundaryElementData(
                    tag=elem_tag,
                    nodes=nodes_tuple,
                    btype=str(btype_array[idx]),
                    G=float(G_all[idx]),  # Computed as rho * vs^2 from vs_data
                    poiss=float(nu_all[idx]),
                    rho=float(rho_all[idx]),
                )
            )
        else:
            # Soil element
            # For 8-node elements, add mid-nodes
            if use_8node:
                base_corner_nodes = (ndivy_total + 1) * (ndivx_total + 1)
                base_horiz_mid = base_corner_nodes
                base_vert_mid = base_corner_nodes + (ndivy_total + 1) * ndivx_total

                j_val = j_flat[idx]
                i_val = i_flat[idx]

                N5 = base_horiz_mid + j_val * ndivx_total + i_val + 1
                N6 = base_vert_mid + j_val * (ndivx_total + 1) + (i_val + 1) + 1
                N7 = base_horiz_mid + (j_val + 1) * ndivx_total + i_val + 1
                N8 = base_vert_mid + j_val * (ndivx_total + 1) + i_val + 1

                nodes_tuple = (
                    int(N1[idx]),
                    int(N2[idx]),
                    int(N3[idx]),
                    int(N4[idx]),
                    int(N5),
                    int(N6),
                    int(N7),
                    int(N8),
                )

            mat_tag = int(mat_tags_all_soil[idx])
            gravity = -9.806 * float(rho_all[idx])
            vs_val = float(vs_all[idx])

            soil_elements_list.append(
                SoilElementData(
                    tag=elem_tag,
                    nodes=nodes_tuple,
                    mat_tag=mat_tag,
                    gravity_load=gravity,
                    vs_value=vs_val,
                )
            )

    model.boundary_elements = boundary_elements_list
    model.soil_elements = soil_elements_list

    return model
