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
    """

    tag: int
    nodes: Tuple[int, ...]  # 4 nodes for 4-node, 8 nodes for 8-node
    mat_tag: int
    gravity_load: float


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

    # Generate corner nodes
    for j in range(ndivy_total + 1):
        y = -abs_h if j == 0 else (j - 1) * config.hy
        for i in range(ndivx_total + 1):
            x = (
                -abs_h
                if i == 0
                else (config.Lx + abs_h if i == ndivx_total else (i - 1) * config.hx)
            )
            node_tag = j * (ndivx_total + 1) + i + 1
            model.nodes.append(NodeData(tag=node_tag, x=x - config.Lx / 2.0, y=y))

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

    # 2. Generate Element and Material Data
    for j in range(ndivy_total):
        Yflag = "B" if j == 0 else ""

        for i in range(ndivx_total):
            Etag = j * ndivx_total + i + 1
            N1, N2 = j * (ndivx_total + 1) + i + 1, j * (ndivx_total + 1) + i + 2
            N4, N3 = (
                (j + 1) * (ndivx_total + 1) + i + 1,
                (j + 1) * (ndivx_total + 1) + i + 2,
            )
            nodes = (N1, N2, N3, N4)

            # Determine element boundary type flag
            if i == 0:
                Xflag = "L"
            elif i == ndivx_total - 1:
                Xflag = "R"
            else:
                Xflag = ""

            # Apply different logic based on boundary condition type
            if config.boundary_condition_type == "1D":
                # 1D Site Response: Only bottom boundary is absorbing
                if Yflag == "B":
                    btype = f"{Xflag}{Yflag}"  # "LB", "B", "RB"
                else:
                    btype = ""  # Force side elements to be non-absorbing
            else:  # 2D Free Field
                # 2D Free Field: All boundaries (L, R, B) are absorbing
                btype = f"{Xflag}{Yflag}"  # "L", "R", "B", "LB", "RB", or ""

            if btype != "":
                # Absorbing element
                model.abs_element_tags.append(Etag)
                adj_i = min(max(1, i), ndivx_total - 2)
                adj_j = min(max(1, j), ndivy_total - 1)
                row_idx = ndivy_soil - (adj_j - 1) - 1
                col_idx = adj_i - 1
                vs, rho, poiss = (
                    vs_data[row_idx, col_idx],
                    rho_data[row_idx, col_idx],
                    nu_data[row_idx, col_idx],
                )

                model.boundary_elements.append(
                    BoundaryElementData(
                        tag=Etag,
                        nodes=nodes,
                        btype=btype,
                        G=rho * vs**2,
                        poiss=poiss,
                        rho=rho,
                    )
                )
            else:
                # Soil element
                # Use the same indexing logic as the original implementation
                adj_i = min(max(1, i), ndivx_total - 2)
                adj_j = min(max(1, j), ndivy_total - 1)
                row_idx = ndivy_soil - (adj_j - 1) - 1
                col_idx = adj_i - 1
                vs, rho, poiss = (
                    vs_data[row_idx, col_idx],
                    rho_data[row_idx, col_idx],
                    nu_data[row_idx, col_idx],
                )

                G = rho * vs**2
                E = G * 2.0 * (1.0 + poiss)
                mat_props = (E, poiss, rho)

                if mat_props not in model.material_map:
                    model.material_map[mat_props] = mat_tag_counter
                    mat_tag_counter += 1

                mat_tag = model.material_map[mat_props]
                gravity = -9.806 * rho

                # For 8-node elements, add mid-nodes
                if use_8node:
                    # Node ordering for 8-node quad in OpenSees:
                    # Corners: N1, N2, N3, N4 (same as 4-node)
                    # Mid-edges: N5 (between N1-N2), N6 (between N2-N3),
                    #            N7 (between N3-N4), N8 (between N4-N1)
                    base_corner_nodes = (ndivy_total + 1) * (ndivx_total + 1)
                    base_horiz_mid = base_corner_nodes
                    base_vert_mid = base_corner_nodes + (ndivy_total + 1) * ndivx_total

                    # N5: horizontal mid-node between N1 and N2 (bottom edge)
                    N5 = base_horiz_mid + j * ndivx_total + i + 1
                    # N6: vertical mid-node between N2 and N3 (right edge)
                    N6 = base_vert_mid + j * (ndivx_total + 1) + (i + 1) + 1
                    # N7: horizontal mid-node between N3 and N4 (top edge)
                    N7 = base_horiz_mid + (j + 1) * ndivx_total + i + 1
                    # N8: vertical mid-node between N4 and N1 (left edge)
                    N8 = base_vert_mid + j * (ndivx_total + 1) + i + 1

                    nodes = (N1, N2, N3, N4, N5, N6, N7, N8)
                else:
                    nodes = (N1, N2, N3, N4)

                model.soil_elements.append(
                    SoilElementData(
                        tag=Etag, nodes=nodes, mat_tag=mat_tag, gravity_load=gravity
                    )
                )

    return model
