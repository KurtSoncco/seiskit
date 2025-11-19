"""Recorder configuration utilities for OpenSees analyses.

This module provides flexible recorder setup functions that allow users to configure
which nodes and degrees of freedom to record during analysis.
"""

from pathlib import Path
from typing import Any, Dict

try:
    import openseespy.opensees as ops  # type: ignore
except Exception:  # pragma: no cover - OpenSees not available in test env
    ops = None

from seiskit.config import AnalysisConfig


def setup_recorders(
    config: AnalysisConfig,
    ndivx: int,
    ndivy: int,
    output_path: Path,
) -> Dict[str, Any]:
    """Setup recorders based on configuration options.

    Args:
        config: Analysis configuration containing recorder settings
        ndivx: Number of divisions in X direction (including absorbing boundaries)
        ndivy: Number of divisions in Y direction
        output_path: Directory where recorder files will be saved

    Returns:
        Dictionary containing information about created recorders for logging
    """
    if ops is None:
        return {"error": "OpenSees not available"}

    recorder_info = {
        "center_nodes": [],
        "surface_nodes": [],
        "files_created": [],
        "total_recorders": 0,
    }

    # Calculate mesh parameters
    ndivx_plus_1 = ndivx + 1
    i_rec = int(ndivx / 2)  # Center node index

    # Calculate node IDs for default base and top
    soil_base = 1 * ndivx_plus_1 + i_rec + 1
    soil_top = ndivy * ndivx_plus_1 + i_rec + 1

    # Calculate all surface nodes (excluding absorbing boundaries)
    surface_nodes = [
        ndivy * ndivx_plus_1 + i + 1
        for i in range(1, ndivx)  # i from 1 to ndivx-1
    ]

    # Setup center node recorders
    if config.record_center_nodes:
        # Determine which Y positions to record at
        if config.center_node_y_positions is not None:
            # Use specified Y positions
            y_positions = config.center_node_y_positions
        else:
            # Use default: base (y=0) and surface (y=Ly)
            # Note: base is at row 1 (y=0), surface is at row ndivy (y=Ly)
            y_positions = [0.0, config.Ly]

        # Convert Y positions to row indices and then to node IDs
        # Y coordinate calculation: y = (j - 1) * hy for j > 0
        # So for a given Y: j = int(Y / hy) + 1
        # But we need to account for the absorbing boundary at j=0
        center_node_ids = []
        for y_pos in y_positions:
            if y_pos < 0 or y_pos > config.Ly:
                raise ValueError(
                    f"Y position {y_pos} is outside valid range [0, {config.Ly}]"
                )
            # Calculate row index: j = int(Y / hy) + 1
            # Row 1 corresponds to y=0 (base), row ndivy corresponds to y=Ly (surface)
            j_row = int(y_pos / config.hy) + 1
            if j_row < 1:
                j_row = 1
            elif j_row > ndivy:
                j_row = ndivy

            # Calculate node ID for center X position at this row
            node_id = j_row * ndivx_plus_1 + i_rec + 1
            center_node_ids.append(node_id)

        recorder_info["center_nodes"] = center_node_ids

        for dof in config.recorder_dofs:
            for idx, node_id in enumerate(center_node_ids):
                y_pos = y_positions[idx]
                # Create filename based on Y position
                filename = (
                    f"center_node_y{y_pos:.2f}_dof{dof}_{config.recorder_quantity}.txt"
                )
                filepath = output_path / filename
                ops.recorder(
                    "Node",
                    "-file",
                    str(filepath),
                    "-time",
                    "-node",
                    node_id,
                    "-dof",
                    dof,
                    config.recorder_quantity,
                )
                recorder_info["files_created"].append(str(filepath))
                recorder_info["total_recorders"] += 1

    # Setup all surface node recorders
    if config.record_all_surface_nodes:
        recorder_info["surface_nodes"] = surface_nodes

        for dof in config.recorder_dofs:
            # All surface nodes recorder
            surface_filename = f"surface_nodes_dof{dof}_{config.recorder_quantity}.txt"
            surface_filepath = output_path / surface_filename
            ops.recorder(
                "Node",
                "-file",
                str(surface_filepath),
                "-time",
                "-node",
                *surface_nodes,
                "-dof",
                dof,
                config.recorder_quantity,
            )
            recorder_info["files_created"].append(str(surface_filepath))
            recorder_info["total_recorders"] += 1

    return recorder_info


def print_recorder_summary(recorder_info: Dict[str, Any]) -> None:
    """Print a summary of created recorders.

    Args:
        recorder_info: Dictionary returned by setup_recorders()
    """
    if "error" in recorder_info:
        print(f"Recorder setup failed: {recorder_info['error']}")
        return

    print("Recorder Summary:")
    print(f"  Total recorders created: {recorder_info['total_recorders']}")

    if recorder_info["center_nodes"]:
        print(f"  Center nodes recorded: {recorder_info['center_nodes']}")

    if recorder_info["surface_nodes"]:
        print(f"  Surface nodes recorded: {len(recorder_info['surface_nodes'])} nodes")
        print(
            f"    Node IDs: {recorder_info['surface_nodes'][:5]}{'...' if len(recorder_info['surface_nodes']) > 5 else ''}"
        )

    print("  Output files:")
    for filepath in recorder_info["files_created"]:
        print(f"    - {filepath}")
