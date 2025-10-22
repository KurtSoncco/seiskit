"""Recorder configuration utilities for OpenSees analyses.

This module provides flexible recorder setup functions that allow users to configure
which nodes and degrees of freedom to record during analysis.
"""

from pathlib import Path
from typing import List, Dict, Any

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
        "total_recorders": 0
    }
    
    # Calculate mesh parameters
    ndivx_plus_1 = ndivx + 1
    i_rec = int(ndivx / 2)  # Center node index
    
    # Calculate node IDs
    soil_base = 1 * ndivx_plus_1 + i_rec + 1
    soil_top = ndivy * ndivx_plus_1 + i_rec + 1
    
    # Calculate all surface nodes (excluding absorbing boundaries)
    surface_nodes = [
        ndivy * ndivx_plus_1 + i + 1 
        for i in range(1, ndivx)  # i from 1 to ndivx-1
    ]
    
    # Setup center node recorders
    if config.record_center_nodes:
        recorder_info["center_nodes"] = [soil_base, soil_top]
        
        for dof in config.recorder_dofs:
            # Base node recorder
            base_filename = f"soil_base_dof{dof}_{config.recorder_quantity}.txt"
            base_filepath = output_path / base_filename
            ops.recorder(
                "Node",
                "-file",
                str(base_filepath),
                "-time",
                "-node",
                soil_base,
                "-dof",
                dof,
                config.recorder_quantity,
            )
            recorder_info["files_created"].append(str(base_filepath))
            recorder_info["total_recorders"] += 1
            
            # Top node recorder
            top_filename = f"soil_top_dof{dof}_{config.recorder_quantity}.txt"
            top_filepath = output_path / top_filename
            ops.recorder(
                "Node",
                "-file",
                str(top_filepath),
                "-time",
                "-node",
                soil_top,
                "-dof",
                dof,
                config.recorder_quantity,
            )
            recorder_info["files_created"].append(str(top_filepath))
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
    
    print(f"Recorder Summary:")
    print(f"  Total recorders created: {recorder_info['total_recorders']}")
    
    if recorder_info["center_nodes"]:
        print(f"  Center nodes recorded: {recorder_info['center_nodes']}")
    
    if recorder_info["surface_nodes"]:
        print(f"  Surface nodes recorded: {len(recorder_info['surface_nodes'])} nodes")
        print(f"    Node IDs: {recorder_info['surface_nodes'][:5]}{'...' if len(recorder_info['surface_nodes']) > 5 else ''}")
    
    print(f"  Output files:")
    for filepath in recorder_info["files_created"]:
        print(f"    - {filepath}")
