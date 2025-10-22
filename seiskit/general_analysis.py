"""Compatibility module for general_analysis functions.

This module provides backward compatibility by re-exporting functions
that have been moved to more appropriate modules in the refactored
structure.
"""

# Import functions from their new locations
from .analysis import run_analysis, perform_analysis_spatial
from .utils import (
    load_material_properties,
    build_mesh_and_materials,
    compute_ricker,
)

# Re-export for backward compatibility
__all__ = [
    "run_analysis",
    "perform_analysis_spatial", 
    "load_material_properties",
    "build_mesh_and_materials",
    "compute_ricker",
]
