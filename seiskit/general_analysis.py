"""Compatibility module for general_analysis functions.

This module provides backward compatibility by re-exporting functions
that have been moved to more appropriate modules in the refactored
structure.
"""

# Import functions from their new locations
from .analysis import perform_analysis_spatial, run_analysis
from .utils import (
    build_mesh_and_materials,
    compute_ricker,
    load_material_properties,
)

# Re-export for backward compatibility
__all__ = [
    "run_analysis",
    "perform_analysis_spatial",
    "load_material_properties",
    "build_mesh_and_materials",
    "compute_ricker",
]
