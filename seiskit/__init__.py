"""seiskit package: lightweight tools for seismic processing and OpenSees utilities.

This package consolidates ttf (time-to-frequency) helpers and OpenSees-related
utilities to avoid copying code across example folders.

Expose small, pure-Python utilities for testing without requiring OpenSees.
"""

from . import (
    analysis,
    damping,
    general_analysis,
    isolated_runner,
    optools,
    parallel,
    plot_results,
    solver_utils,
    utils,
)

# Core analysis functions
from .analysis import (
    perform_analysis_spatial,
    run_analysis,
    run_opensees_analysis,
)
from .builder import ModelData, build_model_data
from .config import AnalysisConfig
from .damping import (
    compute_average_damping_harmonic,
    compute_damping_from_Q,
    compute_quality_factor,
    compute_rayleigh_coefficients,
    compute_rayleigh_mass_only,
)
from .gaussian_field import extend_profile, generate_vs_variability_field
from .isolated_runner import run_isolated_analysis

# Parallel execution functions
from .parallel import (
    AnalysisResult,
    AnalysisTask,
    collect_results,
    prepare_analysis_tasks,
    run_analyses_parallel,
    run_parallel_analyses,
    run_parameter_study,
)
from .plot_results import plot_damping_realization, plot_realization
from .solver_utils import check_mumps_availability, get_solver_info, setup_solver
from .utils import (
    build_mesh_and_materials,
    compute_ricker,
    load_material_properties,
)

__all__ = [
    # Main modules
    "analysis",
    "damping",
    "general_analysis",
    "isolated_runner",
    "optools",
    "parallel",
    "plot_results",
    "solver_utils",
    "utils",
    # Plotting functions
    "plot_damping_realization",
    "plot_realization",
    # Core classes and functions
    "AnalysisConfig",
    "ModelData",
    "build_model_data",
    "build_mesh_and_materials",
    "compute_ricker",
    "load_material_properties",
    "perform_analysis_spatial",
    "run_analysis",
    "run_opensees_analysis",
    # Damping functions
    "compute_rayleigh_coefficients",
    "compute_rayleigh_mass_only",
    "compute_quality_factor",
    "compute_damping_from_Q",
    "compute_average_damping_harmonic",
    # Parallel execution
    "AnalysisResult",
    "AnalysisTask",
    "collect_results",
    "prepare_analysis_tasks",
    "run_analyses_parallel",
    "run_parallel_analyses",
    "run_parameter_study",
    "run_isolated_analysis",
    "generate_vs_variability_field",
    "extend_profile",
    # Solver utilities
    "check_mumps_availability",
    "get_solver_info",
    "setup_solver",
]
