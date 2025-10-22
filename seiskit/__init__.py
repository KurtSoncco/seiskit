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
    run_analysis,
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
    "run_analysis",
    "utils",
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
    # Parallel execution
    "AnalysisResult",
    "AnalysisTask",
    "collect_results",
    "prepare_analysis_tasks",
    "run_analyses_parallel",
    "run_parallel_analyses",
    "run_parameter_study",
    "run_isolated_analysis",
]
