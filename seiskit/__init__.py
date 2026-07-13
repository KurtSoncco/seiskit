"""seiskit: OpenSees site response with spatial Vs variability.

Core flow
---------
1. ``AnalysisConfig`` — geometry, motion, damping, ``"1D"`` or ``"2D"`` BCs
2. ``build_model_data`` — mesh + materials (no OpenSees calls)
3. ``run_opensees_analysis`` / ``run_isolated_analysis`` — OpenSees solve

- **1D**: simple-shear column (equalDOF, bottom ASDA only). Validated against
  closed-form multilayer TFs in ``seiskit.theory``.
- **2D**: free-field domain (side + bottom ASDA) for heterogeneous / GRF fields.
"""

from . import (
    analysis,
    damping,
    general_analysis,
    isolated_runner,
    optools,
    parallel,
    plot_config,
    plot_results,
    solver_utils,
    theory,
    utils,
)
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
from .theory import Layer, RockHalfspace, layered_transfer_function
from .utils import (
    acceleration_to_velocity,
    build_mesh_and_materials,
    compute_ricker,
    compute_ricker_velocity,
    load_material_properties,
)

__all__ = [
    "analysis",
    "damping",
    "general_analysis",
    "isolated_runner",
    "optools",
    "parallel",
    "plot_config",
    "plot_results",
    "solver_utils",
    "theory",
    "utils",
    "plot_damping_realization",
    "plot_realization",
    "AnalysisConfig",
    "ModelData",
    "build_model_data",
    "build_mesh_and_materials",
    "compute_ricker",
    "compute_ricker_velocity",
    "acceleration_to_velocity",
    "load_material_properties",
    "perform_analysis_spatial",
    "run_analysis",
    "run_opensees_analysis",
    "Layer",
    "RockHalfspace",
    "layered_transfer_function",
    "compute_rayleigh_coefficients",
    "compute_rayleigh_mass_only",
    "compute_quality_factor",
    "compute_damping_from_Q",
    "compute_average_damping_harmonic",
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
    "check_mumps_availability",
    "get_solver_info",
    "setup_solver",
]
