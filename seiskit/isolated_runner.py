"""Isolated OpenSees runner for parallel execution.

This module provides a completely isolated OpenSees execution environment
that can be safely called in separate processes without global state conflicts.
"""

import timeit
from pathlib import Path
from typing import Optional

try:
    import openseespy.opensees as ops  # type: ignore
except Exception:  # pragma: no cover - OpenSees not available in test env
    ops = None

from seiskit.builder import ModelData
from seiskit.config import AnalysisConfig
from seiskit.damping import compute_rayleigh_coefficients
from seiskit.recorders import setup_recorders, print_recorder_summary
from seiskit.utils import compute_ricker


def run_isolated_analysis(
    config: AnalysisConfig,
    model_data: ModelData,
    run_id: str,
    output_dir: str = "results",
) -> str:
    """Run a completely isolated OpenSees analysis.

    This function creates a fresh OpenSees environment, runs the analysis,
    and cleans up completely. It's designed to be called in separate processes
    for parallel execution without any global state conflicts.

    Args:
        config: Analysis configuration parameters
        model_data: Pre-built model geometry and materials
        run_id: Unique identifier for this analysis run
        output_dir: Base directory for output files

    Returns:
        Status message indicating success or failure
    """
    if ops is None:
        return f"No OpenSees - {run_id}"

    # Create fresh output directory
    run_output_path = Path(output_dir) / run_id
    run_output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Start timing
        start_time = timeit.default_timer()

        # Create completely fresh OpenSees environment
        ops.wipe()
        ops.model("basic", "-ndm", 2, "-ndf", 2)

        # 1. Create Time Series
        ts_vals = compute_ricker(
            config.motion_freq, config.motion_t_shift, config.duration, config.dt
        )
        ts_tag = 1
        ops.timeSeries(
            "Path", ts_tag, "-dt", config.dt, "-values", *ts_vals, "-factor", 1
        )

        # 2. Create Nodes
        for node in model_data.nodes:
            ops.node(node.tag, node.x, node.y)

        # 2.5. Apply Boundary Conditions (1D or 2D)
        if config.boundary_condition_type == "1D":
            _apply_1d_boundary_conditions(config)
        elif config.boundary_condition_type == "2D":
            _apply_2d_boundary_conditions(config)
        else:
            raise ValueError(
                f"Unknown boundary condition type: {config.boundary_condition_type}. Use '1D' or '2D'."
            )

        # 3. Create Materials
        for mat_props, mat_tag in model_data.material_map.items():
            E, poiss, rho = mat_props
            ops.nDMaterial("ElasticIsotropic", mat_tag, E, poiss, rho)

        # 4. Create Soil Elements
        for elem in model_data.soil_elements:
            ops.element(
                "quad",
                elem.tag,
                *elem.nodes,
                1.0,
                "PlaneStrain",
                elem.mat_tag,
                0.0,
                0.0,
                0.0,
                elem.gravity_load,
            )

        # 5. Create Boundary Elements
        for elem in model_data.boundary_elements:
            if "B" in elem.btype:
                ops.element(
                    "ASDAbsorbingBoundary2D",
                    elem.tag,
                    *elem.nodes,
                    elem.G,
                    elem.poiss,
                    elem.rho,
                    1.0,
                    elem.btype,
                    "-fx",
                    ts_tag,
                )
            else:
                ops.element(
                    "ASDAbsorbingBoundary2D",
                    elem.tag,
                    *elem.nodes,
                    elem.G,
                    elem.poiss,
                    elem.rho,
                    1.0,
                    elem.btype,
                )

        # 6. Run Gravity Analysis
        if config.boundary_condition_type == "2D":
            # For 2D case, apply temporary constraints, run gravity, then remove constraints
            bottom_nodes = _apply_gravity_constraints_2d(config)
            _run_gravity_analysis_isolated(config, run_id)
            _remove_gravity_constraints_2d(bottom_nodes)
        else:
            # For 1D case, run gravity analysis normally
            _run_gravity_analysis_isolated(config, run_id)

        # 7. Setup Recorders
        ops.setParameter("-val", 1, "-ele", *model_data.abs_element_tags, "stage")
        ndivx_total = int(config.Lx / config.hx) + 2
        ndivy_total = int(config.Ly / config.hy) + 1

        # Setup recorders based on configuration
        recorder_info = setup_recorders(config, ndivx_total, ndivy_total, run_output_path)
        print_recorder_summary(recorder_info)

        # 8. Run Dynamic Analysis
        _run_dynamic_analysis_isolated(config, run_id)

        # 9. Clean up completely
        ops.wipe()

        elapsed = timeit.default_timer() - start_time
        return f"Finished {run_id} (t={elapsed:.2f}s)"

    except Exception as e:
        # Ensure cleanup even on error
        try:
            ops.wipe()
        except:
            pass
        return f"Failed {run_id}: {str(e)}"


def _run_gravity_analysis_isolated(config: AnalysisConfig, run_id: str) -> None:
    """Run gravity analysis in isolated environment."""
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("UmfPack")
    ops.test("NormUnbalance", config.gravity_tolerance, config.max_gravity_iter, 1)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")

    if ops.analyze(1) != 0:
        raise RuntimeError(f"Gravity analysis failed for run {run_id}")

    ops.loadConst("-time", 0.0)
    ops.wipeAnalysis()


def _run_dynamic_analysis_isolated(config: AnalysisConfig, run_id: str) -> None:
    """Run dynamic analysis in isolated environment."""
    # Setup damping
    alphaM, betaK = compute_rayleigh_coefficients(
        config.damping_zeta, config.damping_freqs[0], config.damping_freqs[1]
    )
    ops.rayleigh(alphaM, betaK, 0.0, 0.0)

    # Setup analysis
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("UmfPack")
    ops.test("NormUnbalance", config.dynamic_tolerance, config.max_dynamic_iter, 1)
    ops.algorithm("Newton", "-initial")
    ops.integrator("TRBDF2")
    ops.analysis("Transient")

    # Run analysis
    nsteps = int(config.duration / config.dt)
    if ops.analyze(nsteps, config.dt) != 0:
        raise RuntimeError(f"Dynamic analysis failed for run {run_id}")


def validate_analysis_setup(
    config: AnalysisConfig,
    model_data: ModelData,
) -> Optional[str]:
    """Validate that the analysis setup is correct without running OpenSees.

    Args:
        config: Analysis configuration
        model_data: Model data

    Returns:
        None if valid, error message if invalid
    """
    # Check configuration
    if config.duration <= 0:
        return "Duration must be positive"
    if config.dt <= 0:
        return "Time step must be positive"
    if config.dt >= config.duration:
        return "Time step must be smaller than duration"

    # Check model data
    if not model_data.nodes:
        return "No nodes defined"
    if not model_data.soil_elements:
        return "No soil elements defined"
    if not model_data.material_map:
        return "No materials defined"

    # Check material data consistency
    for elem in model_data.soil_elements:
        if elem.mat_tag not in model_data.material_map.values():
            return f"Element {elem.tag} references undefined material {elem.mat_tag}"

    return None  # Valid


def _apply_1d_boundary_conditions(config: AnalysisConfig) -> None:
    """Apply 1D site response boundary conditions.

    This is the correct implementation for 1D site response (simple shear deformation).
    It fixes all nodes in the Y-direction and ties X-DOFs of all other nodes to the master node
    at each elevation.
    """
    ndivx_total = int(config.Lx / config.hx) + 2
    ndivy_total = int(config.Ly / config.hy) + 1

    # Store node IDs for applying boundary conditions
    nodes_by_elevation = []  # List of lists

    for j in range(ndivy_total + 1):
        current_elevation_nodes = []
        for i in range(ndivx_total + 1):
            node_id = j * (ndivx_total + 1) + i + 1
            current_elevation_nodes.append(node_id)
        nodes_by_elevation.append(current_elevation_nodes)

    for j, node_row in enumerate(nodes_by_elevation):
        # Get the master node (the first node, i=0, at this elevation)
        master_node_id = node_row[0]

        for i, node_id in enumerate(node_row):
            # 1. Fix ALL nodes in the Y-direction (DOF 2)
            #    (1=fixed, 0=free)
            ops.fix(node_id, 0, 1)

            # 2. Tie X-DOF (DOF 1) of all other nodes to the master node
            if i > 0:
                # ops.equalDOF(masterNode, slaveNode, *dofs)
                ops.equalDOF(master_node_id, node_id, 1)


def _apply_2d_boundary_conditions(config: AnalysisConfig) -> None:
    """Apply 2D free field boundary conditions.

    This implements 2D free field conditions where the model is free to deform in 2D.
    No equalDOF or Y-direction constraints are applied.
    """
    print("Applying 2D free field boundary conditions (no constraints applied).")


def _apply_gravity_constraints_2d(config: AnalysisConfig) -> None:
    """Apply temporary constraints for gravity analysis in 2D case.

    For 2D free field, we need to temporarily fix some nodes to prevent rigid body motion
    during gravity analysis, then remove these constraints before dynamic analysis.
    """
    ndivx_total = int(config.Lx / config.hx) + 2

    # Fix bottom nodes for gravity analysis to prevent rigid body motion
    bottom_nodes = []
    for i in range(ndivx_total + 1):
        bottom_nodes.append(i + 1)  # Node IDs at j=0

    # Fix bottom nodes in X and Y for gravity
    # This assumes the absorbing layer (j=0) has nodes (1) to (ndivx+1)
    ops.fix(bottom_nodes[0], 1, 1)  # Fix first node in X and Y
    ops.fix(bottom_nodes[-1], 0, 1)  # Fix last node in Y only (allows contraction)
    for node_id in bottom_nodes[1:-1]:
        ops.fix(node_id, 0, 1)  # Fix intermediate nodes in Y only

    print("Applying temporary fixes for gravity analysis...")

    return bottom_nodes


def _remove_gravity_constraints_2d(bottom_nodes: list) -> None:
    """Remove temporary constraints after gravity analysis in 2D case."""
    ops.remove("sp", *bottom_nodes)
    print("Removed temporary gravity fixes.")
