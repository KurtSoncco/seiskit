"""Isolated OpenSees runner for parallel execution.

This module provides a completely isolated OpenSees execution environment
that can be safely called in separate processes without global state conflicts.
"""

import contextlib
import signal
import timeit
from pathlib import Path
from typing import Optional

try:
    import openseespy.opensees as ops  # type: ignore
except Exception:  # pragma: no cover - OpenSees not available in test env
    ops = None

from seiskit.builder import ModelData
from seiskit.config import AnalysisConfig
from seiskit.damping import (
    compute_average_damping_harmonic,
    compute_damping_from_Q,
    compute_quality_factor,
    compute_rayleigh_coefficients,
    compute_rayleigh_mass_only,
)
from seiskit.recorders import print_recorder_summary, setup_recorders
from seiskit.solver_utils import setup_solver
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

        # ---------------------------------------------------------
        # 1. TIME SERIES SETUP
        # ---------------------------------------------------------
        # Compute Ricker values (acceleration input)
        ts_vals = compute_ricker(
            config.motion_freq, config.motion_t_shift, config.duration, config.dt
        )

        ts_tag = 1
        ops.timeSeries("Path", ts_tag, "-dt", config.dt, "-values", *ts_vals, "-factor", 1)

        # ---------------------------------------------------------
        # 3. MESH GENERATION
        # ---------------------------------------------------------
        # Create Nodes
        for node in model_data.nodes:
            ops.node(node.tag, node.x, node.y)

        # Apply Boundary Conditions
        if config.boundary_condition_type == "1D":
            _apply_1d_boundary_conditions(config)
        elif config.boundary_condition_type == "2D":
            _apply_2d_boundary_conditions(config)  # Assuming free field
        else:
            raise ValueError(
                f"Unknown boundary condition type: {config.boundary_condition_type}. Use '1D' or '2D'."
            )

        # 3. Create Materials
        for mat_props, mat_tag in model_data.material_map.items():
            E, poiss, rho = mat_props
            ops.nDMaterial("ElasticIsotropic", mat_tag, E, poiss, rho)

        # 4. Create Soil Elements
        interior_soil_element_tags = []
        for elem in model_data.soil_elements:
            if config.element_type == "8node":
                nodes_to_use = elem.nodes[:4] if len(elem.nodes) == 8 else elem.nodes
                ops.element(
                    "enhancedQuad",
                    elem.tag,
                    *nodes_to_use,
                    1.0,
                    "PlaneStrain",
                    elem.mat_tag,
                    0.0,
                    0.0,
                    0.0,
                    elem.gravity_load,
                )
            else:
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
            interior_soil_element_tags.append(elem.tag)

        # 5. Create Boundary Elements (ASDAbsorbingBoundary2D)
        # Stage 0 (default): Elements act as stiff constraints for gravity analysis
        # Stage 1 (switched later): Elements act as viscous dashpots (absorbing boundaries)
        # Bottom elements (containing "B") also receive time series for input motion
        for elem in model_data.boundary_elements:
            if "B" in elem.btype:
                # Bottom boundary: Absorber + input source (Force-Based Input method)
                # The -fx argument introduces shear stress history for upward wave injection
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
                # Side boundaries (L, R): Pure absorbers (no input motion)
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

        # ---------------------------------------------------------
        # 6. RUN GRAVITY ANALYSIS
        # ---------------------------------------------------------
        # ASDA elements are in Stage 0 by default, acting as stiff constraints
        # Stage 0: Boundaries act as rigid supports (fixed in X and Y) to support soil weight
        # - For 1D: equalDOF constraints + ASDA elements in Stage 0 provide support
        # - For 2D: ASDA elements in Stage 0 provide rigid support (no equalDOF - allows 2D behavior)
        bc_type_str = "2D" if config.boundary_condition_type == "2D" else "1D"
        print(f"Gravity analysis ({bc_type_str}): ASDA elements in Stage 0 (rigid supports)")
        _run_gravity_analysis_isolated(config, run_id)

        # ---------------------------------------------------------
        # 7. SWITCH TO DYNAMIC MODE (Stage 1)
        # ---------------------------------------------------------
        # Switch ASDA elements to Stage 1: viscous dashpots (absorbing boundaries)
        # Stage 1 behavior:
        #   - The internal "fix" is removed (boundaries become compliant)
        #   - A constant vertical force (from Stage 0 reaction) maintains equilibrium
        #   - Viscous dashpots activate to absorb reflected waves
        #   - Bottom boundary injects input motion via Force-Based Input method
        # Note: For 2D, nodes can move independently (no equalDOF); for 1D, equalDOF remains active
        print("Switching ASDA elements to Stage 1 (absorbing boundaries) for dynamic analysis")
        ops.setParameter("-val", 1, "-ele", *model_data.abs_element_tags, "stage")

        # 8. Setup Recorders
        ndivx_total = int(config.Lx / config.hx) + 2
        ndivy_total = int(config.Ly / config.hy) + 1

        # Setup recorders based on configuration
        recorder_info = setup_recorders(config, ndivx_total, ndivy_total, run_output_path)
        print_recorder_summary(recorder_info)

        # 9. Run Dynamic Analysis
        _run_dynamic_analysis_isolated(config, model_data, interior_soil_element_tags, run_id)

        # 10. Clean up completely
        ops.wipe()

        elapsed = timeit.default_timer() - start_time
        return f"Finished {run_id} (t={elapsed:.2f}s)"

    except Exception as e:
        # Ensure cleanup even on error
        try:
            ops.wipe()
        except Exception:
            pass
        return f"Failed {run_id}: {str(e)}"


def _run_gravity_analysis_isolated(config: AnalysisConfig, run_id: str) -> None:
    """Run gravity analysis in isolated environment."""
    # Setup solver based on configuration
    original_solver = config.solver_type
    try:
        setup_solver(config, analysis_type="gravity")
    except Exception as e:
        # Fall back to UmfPack if MUMPS is not available or fails
        if original_solver != "UmfPack":
            print(f"Warning: Failed to use solver '{original_solver}': {e}")
            print(f"Warning: Falling back to UmfPack solver for run {run_id}.")
            config.solver_type = "UmfPack"
            setup_solver(config, analysis_type="gravity")
        else:
            # If UmfPack also fails, re-raise the error
            raise

    ops.test("NormUnbalance", config.gravity_tolerance, config.max_gravity_iter, 1)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")

    # Guard against rare hangs during gravity solve
    with _time_limit(300, msg=f"Gravity analysis timeout for run {run_id}"):
        result = ops.analyze(1)
    if result != 0:
        raise RuntimeError(f"Gravity analysis failed for run {run_id}")

    ops.loadConst("-time", 0.0)
    ops.wipeAnalysis()


def _identify_bedrock_from_mask(
    model_data: ModelData,
) -> tuple[list, list]:
    """Identify bedrock and soil elements using the is_bedrock flag from ModelData.

    This is the preferred method when bedrock_mask is available from create_vs_realization.

    Args:
        model_data: ModelData containing soil elements with is_bedrock flags

    Returns:
        Tuple of (soil_elements, bedrock_elements) lists
    """
    if not model_data.soil_elements:
        return [], []

    soil_elements = []
    bedrock_elements = []

    for elem in model_data.soil_elements:
        if elem.is_bedrock:
            bedrock_elements.append(elem)
        else:
            soil_elements.append(elem)

    return soil_elements, bedrock_elements


def _apply_damping(
    config: AnalysisConfig,
    model_data: ModelData,
    interior_soil_element_tags: list[int],
) -> None:
    """Apply damping based on the configured damping method."""

    if config.damping_method == "none":
        # No damping: skip damping setup entirely
        return

    elif config.damping_method == "global_avg":
        # Model A: Global Average Damping
        # Separate soil and bedrock elements using bedrock mask from create_vs_realization
        # Use bedrock mask if available (is_bedrock flags set), otherwise treat all as soil
        # Note: bedrock_mask should be provided via build_model_data() for accurate identification
        soil_elements, bedrock_elements = _identify_bedrock_from_mask(model_data)

        # Calculate harmonic mean Q from soil layer elements only
        if not soil_elements:
            raise ValueError("No soil layer elements found for global average damping")

        Q_values_soil = [compute_quality_factor(elem.vs_value) for elem in soil_elements]
        avg_damping_soil = compute_average_damping_harmonic(Q_values_soil)

        # Compute Rayleigh coefficients for soil layer
        alphaM_soil, betaK_soil = compute_rayleigh_coefficients(
            avg_damping_soil, config.damping_freqs[0], config.damping_freqs[1]
        )

        # Apply average damping to all soil layer elements
        soil_tags = [elem.tag for elem in soil_elements]
        if soil_tags:
            ops.region(
                1,
                "-ele",
                *soil_tags,
                "-rayleigh",
                alphaM_soil,
                betaK_soil,
                0.0,
                0.0,
            )

        # Apply bedrock-specific damping to bedrock elements
        if bedrock_elements:
            # Use bedrock Vs (typically 1500 m/s) for consistent damping
            bedrock_Vs = 1500.0
            Q_bedrock = compute_quality_factor(bedrock_Vs)
            xi_bedrock = compute_damping_from_Q(Q_bedrock)
            alphaM_bedrock, betaK_bedrock = compute_rayleigh_coefficients(
                xi_bedrock, config.damping_freqs[0], config.damping_freqs[1]
            )

            bedrock_tags = [elem.tag for elem in bedrock_elements]
            # Use region tag 2 for bedrock
            ops.region(
                2,
                "-ele",
                *bedrock_tags,
                "-rayleigh",
                alphaM_bedrock,
                betaK_bedrock,
                0.0,
                0.0,
            )

    elif config.damping_method == "elemental_varying":
        # Model B: Elemental Varying Damping
        # Each element gets its own damping based on its Vs
        region_tag = 1
        for elem in model_data.soil_elements:
            Q = compute_quality_factor(elem.vs_value)
            xi = compute_damping_from_Q(Q)
            alphaM, betaK = compute_rayleigh_coefficients(
                xi, config.damping_freqs[0], config.damping_freqs[1]
            )
            # Use element tag as region tag for uniqueness
            ops.region(
                region_tag,
                "-ele",
                elem.tag,
                "-rayleigh",
                alphaM,
                betaK,
                0.0,
                0.0,
            )
            region_tag += 1

    elif config.damping_method == "elemental_mass_only":
        # Model C: Elemental Mass-Only Damping
        # Each element gets mass-only damping based on its Vs
        f_target = config.damping_f_target if config.damping_f_target > 0 else config.motion_freq
        region_tag = 1
        for elem in model_data.soil_elements:
            Q = compute_quality_factor(elem.vs_value)
            xi = compute_damping_from_Q(Q)
            alphaM, betaK = compute_rayleigh_mass_only(xi, f_target)
            # Use element tag as region tag for uniqueness
            ops.region(
                region_tag,
                "-ele",
                elem.tag,
                "-rayleigh",
                alphaM,
                betaK,
                0.0,
                0.0,
            )
            region_tag += 1

    elif config.damping_method == "uniform":
        # Model D: Uniform Damping
        # Same damping for all elements: zeta=0.0075 at frequencies (0.75, 8.25)
        uniform_zeta = config.damping_zeta  # Use config value (0.0075)
        alphaM_uniform, betaK_uniform = compute_rayleigh_coefficients(
            uniform_zeta, config.damping_freqs[0], config.damping_freqs[1]
        )
        # Apply uniform damping to all soil elements
        all_soil_tags = [elem.tag for elem in model_data.soil_elements]
        if all_soil_tags:
            ops.region(
                1,
                "-ele",
                *all_soil_tags,
                "-rayleigh",
                alphaM_uniform,
                betaK_uniform,
                0.0,
                0.0,
            )

    elif config.damping_method == "uniform_soil_only":
        # Model E: Uniform Soil-Only Damping
        # Specified damping for soil, fixed 0.75% for rock
        soil_elements, bedrock_elements = _identify_bedrock_from_mask(model_data)

        # Apply specified damping to soil elements
        soil_zeta = config.damping_zeta  # Use config value (e.g., 0.025 for 2.5%)
        alphaM_soil, betaK_soil = compute_rayleigh_coefficients(
            soil_zeta, config.damping_freqs[0], config.damping_freqs[1]
        )
        soil_tags = [elem.tag for elem in soil_elements]
        if soil_tags:
            ops.region(
                1,
                "-ele",
                *soil_tags,
                "-rayleigh",
                alphaM_soil,
                betaK_soil,
                0.0,
                0.0,
            )

        # Apply fixed 0.75% damping to bedrock elements
        if bedrock_elements:
            bedrock_zeta = 0.0075  # Fixed 0.75% for rock
            alphaM_bedrock, betaK_bedrock = compute_rayleigh_coefficients(
                bedrock_zeta, config.damping_freqs[0], config.damping_freqs[1]
            )
            bedrock_tags = [elem.tag for elem in bedrock_elements]
            ops.region(
                2,
                "-ele",
                *bedrock_tags,
                "-rayleigh",
                alphaM_bedrock,
                betaK_bedrock,
                0.0,
                0.0,
            )

    else:
        raise ValueError(
            f"Unknown damping method: {config.damping_method}. "
            f"Must be one of: 'none', 'global_avg', 'elemental_varying', 'elemental_mass_only', 'uniform', 'uniform_soil_only'"
        )


def _run_dynamic_analysis_isolated(
    config: AnalysisConfig,
    model_data: ModelData,
    interior_soil_element_tags: list[int],
    run_id: str,
) -> None:
    """Run dynamic analysis in isolated environment with progress tracking."""
    import time

    # Setup damping based on configured method
    _apply_damping(config, model_data, interior_soil_element_tags)

    # Setup analysis
    # Setup solver based on configuration
    original_solver = config.solver_type
    try:
        setup_solver(config, analysis_type="dynamic")
        # Print solver information
        solver_info = f"Solver: {config.solver_type}"
        if config.solver_type == "MumpsParallel" and config.mumps_parallel_procs > 1:
            solver_info += f" (parallel, {config.mumps_parallel_procs} procs)"
        print(solver_info)
    except Exception as e:
        # Fall back to UmfPack if MUMPS is not available or fails
        if original_solver != "UmfPack":
            print(f"Warning: Failed to use solver '{original_solver}': {e}")
            print(f"Warning: Falling back to UmfPack solver for run {run_id}.")
            config.solver_type = "UmfPack"
            setup_solver(config, analysis_type="dynamic")
            print(f"Solver: {config.solver_type}")
        else:
            # If UmfPack also fails, re-raise the error
            raise

    ops.test("NormUnbalance", config.dynamic_tolerance, config.max_dynamic_iter, 1)
    ops.algorithm("Newton", "-initial")
    ops.integrator("TRBDF2")
    ops.analysis("Transient")

    # Run analysis in batches with progress tracking to detect hangs
    # Ensure nsteps is an integer
    nsteps = int(round(config.duration / config.dt))
    print(
        f"[{run_id}] Starting dynamic analysis: {nsteps} steps (dt={config.dt}s, duration={config.duration}s)"
    )

    # Use batch processing: run multiple steps per analyze() call for efficiency
    # but small enough to detect hangs quickly
    # Ensure batch_size is an integer
    batch_size = int(
        max(1, min(100, nsteps // 10))
    )  # ~10 batches total, but at least 1 step per batch

    # Maximum time per batch (seconds) - if a batch takes longer, likely hung
    # Use configurable timeout from AnalysisConfig (default: 600s = 10 min)
    max_time_per_batch = float(config.max_time_per_batch)

    successful_steps = 0
    start_time = time.time()
    step = 0

    while step < nsteps:
        batch_start_time = time.time()
        remaining_steps = int(nsteps - step)
        # Ensure current_batch_size is an integer - OpenSees analyze() requires integer steps
        current_batch_size = int(min(batch_size, remaining_steps))

        # Double-check that we have a valid integer
        if not isinstance(current_batch_size, int) or current_batch_size < 1:
            raise ValueError(
                f"Invalid batch size: {current_batch_size} (type: {type(current_batch_size)})"
            )

        # Run batch of steps with a guard timeout to avoid indefinite hangs
        # Convert to int for signal.alarm() which requires an integer
        timeout_seconds = int(round(max_time_per_batch + 30))
        with _time_limit(
            timeout_seconds,
            msg=(
                f"Dynamic batch timeout for run {run_id} at steps {step + 1}-{step + current_batch_size}/{nsteps}"
            ),
        ):
            # OpenSees analyze() requires integer number of steps
            result = ops.analyze(current_batch_size, float(config.dt))

        batch_time = time.time() - batch_start_time

        # Check for timeout on this batch
        if batch_time > max_time_per_batch:
            raise RuntimeError(
                f"Dynamic analysis batch (steps {step + 1}-{step + current_batch_size}/{nsteps}) "
                f"for run {run_id} took {batch_time:.1f}s (> {max_time_per_batch}s timeout). "
                f"This suggests the analysis is hung. Last {successful_steps} steps completed successfully."
            )

        # Check for analysis failure
        if result != 0:
            raise RuntimeError(
                f"Dynamic analysis failed at step {step + current_batch_size}/{nsteps} for run {run_id}. "
                f"Completed {successful_steps} steps successfully. "
                f"Error code: {result}"
            )

        successful_steps += current_batch_size
        step += current_batch_size

        # Print progress after each batch
        elapsed = time.time() - start_time
        progress = 100.0 * step / nsteps
        avg_step_time = elapsed / step if step > 0 else 0
        remaining_steps_count = nsteps - step
        estimated_remaining = avg_step_time * remaining_steps_count if avg_step_time > 0 else 0
        print(
            f"[{run_id}] Progress: {step}/{nsteps} steps ({progress:.1f}%) | "
            f"Elapsed: {elapsed:.1f}s | Avg: {avg_step_time:.3f}s/step | "
            f"Est. remaining: {estimated_remaining:.1f}s"
        )

    total_time = time.time() - start_time
    print(
        f"[{run_id}] Dynamic analysis completed: {nsteps} steps in {total_time:.1f}s ({total_time / nsteps:.3f}s/step)"
    )


@contextlib.contextmanager
def _time_limit(seconds: int, msg: Optional[str] = None):
    """Raise TimeoutError if the with-block exceeds the given number of seconds.

    Uses SIGALRM; only works on Unix and in the main thread of the process.
    """

    def _handle_signum(signum, frame):
        raise TimeoutError(msg or f"Operation exceeded {seconds} seconds")

    if seconds is None or seconds <= 0:
        yield
        return
    prev_handler = signal.getsignal(signal.SIGALRM)
    try:
        signal.signal(signal.SIGALRM, _handle_signum)
        signal.alarm(seconds)
        yield
    finally:
        signal.alarm(0)
        try:
            signal.signal(signal.SIGALRM, prev_handler)
        except Exception:
            pass


def validate_analysis_setup(
    config: AnalysisConfig,
    model_data: ModelData,
) -> Optional[str]:
    """Validate that the analysis setup is correct without running OpenSees."""
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
    """Apply 1D site response boundary conditions."""
    ndivx_total = int(config.Lx / config.hx) + 2
    ndivy_total = int(config.Ly / config.hy) + 1

    nodes_by_elevation = []

    for j in range(ndivy_total + 1):
        current_elevation_nodes = []
        for i in range(ndivx_total + 1):
            node_id = j * (ndivx_total + 1) + i + 1
            current_elevation_nodes.append(node_id)
        nodes_by_elevation.append(current_elevation_nodes)

    for j, node_row in enumerate(nodes_by_elevation):
        master_node_id = node_row[0]

        for i, node_id in enumerate(node_row):
            ops.fix(node_id, 0, 1)

            if i > 0:
                ops.equalDOF(master_node_id, node_id, 1)


def _apply_2d_boundary_conditions(config: AnalysisConfig) -> None:
    """Apply 2D free field boundary conditions.

    For 2D analysis, NO kinematic constraints (equalDOF) are applied.
    The ASDAbsorbingBoundary2D elements on left and right sides will handle boundary
    conditions, allowing independent motion of nodes on each side. This enables true
    2D wave propagation with heterogeneity and scattering effects.

    The ASDA elements use the stage mechanism:
    - Stage 0: Act as rigid supports during gravity analysis (fixed in X and Y)
    - Stage 1: Switch to absorbing boundaries during dynamic analysis (compliant with viscous dashpots)
    """
    print("Applying 2D free field boundary conditions (no kinematic constraints).")


def _apply_gravity_constraints_2d(config: AnalysisConfig) -> list[int]:
    """Apply temporary constraints for gravity analysis in 2D case."""
    ndivx_total = int(config.Lx / config.hx) + 2

    bottom_nodes = []
    for i in range(ndivx_total + 1):
        bottom_nodes.append(i + 1)

    ops.fix(bottom_nodes[0], 1, 1)
    ops.fix(bottom_nodes[-1], 0, 1)
    for node_id in bottom_nodes[1:-1]:
        ops.fix(node_id, 0, 1)

    print("Applying temporary fixes for gravity analysis...")

    return bottom_nodes


def _remove_gravity_constraints_2d(bottom_nodes: list) -> None:
    """Remove temporary constraints after gravity analysis in 2D case."""
    ops.remove("sp", *bottom_nodes)
    print("Removed temporary gravity fixes.")
