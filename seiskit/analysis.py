"""Core analysis functions for 2D site response analysis.

This module consolidates the main analysis functions, providing both
high-level entry points and lower-level OpenSees execution functions.
All functions are designed to work with the structured ModelData and
AnalysisConfig classes for parallelization readiness.
"""

import timeit
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import openseespy.opensees as ops  # type: ignore
except Exception:  # pragma: no cover - OpenSees not available in test env
    ops = None

from seiskit.builder import ModelData, build_model_data
from seiskit.config import AnalysisConfig
from seiskit.utils import compute_ricker, load_material_properties
from seiskit.damping import compute_rayleigh_coefficients


def _run_gravity_analysis(config: AnalysisConfig, run_id: str) -> None:
    """Helper function to run the static gravity analysis."""
    if ops is None:
        return
        
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("UmfPack")
    ops.test("NormUnbalance", config.gravity_tolerance, config.max_gravity_iter, 1)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    if ops.analyze(1) != 0:
        print(f"WARNING: Gravity analysis failed for run {run_id}.")
    ops.loadConst("-time", 0.0)
    ops.wipeAnalysis()


def _run_dynamic_analysis(config: AnalysisConfig, run_id: str) -> None:
    """Helper function to run the transient dynamic analysis."""
    if ops is None:
        return
        
    alphaM, betaK = compute_rayleigh_coefficients(config.damping_zeta, config.damping_freqs[0], config.damping_freqs[1])
    ops.rayleigh(alphaM, betaK, 0.0, 0.0)

    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("UmfPack")
    ops.test("NormUnbalance", config.dynamic_tolerance, config.max_dynamic_iter, 1)
    ops.algorithm("Newton")
    ops.integrator("TRBDF2")
    ops.analysis("Transient")

    nsteps = int(config.duration / config.dt)
    if ops.analyze(nsteps, config.dt) != 0:
        print(f"WARNING: Dynamic analysis failed for run {run_id}.")


def run_opensees_analysis(
    config: AnalysisConfig,
    model_data: ModelData,
    run_id: str,
    output_dir: str = "results",
) -> str:
    """Perform a 2D site response analysis using pre-built model data structure.
    
    This function delegates to the isolated runner for parallelization compatibility.
    For parallel execution, use the parallel module instead.

    Args:
        config: Analysis configuration parameters
        model_data: Pre-built model geometry and materials
        run_id: Unique identifier for this analysis run
        output_dir: Base directory for output files

    Returns:
        Status message indicating success or failure
    """
    print(f"--- Starting OpenSees Analysis: {run_id} ---")
    
    # Use the isolated runner for consistency with parallel execution
    from .isolated_runner import run_isolated_analysis
    return run_isolated_analysis(config, model_data, run_id, output_dir)


def perform_analysis_spatial(
    run_id: str,
    vs_data: np.ndarray,
    rho_data: np.ndarray,
    nu_data: np.ndarray,
    ts_vals: Optional[np.ndarray] = None,
    abs_elements: Optional[List[int]] = None,
    material_map: Optional[Dict[Tuple[float, float, float], int]] = None,
    output_dir: str = "results",
    Ly: Optional[float] = None,
    Lx: Optional[float] = None,
    hx: Optional[float] = None,
    duration: Optional[float] = None,
    dt: Optional[float] = None,
    motion_freq: Optional[float] = None,
    motion_t_shift: Optional[float] = None,
) -> str:
    """Compatibility wrapper for older call sites.

    This function accepts pre-loaded numpy arrays and optional precomputed
    mesh/material info. It constructs an AnalysisConfig and ModelData and
    then delegates to run_opensees_analysis.

    Args:
        run_id: Unique identifier for this analysis run
        vs_data: Shear wave velocity grid data
        rho_data: Density grid data
        nu_data: Poisson's ratio grid data
        ts_vals: Optional pre-computed time series values (ignored, recomputed)
        abs_elements: Optional pre-computed absorbing element list
        material_map: Optional pre-computed material mapping
        output_dir: Base directory for output files
        Ly: Domain height (optional, uses default if not provided)
        Lx: Domain width (optional, uses default if not provided)
        hx: Element size (optional, uses default if not provided)
        duration: Analysis duration (optional, uses default if not provided)
        dt: Time step (optional, uses default if not provided)
        motion_freq: Motion frequency (optional, uses default if not provided)
        motion_t_shift: Motion time shift (optional, uses default if not provided)

    Returns:
        Status message from the analysis
    """
    # Check if OpenSees is available
    if ops is None:
        return "no-opensees"
        
    # Create config from provided values or defaults
    cfg = AnalysisConfig()
    if Ly is not None:
        cfg.Ly = Ly
    if Lx is not None:
        cfg.Lx = Lx
    if hx is not None:
        cfg.hx = hx
    if duration is not None:
        cfg.duration = duration
    if dt is not None:
        cfg.dt = dt
    if motion_freq is not None:
        cfg.motion_freq = motion_freq
    if motion_t_shift is not None:
        cfg.motion_t_shift = motion_t_shift

    # Build model data (always use builder for consistency)
    model = build_model_data(cfg, vs_data, rho_data, nu_data)

    # Delegate to the main runner
    return run_opensees_analysis(cfg, model, run_id, output_dir=output_dir)


def run_analysis(
    run_id: str,
    vs_file: str,
    rho_file: str,
    nu_file: str,
    output_dir: str = "results",
    Ly: float = 140.0,
    Lx: float = 260.0,
    hx: float = 5.0,
    duration: float = 15.0,
    dt: float = 0.001,
    freq: float = 0.75,
    t_shift: float = 1.4,
) -> str:
    """High-level entry point for spatial analysis with file-based inputs.
    
    This function loads material property files and delegates to the 
    structured analysis pipeline.

    Args:
        run_id: Unique identifier for this analysis run
        vs_file: Path to shear wave velocity data file
        rho_file: Path to density data file
        nu_file: Path to Poisson's ratio data file
        output_dir: Base directory for output files
        Ly: Domain height
        Lx: Domain width
        hx: Element size
        duration: Analysis duration
        dt: Time step
        freq: Motion frequency
        t_shift: Motion time shift

    Returns:
        Status message from the analysis
    """
    material_paths = {"vs": vs_file, "rho": rho_file, "nu": nu_file}
    loaded_materials = load_material_properties(material_paths)

    vs_data = loaded_materials.get("vs")
    rho_data = loaded_materials.get("rho")
    nu_data = loaded_materials.get("nu")

    if vs_data is None:
        raise ValueError("Vs data must be provided and loaded as a numpy array.")
    if rho_data is None:
        raise ValueError("Rho data must be provided and loaded as a numpy array.")
    if nu_data is None:
        raise ValueError("Nu data must be provided and loaded as a numpy array.")

    return perform_analysis_spatial(
        run_id=run_id,
        vs_data=vs_data,
        rho_data=rho_data,
        nu_data=nu_data,
        output_dir=output_dir,
        Ly=Ly,
        Lx=Lx,
        hx=hx,
        duration=duration,
        dt=dt,
        motion_freq=freq,
        motion_t_shift=t_shift,
    )
