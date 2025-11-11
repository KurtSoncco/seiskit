"""Solver utility functions for checking MUMPS availability and configuration.

This module provides utilities to check if OpenSees supports MUMPS solvers
and to configure solver settings appropriately.
"""

from typing import Any, Dict

try:
    import openseespy.opensees as ops  # type: ignore
except Exception:  # pragma: no cover - OpenSees not available in test env
    ops = None


def check_mumps_availability() -> Dict[str, bool]:
    """Check which MUMPS solvers are available in OpenSees.

    This function attempts to determine if Mumps and MumpsParallel solvers
    are available. Since OpenSees doesn't provide a direct API to check
    solver availability, this function makes a best-effort check by
    examining the system or returns conservative defaults.

    Note: This is a heuristic check. The actual availability will be
    determined at runtime when the solver is used.

    Returns:
        Dictionary with availability status for each solver type:
        - "UmfPack": Always True (default solver)
        - "Mumps": True if Mumps might be available (conservative guess)
        - "MumpsParallel": True if MumpsParallel might be available (conservative guess)

    Example:
        >>> availability = check_mumps_availability()
        >>> if availability["Mumps"]:
        ...     print("MUMPS solver may be available")
    """
    if ops is None:
        return {"UmfPack": False, "Mumps": False, "MumpsParallel": False}

    # UmfPack is always available as the default solver
    availability = {"UmfPack": True}

    # For MUMPS, we can't reliably check without a model, so we assume
    # it might be available and let runtime errors handle it
    # This is conservative - we'll try to use it and fall back if it fails
    availability["Mumps"] = True  # Assume available, fallback on error
    availability["MumpsParallel"] = True  # Assume available, fallback on error

    return availability


def setup_solver(config, analysis_type: str = "dynamic") -> None:
    """Setup the solver based on configuration.

    This function sets up the OpenSees solver system based on the AnalysisConfig.
    It handles UmfPack, Mumps, and MumpsParallel solvers with appropriate
    configuration.

    Args:
        config: AnalysisConfig instance with solver settings
        analysis_type: Type of analysis ("gravity" or "dynamic") - currently unused
                       but kept for future extensibility

    Raises:
        ValueError: If solver_type is not supported
        RuntimeError: If MUMPS solver is requested but not available (raised by OpenSees)
    """
    if ops is None:
        return

    # Set up constraints and numberer (same for all solvers)
    ops.constraints("Transformation")
    ops.numberer("RCM")

    # Set up solver based on type
    if config.solver_type == "UmfPack":
        ops.system("UmfPack")
    elif config.solver_type == "Mumps":
        # Try Mumps solver - will raise error if not available
        ops.system("Mumps")
        # Apply MUMPS control parameters if provided
        # Note: MUMPS ICNTL parameters are typically set via environment variables
        # or OpenSees parameters, but the exact API may vary by OpenSees version
        if config.mumps_icntl:
            # Try to set ICNTL parameters if supported
            # This is version-dependent and may not work in all OpenSees builds
            for key, value in config.mumps_icntl.items():
                try:
                    # Attempt to set parameter - this may not be supported
                    ops.setParameter("-val", value, "MumpsICNTL", key)
                except Exception:
                    # If setting parameters fails, continue without them
                    pass
    elif config.solver_type == "MumpsParallel":
        # MumpsParallel syntax may vary - try with number of processes
        if config.mumps_parallel_procs > 1:
            try:
                # Try MumpsParallel with process count
                ops.system("MumpsParallel", config.mumps_parallel_procs)
            except Exception:
                # If that fails, try without process count (OpenSees may auto-detect)
                try:
                    ops.system("MumpsParallel")
                except Exception:
                    # If MumpsParallel fails, fall back to regular Mumps
                    ops.system("Mumps")
        else:
            # If only 1 process requested, use regular Mumps
            ops.system("Mumps")
        # Apply MUMPS control parameters if provided
        if config.mumps_icntl:
            for key, value in config.mumps_icntl.items():
                try:
                    ops.setParameter("-val", value, "MumpsICNTL", key)
                except Exception:
                    pass
    else:
        raise ValueError(
            f"Unknown solver_type: {config.solver_type}. "
            f"Must be one of: 'UmfPack', 'Mumps', 'MumpsParallel'"
        )


def get_solver_info(config) -> Dict[str, Any]:
    """Get information about the configured solver.

    Args:
        config: AnalysisConfig instance

    Returns:
        Dictionary with solver information
    """
    availability = check_mumps_availability()

    return {
        "solver_type": config.solver_type,
        "available": availability.get(config.solver_type, False),
        "all_available": availability,
        "mumps_parallel_procs": config.mumps_parallel_procs
        if config.solver_type == "MumpsParallel"
        else None,
        "mumps_icntl": config.mumps_icntl if config.mumps_icntl else None,
    }
