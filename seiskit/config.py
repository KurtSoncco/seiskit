"""Configuration settings for OpenSees 2D site response analysis.

This module defines the AnalysisConfig dataclass which holds all configuration
parameters for running seismic site response analyses.
"""

from dataclasses import dataclass, field


@dataclass
class AnalysisConfig:
    """Configuration settings for a 2D site response analysis.
    
    This class holds all configuration parameters for running seismic site
    response analyses, including domain geometry, analysis parameters, damping,
    boundary conditions, recorders, and solver configuration.
    
    Solver Configuration:
        The solver_type field controls which linear solver is used:
        - "UmfPack": Default solver, always available
        - "Mumps": MUMPS solver (requires OpenSees built with MUMPS support)
        - "MumpsParallel": Parallel MUMPS solver (requires MPI and MUMPS support)
        
        Note: MUMPS support is compile-time in OpenSees. If MUMPS is requested
        but not available, the code will automatically fall back to UmfPack.
        
        For MumpsParallel, set mumps_parallel_procs to the number of MPI processes.
        MUMPS control parameters can be set via mumps_icntl dictionary.
    """

    # Domain and Mesh
    Ly: float = 140.0
    Lx: float = 260.0
    hx: float = 5.0

    # Dynamic Analysis
    duration: float = 15.0
    dt: float = 0.001

    # Input Motion (Ricker Wavelet)
    motion_freq: float = 0.75
    motion_t_shift: float = 1.4

    # Damping Parameters
    damping_zeta: float = 0.02
    damping_freqs: tuple[float, float] = field(default_factory=lambda: (1.0, 5.0))

    # Analysis Constants
    gravity_tolerance: float = 1.0e-4
    max_gravity_iter: int = 10
    dynamic_tolerance: float = 1.0e-4
    max_dynamic_iter: int = 10

    # Boundary Condition Type
    boundary_condition_type: str = "1D"  # "1D" for site response, "2D" for free field

    # Recorder Configuration
    record_center_nodes: bool = (
        True  # Enable/disable recording of center nodes (base and surface)
    )
    record_all_surface_nodes: bool = (
        False  # Enable/disable recording of all surface nodes
    )
    recorder_dofs: list[int] = field(
        default_factory=lambda: [1]
    )  # List of DOFs to record (1=X, 2=Y)
    recorder_quantity: str = "accel"  # What to record ("accel", "disp", "vel", etc.)

    # Element Type
    element_type: str = "4node"  # "4node" for 4-node quads, "8node" for 8-node quads

    # Solver Configuration
    solver_type: str = "UmfPack"  # "UmfPack", "Mumps", or "MumpsParallel"
    mumps_parallel_procs: int = 1  # Number of MPI processes for MumpsParallel (ignored for other solvers)
    mumps_icntl: dict[int, int] = field(
        default_factory=dict
    )  # Optional MUMPS control parameters (ICNTL array)
    
    # Analysis Timeout Configuration
    max_time_per_batch: float = 600.0  # Maximum time (seconds) per dynamic analysis batch before timeout (default: 600s = 10 min)

    @property
    def hy(self) -> float:
        """Element size in the vertical direction (assumed equal to hx)."""
        return self.hx
