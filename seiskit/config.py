"""Configuration for OpenSees site-response analyses.

``AnalysisConfig`` holds geometry, motion, damping, boundary conditions,
recorders, and solver settings used by ``build_model_data`` and
``run_isolated_analysis``.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(slots=False)
class AnalysisConfig:
    """Parameters for a 1D or 2D linear site-response analysis.

    Boundary conditions
    -------------------
    ``boundary_condition_type``:

    - ``"1D"`` — simple shear: fix uy, tie ux with ``equalDOF`` at each elevation;
      absorbing boundaries on the **bottom only**. Use a single soil column
      (``Lx = hx``) for classical 1D profiles.
    - ``"2D"`` — free field: no kinematic ties; absorbing boundaries on left,
      right, and bottom (for laterally heterogeneous domains).

    Input motion
    ------------
    The Ricker wavelet is generated as acceleration and integrated to
    **velocity** before being passed to ``ASDAbsorbingBoundary2D -fx``
    (Lysmer–Kuhlemeyer injection requires velocity).

    Damping
    -------
    ``damping_method``: ``"none"``, ``"global_avg"``, ``"elemental_varying"``,
    ``"elemental_mass_only"``, ``"uniform"``, or ``"uniform_soil_only"``.

    Recorders
    ---------
    By default, center-line nodes at ``y = 0`` (base) and ``y = Ly`` (surface).
    For layered 1D columns with a bedrock buffer, set
    ``center_node_y_positions`` to the soil–bedrock interface and the surface.
    """

    # Domain and mesh
    Ly: float = 140.0
    Lx: float = 260.0
    hx: float = 5.0

    # Dynamic analysis
    duration: float = 15.0
    dt: float = 0.001

    # Input motion (Ricker wavelet → velocity for ASDA -fx)
    motion_freq: float = 0.75
    motion_t_shift: float = 1.4

    # Damping
    damping_zeta: float = 0.02
    damping_freqs: tuple[float, float] = field(default_factory=lambda: (1.0, 5.0))
    damping_method: str = "global_avg"
    damping_f_target: float = 0.75
    dmin_multiplier: float = 1.0

    # Solver tolerances
    gravity_tolerance: float = 1.0e-4
    max_gravity_iter: int = 10
    dynamic_tolerance: float = 1.0e-4
    max_dynamic_iter: int = 10

    # Boundary conditions: "1D" (simple shear) or "2D" (free field)
    boundary_condition_type: str = "1D"

    # Recorders
    record_center_nodes: bool = True
    center_node_y_positions: Optional[list[float]] = field(default_factory=lambda: None)
    record_lateral_span_at_center_depths: Optional[Tuple[int, float]] = None
    record_all_surface_nodes: bool = False
    recorder_dofs: list[int] = field(default_factory=lambda: [1])
    recorder_quantity: str = "accel"

    # Elements / solver
    element_type: str = "4node"  # "4node" | "8node"
    solver_type: str = "UmfPack"  # "UmfPack" | "Mumps" | "MumpsParallel"
    mumps_parallel_procs: int = 1
    mumps_icntl: dict[int, int] = field(default_factory=dict)

    max_time_per_batch: float = 600.0

    @property
    def hy(self) -> float:
        """Element size in the vertical direction (equal to ``hx``)."""
        return self.hx
