# File: config.py
from dataclasses import dataclass, field


@dataclass
class AnalysisConfig:
    """Configuration settings for a 2D site response analysis."""

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
    record_center_nodes: bool = True  # Enable/disable recording of center nodes (base and surface)
    record_all_surface_nodes: bool = False  # Enable/disable recording of all surface nodes
    recorder_dofs: list[int] = field(default_factory=lambda: [1])  # List of DOFs to record (1=X, 2=Y)
    recorder_quantity: str = "accel"  # What to record ("accel", "disp", "vel", etc.)

    @property
    def hy(self) -> float:
        """Element size in the vertical direction (assumed equal to hx)."""
        return self.hx
