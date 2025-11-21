"""Data generation script for transfer function emulator.

Generates material grids, base motions, and runs OpenSees simulations
to create training/validation/test datasets using seiskit.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

# Add parent directory to path to allow importing emulator
sys.path.insert(0, str(Path(__file__).parent.parent))

from emulator.data_utils import load_recorder_outputs
from seiskit.analysis import run_opensees_analysis
from seiskit.builder import build_model_data
from seiskit.config import AnalysisConfig
from seiskit.gaussian_field import create_vs_realization
from seiskit.utils import compute_ricker


def _sample_hf_to_lf_with_jitter(
    material_grid_hf: np.ndarray,
    h_lf: int,
    w_lf: int,
    hx_lf: float,
    hx_hf: float,
    jitter_percent: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample HF material grid to LF resolution with random jitter.

    Args:
        material_grid_hf: HF material grid (H_hf, W_hf, 2)
        h_lf, w_lf: LF grid dimensions
        hx_lf: LF element size
        hx_hf: HF element size
        jitter_percent: Maximum jitter as percentage of LF element size (0.0 to 0.5)
        rng: Random number generator

    Returns:
        LF material grid (H_lf, W_lf, 2)
    """
    h_hf, w_hf = material_grid_hf.shape[:2]
    material_grid_lf = np.zeros((h_lf, w_lf, 2))

    # Calculate max jitter in meters
    max_jitter = jitter_percent * hx_lf

    # Generate random jitter for the entire grid (uniform shift)
    # We shift the sampling points, effectively shifting the grid
    delta_x = rng.uniform(-max_jitter, max_jitter)
    delta_z = rng.uniform(-max_jitter, max_jitter)

    for c in range(2):
        # Create interpolator for HF grid
        x_hf = np.linspace(0, (w_hf - 1) * hx_hf, w_hf)
        y_hf = np.linspace(0, (h_hf - 1) * hx_hf, h_hf)  # Assuming dz_hf = dx_hf
        
        # Use nearest neighbor to preserve sharp boundaries if preferred, 
        # or linear for smoother interpolation. Linear is safer for now.
        interp_func = RegularGridInterpolator(
            (y_hf, x_hf), material_grid_hf[:, :, c], method="linear", bounds_error=False, fill_value=None
        )

        # Create coordinate grids for LF sampling points
        # Nominal LF centers: dx/2, 3dx/2, ...
        x_lf = np.linspace(hx_lf / 2, (w_lf - 1) * hx_lf + hx_lf / 2, w_lf)
        y_lf = np.linspace(hx_lf / 2, (h_lf - 1) * hx_lf + hx_lf / 2, h_lf)
        
        # Apply jitter
        x_lf_jittered = x_lf + delta_x
        y_lf_jittered = y_lf + delta_z

        # Create meshgrid of sampling points
        X_lf, Y_lf = np.meshgrid(x_lf_jittered, y_lf_jittered)
        points = np.column_stack([Y_lf.ravel(), X_lf.ravel()])

        # Interpolate
        sampled_values = interp_func(points).reshape(h_lf, w_lf)
        
        # Handle out-of-bounds (fill with nearest edge value or similar)
        # RegularGridInterpolator with bounds_error=False and fill_value=None 
        # extrapolates, which is okay for small jitters.
        
        material_grid_lf[:, :, c] = sampled_values

    return material_grid_lf, delta_x, delta_z


def compute_pga(accel: np.ndarray) -> float:
    """Compute Peak Ground Acceleration from acceleration time series.

    Args:
        accel: Acceleration time-series (T,) or (T, channels)

    Returns:
        PGA value (scalar for 1D, or max across channels for multi-channel)
    """
    if accel.ndim == 1:
        return float(np.max(np.abs(accel)))
    else:
        # For multi-channel, take max across all channels
        return float(np.max(np.abs(accel)))


def generate_dataset(
    data_dir: Path,
    n_simulations: int,
    split: str = "train",
    hx: float = 10.0,  # LF grid (coarse, e.g., 15x15 for 150m domain)
    Lx: float = 150.0,
    Lz: float = 150.0,
    duration: float = 100.0,  # Duration in seconds
    dt_lf: float = 0.2,  # LF time step (coarse, ~500 steps for 100s)
    dt_hf: float = 0.01,  # HF time step (fine, ~10,000 steps for 100s)
    start_idx: int = 0,
    run_hf: bool = False,  # Whether to also run HF for test/val
    hx_hf: float = 1.0,  # HF grid (fine, e.g., 150x150 for 150m domain)
    max_workers: int = 1,
    jitter_percent: float = 0.0, # Jitter percentage (0.0 to 0.5)
):
    """Generate dataset of simulations using seiskit.

    Generates:
    - LF solver: Coarse grid (hx), large time step (dt_lf) → ~500 steps
    - HF solver: Fine grid (hx_hf), small time step (dt_hf) → ~10,000 steps

    Args:
        data_dir: Base data directory
        n_simulations: Number of simulations to generate
        split: Dataset split ("train", "val", "test")
        hx: Element size for LF solver (m) - coarse grid
        Lx: Domain width (m)
        Lz: Domain height (m)
        duration: Simulation duration (s)
        dt_lf: Time step for LF solver (s) - coarse, e.g., 0.2s
        dt_hf: Time step for HF solver (s) - fine, e.g., 0.01s
        start_idx: Starting index for simulation numbering
        run_hf: Whether to also run HF solver (for test/val)
        hx_hf: Element size for HF solver (m) - fine grid
        max_workers: Number of parallel workers (not used yet, can be added with joblib)
        jitter_percent: Max grid jitter as fraction of hx (e.g. 0.5 = 50% of element size)
    """
    data_dir = Path(data_dir)
    materials_dir = data_dir / "materials"
    material_params_dir = data_dir / "material_params"
    base_motion_dir = data_dir / "base_motion"
    lf_output_dir = data_dir / "low_fidelity" / "output_accel"
    hf_output_dir = data_dir / "high_fidelity" / "output_accel"
    lf_pga_dir = data_dir / "low_fidelity" / "pga"
    hf_pga_dir = data_dir / "high_fidelity" / "pga"
    temp_output_dir = data_dir / "temp_outputs"

    # Create directories
    for d in [
        materials_dir,
        material_params_dir,
        base_motion_dir,
        lf_output_dir,
        lf_pga_dir,
        temp_output_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)
    if run_hf:
        hf_output_dir.mkdir(parents=True, exist_ok=True)
        hf_pga_dir.mkdir(parents=True, exist_ok=True)
    
    # Also create materials_hf directory
    materials_hf_dir = data_dir / "materials_hf"
    materials_hf_dir.mkdir(parents=True, exist_ok=True)

    # Parameter ranges
    rng = np.random.default_rng(42)

    # Material parameters
    Vs_profile_1D = np.array([180.0] * 8 + [1300.0] * 1)  # 8 layers of soil, 1 bedrock
    rH_min, rH_max = 10.0, 100.0
    CV_min, CV_max = 0.1, 0.4
    aHV = 10.0  # Fixed anisotropy ratio

    # Base motion parameters
    freq_min, freq_max = 3.0, 5.0
    t_shift = 0.8

    print(f"Generating {split} dataset with {n_simulations} simulations...")

    # Generate all simulations
    for i in tqdm(range(n_simulations), desc=f"Generating {split}"):
        sim_id = start_idx + i
        seed = 42 + sim_id  # Deterministic but different for each sim

        # Sample material parameters
        rH = rng.uniform(rH_min, rH_max)
        CV = rng.uniform(CV_min, CV_max)

        # Generate material grid at HF RESOLUTION (Ground Truth)
        # Note: using hx_hf instead of hx
        Vs_realization, x_coords, z_coords, h_mean = create_vs_realization(
            Vs_profile=Vs_profile_1D,
            Lx=150,
            Lx_variability=50,  # Central part with spatial variability
            Lz=150,
            dx=hx_hf, # Generate at HF resolution
            dz=hx_hf, # Generate at HF resolution
            rH=rH,
            aHV=aHV,
            CV=CV,
            seed=seed,
            dz_1D=5.0,
        )

        # Create density grid (correlated with Vs)
        density = np.full_like(Vs_realization, 2000.0)

        # Stack into (H, W, 2) format - This is the HF Ground Truth
        material_grid_hf = np.stack([Vs_realization, density], axis=2)
        
        # Save HF material grid (Ground Truth / Model Input)
        np.save(materials_hf_dir / f"sim_{sim_id:04d}.npy", material_grid_hf)

        # Create LF material grid by sampling HF grid with Jitter
        h_lf, w_lf = int(Lz / hx), int(Lx / hx)
        
        material_grid_lf, delta_x, delta_z = _sample_hf_to_lf_with_jitter(
            material_grid_hf, 
            h_lf, w_lf, 
            hx, hx_hf, 
            jitter_percent, 
            rng
        )

        # Save LF material grid (for LF solver)
        np.save(materials_dir / f"sim_{sim_id:04d}.npy", material_grid_lf)

        # Sample base motion parameters
        freq_x = rng.uniform(freq_min, freq_max)
        freq_y = rng.uniform(freq_min, freq_max)

        # Generate base motion using seiskit.compute_ricker (use HF dt for base motion)
        ricker_x = compute_ricker(freq_x, t_shift, duration, dt_hf)
        ricker_y = compute_ricker(freq_y, t_shift, duration, dt_hf)
        base_motion = np.stack([ricker_x, ricker_y], axis=1)  # (T, 2)

        # Save base motion
        np.save(base_motion_dir / f"sim_{sim_id:04d}.npy", base_motion)

        # Save parameters
        params = {
            "rH": float(rH),
            "aHV": float(aHV),
            "CV": float(CV),
            "seed": int(seed),
            "Lx": float(Lx),
            "Lz": float(Lz),
            "dx": float(hx),
            "dz": float(hx),
            "freq_x": float(freq_x),
            "freq_y": float(freq_y),
            "t_shift": float(t_shift),
            "jitter_percent": float(jitter_percent),
            "delta_x": float(delta_x),
            "delta_z": float(delta_z),
        }
        with open(material_params_dir / f"sim_{sim_id:04d}.json", "w") as f:
            json.dump(params, f, indent=2)

        # Run LF simulation (coarse grid, large time step)
        config_lf = AnalysisConfig(
            Ly=Lz,
            Lx=Lx,
            hx=hx,
            duration=duration,
            dt=dt_lf,  # Coarse time step (0.2s)
            motion_freq=freq_x,  # Use X component frequency for Ricker wave
            motion_t_shift=t_shift,
            damping_zeta=0.0075,
            damping_freqs=(0.75, 11.25),
            boundary_condition_type="1D",
            recorder_dofs=[1, 2],  # Record both DOFs
            recorder_quantity="accel",
            record_center_nodes=True,
        )

        # Build model data and run analysis using seiskit
        vs_data = material_grid[:, :, 0]
        rho_data = material_grid[:, :, 1]
        nu_data = np.full_like(vs_data, 0.3)  # Poisson's ratio

        run_id_lf = f"{split}_lf_{sim_id:04d}"
        try:
            model_data_lf = build_model_data(config_lf, vs_data, rho_data, nu_data)
            result_lf = run_opensees_analysis(
                config=config_lf,
                model_data=model_data_lf,
                run_id=run_id_lf,
                output_dir=str(temp_output_dir),
            )

            if result_lf.startswith("Finished"):
                # Load recorder outputs
                run_output_path_lf = temp_output_dir / run_id_lf
                output_accel_lf = load_recorder_outputs(
                    run_output_path_lf, run_id_lf, recorder_quantity="accel", Ly=Lz
                )
                if output_accel_lf is not None:
                    np.save(lf_output_dir / f"sim_{sim_id:04d}.npy", output_accel_lf)
                    # Extract PGA from surface acceleration (channel 2 = surf_ax)
                    pga_lf = compute_pga(output_accel_lf[:, 2])
                    np.save(lf_pga_dir / f"sim_{sim_id:04d}.npy", pga_lf)
            else:
                print(f"Warning: LF simulation {sim_id} failed: {result_lf}")
        except Exception as e:
            print(f"Error running LF simulation {sim_id}: {e}")

        # Run HF simulation if requested
        if run_hf:
            # Interpolate material grid to HF resolution
            h_lf, w_lf = material_grid.shape[:2]
            h_hf, w_hf = int(Lz / hx_hf), int(Lx / hx_hf)
            material_grid_hf = _interpolate_material_grid(
                material_grid, h_lf, w_lf, h_hf, w_hf
            )

            config_hf = AnalysisConfig(
                Ly=Lz,
                Lx=Lx,
                hx=hx_hf,
                duration=duration,
                dt=dt_hf,  # Fine time step (0.01s)
                motion_freq=freq_x,
                motion_t_shift=t_shift,
                damping_zeta=0.0075,
                damping_freqs=(0.75, 11.25),
                boundary_condition_type="1D",
                recorder_dofs=[1, 2],
                recorder_quantity="accel",
                record_center_nodes=True,
            )

            # Build model data and run analysis using seiskit
            vs_data_hf = material_grid_hf[:, :, 0]
            rho_data_hf = material_grid_hf[:, :, 1]
            nu_data_hf = np.full_like(vs_data_hf, 0.3)

            run_id_hf = f"{split}_hf_{sim_id:04d}"
            try:
                model_data_hf = build_model_data(
                    config_hf, vs_data_hf, rho_data_hf, nu_data_hf
                )
                result_hf = run_opensees_analysis(
                    config=config_hf,
                    model_data=model_data_hf,
                    run_id=run_id_hf,
                    output_dir=str(temp_output_dir),
                )

                if result_hf.startswith("Finished"):
                    # Load recorder outputs
                    run_output_path_hf = temp_output_dir / run_id_hf
                    output_accel_hf = load_recorder_outputs(
                        run_output_path_hf, run_id_hf, recorder_quantity="accel", Ly=Lz
                    )
                    if output_accel_hf is not None:
                        np.save(
                            hf_output_dir / f"sim_{sim_id:04d}.npy", output_accel_hf
                        )
                        # Extract PGA from surface acceleration (channel 2 = surf_ax)
                        pga_hf = compute_pga(output_accel_hf[:, 2])
                        np.save(hf_pga_dir / f"sim_{sim_id:04d}.npy", pga_hf)
                else:
                    print(f"Warning: HF simulation {sim_id} failed: {result_hf}")
            except Exception as e:
                print(f"Error running HF simulation {sim_id}: {e}")

    print(f"Completed generating {split} dataset")


if __name__ == "__main__":
    # Example usage
    data_dir = Path("data")

    # Generate training set (LF only)
    # LF: coarse grid (hx=10m → ~15x15), dt=0.2s → ~500 steps for 100s
    generate_dataset(
        data_dir=data_dir,
        n_simulations=1,  # Start with small number for testing
        split="train",
        start_idx=0,
        run_hf=False,
        dt_lf=0.2,  # Coarse time step
        dt_hf=0.01,  # Fine time step (for base motion)
        duration=25.0,  # 100s duration → ~500 LF steps, ~10,000 HF steps
        jitter_percent=0.5, # Enable Grid Jitter
    )

    # Generate validation set (LF only)
    generate_dataset(
        data_dir=data_dir,
        n_simulations=1,
        split="val",
        start_idx=100,
        run_hf=False,
        dt_lf=0.2,
        dt_hf=0.01,
        duration=25.0,
        jitter_percent=0.5, # Enable Grid Jitter
    )

    # Generate test set (LF + HF)
    generate_dataset(
        data_dir=data_dir,
        n_simulations=1,
        split="test",
        start_idx=120,
        run_hf=True,
        dt_lf=0.2,
        dt_hf=0.01,
        duration=25.0,
        jitter_percent=0.5, # Enable Grid Jitter
    )
