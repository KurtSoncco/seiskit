"""Unified SLURM-compatible data generation script.

This script generates:
1. Materials (material grids, parameters, base motions)
2. LF simulations (coarse grid, large time step)
3. Optionally HF simulations (fine grid, small time step)

Modes:
- "lf": Generates materials + runs LF simulation
- "both": Generates materials + runs LF + HF simulations (default)

Can be run as a single job or as an array job for parallel generation.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Add parent directory to path to allow importing emulator
sys.path.insert(0, str(Path(__file__).parent.parent))

from emulator.data_utils import load_recorder_outputs
from seiskit.analysis import run_opensees_analysis
from seiskit.builder import build_model_data
from seiskit.config import AnalysisConfig
from seiskit.gaussian_field import create_vs_realization


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
    """Compute Peak Ground Acceleration from acceleration time series."""
    if accel.ndim == 1:
        return float(np.max(np.abs(accel)))
    else:
        return float(np.max(np.abs(accel)))


def generate_materials(
    sim_id: int,
    data_dir: Path,
    hx: float = 10.0,
    Lx: float = 150.0,
    Lz: float = 150.0,
    hx_hf: float = 1.0,
    seed_base: int = 42,
    jitter_percent: float = 0.0,
) -> bool:
    """Generate material grid and parameters for a single simulation."""
    t0 = time.time()

    data_dir = Path(data_dir)
    materials_dir = data_dir / "materials"
    material_params_dir = data_dir / "material_params"
    materials_hf_dir = data_dir / "materials_hf"
    base_motion_dir = data_dir / "base_motion"

    # Create directories
    materials_dir.mkdir(parents=True, exist_ok=True)
    material_params_dir.mkdir(parents=True, exist_ok=True)
    materials_hf_dir.mkdir(parents=True, exist_ok=True)
    base_motion_dir.mkdir(parents=True, exist_ok=True)

    print(f"[generate_materials] Generating material data for sim_id={sim_id:04d}")

    # Parameter ranges (matching generate_data.py)
    rng = np.random.default_rng(seed_base + sim_id)
    Vs_profile_1D = np.array([180.0] * 8 + [1300.0] * 1)
    rH_min, rH_max = 10.0, 100.0
    CV_min, CV_max = 0.1, 0.4
    aHV = 10.0
    freq_min, freq_max = 3.0, 5.0
    t_shift = 0.8

    # Sample material parameters
    rH = rng.uniform(rH_min, rH_max)
    CV = rng.uniform(CV_min, CV_max)
    seed = seed_base + sim_id

    # Generate material grid at HF RESOLUTION (Ground Truth)
    Vs_realization, x_coords, z_coords, h_mean = create_vs_realization(
        Vs_profile=Vs_profile_1D,
        Lx=150,
        Lx_variability=50,
        Lz=150,
        dx=hx_hf, # Generate at HF resolution
        dz=hx_hf, # Generate at HF resolution
        rH=rH,
        aHV=aHV,
        CV=CV,
        seed=seed,
        dz_1D=5.0,
    )

    density = np.full_like(Vs_realization, 2000.0)
    material_grid_hf = np.stack([Vs_realization, density], axis=2)

    # Save HF material grid (Ground Truth)
    material_hf_file = materials_hf_dir / f"sim_{sim_id:04d}.npy"
    np.save(material_hf_file, material_grid_hf)

    # Create LF material grid by sampling HF grid with Jitter
    h_lf, w_lf = int(Lz / hx), int(Lx / hx)
    
    material_grid_lf, delta_x, delta_z = _sample_hf_to_lf_with_jitter(
        material_grid_hf, 
        h_lf, w_lf, 
        hx, hx_hf, 
        jitter_percent, 
        rng
    )

    # Save LF material grid
    material_file = materials_dir / f"sim_{sim_id:04d}.npy"
    np.save(material_file, material_grid_lf)

    # Sample base motion parameters
    freq_x = rng.uniform(freq_min, freq_max)
    freq_y = rng.uniform(freq_min, freq_max)

    # Generate base motion (using HF dt for consistency)
    from seiskit.utils import compute_ricker

    duration = 25.0  # Default duration for base motion
    dt_hf = 0.01  # Default HF dt
    ricker_x = compute_ricker(freq_x, t_shift, duration, dt_hf)
    ricker_y = compute_ricker(freq_y, t_shift, duration, dt_hf)
    base_motion = np.stack([ricker_x, ricker_y], axis=1)

    # Save base motion
    base_motion_file = base_motion_dir / f"sim_{sim_id:04d}.npy"
    np.save(base_motion_file, base_motion)

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
    params_file = material_params_dir / f"sim_{sim_id:04d}.json"
    with open(params_file, "w") as f:
        json.dump(params, f, indent=2)

    elapsed = time.time() - t0
    print(
        f"[generate_materials] ✓ Successfully generated materials for sim_id={sim_id:04d} "
        f"in {elapsed:.2f}s"
    )
    return True


def run_lf_simulation(
    sim_id: int,
    data_dir: Path,
    hx: float = 10.0,
    Lx: float = 150.0,
    Lz: float = 150.0,
    duration: float = 25.0,
    dt_lf: float = 0.2,
) -> bool:
    """Run a single LF simulation for the given simulation ID."""
    t0 = time.time()

    data_dir = Path(data_dir)
    materials_dir = data_dir / "materials"
    material_params_dir = data_dir / "material_params"
    lf_output_dir = data_dir / "low_fidelity" / "output_accel"
    lf_pga_dir = data_dir / "low_fidelity" / "pga"
    temp_output_dir = data_dir / "temp_outputs"

    # Create output directories
    lf_output_dir.mkdir(parents=True, exist_ok=True)
    lf_pga_dir.mkdir(parents=True, exist_ok=True)
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run_lf_simulation] Starting LF simulation for sim_id={sim_id:04d}")

    # Load existing material grid
    material_file = materials_dir / f"sim_{sim_id:04d}.npy"
    if not material_file.exists():
        raise FileNotFoundError(
            f"Material file not found: {material_file.resolve()}\n"
            f"Make sure materials have been generated first."
        )

    material_grid = np.load(material_file)

    # Load material parameters
    params_file = material_params_dir / f"sim_{sim_id:04d}.json"
    if not params_file.exists():
        raise FileNotFoundError(
            f"Parameters file not found: {params_file.resolve()}\n"
            f"Make sure materials have been generated first."
        )

    with open(params_file, "r") as f:
        params = json.load(f)

    freq_x = params.get("freq_x", 3.0)
    t_shift = params.get("t_shift", 0.8)

    # Build LF analysis config
    config_lf = AnalysisConfig(
        Ly=Lz,
        Lx=Lx,
        hx=hx,
        duration=duration,
        dt=dt_lf,  # Coarse time step (0.2s)
        motion_freq=freq_x,
        motion_t_shift=t_shift,
        damping_zeta=0.0075,
        damping_freqs=(0.75, 11.25),
        boundary_condition_type="1D",
        recorder_dofs=[1, 2],
        recorder_quantity="accel",
        record_center_nodes=True,
    )

    # Build model data and run analysis
    vs_data = material_grid[:, :, 0]
    rho_data = material_grid[:, :, 1]
    nu_data = np.full_like(vs_data, 0.3)

    run_id_lf = f"test_lf_{sim_id:04d}"

    print("[run_lf_simulation] Building model and running OpenSees...")
    print(f"  Run ID: {run_id_lf}")
    print(f"  Grid: {material_grid.shape[0]}x{material_grid.shape[1]} ({hx}m elements)")
    print(f"  Time step: {dt_lf}s, Duration: {duration}s")

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
                # Save acceleration output
                np.save(lf_output_dir / f"sim_{sim_id:04d}.npy", output_accel_lf)
                print(
                    f"[run_lf_simulation] Saved acceleration to {lf_output_dir / f'sim_{sim_id:04d}.npy'}"
                )

                # Extract PGA from surface acceleration (channel 2 = surf_ax)
                pga_lf = compute_pga(output_accel_lf[:, 2])
                np.save(lf_pga_dir / f"sim_{sim_id:04d}.npy", pga_lf)
                print(
                    f"[run_lf_simulation] Saved PGA ({pga_lf:.6f} m/s²) to {lf_pga_dir / f'sim_{sim_id:04d}.npy'}"
                )

                elapsed = time.time() - t0
                print(
                    f"[run_lf_simulation] ✓ Successfully completed sim_id={sim_id:04d} "
                    f"in {elapsed:.1f}s"
                )
                return True
            else:
                print(f"[run_lf_simulation] ✗ Warning: No acceleration data found for {run_id_lf}")
                return False
        else:
            print(f"[run_lf_simulation] ✗ Simulation failed: {result_lf}")
            return False
    except Exception as e:
        print(f"[run_lf_simulation] ✗ Error running LF simulation {sim_id}: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_hf_simulation(
    sim_id: int,
    data_dir: Path,
    hx: float = 10.0,
    hx_hf: float = 1.0,
    Lx: float = 150.0,
    Lz: float = 150.0,
    duration: float = 25.0,
    dt_hf: float = 0.01,
) -> bool:
    """Run a single HF simulation for the given simulation ID."""
    t0 = time.time()

    data_dir = Path(data_dir)
    materials_dir = data_dir / "materials"
    material_params_dir = data_dir / "material_params"
    hf_output_dir = data_dir / "high_fidelity" / "output_accel"
    hf_pga_dir = data_dir / "high_fidelity" / "pga"
    temp_output_dir = data_dir / "temp_outputs"

    # Create output directories
    hf_output_dir.mkdir(parents=True, exist_ok=True)
    hf_pga_dir.mkdir(parents=True, exist_ok=True)
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run_hf_simulation] Starting HF simulation for sim_id={sim_id:04d}")

    # Load existing material grid (LF resolution)
    material_file = materials_dir / f"sim_{sim_id:04d}.npy"
    if not material_file.exists():
        raise FileNotFoundError(
            f"Material file not found: {material_file.resolve()}\n"
            f"Current working directory: {os.getcwd()}\n"
            f"Materials directory: {materials_dir.resolve()}\n"
            f"Make sure materials have been generated first."
        )

    material_grid = np.load(material_file)

    # Load material parameters
    params_file = material_params_dir / f"sim_{sim_id:04d}.json"
    if not params_file.exists():
        raise FileNotFoundError(
            f"Parameters file not found: {params_file.resolve()}\n"
            f"Make sure materials have been generated first."
        )

    with open(params_file, "r") as f:
        params = json.load(f)

    freq_x = params.get("freq_x", 3.0)
    t_shift = params.get("t_shift", 0.8)

    # Interpolate material grid to HF resolution
    h_lf, w_lf = material_grid.shape[:2]
    h_hf, w_hf = int(Lz / hx_hf), int(Lx / hx_hf)
    material_grid_hf = _interpolate_material_grid(material_grid, h_lf, w_lf, h_hf, w_hf)

    # Build HF analysis config
    config_hf = AnalysisConfig(
        Ly=Lz,
        Lx=Lx,
        hx=hx_hf,
        duration=duration,
        dt=dt_hf,
        motion_freq=freq_x,
        motion_t_shift=t_shift,
        damping_zeta=0.0075,
        damping_freqs=(0.75, 11.25),
        boundary_condition_type="1D",
        recorder_dofs=[1, 2],
        recorder_quantity="accel",
        record_center_nodes=True,
    )

    # Build model data and run analysis
    vs_data_hf = material_grid_hf[:, :, 0]
    rho_data_hf = material_grid_hf[:, :, 1]
    nu_data_hf = np.full_like(vs_data_hf, 0.3)

    run_id_hf = f"test_hf_{sim_id:04d}"

    print("[run_hf_simulation] Building model and running OpenSees...")
    print(f"  Run ID: {run_id_hf}")
    print(f"  Grid: {h_hf}x{w_hf} ({hx_hf}m elements)")
    print(f"  Time step: {dt_hf}s, Duration: {duration}s")

    try:
        model_data_hf = build_model_data(config_hf, vs_data_hf, rho_data_hf, nu_data_hf)
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
                # Save acceleration output
                np.save(hf_output_dir / f"sim_{sim_id:04d}.npy", output_accel_hf)
                print(
                    f"[run_hf_simulation] Saved acceleration to {hf_output_dir / f'sim_{sim_id:04d}.npy'}"
                )

                # Extract PGA from surface acceleration (channel 2 = surf_ax)
                pga_hf = compute_pga(output_accel_hf[:, 2])
                np.save(hf_pga_dir / f"sim_{sim_id:04d}.npy", pga_hf)
                print(
                    f"[run_hf_simulation] Saved PGA ({pga_hf:.6f} m/s²) to {hf_pga_dir / f'sim_{sim_id:04d}.npy'}"
                )

                elapsed = time.time() - t0
                print(
                    f"[run_hf_simulation] ✓ Successfully completed sim_id={sim_id:04d} "
                    f"in {elapsed:.1f}s"
                )
                return True
            else:
                print(f"[run_hf_simulation] ✗ Warning: No acceleration data found for {run_id_hf}")
                return False
        else:
            print(f"[run_hf_simulation] ✗ Simulation failed: {result_hf}")
            return False
    except Exception as e:
        print(f"[run_hf_simulation] ✗ Error running HF simulation {sim_id}: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main entry point for SLURM array job."""
    parser = argparse.ArgumentParser(
        description="Unified data generation: materials and/or HF simulations"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Base data directory (default: data)",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Array index (overrides SLURM_ARRAY_TASK_ID if provided)",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=1000,
        help="Starting index for simulation IDs (default: 1000)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["lf", "both"],
        default="both",
        help="Generation mode: 'lf' (materials+LF simulation), 'both' (materials+LF+HF simulation, default)",
    )
    parser.add_argument(
        "--hx",
        type=float,
        default=10.0,
        help="LF element size (m) (default: 10.0)",
    )
    parser.add_argument(
        "--hx_hf",
        type=float,
        default=1.0,
        help="HF element size (m) (default: 1.0)",
    )
    parser.add_argument(
        "--Lx",
        type=float,
        default=150.0,
        help="Domain width (m) (default: 150.0)",
    )
    parser.add_argument(
        "--Lz",
        type=float,
        default=150.0,
        help="Domain height (m) (default: 150.0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=25.0,
        help="Simulation duration (s) (default: 25.0)",
    )
    parser.add_argument(
        "--dt_lf",
        type=float,
        default=0.2,
        help="LF time step (s) (default: 0.2)",
    )
    parser.add_argument(
        "--dt_hf",
        type=float,
        default=0.01,
        help="HF time step (s) (default: 0.01)",
    )
    parser.add_argument(
        "--seed_base",
        type=int,
        default=42,
        help="Base seed for random number generation (default: 42)",
    )

    args = parser.parse_args()

    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Get array index
    idx = args.index
    if idx is None:
        env_idx = os.getenv("SLURM_ARRAY_TASK_ID")
        if env_idx is not None and env_idx != "":
            try:
                idx = int(env_idx)
            except ValueError:
                print(
                    f"Invalid SLURM_ARRAY_TASK_ID={env_idx!r}",
                    file=sys.stderr,
                )
                sys.exit(2)
        else:
            print(
                "Error: No array index provided. Use --index or set SLURM_ARRAY_TASK_ID",
                file=sys.stderr,
            )
            sys.exit(1)

    # Calculate simulation ID from array index
    sim_id = args.start_idx + idx

    print("=" * 70)
    print("Unified Data Generation - SLURM Array Job")
    print("=" * 70)
    print(f"Array Index: {idx}")
    print(f"Simulation ID: {sim_id:04d}")
    print(f"Mode: {args.mode}")
    print(f"Data Directory: {args.data_dir}")
    print("=" * 70)

    program_start = time.time()
    success = True

    # Always generate materials first
    print("\n[Step 1] Generating materials...")
    success = generate_materials(
        sim_id=sim_id,
        data_dir=Path(args.data_dir),
        hx=args.hx,
        Lx=args.Lx,
        Lz=args.Lz,
        hx_hf=args.hx_hf,
        seed_base=args.seed_base,
    )
    if not success:
        print("[ERROR] Material generation failed")
        sys.exit(1)

    # Always run LF simulation
    print("\n[Step 2] Running LF simulation...")
    success = run_lf_simulation(
        sim_id=sim_id,
        data_dir=Path(args.data_dir),
        hx=args.hx,
        Lx=args.Lx,
        Lz=args.Lz,
        duration=args.duration,
        dt_lf=args.dt_lf,
    )
    if not success:
        print("[ERROR] LF simulation failed")
        sys.exit(1)

    # Run HF simulation only if mode is "both"
    if args.mode == "both":
        print("\n[Step 3] Running HF simulation...")
        success = run_hf_simulation(
            sim_id=sim_id,
            data_dir=Path(args.data_dir),
            hx=args.hx,
            hx_hf=args.hx_hf,
            Lx=args.Lx,
            Lz=args.Lz,
            duration=args.duration,
            dt_hf=args.dt_hf,
        )
        if not success:
            print("[ERROR] HF simulation failed")
            sys.exit(1)

    total_time = time.time() - program_start
    print("=" * 70)
    print(f"Total wall time: {total_time:.1f}s")
    print("=" * 70)

    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

