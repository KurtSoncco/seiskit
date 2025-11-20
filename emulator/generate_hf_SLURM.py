"""SLURM-compatible script to generate HF datasets using array jobs.

This script runs a single HF simulation based on SLURM_ARRAY_TASK_ID.
Each array task processes one simulation ID from the test set.

Usage:
    # Run with explicit index (for testing)
    python generate_hf_SLURM.py --index 0
    
    # Run with SLURM array (automatically reads SLURM_ARRAY_TASK_ID)
    sbatch job_generate_hf.sh
"""

import argparse
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
from seiskit.utils import compute_ricker


def _fmt_hms(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    seconds = int(round(seconds))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def _interpolate_material_grid(
    material_grid: np.ndarray, h_lf: int, w_lf: int, h_hf: int, w_hf: int
) -> np.ndarray:
    """Interpolate material grid from LF to HF resolution."""
    material_grid_hf = np.zeros((h_hf, w_hf, 2))
    for c in range(2):
        x_lf = np.linspace(0, 1, w_lf)
        y_lf = np.linspace(0, 1, h_lf)
        interp_func = RegularGridInterpolator(
            (y_lf, x_lf), material_grid[:, :, c], method="linear"
        )
        x_hf = np.linspace(0, 1, w_hf)
        y_hf = np.linspace(0, 1, h_hf)
        X_hf, Y_hf = np.meshgrid(x_hf, y_hf)
        points = np.column_stack([Y_hf.ravel(), X_hf.ravel()])
        material_grid_hf[:, :, c] = interp_func(points).reshape(h_hf, w_hf)
    return material_grid_hf


def compute_pga(accel: np.ndarray) -> float:
    """Compute Peak Ground Acceleration from acceleration time series."""
    if accel.ndim == 1:
        return float(np.max(np.abs(accel)))
    else:
        return float(np.max(np.abs(accel)))


def run_hf_simulation(
    array_index: int,
    data_dir: Path,
    test_start_idx: int = 1200,  # Default: after train (1000) + val (100) + some buffer
    hx: float = 10.0,
    hx_hf: float = 1.0,
    Lx: float = 150.0,
    Lz: float = 150.0,
    duration: float = 25.0,
    dt_lf: float = 0.2,
    dt_hf: float = 0.01,
):
    """
    Run a single HF simulation for the test set.
    
    Args:
        array_index: Array task index (0-based)
        data_dir: Base data directory
        test_start_idx: Starting index for test simulations
        hx: LF element size (m)
        hx_hf: HF element size (m)
        Lx: Domain width (m)
        Lz: Domain height (m)
        duration: Simulation duration (s)
        dt_lf: LF time step (s) - used for base motion generation
        dt_hf: HF time step (s)
    """
    t0 = time.time()
    
    # Map array index to simulation ID
    sim_id = test_start_idx + array_index
    
    print("=" * 60)
    print(f"HF Simulation Task")
    print("=" * 60)
    print(f"Array Index: {array_index}")
    print(f"Simulation ID: {sim_id:04d}")
    print(f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()
    
    # Setup directories
    data_dir = Path(data_dir)
    materials_dir = data_dir / "materials"
    material_params_dir = data_dir / "material_params"
    base_motion_dir = data_dir / "base_motion"
    hf_output_dir = data_dir / "high_fidelity" / "output_accel"
    hf_pga_dir = data_dir / "high_fidelity" / "pga"
    temp_output_dir = data_dir / "temp_outputs"
    
    # Create output directories
    hf_output_dir.mkdir(parents=True, exist_ok=True)
    hf_pga_dir.mkdir(parents=True, exist_ok=True)
    temp_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if materials and base motion already exist (from LF generation)
    material_file = materials_dir / f"sim_{sim_id:04d}.npy"
    base_motion_file = base_motion_dir / f"sim_{sim_id:04d}.npy"
    params_file = material_params_dir / f"sim_{sim_id:04d}.json"
    
    # If materials don't exist, generate them
    if not material_file.exists():
        print(f"Generating materials for sim_{sim_id:04d}...")
        
        # Parameter ranges (same as generate_data.py)
        rng = np.random.default_rng(42 + sim_id)  # Deterministic seed
        
        Vs_profile_1D = np.array([180.0] * 8 + [1300.0] * 1)
        rH_min, rH_max = 10.0, 100.0
        CV_min, CV_max = 0.1, 0.4
        aHV = 10.0
        freq_min, freq_max = 3.0, 5.0
        t_shift = 0.8
        
        # Sample material parameters
        rH = rng.uniform(rH_min, rH_max)
        CV = rng.uniform(CV_min, CV_max)
        seed = 42 + sim_id
        
        # Generate material grid
        from seiskit.gaussian_field import create_vs_realization
        
        Vs_realization, x_coords, z_coords, h_mean = create_vs_realization(
            Vs_profile=Vs_profile_1D,
            Lx=150,
            Lx_variability=50,
            Lz=150,
            dx=hx,
            dz=hx,
            rH=rH,
            aHV=aHV,
            CV=CV,
            seed=seed,
            dz_1D=5.0,
        )
        
        density = np.full_like(Vs_realization, 2000.0)
        material_grid = np.stack([Vs_realization, density], axis=2)
        
        # Save material grid (LF resolution)
        materials_dir.mkdir(parents=True, exist_ok=True)
        np.save(material_file, material_grid)
        
        # Sample base motion parameters
        freq_x = rng.uniform(freq_min, freq_max)
        freq_y = rng.uniform(freq_min, freq_max)
        
        # Generate base motion
        ricker_x = compute_ricker(freq_x, t_shift, duration, dt_hf)
        ricker_y = compute_ricker(freq_y, t_shift, duration, dt_hf)
        base_motion = np.stack([ricker_x, ricker_y], axis=1)
        
        # Save base motion
        base_motion_dir.mkdir(parents=True, exist_ok=True)
        np.save(base_motion_file, base_motion)
        
        # Save parameters
        import json
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
        }
        material_params_dir.mkdir(parents=True, exist_ok=True)
        with open(params_file, "w") as f:
            json.dump(params, f, indent=2)
        
        print(f"Generated and saved materials for sim_{sim_id:04d}")
    else:
        print(f"Using existing materials from {material_file}")
        # Load parameters to get motion frequency
        import json
        with open(params_file, "r") as f:
            params = json.load(f)
        freq_x = params["freq_x"]
        t_shift = params["t_shift"]
    
    # Load material grid (LF resolution)
    material_grid = np.load(material_file)
    h_lf, w_lf = material_grid.shape[:2]
    
    # Interpolate to HF resolution
    print(f"Interpolating material grid from LF ({h_lf}x{w_lf}) to HF...")
    h_hf, w_hf = int(Lz / hx_hf), int(Lx / hx_hf)
    material_grid_hf = _interpolate_material_grid(
        material_grid, h_lf, w_lf, h_hf, w_hf
    )
    print(f"HF grid size: {h_hf}x{w_hf}")
    
    # Setup HF analysis config
    config_hf = AnalysisConfig(
        Ly=Lz,
        Lx=Lx,
        hx=hx_hf,
        duration=duration,
        dt=dt_hf,
        motion_freq=freq_x,
        motion_t_shift=t_shift,
        damping_zeta=0.0075,
        damping_freqs=(0.75, 2.25),
        boundary_condition_type="1D",
        recorder_dofs=[1, 2],
        recorder_quantity="accel",
        record_center_nodes=True,
    )
    
    # Build model data
    vs_data_hf = material_grid_hf[:, :, 0]
    rho_data_hf = material_grid_hf[:, :, 1]
    nu_data_hf = np.full_like(vs_data_hf, 0.3)
    
    run_id_hf = f"test_hf_{sim_id:04d}"
    
    print(f"Running HF OpenSees analysis for sim_{sim_id:04d}...")
    print(f"  Grid: {h_hf}x{w_hf} elements")
    print(f"  Time step: {dt_hf} s")
    print(f"  Duration: {duration} s")
    print(f"  Total steps: {int(duration / dt_hf)}")
    
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
                run_output_path_hf, run_id_hf, recorder_quantity="accel"
            )
            
            if output_accel_hf is not None:
                # Save acceleration output
                np.save(hf_output_dir / f"sim_{sim_id:04d}.npy", output_accel_hf)
                
                # Extract and save PGA
                pga_hf = compute_pga(output_accel_hf[:, 2])  # Channel 2 = surface acceleration
                np.save(hf_pga_dir / f"sim_{sim_id:04d}.npy", pga_hf)
                
                elapsed = time.time() - t0
                print()
                print("=" * 60)
                print(f"✓ Successfully completed sim_{sim_id:04d}")
                print(f"  PGA: {pga_hf:.6f} m/s²")
                print(f"  Wall time: {_fmt_hms(elapsed)}")
                print("=" * 60)
                return 0
            else:
                print(f"✗ Error: Failed to load recorder outputs for sim_{sim_id:04d}")
                return 1
        else:
            print(f"✗ Error: HF simulation failed: {result_hf}")
            return 1
            
    except Exception as e:
        print(f"✗ Error running HF simulation {sim_id}: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Generate HF dataset using SLURM array jobs"
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Array index (overrides SLURM_ARRAY_TASK_ID if provided)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Data directory",
    )
    parser.add_argument(
        "--test_start_idx",
        type=int,
        default=1200,
        help="Starting index for test simulations (default: 1200)",
    )
    parser.add_argument(
        "--hx",
        type=float,
        default=10.0,
        help="LF element size (m)",
    )
    parser.add_argument(
        "--hx_hf",
        type=float,
        default=1.0,
        help="HF element size (m)",
    )
    parser.add_argument(
        "--Lx",
        type=float,
        default=150.0,
        help="Domain width (m)",
    )
    parser.add_argument(
        "--Lz",
        type=float,
        default=150.0,
        help="Domain height (m)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=25.0,
        help="Simulation duration (s)",
    )
    parser.add_argument(
        "--dt_hf",
        type=float,
        default=0.01,
        help="HF time step (s)",
    )
    
    args = parser.parse_args()
    
    # Get array index from argument or environment variable
    array_index = args.index
    if array_index is None:
        env_idx = os.getenv("SLURM_ARRAY_TASK_ID")
        if env_idx is not None and env_idx != "":
            try:
                array_index = int(env_idx)
            except ValueError:
                print(f"Invalid SLURM_ARRAY_TASK_ID={env_idx!r}", file=sys.stderr)
                sys.exit(2)
        else:
            print("Error: No array index provided. Use --index or set SLURM_ARRAY_TASK_ID", file=sys.stderr)
            sys.exit(1)
    
    # Run the HF simulation
    exit_code = run_hf_simulation(
        array_index=array_index,
        data_dir=Path(args.data_dir),
        test_start_idx=args.test_start_idx,
        hx=args.hx,
        hx_hf=args.hx_hf,
        Lx=args.Lx,
        Lz=args.Lz,
        duration=args.duration,
        dt_hf=args.dt_hf,
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

