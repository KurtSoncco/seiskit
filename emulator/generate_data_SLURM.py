"""SLURM-compatible data generation script for HF simulations.

This script runs a single HF simulation based on SLURM_ARRAY_TASK_ID.
It assumes that LF data (materials, base motion, parameters) have already been generated.
Only the HF OpenSees simulation is run here.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add parent directory to path to allow importing emulator
sys.path.insert(0, str(Path(__file__).parent.parent))

from scipy.interpolate import RegularGridInterpolator

from emulator.data_utils import load_recorder_outputs
from seiskit.analysis import run_opensees_analysis
from seiskit.builder import build_model_data
from seiskit.config import AnalysisConfig


def _interpolate_material_grid(
    material_grid: np.ndarray, h_lf: int, w_lf: int, h_hf: int, w_hf: int
) -> np.ndarray:
    """Interpolate material grid from LF to HF resolution.

    Args:
        material_grid: Material grid (H_lf, W_lf, 2) = [Vs, density]
        h_lf, w_lf: LF grid dimensions
        h_hf, w_hf: HF grid dimensions

    Returns:
        Interpolated material grid (H_hf, W_hf, 2)
    """
    material_grid_hf = np.zeros((h_hf, w_hf, 2))
    for c in range(2):
        # Create coordinate grids for LF
        x_lf = np.linspace(0, 1, w_lf)
        y_lf = np.linspace(0, 1, h_lf)

        # Create interpolator
        interp_func = RegularGridInterpolator(
            (y_lf, x_lf), material_grid[:, :, c], method="linear"
        )

        # Create coordinate grids for HF
        x_hf = np.linspace(0, 1, w_hf)
        y_hf = np.linspace(0, 1, h_hf)
        X_hf, Y_hf = np.meshgrid(x_hf, y_hf)

        # Interpolate
        points = np.column_stack([Y_hf.ravel(), X_hf.ravel()])
        material_grid_hf[:, :, c] = interp_func(points).reshape(h_hf, w_hf)

    return material_grid_hf


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


def _fmt_hms(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    seconds = int(round(seconds))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def run_hf_simulation(
    sim_id: int,
    data_dir: Path,
    hx: float = 10.0,  # LF grid size (for reading existing data)
    hx_hf: float = 1.0,  # HF grid size
    Lx: float = 150.0,
    Lz: float = 150.0,
    duration: float = 100.0,
    dt_hf: float = 0.01,  # HF time step
):
    """
    Run a single HF simulation for the given simulation ID.

    Args:
        sim_id: Simulation ID (0-indexed)
        data_dir: Base data directory
        hx: LF element size (for reading existing material grid)
        hx_hf: HF element size
        Lx: Domain width (m)
        Lz: Domain height (m)
        duration: Simulation duration (s)
        dt_hf: HF time step (s)
    """
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
            f"Material file not found: {material_file}. "
            f"Make sure LF data generation has been completed first."
        )

    material_grid = np.load(material_file)
    print(f"[run_hf_simulation] Loaded material grid from {material_file}")
    print(f"  Material grid shape: {material_grid.shape}")

    # Load material parameters (for frequency and t_shift)
    params_file = material_params_dir / f"sim_{sim_id:04d}.json"
    if not params_file.exists():
        raise FileNotFoundError(
            f"Parameters file not found: {params_file}. "
            f"Make sure LF data generation has been completed first."
        )

    import json

    with open(params_file, "r") as f:
        params = json.load(f)

    freq_x = params.get("freq_x", 3.0)
    t_shift = params.get("t_shift", 0.8)

    # Interpolate material grid to HF resolution
    h_lf, w_lf = material_grid.shape[:2]
    h_hf, w_hf = int(Lz / hx_hf), int(Lx / hx_hf)
    print(
        f"[run_hf_simulation] Interpolating from LF ({h_lf}x{w_lf}) to HF ({h_hf}x{w_hf})"
    )

    material_grid_hf = _interpolate_material_grid(material_grid, h_lf, w_lf, h_hf, w_hf)

    # Build HF analysis config
    config_hf = AnalysisConfig(
        Ly=Lz,
        Lx=Lx,
        hx=hx_hf,
        duration=duration,
        dt=dt_hf,  # Fine time step (0.01s)
        motion_freq=freq_x,
        motion_t_shift=t_shift,
        damping_zeta=0.0075,
        damping_freqs=(0.75, 2.25),
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
    print(f"  Steps: {int(duration / dt_hf)}")

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
                    f"in {_fmt_hms(elapsed)}"
                )
                return True
            else:
                print(
                    f"[run_hf_simulation] ✗ Warning: No acceleration data found for {run_id_hf}"
                )
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
        description="Run a single HF simulation for SLURM array job"
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
        "--test_start_idx",
        type=int,
        default=1000,
        help="Starting index for test data (default: 1000)",
    )
    parser.add_argument(
        "--hx",
        type=float,
        default=10.0,
        help="LF element size (m) - for reading existing data (default: 10.0)",
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
        "--dt_hf",
        type=float,
        default=0.01,
        help="HF time step (s) (default: 0.01)",
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
    sim_id = args.test_start_idx + idx

    print("=" * 70)
    print("HF Simulation - SLURM Array Job")
    print("=" * 70)
    print(f"Array Index: {idx}")
    print(f"Simulation ID: {sim_id:04d}")
    print(f"Data Directory: {args.data_dir}")
    print("=" * 70)

    program_start = time.time()

    # Run HF simulation
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

    total_time = time.time() - program_start
    print("=" * 70)
    print(f"Total wall time: {_fmt_hms(total_time)}")
    print("=" * 70)

    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
