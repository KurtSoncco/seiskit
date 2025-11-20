"""SLURM-compatible script for generating LF material grids and parameters.

This script generates only the material grids and parameters needed for HF generation.
It does NOT run LF or HF simulations - those are handled separately.

Can be run as a single job or as an array job for parallel generation.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add parent directory to path to allow importing emiskit
sys.path.insert(0, str(Path(__file__).parent.parent))

from seiskit.gaussian_field import create_vs_realization


def generate_material_data(
    sim_id: int,
    data_dir: Path,
    hx: float = 10.0,
    Lx: float = 150.0,
    Lz: float = 150.0,
    hx_hf: float = 1.0,
    seed_base: int = 42,
):
    """Generate material grid and parameters for a single simulation.

    Args:
        sim_id: Simulation ID (0-indexed)
        data_dir: Base data directory
        hx: LF element size (m)
        Lx: Domain width (m)
        Lz: Domain height (m)
        hx_hf: HF element size (m) - for generating HF-resolution input
        seed_base: Base seed for random number generation
    """
    t0 = time.time()

    data_dir = Path(data_dir)
    materials_dir = data_dir / "materials"
    material_params_dir = data_dir / "material_params"
    materials_hf_dir = data_dir / "materials_hf"

    # Create directories
    materials_dir.mkdir(parents=True, exist_ok=True)
    material_params_dir.mkdir(parents=True, exist_ok=True)
    materials_hf_dir.mkdir(parents=True, exist_ok=True)

    print(f"[generate_material_data] Generating material data for sim_id={sim_id:04d}")

    # Parameter ranges (matching generate_data.py)
    rng = np.random.default_rng(seed_base + sim_id)

    # Material parameters
    Vs_profile_1D = np.array([180.0] * 8 + [1300.0] * 1)  # 8 layers of soil, 1 bedrock
    rH_min, rH_max = 10.0, 100.0
    CV_min, CV_max = 0.1, 0.4
    aHV = 10.0  # Fixed anisotropy ratio

    # Base motion parameters (needed for params file, but motion not generated here)
    freq_min, freq_max = 3.0, 5.0
    t_shift = 0.8

    # Sample material parameters
    rH = rng.uniform(rH_min, rH_max)
    CV = rng.uniform(CV_min, CV_max)
    seed = seed_base + sim_id

    # Generate material grid using seiskit.create_vs_realization
    Vs_realization, x_coords, z_coords, h_mean = create_vs_realization(
        Vs_profile=Vs_profile_1D,
        Lx=150,
        Lx_variability=50,  # Central part with spatial variability
        Lz=150,
        dx=hx,
        dz=hx,
        rH=rH,
        aHV=aHV,
        CV=CV,
        seed=seed,
        dz_1D=5.0,
    )

    # Create density grid (correlated with Vs)
    density = np.full_like(Vs_realization, 2000.0)

    # Stack into (H, W, 2) format: [Vs, density]
    material_grid = np.stack([Vs_realization, density], axis=2)

    # Save material grid (LF resolution for LF solver)
    material_file = materials_dir / f"sim_{sim_id:04d}.npy"
    np.save(material_file, material_grid)
    print(f"[generate_material_data] Saved LF material grid: {material_file}")
    print(f"  Material grid shape: {material_grid.shape}")

    # For model input, we need HF resolution (150x150)
    # Interpolate to HF resolution for input
    from scipy.interpolate import RegularGridInterpolator

    h_lf, w_lf = material_grid.shape[:2]
    h_hf, w_hf = int(Lz / hx_hf), int(Lx / hx_hf)

    material_grid_hf_input = np.zeros((h_hf, w_hf, 2))
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
        material_grid_hf_input[:, :, c] = interp_func(points).reshape(h_hf, w_hf)

    # Save HF material grid as input (for model)
    material_hf_file = materials_hf_dir / f"sim_{sim_id:04d}.npy"
    np.save(material_hf_file, material_grid_hf_input)
    print(f"[generate_material_data] Saved HF material grid: {material_hf_file}")
    print(f"  HF material grid shape: {material_grid_hf_input.shape}")

    # Sample base motion parameters (for params file)
    freq_x = rng.uniform(freq_min, freq_max)
    freq_y = rng.uniform(freq_min, freq_max)

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
    }
    params_file = material_params_dir / f"sim_{sim_id:04d}.json"
    with open(params_file, "w") as f:
        json.dump(params, f, indent=2)
    print(f"[generate_material_data] Saved parameters: {params_file}")

    elapsed = time.time() - t0
    print(
        f"[generate_material_data] ✓ Successfully generated material data for sim_id={sim_id:04d} "
        f"in {elapsed:.2f}s"
    )

    return True


def main():
    """Main entry point for SLURM array job."""
    parser = argparse.ArgumentParser(
        description="Generate LF material grids and parameters for SLURM array job"
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
        default=0,
        help="Starting index for simulation IDs (default: 0)",
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
    print("LF Material Generation - SLURM Array Job")
    print("=" * 70)
    print(f"Array Index: {idx}")
    print(f"Simulation ID: {sim_id:04d}")
    print(f"Data Directory: {args.data_dir}")
    print("=" * 70)

    program_start = time.time()

    # Generate material data
    success = generate_material_data(
        sim_id=sim_id,
        data_dir=Path(args.data_dir),
        hx=args.hx,
        Lx=args.Lx,
        Lz=args.Lz,
        hx_hf=args.hx_hf,
        seed_base=args.seed_base,
    )

    total_time = time.time() - program_start
    print("=" * 70)
    print(f"Total wall time: {total_time:.2f}s")
    print("=" * 70)

    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

