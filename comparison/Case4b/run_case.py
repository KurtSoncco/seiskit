"""Example runner for comparison/Case1a using the refactored seiskit package.

This runner prepares inputs and calls `seiskit.run_analysis.run_case_basic`.
"""

import os
import sys
from pathlib import Path

import numpy as np

from seiskit.analysis import run_opensees_analysis
from seiskit.builder import build_model_data
from seiskit.config import AnalysisConfig
from seiskit.plot_results import (
    load_datasets,
    plot_acceleration_comparison,
    plot_acceleration_differences,
    plot_transfer_functions,
)

if __name__ == "__main__":
    # Change to the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # 1. Define the analysis configuration
    config = AnalysisConfig(
        Ly=150.0,  # Domain height
        Lx=500.0,  # Domain width
        hx=5.0,  # Element size
        dt=0.01,  # Time step
        duration=15.0,  # Analysis duration
        motion_freq=0.75,  # Motion frequency
        motion_t_shift=1.4 - 0.01,  # Motion time shift
        damping_zeta=0.0075,  # Damping ratio
        damping_freqs=(0.75, 2.25),  # Damping frequencies
        boundary_condition_type="2D",  # Use 2D BCs
    )
    RUN_ID = "Case_4b"
    OUTPUT_DIR = "results"

    # 2. Create homogeneous material property arrays
    ndivx = int(config.Lx / config.hx)
    ndivy = int(config.Ly / config.hy)
    vs_data = np.full((ndivy, ndivx), 140.0)  # Vs = 140 m/s
    rho_data = np.full((ndivy, ndivx), 2000.0)  # rho = 2000 kg/m³
    nu_data = np.full((ndivy, ndivx), 0.3)  # nu = 0.3

    # 3. Build the model data structure (no OpenSees calls)
    print("--- Building model geometry and material maps... ---")
    model_data = build_model_data(config, vs_data, rho_data, nu_data)

    # 4. Run the OpenSees simulation with the prepared data
    res = run_opensees_analysis(
        config=config,
        model_data=model_data,
        run_id=RUN_ID,
        output_dir=OUTPUT_DIR,
    )
    print(res)

    RESULTS_DIR = Path("./results/Case_4b")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)  # Ensure dir exists

    # 1. Define the configuration for all data sources
    DATA_CONFIG = {
        "My_New_Run": {
            "base": RESULTS_DIR / "soil_base.txt",
            "top": RESULTS_DIR / "soil_top.txt",
        },
        "FLAC": {
            "base": RESULTS_DIR / "BaseRight.txt",
            "top": RESULTS_DIR / "SurfaceRight.txt",
        },
    }

    # 2. Load all the data into a standardized structure
    try:
        all_data = load_datasets(DATA_CONFIG)

    except FileNotFoundError as e:
        print(f"Error: Data file not found. {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Generate the plots
    REFERENCE_MODEL = "My_New_Run"

    plot_acceleration_comparison(
        datasets=all_data,
        reference_name=REFERENCE_MODEL,
        output_path=Path("acceleration_time_histories_comparison.html"),
    )

    plot_acceleration_differences(
        datasets=all_data,
        reference_name=REFERENCE_MODEL,
        output_path=Path("acceleration_time_histories_differences.html"),
    )

    plot_transfer_functions(
        datasets=all_data,
        reference_name=REFERENCE_MODEL,
        output_path=Path("transfer_functions.html"),
    )
