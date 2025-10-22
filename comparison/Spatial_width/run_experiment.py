import os
from pathlib import Path

import numpy as np

from seiskit.config import AnalysisConfig
from seiskit.gaussian_field import (
    _extend_profile,
    _generate_vs_variability_field,
    plot_realization,
)
from seiskit.joblib_parallel import (
    JoblibAnalysisResult,
    collect_joblib_results,
    run_joblib_parallel_analyses,
)
from seiskit.plot_results import (
    load_datasets,
    plot_acceleration_comparison,
    plot_transfer_functions,
)


def main():
    """
    In this experiment, we are going to generate ht emost variable case for a single Vs profile.
    We are going to gradually decrease the width of the model until we observe significant errors in the PGA values and differences in the transfer functions.

    Experiment Layout:
    Vs = np.array([140.0] * 8 + [1300.0] * 1)
    Lz = 50.0
    rH, aHV, CV = 10.0, 10.0, 0.3


    Lx = [500.0, 400.0, 300.0, 200.0]
    """

    ## Set base case
    Vs_profile_1D = np.array([180.0] * 8 + [1300.0] * 1)
    Lx = 1000
    Lz = 50.0
    dx, dz = 2.5, 2.5
    rH, aHV, CV = 10.0, 2.0, 0.3
    seed = 42

    # Lx_values = [500, 250, 100, 50, 25, 10]
    Lx_variability_values = [800, 700, 600, 500, 400, 300, 200, 100]

    results = {Lx: f"results/Lx_{Lx}" for Lx in Lx_variability_values}

    print(f"Generating Vs realization for Lx = {np.max(Lx_variability_values)} m...")
    Vs_total_realization, x_coords, z_coords, h_mean = _generate_vs_variability_field(
        Vs_profile_1D,
        np.max(Lx_variability_values),
        Lz,
        dx,
        dz,
        rH,
        aHV,
        CV,
        seed=seed,
    )

    # Prepare all tasks for parallel execution
    print("Preparing tasks for parallel execution...")
    task_configs = []
    material_data_list = []

    for Lx_i in Lx_variability_values:
        print(f"Preparing Vs realization for Lx = {Lx_i} m...")
        # Compute the central region given the desired Lx
        center_x = np.max(Lx_variability_values) / 2.0
        half_Lx = Lx_i / 2.0
        x_start = center_x - half_Lx
        x_end = center_x + half_Lx
        start_idx = int(x_start / dx)
        end_idx = int(x_end / dx)
        Vs_realization = Vs_total_realization[:, start_idx:end_idx]

        Vs_realization, x_total = _extend_profile(
            Vs_realization,
            Lx=Lx,
            dx=dx,
        )

        # Make sure output directory exists
        os.makedirs(results[Lx_i], exist_ok=True)

        # Create plot (this is done sequentially as it's fast)
        plot_realization(
            Vs_profile_1D,
            Vs_realization,
            Lx,
            Lz,
            dx,
            dz,
            save_path=f"{results[Lx_i]}/Vs_realization.png",
        )

        # Prepare task configuration
        task_config = {
            "task_id": f"Lx_{Lx_i}",
            "Ly": Lz,  # Domain height
            "Lx": Lx,  # Domain width
            "hx": dx,  # Element size
            "dt": 0.01,  # Time step
            "duration": 15.0,  # Analysis duration
            "motion_freq": 0.75,  # Motion frequency
            "motion_t_shift": 1.4,  # Motion time shift
            "damping_zeta": 0.0075,  # Damping ratio
            "damping_freqs": (0.75, 2.25),  # Damping frequencies
            "boundary_condition_type": "2D",  # Use 2D BCs
        }

        # Prepare material data
        material_data = {
            "vs": Vs_realization,
            "rho": np.ones_like(Vs_realization) * 2000,
            "nu": np.ones_like(Vs_realization) * 0.3,
        }

        task_configs.append(task_config)
        material_data_list.append(material_data)

        print(f"Task preparation complete for Lx = {Lx} m.")

    # Run analyses in parallel using joblib
    print(f"\nRunning {len(task_configs)} analyses in parallel...")

    def progress_callback(result: JoblibAnalysisResult):
        """Progress callback for joblib execution."""
        if result.success:
            print(f"✓ Completed: {result.task_id} (t={result.execution_time:.1f}s)")
        else:
            print(f"✗ Failed: {result.task_id} - {result.error}")

    # Prepare all tasks
    all_tasks = []
    for i, (task_config, material_data) in enumerate(
        zip(task_configs, material_data_list)
    ):
        # Create AnalysisConfig from task parameters
        config = AnalysisConfig(
            **{k: v for k, v in task_config.items() if k != "task_id"}
        )

        # Create output directory
        output_dir = results[Lx_variability_values[i]]

        from seiskit.joblib_parallel import JoblibAnalysisTask

        task = JoblibAnalysisTask(
            task_id=task_config["task_id"],
            config=config,
            vs_data=material_data["vs"],
            rho_data=material_data["rho"],
            nu_data=material_data["nu"],
            output_dir=output_dir,
        )
        all_tasks.append(task)

    # Run parallel analyses
    results_list = run_joblib_parallel_analyses(
        all_tasks,
        n_jobs=-1,  # Use all available CPUs
        backend="loky",  # Use loky backend for better process isolation
        verbose=1,  # Show progress
        progress_callback=progress_callback,
    )

    # Collect and display results
    summary = collect_joblib_results(results_list, "summary")
    print("\n=== Parallel Execution Summary ===")
    print(f"Total tasks: {summary['total_tasks']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    print(f"Success rate: {summary['success_rate']:.1%}")
    print(f"Total execution time: {summary['total_execution_time']:.1f}s")

    if summary["failed"] > 0:
        print(f"Failed tasks: {summary['failed_task_ids']}")


def analysis_results():
    """After the analysis are completed, we can plot the results.
    1. Comparison of base and top motions in plotly
    2. Comparison of transfer functions in plotly
    """

    RESULTS_DIR = Path("./results")
    DATA_CONFIG = {
        "Lx_800": {
            "base": RESULTS_DIR / "Lx_800/soil_base.txt",
            "top": RESULTS_DIR / "Lx_800/soil_top.txt",
        },
        "Lx_700": {
            "base": RESULTS_DIR / "Lx_700/soil_base.txt",
            "top": RESULTS_DIR / "Lx_700/soil_top.txt",
        },
        "Lx_600": {
            "base": RESULTS_DIR / "Lx_600/soil_base.txt",
            "top": RESULTS_DIR / "Lx_600/soil_top.txt",
        },
        "Lx_500": {
            "base": RESULTS_DIR / "Lx_500/soil_base.txt",
            "top": RESULTS_DIR / "Lx_500/soil_top.txt",
        },
        "Lx_400": {
            "base": RESULTS_DIR / "Lx_400/soil_base.txt",
            "top": RESULTS_DIR / "Lx_400/soil_top.txt",
        },
        "Lx_300": {
            "base": RESULTS_DIR / "Lx_300/soil_base.txt",
            "top": RESULTS_DIR / "Lx_300/soil_top.txt",
        },
        "Lx_200": {
            "base": RESULTS_DIR / "Lx_200/soil_base.txt",
            "top": RESULTS_DIR / "Lx_200/soil_top.txt",
        },
        "Lx_100": {
            "base": RESULTS_DIR / "Lx_100/soil_base.txt",
            "top": RESULTS_DIR / "Lx_100/soil_top.txt",
        },
    }

    # 2. Load all the data into a standardized structure
    data = load_datasets(DATA_CONFIG)
    # 3. Plot transfer functions comparison
    plot_transfer_functions(
        datasets=data,
        reference_name="",
        output_path=Path("transfer_functions_comparison.html"),
        dz=2.5,
    )
    # 4. Plot acceleration comparison at the top and base
    plot_acceleration_comparison(
        datasets=data,
        reference_name="",
        output_path=Path("acceleration_time_histories_comparison.html"),
    )


def compare_sequential_vs_parallel():
    """Compare sequential vs parallel execution performance."""
    import time

    # Set base case
    Vs_profile_1D = np.array([180.0] * 8 + [1300.0] * 1)
    Lx = 1000
    Lz = 50.0
    dx, dz = 2.5, 2.5
    rH, aHV, CV = 10.0, 2.0, 0.3
    seed = 42

    # Use smaller set for comparison
    Lx_variability_values = [200, 100]  # Just 2 cases for quick comparison

    print("=== Performance Comparison: Sequential vs Parallel ===")

    # Generate Vs realization
    Vs_total_realization, x_coords, z_coords, h_mean = _generate_vs_variability_field(
        Vs_profile_1D,
        np.max(Lx_variability_values),
        Lz,
        dx,
        dz,
        rH,
        aHV,
        CV,
        seed=seed,
    )

    # Prepare material data for both approaches
    material_data_list = []
    task_configs = []

    for Lx_i in Lx_variability_values:
        center_x = np.max(Lx_variability_values) / 2.0
        half_Lx = Lx_i / 2.0
        x_start = center_x - half_Lx
        x_end = center_x + half_Lx
        start_idx = int(x_start / dx)
        end_idx = int(x_end / dx)
        Vs_realization = Vs_total_realization[:, start_idx:end_idx]

        Vs_realization, x_total = _extend_profile(Vs_realization, Lx=Lx, dx=dx)

        material_data = {
            "vs": Vs_realization,
            "rho": np.ones_like(Vs_realization) * 2000,
            "nu": np.ones_like(Vs_realization) * 0.3,
        }

        task_config = {
            "task_id": f"Lx_{Lx_i}",
            "Ly": Lz,
            "Lx": Lx,  # The domain width is Lx, not the variability width
            "hx": dx,
            "dt": 0.01,
            "duration": 15.0,
            "motion_freq": 0.75,
            "motion_t_shift": 1.4,
            "damping_zeta": 0.0075,
            "damping_freqs": (0.75, 2.25),
            "boundary_condition_type": "2D",
        }

        material_data_list.append(material_data)
        task_configs.append(task_config)

    # Sequential execution
    print("\n1. Sequential Execution:")
    start_time = time.time()

    for i, (task_config, material_data) in enumerate(
        zip(task_configs, material_data_list)
    ):
        config = AnalysisConfig(
            **{k: v for k, v in task_config.items() if k != "task_id"}
        )
        output_dir = f"results/comparison/Lx_{Lx_variability_values[i]}"
        os.makedirs(output_dir, exist_ok=True)

        from seiskit.analysis import run_opensees_analysis
        from seiskit.builder import build_model_data

        model_data = build_model_data(
            config, material_data["vs"], material_data["rho"], material_data["nu"]
        )
        result = run_opensees_analysis(
            config, model_data, task_config["task_id"], output_dir
        )
        print(f"  Sequential {task_config['task_id']}: {result}")

    sequential_time = time.time() - start_time
    print(f"  Sequential total time: {sequential_time:.1f}s")

    # Parallel execution
    print("\n2. Parallel Execution:")
    start_time = time.time()

    all_tasks = []
    for i, (task_config, material_data) in enumerate(
        zip(task_configs, material_data_list)
    ):
        config = AnalysisConfig(
            **{k: v for k, v in task_config.items() if k != "task_id"}
        )
        output_dir = f"results/comparison_parallel/Lx_{Lx_variability_values[i]}"
        os.makedirs(output_dir, exist_ok=True)

        from seiskit.joblib_parallel import JoblibAnalysisTask

        task = JoblibAnalysisTask(
            task_id=task_config["task_id"],
            config=config,
            vs_data=material_data["vs"],
            rho_data=material_data["rho"],
            nu_data=material_data["nu"],
            output_dir=output_dir,
        )
        all_tasks.append(task)

    results_list = run_joblib_parallel_analyses(all_tasks, n_jobs=2, verbose=1)
    parallel_time = time.time() - start_time

    print(f"  Parallel total time: {parallel_time:.1f}s")

    # Summary
    speedup = sequential_time / parallel_time if parallel_time > 0 else float("inf")
    print("\n=== Performance Summary ===")
    print(f"Sequential time: {sequential_time:.1f}s")
    print(f"Parallel time: {parallel_time:.1f}s")
    print(f"Speedup: {speedup:.1f}x")
    print(f"Efficiency: {speedup / 2:.1%}")  # Assuming 2 cores


if __name__ == "__main__":
    # Change to the script's directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Uncomment the function you want to run:
    main()  # Run the full parallel experiment
    # analysis_results()  # Plot results after analysis
    # compare_sequential_vs_parallel()  # Compare performance
