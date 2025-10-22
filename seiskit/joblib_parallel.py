"""Joblib-based parallel execution utilities for seiskit analyses.

This module provides joblib-based parallelization for running multiple
seismic analyses concurrently. It's designed as an alternative to the
existing ProcessPoolExecutor-based parallel system.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, cast

import numpy as np
from joblib import Parallel, delayed

from seiskit.analysis import run_opensees_analysis
from seiskit.builder import build_model_data
from seiskit.config import AnalysisConfig


@dataclass
class JoblibAnalysisTask:
    """Represents a single analysis task for joblib execution."""

    task_id: str
    config: AnalysisConfig
    vs_data: np.ndarray
    rho_data: np.ndarray
    nu_data: np.ndarray
    output_dir: str

    def __post_init__(self):
        """Validate the task data."""
        if not isinstance(self.config, AnalysisConfig):
            raise TypeError("config must be an AnalysisConfig instance")
        if not isinstance(self.vs_data, np.ndarray):
            raise TypeError("vs_data must be a numpy array")
        if not isinstance(self.rho_data, np.ndarray):
            raise TypeError("rho_data must be a numpy array")
        if not isinstance(self.nu_data, np.ndarray):
            raise TypeError("nu_data must be a numpy array")


@dataclass
class JoblibAnalysisResult:
    """Represents the result of a single joblib analysis."""

    task_id: str
    status: str
    output_path: Optional[Path] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None

    @property
    def success(self) -> bool:
        """Check if the analysis was successful."""
        return self.status.startswith("Finished") and self.error is None


def _run_single_joblib_analysis(task: JoblibAnalysisTask) -> JoblibAnalysisResult:
    """Run a single analysis using joblib.

    This function is designed to be called by joblib.Parallel.
    It builds the model data and runs the analysis in an isolated way.

    Args:
        task: JoblibAnalysisTask containing all required data

    Returns:
        JoblibAnalysisResult with status and output information
    """
    start_time = time.time()

    try:
        # Build model data (this is safe to do in parallel)
        model_data = build_model_data(
            task.config, task.vs_data, task.rho_data, task.nu_data
        )

        # Run the analysis
        status = run_opensees_analysis(
            config=task.config,
            model_data=model_data,
            run_id=task.task_id,
            output_dir=task.output_dir,
        )

        execution_time = time.time() - start_time

        return JoblibAnalysisResult(
            task_id=task.task_id,
            status=status,
            output_path=Path(task.output_dir),
            execution_time=execution_time,
        )

    except Exception as e:
        execution_time = time.time() - start_time
        return JoblibAnalysisResult(
            task_id=task.task_id,
            status="failed",
            error=str(e),
            execution_time=execution_time,
        )


def run_joblib_parallel_analyses(
    tasks: List[JoblibAnalysisTask],
    n_jobs: Optional[int] = None,
    backend: str = "loky",  # loky, threading, or multiprocessing
    verbose: int = 0,
    progress_callback: Optional[Callable[[JoblibAnalysisResult], None]] = None,
) -> List[JoblibAnalysisResult]:
    """Run multiple analyses in parallel using joblib.

    Args:
        tasks: List of JoblibAnalysisTask objects to execute
        n_jobs: Number of parallel jobs (default: -1 for all CPUs)
        backend: Joblib backend ('loky', 'threading', 'multiprocessing')
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed)
        progress_callback: Optional callback function for progress updates

    Returns:
        List of JoblibAnalysisResult objects in the same order as input tasks

    Example:
        >>> results = run_joblib_parallel_analyses(tasks, n_jobs=4, verbose=1)
        >>> successful_results = [r for r in results if r.success]
    """
    if n_jobs is None:
        n_jobs = -1  # Use all available CPUs

    # Create a wrapper function that includes progress callback
    def _run_with_callback(task: JoblibAnalysisTask) -> JoblibAnalysisResult:
        result = _run_single_joblib_analysis(task)
        if progress_callback:
            progress_callback(result)
        return result

    # Run analyses in parallel
    results = Parallel(
        n_jobs=n_jobs, backend=backend, verbose=verbose, return_as="list"
    )(delayed(_run_with_callback)(task) for task in tasks)
    assert isinstance(results, list) and all(
        isinstance(r, JoblibAnalysisResult) for r in results
    )

    return cast(List[JoblibAnalysisResult], results)


def prepare_joblib_tasks(
    task_configs: List[Dict[str, Any]],
    material_data: Dict[str, np.ndarray],
    base_output_dir: str = "results",
) -> List[JoblibAnalysisTask]:
    """Prepare joblib analysis tasks from configurations and material data.

    Args:
        task_configs: List of dictionaries containing task parameters
        material_data: Dictionary with 'vs', 'rho', 'nu' arrays
        base_output_dir: Base directory for outputs

    Returns:
        List of JoblibAnalysisTask objects ready for parallel execution

    Example:
        >>> task_configs = [
        ...     {"task_id": "case1", "Ly": 140.0, "Lx": 260.0, "hx": 5.0},
        ...     {"task_id": "case2", "Ly": 140.0, "Lx": 260.0, "hx": 2.5},
        ... ]
        >>> material_data = {
        ...     "vs": np.loadtxt("vs_data.txt"),
        ...     "rho": np.loadtxt("rho_data.txt"),
        ...     "nu": np.loadtxt("nu_data.txt"),
        ... }
        >>> tasks = prepare_joblib_tasks(task_configs, material_data)
    """
    tasks = []

    for task_config in task_configs:
        # Extract task_id and create output directory
        task_id = task_config.pop("task_id")
        output_dir = Path(base_output_dir) / task_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create AnalysisConfig from task parameters
        config = AnalysisConfig(**task_config)

        tasks.append(
            JoblibAnalysisTask(
                task_id=task_id,
                config=config,
                vs_data=material_data["vs"],
                rho_data=material_data["rho"],
                nu_data=material_data["nu"],
                output_dir=str(output_dir),
            )
        )

    return tasks


def run_joblib_parameter_study(
    base_config: Dict[str, Any],
    parameter_variations: Dict[str, List[Any]],
    material_data: Dict[str, np.ndarray],
    output_dir: str = "results",
    n_jobs: Optional[int] = None,
    backend: str = "loky",
    verbose: int = 0,
) -> List[JoblibAnalysisResult]:
    """Run a parameter study with joblib parallelization.

    Args:
        base_config: Base configuration parameters
        parameter_variations: Dictionary mapping parameter names to lists of values
        material_data: Material property arrays
        output_dir: Base output directory
        n_jobs: Number of parallel jobs
        backend: Joblib backend
        verbose: Verbosity level

    Returns:
        List of JoblibAnalysisResult objects

    Example:
        >>> base_config = {"Ly": 140.0, "Lx": 260.0, "duration": 15.0}
        >>> variations = {
        ...     "hx": [2.5, 5.0, 10.0],
        ...     "motion_freq": [0.5, 0.75, 1.0]
        ... }
        >>> results = run_joblib_parameter_study(base_config, variations, material_data)
    """
    import itertools

    # Generate all parameter combinations
    param_names = list(parameter_variations.keys())
    param_values = list(parameter_variations.values())

    task_configs = []
    for combination in itertools.product(*param_values):
        task_config = base_config.copy()

        # Create task ID from parameter values
        param_str = "_".join(
            f"{name}{val}" for name, val in zip(param_names, combination)
        )
        task_config["task_id"] = f"param_study_{param_str}"

        # Add parameter values to config
        for name, val in zip(param_names, combination):
            task_config[name] = val

        task_configs.append(task_config)

    # Prepare and run tasks
    tasks = prepare_joblib_tasks(task_configs, material_data, output_dir)
    return run_joblib_parallel_analyses(
        tasks, n_jobs=n_jobs, backend=backend, verbose=verbose
    )


def collect_joblib_results(
    results: List[JoblibAnalysisResult], output_format: str = "summary"
) -> Dict[str, Any]:
    """Collect and summarize results from joblib parallel analyses.

    Args:
        results: List of JoblibAnalysisResult objects
        output_format: Format for output ('summary', 'detailed', 'failed_only')

    Returns:
        Dictionary containing collected results
    """
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    if output_format == "summary":
        return {
            "total_tasks": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(results) if results else 0.0,
            "total_execution_time": sum(r.execution_time or 0 for r in results),
            "failed_task_ids": [r.task_id for r in failed],
        }
    elif output_format == "detailed":
        return {
            "results": results,
            "successful": successful,
            "failed": failed,
            "summary": collect_joblib_results(results, "summary"),
        }
    elif output_format == "failed_only":
        return {"failed": failed}
    else:
        raise ValueError(f"Unknown output_format: {output_format}")


# Convenience function for simple joblib parallel execution
def run_analyses_joblib_parallel(
    configs: List[Dict[str, Any]],
    material_data: Dict[str, np.ndarray],
    output_dir: str = "results",
    n_jobs: Optional[int] = None,
    backend: str = "loky",
    verbose: int = 0,
) -> List[JoblibAnalysisResult]:
    """Convenience function for running multiple analyses in parallel with joblib.

    Args:
        configs: List of configuration dictionaries
        material_data: Material property arrays
        output_dir: Base output directory
        n_jobs: Number of parallel jobs
        backend: Joblib backend
        verbose: Verbosity level

    Returns:
        List of JoblibAnalysisResult objects

    Example:
        >>> configs = [
        ...     {"task_id": "case1", "hx": 5.0},
        ...     {"task_id": "case2", "hx": 2.5},
        ... ]
        >>> results = run_analyses_joblib_parallel(configs, material_data)
    """
    tasks = prepare_joblib_tasks(configs, material_data, output_dir)
    return run_joblib_parallel_analyses(
        tasks, n_jobs=n_jobs, backend=backend, verbose=verbose
    )
