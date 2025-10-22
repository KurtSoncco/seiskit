"""Parallel execution utilities for seiskit analyses.

This module provides parallelization-ready interfaces for running multiple
seismic analyses concurrently. It separates data preparation (parallel-safe)
from OpenSees execution (requires process isolation).
"""

import multiprocessing as mp
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .builder import ModelData, build_model_data
from .config import AnalysisConfig


@dataclass
class AnalysisTask:
    """Represents a single analysis task with all required data."""
    
    task_id: str
    config: AnalysisConfig
    model_data: ModelData
    output_dir: str
    
    def __post_init__(self):
        """Validate the task data."""
        if not isinstance(self.config, AnalysisConfig):
            raise TypeError("config must be an AnalysisConfig instance")
        if not isinstance(self.model_data, ModelData):
            raise TypeError("model_data must be a ModelData instance")


@dataclass
class AnalysisResult:
    """Represents the result of a single analysis."""
    
    task_id: str
    status: str
    output_path: Optional[Path] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    
    @property
    def success(self) -> bool:
        """Check if the analysis was successful."""
        return self.status.startswith("Finished") and self.error is None


def prepare_analysis_tasks(
    task_configs: List[Dict[str, Any]],
    material_data: Dict[str, np.ndarray],
    base_output_dir: str = "results",
) -> List[AnalysisTask]:
    """Prepare analysis tasks from configurations and material data.
    
    This function is parallelization-safe and can be called in the main process.
    It prepares all the data needed for analysis without making any OpenSees calls.
    
    Args:
        task_configs: List of dictionaries containing task parameters
        material_data: Dictionary with 'vs', 'rho', 'nu' arrays
        base_output_dir: Base directory for outputs
        
    Returns:
        List of AnalysisTask objects ready for parallel execution
        
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
        >>> tasks = prepare_analysis_tasks(task_configs, material_data)
    """
    tasks = []
    
    for task_config in task_configs:
        # Extract task_id and create output directory
        task_id = task_config.pop("task_id")
        output_dir = Path(base_output_dir) / task_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create AnalysisConfig from task parameters
        config = AnalysisConfig(**task_config)
        
        # Build model data (no OpenSees calls)
        model_data = build_model_data(
            config, 
            material_data["vs"], 
            material_data["rho"], 
            material_data["nu"]
        )
        
        tasks.append(AnalysisTask(
            task_id=task_id,
            config=config,
            model_data=model_data,
            output_dir=str(output_dir)
        ))
    
    return tasks


def _run_isolated_analysis(task: AnalysisTask) -> AnalysisResult:
    """Run a single analysis in an isolated process.
    
    This function is designed to be called in a separate process to avoid
    OpenSees global state conflicts. It imports OpenSees locally and runs
    the analysis with the provided task data.
    
    Args:
        task: AnalysisTask containing all required data
        
    Returns:
        AnalysisResult with status and output information
    """
    import timeit
    from pathlib import Path
    
    # Import OpenSees locally in this process
    try:
        import openseespy.opensees as ops
    except ImportError:
        return AnalysisResult(
            task_id=task.task_id,
            status="no-opensees",
            error="OpenSees not available"
        )
    
    start_time = timeit.default_timer()
    
    try:
        # Import the analysis function locally
        from seiskit.analysis import run_opensees_analysis
        
        # Run the analysis (this will create its own isolated OpenSees instance)
        status = run_opensees_analysis(
            config=task.config,
            model_data=task.model_data,
            run_id=task.task_id,
            output_dir=task.output_dir
        )
        
        execution_time = timeit.default_timer() - start_time
        
        return AnalysisResult(
            task_id=task.task_id,
            status=status,
            output_path=Path(task.output_dir),
            execution_time=execution_time
        )
        
    except Exception as e:
        execution_time = timeit.default_timer() - start_time
        return AnalysisResult(
            task_id=task.task_id,
            status="failed",
            error=str(e),
            execution_time=execution_time
        )


def run_parallel_analyses(
    tasks: List[AnalysisTask],
    max_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[AnalysisResult], None]] = None,
) -> List[AnalysisResult]:
    """Run multiple analyses in parallel using process pool.
    
    Args:
        tasks: List of AnalysisTask objects to execute
        max_workers: Maximum number of worker processes (default: CPU count)
        progress_callback: Optional callback function for progress updates
        
    Returns:
        List of AnalysisResult objects in the same order as input tasks
        
    Example:
        >>> results = run_parallel_analyses(tasks, max_workers=4)
        >>> successful_results = [r for r in results if r.success]
    """
    if max_workers is None:
        max_workers = min(len(tasks), mp.cpu_count())
    
    results = [None] * len(tasks)
    task_indices = {task.task_id: i for i, task in enumerate(tasks)}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(_run_isolated_analysis, task): task 
            for task in tasks
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                results[task_indices[task.task_id]] = result
                
                if progress_callback:
                    progress_callback(result)
                    
            except Exception as e:
                # Handle unexpected errors in the executor
                result = AnalysisResult(
                    task_id=task.task_id,
                    status="executor-error",
                    error=str(e)
                )
                results[task_indices[task.task_id]] = result
                
                if progress_callback:
                    progress_callback(result)
    
    return results


def run_parameter_study(
    base_config: Dict[str, Any],
    parameter_variations: Dict[str, List[Any]],
    material_data: Dict[str, np.ndarray],
    output_dir: str = "results",
    max_workers: Optional[int] = None,
) -> List[AnalysisResult]:
    """Run a parameter study with multiple parameter combinations.
    
    This is a high-level interface for running parameter studies where
    you want to vary specific parameters and see how they affect results.
    
    Args:
        base_config: Base configuration parameters
        parameter_variations: Dictionary mapping parameter names to lists of values
        material_data: Material property arrays
        output_dir: Base output directory
        max_workers: Maximum number of worker processes
        
    Returns:
        List of AnalysisResult objects
        
    Example:
        >>> base_config = {"Ly": 140.0, "Lx": 260.0, "duration": 15.0}
        >>> variations = {
        ...     "hx": [2.5, 5.0, 10.0],
        ...     "motion_freq": [0.5, 0.75, 1.0]
        ... }
        >>> results = run_parameter_study(base_config, variations, material_data)
    """
    import itertools
    
    # Generate all parameter combinations
    param_names = list(parameter_variations.keys())
    param_values = list(parameter_variations.values())
    
    task_configs = []
    for combination in itertools.product(*param_values):
        task_config = base_config.copy()
        
        # Create task ID from parameter values
        param_str = "_".join(f"{name}{val}" for name, val in zip(param_names, combination))
        task_config["task_id"] = f"param_study_{param_str}"
        
        # Add parameter values to config
        for name, val in zip(param_names, combination):
            task_config[name] = val
            
        task_configs.append(task_config)
    
    # Prepare and run tasks
    tasks = prepare_analysis_tasks(task_configs, material_data, output_dir)
    return run_parallel_analyses(tasks, max_workers=max_workers)


def collect_results(
    results: List[AnalysisResult],
    output_format: str = "summary"
) -> Dict[str, Any]:
    """Collect and summarize results from parallel analyses.
    
    Args:
        results: List of AnalysisResult objects
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
            "failed_task_ids": [r.task_id for r in failed]
        }
    elif output_format == "detailed":
        return {
            "results": results,
            "successful": successful,
            "failed": failed,
            "summary": collect_results(results, "summary")
        }
    elif output_format == "failed_only":
        return {"failed": failed}
    else:
        raise ValueError(f"Unknown output_format: {output_format}")


# Convenience function for simple parallel execution
def run_analyses_parallel(
    configs: List[Dict[str, Any]],
    material_data: Dict[str, np.ndarray],
    output_dir: str = "results",
    max_workers: Optional[int] = None,
) -> List[AnalysisResult]:
    """Convenience function for running multiple analyses in parallel.
    
    This is the simplest interface for parallel execution.
    
    Args:
        configs: List of configuration dictionaries
        material_data: Material property arrays
        output_dir: Base output directory
        max_workers: Maximum number of worker processes
        
    Returns:
        List of AnalysisResult objects
        
    Example:
        >>> configs = [
        ...     {"task_id": "case1", "hx": 5.0},
        ...     {"task_id": "case2", "hx": 2.5},
        ... ]
        >>> results = run_analyses_parallel(configs, material_data)
    """
    tasks = prepare_analysis_tasks(configs, material_data, output_dir)
    return run_parallel_analyses(tasks, max_workers=max_workers)
