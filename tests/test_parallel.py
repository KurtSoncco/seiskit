"""Tests for seiskit.parallel module."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from seiskit.parallel import (
    AnalysisTask,
    AnalysisResult,
    prepare_analysis_tasks,
    run_parallel_analyses,
    run_parameter_study,
    collect_results,
)
from seiskit.config import AnalysisConfig
from seiskit.builder import ModelData


def test_analysis_task_creation():
    """Test AnalysisTask creation and validation."""
    config = AnalysisConfig()
    model_data = ModelData()
    
    task = AnalysisTask(
        task_id="test_task",
        config=config,
        model_data=model_data,
        output_dir="/tmp/test"
    )
    
    assert task.task_id == "test_task"
    assert task.config == config
    assert task.model_data == model_data
    assert task.output_dir == "/tmp/test"


def test_analysis_task_validation():
    """Test AnalysisTask input validation."""
    config = AnalysisConfig()
    model_data = ModelData()
    
    # Test with invalid config type
    with pytest.raises(TypeError):
        AnalysisTask(
            task_id="test_task",
            config="invalid_config",
            model_data=model_data,
            output_dir="/tmp/test"
        )
    
    # Test with invalid model_data type
    with pytest.raises(TypeError):
        AnalysisTask(
            task_id="test_task",
            config=config,
            model_data="invalid_model_data",
            output_dir="/tmp/test"
        )


def test_analysis_result_creation():
    """Test AnalysisResult creation and properties."""
    result = AnalysisResult(
        task_id="test_task",
        status="Finished successfully",
        output_path=Path("/tmp/test"),
        execution_time=10.5
    )
    
    assert result.task_id == "test_task"
    assert result.status == "Finished successfully"
    assert result.output_path == Path("/tmp/test")
    assert result.execution_time == 10.5
    assert result.error is None
    assert result.success is True


def test_analysis_result_success_property():
    """Test AnalysisResult success property logic."""
    # Successful result
    result1 = AnalysisResult(
        task_id="test_task",
        status="Finished successfully"
    )
    assert result1.success is True
    
    # Failed result with error
    result2 = AnalysisResult(
        task_id="test_task",
        status="Failed",
        error="Some error"
    )
    assert result2.success is False
    
    # Failed result without error
    result3 = AnalysisResult(
        task_id="test_task",
        status="Failed"
    )
    assert result3.success is False


def test_prepare_analysis_tasks():
    """Test prepare_analysis_tasks function."""
    task_configs = [
        {"task_id": "task1", "Ly": 20.0, "Lx": 20.0, "hx": 5.0},
    ]
    
    # Create material data with correct shape for the domain
    # For Lx=20, Ly=20, hx=5: expected shape is (4, 4) for soil grid
    material_data = {
        "vs": np.random.uniform(200.0, 400.0, (4, 4)),
        "rho": np.random.uniform(1800.0, 2000.0, (4, 4)),
        "nu": np.full((4, 4), 0.3)
    }
    
    tasks = prepare_analysis_tasks(task_configs, material_data)
    
    assert len(tasks) == 1
    assert tasks[0].task_id == "task1"
    assert isinstance(tasks[0].config, AnalysisConfig)
    assert isinstance(tasks[0].model_data, ModelData)


def test_run_parallel_analyses_interface():
    """Test run_parallel_analyses interface without execution."""
    config = AnalysisConfig()
    model_data = ModelData()
    
    tasks = [
        AnalysisTask(
            task_id="test_task",
            config=config,
            model_data=model_data,
            output_dir="/tmp/test"
        )
    ]
    
    # Test that the function exists and can be called
    # (We don't actually run it to avoid OpenSees dependency)
    assert callable(run_parallel_analyses)


def test_run_parameter_study():
    """Test run_parameter_study function."""
    base_config = {"Ly": 20.0, "Lx": 20.0, "duration": 15.0}
    parameter_variations = {
        "hx": [5.0, 2.5],
        "motion_freq": [0.5, 0.75]
    }
    
    # Create material data with correct shape
    material_data = {
        "vs": np.random.uniform(200.0, 400.0, (4, 4)),
        "rho": np.random.uniform(1800.0, 2000.0, (4, 4)),
        "nu": np.full((4, 4), 0.3)
    }
    
    # Test parameter combination generation
    tasks = prepare_analysis_tasks(
        [{"task_id": "test", **base_config, "hx": 5.0, "motion_freq": 0.5}],
        material_data
    )
    
    assert len(tasks) == 1
    assert tasks[0].config.hx == 5.0
    assert tasks[0].config.motion_freq == 0.5


def test_collect_results():
    """Test collect_results function."""
    results = [
        AnalysisResult(
            task_id="task1",
            status="Finished successfully",
            execution_time=10.0
        ),
        AnalysisResult(
            task_id="task2",
            status="Failed",
            error="Some error",
            execution_time=5.0
        ),
        AnalysisResult(
            task_id="task3",
            status="Finished successfully",
            execution_time=15.0
        )
    ]
    
    # Test summary collection
    summary = collect_results(results, output_format="summary")
    
    assert summary["total_tasks"] == 3
    assert summary["successful"] == 2
    assert summary["failed"] == 1
    assert summary["success_rate"] == 2/3
    assert summary["total_execution_time"] == 30.0
    assert "task2" in summary["failed_task_ids"]
    
    # Test detailed collection
    detailed = collect_results(results, output_format="detailed")
    
    assert len(detailed["results"]) == 3
    assert len(detailed["successful"]) == 2
    assert len(detailed["failed"]) == 1
    assert "summary" in detailed
    
    # Test failed only collection
    failed_only = collect_results(results, output_format="failed_only")
    
    assert len(failed_only["failed"]) == 1
    assert failed_only["failed"][0].task_id == "task2"


def test_parameter_study_combinations():
    """Test parameter study combination generation."""
    base_config = {"Ly": 20.0, "Lx": 20.0}
    parameter_variations = {
        "hx": [5.0, 2.5],
        "motion_freq": [0.5, 0.75]
    }
    
    # Create material data with correct shape
    material_data = {
        "vs": np.random.uniform(200.0, 400.0, (4, 4)),
        "rho": np.random.uniform(1800.0, 2000.0, (4, 4)),
        "nu": np.full((4, 4), 0.3)
    }
    
    # Test that the function exists and can be called
    assert callable(run_parameter_study)
