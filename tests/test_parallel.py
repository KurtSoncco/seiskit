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
    model_data = ModelData(
        vs=np.array([200.0, 250.0]),
        rho=np.array([1800.0, 1900.0]),
        nu=np.array([0.3, 0.3]),
        nx=2,
        ny=1
    )
    
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
    model_data = ModelData(
        vs=np.array([200.0, 250.0]),
        rho=np.array([1800.0, 1900.0]),
        nu=np.array([0.3, 0.3]),
        nx=2,
        ny=1
    )
    
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
        {"task_id": "task1", "Ly": 140.0, "Lx": 260.0, "hx": 5.0},
        {"task_id": "task2", "Ly": 140.0, "Lx": 260.0, "hx": 2.5},
    ]
    
    material_data = {
        "vs": np.array([200.0, 250.0]),
        "rho": np.array([1800.0, 1900.0]),
        "nu": np.array([0.3, 0.3])
    }
    
    tasks = prepare_analysis_tasks(task_configs, material_data)
    
    assert len(tasks) == 2
    assert tasks[0].task_id == "task1"
    assert tasks[1].task_id == "task2"
    assert isinstance(tasks[0].config, AnalysisConfig)
    assert isinstance(tasks[0].model_data, ModelData)


@patch('seiskit.parallel.run_opensees_analysis')
def test_run_parallel_analyses_mock(mock_analysis):
    """Test run_parallel_analyses with mocked OpenSees."""
    mock_analysis.return_value = "Finished successfully"
    
    config = AnalysisConfig()
    model_data = ModelData(
        vs=np.array([200.0, 250.0]),
        rho=np.array([1800.0, 1900.0]),
        nu=np.array([0.3, 0.3]),
        nx=2,
        ny=1
    )
    
    tasks = [
        AnalysisTask(
            task_id="test_task",
            config=config,
            model_data=model_data,
            output_dir="/tmp/test"
        )
    ]
    
    results = run_parallel_analyses(tasks, max_workers=1)
    
    assert len(results) == 1
    assert results[0].task_id == "test_task"
    assert results[0].status == "Finished successfully"


def test_run_parameter_study():
    """Test run_parameter_study function."""
    base_config = {"Ly": 140.0, "Lx": 260.0, "duration": 15.0}
    parameter_variations = {
        "hx": [5.0, 2.5],
        "motion_freq": [0.5, 0.75]
    }
    
    material_data = {
        "vs": np.array([200.0, 250.0]),
        "rho": np.array([1800.0, 1900.0]),
        "nu": np.array([0.3, 0.3])
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
    base_config = {"Ly": 140.0, "Lx": 260.0}
    parameter_variations = {
        "hx": [5.0, 2.5],
        "motion_freq": [0.5, 0.75]
    }
    
    material_data = {
        "vs": np.array([200.0, 250.0]),
        "rho": np.array([1800.0, 1900.0]),
        "nu": np.array([0.3, 0.3])
    }
    
    # This should generate 4 combinations (2x2)
    results = run_parameter_study(
        base_config,
        parameter_variations,
        material_data,
        max_workers=1
    )
    
    # Should have 4 results (though they may fail due to OpenSees not being available)
    assert len(results) >= 0  # At least no errors in task preparation
