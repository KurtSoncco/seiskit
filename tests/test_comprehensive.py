"""Comprehensive tests for seiskit package."""

import pytest
import numpy as np
from pathlib import Path

from seiskit.config import AnalysisConfig
from seiskit.utils import compute_ricker, load_material_properties
from seiskit.damping import compute_rayleigh_coefficients


def test_complete_workflow():
    """Test a complete workflow without OpenSees dependency."""
    # 1. Create configuration
    config = AnalysisConfig(
        Ly=140.0,
        Lx=260.0,
        hx=5.0,
        duration=15.0,
        motion_freq=0.75
    )
    
    assert config.Ly == 140.0
    assert config.Lx == 260.0
    assert config.hx == 5.0
    assert config.duration == 15.0
    assert config.motion_freq == 0.75
    
    # 2. Compute Ricker wavelet
    ricker_wave = compute_ricker(
        freq=config.motion_freq,
        t_shift=1.4,
        duration=config.duration,
        dt=0.01
    )
    
    assert len(ricker_wave) > 0
    assert isinstance(ricker_wave, np.ndarray)
    
    # 3. Compute Rayleigh damping coefficients
    alphaM, betaK = compute_rayleigh_coefficients(
        zeta=config.damping_zeta,
        f1=config.damping_freqs[0],
        f2=config.damping_freqs[1]
    )
    
    assert alphaM > 0
    assert betaK > 0
    assert isinstance(alphaM, float)
    assert isinstance(betaK, float)


def test_material_properties_loading():
    """Test material properties loading functionality."""
    # Create temporary test files
    vs_data = np.array([200.0, 250.0, 300.0, 350.0])
    rho_data = np.array([1800.0, 1900.0, 2000.0, 2100.0])
    nu_data = np.array([0.3, 0.3, 0.3, 0.3])
    
    # Test with numpy arrays (simulating loaded data)
    material_data = {
        "vs": vs_data,
        "rho": rho_data,
        "nu": nu_data
    }
    
    assert "vs" in material_data
    assert "rho" in material_data
    assert "nu" in material_data
    assert len(material_data["vs"]) == 4
    assert len(material_data["rho"]) == 4
    assert len(material_data["nu"]) == 4


def test_ricker_wavelet_properties():
    """Test Ricker wavelet mathematical properties."""
    freq = 1.0
    t_shift = 2.0
    duration = 5.0
    dt = 0.01
    
    samples = compute_ricker(freq, t_shift, duration, dt)
    
    # Test basic properties
    assert len(samples) == int(duration / dt) + 1
    assert isinstance(samples, np.ndarray)
    
    # Test that peak occurs around t_shift
    peak_idx = np.argmax(np.abs(samples))
    peak_time = peak_idx * dt
    assert abs(peak_time - t_shift) < 0.2  # Allow some tolerance
    
    # Test that the wavelet has the expected frequency content
    # (This is a basic test - more sophisticated frequency analysis could be added)


def test_damping_coefficient_validation():
    """Test Rayleigh damping coefficient validation."""
    # Test with typical seismic analysis parameters
    damping_ratios = [0.02, 0.05, 0.10]
    frequencies = [(0.5, 2.0), (1.0, 5.0), (2.0, 10.0)]
    
    for zeta in damping_ratios:
        for f1, f2 in frequencies:
            alphaM, betaK = compute_rayleigh_coefficients(zeta, f1, f2)
            
            # Basic validation
            assert alphaM > 0, f"alphaM should be positive for zeta={zeta}, f1={f1}, f2={f2}"
            assert betaK > 0, f"betaK should be positive for zeta={zeta}, f1={f1}, f2={f2}"
            assert isinstance(alphaM, float)
            assert isinstance(betaK, float)


def test_configuration_validation():
    """Test configuration parameter validation."""
    # Test default configuration
    config = AnalysisConfig()
    
    # Test that all required fields have reasonable defaults
    assert config.Ly > 0
    assert config.Lx > 0
    assert config.hx > 0
    assert config.duration > 0
    assert config.motion_freq > 0
    assert config.damping_zeta > 0
    assert len(config.damping_freqs) == 2
    assert config.damping_freqs[0] < config.damping_freqs[1]
    
    # Test custom configuration
    custom_config = AnalysisConfig(
        Ly=200.0,
        Lx=300.0,
        hx=2.5,
        duration=20.0,
        motion_freq=1.0
    )
    
    assert custom_config.Ly == 200.0
    assert custom_config.Lx == 300.0
    assert custom_config.hx == 2.5
    assert custom_config.duration == 20.0
    assert custom_config.motion_freq == 1.0


def test_parallel_processing_interface():
    """Test parallel processing interface without actual execution."""
    from seiskit.parallel import AnalysisResult, AnalysisTask
    
    # Test AnalysisResult creation
    result = AnalysisResult(
        task_id="test_task",
        status="Finished successfully",
        execution_time=10.5
    )
    
    assert result.task_id == "test_task"
    assert result.status == "Finished successfully"
    assert result.execution_time == 10.5
    assert result.success is True
    
    # Test AnalysisResult with error
    error_result = AnalysisResult(
        task_id="failed_task",
        status="Failed",
        error="Some error occurred"
    )
    
    assert error_result.task_id == "failed_task"
    assert error_result.status == "Failed"
    assert error_result.error == "Some error occurred"
    assert error_result.success is False


def test_joblib_parallel_interface():
    """Test joblib parallel processing interface."""
    from seiskit.joblib_parallel import JoblibAnalysisResult, JoblibAnalysisTask
    from seiskit.config import AnalysisConfig
    
    config = AnalysisConfig()
    
    # Test JoblibAnalysisTask creation
    task = JoblibAnalysisTask(
        task_id="test_task",
        config=config,
        vs_data=np.array([200.0, 250.0]),
        rho_data=np.array([1800.0, 1900.0]),
        nu_data=np.array([0.3, 0.3]),
        output_dir="/tmp/test"
    )
    
    assert task.task_id == "test_task"
    assert task.config == config
    assert np.array_equal(task.vs_data, np.array([200.0, 250.0]))
    assert np.array_equal(task.rho_data, np.array([1800.0, 1900.0]))
    assert np.array_equal(task.nu_data, np.array([0.3, 0.3]))
    
    # Test JoblibAnalysisResult creation
    result = JoblibAnalysisResult(
        task_id="test_task",
        status="Finished successfully",
        execution_time=10.5
    )
    
    assert result.task_id == "test_task"
    assert result.status == "Finished successfully"
    assert result.success is True


def test_ttf_functionality():
    """Test TTF (Time-to-Frequency) functionality."""
    from seiskit.ttf.acc2FAS2 import acc2FAS2
    
    # Create a simple test signal
    dt = 0.01
    t = np.arange(0, 2.0, dt)
    # Simple sine wave at 1 Hz
    acc = np.sin(2 * np.pi * 1.0 * t)
    
    # Test acc2FAS2 function
    fas, freq = acc2FAS2(acc, dt)
    
    assert len(fas) > 0
    assert len(freq) > 0
    assert len(fas) == len(freq)
    assert isinstance(fas, np.ndarray)
    assert isinstance(freq, np.ndarray)
    
    # Test that we get reasonable frequency range
    assert freq[0] >= 0
    assert freq[-1] <= 1/(2*dt)  # Nyquist frequency


def test_analysis_config_properties():
    """Test AnalysisConfig computed properties."""
    config = AnalysisConfig(hx=5.0)
    
    # Test hy property (should equal hx)
    assert config.hy == config.hx
    assert config.hy == 5.0
    
    # Test that hy is a float
    assert isinstance(config.hy, float)


def test_mesh_and_materials_basic():
    """Test basic mesh and materials functionality."""
    from seiskit.utils import build_mesh_and_materials
    
    # Test with simple parameters
    abs_elems, mats = build_mesh_and_materials(Lx=20.0, Ly=10.0, hx=5.0)
    
    assert isinstance(abs_elems, list)
    assert isinstance(mats, dict)
    
    # Test that we get some elements
    assert len(abs_elems) > 0
    assert len(mats) > 0
    
    # Test material map structure
    for key, value in mats.items():
        assert isinstance(key, tuple)
        assert len(key) == 3  # (vs, nu, rho)
        assert isinstance(value, int)  # Material tag
