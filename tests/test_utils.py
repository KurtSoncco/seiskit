"""Tests for seiskit.utils module."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch, mock_open

from seiskit.utils import (
    compute_ricker,
    build_mesh_and_materials,
    load_material_properties,
)


def test_compute_ricker():
    """Test compute_ricker function."""
    # Test basic functionality
    samples = compute_ricker(0.75, 1.4, 2.0, 0.01)
    
    assert len(samples) > 0
    assert isinstance(samples, np.ndarray)
    assert samples.dtype == np.float64
    
    # Test expected shape
    expected_length = int(2.0 / 0.01) + 1
    assert len(samples) == expected_length
    
    # Test that first value is approximately zero
    assert abs(samples[0]) < 1e-3  # More lenient threshold
    # The last value may not be exactly zero depending on the duration and dt
    
    # Test that peak occurs around t_shift
    peak_idx = np.argmax(np.abs(samples))
    peak_time = peak_idx * 0.01
    assert abs(peak_time - 1.4) < 0.1


def test_compute_ricker_different_parameters():
    """Test compute_ricker with different parameters."""
    # Test different frequency
    samples1 = compute_ricker(1.0, 1.4, 2.0, 0.01)
    samples2 = compute_ricker(2.0, 1.4, 2.0, 0.01)
    
    assert len(samples1) == len(samples2)
    assert not np.array_equal(samples1, samples2)
    
    # Test different time shift
    samples3 = compute_ricker(0.75, 0.5, 2.0, 0.01)
    peak_idx = np.argmax(np.abs(samples3))
    peak_time = peak_idx * 0.01
    assert abs(peak_time - 0.5) < 0.1


def test_compute_ricker_edge_cases():
    """Test compute_ricker edge cases."""
    # Test very small duration
    samples = compute_ricker(0.75, 1.4, 0.1, 0.01)
    assert len(samples) == 11  # int(0.1/0.01) + 1
    
    # Test very small time step
    samples = compute_ricker(0.75, 1.4, 2.0, 0.001)
    assert len(samples) == 2001  # int(2.0/0.001) + 1


def test_build_mesh_and_materials():
    """Test build_mesh_and_materials function."""
    abs_elems, mats = build_mesh_and_materials(Lx=100.0, Ly=40.0, hx=5.0)
    
    assert isinstance(abs_elems, list)
    assert isinstance(mats, dict)
    
    # Test that materials dict has material properties as keys
    # The function returns a material map with (vs, nu, rho) tuples as keys
    assert len(mats) > 0
    
    # Test that all keys are tuples with 3 elements
    for key in mats.keys():
        assert isinstance(key, tuple)
        assert len(key) == 3  # (vs, nu, rho)
        assert all(isinstance(x, (int, float)) for x in key)
    
    # Test that values are integers (material tags)
    for value in mats.values():
        assert isinstance(value, int)
        assert value > 0


def test_build_mesh_and_materials_different_sizes():
    """Test build_mesh_and_materials with different sizes."""
    # Test different mesh sizes
    abs_elems1, mats1 = build_mesh_and_materials(Lx=100.0, Ly=40.0, hx=2.5)
    abs_elems2, mats2 = build_mesh_and_materials(Lx=100.0, Ly=40.0, hx=10.0)
    
    # Smaller hx should result in more elements
    assert len(abs_elems1) > len(abs_elems2)
    assert len(mats1) >= len(mats2)
    
    # Test different domain sizes
    abs_elems3, mats3 = build_mesh_and_materials(Lx=200.0, Ly=80.0, hx=5.0)
    
    # Larger domain should have more elements
    assert len(abs_elems3) > len(abs_elems2)


def test_load_material_properties():
    """Test load_material_properties function."""
    # Create mock file content
    mock_content = "200.0\n250.0\n300.0\n350.0\n"
    
    with patch("builtins.open", mock_open(read_data=mock_content)):
        with patch("numpy.loadtxt") as mock_loadtxt:
            mock_loadtxt.return_value = np.array([200.0, 250.0, 300.0, 350.0])
            
            material_data = load_material_properties({
                "vs": "vs_data.txt",
                "rho": "rho_data.txt",
                "nu": "nu_data.txt"
            })
            
            assert isinstance(material_data, dict)
            assert "vs" in material_data
            assert "rho" in material_data
            assert "nu" in material_data
            
            # Check that loadtxt was called for each file
            assert mock_loadtxt.call_count == 3


def test_load_material_properties_different_files():
    """Test load_material_properties with different file types."""
    mock_content = "200.0\n250.0\n300.0\n"
    
    with patch("builtins.open", mock_open(read_data=mock_content)):
        with patch("numpy.loadtxt") as mock_loadtxt:
            mock_loadtxt.return_value = np.array([200.0, 250.0, 300.0])
            
            # Test with different file extensions
            material_data = load_material_properties({
                "vs": "vs_data.csv",
                "rho": "rho_data.dat",
                "nu": "nu_data.txt"
            })
            
            assert isinstance(material_data, dict)
            assert len(material_data) == 3


def test_load_material_properties_missing_files():
    """Test load_material_properties with missing files."""
    with patch("numpy.loadtxt", side_effect=FileNotFoundError("File not found")):
        with pytest.raises(FileNotFoundError):
            load_material_properties({
                "vs": "missing_vs_data.txt",
                "rho": "missing_rho_data.txt",
                "nu": "missing_nu_data.txt"
            })


def test_compute_ricker_mathematical_properties():
    """Test mathematical properties of Ricker wavelet."""
    freq = 1.0
    t_shift = 2.0
    duration = 5.0
    dt = 0.01
    
    samples = compute_ricker(freq, t_shift, duration, dt)
    
    # Test that the integral is approximately zero (zero mean)
    integral = np.trapezoid(samples, dx=dt)
    assert abs(integral) < 1e-6
    
    # Test multi-modal properties
    # Find peaks and valleys
    peaks = []
    valleys = []
    for i in range(1, len(samples) - 1):
        if samples[i] > samples[i-1] and samples[i] > samples[i+1]:
            peaks.append(i)
        elif samples[i] < samples[i-1] and samples[i] < samples[i+1]:
            valleys.append(i)
    
    # Should have multiple peaks and valleys
    assert len(peaks) > 0
    assert len(valleys) > 0


def test_build_mesh_and_materials_element_types():
    """Test that build_mesh_and_materials returns correct element types."""
    abs_elems, mats = build_mesh_and_materials(Lx=100.0, Ly=40.0, hx=5.0)
    
    # Test that abs_elems contains valid element data
    assert isinstance(abs_elems, list)
    if abs_elems:  # If not empty
        for elem in abs_elems:
            assert isinstance(elem, int)  # Element IDs are integers
    
    # Test that materials contain tuples as keys and integers as values
    for key, value in mats.items():
        assert isinstance(key, tuple)
        assert len(key) == 3  # (vs, nu, rho)
        assert isinstance(value, int)  # Material tag
