"""Tests for seiskit.builder module."""

import pytest
import numpy as np
from seiskit.builder import ModelData, build_model_data
from seiskit.config import AnalysisConfig


def test_model_data_creation():
    """Test ModelData creation and properties."""
    # Create sample data
    from seiskit.builder import NodeData, SoilElementData, BoundaryElementData
    
    model_data = ModelData()
    
    # Add some test data
    model_data.nodes = [NodeData(1, 0.0, 0.0), NodeData(2, 1.0, 0.0)]
    model_data.soil_elements = [SoilElementData(1, (1, 2, 3, 4), 1, 0.0)]
    model_data.material_map = {(200.0, 0.3, 1800.0): 1}
    model_data.abs_element_tags = [1, 2]
    
    assert len(model_data.nodes) == 2
    assert len(model_data.soil_elements) == 1
    assert len(model_data.material_map) == 1
    assert len(model_data.abs_element_tags) == 2


def test_model_data_validation():
    """Test ModelData input validation."""
    vs_data = np.array([200.0, 250.0, 300.0])
    rho_data = np.array([1800.0, 1900.0, 2000.0])
    nu_data = np.array([0.3, 0.3, 0.3])
    
    # Test with mismatched array sizes
    with pytest.raises((ValueError, AssertionError)):
        ModelData(
            vs=vs_data,
            rho=np.array([1800.0, 1900.0]),  # Different size
            nu=nu_data,
            nx=3,
            ny=3
        )
    
    # Test with negative values
    with pytest.raises((ValueError, AssertionError)):
        ModelData(
            vs=np.array([-200.0, 250.0, 300.0]),  # Negative vs
            rho=rho_data,
            nu=nu_data,
            nx=3,
            ny=3
        )


def test_build_model_data():
    """Test build_model_data function."""
    config = AnalysisConfig(Lx=100.0, Ly=50.0, hx=5.0)
    
    # Create sample material data
    vs_data = np.random.uniform(200.0, 400.0, (10, 20))  # ny, nx
    rho_data = np.random.uniform(1800.0, 2000.0, (10, 20))
    nu_data = np.full((10, 20), 0.3)
    
    model_data = build_model_data(config, vs_data, rho_data, nu_data)
    
    assert isinstance(model_data, ModelData)
    assert model_data.nx == 20
    assert model_data.ny == 10
    assert np.array_equal(model_data.vs, vs_data.flatten())
    assert np.array_equal(model_data.rho, rho_data.flatten())
    assert np.array_equal(model_data.nu, nu_data.flatten())


def test_build_model_data_shape_validation():
    """Test build_model_data with different input shapes."""
    config = AnalysisConfig(Lx=100.0, Ly=50.0, hx=5.0)
    
    # Test with 1D arrays
    vs_data = np.array([200.0, 250.0, 300.0, 350.0])
    rho_data = np.array([1800.0, 1900.0, 2000.0, 2100.0])
    nu_data = np.array([0.3, 0.3, 0.3, 0.3])
    
    model_data = build_model_data(config, vs_data, rho_data, nu_data)
    
    assert isinstance(model_data, ModelData)
    assert len(model_data.vs) == 4
    assert len(model_data.rho) == 4
    assert len(model_data.nu) == 4


def test_model_data_properties():
    """Test ModelData computed properties."""
    vs_data = np.array([200.0, 250.0, 300.0])
    rho_data = np.array([1800.0, 1900.0, 2000.0])
    nu_data = np.array([0.3, 0.3, 0.3])
    
    model_data = ModelData(
        vs=vs_data,
        rho=rho_data,
        nu=nu_data,
        nx=3,
        ny=3
    )
    
    # Test that all arrays are numpy arrays
    assert isinstance(model_data.vs, np.ndarray)
    assert isinstance(model_data.rho, np.ndarray)
    assert isinstance(model_data.nu, np.ndarray)
    
    # Test data types
    assert model_data.vs.dtype == np.float64
    assert model_data.rho.dtype == np.float64
    assert model_data.nu.dtype == np.float64


def test_model_data_repr():
    """Test ModelData string representation."""
    vs_data = np.array([200.0, 250.0])
    rho_data = np.array([1800.0, 1900.0])
    nu_data = np.array([0.3, 0.3])
    
    model_data = ModelData(
        vs=vs_data,
        rho=rho_data,
        nu=nu_data,
        nx=2,
        ny=1
    )
    
    repr_str = repr(model_data)
    assert "ModelData" in repr_str
    assert "nx=2" in repr_str
    assert "ny=1" in repr_str
