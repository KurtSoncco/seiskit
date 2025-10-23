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
    # ModelData is a dataclass with specific fields, not vs/rho/nu
    model_data = ModelData()
    
    # Test that we can add nodes and elements
    from seiskit.builder import NodeData, SoilElementData
    
    model_data.nodes = [NodeData(1, 0.0, 0.0)]
    model_data.soil_elements = [SoilElementData(1, (1, 2, 3, 4), 1, 0.0)]
    
    assert len(model_data.nodes) == 1
    assert len(model_data.soil_elements) == 1


def test_build_model_data():
    """Test build_model_data function."""
    config = AnalysisConfig(Lx=100.0, Ly=50.0, hx=5.0)
    
    # Create sample material data with correct shape
    # For Lx=100, Ly=50, hx=5: expected shape is (10, 20) for soil grid
    vs_data = np.random.uniform(200.0, 400.0, (10, 20))  # ny, nx
    rho_data = np.random.uniform(1800.0, 2000.0, (10, 20))
    nu_data = np.full((10, 20), 0.3)
    
    model_data = build_model_data(config, vs_data, rho_data, nu_data)
    
    assert isinstance(model_data, ModelData)
    assert len(model_data.nodes) > 0
    assert len(model_data.soil_elements) > 0
    assert len(model_data.material_map) > 0


def test_build_model_data_shape_validation():
    """Test build_model_data with different input shapes."""
    config = AnalysisConfig(Lx=20.0, Ly=10.0, hx=5.0)
    
    # Test with correct 2D arrays for the domain size
    # For Lx=20, Ly=10, hx=5: expected shape is (2, 4) for soil grid
    vs_data = np.array([[200.0, 250.0, 300.0, 350.0], 
                       [180.0, 230.0, 280.0, 330.0]])
    rho_data = np.array([[1800.0, 1900.0, 2000.0, 2100.0],
                        [1750.0, 1850.0, 1950.0, 2050.0]])
    nu_data = np.full((2, 4), 0.3)
    
    model_data = build_model_data(config, vs_data, rho_data, nu_data)
    
    assert isinstance(model_data, ModelData)
    assert len(model_data.nodes) > 0
    assert len(model_data.soil_elements) > 0


def test_model_data_properties():
    """Test ModelData computed properties."""
    model_data = ModelData()
    
    # Test that we can add properties
    from seiskit.builder import NodeData, SoilElementData
    
    model_data.nodes = [NodeData(1, 0.0, 0.0), NodeData(2, 1.0, 0.0)]
    model_data.soil_elements = [SoilElementData(1, (1, 2, 3, 4), 1, 0.0)]
    model_data.material_map = {(200.0, 0.3, 1800.0): 1}
    
    # Test that all properties are accessible
    assert len(model_data.nodes) == 2
    assert len(model_data.soil_elements) == 1
    assert len(model_data.material_map) == 1


def test_model_data_repr():
    """Test ModelData string representation."""
    model_data = ModelData()
    
    from seiskit.builder import NodeData
    model_data.nodes = [NodeData(1, 0.0, 0.0)]
    
    repr_str = repr(model_data)
    assert "ModelData" in repr_str
