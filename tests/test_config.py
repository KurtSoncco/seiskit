"""Tests for seiskit.config module."""

import pytest
from seiskit.config import AnalysisConfig


def test_analysis_config_defaults():
    """Test AnalysisConfig with default values."""
    config = AnalysisConfig()
    
    # Test that all required fields have defaults
    assert hasattr(config, 'Ly')
    assert hasattr(config, 'Lx')
    assert hasattr(config, 'hx')
    assert hasattr(config, 'duration')
    assert hasattr(config, 'motion_freq')
    
    # Test default values
    assert config.Ly == 140.0
    assert config.Lx == 260.0
    assert config.hx == 5.0
    assert config.duration == 15.0
    assert config.motion_freq == 0.75


def test_analysis_config_custom_values():
    """Test AnalysisConfig with custom values."""
    config = AnalysisConfig(
        Ly=200.0,
        Lx=300.0,
        hx=2.5,
        duration=20.0,
        motion_freq=1.0
    )
    
    assert config.Ly == 200.0
    assert config.Lx == 300.0
    assert config.hx == 2.5
    assert config.duration == 20.0
    assert config.motion_freq == 1.0


def test_analysis_config_validation():
    """Test AnalysisConfig parameter validation."""
    # AnalysisConfig is a dataclass, so it accepts any values
    # The validation would need to be added to the dataclass if needed
    config = AnalysisConfig(Ly=-10.0)
    assert config.Ly == -10.0
    
    config = AnalysisConfig(hx=0.0)
    assert config.hx == 0.0
    
    config = AnalysisConfig(duration=-5.0)
    assert config.duration == -5.0


def test_analysis_config_repr():
    """Test AnalysisConfig string representation."""
    config = AnalysisConfig(Ly=100.0, Lx=200.0)
    repr_str = repr(config)
    assert "AnalysisConfig" in repr_str
    assert "Ly=100.0" in repr_str
    assert "Lx=200.0" in repr_str


def test_analysis_config_equality():
    """Test AnalysisConfig equality comparison."""
    config1 = AnalysisConfig(Ly=100.0, Lx=200.0)
    config2 = AnalysisConfig(Ly=100.0, Lx=200.0)
    config3 = AnalysisConfig(Ly=150.0, Lx=200.0)
    
    assert config1 == config2
    assert config1 != config3


def test_analysis_config_copy():
    """Test AnalysisConfig copying."""
    config1 = AnalysisConfig(Ly=100.0, Lx=200.0)
    config2 = AnalysisConfig(**config1.__dict__)
    
    assert config1 == config2
    assert config1 is not config2  # Different objects
