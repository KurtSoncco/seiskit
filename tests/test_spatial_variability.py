"""Tests for seiskit.spatial_variability module."""

import numpy as np

from seiskit.spatial_variability import Base_case_extraction


def test_base_case_extraction():
    """Base_case_extraction returns Vs1, Vs2, h from Vs array."""
    # Two-layer profile: 3 cells of 200, 2 cells of 500
    vs_array = np.array([200.0, 200.0, 200.0, 500.0, 500.0])
    Vs1, Vs2, h = Base_case_extraction(vs_array, dz=5)
    assert Vs1 == 200.0
    assert Vs2 == 500.0
    assert h == 15.0  # 3 * 5


def test_base_case_extraction_single_layer():
    """Base_case_extraction with two unique values."""
    vs_array = np.array([100.0, 100.0, 400.0, 400.0])
    Vs1, Vs2, h = Base_case_extraction(vs_array, dz=2)
    assert Vs1 == 100.0
    assert Vs2 == 400.0
    assert h == 4.0  # 2 * 2
