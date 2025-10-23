"""Tests for seiskit.damping module."""

import pytest
import numpy as np
from seiskit.damping import compute_rayleigh_coefficients


def test_compute_rayleigh_coefficients():
    """Test compute_rayleigh_coefficients function."""
    damping_zeta = 0.0075
    freq1 = 0.75
    freq2 = 2.25
    
    alphaM, betaK = compute_rayleigh_coefficients(damping_zeta, freq1, freq2)
    
    assert isinstance(alphaM, float)
    assert isinstance(betaK, float)
    
    # Test that both coefficients are positive
    assert alphaM > 0
    assert betaK > 0
    
    # Test with known values (from existing test)
    expected_alphaM = 0.053014376
    expected_betaK = 0.000795775
    assert abs(alphaM - expected_alphaM) < 1e-8
    assert abs(betaK - expected_betaK) < 1e-8


def test_compute_rayleigh_coefficients_different_frequencies():
    """Test compute_rayleigh_coefficients with different frequency pairs."""
    damping_zeta = 0.05
    
    # Test different frequency ranges
    freq_pairs = [
        (1.0, 5.0),
        (0.5, 2.0),
        (2.0, 10.0),
        (0.1, 1.0)
    ]
    
    for freq1, freq2 in freq_pairs:
        alphaM, betaK = compute_rayleigh_coefficients(damping_zeta, freq1, freq2)
        
        assert alphaM > 0
        assert betaK > 0
        assert isinstance(alphaM, float)
        assert isinstance(betaK, float)


def test_compute_rayleigh_coefficients_different_damping():
    """Test compute_rayleigh_coefficients with different damping ratios."""
    freq1 = 1.0
    freq2 = 3.0
    
    damping_ratios = [0.01, 0.02, 0.05, 0.1]
    
    for damping_zeta in damping_ratios:
        alphaM, betaK = compute_rayleigh_coefficients(damping_zeta, freq1, freq2)
        
        assert alphaM > 0
        assert betaK > 0
        
        # Higher damping should generally result in higher coefficients
        # (though the relationship is complex)


def test_compute_rayleigh_coefficients_edge_cases():
    """Test compute_rayleigh_coefficients edge cases."""
    damping_zeta = 0.02
    
    # Test with very close frequencies
    freq1 = 1.0
    freq2 = 1.01
    alphaM, betaK = compute_rayleigh_coefficients(damping_zeta, freq1, freq2)
    
    assert alphaM > 0
    assert betaK > 0
    
    # Test with very different frequencies
    freq1 = 0.1
    freq2 = 10.0
    alphaM, betaK = compute_rayleigh_coefficients(damping_zeta, freq1, freq2)
    
    assert alphaM > 0
    assert betaK > 0


def test_compute_rayleigh_coefficients_mathematical_properties():
    """Test mathematical properties of Rayleigh coefficients."""
    damping_zeta = 0.05
    freq1 = 1.0
    freq2 = 5.0
    
    alphaM, betaK = compute_rayleigh_coefficients(damping_zeta, freq1, freq2)
    
    # Test that the coefficients satisfy the Rayleigh damping equation
    # For any frequency f, the damping ratio should be:
    # zeta = (alphaM / (4*pi*f)) + (betaK * pi * f)
    
    test_frequencies = [0.5, 1.0, 2.0, 3.0, 5.0]
    
    for f in test_frequencies:
        computed_zeta = (alphaM / (4 * np.pi * f)) + (betaK * np.pi * f)
        
        # The computed damping should be close to the target damping
        # at the reference frequencies
        if f == freq1 or f == freq2:
            assert abs(computed_zeta - damping_zeta) < 1e-6


def test_compute_rayleigh_coefficients_input_validation():
    """Test input validation for compute_rayleigh_coefficients."""
    # Test with negative damping (should still work but give negative coefficients)
    alphaM, betaK = compute_rayleigh_coefficients(-0.01, 1.0, 3.0)
    assert alphaM < 0  # Should be negative
    assert betaK < 0   # Should be negative
    
    # Test with zero frequencies (should still work, just gives specific values)
    alphaM, betaK = compute_rayleigh_coefficients(0.02, 0.0, 3.0)
    assert isinstance(alphaM, float)
    assert isinstance(betaK, float)
    
    alphaM, betaK = compute_rayleigh_coefficients(0.02, 1.0, 0.0)
    assert isinstance(alphaM, float)
    assert isinstance(betaK, float)


def test_compute_rayleigh_coefficients_symmetric_properties():
    """Test symmetric properties of Rayleigh coefficients."""
    damping_zeta = 0.03
    
    # Test that swapping frequencies gives the same coefficients
    # (Rayleigh damping is symmetric with respect to frequency order)
    alphaM1, betaK1 = compute_rayleigh_coefficients(damping_zeta, 1.0, 3.0)
    alphaM2, betaK2 = compute_rayleigh_coefficients(damping_zeta, 3.0, 1.0)
    
    # The coefficients should be the same when frequencies are swapped
    assert alphaM1 == alphaM2
    assert betaK1 == betaK2
    
    # Both should be positive
    assert alphaM1 > 0 and alphaM2 > 0
    assert betaK1 > 0 and betaK2 > 0


def test_compute_rayleigh_coefficients_frequency_dependence():
    """Test how coefficients vary with frequency range."""
    damping_zeta = 0.02
    
    # Test with increasing frequency range
    freq_pairs = [(1.0, 2.0), (1.0, 5.0), (1.0, 10.0)]
    alphas = []
    betas = []
    
    for freq1, freq2 in freq_pairs:
        alphaM, betaK = compute_rayleigh_coefficients(damping_zeta, freq1, freq2)
        alphas.append(alphaM)
        betas.append(betaK)
    
    # Test that we get different coefficients for different frequency ranges
    assert len(set(alphas)) > 1  # Not all alphas are the same
    assert len(set(betas)) > 1   # Not all betas are the same
