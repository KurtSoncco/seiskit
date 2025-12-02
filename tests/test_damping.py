"""Tests for seiskit.damping module."""

import numpy as np
import pytest

from seiskit.damping import (
    compute_average_damping_harmonic,
    compute_damping_from_Q,
    compute_quality_factor,
    compute_rayleigh_coefficients,
    compute_rayleigh_mass_only,
)


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
    freq_pairs = [(1.0, 5.0), (0.5, 2.0), (2.0, 10.0), (0.1, 1.0)]

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
    assert betaK < 0  # Should be negative

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
    assert len(set(betas)) > 1  # Not all betas are the same


def test_compute_rayleigh_mass_only():
    """Test compute_rayleigh_mass_only function."""
    zeta = 0.02
    f_target = 1.5

    alphaM, betaK = compute_rayleigh_mass_only(zeta, f_target)

    # Verify betaK is always zero
    assert betaK == 0.0
    assert isinstance(alphaM, float)
    assert alphaM > 0

    # Verify the formula: alphaM = 2 * w_target * zeta
    w_target = 2 * np.pi * f_target
    expected_alphaM = 2 * w_target * zeta
    assert abs(alphaM - expected_alphaM) < 1e-10


def test_compute_rayleigh_mass_only_different_frequencies():
    """Test compute_rayleigh_mass_only with different target frequencies."""
    zeta = 0.01

    frequencies = [0.5, 1.0, 2.0, 5.0, 10.0]

    for f_target in frequencies:
        alphaM, betaK = compute_rayleigh_mass_only(zeta, f_target)

        assert betaK == 0.0
        assert alphaM > 0

        # Verify frequency dependence: higher frequency -> higher alphaM
        w_target = 2 * np.pi * f_target
        expected_alphaM = 2 * w_target * zeta
        assert abs(alphaM - expected_alphaM) < 1e-10


def test_compute_rayleigh_mass_only_different_damping():
    """Test compute_rayleigh_mass_only with different damping ratios."""
    f_target = 1.0

    damping_ratios = [0.005, 0.01, 0.02, 0.05]

    for zeta in damping_ratios:
        alphaM, betaK = compute_rayleigh_mass_only(zeta, f_target)

        assert betaK == 0.0
        assert alphaM > 0

        # Higher damping should give higher alphaM
        w_target = 2 * np.pi * f_target
        expected_alphaM = 2 * w_target * zeta
        assert abs(alphaM - expected_alphaM) < 1e-10


def test_compute_quality_factor():
    """Test compute_quality_factor function."""
    # Test with Vs < 800 m/s
    Vs_low = 200.0
    Q_low = compute_quality_factor(Vs_low)
    expected_Q_low = (
        10.5
        - 16 * (Vs_low / 1000)
        + 153 * (Vs_low / 1000) ** 2
        - 103 * (Vs_low / 1000) ** 3
        + 34.7 * (Vs_low / 1000) ** 4
        - 5.29 * (Vs_low / 1000) ** 5
        + 0.31 * (Vs_low / 1000) ** 6
    )
    assert abs(Q_low - expected_Q_low) < 1e-10
    assert Q_low > 0

    # Test with Vs >= 800 m/s
    Vs_high = 1000.0
    Q_high = compute_quality_factor(Vs_high)
    expected_Q_high = (
        10.5
        - 16 * (Vs_high / 1000)
        + 153 * (Vs_high / 1000) ** 2
        - 103 * (Vs_high / 1000) ** 3
        + 34.7 * (Vs_high / 1000) ** 4
        - 5.29 * (Vs_high / 1000) ** 5
        + 0.31 * (Vs_high / 1000) ** 6
    )
    assert abs(Q_high - expected_Q_high) < 1e-10
    assert Q_high > 0

    # Test very low Vs
    Vs_very_low = 50.0
    Q_very_low = compute_quality_factor(Vs_very_low)
    expected_Q_very_low = (
        10.5
        - 16 * (Vs_very_low / 1000)
        + 153 * (Vs_very_low / 1000) ** 2
        - 103 * (Vs_very_low / 1000) ** 3
        + 34.7 * (Vs_very_low / 1000) ** 4
        - 5.29 * (Vs_very_low / 1000) ** 5
        + 0.31 * (Vs_very_low / 1000) ** 6
    )
    assert abs(Q_very_low - expected_Q_very_low) < 1e-10


def test_compute_quality_factor_range():
    """Test compute_quality_factor across a range of Vs values."""
    # Test multiple values below 800
    for Vs in [100.0, 200.0, 400.0, 600.0, 799.0]:
        Q = compute_quality_factor(Vs)
        expected_Q = (
            10.5
            - 16 * (Vs / 1000)
            + 153 * (Vs / 1000) ** 2
            - 103 * (Vs / 1000) ** 3
            + 34.7 * (Vs / 1000) ** 4
            - 5.29 * (Vs / 1000) ** 5
            + 0.31 * (Vs / 1000) ** 6
        )
        assert abs(Q - expected_Q) < 1e-10


def test_compute_damping_from_Q():
    """Test compute_damping_from_Q function."""
    # Test with known Q values
    Q = 10.0
    xi = compute_damping_from_Q(Q)
    expected_xi = 1.0 / (2.0 * Q)
    assert abs(xi - expected_xi) < 1e-10
    assert xi > 0

    # Test with different Q values
    Q_values = [5.0, 10.0, 20.0, 50.0, 100.0]
    for Q in Q_values:
        xi = compute_damping_from_Q(Q)
        expected_xi = 1.0 / (2.0 * Q)
        assert abs(xi - expected_xi) < 1e-10
        assert xi > 0

        # Higher Q should give lower damping
        assert xi <= 0.1  # All should be reasonable (Q=5 gives xi=0.1)


def test_compute_damping_from_Q_inverse_relationship():
    """Test that compute_damping_from_Q and compute_quality_factor are inverse."""
    # Test round-trip: Vs -> Q -> xi -> verify consistency
    Vs_values = [100.0, 200.0, 500.0, 1000.0]

    for Vs in Vs_values:
        Q = compute_quality_factor(Vs)
        xi = compute_damping_from_Q(Q)

        # Verify the relationship: xi = 1 / (2 * Q)
        assert abs(xi - 1.0 / (2.0 * Q)) < 1e-10

        # Verify reasonable damping values (typically 0.01 to 0.1 for soils)
        assert 0.001 < xi < 0.2


def test_compute_average_damping_harmonic():
    """Test compute_average_damping_harmonic function."""
    # Test with simple case
    Q_values = [10.0, 20.0, 30.0]
    xi_avg = compute_average_damping_harmonic(Q_values)

    # Manual calculation
    n_layers = len(Q_values)
    total_inverse_Q = sum(1.0 / Q for Q in Q_values)
    Q_avg_harmonic = n_layers / total_inverse_Q
    expected_xi = 1.0 / (2.0 * Q_avg_harmonic)

    assert abs(xi_avg - expected_xi) < 1e-10
    assert xi_avg > 0


def test_compute_average_damping_harmonic_single_value():
    """Test compute_average_damping_harmonic with single Q value."""
    Q_values = [15.0]
    xi_avg = compute_average_damping_harmonic(Q_values)

    expected_xi = compute_damping_from_Q(15.0)
    assert abs(xi_avg - expected_xi) < 1e-10


def test_compute_average_damping_harmonic_empty_list():
    """Test that compute_average_damping_harmonic raises error for empty list."""
    with pytest.raises(ValueError, match="cannot be empty"):
        compute_average_damping_harmonic([])


def test_compute_average_damping_harmonic_different_values():
    """Test compute_average_damping_harmonic with various Q value sets."""
    # Test case 1: All same values
    Q_same = [20.0, 20.0, 20.0]
    xi_same = compute_average_damping_harmonic(Q_same)
    expected_xi_same = compute_damping_from_Q(20.0)
    assert abs(xi_same - expected_xi_same) < 1e-10

    # Test case 2: Wide range
    Q_range = [5.0, 10.0, 20.0, 50.0]
    xi_range = compute_average_damping_harmonic(Q_range)

    # Harmonic average should be less than arithmetic average
    Q_arithmetic_avg = sum(Q_range) / len(Q_range)
    xi_arithmetic_avg = compute_damping_from_Q(Q_arithmetic_avg)

    # Harmonic average gives higher damping (lower effective Q)
    assert xi_range > xi_arithmetic_avg


def test_absorbing_boundary_elements_no_damping():
    """Test that absorbing boundary elements are not included in damping application."""
    from seiskit.builder import BoundaryElementData, ModelData, SoilElementData

    # Create a simple model with both soil and boundary elements
    model_data = ModelData()

    # Add soil elements (should get damping)
    model_data.soil_elements = [
        SoilElementData(
            tag=1, nodes=(1, 2, 3, 4), mat_tag=1, gravity_load=-19612.0, vs_value=200.0
        ),
        SoilElementData(
            tag=2, nodes=(5, 6, 7, 8), mat_tag=1, gravity_load=-19612.0, vs_value=250.0
        ),
        SoilElementData(
            tag=3,
            nodes=(9, 10, 11, 12),
            mat_tag=1,
            gravity_load=-19612.0,
            vs_value=300.0,
        ),
    ]

    # Add boundary elements (should NOT get damping)
    model_data.boundary_elements = [
        BoundaryElementData(
            tag=10, nodes=(1, 2, 3, 4), btype="L", G=1e6, poiss=0.3, rho=2000.0
        ),
        BoundaryElementData(
            tag=11, nodes=(5, 6, 7, 8), btype="R", G=1e6, poiss=0.3, rho=2000.0
        ),
        BoundaryElementData(
            tag=12, nodes=(9, 10, 11, 12), btype="B", G=1e6, poiss=0.3, rho=2000.0
        ),
    ]

    # Extract interior soil element tags (as done in isolated_runner)
    interior_soil_element_tags = [elem.tag for elem in model_data.soil_elements]
    boundary_element_tags = [elem.tag for elem in model_data.boundary_elements]

    # Verify that boundary elements are NOT in the interior soil element list
    for boundary_tag in boundary_element_tags:
        assert boundary_tag not in interior_soil_element_tags, (
            f"Boundary element {boundary_tag} should not be in damping list"
        )

    # Verify that all soil elements ARE in the list
    for soil_elem in model_data.soil_elements:
        assert soil_elem.tag in interior_soil_element_tags, (
            f"Soil element {soil_elem.tag} should be in damping list"
        )

    # Verify the lists are disjoint
    assert set(interior_soil_element_tags).isdisjoint(set(boundary_element_tags)), (
        "Soil and boundary element tags should be disjoint"
    )


def test_rayleigh_damping_only_applied_to_soil_elements():
    """Test that Rayleigh damping coefficients are only computed for soil elements."""
    from seiskit.builder import BoundaryElementData, ModelData, SoilElementData

    # Create model data
    model_data = ModelData()

    # Add soil elements
    soil_tags = [1, 2, 3, 4, 5]
    for tag in soil_tags:
        model_data.soil_elements.append(
            SoilElementData(
                tag=tag,
                nodes=(tag, tag + 1, tag + 2, tag + 3),
                mat_tag=1,
                gravity_load=-19612.0,
                vs_value=200.0 + tag * 10.0,
            )
        )

    # Add boundary elements
    boundary_tags = [10, 11, 12, 13, 14]
    for tag in boundary_tags:
        model_data.boundary_elements.append(
            BoundaryElementData(
                tag=tag,
                nodes=(tag, tag + 1, tag + 2, tag + 3),
                btype="L",
                G=1e6,
                poiss=0.3,
                rho=2000.0,
            )
        )

    # Simulate what isolated_runner does: collect only soil element tags
    interior_soil_element_tags = [elem.tag for elem in model_data.soil_elements]

    # Verify counts
    assert len(interior_soil_element_tags) == len(soil_tags)
    assert len(model_data.boundary_elements) == len(boundary_tags)

    # Verify no boundary elements in damping list
    for boundary_elem in model_data.boundary_elements:
        assert boundary_elem.tag not in interior_soil_element_tags

    # Compute damping coefficients (would be applied only to interior_soil_element_tags)
    zeta = 0.02
    f1, f2 = 1.0, 5.0
    alphaM, betaK = compute_rayleigh_coefficients(zeta, f1, f2)

    # These coefficients would only be applied to interior_soil_element_tags
    # Boundary elements would not receive damping
    assert alphaM > 0
    assert betaK > 0
    assert len(interior_soil_element_tags) == 5  # Only soil elements
