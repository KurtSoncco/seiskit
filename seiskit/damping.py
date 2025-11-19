"""Utilities for calculating damping coefficients and material properties.

This module contains functions for computing Rayleigh damping coefficients,
quality factors (Q), and other damping-related calculations.
"""

import numpy as np


def compute_rayleigh_coefficients(
    zeta: float, f1: float, f2: float
) -> tuple[float, float]:
    """Calculate Rayleigh damping coefficients alphaM and betaK.

    Uses a matrix solve approach which is more general and handles edge cases
    better than the simplified formula.

    Args:
        zeta: Damping ratio (e.g., 0.0075 for 0.75%)
        f1: First frequency in Hz
        f2: Second frequency in Hz

    Returns:
        Tuple of (alphaM, betaK) coefficients

    Raises:
        ValueError: If the matrix is singular (e.g., f1 == f2) or both frequencies are zero
    """
    # Handle edge case where one frequency is zero
    # If f1 is zero, use only f2 (mass-proportional damping limit)
    if f1 == 0.0 and f2 != 0.0:
        w2 = 2 * np.pi * f2
        # In the limit as w1 -> 0, alphaM dominates
        # We solve: zeta = alphaM / (2*w2) + betaK * w2 / 2
        # For w1=0, we need another constraint. Use betaK = 0 (mass-proportional only)
        alphaM = 2 * w2 * zeta
        betaK = 0.0
        return alphaM, betaK

    # If f2 is zero, use only f1 (stiffness-proportional damping limit)
    if f2 == 0.0 and f1 != 0.0:
        w1 = 2 * np.pi * f1
        # In the limit as w2 -> 0, betaK dominates
        # We solve: zeta = alphaM / (2*w1) + betaK * w1 / 2
        # For w2=0, we need another constraint. Use alphaM = 0 (stiffness-proportional only)
        alphaM = 0.0
        betaK = 2 * zeta / w1
        return alphaM, betaK

    # If both are zero, raise an error
    if f1 == 0.0 and f2 == 0.0:
        raise ValueError(
            "Both frequencies cannot be zero. At least one frequency must be positive."
        )

    # Convert frequencies to rad/s
    w1 = 2 * np.pi * f1
    w2 = 2 * np.pi * f2

    # Check if frequencies are too close (would cause singular matrix)
    if abs(f1 - f2) < 1e-10:
        raise ValueError(
            f"Frequencies f1={f1} Hz and f2={f2} Hz are too close or equal. "
            f"They must be distinct for Rayleigh damping calculation."
        )

    # Use matrix solve approach (more general)
    A = 0.5 * np.array([[1 / w1, w1], [1 / w2, w2]])
    b = np.array([zeta, zeta])

    try:
        coeffs = np.linalg.solve(A, b)
        alphaM, betaK = coeffs[0], coeffs[1]
    except np.linalg.LinAlgError:
        raise ValueError(
            f"Singular matrix in Rayleigh calculation. "
            f"Frequencies f1={f1} Hz and f2={f2} Hz may be too close or equal."
        )

    return alphaM, betaK


def compute_rayleigh_mass_only(zeta: float, f_target: float) -> tuple[float, float]:
    """Calculate mass-proportional Rayleigh damping coefficients.

    Calculates alpha_M for mass-proportional damping at ONE target frequency.
    Sets beta_K to zero. This is useful when you want frequency-independent
    damping that scales with mass only.

    Args:
        zeta: Damping ratio (e.g., 0.0075 for 0.75%)
        f_target: Target frequency in Hz

    Returns:
        Tuple of (alphaM, betaK) where betaK = 0.0
    """
    w_target = 2 * np.pi * f_target
    alphaM = 2 * w_target * zeta
    betaK = 0.0
    return alphaM, betaK


def compute_quality_factor(Vs: float) -> float:
    """Calculate quality factor Q from shear wave velocity.
    Model 2 by Campbell (2009)

    Uses a common empirical relationship for soil materials:
    - For Vs < 800 m/s: Q = 7.17 + 0.00276 * Vs
    - For Vs >= 800 m/s: Q = 50.0

    Args:
        Vs: Shear wave velocity in m/s

    Returns:
        Quality factor Q (dimensionless)
    """
    if Vs >= 800.0:
        return 50.0
    else:
        return 7.17 + 0.00276 * Vs


def compute_damping_from_Q(Q: float) -> float:
    """Calculate damping ratio from quality factor.

    The relationship is: xi = 1 / (2 * Q)

    Args:
        Q: Quality factor

    Returns:
        Damping ratio xi
    """
    return 1.0 / (2.0 * Q)


def compute_average_damping_harmonic(Q_values: list[float]) -> float:
    """Calculate harmonic average damping from a list of Q values.

    This is useful when you have multiple layers with different Q values
    and want to compute an effective average damping.

    Args:
        Q_values: List of quality factors

    Returns:
        Average damping ratio xi_avg
    """
    if not Q_values:
        raise ValueError("Q_values list cannot be empty")

    n_layers = len(Q_values)
    total_inverse_Q = sum(1.0 / Q for Q in Q_values)
    Q_avg_harmonic = n_layers / total_inverse_Q
    return compute_damping_from_Q(Q_avg_harmonic)


def print_rayleigh_coefficients(
    zeta: float, f1: float, f2: float, alphaM: float, betaK: float
) -> None:
    """Print formatted Rayleigh damping coefficients.

    Args:
        zeta: Damping ratio used
        f1: First frequency used
        f2: Second frequency used
        alphaM: Mass damping coefficient
        betaK: Stiffness damping coefficient
    """
    print(f"Damping ratio: {zeta} ({zeta * 100:.2f}%)")
    print(f"Frequencies: {f1} Hz and {f2} Hz")
    print(f"Calculated alphaM: {alphaM}")
    print(f"Calculated betaK: {betaK}")


# Example usage
if __name__ == "__main__":
    zeta = 0.0075  # 0.75% damping
    f1 = 0.75  # Hz
    f2 = 2.25  # Hz

    alphaM, betaK = compute_rayleigh_coefficients(zeta, f1, f2)
    print_rayleigh_coefficients(zeta, f1, f2, alphaM, betaK)

    # Test mass-only damping
    alphaM_mass, betaK_mass = compute_rayleigh_mass_only(zeta, f1)
    print(f"\nMass-only damping: alphaM={alphaM_mass}, betaK={betaK_mass}")

    # Test Q factor calculations
    Vs_test = 200.0
    Q = compute_quality_factor(Vs_test)
    xi = compute_damping_from_Q(Q)
    print(f"\nFor Vs={Vs_test} m/s: Q={Q:.2f}, xi={xi:.6f}")
