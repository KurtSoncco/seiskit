"""Utilities for calculating damping coefficients.

This module contains functions for computing Rayleigh damping coefficients
and other damping-related calculations.
"""

import numpy as np


def compute_rayleigh_coefficients(
    zeta: float, f1: float, f2: float
) -> tuple[float, float]:
    """Calculate Rayleigh damping coefficients alphaM and betaK.
    
    Args:
        zeta: Damping ratio (e.g., 0.0075 for 0.75%)
        f1: First frequency in Hz
        f2: Second frequency in Hz
        
    Returns:
        Tuple of (alphaM, betaK) coefficients
    """
    # Convert frequencies to rad/s
    w1 = 2 * np.pi * f1
    w2 = 2 * np.pi * f2
    
    # Calculate Rayleigh coefficients
    alphaM = zeta * (2 * w1 * w2) / (w1 + w2)
    betaK = zeta * 2 / (w1 + w2)
    
    return alphaM, betaK


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
    print(f"Damping ratio: {zeta} ({zeta*100:.2f}%)")
    print(f"Frequencies: {f1} Hz and {f2} Hz")
    print(f"Calculated alphaM: {alphaM}")
    print(f"Calculated betaK: {betaK}")


# Example usage from the original compute_coeff.py
if __name__ == "__main__":
    zeta = 0.0075  # 0.75% damping
    f1 = 0.75      # Hz
    f2 = 2.25      # Hz
    
    alphaM, betaK = compute_rayleigh_coefficients(zeta, f1, f2)
    print_rayleigh_coefficients(zeta, f1, f2, alphaM, betaK)
