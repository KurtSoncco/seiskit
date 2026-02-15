import numpy as np
from scipy.interpolate import interp1d

from .acc2FAS2 import acc2FAS2
from .kohmachi import kohmachi


def TTF(
    surface_acc,
    base_acc,
    dt=1e-4,
    n_points=1000,
    Vsmin=None,
    dz: float = 5,
    smooth_coeff: float = 500,
):
    """
    Transfer function between surface and base acceleration

    Parameters
    ----------
    surface_acc : array_like
        Surface acceleration time history
    base_acc : array_like
        Base acceleration time history
    dt : float, optional
        Time step of the acceleration time history, by default 0.01
    n_points : int, optional
        Number of points to downsample the frequency, by default 1000
    Vsmin : float, optional
        Minimum Vs value to calculate the maximum frequency, by default None
    dz : float, optional
        Depth increment used in the model, by default 5.0
    smooth_coeff : float, optional
        Smoothing coefficient for kohmachi smoothing, by default 500.
        Lower values result in more smoothing.

    Returns
    -------
    freq : array_like
        Frequency vector
    TF : array_like
        Transfer function between surface and base acceleration
    """

    # Calculation of maximum frequency
    if Vsmin is not None:
        fmax = Vsmin / (10 * dz)
    else:
        fmax = 10.0  # Updating to 10 Hz

    # get FAS surface
    FAS_s, freq = acc2FAS2(surface_acc, dt, 10**6)
    # downsample
    f = interp1d(freq, FAS_s)
    FAS_s = f(np.logspace(np.log10(0.1), np.log10(fmax), n_points))

    # get FAS base
    FAS_b, freq = acc2FAS2(base_acc, dt, 10**6)
    # downsample
    f = interp1d(freq, FAS_b)
    FAS_b = f(np.logspace(np.log10(0.1), np.log10(fmax), n_points))

    # define downsampled freq
    freq = np.logspace(np.log10(0.1), np.log10(fmax), n_points)

    # get TF
    assert isinstance(smooth_coeff, int)
    kohmachi_s = kohmachi(FAS_s, freq, smooth_coeff)
    kohmachi_b = kohmachi(FAS_b, freq, smooth_coeff)

    # Handle division by zero by adding a small epsilon
    epsilon = 1e-12
    TF = np.divide(
        kohmachi_s,
        kohmachi_b + epsilon,
        out=np.zeros_like(kohmachi_s),
        where=(kohmachi_b + epsilon) != 0,
    )
    # TF = kohmachi(FAS_s, freq, 150) / kohmachi(FAS_b, freq, 150)

    return freq, TF
