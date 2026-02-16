import numpy as np
from scipy.interpolate import interp1d

from .acc2FAS2 import acc2FAS2, acc2FAS_complex
from .kohmachi import kohmachi


def TTF_full(
    surface_acc,
    base_acc,
    dt=1e-4,
    n_points=1000,
    Vsmin=None,
    dz: float = 5,
    smooth_coeff: float = 500,
):
    """
    Transfer function between surface and base acceleration with magnitude and phase.

    Uses the same frequency range and Konno–Ohmachi smoothing as TTF(), but returns
    both magnitude and phase (radians) of the complex transfer function H(f) =
    FAS_surface(f) / FAS_base(f).

    Parameters
    ----------
    surface_acc : array_like
        Surface acceleration time history.
    base_acc : array_like
        Base acceleration time history.
    dt : float, optional
        Time step of the acceleration time history. Default 1e-4.
    n_points : int, optional
        Number of log-spaced frequency points. Default 1000.
    Vsmin : float, optional
        Minimum Vs (m/s) used to set maximum frequency: fmax = Vsmin / (10 * dz).
        If None, fmax = 10 Hz.
    dz : float, optional
        Depth increment (m) used for fmax. Default 5.
    smooth_coeff : float, optional
        Konno–Ohmachi smoothing coefficient. Lower = more smoothing. Default 500.

    Returns
    -------
    freq : np.ndarray
        Frequency vector (Hz).
    magnitude : np.ndarray
        Transfer function magnitude (smoothed with Konno–Ohmachi).
    phase : np.ndarray
        Transfer function phase in radians (unsmoothed).
    """
    if Vsmin is not None:
        fmax = Vsmin / (10 * dz)
    else:
        fmax = 10.0

    # Complex FAS (magnitude + phase) for surface and base
    FAS_s, phase_s, freq_raw = acc2FAS_complex(surface_acc, dt, 10**6)
    FAS_b, phase_b, _ = acc2FAS_complex(base_acc, dt, 10**6)

    # Log-spaced frequency grid (same as TTF)
    freq = np.logspace(np.log10(0.1), np.log10(fmax), n_points)

    # Interpolate magnitude and phase onto log-spaced grid
    interp_mag_s = interp1d(freq_raw, FAS_s, kind="linear", fill_value=0.0, bounds_error=False)
    interp_ph_s = interp1d(freq_raw, phase_s, kind="linear", fill_value=0.0, bounds_error=False)
    interp_mag_b = interp1d(freq_raw, FAS_b, kind="linear", fill_value=0.0, bounds_error=False)
    interp_ph_b = interp1d(freq_raw, phase_b, kind="linear", fill_value=0.0, bounds_error=False)

    FAS_s_log = interp_mag_s(freq)
    phase_s_log = interp_ph_s(freq)
    FAS_b_log = interp_mag_b(freq)
    phase_b_log = interp_ph_b(freq)

    # Complex spectra and transfer function H(f) = Y(f)/X(f)
    epsilon = 1e-12
    H_num = FAS_s_log * np.exp(1j * phase_s_log)
    H_den = FAS_b_log + epsilon
    H = np.divide(
        H_num,
        H_den * np.exp(1j * phase_b_log),
        out=np.zeros_like(H_num, dtype=complex),
        where=H_den != 0,
    )

    magnitude = np.abs(H)
    phase = np.angle(H)

    # Apply Konno–Ohmachi smoothing to magnitude only (match TTF behavior)
    smooth_coeff_int = int(smooth_coeff)
    magnitude = kohmachi(magnitude, freq, smooth_coeff_int)

    return freq, magnitude, phase


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
    # downsample (bounds_error=False to handle fmax at or just above freq[-1])
    f = interp1d(freq, FAS_s, bounds_error=False, fill_value=(FAS_s[0], FAS_s[-1]))
    FAS_s = f(np.logspace(np.log10(0.1), np.log10(fmax), n_points))

    # get FAS base
    FAS_b, freq = acc2FAS2(base_acc, dt, 10**6)
    # downsample
    f = interp1d(freq, FAS_b, bounds_error=False, fill_value=(FAS_b[0], FAS_b[-1]))
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
