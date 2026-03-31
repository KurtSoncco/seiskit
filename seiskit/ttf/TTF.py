import numpy as np
from scipy.interpolate import interp1d
from functools import lru_cache

from .acc2FAS2 import acc2FAS2, acc2FAS2_batch, acc2FAS_complex
from .kohmachi import kohmachi


def TTF_full(
    surface_acc,
    base_acc,
    dt=1e-4,
    n_points=1000,
    Vsmin=None,
    dz: float = 5,
    smooth_coeff: float = 500,
    nfreq=None,
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
    nfreq : int, optional
        FFT length for FAS. If None, uses next_fast_len(max(len(acc), 8192)).

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
    FAS_s, phase_s, freq_raw = acc2FAS_complex(surface_acc, dt, nfreq)
    FAS_b, phase_b, _ = acc2FAS_complex(base_acc, dt, nfreq)

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
    nfreq=None,
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
    nfreq : int, optional
        FFT length for FAS. If None, uses next_fast_len(max(len(acc), 8192)).

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
    FAS_s, freq = acc2FAS2(surface_acc, dt, nfreq)
    # downsample (bounds_error=False to handle fmax at or just above freq[-1])
    # fill_value tuple (below, above) is supported at runtime; stubs often type it as float only
    f = interp1d(
        freq,
        FAS_s,
        bounds_error=False,
        fill_value=(FAS_s[0], FAS_s[-1]),  # type: ignore[arg-type]
    )
    FAS_s = f(np.logspace(np.log10(0.1), np.log10(fmax), n_points))

    # get FAS base
    FAS_b, freq = acc2FAS2(base_acc, dt, nfreq)
    # downsample
    f = interp1d(
        freq,
        FAS_b,
        bounds_error=False,
        fill_value=(FAS_b[0], FAS_b[-1]),  # type: ignore[arg-type]
    )
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


def TTF_batch(
    base_2d,
    surf_2d,
    dt=1e-4,
    n_points=1000,
    Vsmin=None,
    dz: float = 5,
    smooth_coeff: float = 500,
    nfreq=None,
):
    """
    Batched transfer function between base and surface accelerations.

    Parameters
    ----------
    base_2d : array_like, shape (n_channels, n_time)
        Base acceleration time histories.
    surf_2d : array_like, shape (n_channels, n_time)
        Surface acceleration time histories.
    dt : float, optional
        Time step.
    n_points : int, optional
        Number of log-spaced frequency points.
    Vsmin : float, optional
        Minimum Vs for fmax.
    dz : float, optional
        Depth increment for fmax.
    smooth_coeff : float, optional
        Konno–Ohmachi smoothing coefficient.
    nfreq : int, optional
        FFT length for FAS.

    Returns
    -------
    freq : np.ndarray
        Log-spaced frequency vector (Hz).
    mag : np.ndarray, shape (n_channels, n_points)
        Transfer function magnitude per channel.
    """
    if Vsmin is not None:
        fmax = Vsmin / (10 * dz)
    else:
        fmax = 10.0

    FAS_base, freq_raw = acc2FAS2_batch(base_2d, dt, nfreq)
    FAS_surf, _ = acc2FAS2_batch(surf_2d, dt, nfreq)

    freq = np.logspace(np.log10(0.1), np.log10(fmax), n_points)
    n_ch = FAS_base.shape[0]
    smooth_coeff_int = int(smooth_coeff)

    mag = np.zeros((n_ch, len(freq)), dtype=np.float64)
    for i in range(n_ch):
        f_b = interp1d(
            freq_raw,
            FAS_base[i],
            bounds_error=False,
            fill_value=(FAS_base[i, 0], FAS_base[i, -1]),
        )
        f_s = interp1d(
            freq_raw,
            FAS_surf[i],
            bounds_error=False,
            fill_value=(FAS_surf[i, 0], FAS_surf[i, -1]),
        )
        fas_b_log = f_b(freq)
        fas_s_log = f_s(freq)
        k_s = kohmachi(fas_s_log, freq, smooth_coeff_int)
        k_b = kohmachi(fas_b_log, freq, smooth_coeff_int)
        epsilon = 1e-12
        mag[i] = np.divide(
            k_s,
            k_b + epsilon,
            out=np.zeros_like(k_s),
            where=(k_b + epsilon) != 0,
        )
    return freq, mag


@lru_cache(maxsize=8)
def _kohmachi_weight_norm(n_points: int, fmax: float, smooth_coeff_int: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Precompute Konno-Ohmachi normalized weight matrix for a fixed (freq grid, smooth_coeff).

    Returns:
      freq: (n_points,)
      Wnorm: (n_points, n_points) where Wnorm[i, j] is the normalized weight applied to x[j]
             to compute y[i] = sum_j Wnorm[i,j] * x[j].
    """
    freq = np.logspace(np.log10(0.1), np.log10(fmax), n_points)
    b = float(smooth_coeff_int)

    # Match kohmachi implementation: f_shifted = f / (1 + 1e-4)
    f_shifted = freq / (1.0 + 1e-4)

    # Vectorized weight computation:
    # z_j(i) = f_shifted[j] / f[i]
    # w_j(i) = ( sin(b*log10(z)) / (b*log10(z)) )^4 with special-case for log10(z) ~ 0
    z = f_shifted[None, :] / freq[:, None]  # (n_points, n_points)
    logz = np.log10(z)
    denom = b * logz

    w = np.zeros_like(logz, dtype=np.float64)
    mask = np.abs(logz) >= 1e-12
    # For masked values: (sin(denom)/denom)^4
    ratio = np.sin(denom[mask]) / denom[mask]
    w[mask] = ratio**4

    # Ensure numerical nans do not propagate
    w[np.isnan(w)] = 0.0

    sw = w.sum(axis=1)  # (n_points,)
    Wnorm = np.divide(w, sw[:, None], out=np.zeros_like(w), where=sw[:, None] != 0)

    # Match kohmachi endpoints behavior: y[0]=y[1], y[-1]=y[-2]
    if n_points >= 2:
        Wnorm[0, :] = Wnorm[1, :]
        Wnorm[-1, :] = Wnorm[-2, :]

    return freq, Wnorm


def _interp_linear_batch(y_raw: np.ndarray, x_raw: np.ndarray, x_target: np.ndarray) -> np.ndarray:
    """
    Vectorized linear interpolation along the last axis.

    Args:
      y_raw: (n_channels, n_raw)
      x_raw: (n_raw,)
      x_target: (n_target,)

    Returns:
      y_target: (n_channels, n_target)
    """
    if y_raw.ndim != 2:
        raise ValueError("y_raw must have shape (n_channels, n_raw)")
    x_raw = np.asarray(x_raw, dtype=np.float64)
    x_target = np.asarray(x_target, dtype=np.float64)

    # Indices of the right bin edges for each target x
    hi = np.searchsorted(x_raw, x_target, side="right")
    hi = np.clip(hi, 1, len(x_raw) - 1)
    lo = hi - 1

    x0 = x_raw[lo]  # (n_target,)
    x1 = x_raw[hi]  # (n_target,)

    y0 = y_raw[:, lo]  # (n_channels, n_target)
    y1 = y_raw[:, hi]  # (n_channels, n_target)

    denom = (x1 - x0)
    denom_safe = np.where(denom == 0.0, 1.0, denom)
    t = (x_target - x0) / denom_safe  # (n_target,)
    y = y0 + (y1 - y0) * t[None, :]

    # Match scipy interp1d fill_value=(y[0], y[-1]) outside bounds
    below = x_target <= x_raw[0]
    above = x_target >= x_raw[-1]
    if np.any(below):
        y[:, below] = y_raw[:, 0:1]
    if np.any(above):
        y[:, above] = y_raw[:, -1:]
    return y


def TTF_batch_fast(
    base_2d,
    surf_2d,
    dt: float = 1e-4,
    n_points: int = 1000,
    Vsmin=None,
    dz: float = 5,
    smooth_coeff: float = 500,
    nfreq=None,
):
    """
    Faster batched transfer function magnitude.

    Optimization vs `TTF_batch`:
      - vectorized interpolation onto the log-spaced frequency grid
      - batched Konno-Ohmachi smoothing using a precomputed weight matrix for the target freq grid
    """
    if Vsmin is not None:
        fmax = Vsmin / (10 * dz)
    else:
        fmax = 10.0

    smooth_coeff_int = int(smooth_coeff)
    freq, Wnorm = _kohmachi_weight_norm(n_points, float(fmax), smooth_coeff_int)

    FAS_base, freq_raw = acc2FAS2_batch(base_2d, dt, nfreq)
    FAS_surf, _ = acc2FAS2_batch(surf_2d, dt, nfreq)

    # Interpolate each channel's FAS magnitudes onto the common log-frequency grid.
    fas_b_log = _interp_linear_batch(FAS_base, freq_raw, freq)  # (n_ch, n_points)
    fas_s_log = _interp_linear_batch(FAS_surf, freq_raw, freq)  # (n_ch, n_points)

    # Batched Konno-Ohmachi smoothing:
    # k[i] = sum_j Wnorm[i,j] * fas[j]  -> y = fas @ Wnorm.T
    k_b = fas_b_log @ Wnorm.T
    k_s = fas_s_log @ Wnorm.T

    epsilon = 1e-12
    mag = k_s / (k_b + epsilon)
    return freq, mag
