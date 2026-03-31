import numpy as np
from scipy.fft import fft, next_fast_len


def _default_nfreq(numpts: int) -> int:
    """FFT length for FAS: fast length >= max(8192, numpts) to avoid wasteful zero-padding."""
    return next_fast_len(max(numpts, 8192))  # type: ignore


def acc2FAS_complex(acc, dt, nfreq=None):
    """
    Convert acceleration time history to Fourier Amplitude Spectrum (FAS) and phase.

    Parameters
    ----------
    acc : array_like
        Acceleration time history.
    dt : float
        Time step.
    nfreq : int, optional
        Number of frequency points (FFT length). Default is len(acc).

    Returns
    -------
    FAS : np.ndarray
        One-sided Fourier amplitude spectrum (magnitude).
    phase : np.ndarray
        Phase in radians for positive frequencies.
    freq : np.ndarray
        Frequency vector (Hz) corresponding to FAS and phase.
    """
    numpts = len(acc)
    if nfreq is None:
        n = _default_nfreq(numpts)
    else:
        n = nfreq

    fs = 1 / dt
    fnyq = 0.5 * fs
    df = 1 / (n * dt)
    n_half = n // 2
    freq = np.arange(0, fnyq, df)[:n_half]

    Acc = np.asarray(fft(acc, n=n, axis=0))
    # One-sided: positive frequencies only
    Acc_one = Acc[:n_half]
    FAS = (2 / numpts) * np.abs(Acc_one)
    phase = np.angle(Acc_one)

    FAS = np.asarray(FAS).reshape(-1)
    phase = np.asarray(phase).reshape(-1)
    freq = np.asarray(freq).reshape(-1)
    return FAS, phase, freq


def acc2FAS2(acc, dt, nfreq=None):
    """
    Convert acceleration time history to Fourier Amplitude Spectrum (FAS)

    Parameters:
    acc (array): Acceleration time history
    dt (float): Time step
    nfreq (int): Number of frequency points for FAS (default is length of acc)

    Returns:
    tuple: FAS, freq, Phase, acc_t
    FAS (array): Fourier amplitude spectrum (unit acc)
    freq (array): Frequency vector corresponding to FAS
    """

    numpts = len(acc)

    if nfreq is None:
        n = _default_nfreq(numpts)
    else:
        if numpts > nfreq:
            print("Warning: numpts > nfreq")
        n = nfreq

    fs = 1 / dt
    fnyq = 0.5 * fs
    df = 1 / (n * dt)
    freq = np.arange(0, fnyq, df)

    Acc = np.asarray(fft(acc, n=n, axis=0))
    n_half = n // 2
    FAS = (2 / numpts) * np.abs(Acc[:n_half])

    FAS = np.asarray(FAS).reshape(-1)
    freq = np.asarray(freq).reshape(-1)
    # np.arange can overshoot by 1 due to floating point; ensure same length as FAS
    if len(freq) != len(FAS):
        freq = freq[: len(FAS)]

    return FAS, freq


def acc2FAS2_batch(acc, dt, nfreq=None):
    """
    Batched conversion of acceleration time histories to Fourier Amplitude Spectrum (FAS).

    Parameters
    ----------
    acc : array_like, shape (n_channels, n_time)
        Acceleration time histories. Time is along the last axis.
    dt : float
        Time step.
    nfreq : int, optional
        FFT length. If None, uses next_fast_len(max(n_time, 8192)).

    Returns
    -------
    FAS : np.ndarray, shape (n_channels, n_freq)
        One-sided Fourier amplitude spectrum per channel.
    freq : np.ndarray
        Frequency vector (Hz) corresponding to FAS.
    """
    acc = np.asarray(acc)
    if acc.ndim != 2:
        raise ValueError("acc must be 2D (n_channels, n_time)")
    numpts = acc.shape[1]
    n_channels = acc.shape[0]
    assert n_channels > 0, "acc must have at least one channel"

    if nfreq is None:
        n = _default_nfreq(numpts)
    else:
        n = nfreq

    fs = 1 / dt
    fnyq = 0.5 * fs
    df = 1 / (n * dt)
    n_half = n // 2
    freq = np.arange(0, fnyq, df)[:n_half]

    Acc = np.asarray(fft(acc, n=n, axis=1))
    Acc_one = Acc[:, :n_half]
    FAS = (2 / numpts) * np.abs(Acc_one)

    freq = np.asarray(np.arange(0, fnyq, df)[:n_half], dtype=np.float64).ravel()
    if len(freq) != FAS.shape[1]:
        freq = freq[: FAS.shape[1]]
    return FAS.astype(np.float64), freq
