import numpy as np

try:
    import numba

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def _kohmachi_numpy(signal, freq_array, smooth_coeff: float):
    """Pure NumPy implementation (fallback when Numba unavailable)."""
    x = np.asarray(signal, dtype=np.float64)
    f = np.asarray(freq_array, dtype=np.float64)
    f_shifted = f / (1 + 1e-4)
    L = len(x)
    y = np.zeros(L)

    for i in range(L):
        if i != 0 and i != L - 1:
            z = f_shifted / f[i]
            w = (np.sin(smooth_coeff * np.log10(z)) / (smooth_coeff * np.log10(z))) ** 4
            w[np.isnan(w)] = 0
            y[i] = np.dot(w, x) / np.sum(w)

    y[0] = y[1]
    y[L - 1] = y[L - 2]
    return y


if HAS_NUMBA:

    @numba.jit(nopython=True, cache=True)
    def _kohmachi_numba(x, f, smooth_coeff):
        f_shifted = f / (1.0 + 1e-4)
        L = len(x)
        y = np.zeros(L)

        for i in range(1, L - 1):
            z = f_shifted / f[i]
            w = np.empty(L)
            for j in range(L):
                logz = np.log10(z[j])
                if np.abs(logz) < 1e-12:
                    w[j] = 0.0
                else:
                    val = np.sin(smooth_coeff * logz) / (smooth_coeff * logz)
                    w[j] = val**4
            sw = np.sum(w)
            if sw > 0:
                y[i] = np.dot(w, x) / sw
            else:
                y[i] = x[i]

        y[0] = y[1]
        y[L - 1] = y[L - 2]
        return y


def kohmachi(signal, freq_array, smooth_coeff: float = 500):
    """
    Efficient way of smoothing low-frequency microtremor signals.
    Original paper:
        K. Konno & T. Ohmachi (1998) "Ground-motion characteristics estimated
        from spectral ratio between horizontal and vertical components of
        microtremor." Bulletin of the Seismological Society of America.
        Vol.88, No.1, 228-241.

    Parameters:
    signal (array-like): Signal to be smoothed in frequency domain.
    freq_array (array-like): Frequency array corresponding to the signal.
                             It must have the same length as "signal".
    smooth_coeff (float): A parameter determining the degree of smoothing.
                          The lower this parameter, the more the signal
                          is smoothed.

    Returns:
    y (numpy array): Smoothed signal.
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    f = np.asarray(freq_array, dtype=np.float64).ravel()
    if len(x) != len(f):
        raise ValueError("signal and freq_array must have the same length")
    b = float(smooth_coeff)

    if HAS_NUMBA:
        return _kohmachi_numba(x, f, b)
    return _kohmachi_numpy(signal, freq_array, b)
