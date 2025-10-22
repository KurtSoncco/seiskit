import numpy as np 

def kohmachi(signal, freq_array, smooth_coeff=500):
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

    x = np.asarray(signal)
    f = np.asarray(freq_array)
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