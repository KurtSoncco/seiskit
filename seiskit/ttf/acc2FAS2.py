import numpy as np
from scipy.fft import fft

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

    #if acc.ndim == 1:
    #    acc = acc.reshape(-1, 1)

    if nfreq is None:
        n = numpts
    else:
        if numpts > nfreq:
            print('Warning: numpts > nfreq')
        n = nfreq

    fs = 1 / dt
    fnyq = 0.5 * fs
    df = 1 / (n * dt)
    freq = np.arange(0, fnyq, df)


    Acc = fft(acc, n=n, axis=0)
    FAS = (2 / numpts) * np.abs(Acc[:n // 2])

    FAS = FAS.reshape(-1,)
    freq = freq.reshape(-1,)
    
    return FAS, freq