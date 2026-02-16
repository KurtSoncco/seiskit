"""Tests for seiskit.ttf.kohmachi module."""

import numpy as np

from seiskit.ttf.kohmachi import kohmachi


def test_kohmachi_returns_same_length():
    """kohmachi returns array of same length as signal."""
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    freq = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    out = kohmachi(signal, freq, smooth_coeff=20)
    assert out.shape == signal.shape
    assert len(out) == len(signal)


def test_kohmachi_finite():
    """kohmachi output is finite."""
    signal = np.random.rand(10)
    freq = np.linspace(0.1, 5.0, 10)
    out = kohmachi(signal, freq, smooth_coeff=50)
    assert np.all(np.isfinite(out))


def test_kohmachi_boundary_values():
    """First and last values are set from neighbors."""
    signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    freq = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    out = kohmachi(signal, freq, smooth_coeff=20)
    assert out[0] == out[1]
    assert out[-1] == out[-2]
