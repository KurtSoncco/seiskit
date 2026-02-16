import numpy as np

from seiskit import optools
from seiskit.ttf.acc2FAS2 import acc2FAS2, acc2FAS_complex
from seiskit.ttf.TTF import TTF, TTF_full


def test_ttf_empty():
    # Test with minimal input to avoid division by zero
    surface_acc = [0.0, 0.0, 0.0]
    base_acc = [0.0, 0.0, 0.0]
    freq, tf = TTF(surface_acc, base_acc, dt=0.01)
    assert len(freq) > 0  # Should still return frequency array


def test_ttf_simple_sine():
    # create a simple sine wave at 1 Hz sampled at 20 Hz
    import math

    fs = 20.0
    dt = 1.0 / fs
    n = 40
    surface_acc = [math.sin(2 * math.pi * 1.0 * (i * dt)) for i in range(n)]
    base_acc = [0.5 * math.sin(2 * math.pi * 1.0 * (i * dt)) for i in range(n)]

    freq, tf = TTF(surface_acc, base_acc, dt=dt)
    # expect non-empty output
    assert len(freq) > 0
    assert len(tf) > 0


def test_acc2fas_functionality():
    data = [0.0, 1.0, 0.0, -1.0]
    fas, freq = acc2FAS2(data, 0.1)
    assert len(fas) > 0
    assert len(freq) > 0
    assert len(fas) == len(freq)


def test_acc2fas_complex_returns_magnitude_and_phase():
    data = np.random.randn(128)
    fas_mag, phase, freq = acc2FAS_complex(data, 0.01)
    assert len(fas_mag) == len(phase) == len(freq)
    assert np.all(np.isfinite(fas_mag))
    assert np.all(np.isfinite(phase))
    assert np.all(np.abs(phase) <= np.pi)


def test_ttf_full_returns_magnitude_and_phase():
    surface_acc = np.sin(2 * np.pi * 1.0 * np.arange(200) * 0.01)
    base_acc = 0.5 * np.sin(2 * np.pi * 1.0 * np.arange(200) * 0.01)
    freq, magnitude, phase = TTF_full(surface_acc, base_acc, dt=0.01)
    assert len(freq) == len(magnitude) == len(phase)
    assert np.all(np.isfinite(magnitude))
    assert np.all(np.isfinite(phase))
    assert np.all(magnitude >= 0)
    assert np.all(np.abs(phase) <= np.pi)


def test_read_time_series(tmp_path):
    p = tmp_path / "vals.txt"
    p.write_text("# comment\n0.1\n0.2\n\n0.3\n")
    vals = optools.read_time_series(str(p))
    assert vals == [0.1, 0.2, 0.3]
