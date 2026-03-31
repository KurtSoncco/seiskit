import numpy as np

from seiskit import optools
from seiskit.ttf.acc2FAS2 import acc2FAS2, acc2FAS_complex
from seiskit.ttf.TTF import TTF, TTF_batch, TTF_full


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


def test_ttf_vs_ttf_batch_single_pair():
    """TTF and TTF_batch with a single channel should match."""
    np.random.seed(42)
    n_time = 500
    dt = 0.02
    base_acc = np.random.randn(n_time).astype(np.float64) * 0.1
    surf_acc = base_acc * 2 + np.random.randn(n_time).astype(np.float64) * 0.05

    freq_tf, mag_tf = TTF(surf_acc, base_acc, dt=dt)
    base_2d = base_acc[np.newaxis, :]
    surf_2d = surf_acc[np.newaxis, :]
    freq_batch, mag_batch = TTF_batch(base_2d, surf_2d, dt=dt)

    np.testing.assert_allclose(freq_tf, freq_batch)
    np.testing.assert_allclose(mag_tf, mag_batch[0], rtol=1e-10, atol=1e-10)


def test_ttf_nfreq_auto_vs_large_on_1500pts():
    """TTF with nfreq=None (auto) vs nfreq=10**6 on 1500-pt signal: acceptable agreement in 0.1–10 Hz."""
    np.random.seed(42)
    n_time = 1500
    dt = 0.02
    base_acc = np.random.randn(n_time).astype(np.float64) * 0.1
    surf_acc = base_acc * 1.5 + np.random.randn(n_time).astype(np.float64) * 0.05

    freq_auto, mag_auto = TTF(surf_acc, base_acc, dt=dt, nfreq=None)
    freq_lg, mag_lg = TTF(surf_acc, base_acc, dt=dt, nfreq=10**6)

    np.testing.assert_allclose(freq_auto, freq_lg)
    # In 0.1–10 Hz band, exclude near-zero magnitude (unstable rel err) and check median
    mask = (freq_auto >= 0.1) & (freq_auto <= 10) & (np.abs(mag_lg) > 1e-4)
    if mask.sum() > 0:
        rel_err = np.abs(mag_auto[mask] - mag_lg[mask]) / (np.abs(mag_lg[mask]) + 1e-12)
        assert np.median(rel_err) < 0.05, (
            f"Median relative error {np.median(rel_err):.4f} exceeds 5%"
        )


def test_acc2fas2_batch_shape():
    """acc2FAS2_batch returns correct shapes."""
    from seiskit.ttf.acc2FAS2 import acc2FAS2_batch

    n_ch, n_time = 5, 200
    acc = np.random.randn(n_ch, n_time).astype(np.float64)
    fas, freq = acc2FAS2_batch(acc, dt=0.02)
    assert fas.shape[0] == n_ch
    assert fas.shape[1] == len(freq)
    assert np.all(np.isfinite(fas))


def test_read_time_series(tmp_path):
    p = tmp_path / "vals.txt"
    p.write_text("# comment\n0.1\n0.2\n\n0.3\n")
    vals = optools.read_time_series(str(p))
    assert vals == [0.1, 0.2, 0.3]
