from pathlib import Path

import numpy as np
import pytest

from seiskit.utils import compute_ricker


def test_compute_ricker():
    samples = compute_ricker(0.75, 1.4, 2.0, 0.01)
    assert len(samples) > 0


def test_compute_ricker_shape():
    samples = compute_ricker(0.75, 1.4, 2.0, 0.01)
    expected_shape = (int(2.0 / 0.01) + 1,)
    assert samples.shape == expected_shape


def test_compute_ricker_values():
    samples = compute_ricker(0.75, 1.4, 15.0, 0.01)
    # Use tolerance to avoid fragile exact float equality
    expected_first = -0.0003904965392413389
    assert np.isclose(samples[0], expected_first, atol=1e-12, rtol=0)
    assert abs(samples[-1]) < 1e-10


def test_with_real_data(tmp_path):
    ricker_in = Path(__file__).parent / "ricker.in"
    if not ricker_in.exists():
        pytest.skip("tests/ricker.in not found; required for validation against reference data")

    samples = compute_ricker(freq=0.75, t_shift=1.4, duration=15.0, dt=0.01)
    real_data = np.loadtxt(ricker_in, skiprows=1)

    difference = samples - real_data
    # Write plot to tmp_path so CI and local runs don't leave artifacts under tests/
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.plot(difference)
    plt.savefig(tmp_path / "ricker_difference.png")
    plt.close()

    mae = np.mean(np.abs(samples - real_data))
    mse = np.mean(np.square(samples - real_data))
    assert mae < 1e-8
    assert mse < 1e-8
