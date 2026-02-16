"""Smoke tests for seiskit.plot_results module."""

import numpy as np

from seiskit.plot_results import plot_realization


def test_plot_realization_smoke(tmp_path):
    """plot_realization runs without error with minimal synthetic data."""
    vs_1d = np.array([200.0, 500.0])
    vs_realization = np.random.uniform(200.0, 250.0, (4, 6))
    save_path = tmp_path / "realization.png"
    plot_realization(
        vs_1d,
        vs_realization,
        Lx=30.0,
        Lz=20.0,
        dx=5.0,
        dz=5.0,
        save_path=str(save_path),
        title="Test",
    )
    assert save_path.exists()
