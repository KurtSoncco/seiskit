"""Tests for seiskit.isolated_runner module."""

import numpy as np

from seiskit.builder import build_model_data
from seiskit.config import AnalysisConfig
from seiskit.isolated_runner import run_isolated_analysis


def test_run_isolated_analysis_smoke(tmp_path):
    """Smoke test: run_isolated_analysis returns a status string.

    Without OpenSees returns 'No OpenSees - {run_id}'; with OpenSees may return
    'Finished...' or an error message. This documents behavior and catches regressions.
    """
    config = AnalysisConfig(Lx=20.0, Ly=10.0, hx=5.0, duration=0.1)
    vs = np.full((2, 4), 200.0)
    rho = np.full((2, 4), 1800.0)
    nu = np.full((2, 4), 0.3)
    model_data = build_model_data(config, vs, rho, nu)
    status = run_isolated_analysis(config, model_data, "smoke_run", output_dir=str(tmp_path))
    assert isinstance(status, str)
    assert status.startswith("No OpenSees") or status.startswith("Finished") or "Failed" in status


def test_apply_damping_global_avg_base_override(monkeypatch):
    """xi_soil_base / xi_rock_base replace ξ_Q and are scaled by dmin_multiplier."""
    import seiskit.isolated_runner as runner
    from seiskit.damping import compute_rayleigh_coefficients

    calls = {}

    class FakeOps:
        @staticmethod
        def region(tag, *args):
            calls[tag] = args[-4:-2]

    monkeypatch.setattr(runner, "ops", FakeOps)
    config = AnalysisConfig(
        Lx=5.0, Ly=10.0, hx=5.0, dmin_multiplier=4.0, xi_soil_base=0.008, xi_rock_base=0.005
    )
    vs = np.array([[1500.0], [200.0]])
    mask = np.array([[True], [False]])
    model_data = build_model_data(
        config, vs, np.full_like(vs, 2000.0), np.full_like(vs, 0.3), bedrock_mask=mask
    )
    runner._apply_damping(config, model_data, [])
    f1, f2 = config.damping_freqs
    assert np.allclose(calls[1], compute_rayleigh_coefficients(0.032, f1, f2))
    assert np.allclose(calls[2], compute_rayleigh_coefficients(0.020, f1, f2))
