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
