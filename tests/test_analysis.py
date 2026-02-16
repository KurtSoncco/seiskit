from pathlib import Path
from unittest.mock import patch

import numpy as np

from seiskit.analysis import run_opensees_analysis
from seiskit.builder import ModelData, build_model_data
from seiskit.config import AnalysisConfig
from seiskit.damping import compute_rayleigh_coefficients
from seiskit.run_analysis import run_case_basic
from seiskit.utils import build_mesh_and_materials, compute_ricker


def test_compute_ricker():
    samples = compute_ricker(0.75, 1.4, 2.0, 0.01)
    assert len(samples) > 0


def test_run_case_basic_smoke(tmp_path):
    """Smoke test that doesn't require OpenSees.

    Ensures the time series file is created and the function returns a status.
    Accepts either 'no-opensees' (when OpenSees isn't installed) or a
    'Finished...' status when available.
    """
    out = run_case_basic("testcase", output_dir=str(tmp_path), duration=0.1, dt=0.01)
    assert Path(out["ts"]).exists()
    assert isinstance(out["status"], str)
    assert out["status"] == "no-opensees" or out["status"].startswith("Finished")


def test_build_mesh_and_materials_default():
    abs_elems, mats = build_mesh_and_materials(Lx=100.0, Ly=40.0, hx=5.0)
    assert isinstance(abs_elems, list)
    assert isinstance(mats, dict)


def test_rayleigh_damping():
    damping_zeta = 0.0075
    damping_freqs = (0.75, 2.25)

    expected_alphaM = 0.053014376
    expected_betaK = 0.000795775
    alphaM, betaK = compute_rayleigh_coefficients(damping_zeta, damping_freqs[0], damping_freqs[1])
    assert abs(alphaM - expected_alphaM) < 1e-8
    assert abs(betaK - expected_betaK) < 1e-8


def test_run_opensees_analysis_forwards_to_isolated_runner(tmp_path):
    """run_opensees_analysis returns the string returned by run_isolated_analysis."""
    config = AnalysisConfig(Lx=20.0, Ly=10.0, hx=5.0)
    vs = np.full((2, 4), 200.0)
    rho = np.full((2, 4), 1800.0)
    nu = np.full((2, 4), 0.3)
    model_data = build_model_data(config, vs, rho, nu)
    with patch("seiskit.isolated_runner.run_isolated_analysis", return_value="Finished OK") as m:
        status = run_opensees_analysis(config, model_data, "test_run", output_dir=str(tmp_path))
    assert status == "Finished OK"
    m.assert_called_once()
    args, kwargs = m.call_args
    assert args[2] == "test_run"
    assert (kwargs.get("output_dir") or (args[3] if len(args) > 3 else None)) == str(tmp_path)


def test_perform_analysis_spatial_calls_run_opensees_analysis(tmp_path):
    """perform_analysis_spatial builds config/model and calls run_opensees_analysis."""
    from seiskit.analysis import perform_analysis_spatial

    vs = np.full((2, 4), 250.0)
    rho = np.full((2, 4), 1900.0)
    nu = np.full((2, 4), 0.3)
    # Ensure code path reaches run_opensees_analysis (ops is not None)
    with (
        patch("seiskit.analysis.ops", object()),
        patch("seiskit.analysis.run_opensees_analysis", return_value="Finished") as m,
    ):
        status = perform_analysis_spatial(
            "spatial_test",
            vs_data=vs,
            rho_data=rho,
            nu_data=nu,
            output_dir=str(tmp_path),
            Ly=10.0,
            Lx=20.0,
            hx=5.0,
        )
    assert status == "Finished"
    m.assert_called_once()
    args = m.call_args[0]
    assert isinstance(args[0], AnalysisConfig)
    assert isinstance(args[1], ModelData)
    assert args[2] == "spatial_test"
