from pathlib import Path

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
    alphaM, betaK = compute_rayleigh_coefficients(
        damping_zeta, damping_freqs[0], damping_freqs[1]
    )
    assert abs(alphaM - expected_alphaM) < 1e-8
    assert abs(betaK - expected_betaK) < 1e-8
