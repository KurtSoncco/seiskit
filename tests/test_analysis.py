from pathlib import Path

from seiskit.run_analysis import run_case_basic
from seiskit.utils import compute_ricker, build_mesh_and_materials
from seiskit.damping import compute_rayleigh_coefficients


def test_compute_ricker():
    samples = compute_ricker(0.75, 1.4, 2.0, 0.01)
    assert len(samples) > 0

def test_run_case_basic_with_opensees(tmp_path, monkeypatch):
    # Mock the openseespy.opensees module to simulate OpenSees availability
    import sys
    mock_ops = type('MockOps', (), {})()
    monkeypatch.setitem(sys.modules, "openseespy.opensees", mock_ops)
    
    # Also mock the perform_analysis function to return a success status
    from unittest.mock import patch
    with patch('seiskit.analysis.perform_analysis_spatial') as mock_perform:
        mock_perform.return_value = "Finished successfully"
        
        out = run_case_basic(
            "testcase", output_dir=str(tmp_path), duration=0.1, dt=0.01
        )
        assert Path(out["ts"]).exists()
        assert out["status"].startswith("Finished")

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