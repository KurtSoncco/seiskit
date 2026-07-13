"""Optional OpenSees integration: 2-layer AF matches closed-form theory."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("openseespy.opensees")

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "comparison" / "1D_theory_validation" / "run_experiment.py"


def _load_experiment():
    spec = importlib.util.spec_from_file_location("oned_theory_exp", EXP)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    import sys

    sys.modules["oned_theory_exp"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.opensees
def test_opensees_2layer_matches_theory(tmp_path, monkeypatch):
    """Short 2-layer undamped 1D run matches closed-form AF."""
    exp = _load_experiment()
    monkeypatch.setattr(exp, "OUT", tmp_path / "results")
    result = exp.run_case(
        exp.CASES[0],
        hx=2.0,
        dt=1e-3,
        duration=12.0,
        motion_freq=2.0,
        rock_buffer=2.0,
    )
    assert result["outcrop"]["median_rel_err"] < 0.05
    assert result["outcrop"]["p95_rel_err"] < 0.15
    assert result["peak_freq_rel_err"] < 0.03
    assert result["peak_amp_rel_err"] < 0.10
    assert np.isfinite(result["f_peak_os"])
