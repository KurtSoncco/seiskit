"""Tests for seiskit.recorders module."""

from pathlib import Path
from unittest.mock import patch

from seiskit.config import AnalysisConfig
from seiskit.recorders import print_recorder_summary, setup_recorders


def test_setup_recorders_returns_error_when_opensees_unavailable():
    """When ops is None, setup_recorders returns a dict with 'error'."""
    from seiskit import recorders

    config = AnalysisConfig()
    output_path = Path("/tmp/out")
    with patch.object(recorders, "ops", None):
        result = setup_recorders(config, ndivx=10, ndivy=5, output_path=output_path)
    assert isinstance(result, dict)
    assert "error" in result
    assert "OpenSees" in result["error"] or "not available" in result["error"].lower()


def test_print_recorder_summary_with_error_dict(capsys):
    """print_recorder_summary with error dict prints message and returns."""
    print_recorder_summary({"error": "OpenSees not available"})
    out, _ = capsys.readouterr()
    assert "Recorder setup failed" in out or "error" in out.lower()


def test_print_recorder_summary_with_success_dict(capsys):
    """print_recorder_summary with success dict prints summary."""
    info = {
        "center_nodes": [1, 2],
        "surface_nodes": [10, 11, 12],
        "row_nodes": [],
        "files_created": [],
        "total_recorders": 2,
    }
    print_recorder_summary(info)
    out, _ = capsys.readouterr()
    assert "Total recorders" in out or "2" in out
