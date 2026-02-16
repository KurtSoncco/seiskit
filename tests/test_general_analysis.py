"""Tests for seiskit.general_analysis re-exports."""

from seiskit import general_analysis


def test_general_analysis_re_exports():
    """general_analysis re-exports expected names from analysis and utils."""
    assert hasattr(general_analysis, "run_analysis")
    assert hasattr(general_analysis, "perform_analysis_spatial")
    assert hasattr(general_analysis, "load_material_properties")
    assert hasattr(general_analysis, "build_mesh_and_materials")
    assert hasattr(general_analysis, "compute_ricker")
    assert callable(general_analysis.run_analysis)
    assert callable(general_analysis.perform_analysis_spatial)
    assert callable(general_analysis.compute_ricker)
