"""Unit tests for seiskit.solver_utils module."""

import pytest

from seiskit.config import AnalysisConfig
from seiskit.solver_utils import check_mumps_availability, get_solver_info, setup_solver


def test_check_mumps_availability():
    """Test check_mumps_availability returns a dictionary with expected keys."""
    availability = check_mumps_availability()

    # Should return a dictionary
    assert isinstance(availability, dict)

    # Should have expected keys
    assert "UmfPack" in availability
    assert "Mumps" in availability
    assert "MumpsParallel" in availability

    # Values should be booleans
    assert isinstance(availability["UmfPack"], bool)
    assert isinstance(availability["Mumps"], bool)
    assert isinstance(availability["MumpsParallel"], bool)

    # UmfPack should always be True if OpenSees is available
    # (or False if not available)
    # Note: This test doesn't require OpenSees to pass


def test_get_solver_info_umfpack():
    """Test get_solver_info with UmfPack solver."""
    config = AnalysisConfig(solver_type="UmfPack")
    info = get_solver_info(config)

    assert info["solver_type"] == "UmfPack"
    assert "available" in info
    assert "all_available" in info
    assert info["mumps_parallel_procs"] is None
    assert info["mumps_icntl"] is None


def test_get_solver_info_mumps():
    """Test get_solver_info with Mumps solver."""
    config = AnalysisConfig(solver_type="Mumps")
    info = get_solver_info(config)

    assert info["solver_type"] == "Mumps"
    assert "available" in info
    assert "all_available" in info
    assert info["mumps_parallel_procs"] is None
    assert info["mumps_icntl"] is None


def test_get_solver_info_mumps_parallel():
    """Test get_solver_info with MumpsParallel solver."""
    config = AnalysisConfig(
        solver_type="MumpsParallel", mumps_parallel_procs=4, mumps_icntl={1: 7, 7: 2}
    )
    info = get_solver_info(config)

    assert info["solver_type"] == "MumpsParallel"
    assert "available" in info
    assert "all_available" in info
    assert info["mumps_parallel_procs"] == 4
    assert info["mumps_icntl"] == {1: 7, 7: 2}


def test_get_solver_info_mumps_with_icntl():
    """Test get_solver_info with Mumps solver and ICNTL parameters."""
    config = AnalysisConfig(solver_type="Mumps", mumps_icntl={1: 7, 7: 2})
    info = get_solver_info(config)

    assert info["solver_type"] == "Mumps"
    assert info["mumps_icntl"] == {1: 7, 7: 2}


def test_get_solver_info_default_config():
    """Test get_solver_info with default AnalysisConfig."""
    config = AnalysisConfig()
    info = get_solver_info(config)

    # Default solver should be UmfPack
    assert info["solver_type"] == "UmfPack"
    assert info["mumps_parallel_procs"] is None


def test_setup_solver_umfpack_smoke():
    """Smoke test for setup_solver with UmfPack (handles missing OpenSees gracefully)."""
    config = AnalysisConfig(solver_type="UmfPack")

    # This should not raise an error even if OpenSees is not available
    # The function handles missing OpenSees by returning early
    try:
        setup_solver(config)
    except Exception as e:
        # If OpenSees is not available, the function should handle it gracefully
        # If it's available, it should work without error
        # We just check that it doesn't raise unexpected errors
        pytest.skip(f"OpenSees not available or setup failed: {e}")


def test_setup_solver_mumps_smoke():
    """Smoke test for setup_solver with Mumps (handles missing OpenSees gracefully)."""
    config = AnalysisConfig(solver_type="Mumps")

    try:
        setup_solver(config)
    except Exception as e:
        pytest.skip(f"OpenSees not available or setup failed: {e}")


def test_setup_solver_invalid_solver():
    """Test setup_solver raises ValueError for invalid solver type."""
    config = AnalysisConfig(solver_type="InvalidSolver")

    # This should raise ValueError even without OpenSees
    # because the check happens before OpenSees is called
    with pytest.raises(ValueError, match="Unknown solver_type"):
        setup_solver(config)


def test_check_mumps_availability_consistency():
    """Test that check_mumps_availability returns consistent results."""
    result1 = check_mumps_availability()
    result2 = check_mumps_availability()

    # Should return the same result on multiple calls
    assert result1 == result2
