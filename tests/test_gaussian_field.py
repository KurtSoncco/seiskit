"""Tests for seiskit.gaussian_field module."""

import numpy as np

from seiskit.gaussian_field import create_vs_realization, generate_gaussian_field_fft


def test_generate_gaussian_field_fft_shape():
    """generate_gaussian_field_fft returns array of expected shape."""
    nx, nz = 4, 6
    rng = np.random.default_rng(42)
    out = generate_gaussian_field_fft(nx, nz, 1.0, 1.0, 1.0, 1.0, rng=rng)
    assert out.shape == (nz, nx)
    assert np.all(np.isfinite(out))


def test_create_vs_realization_returns_expected_shapes():
    """create_vs_realization returns (final_Vs, x_total, z, h, bedrock_mask) with correct shapes."""
    Vs_profile = np.array([200.0, 800.0])
    Lx = 40.0
    Lx_variability = 40.0
    Lz = 30.0
    dx = 5.0
    dz = 5.0
    rH = 10.0
    aHV = 1.0
    CV = 0.2
    result = create_vs_realization(
        Vs_profile=Vs_profile,
        Lx=Lx,
        Lx_variability=Lx_variability,
        Lz=Lz,
        dx=dx,
        dz=dz,
        rH=rH,
        aHV=aHV,
        CV=CV,
        seed=42,
    )
    final_Vs, x_total, z, h_out, bedrock_mask = result
    nz = int(Lz / dz)
    assert final_Vs.shape[0] == nz
    assert final_Vs.shape[1] >= int(Lx / dx)
    assert isinstance(z, np.ndarray)
    assert isinstance(h_out, (float, np.floating))
    assert bedrock_mask.shape == final_Vs.shape
    assert np.all(np.isfinite(final_Vs))
