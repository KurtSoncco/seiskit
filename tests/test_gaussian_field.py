"""Tests for seiskit.gaussian_field module."""

import numpy as np

from seiskit.gaussian_field import (
    create_dipping_vs_realization,
    create_three_layer_vs_realization,
    create_vs_realization,
    generate_gaussian_field_fft,
)


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


def test_create_three_layer_vs_realization_returns_expected_shapes():
    """create_three_layer_vs_realization returns correctly shaped outputs."""
    Lx = 40.0
    Lx_variability = 40.0
    Lz = 30.0
    dx = 5.0
    dz = 5.0
    result = create_three_layer_vs_realization(
        Vs1=150.0,
        Vs_mid=500.0,
        Vs_bedrock=1500.0,
        H1=10.0,
        H2=10.0,
        Lx=Lx,
        Lx_variability=Lx_variability,
        Lz=Lz,
        dx=dx,
        dz=dz,
        rH1=10.0,
        aHV1=1.0,
        CV1=0.2,
        rH2=10.0,
        aHV2=1.0,
        CV2=0.2,
        seed1=42,
        seed2=43,
    )
    final_Vs, x_total, z, (h1, h2), bedrock_mask = result
    nz = int(Lz / dz)
    assert final_Vs.shape[0] == nz
    assert final_Vs.shape[1] >= int(Lx / dx)
    assert isinstance(z, np.ndarray)
    assert isinstance(h1, (float, np.floating))
    assert isinstance(h2, (float, np.floating))
    assert bedrock_mask.shape == final_Vs.shape
    assert bedrock_mask.dtype == np.bool_
    assert np.all(np.isfinite(final_Vs))


def test_create_dipping_vs_realization_returns_expected_shapes():
    """create_dipping_vs_realization returns correctly shaped outputs for both dip directions."""
    Vs_profile = np.array([200.0, 800.0])
    Lx = 40.0
    Lx_variability = 40.0
    Lz = 30.0
    dx = 5.0
    dz = 5.0

    for dip_angle_deg in (5.0, -5.0):
        result = create_dipping_vs_realization(
            Vs_profile=Vs_profile,
            Lx=Lx,
            Lx_variability=Lx_variability,
            Lz=Lz,
            dx=dx,
            dz=dz,
            rH=10.0,
            aHV=1.0,
            CV=0.2,
            dip_angle_deg=dip_angle_deg,
            seed=42,
        )
        final_Vs, x_total, z, h_out, bedrock_mask = result
        nz = int(Lz / dz)
        assert final_Vs.shape[0] == nz
        assert final_Vs.shape[1] >= int(Lx / dx)
        assert isinstance(h_out, (float, np.floating))
        assert bedrock_mask.shape == final_Vs.shape
        assert np.all(np.isfinite(final_Vs))
        # Dip should make the interface deeper on one side than the other.
        assert bedrock_mask[:, 0].sum() != bedrock_mask[:, -1].sum()
