"""Unit tests for closed-form multilayer 1D transfer functions."""

import numpy as np
import pytest

from seiskit.theory.layered_1d_tf import (
    Layer,
    RockHalfspace,
    amplification_single_layer_elastic,
    layered_transfer_function,
    resonance_frequencies_quarter_wave,
)


def test_static_limit():
    layers = [Layer(30.0, 200.0, 2000.0, 0.0)]
    rock = RockHalfspace(1000.0, 2000.0, 0.0)
    _, aw, ao = layered_transfer_function(0.0, layers, rock)
    assert aw[0] == pytest.approx(1.0)
    assert ao[0] == pytest.approx(1.0)


def test_single_layer_matches_kramer_closed_form():
    H, Vs, rho = 30.0, 200.0, 2000.0
    Vs_r, rho_r = 1000.0, 2200.0
    freq = np.linspace(0.05, 25.0, 400)
    aw_cf, ao_cf = amplification_single_layer_elastic(freq, H, Vs, rho, Vs_r, rho_r)
    _, aw, ao = layered_transfer_function(
        freq, [Layer(H, Vs, rho, 0.0)], RockHalfspace(Vs_r, rho_r, 0.0)
    )
    assert np.max(np.abs(ao - ao_cf) / np.maximum(ao_cf, 1e-12)) < 1e-10
    assert np.max(np.abs(aw - aw_cf) / np.maximum(aw_cf, 1e-12)) < 1e-10


def test_homogeneous_outcrop_is_unity():
    """Finite layer with Vs = Vs_rock → AF_outcrop ≡ 1 (no impedance contrast)."""
    freq = np.linspace(0.1, 15.0, 200)
    _, _, ao = layered_transfer_function(
        freq,
        [Layer(50.0, 800.0, 2000.0, 0.0)],
        RockHalfspace(800.0, 2000.0, 0.0),
    )
    assert np.allclose(ao, 1.0, atol=1e-10)


def test_rigid_rock_peaks_at_quarter_wave():
    H, Vs, rho = 40.0, 200.0, 2000.0
    freq = np.linspace(0.05, 10.0, 2000)
    _, _, ao = layered_transfer_function(
        freq,
        [Layer(H, Vs, rho, 0.0)],
        RockHalfspace(1.0e7, rho, 0.0),
    )
    f_modes = resonance_frequencies_quarter_wave(Vs, H, n_modes=3)
    for fn in f_modes:
        # Peak should lie near theoretical fn
        band = (freq > fn * 0.9) & (freq < fn * 1.1)
        assert band.any()
        f_peak = freq[band][np.argmax(ao[band])]
        assert abs(f_peak - fn) / fn < 0.02


def test_two_layer_viscoelastic_finite_peaks():
    layers = [
        Layer(15.0, 180.0, 1900.0, 0.02),
        Layer(25.0, 300.0, 2000.0, 0.02),
    ]
    rock = RockHalfspace(1000.0, 2200.0, 0.01)
    freq = np.linspace(0.1, 20.0, 500)
    _, aw, ao = layered_transfer_function(freq, layers, rock)
    assert np.all(np.isfinite(aw))
    assert np.all(np.isfinite(ao))
    assert np.max(ao) > 1.0
    assert np.max(ao) < 200.0  # radiation + material damping; still finite


def test_three_and_four_layer_smoke():
    rock = RockHalfspace(1200.0, 2200.0, 0.0)
    cases = [
        [Layer(15.0, 180.0, 1900.0), Layer(25.0, 300.0, 2000.0)],
        [
            Layer(10.0, 150.0, 1800.0),
            Layer(15.0, 220.0, 1900.0),
            Layer(20.0, 350.0, 2000.0),
        ],
    ]
    freq = np.logspace(-1, 1.3, 100)
    for layers in cases:
        _, aw, ao = layered_transfer_function(freq, layers, rock)
        assert aw.shape == freq.shape
        assert ao[0] == pytest.approx(1.0, rel=0.05)


def test_identical_split_layers_match_single_layer():
    """Two stacked half-layers must match one full layer (transfer-matrix check)."""
    freq = np.linspace(0.1, 20.0, 300)
    rock = RockHalfspace(1000.0, 2200.0, 0.0)
    _, aw1, ao1 = layered_transfer_function(freq, [Layer(30.0, 200.0, 2000.0)], rock)
    _, aw2, ao2 = layered_transfer_function(
        freq,
        [Layer(15.0, 200.0, 2000.0), Layer(15.0, 200.0, 2000.0)],
        rock,
    )
    assert np.max(np.abs(ao1 - ao2) / np.maximum(ao1, 1e-12)) < 1e-8
    assert np.max(np.abs(aw1 - aw2) / np.maximum(aw1, 1e-12)) < 1e-8


def test_empty_layers_raises():
    with pytest.raises(ValueError):
        layered_transfer_function(1.0, [], RockHalfspace(1000.0, 2000.0))
