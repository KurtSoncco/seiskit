"""Tests for intensity measures and GOF."""

import numpy as np

from seiskit.gof import anderson_frequency_domain, anderson_time_domain, log_residual_bias
from seiskit.intensity_measures import compute_sa, pga, sigma_ln


def test_pga_sine():
    t = np.linspace(0, 1, 100)
    a = 2.0 * np.sin(2 * np.pi * t)
    assert abs(pga(a) - 2.0) < 0.01


def test_sigma_ln_positive():
    x = np.array([1.0, 1.5, 2.0, 2.5])
    s = sigma_ln(x)
    assert s > 0


def test_compute_sa_shape():
    dt = 0.01
    n = 500
    a = np.random.default_rng(0).standard_normal(n)
    periods = np.array([0.2, 0.5, 1.0])
    sa = compute_sa(a, dt, periods)
    assert sa.shape == periods.shape
    assert np.all(sa >= 0)


def test_anderson_identical_signals():
    dt = 0.01
    a = np.sin(2 * np.pi * np.arange(200) * dt)
    gof = anderson_time_domain(a, a, dt)
    assert gof["GOF_A"] < 1e-6


def test_log_residual_bias():
    ref = np.array([1.0, 2.0, 4.0])
    cand = ref * np.e
    assert abs(log_residual_bias(ref, cand) - 1.0) < 1e-6


def test_anderson_freq_zero_for_match():
    f = np.logspace(-1, 1, 20)
    af = np.linspace(1, 10, 20)
    assert anderson_frequency_domain(f, af, af) == 0.0
