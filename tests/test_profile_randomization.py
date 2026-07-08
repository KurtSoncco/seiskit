"""Tests for profile randomization utilities."""

import numpy as np

from seiskit.profile_randomization import (
    ProfileRandomizationConfig,
    acf_rmse,
    generate_tts_randomized_profile,
    generate_vs_randomized_profile,
    profile_cov,
    vertical_acf_ln_vs,
)


def test_vs_profile_cov_near_target():
    cfg = ProfileRandomizationConfig(vs_mean=230.0, thickness=15.0, dz=0.5, cov=0.2)
    rng = np.random.default_rng(42)
    profiles = [generate_vs_randomized_profile(cfg, rng, rho=0.7) for _ in range(200)]
    mean_cov = np.mean([profile_cov(p) for p in profiles])
    assert 0.12 < mean_cov < 0.28


def test_tts_profile_positive_vs():
    cfg = ProfileRandomizationConfig(vs_mean=230.0, thickness=15.0, dz=0.5)
    rng = np.random.default_rng(7)
    vs = generate_tts_randomized_profile(cfg, rng, rho=0.8)
    assert np.all(vs > 0)
    assert len(vs) == int(round(15.0 / 0.5))


def test_vertical_acf_at_zero_is_one():
    vs = 230.0 * np.exp(0.1 * np.random.default_rng(0).standard_normal(30))
    lags, acf = vertical_acf_ln_vs(vs, dz=0.5)
    assert lags[0] == 0.0
    assert abs(acf[0] - 1.0) < 1e-6


def test_acf_rmse_identical_is_zero():
    lags = np.array([0.0, 0.5, 1.0])
    acf = np.array([1.0, 0.8, 0.6])
    assert acf_rmse(lags, acf, lags, acf) == 0.0
