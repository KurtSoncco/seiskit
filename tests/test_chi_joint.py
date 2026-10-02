"""Tests for statistical_analysis/full_paper/analysis/code/chi_joint/covariance.py."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import multivariate_normal

CODE = Path(__file__).resolve().parents[1] / "statistical_analysis/full_paper/analysis/code"
sys.path.insert(0, str(CODE / "chi_joint"))
sys.path.insert(0, str(CODE / "chi_spatial"))
sys.path.insert(0, str(CODE))

cov = pytest.importorskip("covariance")
spatial_acf = pytest.importorskip("spatial_acf")
spatial_coherence = pytest.importorskip("spatial_coherence")

P = 40


def test_diag_is_one_and_pd():
    rng = np.random.default_rng(0)
    for model in cov.MODEL_SPECS:
        for _ in range(20):
            lo = np.array([b[0] for b in cov.bounds(model)])
            hi = np.array([b[1] for b in cov.bounds(model)])
            theta = rng.uniform(lo, hi) if lo.size else np.zeros(0)
            R = cov.build_R(model, theta, P)
            np.testing.assert_allclose(np.diag(R), 1.0, atol=1e-10)
            assert np.all(np.linalg.eigvalsh(R + cov.JITTER * np.eye(P)) > 0)


def test_loglik_matches_scipy():
    rng = np.random.default_rng(1)
    theta = np.array([-0.5, 1.0, np.log(20.0), np.log(0.8), 2.0])
    R = cov.build_R("coswm", theta, P)
    Z = cov.simulate(R, 30, rng)
    ref = multivariate_normal(mean=np.zeros(P), cov=R + cov.JITTER * np.eye(P)).logpdf(Z)
    np.testing.assert_allclose(cov.loglik_profiles(R, Z), ref, rtol=1e-8)
    S, n = cov.suffstat(Z)
    np.testing.assert_allclose(cov.loglik_suffstat(R, S, n), ref.sum(), rtol=1e-8)


def test_coswm_nests_wm():
    theta_wm = np.array([0.2, 1.0, np.log(30.0), np.log(0.7)])
    theta_cos = np.append(theta_wm, 0.0)
    np.testing.assert_allclose(
        cov.build_R("wm", theta_wm, P), cov.build_R("coswm", theta_cos, P), atol=1e-12
    )


def test_fit_recovers_shared_matern():
    rng = np.random.default_rng(2)
    true = np.array([np.log(0.3 / 0.1), np.log(0.6 / 0.1), np.log(25.0)])  # w = .3/.6/.1
    R = cov.build_R("matern32", true, P)
    S, n = cov.suffstat(cov.simulate(R, 3000, rng))
    theta, _ = cov.fit_model("matern32", S, n)
    w_true = cov.weights("matern32", true)
    w_hat = cov.weights("matern32", theta)
    np.testing.assert_allclose(w_hat, w_true, atol=0.04)
    assert abs(np.exp(theta[2]) - 25.0) / 25.0 < 0.15


def test_ridge_pulls_toward_prior():
    rng = np.random.default_rng(3)
    R = cov.build_R("matern32", np.array([0.0, 1.0, np.log(20.0)]), P)
    S, n = cov.suffstat(cov.simulate(R, 20, rng))
    prior = np.array([2.0, 0.0, np.log(60.0)])
    free, _ = cov.fit_model("matern32", S, n)
    tight, _ = cov.fit_model("matern32", S, n, prior_mean=prior, lam=1e5)
    assert np.linalg.norm(tight - prior) < 0.05 < np.linalg.norm(free - prior)


def test_diagnostics_match_chi_spatial_estimators():
    rng = np.random.default_rng(4)
    R = cov.build_R("coswm", np.array([0.0, 1.0, np.log(30.0), np.log(0.5), 3.0]), P)
    Z = cov.simulate(R, 25, rng)  # (seeds, nodes)
    # spatial_acf.empirical_acf works on its fixed 100-lag grid → compare first lags only
    h, rho, _, _ = spatial_acf.empirical_acf(Z.T)
    lags = np.arange(1, 11)
    np.testing.assert_allclose(cov.demeaned_acf(Z, lags), rho[:10], rtol=1e-10)

    _, rho_ik, _, _ = spatial_coherence.between_seed_C_rho(Z.T)
    ref = [np.mean(np.diagonal(rho_ik, offset=k)) for k in lags]
    np.testing.assert_allclose(cov.across_seed_corr(Z, lags), ref, rtol=1e-10)
