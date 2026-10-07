"""Tests for profile randomization utilities."""

import numpy as np
import pytest

from seiskit.profile_randomization import (
    DmultMethod,
    PasseriMethod,
    ProfileRandomizationConfig,
    RandomizedProfile,
    SpatialVariabilityMethod,
    ToroMethod,
    acf_rmse,
    build_base_case_profile,
    dmult_from_vs_contrast,
    generate_nhpp_layer_thicknesses,
    generate_passeri_profile,
    generate_toro_profile,
    generate_tts_randomized_profile,
    generate_vs_randomized_profile,
    get_method,
    hallal_profile_config,
    profile_to_opensees_column,
    toro_adjacent_correlation,
    vertical_acf_ln_vs,
    vs_contrast,
)


def _cfg(**kw) -> ProfileRandomizationConfig:
    defaults = dict(
        vs_mean=230.0,
        thickness=15.0,
        dz=0.5,
        vs_bedrock=1500.0,
        bedrock_thickness=10.0,
        cov=0.2,
    )
    defaults.update(kw)
    return ProfileRandomizationConfig(**defaults)


def _ensemble_cov(profiles: list[np.ndarray]) -> float:
    arr = np.asarray(profiles, dtype=float)
    m = float(np.mean(arr))
    if m <= 0:
        return 0.0
    return float(np.std(arr) / m)


def test_base_case_includes_bedrock():
    cfg = _cfg()
    base = build_base_case_profile(cfg)
    assert len(base) == 50
    assert np.all(base[:30] == 230.0)
    assert np.all(base[30:] == 1500.0)


def test_nhpp_generates_multiple_layers():
    rng = np.random.default_rng(0)
    thick, _, _ = generate_nhpp_layer_thicknesses(15.0, rng, min_thickness=1.0)
    assert len(thick) >= 2
    assert abs(float(thick.sum()) - 15.0) < 0.05


def test_full_toro_returns_profile_metadata():
    cfg = _cfg(use_full_model=True, randomize_bedrock_depth=False)
    prof = generate_toro_profile(cfg, np.random.default_rng(1))
    assert isinstance(prof, RandomizedProfile)
    assert len(prof.vs_depth) == 50
    assert prof.n_soil_samples > 0
    assert np.all(prof.vs_depth[prof.n_soil_samples :] >= 1400.0)


def test_full_passeri_varies_bedrock_vs():
    cfg = _cfg(use_full_model=True, randomize_bedrock_depth=False, vary_bedrock_vs=True)
    rng = np.random.default_rng(2)
    beds = [generate_passeri_profile(cfg, rng).vs_depth[-1] for _ in range(100)]
    assert np.std(beds) > 1.0


def test_full_passeri_fixed_bedrock_when_disabled():
    cfg = _cfg(use_full_model=True, randomize_bedrock_depth=False, vary_bedrock_vs=False)
    rng = np.random.default_rng(2)
    beds = [generate_passeri_profile(cfg, rng).vs_depth[-1] for _ in range(50)]
    assert np.allclose(beds, 1500.0)


def test_full_toro_varies_bedrock_vs():
    cfg = _cfg(use_full_model=True, randomize_bedrock_depth=False, vary_bedrock_vs=True)
    rng = np.random.default_rng(2)
    beds = [generate_toro_profile(cfg, rng).vs_depth[-1] for _ in range(100)]
    assert np.std(beds) > 1.0


def test_full_toro_soil_not_flat_when_nhpp_on():
    cfg = _cfg(use_full_model=True, randomize_bedrock_depth=False)
    prof = generate_toro_profile(cfg, np.random.default_rng(3))
    soil = prof.vs_depth[: prof.n_soil_samples]
    assert np.std(soil) > 1.0


def test_simplified_frozen_h_soil_varies_with_depth():
    cfg = _cfg(use_full_model=False, randomize_layer_thickness=False, randomize_bedrock_depth=False)
    vs = generate_vs_randomized_profile(cfg, np.random.default_rng(42))
    assert vs[:30].std() > 1.0
    assert np.all(vs[30:] == 1500.0)
    assert np.all(np.isfinite(vs))
    assert np.all(vs > 0)


def test_vs_ensemble_cov_simplified():
    cfg = _cfg(use_full_model=False, randomize_bedrock_depth=False, randomize_layer_thickness=False)
    rng = np.random.default_rng(42)
    profiles = [generate_vs_randomized_profile(cfg, rng)[:30] for _ in range(400)]
    assert 0.08 < _ensemble_cov(profiles) < 0.35
    assert not np.allclose(profiles[0], profiles[1])


def test_profile_to_opensees_auto_interface():
    cfg = _cfg()
    prof = generate_toro_profile(cfg, np.random.default_rng(5))
    col, mask = profile_to_opensees_column(prof.vs_depth, prof.n_soil_samples)
    assert col.shape[0] == len(prof.vs_depth)
    assert mask.sum() == len(prof.vs_depth) - prof.n_soil_samples


def test_tts_profile_positive_vs():
    cfg = _cfg()
    vs = generate_tts_randomized_profile(cfg, np.random.default_rng(7))
    assert np.all(vs > 0)
    assert len(vs) == 50


def test_toro_correlation_in_valid_range():
    depth = np.array([0.25, 0.75, 1.25, 2.0])
    rho = toro_adjacent_correlation(depth, rho_0=0.99, delta=3.9, rho_200=0.98, b=0.344)
    assert len(rho) == len(depth) - 1
    assert np.all((rho >= 0) & (rho <= 1))


def test_toro_bedrock_correlation_overwrites_last_pair():
    depth = np.array([1.0, 3.0, 6.0, 10.0])
    rho_soil = toro_adjacent_correlation(
        depth, rho_0=0.99, delta=3.9, rho_200=0.98, b=0.344, bedrock_interface=False
    )
    rho_bed = toro_adjacent_correlation(
        depth,
        rho_0=0.99,
        delta=3.9,
        rho_200=0.98,
        b=0.344,
        bedrock_interface=True,
        bedrock_interface_rho=0.42,
    )
    assert len(rho_bed) == len(depth) - 1
    assert np.allclose(rho_bed[:-1], rho_soil[:-1])
    assert rho_bed[-1] == 0.42


def test_vertical_acf_at_zero_is_one():
    vs = 230.0 * np.exp(0.1 * np.random.default_rng(0).standard_normal(30))
    lags, acf = vertical_acf_ln_vs(vs, dz=0.5)
    assert lags[0] == 0.0
    assert abs(acf[0] - 1.0) < 1e-6


def test_acf_rmse_identical_is_zero():
    lags = np.array([0.0, 0.5, 1.0])
    acf = np.array([1.0, 0.8, 0.6])
    assert acf_rmse(lags, acf, lags, acf) == 0.0


def test_passeri_soil_vs_finite():
    cfg = _cfg(use_full_model=True, randomize_layer_thickness=True)
    rng = np.random.default_rng(99)
    for _ in range(200):
        prof = generate_passeri_profile(cfg, rng)
        soil = prof.vs_depth[: prof.n_soil_samples]
        assert np.all(np.isfinite(soil))
        assert np.all(soil > 0)
        assert np.all(soil < 1e6)


def test_passeri_joint_bedrock_correlation():
    from seiskit.profile_randomization.passeri import _passeri_joint_bedrock_draw

    cfg = _cfg(randomize_bedrock_depth=True, vary_bedrock_vs=True)
    rng = np.random.default_rng(0)
    depths = []
    vss = []
    for _ in range(5000):
        depth, vs = _passeri_joint_bedrock_draw(cfg, rng)
        depths.append(depth)
        vss.append(vs)
    corr = float(np.corrcoef(np.log(depths), np.log(vss))[0, 1])
    assert 0.40 < corr < 0.60


def test_dmult_formula_clip_and_linear():
    # High contrast → clips to 2 (Hallal-like column ~10.2)
    assert dmult_from_vs_contrast(225.6, 2298.12) == pytest.approx(2.0)
    # Low contrast → clips to 10
    assert dmult_from_vs_contrast(400.0, 800.0) == pytest.approx(10.0)
    # Mid-range linear: contrast=5 → -1.3*5+13.9 = 7.4
    assert dmult_from_vs_contrast(200.0, 1000.0) == pytest.approx(7.4)
    assert vs_contrast(200.0, 1000.0) == pytest.approx(5.0)


def test_get_method_aliases():
    assert isinstance(get_method("toro"), ToroMethod)
    assert isinstance(get_method("hallal_vs"), ToroMethod)
    assert isinstance(get_method("passeri"), PasseriMethod)
    assert isinstance(get_method("hallal_tts"), PasseriMethod)
    assert isinstance(get_method("dmult"), DmultMethod)
    assert isinstance(get_method("hallal_dmin"), DmultMethod)
    with pytest.raises(ValueError):
        get_method("unknown")


def test_method_generate_profiles_vs_only():
    cfg = hallal_profile_config(
        vs1=230.0, H=15.0, cov=0.20, vs2=1500.0, dz=0.5, bedrock_thickness=10.0
    )
    assert cfg.sigma_ln_vs == pytest.approx(0.20)
    assert cfg.sigma_ln_tts == pytest.approx(0.20)
    assert cfg.randomize_layer_thickness is False

    toro = get_method("toro")
    passeri = get_method("passeri")
    dmult = get_method("dmult")
    assert isinstance(toro, SpatialVariabilityMethod)

    p1 = toro.generate_profile(cfg, np.random.default_rng(1))
    p2 = passeri.generate_profile(cfg, np.random.default_rng(1))
    p3 = dmult.generate_profile(cfg, np.random.default_rng(1))
    p3b = dmult.generate_profile(cfg, np.random.default_rng(99))
    assert isinstance(p1, RandomizedProfile)
    assert isinstance(p2, RandomizedProfile)
    assert p1.n_soil_samples == 30
    assert p2.n_soil_samples == 30
    assert np.allclose(p3.vs_depth, p3b.vs_depth)  # deterministic
    assert dmult.damping_multiplier(230.0, 1500.0) == dmult_from_vs_contrast(230.0, 1500.0)
    assert dmult.uses_elemental_damping() is False
    assert toro.damping_multiplier(230.0, 1500.0) == 1.0


def test_toro_sigma_ln_spid_taper():
    from seiskit.profile_randomization import toro_sigma_ln_vs

    cfg = _cfg()
    assert float(toro_sigma_ln_vs(0.0, cfg)) == pytest.approx(0.25)
    assert float(toro_sigma_ln_vs(15.0, cfg)) == pytest.approx(0.15)
    assert float(toro_sigma_ln_vs(40.0, cfg)) == pytest.approx(0.15)
    mid = float(toro_sigma_ln_vs(7.5, cfg))
    assert 0.15 < mid < 0.25


def test_toro_h0_not_rho0_in_depth_term():
    depth = np.array([0.25, 0.75, 1.25, 2.0])
    rho_h0 = toro_adjacent_correlation(depth, rho_0=0.99, delta=3.9, rho_200=0.98, b=0.344, h0=0.0)
    rho_old = toro_adjacent_correlation(
        depth, rho_0=0.99, delta=3.9, rho_200=0.98, b=0.344, h0=0.99
    )
    assert not np.allclose(rho_h0, rho_old)


def test_frozen_h_full_matches_simplified():
    kw = dict(
        use_full_model=True,
        randomize_layer_thickness=False,
        randomize_bedrock_depth=False,
        vary_bedrock_vs=False,
    )
    cfg_full = _cfg(**kw)
    cfg_simp = _cfg(
        use_full_model=False, randomize_layer_thickness=False, randomize_bedrock_depth=False
    )
    vs_full = generate_toro_profile(cfg_full, np.random.default_rng(42)).vs_depth
    vs_simp = generate_vs_randomized_profile(cfg_simp, np.random.default_rng(42))
    assert np.allclose(vs_full, vs_simp)


def test_frozen_h_adjacent_ln_corr_high():
    cfg = _cfg(
        use_full_model=False,
        randomize_layer_thickness=False,
        randomize_bedrock_depth=False,
        thickness=40.0,
    )
    rng = np.random.default_rng(0)
    n_soil = int(round(cfg.thickness / cfg.dz))
    stack = np.vstack([generate_vs_randomized_profile(cfg, rng)[:n_soil] for _ in range(200)])
    ln = np.log(np.clip(stack, 1e-6, None))
    corr = float(np.corrcoef(ln[:, 0], ln[:, 1])[0, 1])
    assert corr > 0.7


def test_bedrock_depth_model_validation():
    with pytest.raises(ValueError, match="bedrock_depth_model"):
        _cfg(bedrock_depth_model="normal")
    with pytest.raises(ValueError, match="dip_half_span_m"):
        _cfg(bedrock_depth_model="dip", dip_half_span_m=0.0)
    with pytest.raises(ValueError, match="dip_angle_max_deg"):
        _cfg(bedrock_depth_model="dip", dip_angle_min_deg=3.0, dip_angle_max_deg=-3.0)
    cfg = _cfg(bedrock_depth_model="DIP")
    assert cfg.bedrock_depth_model == "dip"


def test_dip_interface_depth_bounds_mean_and_shape():
    from seiskit.profile_randomization.nhpp import _sample_interface_depth

    H = 40.0
    half_span = 250.0
    angle_max = 3.0
    max_shift = half_span * np.tan(np.radians(angle_max))
    cfg = _cfg(
        thickness=H,
        bedrock_thickness=30.0,
        randomize_bedrock_depth=True,
        bedrock_depth_model="dip",
        dip_angle_min_deg=-angle_max,
        dip_angle_max_deg=angle_max,
        dip_half_span_m=half_span,
    )
    rng = np.random.default_rng(0)
    depths = np.array([_sample_interface_depth(cfg, rng) for _ in range(8000)])
    assert depths.min() >= H - max_shift - 1e-9
    assert depths.max() <= H + max_shift + 1e-9
    assert abs(float(np.mean(depths)) - H) < 0.35
    # x tan θ is more concentrated at H than Uniform[H±max_shift]
    n_center = int(np.sum(np.abs(depths - H) < 0.25 * max_shift))
    n_edge = int(np.sum(np.abs(depths - H) > 0.75 * max_shift))
    assert n_center > n_edge


def test_toro_passeri_use_dip_interface_depth():
    cfg = _cfg(
        thickness=40.0,
        bedrock_thickness=30.0,
        randomize_layer_thickness=False,
        randomize_bedrock_depth=True,
        bedrock_depth_model="dip",
        vary_bedrock_vs=False,
    )
    max_shift = cfg.dip_half_span_m * np.tan(np.radians(cfg.dip_angle_max_deg))
    rng_t = np.random.default_rng(11)
    rng_p = np.random.default_rng(11)
    toro_depths = [generate_toro_profile(cfg, rng_t).interface_depth for _ in range(200)]
    passeri_depths = [generate_passeri_profile(cfg, rng_p).interface_depth for _ in range(200)]
    for depths in (toro_depths, passeri_depths):
        arr = np.asarray(depths)
        assert arr.min() >= 40.0 - max_shift - 1e-9
        assert arr.max() <= 40.0 + max_shift + 1e-9
        assert abs(float(np.mean(arr)) - 40.0) < 1.0
        assert float(np.std(arr)) > 1.0  # not stuck at nominal H


def test_passeri_dip_depth_independent_of_bedrock_vs():
    from seiskit.profile_randomization.passeri import _passeri_joint_bedrock_draw

    cfg = _cfg(
        thickness=40.0,
        bedrock_thickness=30.0,
        randomize_bedrock_depth=True,
        bedrock_depth_model="dip",
        vary_bedrock_vs=True,
    )
    rng = np.random.default_rng(0)
    depths = []
    vss = []
    for _ in range(5000):
        depth, vs = _passeri_joint_bedrock_draw(cfg, rng)
        depths.append(depth)
        vss.append(vs)
    corr = float(np.corrcoef(depths, np.log(vss))[0, 1])
    assert abs(corr) < 0.08
