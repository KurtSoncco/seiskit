"""
Response_Variability comparison campaign runner.

Sobol base cases (Vs1, H, CoV, Vs2) with fixed rH=10, aHV=50.
Methods: grf_2d (GIFNO), opensees_2d (OpenSees baseline), pretell (1D, full strip),
Hallal 1D arms.

Usage:
  python run_experiment.py --index 0
  RV_SMOKE=1 python run_experiment.py --index 0
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from manifest import (
    PRETELL_SAMPLE_WIDTH_M,
    RF_METHODS,
    CaseParams,
    active_duration,
    active_dx,
    active_dz,
    active_lx_total,
    active_pretell_n_samples,
    active_rf_bc_width,
    active_rf_dx,
    active_rf_dz,
    active_rf_lx_total,
    active_rf_lx_var,
    case_tag,
    damping_method_for,
    dmin_multiplier_for,
    index_to_params,
    motion_frequency,
    pretell_column_indices,
    total_combinations,
)

from seiskit.analysis import run_opensees_analysis
from seiskit.builder import build_model_data
from seiskit.config import AnalysisConfig
from seiskit.gaussian_field import (
    _extend_profile,
    _generate_vs_variability_field,
)
from seiskit.intensity_measures import compute_sa, default_periods, pga, sigma_ln
from seiskit.plot_results import get_damping_zeta_grid
from seiskit.profile_randomization import (
    ProfileRandomizationConfig,
    RandomizedProfile,
    generate_tts_randomized_profile_full,
    generate_vs_randomized_profile_full,
    profile_to_opensees_column,
)
from seiskit.solver_utils import get_solver_info
from seiskit.ttf.TTF import TTF, TTF_batch_fast

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _results_root() -> Path:
    base = os.getenv("RV_OUTDIR")
    if base:
        return Path(base)
    return Path(__file__).resolve().parent / "results"


def _h5_path(index: int) -> Path:
    base = os.getenv("RV_H5_DIR")
    if base:
        return Path(base) / f"run_{index}.h5"
    return _results_root() / "h5" / f"run_{index}.h5"


def _output_dir(p: CaseParams) -> Path:
    return _results_root() / f"idx_{p.index:05d}"


# ---------------------------------------------------------------------------
# Recorders / transfer functions
# ---------------------------------------------------------------------------


def _recorder_y_sort_key(path: Path) -> float:
    m = re.search(r"(?:center_node_|row_)y([\d.]+)_", path.name)
    return float(m.group(1)) if m else 0.0


def _load_surface_base(recorder_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = sorted(recorder_dir.glob("center_node_y*_dof1_accel.txt"), key=_recorder_y_sort_key)
    if len(center) >= 2:
        base_arr = np.loadtxt(center[0])
        surf_arr = np.loadtxt(center[-1])
        if base_arr.size == 0 or surf_arr.size == 0:
            pass
        else:
            if base_arr.ndim == 1:
                base_arr = base_arr.reshape(-1, 1)
            if surf_arr.ndim == 1:
                surf_arr = surf_arr.reshape(-1, 1)
            if base_arr.shape[1] >= 2 and surf_arr.shape[1] >= 2:
                t = base_arr[:, 0]
                return t, surf_arr[:, 1], base_arr[:, 1]

    rows = sorted(recorder_dir.glob("row_y*_dof1_accel.txt"), key=_recorder_y_sort_key)
    if len(rows) >= 2:
        base_arr = np.loadtxt(rows[0])
        surf_arr = np.loadtxt(rows[-1])
        if base_arr.ndim == 1:
            base_arr = base_arr.reshape(-1, 1)
        if surf_arr.ndim == 1:
            surf_arr = surf_arr.reshape(-1, 1)
        t = base_arr[:, 0]
        mid = base_arr.shape[1] // 2
        return t, surf_arr[:, mid], base_arr[:, mid]

    time_arr, accel = _load_recorder_txt(recorder_dir)
    if len(time_arr) == 0:
        return np.array([]), np.array([]), np.array([])
    if accel.ndim == 2 and accel.shape[1] >= 2:
        return time_arr, accel[:, -1], accel[:, 0]
    return time_arr, accel.ravel(), accel.ravel()


def _load_lateral_surface_base(
    recorder_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    rows = sorted(recorder_dir.glob("row_y*_dof1_accel.txt"), key=_recorder_y_sort_key)
    if len(rows) < 2:
        return None
    base_arr = np.loadtxt(rows[0])
    surf_arr = np.loadtxt(rows[-1])
    if base_arr.ndim == 1:
        base_arr = base_arr.reshape(-1, 1)
    if surf_arr.ndim == 1:
        surf_arr = surf_arr.reshape(-1, 1)
    if base_arr.shape[1] < 3 or surf_arr.shape[1] < 3:
        return None
    t = base_arr[:, 0]
    n = min(base_arr.shape[1], surf_arr.shape[1]) - 1
    return t, surf_arr[:, 1 : n + 1], base_arr[:, 1 : n + 1]


def _spatial_af_percentiles(
    surf_nodes: np.ndarray,
    base_nodes: np.ndarray,
    dt: float,
    *,
    vs_min: float,
    dz: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    freq, af_ch = TTF_batch_fast(
        base_nodes.T,
        surf_nodes.T,
        dt=dt,
        Vsmin=vs_min,
        dz=dz,
    )
    return (
        freq,
        af_ch.astype(np.float32),
        {
            "median": np.median(af_ch, axis=0).astype(np.float32),
            "p16": np.percentile(af_ch, 16, axis=0).astype(np.float32),
            "p84": np.percentile(af_ch, 84, axis=0).astype(np.float32),
            "sigma_ln": np.array(
                [sigma_ln(af_ch[:, j]) for j in range(af_ch.shape[1])], dtype=np.float32
            ),
        },
    )


def _load_recorder_txt(
    recorder_dir: Path, quantity: str = "accel"
) -> tuple[np.ndarray, np.ndarray]:
    center_glob = list(recorder_dir.glob(f"center_node_y*_dof1_{quantity}.txt"))
    row_glob = list(recorder_dir.glob(f"row_y*_dof1_{quantity}.txt"))
    center_glob.sort(key=_recorder_y_sort_key)
    row_glob.sort(key=_recorder_y_sort_key)
    chunks: list[np.ndarray] = []
    time_arr = None
    for f in center_glob + row_glob:
        arr = np.loadtxt(f)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if time_arr is None:
            time_arr = arr[:, 0]
        if arr.shape[1] > 1:
            chunks.append(arr[:, 1:])
    if not chunks:
        return np.array([]), np.array([]).reshape(0, 0)
    return time_arr, np.hstack(chunks)


# ---------------------------------------------------------------------------
# Profile builders
# ---------------------------------------------------------------------------


def _profile_config(p: CaseParams) -> ProfileRandomizationConfig:
    # Hallal arms: fixed H and bedrock; only σ_ln(Vs) or σ_ln(tts) on the base profile.
    return ProfileRandomizationConfig(
        vs_mean=p.vs1,
        thickness=p.H,
        dz=active_dz(),
        vs_bedrock=p.vs2,
        bedrock_thickness=p.bedrock_thickness,
        cov=p.cov,
        use_full_model=False,
        randomize_layer_thickness=False,
        randomize_bedrock_depth=False,
        vary_bedrock_vs=False,
    )


def _is_rf_method(p: CaseParams) -> bool:
    return p.method in RF_METHODS


def _grid_spacing(p: CaseParams) -> tuple[float, float]:
    if _is_rf_method(p):
        return active_rf_dx(), active_rf_dz()
    return active_dx(), active_dz()


def _discretized_profile(p: CaseParams) -> tuple[np.ndarray, int, int]:
    _, dz = _grid_spacing(p)
    n_soil = max(1, int(round(p.H / dz)))
    n_bed = max(1, int(round(p.bedrock_thickness / dz)))
    vs_profile_1d = np.array([p.vs1] * n_soil + [p.vs2] * n_bed)
    return vs_profile_1d, n_soil, n_bed


def _build_2d_grf(
    p: CaseParams,
    rf_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Vs_extended nz×nx, bedrock_mask_extended, template Vs_profile_1D).

    Built on the neural-operator grid: 1500 m × nz (dx=dz=1 m).
    """
    dx, dz = active_rf_dx(), active_rf_dz()
    vs_profile_1d, _, _ = _discretized_profile(p)
    lz = len(vs_profile_1d) * dz

    vs_var, _, _, _, bedrock_mask_var = _generate_vs_variability_field(
        vs_profile_1d,
        active_rf_lx_var(),
        lz,
        dx,
        dz,
        p.rH,
        p.aHV,
        p.cov,
        seed=rf_seed,
        dz_1D=dz,
        interlayer_seed=14,
        interlayer_amplitude=0.0,
    )
    vs_ext, _ = _extend_profile(vs_var, Lx=active_rf_lx_total(), dx=dx)
    bedrock_ext, _ = _extend_profile(bedrock_mask_var.astype(float), Lx=active_rf_lx_total(), dx=dx)
    return vs_ext, bedrock_ext.astype(bool), vs_profile_1d


def _geomean_af(af_stack: np.ndarray) -> np.ndarray:
    clipped = np.clip(af_stack, 1e-12, None)
    return np.exp(np.nanmean(np.log(clipped), axis=0)).astype(np.float32)


def _interp_af(freq_src: np.ndarray, af: np.ndarray, freq_dst: np.ndarray) -> np.ndarray:
    from scipy.interpolate import interp1d

    fn = interp1d(freq_src, af, kind="linear", bounds_error=False, fill_value=np.nan)
    return fn(freq_dst).astype(np.float32)


def _run_pretell_profiles(
    p: CaseParams,
    *,
    vs_field_2d: np.ndarray,
    bedrock_mask_2d: np.ndarray,
    vs_profile_1d: np.ndarray,
    out_dir: Path,
    task_id: str,
    motion_freq: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run 1D OpenSees on Pretell strip columns; return geomean AF."""
    _dx, dz = active_rf_dx(), active_rf_dz()
    lz = vs_field_2d.shape[0] * dz
    cols = pretell_column_indices()
    n_samples = len(cols)
    af_rows: list[np.ndarray] = []
    freq_ref: np.ndarray | None = None

    for col in cols:
        vs_col = vs_field_2d[:, col : col + 1]
        mask_col = bedrock_mask_2d[:, col : col + 1]
        sub_id = f"{task_id}_x{col}"
        config = _analysis_config(p, lz, motion_freq, bc_2d=False, grid_dx=dz, grid_dz=dz)
        get_damping_zeta_grid(
            vs_col,
            config.damping_method,
            config.Lx,
            lz,
            config.hx,
            dz,
            bedrock_mask=mask_col,
            config=config,
        )
        rho = np.ones_like(vs_col, dtype=float) * 2000.0
        nu = np.ones_like(vs_col, dtype=float) * 0.3
        model_data = build_model_data(config, vs_col, rho, nu, bedrock_mask=mask_col)
        status = run_opensees_analysis(config, model_data, sub_id, str(out_dir))
        if status.startswith("Failed"):
            raise RuntimeError(f"Pretell column {col}: {status}")

        recorder_dir = out_dir / sub_id
        time_arr, surface, base = _load_surface_base(recorder_dir)
        if len(time_arr) == 0:
            raise RuntimeError(f"No recorder output for Pretell column {col} in {recorder_dir}")
        dt = config.dt
        if len(time_arr) > 1:
            dt = float(time_arr[1] - time_arr[0])
        vs_min = float(np.min(vs_col[~mask_col])) if np.any(~mask_col) else float(p.vs1)
        freq, af = TTF(surface, base, dt=dt, Vsmin=vs_min, dz=dz)
        if freq_ref is None:
            freq_ref = freq
        else:
            af = _interp_af(freq, af, freq_ref)
        af_rows.append(af)

    assert freq_ref is not None
    af_geo = _geomean_af(np.vstack(af_rows))
    return freq_ref.astype(np.float32), af_geo, n_samples


def theoretical_f0(vs1: float, H: float) -> float:
    """Quarter-wave fundamental frequency f₀ = Vs1 / (4 H)."""
    return float(vs1) / (4.0 * float(H))


def _build_1d_hallal(
    p: CaseParams,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = _profile_config(p)

    if p.method == "hallal_vs":
        prof = generate_vs_randomized_profile_full(cfg, rng)
    elif p.method == "hallal_tts":
        prof = generate_tts_randomized_profile_full(cfg, rng)
    elif p.method == "hallal_dmin":
        from seiskit.profile_randomization import build_base_case_profile

        vs_profile = build_base_case_profile(cfg)
        prof = RandomizedProfile(
            vs_depth=vs_profile,
            n_soil_samples=max(1, int(round(p.H / active_dz()))),
            interface_depth=p.H,
        )
    else:
        raise ValueError(f"Not a 1D Hallal method: {p.method}")

    vs_col, mask = profile_to_opensees_column(prof.vs_depth, prof.n_soil_samples)
    return vs_col, mask, prof.vs_depth


def compute_base_1d_tf(
    p: CaseParams,
    *,
    out_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Deterministic 1D OpenSees TF on the base Vs profile (no randomization).

    Uses the same damping as hallal_vs/tts (``global_avg``, Dmin×1).
    Returns ``(freq, af, f0)`` with ``f0 = Vs1/(4H)``.
    """
    from seiskit.profile_randomization import build_base_case_profile

    f0 = theoretical_f0(p.vs1, p.H)
    cfg = _profile_config(p)
    vs_profile = build_base_case_profile(cfg)
    n_soil = max(1, int(round(p.H / active_dz())))
    vs_col, mask = profile_to_opensees_column(vs_profile, n_soil)
    dz = active_dz()
    lz = float(vs_col.shape[0] * dz)
    motion_freq = motion_frequency(p.vs1, p.motion_id, H=p.H)

    # Force base-case damping regardless of p.method (e.g. if called with hallal_dmin).
    base_p = CaseParams(
        index=p.index,
        sobol_id=p.sobol_id,
        vs1=p.vs1,
        H=p.H,
        cov=p.cov,
        vs2=p.vs2,
        method="hallal_vs",
        motion_id=p.motion_id,
        seed=0,
        seed_kind="realization",
        rH=p.rH,
        aHV=p.aHV,
        bedrock_thickness=p.bedrock_thickness,
        dmin_multiplier=1.0,
    )
    config = _analysis_config(base_p, lz, motion_freq, bc_2d=False, grid_dx=dz, grid_dz=dz)
    get_damping_zeta_grid(
        vs_col,
        config.damping_method,
        config.Lx,
        lz,
        config.hx,
        dz,
        bedrock_mask=mask,
        config=config,
    )
    rho = np.ones_like(vs_col, dtype=float) * 2000.0
    nu = np.ones_like(vs_col, dtype=float) * 0.3
    model_data = build_model_data(config, vs_col, rho, nu, bedrock_mask=mask)

    work = Path(out_dir) if out_dir is not None else Path("results") / "base_1d_tf"
    work.mkdir(parents=True, exist_ok=True)
    task_id = f"base1d_s{p.sobol_id:02d}_{p.motion_id}"
    status = run_opensees_analysis(config, model_data, task_id, str(work))
    if status.startswith("Failed"):
        raise RuntimeError(f"Base 1D TF failed: {status}")

    recorder_dir = work / task_id
    time_arr, surface, base = _load_surface_base(recorder_dir)
    if len(time_arr) == 0:
        raise RuntimeError(f"No recorder output for base 1D TF in {recorder_dir}")
    dt = config.dt
    if len(time_arr) > 1:
        dt = float(time_arr[1] - time_arr[0])
    vs_min = float(np.min(vs_col[~mask])) if np.any(~mask) else float(p.vs1)
    freq, af = TTF(surface, base, dt=dt, Vsmin=vs_min, dz=dz)
    return freq.astype(np.float32), af.astype(np.float32), f0


# ---------------------------------------------------------------------------
# Analysis config / HDF5
# ---------------------------------------------------------------------------


def _analysis_config(
    p: CaseParams,
    lz: float,
    motion_freq: float,
    *,
    bc_2d: bool,
    grid_dx: float | None = None,
    grid_dz: float | None = None,
) -> AnalysisConfig:
    if grid_dx is None or grid_dz is None:
        grid_dx, grid_dz = _grid_spacing(p)
    f0 = p.vs1 / (4.0 * p.H)
    duration = active_duration(f0)
    damping_freq_first = min(f0, motion_freq)

    lx = (
        active_rf_lx_total()
        if bc_2d and _is_rf_method(p)
        else (active_lx_total() if bc_2d else grid_dz)
    )
    record_lateral = (10, 2.0) if bc_2d else None

    # Large 2D NO-grid meshes (~1500×nz) need hours per 100-step batch; override via env.
    max_batch = float(os.getenv("RV_MAX_TIME_PER_BATCH", "28800"))  # 8 h default

    # 1D: soil–bedrock interface + surface (1D_theory_validation / 1D_cases).
    # 2D: 2 m above model base + surface (neural-operator training convention).
    if bc_2d:
        center_y = [2.0, lz]
    else:
        center_y = [float(p.bedrock_thickness), lz]

    return AnalysisConfig(
        Ly=lz,
        Lx=lx,
        hx=grid_dx if bc_2d else grid_dz,
        dt=0.01,
        duration=duration,
        motion_freq=motion_freq,
        motion_t_shift=0.5,
        damping_freqs=(damping_freq_first, 10.0),
        damping_zeta=0.025,
        damping_method=damping_method_for(p),
        dmin_multiplier=dmin_multiplier_for(p),
        boundary_condition_type="2D" if bc_2d else "1D",
        record_center_nodes=True,
        center_node_y_positions=center_y,
        record_lateral_span_at_center_depths=record_lateral,
        record_all_surface_nodes=False,
        element_type="4node",
        solver_type="Mumps",
        max_time_per_batch=max_batch,
    )


def _write_h5(
    path: Path,
    p: CaseParams,
    *,
    vs_profile_1d: np.ndarray,
    vs_field: np.ndarray,
    zeta_grid: np.ndarray,
    time_arr: np.ndarray,
    accel: np.ndarray,
    dt: float,
    task_id: str,
    freq: np.ndarray,
    af: np.ndarray,
    af_spatial: np.ndarray | None = None,
    af_spatial_stats: dict[str, np.ndarray] | None = None,
    tf_only: bool = False,
    analysis_backend: str = "opensees",
    pretell_n_samples: int = 0,
) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    periods = default_periods(p.vs1, p.H)
    if not tf_only and len(time_arr) == 0:
        raise ValueError("Cannot write HDF5 without time series")
    if tf_only:
        surface = np.array([], dtype=np.float32)
        sa_surf = np.zeros(len(periods), dtype=np.float32)
        pga_val = float("nan")
    else:
        surface = accel[:, 0] if accel.ndim == 2 and accel.shape[1] > 0 else accel.ravel()
        sa_surf = compute_sa(surface, dt, periods)
        pga_val = pga(surface)
    damp_method = damping_method_for(p)

    with h5py.File(path, "w") as f:
        f.attrs["index"] = p.index
        f.attrs["task_id"] = task_id
        f.attrs["method"] = p.method
        f.attrs["motion_id"] = p.motion_id
        f.attrs["damping_method"] = damp_method
        f.attrs["analysis_backend"] = analysis_backend
        if pretell_n_samples > 0:
            f.attrs["pretell_n_samples"] = pretell_n_samples
            f.attrs["pretell_sample_width_m"] = float(PRETELL_SAMPLE_WIDTH_M)
            # Legacy attr name retained for older analyze scripts.
            f.attrs["pretell_central_width_m"] = float(PRETELL_SAMPLE_WIDTH_M)
        if p.method == "hallal_dmin":
            f.attrs["dmin_multiplier"] = dmin_multiplier_for(p)
        grp = f.create_group("params")
        grp.attrs["sobol_id"] = p.sobol_id
        grp.attrs["Vs1"] = p.vs1
        grp.attrs["H"] = p.H
        grp.attrs["Vs2"] = p.vs2
        grp.attrs["CoV"] = p.cov
        grp.attrs["rH"] = p.rH
        grp.attrs["rV"] = p.rV
        grp.attrs["aHV"] = p.aHV
        grp.attrs["seed"] = p.seed
        grp.attrs["seed_kind"] = p.seed_kind
        grp.attrs["bedrock_thickness"] = p.bedrock_thickness
        f.create_dataset("Vs_profile_1D", data=vs_profile_1d.astype(np.float32))
        f.create_dataset("Vs_field", data=vs_field.astype(np.float32))
        f.create_dataset("Damping_zeta", data=zeta_grid.astype(np.float32))
        g = f.create_group("grid")
        adx, adz = _grid_spacing(p) if _is_rf_method(p) else (active_dx(), active_dz())
        g.attrs["Lx"] = vs_field.shape[1] * adx if vs_field.ndim == 2 else adz
        g.attrs["Lz"] = vs_field.shape[0] * adz
        g.attrs["dx"] = adx
        g.attrs["dz"] = adz
        g.attrs["dt"] = dt
        if len(time_arr):
            rec = f.create_group("recorders").create_group("accel")
            rec.create_dataset("time", data=time_arr.astype(np.float32))
            rec.create_dataset("data", data=accel.astype(np.float32))
        ims = f.create_group("ims")
        ims.attrs["PGA_surface"] = pga_val
        ims.create_dataset("Sa_periods", data=periods.astype(np.float32))
        ims.create_dataset("Sa_surface", data=sa_surf.astype(np.float32))
        tf = f.create_group("transfer_function")
        tf.create_dataset("freq", data=freq.astype(np.float32))
        tf.create_dataset("AF", data=af.astype(np.float32))
        if af_spatial is not None and af_spatial_stats is not None:
            tf.create_dataset("AF_spatial", data=af_spatial.astype(np.float32))
            tf.create_dataset("AF_spatial_median", data=af_spatial_stats["median"])
            tf.create_dataset("AF_spatial_p16", data=af_spatial_stats["p16"])
            tf.create_dataset("AF_spatial_p84", data=af_spatial_stats["p84"])
            tf.create_dataset("AF_spatial_sigma_ln", data=af_spatial_stats["sigma_ln"])
            tf.attrs["n_spatial_nodes"] = int(af_spatial.shape[0])
            src = "gifno_surrogate" if analysis_backend == "gifno_surrogate" else "row_recorders"
            tf.attrs["spatial_source"] = src


# ---------------------------------------------------------------------------
# Run case
# ---------------------------------------------------------------------------


def run_case(index: int, *, force: bool = False) -> str:
    p = index_to_params(index)
    h5_path = _h5_path(index)
    if h5_path.exists() and not force:
        print(f"[skip] {h5_path} exists")
        return "skipped"

    out_dir = _output_dir(p)
    out_dir.mkdir(parents=True, exist_ok=True)
    motion_freq = motion_frequency(p.vs1, p.motion_id, H=p.H)
    task_id = case_tag(p)
    rng = np.random.default_rng(p.seed)

    is_gifno = p.method == "grf_2d"
    is_ops_2d = p.method == "opensees_2d"
    is_2d = is_gifno or is_ops_2d
    is_pretell = p.method == "pretell"
    dx, dz = _grid_spacing(p)

    if is_2d or is_pretell:
        vs_field, bedrock_mask, vs_profile_1d = _build_2d_grf(p, p.seed)
    else:
        vs_field, bedrock_mask, vs_profile_1d = _build_1d_hallal(p, rng)

    lz = vs_field.shape[0] * dz
    config = _analysis_config(p, lz, motion_freq, bc_2d=is_2d)
    zeta_grid = get_damping_zeta_grid(
        vs_field,
        config.damping_method,
        config.Lx,
        lz,
        config.hx,
        dz,
        bedrock_mask=bedrock_mask,
        config=config,
    )

    rho = np.ones_like(vs_field, dtype=float) * 2000.0
    nu = np.ones_like(vs_field, dtype=float) * 0.3

    print(f"[run] index={index} {task_id}")
    print(
        f"  sobol={p.sobol_id} method={p.method} H={p.H:.1f} "
        f"f={motion_freq:.3f} Hz seed={p.seed} ({p.seed_kind})"
    )
    if _is_rf_method(p):
        print(
            f"  NO grid: Lx={vs_field.shape[1] * dx:.0f} m × Lz={lz:.0f} m "
            f"(dx=dz={dx} m, var={active_rf_lx_var():.0f} m)"
        )

    if is_pretell:
        t0 = time.time()
        n_pretell = active_pretell_n_samples()
        print(
            f"  backend=opensees_pretell ({n_pretell} profiles across "
            f"{PRETELL_SAMPLE_WIDTH_M:.0f} m variability strip)"
        )
        freq, af, n_done = _run_pretell_profiles(
            p,
            vs_field_2d=vs_field,
            bedrock_mask_2d=bedrock_mask,
            vs_profile_1d=vs_profile_1d,
            out_dir=out_dir,
            task_id=task_id,
            motion_freq=motion_freq,
        )
        _write_h5(
            h5_path,
            p,
            vs_profile_1d=vs_profile_1d,
            vs_field=vs_field,
            zeta_grid=zeta_grid,
            time_arr=np.array([]),
            accel=np.array([]).reshape(0, 0),
            dt=config.dt,
            task_id=task_id,
            freq=freq,
            af=af,
            tf_only=True,
            analysis_backend="opensees_pretell",
            pretell_n_samples=n_done,
        )
        print(f"  pretell geomean AF from {n_done} profiles in {time.time() - t0:.1f}s")
        print(f"  wrote {h5_path}")
        return "ok"

    # GIFNO surrogate is only for grf_2d — never for opensees_2d baseline.
    if is_gifno:
        from surrogate_2d import (
            central_af_index,
            predict_transfer_functions,
            use_surrogate_2d,
        )

        if use_surrogate_2d():
            t0 = time.time()
            print("  backend=gifno_surrogate (grf_2d)")
            freq, af_spatial, af_spatial_stats = predict_transfer_functions(
                vs_field,
                zeta_grid,
                dx=dx,
                dz=dz,
                bc_width=active_rf_bc_width(),
                lx_var=active_rf_lx_var(),
            )
            af = af_spatial[central_af_index()]
            _write_h5(
                h5_path,
                p,
                vs_profile_1d=vs_profile_1d,
                vs_field=vs_field,
                zeta_grid=zeta_grid,
                time_arr=np.array([]),
                accel=np.array([]).reshape(0, 0),
                dt=config.dt,
                task_id=task_id,
                freq=freq,
                af=af,
                af_spatial=af_spatial,
                af_spatial_stats=af_spatial_stats,
                tf_only=True,
                analysis_backend="gifno_surrogate",
            )
            print(f"  surrogate done in {time.time() - t0:.1f}s")
            print(f"  spatial AF from {af_spatial.shape[0]} surrogate recorders")
            print(f"  wrote {h5_path}")
            return "ok"

    if is_ops_2d:
        print("  backend=opensees_2d (baseline)")

    model_data = build_model_data(config, vs_field, rho, nu, bedrock_mask=bedrock_mask)
    info = get_solver_info(config)
    print(
        f"  solver={info['solver_type']} domain={config.Lx}x{lz} m damping={config.damping_method}"
    )

    t0 = time.time()
    status = run_opensees_analysis(config, model_data, task_id, str(out_dir))
    print(f"  analysis done in {time.time() - t0:.1f}s")
    if status.startswith("Failed"):
        raise RuntimeError(status)

    recorder_dir = out_dir / task_id
    time_arr, surface, base = _load_surface_base(recorder_dir)
    if len(time_arr) == 0:
        raise RuntimeError(f"No recorder output in {recorder_dir}")

    _, accel = _load_recorder_txt(recorder_dir)
    if accel.size == 0:
        accel = np.column_stack([surface, base])
    dt = config.dt
    if len(time_arr) > 1:
        dt = float(time_arr[1] - time_arr[0])

    vs_min = float(np.min(vs_field[~bedrock_mask])) if np.any(~bedrock_mask) else float(p.vs1)
    freq, af = TTF(surface, base, dt=dt, Vsmin=vs_min, dz=dz)

    af_spatial = None
    af_spatial_stats = None
    if is_2d:
        lateral = _load_lateral_surface_base(recorder_dir)
        if lateral is not None:
            _t_lat, surf_nodes, base_nodes = lateral
            _freq_sp, af_spatial, af_spatial_stats = _spatial_af_percentiles(
                surf_nodes, base_nodes, dt, vs_min=vs_min, dz=dz
            )
            print(f"  spatial AF from {af_spatial.shape[0]} surface nodes")

    _write_h5(
        h5_path,
        p,
        vs_profile_1d=vs_profile_1d,
        vs_field=vs_field,
        zeta_grid=zeta_grid,
        time_arr=time_arr,
        accel=accel,
        dt=dt,
        task_id=task_id,
        freq=freq,
        af=af,
        af_spatial=af_spatial,
        af_spatial_stats=af_spatial_stats,
        analysis_backend="opensees_2d" if is_ops_2d else "opensees",
    )
    print(f"  wrote {h5_path}")
    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser(description="Response_Variability comparison runner")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--index-start", type=int, default=None)
    parser.add_argument("--index-end", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    n = total_combinations()
    print(
        f"Total combinations: {n} "
        f"(smoke={os.getenv('RV_SMOKE', '0')}, smoke_2d={os.getenv('RV_SMOKE_2D', '0')})"
    )

    if args.index_start is not None:
        start = args.index_start
        end = args.index_end if args.index_end is not None else start
        indices = range(start, min(end + 1, n))
    else:
        indices = [args.index]

    failed = []
    for idx in indices:
        try:
            run_case(idx, force=args.force)
        except Exception as e:
            print(f"[ERROR] index={idx}: {e}", file=sys.stderr)
            failed.append(idx)

    if failed:
        print(f"Failed indices: {failed}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
