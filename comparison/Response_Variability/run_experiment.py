"""
Response_Variability comparison campaign runner.

Compares 2D GRF reference, de la Torre protocol, and Hallal 1D methods on
shallow synthetic H<20 m domains. Response-focused outputs (HDF5).

Usage:
  python run_experiment.py --index 0
  RV_SMOKE=1 python run_experiment.py --index 0   # 10 seeds, 1 motion, Vs1=230
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
    AHV,
    BEDROCK_DEPTH,
    COV,
    DAMPING_ZETA_BASE,
    DAMPING_ZETA_DMIN_MULT,
    RH,
    RV,
    THICKNESS,
    VS2,
    CaseParams,
    active_duration,
    active_dx,
    active_dz,
    active_lx_total,
    active_lx_var,
    case_tag,
    index_to_params,
    motion_frequency,
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
    """OpenSees run directory: results/idx_NNNNN/ (recorders in idx_NNNNN/task_id/)."""
    return _results_root() / f"idx_{p.index:05d}"


def _recorder_y_sort_key(path: Path) -> float:
    m = re.search(r"(?:center_node_|row_)y([\d.]+)_", path.name)
    return float(m.group(1)) if m else 0.0


def _load_surface_base(recorder_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load time, center surface accel, center base accel."""
    center = sorted(recorder_dir.glob("center_node_y*_dof1_accel.txt"), key=_recorder_y_sort_key)
    if len(center) >= 2:
        base_arr = np.loadtxt(center[0])
        surf_arr = np.loadtxt(center[-1])
        if base_arr.ndim == 1:
            base_arr = base_arr.reshape(-1, 1)
        if surf_arr.ndim == 1:
            surf_arr = surf_arr.reshape(-1, 1)
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
    """Load time and multi-node surface/base accel from lateral row recorders.

    Returns
    -------
    time, surface (n_time, n_nodes), base (n_time, n_nodes)
    or None if fewer than 2 lateral channels are available.
    """
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
    # Match channel count (time column already removed)
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
    """AF(f) at each surface node vs co-located base node; return percentiles."""
    # TTF_batch_fast expects (n_channels, n_time)
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
    """Load time and multi-channel accel from recorder directory."""
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


def _profile_config(vs1: float) -> ProfileRandomizationConfig:
    dz = active_dz()
    return ProfileRandomizationConfig(
        vs_mean=vs1,
        thickness=THICKNESS,
        dz=dz,
        vs_bedrock=VS2,
        bedrock_thickness=BEDROCK_DEPTH,
        cov=COV,
    )


def _build_2d_grf(
    vs1: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Vs_extended, bedrock_mask_extended, Vs_profile_1D)."""
    dx, dz = active_dx(), active_dz()
    layer_1_count = int(THICKNESS / dz)
    layer_2_count = int(BEDROCK_DEPTH / dz)
    vs_profile_1d = np.array([vs1] * layer_1_count + [VS2] * layer_2_count)
    lz = len(vs_profile_1d) * dz

    rv = RV
    ahv = RH / rv

    vs_var, _, _, _, bedrock_mask_var = _generate_vs_variability_field(
        vs_profile_1d,
        active_lx_var(),
        lz,
        dx,
        dz,
        RH,
        ahv,
        COV,
        seed=seed,
        dz_1D=dz,
        interlayer_seed=14,
        interlayer_amplitude=0.0,
    )
    vs_ext, _ = _extend_profile(vs_var, Lx=active_lx_total(), dx=dx)
    bedrock_ext, _ = _extend_profile(bedrock_mask_var.astype(float), Lx=active_lx_total(), dx=dx)
    return vs_ext, bedrock_ext.astype(bool), vs_profile_1d


def _build_1d_profile(
    p: CaseParams,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (Vs_2d nz x 1, bedrock_mask, Vs_profile_1D)."""
    cfg = _profile_config(p.vs1)

    if p.method == "hallal_vs":
        prof = generate_vs_randomized_profile_full(cfg, rng)
    elif p.method == "hallal_tts":
        prof = generate_tts_randomized_profile_full(cfg, rng)
    elif p.method == "hallal_dmin":
        from seiskit.profile_randomization import build_base_case_profile

        vs_profile = build_base_case_profile(cfg)
        prof = RandomizedProfile(
            vs_depth=vs_profile,
            n_soil_samples=max(1, int(round(THICKNESS / active_dz()))),
            interface_depth=THICKNESS,
        )
    else:
        raise ValueError(f"Not a 1D method: {p.method}")

    if p.method != "hallal_dmin":
        vs_profile = prof.vs_depth
    vs_col, mask = profile_to_opensees_column(
        vs_profile,
        prof.n_soil_samples,
    )
    return vs_col, mask, vs_profile


def _analysis_config(
    p: CaseParams,
    lz: float,
    motion_freq: float,
    *,
    bc_2d: bool,
) -> AnalysisConfig:
    dx, dz = active_dx(), active_dz()
    f0 = p.vs1 / (4.0 * THICKNESS)
    duration = active_duration(f0)
    damping_freq_first = min(f0, motion_freq)

    lx = active_lx_total() if bc_2d else dz
    record_lateral = (10, 2.0) if bc_2d else None

    return AnalysisConfig(
        Ly=lz,
        Lx=lx,
        hx=dx if bc_2d else dz,
        dt=0.01,
        duration=duration,
        motion_freq=motion_freq,
        motion_t_shift=0.5,
        damping_freqs=(damping_freq_first, 10.0),
        damping_zeta=DAMPING_ZETA_BASE,
        damping_method="global_avg",
        boundary_condition_type="2D" if bc_2d else "1D",
        record_center_nodes=True,
        center_node_y_positions=[2.0, lz],
        record_lateral_span_at_center_depths=record_lateral,
        record_all_surface_nodes=False,
        element_type="4node",
        solver_type="Mumps",
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
) -> None:
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    periods = default_periods(p.vs1, THICKNESS)
    if len(time_arr) == 0:
        raise ValueError("Cannot write HDF5 without time series")
    surface = accel[:, 0] if accel.ndim == 2 and accel.shape[1] > 0 else accel.ravel()
    sa_surf = compute_sa(surface, dt, periods)

    with h5py.File(path, "w") as f:
        f.attrs["index"] = p.index
        f.attrs["task_id"] = task_id
        f.attrs["method"] = p.method
        f.attrs["motion_id"] = p.motion_id
        grp = f.create_group("params")
        grp.attrs["Vs1"] = p.vs1
        grp.attrs["CoV"] = COV
        grp.attrs["rH"] = RH
        grp.attrs["rV"] = RV
        grp.attrs["aHV"] = AHV
        grp.attrs["seed"] = p.seed
        grp.attrs["thickness"] = THICKNESS
        f.create_dataset("Vs_profile_1D", data=vs_profile_1d.astype(np.float32))
        f.create_dataset("Vs_field", data=vs_field.astype(np.float32))
        f.create_dataset("Damping_zeta", data=zeta_grid.astype(np.float32))
        g = f.create_group("grid")
        adx, adz = active_dx(), active_dz()
        g.attrs["Lx"] = vs_field.shape[1] * adx if vs_field.ndim == 2 else adz
        g.attrs["Lz"] = vs_field.shape[0] * adz
        g.attrs["dx"] = active_dx()
        g.attrs["dz"] = active_dz()
        g.attrs["dt"] = dt
        if len(time_arr):
            rec = f.create_group("recorders").create_group("accel")
            rec.create_dataset("time", data=time_arr.astype(np.float32))
            rec.create_dataset("data", data=accel.astype(np.float32))
        ims = f.create_group("ims")
        ims.attrs["PGA_surface"] = pga(surface)
        ims.create_dataset("Sa_periods", data=periods.astype(np.float32))
        ims.create_dataset("Sa_surface", data=sa_surf.astype(np.float32))
        tf = f.create_group("transfer_function")
        tf.create_dataset("freq", data=freq.astype(np.float32))
        tf.create_dataset("AF", data=af.astype(np.float32))
        # Spatial AF percentiles across lateral surface nodes (2D arms)
        if af_spatial is not None and af_spatial_stats is not None:
            tf.create_dataset("AF_spatial", data=af_spatial.astype(np.float32))
            tf.create_dataset("AF_spatial_median", data=af_spatial_stats["median"])
            tf.create_dataset("AF_spatial_p16", data=af_spatial_stats["p16"])
            tf.create_dataset("AF_spatial_p84", data=af_spatial_stats["p84"])
            tf.create_dataset("AF_spatial_sigma_ln", data=af_spatial_stats["sigma_ln"])
            tf.attrs["n_spatial_nodes"] = int(af_spatial.shape[0])
            tf.attrs["spatial_source"] = "row_recorders"


def run_case(index: int, *, force: bool = False) -> str:
    p = index_to_params(index)
    h5_path = _h5_path(index)
    if h5_path.exists() and not force:
        print(f"[skip] {h5_path} exists")
        return "skipped"

    out_dir = _output_dir(p)
    out_dir.mkdir(parents=True, exist_ok=True)
    motion_freq = motion_frequency(p.vs1, p.motion_id)
    task_id = case_tag(p)
    rng = np.random.default_rng(p.seed)

    is_2d = p.method in ("grf_2d", "delatorre_2d")
    if is_2d:
        vs_field, bedrock_mask, vs_profile_1d = _build_2d_grf(p.vs1, p.seed)
        lz = vs_field.shape[0] * active_dz()
    else:
        vs_field, bedrock_mask, vs_profile_1d = _build_1d_profile(p, rng)
        lz = vs_field.shape[0] * active_dz()

    config = _analysis_config(p, lz, motion_freq, bc_2d=is_2d)
    zeta_mult = DAMPING_ZETA_DMIN_MULT if p.method == "hallal_dmin" else 1.0
    config.damping_zeta = DAMPING_ZETA_BASE * zeta_mult

    _dx, dz = active_dx(), active_dz()
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
    print(f"  method={p.method} motion={p.motion_id} f={motion_freq:.3f} Hz seed={p.seed}")

    model_data = build_model_data(config, vs_field, rho, nu, bedrock_mask=bedrock_mask)
    info = get_solver_info(config)
    print(f"  solver={info['solver_type']} domain={config.Lx}x{lz} m")

    t0 = time.time()
    run_opensees_analysis(config, model_data, task_id, str(out_dir))
    print(f"  analysis done in {time.time() - t0:.1f}s")

    recorder_dir = out_dir / task_id
    time_arr, surface, base = _load_surface_base(recorder_dir)
    if len(time_arr) == 0:
        raise RuntimeError(f"No recorder output in {recorder_dir}")

    # Full channel block for HDF5 (optional)
    _, accel = _load_recorder_txt(recorder_dir)
    if accel.size == 0:
        accel = np.column_stack([surface, base])
    dt = config.dt
    if len(time_arr) > 1:
        dt = float(time_arr[1] - time_arr[0])

    _dx, dz = active_dx(), active_dz()
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
    print(f"Total combinations: {n} (smoke={os.getenv('RV_SMOKE', '0')})")

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
