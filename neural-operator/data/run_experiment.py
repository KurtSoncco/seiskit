"""
Sobol-driven neural-operator data generation runner.

This mirrors the lifecycle of the emulator runners under ``data_generation``:
``index -> parameters -> run_case(index) -> outputs/HDF5``.

The key difference is that indices map to rows in a persisted Sobol manifest
instead of a hard-coded Cartesian parameter grid.
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import numpy as np

from seiskit.analysis import run_opensees_analysis
from seiskit.builder import build_model_data
from seiskit.config import AnalysisConfig
from seiskit.gaussian_field import _extend_profile, _generate_vs_variability_field
from seiskit.plot_results import get_damping_zeta_grid
from seiskit.solver_utils import get_solver_info
from sobol import (
    DEFAULT_BEDROCK_THICKNESS,
    DEFAULT_DZ_1D,
    DEFAULT_MOTION_FREQ,
    DEFAULT_RF_SEEDS_PER_SAMPLE,
    DEFAULT_RF_SEED_GENERATOR,
    DEFAULT_SAMPLER_SEED,
    DEFAULT_SOBOL_SAMPLE_COUNT,
    ManifestEntry,
    ensure_manifest,
)

DEFAULT_MANIFEST_PATH = THIS_DIR / "sobol_manifest.csv"
CASE_TYPE = "sobol_6d_4node"
ELEMENT_TYPE = "4node"
BEDROCK_THICKNESS = DEFAULT_BEDROCK_THICKNESS
DZ_1D = DEFAULT_DZ_1D
DX = 1.0
DZ = 1.0
LX_VARIABILITY = 500.0
BC_WIDTH = 500.0
LX = LX_VARIABILITY + 2 * BC_WIDTH
INTERLAYER_AMPLITUDE = 0.0
INTERLAYER_SEED = 14
MOTION_FREQ = DEFAULT_MOTION_FREQ
DAMPING_FREQ_SECOND = 10.0
DAMPING_METHOD = "global_avg"
# Per-batch dynamic watchdog in seiskit/isolated_runner (SIGALRM uses this + 30 s).
DEFAULT_SOBOL_MAX_TIME_PER_BATCH_SEC = 8 * 3600.0


def _sobol_max_time_per_batch() -> float:
    """Seconds allowed per dynamic batch (typically 100 steps). Override via SOBOL_MAX_TIME_PER_BATCH."""
    raw = os.getenv("SOBOL_MAX_TIME_PER_BATCH", "").strip()
    if raw:
        return float(raw)
    return DEFAULT_SOBOL_MAX_TIME_PER_BATCH_SEC


def _configure_slurm_environment() -> None:
    """Configure threading and report SLURM/gnu-parallel context."""
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if not os.getenv(var):
            os.environ[var] = "1"

    job_id = os.getenv("SLURM_JOB_ID", "-")
    task_id = os.getenv("SLURM_ARRAY_TASK_ID", "-")
    parallel_seq = os.getenv("PARALLEL_SEQ", "-")
    node = os.getenv("SLURMD_NODENAME", os.uname().nodename)
    print(f"[slurm] job_id={job_id} task_id={task_id} parallel_seq={parallel_seq} node={node}")


def _install_sigterm_handler() -> None:
    def _handler(signum, frame):  # noqa: ARG001
        try:
            print("[signal] Received SIGTERM. Attempting graceful shutdown...")
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(143)

    try:
        signal.signal(signal.SIGTERM, _handler)
    except Exception:
        pass


def _fmt_hms(seconds: float) -> str:
    total_seconds = int(seconds)
    return (
        f"{total_seconds // 3600:02d}:{(total_seconds % 3600) // 60:02d}:{total_seconds % 60:02d}"
    )


def _heartbeat_interval_seconds() -> int:
    """Heartbeat interval from HEARTBEAT_MINUTES (default 5). 0 disables heartbeat."""
    try:
        minutes = int(os.getenv("HEARTBEAT_MINUTES", "5"))
        return max(0, minutes) * 60
    except ValueError:
        return 300


def _start_heartbeat(index: int) -> threading.Event:
    """Start a daemon thread that prints a heartbeat to stderr every N minutes."""
    stop = threading.Event()

    def _run() -> None:
        interval = _heartbeat_interval_seconds()
        if interval <= 0:
            return
        t0 = time.time()
        while not stop.wait(timeout=interval):
            elapsed = time.time() - t0
            try:
                print(
                    f"[heartbeat] index={index} elapsed_sec={elapsed:.0f}",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception:
                break

    threading.Thread(target=_run, daemon=True).start()
    return stop


def _index_to_entry(index: int, manifest_entries: list[ManifestEntry]) -> ManifestEntry:
    total = len(manifest_entries)
    if index < 0 or index >= total:
        raise IndexError(f"Index {index} is out of range for {total} tasks (valid 0..{total - 1}).")
    return manifest_entries[index]


def _param_dir_name(entry: ManifestEntry) -> str:
    """Directory name for one manifest row."""
    return f"sample_{entry.sample_id:04d}_rep_{entry.replicate_id:02d}_s{entry.rf_seed}"


def _output_dir_for_index(index: int, entry: ManifestEntry) -> Path:
    """Output directory for this index. Uses SOBOL_OUTDIR scratch if configured."""
    base = os.getenv("SOBOL_OUTDIR")
    param_dir = _param_dir_name(entry)
    if base:
        return Path(base) / f"run_{index}" / param_dir
    return Path("results") / param_dir


def _run_dir_for_index(index: int) -> Path | None:
    """Run directory to archive (parent of param dir when using scratch)."""
    base = os.getenv("SOBOL_OUTDIR")
    if not base:
        return None
    return Path(base) / f"run_{index}"


def _h5_path_for_index(index: int) -> Path:
    """HDF5 output path for this index."""
    base = os.getenv("SOBOL_H5_DIR")
    if base:
        return Path(base) / f"run_{index}.h5"
    return Path("results") / "h5" / f"run_{index}.h5"


def _archive_run_dir(run_dir: Path, archives_dir: Path, index: int) -> bool:
    """Tar+compress run_dir, verify it, then delete the uncompressed directory."""
    import shutil

    archives_dir.mkdir(parents=True, exist_ok=True)
    arc_zst = archives_dir / f"run_{index}.tar.zst"
    try:
        has_zstd = shutil.which("zstd") is not None
        if has_zstd:
            cmd = f"tar -C '{run_dir.parent}' -cvf - '{run_dir.name}' | zstd -T0 -19 -o '{arc_zst}'"
            subprocess.run(cmd, shell=True, check=True)
            subprocess.run(["zstd", "-t", str(arc_zst)], check=True, capture_output=True)
        else:
            arc_tgz = archives_dir / f"run_{index}.tgz"
            subprocess.run(
                ["tar", "-C", str(run_dir.parent), "-czvf", str(arc_tgz), run_dir.name],
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["tar", "-tzf", str(arc_tgz)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except subprocess.CalledProcessError as exc:
        print(f"[archive] WARNING: compress failed for run_{index}: {exc}", file=sys.stderr)
        return False
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    return True


def _recorder_y_sort_key(path: Path) -> float:
    """Parse depth from ``center_node_y...`` or ``row_y...`` filenames."""
    match = re.search(r"(?:center_node_|row_)y([\d.]+)_", path.name)
    if not match:
        return 0.0
    return float(match.group(1))


def _load_recorder_txt(recorder_dir: Path, quantity: str = "accel") -> tuple[np.ndarray, np.ndarray]:
    """Load OpenSees recorder files and return ``(time, data)``."""
    center_glob = list(recorder_dir.glob(f"center_node_y*_dof1_{quantity}.txt"))
    row_glob = list(recorder_dir.glob(f"row_y*_dof1_{quantity}.txt"))
    center_glob.sort(key=_recorder_y_sort_key)
    row_glob.sort(key=_recorder_y_sort_key)

    chunks: list[np.ndarray] = []
    time_arr: np.ndarray | None = None
    for file_path in center_glob + row_glob:
        arr = np.loadtxt(file_path)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if time_arr is None:
            time_arr = arr[:, 0]
        if arr.shape[1] > 1:
            chunks.append(arr[:, 1:])

    if not chunks:
        return np.array([]), np.array([]).reshape(0, 0)

    data = np.hstack(chunks)
    assert time_arr is not None
    return time_arr, data


def _get_compression_grid():
    """Compression for Vs/damping grids. Prefer Blosc2/zstd, fallback to gzip-9."""
    try:
        import hdf5plugin  # type: ignore[import-untyped]

        return hdf5plugin.Blosc2(cname="zstd", clevel=5, filters=hdf5plugin.Blosc2.SHUFFLE)  # type: ignore[attr-defined]
    except ImportError:
        return {"compression": "gzip", "compression_opts": 9, "shuffle": True}


def _get_compression_recorder_lossless():
    """Compression for recorder time series. Prefer Blosc2/zstd+DELTA, fallback to gzip-9."""
    try:
        import hdf5plugin  # type: ignore[import-untyped]

        return hdf5plugin.Blosc2(cname="zstd", clevel=5, filters=hdf5plugin.Blosc2.DELTA)  # type: ignore[attr-defined]
    except ImportError:
        return {"compression": "gzip", "compression_opts": 9, "shuffle": True}


def _get_compression_recorder_lossy(tolerance: float = 1e-5):
    """Lossy compression for recorder data. Falls back to lossless when unavailable."""
    try:
        import hdf5plugin  # type: ignore[import-untyped]

        return hdf5plugin.Zfp(accuracy=tolerance)  # type: ignore[attr-defined]
    except ImportError:
        return _get_compression_recorder_lossless()


def _write_h5(
    h5_path: Path,
    index: int,
    entry: ManifestEntry,
    manifest_path: Path,
    Vs_profile_1D: np.ndarray,
    Vs_extended: np.ndarray,
    zeta_grid: np.ndarray,
    recorder_dir: Path,
    Lx: float,
    Lz: float,
    dx: float,
    dz: float,
    dt: float,
    task_id: str,
) -> None:
    """Write one HDF5 per manifest index."""
    try:
        import h5py  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("h5py is required for HDF5 output; install with pip install h5py") from exc

    h5_path.parent.mkdir(parents=True, exist_ok=True)
    lossy = os.getenv("SOBOL_H5_LOSSY", "0") == "1"
    lossy_tol = float(os.getenv("SOBOL_H5_LOSSY_TOLERANCE", "1e-5"))
    comp_grid = _get_compression_grid()
    comp_ts = _get_compression_recorder_lossy(lossy_tol) if lossy else _get_compression_recorder_lossless()

    def _kw(compression):
        return compression if isinstance(compression, dict) else {"compression": compression}

    time_arr, data = _load_recorder_txt(recorder_dir, quantity="accel")
    downsample = max(1, int(os.getenv("SOBOL_H5_DOWNSAMPLE", "1")))
    dt_stored = dt
    if len(time_arr) > 0 and downsample > 1:
        time_arr = time_arr[::downsample].copy()
        data = data[::downsample, :].copy()
        dt_stored = dt * downsample

    with h5py.File(h5_path, "w") as handle:
        handle.attrs["index"] = index
        handle.attrs["task_id"] = task_id
        handle.attrs["sample_id"] = entry.sample_id
        handle.attrs["replicate_id"] = entry.replicate_id
        handle.attrs["rf_seed"] = entry.rf_seed
        handle.attrs["manifest_path"] = str(manifest_path.resolve())
        comp_name = "gzip" if isinstance(comp_grid, dict) else "blosc2_zstd"
        handle.attrs["h5_compression"] = f"{comp_name}_lossy_recorder" if lossy else comp_name

        params = handle.create_group("params")
        params.attrs["Vs1"] = entry.Vs1
        params.attrs["H"] = entry.H_discretized
        params.attrs["H_requested"] = entry.H_requested
        params.attrs["H_discretized"] = entry.H_discretized
        params.attrs["soil_layer_count"] = entry.soil_layer_count
        params.attrs["CoV"] = entry.CoV
        params.attrs["rH"] = entry.rH
        params.attrs["aHV"] = entry.aHV
        params.attrs["Vs2"] = entry.Vs2
        params.attrs["seed"] = entry.rf_seed
        params.attrs["rf_seed"] = entry.rf_seed
        params.attrs["bedrock_layer_count"] = entry.bedrock_layer_count
        params.attrs["f0_effective"] = entry.f0_effective
        params.attrs["duration"] = entry.duration
        params.attrs["damping_freq_first"] = entry.damping_freq_first

        handle.create_dataset(
            "Vs_profile_1D",
            data=Vs_profile_1D.astype(np.float32),
            **_kw(comp_grid),
        )
        handle.create_dataset(
            "Vs_realization_2D",
            data=Vs_extended.astype(np.float32),
            **_kw(comp_grid),
        )
        handle.create_dataset(
            "Damping_zeta",
            data=zeta_grid.astype(np.float32),
            **_kw(comp_grid),
        )

        grid = handle.create_group("grid")
        grid.attrs["Lx"] = Lx
        grid.attrs["Lz"] = Lz
        grid.attrs["Lz_discretized"] = entry.Lz_discretized
        grid.attrs["dx"] = dx
        grid.attrs["dz"] = dz
        grid.attrs["dz_1D"] = entry.dz_1D
        grid.attrs["dt"] = dt_stored
        grid.attrs["motion_freq"] = entry.motion_freq
        grid.attrs["bedrock_thickness"] = entry.bedrock_thickness
        grid.attrs["bedrock_thickness_discretized"] = entry.bedrock_thickness_discretized

        if len(time_arr) > 0:
            accel = handle.create_group("recorders").create_group("accel")
            n_time, n_ch = data.shape[0], max(1, data.shape[1])
            chunks_ts = (min(1000, n_time), min(64, n_ch))
            accel.create_dataset("time", data=time_arr.astype(np.float32), **_kw(comp_ts))
            ds_data = accel.create_dataset(
                "data",
                data=data.astype(np.float32),
                chunks=chunks_ts,
                **_kw(comp_ts),
            )
            ds_data.attrs["layout"] = "n_time x n_channels"
            row_paths = list(recorder_dir.glob("row_y*_dof1_accel.txt"))
            if row_paths:
                ys = sorted({_recorder_y_sort_key(path) for path in row_paths})
                ds_data.attrs["row_y_m"] = np.asarray(ys, dtype=np.float64)
            if lossy:
                ds_data.attrs["compression_tolerance"] = lossy_tol


def run_case(index: int, manifest_entries: list[ManifestEntry], manifest_path: Path) -> str:
    """Run a single Sobol case given the flattened manifest index."""
    t0 = time.time()
    heartbeat_stop = _start_heartbeat(index)
    try:
        return _run_case_impl(index, manifest_entries, manifest_path, t0)
    finally:
        heartbeat_stop.set()


def _run_case_impl(
    index: int,
    manifest_entries: list[ManifestEntry],
    manifest_path: Path,
    t0: float,
) -> str:
    """Actual case execution (heartbeat already started by ``run_case``)."""
    entry = _index_to_entry(index, manifest_entries)

    Vs_profile_1D = np.concatenate(
        [
            np.full(entry.soil_layer_count, entry.Vs1, dtype=float),
            np.full(entry.bedrock_layer_count, entry.Vs2, dtype=float),
        ]
    )
    Lz = entry.Lz_discretized
    f0 = entry.f0_effective
    duration = entry.duration
    damping_freq_first = entry.damping_freq_first
    damping_freqs = (damping_freq_first, DAMPING_FREQ_SECOND)

    output_dir = _output_dir_for_index(index, entry)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / "Vs_profile_1D.txt", Vs_profile_1D)

    task_id = f"{CASE_TYPE}_sample{entry.sample_id:04d}_rep{entry.replicate_id:02d}_seed{entry.rf_seed}"

    print(f"[{CASE_TYPE}] Starting task {task_id} (index={index})")
    print(
        "  "
        f"Vs1={entry.Vs1:.3f}, H_requested={entry.H_requested:.3f}, H_discretized={entry.H_discretized:.1f}, "
        f"CoV={entry.CoV:.4f}, rH={entry.rH:.3f}, aHV={entry.aHV:.3f}, Vs2={entry.Vs2:.3f}, seed={entry.rf_seed}"
    )
    print(f"  Lx={LX} m, Lz={Lz} m, dx={DX} m, dz={DZ} m")
    print(f"  f0={f0:.4f} Hz, duration={duration:.1f}s, damping_freqs={damping_freqs}")

    t_field_start = time.time()
    Vs_realization, _, _, _, bedrock_mask_var = _generate_vs_variability_field(
        Vs_profile_1D,
        LX_VARIABILITY,
        Lz,
        DX,
        DZ,
        entry.rH,
        entry.aHV,
        entry.CoV,
        seed=entry.rf_seed,
        dz_1D=entry.dz_1D,
        interlayer_seed=INTERLAYER_SEED,
        interlayer_amplitude=INTERLAYER_AMPLITUDE,
    )
    field_generation_time = time.time() - t_field_start

    Vs_extended, _ = _extend_profile(Vs_realization, Lx=LX, dx=DX)
    bedrock_mask_extended, _ = _extend_profile(bedrock_mask_var.astype(float), Lx=LX, dx=DX)
    bedrock_mask_extended = bedrock_mask_extended.astype(bool)

    config = AnalysisConfig(
        Ly=Lz,
        Lx=LX,
        hx=DX,
        dt=0.01,
        duration=duration,
        motion_freq=entry.motion_freq,
        motion_t_shift=0.5,
        damping_freqs=damping_freqs,
        damping_zeta=0.025,
        damping_method=DAMPING_METHOD,
        boundary_condition_type="2D",
        record_center_nodes=False,
        center_node_y_positions=[2.0, Lz],
        record_lateral_span_at_center_depths=(50, 2.0),
        record_all_surface_nodes=False,
        element_type=ELEMENT_TYPE,
        solver_type="Mumps",
        max_time_per_batch=_sobol_max_time_per_batch(),
    )

    zeta_grid = get_damping_zeta_grid(
        Vs_extended,
        DAMPING_METHOD,
        LX,
        Lz,
        DX,
        DZ,
        bedrock_mask=bedrock_mask_extended,
        config=config,
    )

    solver_info = get_solver_info(config)
    print(f"  Solver: {solver_info['solver_type']}")
    print(
        f"  Dynamic batch timeout: {config.max_time_per_batch:.0f}s "
        f"({config.max_time_per_batch / 3600.0:.1f} h per batch)"
    )
    print(f"  Domain: {LX}m x {Lz}m")
    print(f"  Recording center + lateral span at Y={config.center_node_y_positions}")

    rho = np.ones_like(Vs_extended) * 2000.0
    nu = np.ones_like(Vs_extended) * 0.3

    t_model_start = time.time()
    model_data = build_model_data(
        config,
        Vs_extended,
        rho,
        nu,
        bedrock_mask=bedrock_mask_extended,
    )
    model_build_time = time.time() - t_model_start

    t_analysis_start = time.time()
    print(f"[{CASE_TYPE}] Running OpenSees for {task_id} -> {output_dir}")
    result = run_opensees_analysis(config, model_data, task_id, str(output_dir))
    analysis_time = time.time() - t_analysis_start

    h5_path = _h5_path_for_index(index)
    recorder_dir = output_dir / task_id
    if recorder_dir.exists():
        _write_h5(
            h5_path,
            index=index,
            entry=entry,
            manifest_path=manifest_path,
            Vs_profile_1D=Vs_profile_1D,
            Vs_extended=Vs_extended,
            zeta_grid=zeta_grid,
            recorder_dir=recorder_dir,
            Lx=LX,
            Lz=Lz,
            dx=DX,
            dz=DZ,
            dt=config.dt,
            task_id=task_id,
        )
        print(f"[{CASE_TYPE}] Wrote {h5_path}")

    run_dir = _run_dir_for_index(index)
    if run_dir is not None and run_dir.exists():
        outdir_base = Path(os.getenv("SOBOL_OUTDIR", ""))
        archives_dir = outdir_base / "archives"
        if _archive_run_dir(run_dir, archives_dir, index):
            print(f"[{CASE_TYPE}] Archived run_{index} to {archives_dir}")

    total_time = time.time() - t0
    _write_timing_per_index(
        index=index,
        entry=entry,
        task_id=task_id,
        total_time_sec=total_time,
        field_sec=field_generation_time,
        model_sec=model_build_time,
        analysis_sec=analysis_time,
        status=result,
    )

    print(f"[{CASE_TYPE}] Done: {result} | Wall time: {_fmt_hms(total_time)}")
    return result


def _write_timing_per_index(
    *,
    index: int,
    entry: ManifestEntry,
    task_id: str,
    total_time_sec: float,
    field_sec: float,
    model_sec: float,
    analysis_sec: float,
    status: str,
) -> None:
    """Append one row to timing.db. No-op on DB errors."""
    db_path = os.getenv("SOBOL_TIMING_DB") or str(Path("timing.db").resolve())
    jobid = int(os.getenv("SLURM_JOB_ID", "0"))
    taskid = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
    required_columns = [
        ("ts", "TEXT"),
        ("idx", "INT"),
        ("jobid", "INT"),
        ("taskid", "INT"),
        ("sample_id", "INT"),
        ("replicate_id", "INT"),
        ("Vs1", "REAL"),
        ("H", "REAL"),
        ("H_requested", "REAL"),
        ("H_discretized", "REAL"),
        ("soil_layer_count", "INT"),
        ("CoV", "REAL"),
        ("rH", "REAL"),
        ("aHV", "REAL"),
        ("Vs2", "REAL"),
        ("bedrock_layer_count", "INT"),
        ("Lz_discretized", "REAL"),
        ("f0_effective", "REAL"),
        ("duration", "REAL"),
        ("seed", "INT"),
        ("task_id", "TEXT"),
        ("total_time_sec", "REAL"),
        ("field_sec", "REAL"),
        ("model_sec", "REAL"),
        ("analysis_sec", "REAL"),
        ("status", "TEXT"),
    ]
    try:
        conn = sqlite3.connect(db_path, timeout=10.0)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sobol_timing_per_index (
                ts TEXT,
                idx INT,
                jobid INT,
                taskid INT,
                sample_id INT,
                replicate_id INT,
                Vs1 REAL,
                H REAL,
                H_requested REAL,
                H_discretized REAL,
                soil_layer_count INT,
                CoV REAL,
                rH REAL,
                aHV REAL,
                Vs2 REAL,
                bedrock_layer_count INT,
                Lz_discretized REAL,
                f0_effective REAL,
                duration REAL,
                seed INT,
                task_id TEXT,
                total_time_sec REAL,
                field_sec REAL,
                model_sec REAL,
                analysis_sec REAL,
                status TEXT
            )
            """
        )
        existing_columns = {
            row[1] for row in conn.execute("PRAGMA table_info(sobol_timing_per_index)").fetchall()
        }
        for column_name, column_type in required_columns:
            if column_name not in existing_columns:
                conn.execute(
                    f"ALTER TABLE sobol_timing_per_index ADD COLUMN {column_name} {column_type}"
                )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sobol_timing_per_index_idx ON sobol_timing_per_index(idx)"
        )
        conn.execute(
            """
            INSERT INTO sobol_timing_per_index(
                ts, idx, jobid, taskid, sample_id, replicate_id, Vs1, H, H_requested,
                H_discretized, soil_layer_count, CoV, rH, aHV, Vs2, bedrock_layer_count,
                Lz_discretized, f0_effective, duration, seed, task_id, total_time_sec,
                field_sec, model_sec, analysis_sec, status
            )
            VALUES(datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                index,
                jobid,
                taskid,
                entry.sample_id,
                entry.replicate_id,
                entry.Vs1,
                entry.H_discretized,
                entry.H_requested,
                entry.H_discretized,
                entry.soil_layer_count,
                entry.CoV,
                entry.rH,
                entry.aHV,
                entry.Vs2,
                entry.bedrock_layer_count,
                entry.Lz_discretized,
                entry.f0_effective,
                entry.duration,
                entry.rf_seed,
                task_id,
                total_time_sec,
                field_sec,
                model_sec,
                analysis_sec,
                status,
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def _output_already_exists(index: int, entry: ManifestEntry) -> bool:
    """Return True when this index already has an H5, archive, or raw recorder outputs."""
    h5_path = _h5_path_for_index(index)
    if h5_path.exists():
        print(f"[idempotent] Index {index} already has H5; skipping.")
        return True

    run_dir = _run_dir_for_index(index)
    if run_dir is not None:
        outdir_base = Path(os.getenv("SOBOL_OUTDIR", ""))
        arc = outdir_base / "archives" / f"run_{index}.tar.zst"
        if not arc.exists():
            arc = outdir_base / "archives" / f"run_{index}.tgz"
        if arc.exists():
            print(f"[idempotent] Index {index} already has archive; skipping.")
            return True

    out_dir = _output_dir_for_index(index, entry)
    raw_markers = list(out_dir.glob("**/center_node_*.txt")) + list(out_dir.glob("**/row_y*_dof1_*.txt"))
    if out_dir.exists() and raw_markers:
        print(f"[idempotent] Index {index} already has output; skipping.")
        return True

    return False


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Run Sobol-based neural-operator experiments from a flattened manifest."
    )
    parser.add_argument("--index", type=int, default=None, help="Single task index.")
    parser.add_argument("--index-start", type=int, default=None, help="Start of range (inclusive).")
    parser.add_argument("--index-end", type=int, default=None, help="End of range (exclusive).")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if output already exists (overrides idempotent skip).",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to the persisted Sobol manifest CSV.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_SOBOL_SAMPLE_COUNT,
        help="Number of global Sobol samples to generate when creating a manifest.",
    )
    parser.add_argument(
        "--rf-seeds-per-sample",
        type=int,
        default=DEFAULT_RF_SEEDS_PER_SAMPLE,
        help="Number of RF seed replicates per Sobol sample when creating a manifest.",
    )
    parser.add_argument(
        "--sampler-method",
        choices=("sobol", "lhs"),
        default="sobol",
        help="Sampling method used when creating a manifest.",
    )
    parser.add_argument(
        "--sampler-seed",
        type=int,
        default=DEFAULT_SAMPLER_SEED,
        help="Seed for the Sobol/LHS sampler when creating a manifest.",
    )
    parser.add_argument(
        "--rf-seed-seed",
        type=int,
        default=DEFAULT_RF_SEED_GENERATOR,
        help="Seed used to generate per-sample RF seeds when creating a manifest.",
    )
    parser.add_argument(
        "--overwrite-manifest",
        action="store_true",
        help="Regenerate the manifest even if the CSV already exists.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    os.chdir(THIS_DIR)
    program_start = time.time()
    _configure_slurm_environment()
    _install_sigterm_handler()
    args = _parse_args()

    manifest_entries = ensure_manifest(
        path=args.manifest_path,
        sample_count=args.num_samples,
        seeds_per_sample=args.rf_seeds_per_sample,
        method=args.sampler_method,
        sampler_seed=args.sampler_seed,
        rf_seed_seed=args.rf_seed_seed,
        overwrite=args.overwrite_manifest,
    )
    manifest_path = Path(args.manifest_path).resolve()
    total_combinations = len(manifest_entries)
    print(f"[manifest] Using {manifest_path} ({total_combinations} tasks)")

    if args.index is not None:
        idx = args.index
        if idx < 0 or idx >= total_combinations:
            print(f"Error: --index must be 0..{total_combinations - 1}", file=sys.stderr)
            sys.exit(1)
        entry = _index_to_entry(idx, manifest_entries)
        if not args.force and _output_already_exists(idx, entry):
            sys.exit(0)
        run_case(idx, manifest_entries, manifest_path)
    elif args.index_start is not None and args.index_end is not None:
        start = max(0, args.index_start)
        end = min(total_combinations, args.index_end)
        for idx in range(start, end):
            entry = _index_to_entry(idx, manifest_entries)
            if not args.force and _output_already_exists(idx, entry):
                continue
            run_case(idx, manifest_entries, manifest_path)
    else:
        idx = args.index
        if idx is None:
            env_idx = os.getenv("SLURM_ARRAY_TASK_ID") or os.getenv("PARALLEL_SEQ")
            if env_idx not in (None, ""):
                try:
                    idx = int(env_idx)
                    if os.getenv("PARALLEL_SEQ"):
                        idx -= 1
                except ValueError:
                    idx = None
        if idx is None or idx < 0 or idx >= total_combinations:
            print(
                f"Error: provide --index 0-{total_combinations - 1} or --index-start/--index-end, "
                "or set SLURM_ARRAY_TASK_ID/PARALLEL_SEQ",
                file=sys.stderr,
            )
            sys.exit(1)
        entry = _index_to_entry(idx, manifest_entries)
        if not args.force and _output_already_exists(idx, entry):
            sys.exit(0)
        run_case(idx, manifest_entries, manifest_path)

    print(f"\n[program] Total wall time: {_fmt_hms(time.time() - program_start)}")
    sys.exit(0)
