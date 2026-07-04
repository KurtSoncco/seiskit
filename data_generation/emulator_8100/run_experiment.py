"""
Emulator 8100: 8,100 simulations for surrogate training.

Fixed:
- Height = 15 m, Vs2 = 1500 m/s, no interlayer variability (flat surface).
- Lx = 1500 m with BC_width = 500 m, dx = 1 m, dz = 1 m, Lz = 25 m (15 soil + 10 bedrock).
- Motion 3.0 Hz, damping_method = "global_avg", f0-based duration and damping_freqs.

Sweep:
- Vs1 in {100, 230, 360}, CoV in {0.1, 0.2, 0.3}, rh in {10, 30, 50}, aHV in {1, 10, 50}.
- Random seeds 1--100.

Recordings: center node + lateral span at center depths (101 nodes per row, 2 m spacing, center ±100 m).

Total: 3*3*3*3*100 = 8,100. Index 0--8099.
Supports: --index N (single), or --index-start A --index-end B (range, clamp to 8100).
Idempotent: skips if output already exists for that index. Use --force to re-run anyway.
"""

import argparse
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from seiskit.analysis import run_opensees_analysis
from seiskit.builder import build_model_data
from seiskit.config import AnalysisConfig
from seiskit.gaussian_field import (
    _extend_profile,
    _generate_vs_variability_field,
)
from seiskit.plot_results import get_damping_zeta_grid
from seiskit.solver_utils import get_solver_info

TOTAL_COMBINATIONS = 8100  # 3*3*3*3*100

# Parameter grids (order: Vs1 -> CoV -> rh -> aHV -> seed)
VS1_LIST = [100.0, 230.0, 360.0]
COV_LIST = [0.1, 0.2, 0.3]
RH_LIST = [10.0, 30.0, 50.0]
AHV_LIST = [1.0, 10.0, 50.0]
SEED_VALUES = list(range(1, 101))  # 1..100


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
    parallel_seq = os.getenv("PARALLEL_SEQ", "-")  # gnu-parallel slot (1-based)
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
        m = int(os.getenv("HEARTBEAT_MINUTES", "5"))
        return max(0, m) * 60
    except ValueError:
        return 300


def _start_heartbeat(index: int) -> threading.Event:
    """Start a daemon thread that prints a heartbeat to stderr every N minutes. Returns stop event."""
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

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return stop


def _index_to_params(index: int) -> tuple[float, float, float, float, int]:
    """
    Map index 0..8099 to (Vs1, CoV, rh, aHV, seed).
    Order: Vs1 -> CoV -> rh -> aHV -> seed.
    """
    if index < 0 or index >= TOTAL_COMBINATIONS:
        raise IndexError(
            f"Index {index} is out of range for {TOTAL_COMBINATIONS} tasks "
            f"(valid 0..{TOTAL_COMBINATIONS - 1})."
        )
    r = index
    Vs1_idx = r // (len(COV_LIST) * len(RH_LIST) * len(AHV_LIST) * len(SEED_VALUES))
    r = r % (len(COV_LIST) * len(RH_LIST) * len(AHV_LIST) * len(SEED_VALUES))
    CoV_idx = r // (len(RH_LIST) * len(AHV_LIST) * len(SEED_VALUES))
    r = r % (len(RH_LIST) * len(AHV_LIST) * len(SEED_VALUES))
    rh_idx = r // (len(AHV_LIST) * len(SEED_VALUES))
    r = r % (len(AHV_LIST) * len(SEED_VALUES))
    aHV_idx = r // len(SEED_VALUES)
    seed_idx = r % len(SEED_VALUES)
    return (
        VS1_LIST[Vs1_idx],
        COV_LIST[CoV_idx],
        RH_LIST[rh_idx],
        AHV_LIST[aHV_idx],
        SEED_VALUES[seed_idx],
    )


def _param_dir_name(index: int) -> str:
    """Directory name for this index's parameters (Vs1_CV_..._s seed)."""
    Vs1, CoV, rh, aHV, seed = _index_to_params(index)
    return f"Vs1_{Vs1:.0f}_CV_{CoV:.2f}_rH_{rh:.0f}_aHV_{aHV:.0f}_s{seed}"


def _output_dir_for_index(index: int) -> Path:
    """Output directory for this index. If EMULATOR_8100_OUTDIR is set (scratch), use run_<index>/param_dir."""
    base = os.getenv("EMULATOR_8100_OUTDIR")
    param_dir = _param_dir_name(index)
    if base:
        return Path(base) / f"run_{index}" / param_dir
    return Path("results") / param_dir


def _run_dir_for_index(index: int) -> Path | None:
    """Run directory to archive (parent of param dir when on scratch). None if not using scratch."""
    base = os.getenv("EMULATOR_8100_OUTDIR")
    if not base:
        return None
    return Path(base) / f"run_{index}"


def _h5_path_for_index(index: int) -> Path:
    """H5 file path for this index. Uses EMULATOR_8100_H5_DIR if set (e.g. scratch on Savio)."""
    base = os.getenv("EMULATOR_8100_H5_DIR")
    if base:
        return Path(base) / f"run_{index}.h5"
    return Path("results") / "h5" / f"run_{index}.h5"


def _archive_run_dir(run_dir: Path, archives_dir: Path, index: int) -> bool:
    """Tar+compress run_dir to archives_dir/run_<index>.tar.zst (or .tgz), verify, then rm -rf run_dir. Returns True on success."""
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
    except subprocess.CalledProcessError as e:
        print(f"[archive] WARNING: compress failed for run_{index}: {e}", file=sys.stderr)
        return False
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)
    return True


def _load_recorder_txt(
    recorder_dir: Path, quantity: str = "accel"
) -> tuple[np.ndarray, np.ndarray]:
    """Load time and data from OpenSees recorder .txt files in recorder_dir. Returns (time_1d, data_2d) with data shape (n_time, n_channels)."""
    center_glob = list(recorder_dir.glob(f"center_node_y*_dof1_{quantity}.txt"))
    row_glob = list(recorder_dir.glob(f"row_y*_dof1_{quantity}.txt"))
    center_glob.sort(key=lambda p: p.name)
    row_glob.sort(key=lambda p: p.name)
    chunks = []
    time_arr = None
    for f in center_glob + row_glob:
        arr = np.loadtxt(f)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if time_arr is None:
            time_arr = arr[:, 0]
        # First column is time, rest are channel(s)
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
    """Lossy compression for recorder data. Zfp accuracy mode. Fallback to lossless."""
    try:
        import hdf5plugin  # type: ignore[import-untyped]

        return hdf5plugin.Zfp(accuracy=tolerance)  # type: ignore[attr-defined]
    except ImportError:
        return _get_compression_recorder_lossless()


def _write_h5(
    h5_path: Path,
    index: int,
    Vs1: float,
    CoV: float,
    rh: float,
    aHV: float,
    seed: int,
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
    """Write one HDF5 per index: params, grid, Vs, damping, recorders.
    Uses Blosc2/zstd (or gzip-9 fallback) and float32. Optional lossy Zfp on recorder data.
    Env: EMULATOR_8100_H5_LOSSY=1, EMULATOR_8100_H5_LOSSY_TOLERANCE=1e-5.
    """
    try:
        import h5py  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError("h5py is required for HDF5 output; install with pip install h5py") from e
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    lossy = os.getenv("EMULATOR_8100_H5_LOSSY", "0") == "1"
    lossy_tol = float(os.getenv("EMULATOR_8100_H5_LOSSY_TOLERANCE", "1e-5"))
    comp_grid = _get_compression_grid()
    comp_ts = (
        _get_compression_recorder_lossy(lossy_tol)
        if lossy
        else _get_compression_recorder_lossless()
    )

    def _kw(c):
        return c if isinstance(c, dict) else {"compression": c}

    # Load recorders and apply time downsampling before writing
    time_arr, data = _load_recorder_txt(recorder_dir, quantity="accel")
    downsample = max(1, int(os.getenv("EMULATOR_8100_H5_DOWNSAMPLE", "1")))
    dt_stored = dt
    if time_arr is not None and len(time_arr) > 0 and downsample > 1:
        time_arr = time_arr[::downsample].copy()
        data = data[::downsample, :].copy()
        dt_stored = dt * downsample

    with h5py.File(h5_path, "w") as f:
        f.attrs["index"] = index
        f.attrs["task_id"] = task_id
        comp_name = "gzip" if isinstance(comp_grid, dict) else "blosc2_zstd"
        f.attrs["h5_compression"] = f"{comp_name}_lossy_recorder" if lossy else comp_name
        # params
        grp = f.create_group("params")
        grp.attrs["Vs1"] = Vs1
        grp.attrs["CoV"] = CoV
        grp.attrs["rH"] = rh
        grp.attrs["aHV"] = aHV
        grp.attrs["seed"] = seed
        # grid / fields (float32 halves size vs float64; ample precision for emulator)
        f.create_dataset(
            "Vs_profile_1D",
            data=Vs_profile_1D.astype(np.float32),
            **_kw(comp_grid),
        )
        f.create_dataset(
            "Vs_realization_2D",
            data=Vs_extended.astype(np.float32),
            **_kw(comp_grid),
        )
        f.create_dataset(
            "Damping_zeta",
            data=zeta_grid.astype(np.float32),
            **_kw(comp_grid),
        )
        # grid metadata (dt_stored = recorder timestep after downsampling)
        grp = f.create_group("grid")
        grp.attrs["Lx"] = Lx
        grp.attrs["Lz"] = Lz
        grp.attrs["dx"] = dx
        grp.attrs["dz"] = dz
        grp.attrs["dt"] = dt_stored
        # recorders
        if time_arr is not None and len(time_arr) > 0:
            accel = f.create_group("recorders").create_group("accel")
            n_time, n_ch = data.shape[0], max(1, data.shape[1])
            chunks_ts = (min(1000, n_time), min(64, n_ch))
            accel.create_dataset(
                "time",
                data=time_arr.astype(np.float32),
                **_kw(comp_ts),
            )
            ds_data = accel.create_dataset(
                "data",
                data=data.astype(np.float32),
                chunks=chunks_ts,
                **_kw(comp_ts),
            )
            ds_data.attrs["layout"] = "n_time x n_channels"
            if lossy:
                ds_data.attrs["compression_tolerance"] = lossy_tol


def run_case(index: int = 0) -> str:
    """
    Run a single case given the index (0--8099).

    Returns:
        Result status string.
    """
    t0 = time.time()
    heartbeat_stop = _start_heartbeat(index)
    try:
        return _run_case_impl(index, t0)
    finally:
        heartbeat_stop.set()


def _run_case_impl(index: int, t0: float) -> str:
    """Actual run_case logic (heartbeat already started by run_case)."""
    Vs1, CoV, rh, aHV, seed = _index_to_params(index)

    # Fixed per 8100 plan
    thickness = 15.0
    Vs2 = 1500.0
    dz_1D = 1.0
    dx, dz = 1.0, 1.0
    Lx_variability = 500.0
    BC_width = 500.0
    Lx = Lx_variability + 2 * BC_width
    interlayer_amplitude = 0.0  # No interlayer variability (flat surface)
    interlayer_seed = 14
    motion_freq = 3.0
    damping_method = "global_avg"
    case_type = "1x1_4node"
    element_type = "4node"

    layer_1_count = int(thickness / dz_1D)
    layer_2_count = 10
    Vs_profile_1D = np.array([Vs1] * layer_1_count + [Vs2] * layer_2_count)
    Lz = layer_1_count * dz_1D + layer_2_count * dz_1D

    f0 = Vs1 / (4 * thickness)
    duration = 50.0 if f0 < 1.0 else 30.0
    damping_freq_first = min(f0, motion_freq)
    damping_freqs = (damping_freq_first, 10.0)

    output_dir = _output_dir_for_index(index)
    output_dir = str(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(f"{output_dir}/Vs_profile_1D.txt", Vs_profile_1D)

    task_id = f"{case_type}_Vs1{Vs1:.0f}_CV{CoV:.2f}_rH{rh:.0f}_aHV{aHV:.0f}_s{seed}"

    print(f"[{case_type}] Starting task {task_id} (index={index})")
    print(f"  Vs1={Vs1}, CoV={CoV}, rH={rh}, aHV={aHV}, seed={seed}")
    print(f"  Lx={Lx} m, Lz={Lz} m, dx={dx} m, dz={dz} m")
    print(f"  f0={f0:.4f} Hz, duration={duration:.1f}s, damping_freqs={damping_freqs}")

    t_field_start = time.time()
    np.random.seed(seed)
    Vs_realization, _, _, _, bedrock_mask_var = _generate_vs_variability_field(
        Vs_profile_1D,
        Lx_variability,
        Lz,
        dx,
        dz,
        rh,
        aHV,
        CoV,
        seed=seed,
        dz_1D=dz_1D,
        interlayer_seed=interlayer_seed,
        interlayer_amplitude=interlayer_amplitude,
    )
    field_generation_time = time.time() - t_field_start

    Vs_extended, _ = _extend_profile(Vs_realization, Lx=Lx, dx=dx)
    bedrock_mask_extended, _ = _extend_profile(bedrock_mask_var.astype(float), Lx=Lx, dx=dx)
    bedrock_mask_extended = bedrock_mask_extended.astype(bool)

    config = AnalysisConfig(
        Ly=Lz,
        Lx=Lx,
        hx=dx,
        dt=0.01,
        duration=duration,
        motion_freq=motion_freq,
        motion_t_shift=0.5,
        damping_freqs=damping_freqs,
        damping_zeta=0.025,
        damping_method=damping_method,
        boundary_condition_type="2D",
        record_center_nodes=False,  # center already in lateral span
        center_node_y_positions=[2.0, Lz],
        record_lateral_span_at_center_depths=(
            50,
            2.0,
        ),  # 101 nodes per row (center ±100 m at 2 m spacing)
        record_all_surface_nodes=False,
        element_type=element_type,
        solver_type="Mumps",
    )

    zeta_grid = get_damping_zeta_grid(
        Vs_extended,
        damping_method,
        Lx,
        Lz,
        dx,
        dz,
        bedrock_mask=bedrock_mask_extended,
        config=config,
    )

    solver_info = get_solver_info(config)
    print(f"  Solver: {solver_info['solver_type']}")
    print(f"  Domain: {Lx}m x {Lz}m")
    print(f"  Recording center + lateral span at Y={config.center_node_y_positions}")

    rho = np.ones_like(Vs_extended) * 2000.0
    nu = np.ones_like(Vs_extended) * 0.3

    t_model_start = time.time()
    model_data = build_model_data(config, Vs_extended, rho, nu)
    model_build_time = time.time() - t_model_start

    t_analysis_start = time.time()
    print(f"[{case_type}] Running OpenSees for {task_id} -> {output_dir}")
    result = run_opensees_analysis(config, model_data, task_id, output_dir)
    analysis_time = time.time() - t_analysis_start

    # HDF5 write (default): one file per index for fast reads and portability
    h5_path = _h5_path_for_index(index)
    recorder_dir = Path(output_dir) / task_id
    if recorder_dir.exists():
        _write_h5(
            h5_path,
            index=index,
            Vs1=Vs1,
            CoV=CoV,
            rh=rh,
            aHV=aHV,
            seed=seed,
            Vs_profile_1D=Vs_profile_1D,
            Vs_extended=Vs_extended,
            zeta_grid=zeta_grid,
            recorder_dir=recorder_dir,
            Lx=Lx,
            Lz=Lz,
            dx=dx,
            dz=dz,
            dt=config.dt,
            task_id=task_id,
        )
        print(f"[{case_type}] Wrote {h5_path}")

    # Scratch: aggregate+compress run dir, then remove raw files
    run_dir = _run_dir_for_index(index)
    if run_dir is not None and run_dir.exists():
        outdir_base = Path(os.getenv("EMULATOR_8100_OUTDIR", ""))
        archives_dir = outdir_base / "archives"
        if _archive_run_dir(run_dir, archives_dir, index):
            print(f"[{case_type}] Archived run_{index} to {archives_dir}")

    total_time = time.time() - t0

    # Per-index timing: append to timing.db (same DB as phase1 array-task summary).
    _write_timing_per_index(
        index=index,
        Vs1=Vs1,
        CoV=CoV,
        rh=rh,
        aHV=aHV,
        seed=seed,
        task_id=task_id,
        total_time_sec=total_time,
        field_sec=field_generation_time,
        model_sec=model_build_time,
        analysis_sec=analysis_time,
        status=result,
    )

    print(f"[{case_type}] Done: {result} | Wall time: {_fmt_hms(total_time)}")
    return result


def _write_timing_per_index(
    *,
    index: int,
    Vs1: float,
    CoV: float,
    rh: float,
    aHV: float,
    seed: int,
    task_id: str,
    total_time_sec: float,
    field_sec: float,
    model_sec: float,
    analysis_sec: float,
    status: str,
) -> None:
    """Append one row to timing.db table timing_per_index. No-op on error (e.g. DB locked)."""
    db_path = os.getenv("EMULATOR_8100_TIMING_DB") or str(Path("timing.db").resolve())
    jobid = int(os.getenv("SLURM_JOB_ID", "0"))
    taskid = int(os.getenv("SLURM_ARRAY_TASK_ID", "0"))
    try:
        conn = sqlite3.connect(db_path, timeout=10.0)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS timing_per_index (
                ts TEXT, idx INT, jobid INT, taskid INT,
                Vs1 REAL, CoV REAL, rH REAL, aHV REAL, seed INT, task_id TEXT,
                total_time_sec REAL, field_sec REAL, model_sec REAL, analysis_sec REAL, status TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timing_per_index_idx ON timing_per_index(idx)")
        conn.execute(
            """
            INSERT INTO timing_per_index(ts, idx, jobid, taskid, Vs1, CoV, rH, aHV, seed, task_id,
                total_time_sec, field_sec, model_sec, analysis_sec, status)
            VALUES(datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                index,
                jobid,
                taskid,
                Vs1,
                CoV,
                rh,
                aHV,
                seed,
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


def _parse_args():
    p = argparse.ArgumentParser(
        description="Run emulator_8100 (8,100 sims). Single --index or range --index-start/--index-end."
    )
    p.add_argument("--index", type=int, default=None, help="Single task index 0-8099")
    p.add_argument("--index-start", type=int, default=None, help="Start of range (inclusive)")
    p.add_argument(
        "--index-end",
        type=int,
        default=None,
        help="End of range (exclusive), clamped to 8100",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if output already exists (overrides idempotent skip).",
    )
    return p.parse_args()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    program_start = time.time()
    _configure_slurm_environment()
    _install_sigterm_handler()
    args = _parse_args()

    if args.index is not None:
        # Single run
        idx = args.index
        if idx < 0 or idx >= TOTAL_COMBINATIONS:
            print(f"Error: --index must be 0..{TOTAL_COMBINATIONS - 1}", file=sys.stderr)
            sys.exit(1)
        # Idempotent: skip if H5 exists, or archive (scratch), or legacy center_node_*.txt
        if not args.force:
            h5_path = _h5_path_for_index(idx)
            if h5_path.exists():
                print(f"[idempotent] Index {idx} already has H5; skipping.")
                sys.exit(0)
            run_dir = _run_dir_for_index(idx)
            if run_dir is not None:
                outdir_base = Path(os.getenv("EMULATOR_8100_OUTDIR", ""))
                arc = outdir_base / "archives" / f"run_{idx}.tar.zst"
                if not arc.exists():
                    arc = outdir_base / "archives" / f"run_{idx}.tgz"
                if arc.exists():
                    print(f"[idempotent] Index {idx} already has archive; skipping.")
                    sys.exit(0)
            out_dir = _output_dir_for_index(idx)
            if out_dir.exists():
                done_marker = list(out_dir.glob("**/center_node_*.txt"))
                if done_marker:
                    print(f"[idempotent] Index {idx} already has output; skipping.")
                    sys.exit(0)
        run_case(idx)
    elif args.index_start is not None and args.index_end is not None:
        start = max(0, args.index_start)
        end = min(TOTAL_COMBINATIONS, args.index_end)
        for i in range(start, end):
            if not args.force:
                if _h5_path_for_index(i).exists():
                    print(f"[idempotent] Index {i} already has H5; skipping.")
                    continue
                run_dir = _run_dir_for_index(i)
                if run_dir is not None:
                    outdir_base = Path(os.getenv("EMULATOR_8100_OUTDIR", ""))
                    if (outdir_base / "archives" / f"run_{i}.tar.zst").exists() or (
                        outdir_base / "archives" / f"run_{i}.tgz"
                    ).exists():
                        print(f"[idempotent] Index {i} already has archive; skipping.")
                        continue
                out_dir = _output_dir_for_index(i)
                if out_dir.exists() and list(out_dir.glob("**/center_node_*.txt")):
                    print(f"[idempotent] Index {i} already has output; skipping.")
                    continue
            run_case(i)
    else:
        # Fallback: from environment (SLURM_ARRAY_TASK_ID or PARALLEL_SEQ)
        idx = args.index
        if idx is None:
            env_idx = os.getenv("SLURM_ARRAY_TASK_ID") or os.getenv("PARALLEL_SEQ")
            if env_idx not in (None, ""):
                try:
                    idx = int(env_idx)
                    if os.getenv("PARALLEL_SEQ"):
                        idx -= 1  # gnu-parallel slot is 1-based
                except ValueError:
                    idx = None
        if idx is None or idx < 0 or idx >= TOTAL_COMBINATIONS:
            print(
                f"Error: provide --index 0-{TOTAL_COMBINATIONS - 1} or --index-start/--index-end, or set SLURM_ARRAY_TASK_ID/PARALLEL_SEQ",
                file=sys.stderr,
            )
            sys.exit(1)
        if not args.force:
            if _h5_path_for_index(idx).exists():
                print(f"[idempotent] Index {idx} already has H5; skipping.")
                sys.exit(0)
            run_dir = _run_dir_for_index(idx)
            if run_dir is not None:
                outdir_base = Path(os.getenv("EMULATOR_8100_OUTDIR", ""))
                if (outdir_base / "archives" / f"run_{idx}.tar.zst").exists() or (
                    outdir_base / "archives" / f"run_{idx}.tgz"
                ).exists():
                    print(f"[idempotent] Index {idx} already has archive; skipping.")
                    sys.exit(0)
            out_dir = _output_dir_for_index(idx)
            if out_dir.exists() and list(out_dir.glob("**/center_node_*.txt")):
                print(f"[idempotent] Index {idx} already has output; skipping.")
                sys.exit(0)
        run_case(idx)

    print(f"\n[program] Total wall time: {_fmt_hms(time.time() - program_start)}")
    sys.exit(0)
