"""Runner for the 3-Vs-layer generalization experiment.

See ``neural-operator/experiments/README.md``. Builds each manifest case (4D
topology Sobol draw + RF seeds), runs the OpenSees analysis, and writes one HDF5
per case. Supports local runs and Stampede3 pylauncher batches via
``EXP_H5_DIR`` / ``EXP_OUTDIR`` environment overrides.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from seiskit.analysis import run_opensees_analysis
from seiskit.builder import build_model_data
from seiskit.config import AnalysisConfig
from seiskit.gaussian_field import create_three_layer_vs_realization
from seiskit.plot_results import get_damping_zeta_grid
from seiskit.solver_utils import get_solver_info

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent.parent / "data"
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from manifest import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    ThreeLayerManifestEntry,
    ensure_manifest,
    load_manifest_csv,
)


def _load_module(name: str, path: Path):
    """Load a module from an explicit file path under a unique name."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_data_run_experiment = _load_module("_data_run_experiment", DATA_DIR / "run_experiment.py")
_fmt_hms = _data_run_experiment._fmt_hms
_get_compression_grid = _data_run_experiment._get_compression_grid
_get_compression_recorder_lossless = _data_run_experiment._get_compression_recorder_lossless
_load_recorder_txt = _data_run_experiment._load_recorder_txt

CASE_TYPE = "three_layer"
ELEMENT_TYPE = "4node"
DX = 1.0
DZ = 1.0
LX_VARIABILITY = 500.0
BC_WIDTH = 500.0
LX = LX_VARIABILITY + 2 * BC_WIDTH
DAMPING_FREQ_SECOND = 10.0
DAMPING_METHOD = "global_avg"
MAX_TIME_PER_BATCH_SEC = 8 * 3600.0

RESULTS_DIR = Path(os.environ.get("EXP_OUTDIR", THIS_DIR / "results"))
H5_DIR = Path(os.environ.get("EXP_H5_DIR", THIS_DIR / "h5"))


def _write_h5(
    h5_path: Path,
    entry: ThreeLayerManifestEntry,
    manifest_path: Path,
    Vs_extended: np.ndarray,
    zeta_grid: np.ndarray,
    recorder_dir: Path,
    Lz: float,
    dt: float,
    task_id: str,
) -> None:
    import h5py

    h5_path.parent.mkdir(parents=True, exist_ok=True)
    comp_grid = _get_compression_grid()
    comp_ts = _get_compression_recorder_lossless()

    def _kw(compression):
        return compression if isinstance(compression, dict) else {"compression": compression}

    time_arr, data = _load_recorder_txt(recorder_dir, quantity="accel")

    with h5py.File(h5_path, "w") as handle:
        handle.attrs["index"] = entry.index
        handle.attrs["task_id"] = task_id
        handle.attrs["case_type"] = CASE_TYPE
        handle.attrs["manifest_path"] = str(manifest_path.resolve())
        handle.attrs["split"] = entry.split

        params = handle.create_group("params")
        params.attrs["Vs1"] = entry.Vs1
        params.attrs["H1"] = entry.H1_discretized
        params.attrs["CoV1"] = entry.CoV1
        params.attrs["rH1"] = entry.rH1
        params.attrs["aHV1"] = entry.aHV1
        params.attrs["Vs_mid"] = entry.Vs_mid
        params.attrs["H2"] = entry.H2_discretized
        params.attrs["CoV2"] = entry.CoV2
        params.attrs["rH2"] = entry.rH2
        params.attrs["aHV2"] = entry.aHV2
        params.attrs["Vs_contrast"] = entry.Vs_contrast
        params.attrs["Vs_bedrock"] = entry.Vs_bedrock
        params.attrs["topology_id"] = entry.topology_id
        params.attrs["replicate_id"] = entry.replicate_id
        params.attrs["seed1"] = entry.seed1
        params.attrs["seed2"] = entry.seed2
        params.attrs["f0_effective"] = entry.f0_effective
        params.attrs["duration"] = entry.duration

        handle.create_dataset(
            "Vs_realization_2D", data=Vs_extended.astype(np.float32), **_kw(comp_grid)
        )
        handle.create_dataset("Damping_zeta", data=zeta_grid.astype(np.float32), **_kw(comp_grid))

        grid = handle.create_group("grid")
        grid.attrs["Lx"] = LX
        grid.attrs["Lz"] = Lz
        grid.attrs["dx"] = DX
        grid.attrs["dz"] = DZ
        grid.attrs["dt"] = dt
        grid.attrs["bedrock_thickness"] = entry.bedrock_thickness_discretized

        if len(time_arr) > 0:
            accel = handle.create_group("recorders").create_group("accel")
            n_time, n_ch = data.shape[0], max(1, data.shape[1])
            chunks_ts = (min(1000, n_time), min(64, n_ch))
            accel.create_dataset("time", data=time_arr.astype(np.float32), **_kw(comp_ts))
            accel.create_dataset(
                "data", data=data.astype(np.float32), chunks=chunks_ts, **_kw(comp_ts)
            )


def run_case(entry: ThreeLayerManifestEntry, manifest_path: Path) -> str:
    t0 = time.time()
    task_id = f"{CASE_TYPE}_case{entry.index}"
    output_dir = RESULTS_DIR / f"case_{entry.index}"
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{CASE_TYPE}] Starting {task_id} split={entry.split}")
    print(
        f"  Vs1={entry.Vs1:.1f} H1={entry.H1_discretized:.1f} CoV1={entry.CoV1:.3f} "
        f"rH1={entry.rH1:.1f} aHV1={entry.aHV1:.1f}"
    )
    print(
        f"  Vs_mid={entry.Vs_mid:.1f} H2={entry.H2_discretized:.1f} CoV2={entry.CoV2:.3f} "
        f"rH2={entry.rH2:.1f} aHV2={entry.aHV2:.1f}"
    )
    print(
        f"  Vs_bedrock={entry.Vs_bedrock:.1f} contrast={entry.Vs_contrast:.3f} "
        f"Lz={entry.Lz_discretized:.1f} f0={entry.f0_effective:.4f} duration={entry.duration:.1f}"
    )

    Vs_extended, _x_total, _z, (_h1, _h2), bedrock_mask = create_three_layer_vs_realization(
        Vs1=entry.Vs1,
        Vs_mid=entry.Vs_mid,
        Vs_bedrock=entry.Vs_bedrock,
        H1=entry.H1_discretized,
        H2=entry.H2_discretized,
        Lx=LX,
        Lx_variability=LX_VARIABILITY,
        Lz=entry.Lz_discretized,
        dx=DX,
        dz=DZ,
        rH1=entry.rH1,
        aHV1=entry.aHV1,
        CV1=entry.CoV1,
        rH2=entry.rH2,
        aHV2=entry.aHV2,
        CV2=entry.CoV2,
        seed1=entry.seed1,
        seed2=entry.seed2,
    )

    damping_freqs = (entry.damping_freq_first, DAMPING_FREQ_SECOND)
    config = AnalysisConfig(
        Ly=entry.Lz_discretized,
        Lx=LX,
        hx=DX,
        dt=0.01,
        duration=entry.duration,
        motion_freq=entry.motion_freq,
        motion_t_shift=0.5,
        damping_freqs=damping_freqs,
        damping_zeta=0.025,
        damping_method=DAMPING_METHOD,
        boundary_condition_type="2D",
        record_center_nodes=False,
        center_node_y_positions=[2.0, entry.Lz_discretized],
        record_lateral_span_at_center_depths=(10, 15.0),
        record_all_surface_nodes=False,
        element_type=ELEMENT_TYPE,
        solver_type="Mumps",
        max_time_per_batch=MAX_TIME_PER_BATCH_SEC,
    )

    zeta_grid = get_damping_zeta_grid(
        Vs_extended,
        DAMPING_METHOD,
        LX,
        entry.Lz_discretized,
        DX,
        DZ,
        bedrock_mask=bedrock_mask,
        config=config,
    )

    solver_info = get_solver_info(config)
    print(f"  Solver: {solver_info['solver_type']}, Domain: {LX}m x {entry.Lz_discretized}m")

    rho = np.ones_like(Vs_extended) * 2000.0
    nu = np.ones_like(Vs_extended) * 0.3

    model_data = build_model_data(config, Vs_extended, rho, nu, bedrock_mask=bedrock_mask)

    print(f"[{CASE_TYPE}] Running OpenSees for {task_id} -> {output_dir}")
    result = run_opensees_analysis(config, model_data, task_id, str(output_dir))

    h5_path = H5_DIR / f"case_{entry.index}.h5"
    recorder_dir = output_dir / task_id
    if recorder_dir.exists():
        _write_h5(
            h5_path,
            entry=entry,
            manifest_path=manifest_path,
            Vs_extended=Vs_extended,
            zeta_grid=zeta_grid,
            recorder_dir=recorder_dir,
            Lz=entry.Lz_discretized,
            dt=config.dt,
            task_id=task_id,
        )
        print(f"[{CASE_TYPE}] Wrote {h5_path}")

    print(f"[{CASE_TYPE}] Done: {result} | Wall time: {_fmt_hms(time.time() - t0)}")
    return result


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the 3-Vs-layer generalization experiment.")
    parser.add_argument("--index", type=int, default=None, help="Run only this case index.")
    parser.add_argument("--manifest-path", type=Path, default=None, help="Path to manifest CSV.")
    parser.add_argument(
        "--force", action="store_true", help="Re-run even if an H5 output already exists."
    )
    parser.add_argument(
        "--overwrite-manifest",
        action="store_true",
        help="Regenerate manifest.csv before running.",
    )
    parser.add_argument(
        "--topology-count",
        type=int,
        default=None,
        help="Override manifest Sobol topology count.",
    )
    parser.add_argument(
        "--rf-seeds-per-topology",
        type=int,
        default=None,
        help="Override manifest RF seeds per topology point.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    manifest_path = Path(args.manifest_path or DEFAULT_MANIFEST_PATH)

    if args.overwrite_manifest or not manifest_path.exists():
        from manifest import DEFAULT_RF_SEEDS_PER_TOPOLOGY, DEFAULT_TOPOLOGY_COUNT

        manifest_entries = ensure_manifest(
            path=manifest_path,
            topology_count=args.topology_count or DEFAULT_TOPOLOGY_COUNT,
            rf_seeds_per_topology=args.rf_seeds_per_topology or DEFAULT_RF_SEEDS_PER_TOPOLOGY,
            overwrite=True,
        )
    else:
        manifest_entries = load_manifest_csv(manifest_path)

    manifest_path = manifest_path.resolve()
    print(f"[manifest] Using {manifest_path} ({len(manifest_entries)} cases)")

    indices = [args.index] if args.index is not None else [e.index for e in manifest_entries]
    for idx in indices:
        entry = manifest_entries[idx]
        h5_path = H5_DIR / f"case_{entry.index}.h5"
        if not args.force and h5_path.exists():
            print(f"[idempotent] case_{idx} already has H5; skipping.")
            continue
        run_case(entry, manifest_path)
