"""
Emulator 8100: 8,100 simulations for surrogate training.

Fixed:
- Height = 15 m, Vs2 = 1500 m/s, no interlayer variability (flat surface).
- Lx = 1500 m with BC_width = 500 m, dx = 1 m, dz = 1 m, Lz = 25 m (15 soil + 10 bedrock).
- Motion 3.0 Hz, damping_method = "global_avg", f0-based duration and damping_freqs.

Sweep:
- Vs1 in {100, 230, 360}, CoV in {0.1, 0.2, 0.3}, rh in {10, 30, 50}, aHV in {1, 10, 50}.
- Random seeds 1--100.

Recordings: center node + lateral span at center depths (21 nodes at bedrock and surface, 10 m spacing).

Total: 3*3*3*3*100 = 8,100. Index 0--8099.
Supports: --index N (single), or --index-start A --index-end B (range, clamp to 8100).
Idempotent: skips if output already exists for that index.
"""

import argparse
import csv
import os
import signal
import sys
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
from seiskit.plot_results import plot_damping_realization, plot_realization
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
    print(
        f"[slurm] job_id={job_id} task_id={task_id} parallel_seq={parallel_seq} node={node}"
    )


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
    return f"{total_seconds // 3600:02d}:{(total_seconds % 3600) // 60:02d}:{total_seconds % 60:02d}"


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


def _output_dir_for_index(index: int) -> Path:
    """Unique output directory for this index (for idempotent skip check)."""
    Vs1, CoV, rh, aHV, seed = _index_to_params(index)
    return Path(f"results/Vs1_{Vs1:.0f}_CV_{CoV:.2f}_rH_{rh:.0f}_aHV_{aHV:.0f}_s{seed}")


def run_case(index: int = 0) -> str:
    """
    Run a single case given the index (0--8099).

    Returns:
        Result status string.
    """
    t0 = time.time()
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
    bedrock_mask_extended, _ = _extend_profile(
        bedrock_mask_var.astype(float), Lx=Lx, dx=dx
    )
    bedrock_mask_extended = bedrock_mask_extended.astype(bool)

    plot_realization(
        Vs_profile_1D,
        Vs_extended,
        Lx,
        Lz,
        dx,
        dz,
        save_path=f"{output_dir}/Vs_realization.png",
        title=f"Vs Realization (Vs1={Vs1:.0f}, CV={CoV:.2f}, rH={rh:.0f}, aHV={aHV:.0f}, s{seed})",
        bedrock_mask=bedrock_mask_extended,
    )

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
        record_center_nodes=True,
        center_node_y_positions=[2.0, Lz],
        record_lateral_span_at_center_depths=(
            100,
            1.0,
        ),  # subset: 201 nodes (center ±100 at 1 m spacing)
        record_all_surface_nodes=False,
        element_type=element_type,
        solver_type="Mumps",
    )

    plot_damping_realization(
        Vs_extended,
        damping_method,
        Lx,
        Lz,
        dx,
        dz,
        save_path=f"{output_dir}/Damping_realization.png",
        title=f"Damping (Vs1={Vs1:.0f}, CV={CoV:.2f}, s{seed})",
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
    total_time = time.time() - t0

    timing_file = Path("results") / f"timing_data_task_{index}.csv"
    timing_file.parent.mkdir(parents=True, exist_ok=True)
    with open(timing_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "index",
                "Vs1",
                "CoV",
                "rH",
                "aHV",
                "seed",
                "task_id",
                "total_time_sec",
                "field_sec",
                "model_sec",
                "analysis_sec",
                "status",
            ]
        )
        w.writerow(
            [
                index,
                Vs1,
                CoV,
                rh,
                aHV,
                seed,
                task_id,
                f"{total_time:.3f}",
                f"{field_generation_time:.3f}",
                f"{model_build_time:.3f}",
                f"{analysis_time:.3f}",
                result,
            ]
        )

    print(f"[{case_type}] Done: {result} | Wall time: {_fmt_hms(total_time)}")
    return result


def _parse_args():
    p = argparse.ArgumentParser(
        description="Run emulator_8100 (8,100 sims). Single --index or range --index-start/--index-end."
    )
    p.add_argument("--index", type=int, default=None, help="Single task index 0-8099")
    p.add_argument(
        "--index-start", type=int, default=None, help="Start of range (inclusive)"
    )
    p.add_argument(
        "--index-end",
        type=int,
        default=None,
        help="End of range (exclusive), clamped to 8100",
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
            print(
                f"Error: --index must be 0..{TOTAL_COMBINATIONS - 1}", file=sys.stderr
            )
            sys.exit(1)
        # Idempotent: skip if output dir exists and has OpenSees output (output_dir/task_id/center_node_*.txt)
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
            out_dir = _output_dir_for_index(i)
            if out_dir.exists():
                existing = list(out_dir.glob("**/center_node_*.txt"))
                if existing:
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
        out_dir = _output_dir_for_index(idx)
        if out_dir.exists():
            existing = list(out_dir.glob("**/center_node_*.txt"))
            if existing:
                print(f"[idempotent] Index {idx} already has output; skipping.")
                sys.exit(0)
        run_case(idx)

    print(f"\n[program] Total wall time: {_fmt_hms(time.time() - program_start)}")
    sys.exit(0)
