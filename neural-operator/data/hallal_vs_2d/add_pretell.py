"""Pretell 1D column ensemble with Taborda–Bielak Q–Vs damping vs NO 2D center TF.

For each of the 7680 two-layer OpenSees runs:

1. Load ``Vs_realization_2D`` from the H5
2. Extract evenly spaced 1D columns across the 500 m variability strip
   (Pretell-style; same indexing as ``comparison/Response_Variability``)
3. Thomson–Haskell ``AF_within`` with **elemental** Taborda–Bielak
   :math:`\\xi_Q = 1/(2Q(V_s))` on merged soil layers + rock halfspace
4. Geomean, empirical p16/p84, and :math:`\\sigma_{\\ln}` over columns
5. Pearson :math:`r(\\ln|TF|_{\\mathrm{Pretell}},\\ln|TF|_{2D,\\mathrm{center}})`

Checkpoints every ``PRETELL_BATCH`` (default 128) runs to a **local** H5 so a
reboot can resume. Syncs to Box ``hallal_vs_2d/pretell_ensembles.h5`` after
each checkpoint and at the end.

Usage
-----
  HALLAL_N_JOBS=16 PRETELL_N_SAMPLES=200 python add_pretell.py
  PRETELL_N_SAMPLES=20 python add_pretell.py --smoke   # 2 samples × seeds
  python add_pretell.py          # resume if local checkpoint exists
  python add_pretell.py --force  # wipe and restart
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import hdf5plugin  # noqa: F401
import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
NO_DATA = THIS_DIR.parent
REPO = NO_DATA.parents[1]
sys.path.insert(0, str(NO_DATA))
sys.path.insert(0, str(REPO))

from sobol import (  # noqa: E402
    DEFAULT_RF_SEEDS_PER_SAMPLE,
    DEFAULT_SOBOL_SAMPLE_COUNT,
    build_manifest,
)
from seiskit.damping import (  # noqa: E402
    compute_damping_from_Q,
    compute_quality_factor,
)
from seiskit.theory.layered_1d_tf import Layer, RockHalfspace, layered_transfer_function  # noqa: E402

BOX_ROOT = Path(
    os.getenv(
        "NO_BOX_ROOT",
        "/mnt/box/GIG Lab - UC Berkeley/Projects/Neural Operator/data",
    )
)
H5_DIR = BOX_ROOT / "h5"
OUT_DIR = BOX_ROOT / "hallal_vs_2d"
TF_DIR = BOX_ROOT / "transfer_function"
# Local SSD checkpoint (survives Box sync failures; resume after reboot)
LOCAL_CKPT_DIR = Path(
    os.getenv("PRETELL_CKPT_DIR", str(Path.home() / ".cache" / "hallal_vs_2d"))
)

RHO = 2000.0
DZ = 1.0
BC_WIDTH_M = 500.0
LX_VAR_M = 500.0
EPS = 1e-12
LAYER_REL_TOL = 0.05
PRETELL_N_DEFAULT = 200


def _n_jobs() -> int:
    raw = os.getenv("HALLAL_N_JOBS", "").strip()
    if raw:
        return int(raw)
    slurm = os.getenv("SLURM_CPUS_PER_TASK", "").strip()
    if slurm:
        return int(slurm)
    return -1


def _n_samples() -> int:
    raw = os.getenv("PRETELL_N_SAMPLES", "").strip()
    return max(1, int(raw)) if raw else PRETELL_N_DEFAULT


def _configure_blas() -> None:
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")


def pretell_column_indices(n_samples: int, *, dx: float = 1.0) -> np.ndarray:
    """Evenly spaced columns on the full 1500 m grid (variability strip only)."""
    bc_cols = int(round(BC_WIDTH_M / dx))
    sample_cols = int(round(LX_VAR_M / dx))
    cols_strip = np.linspace(0, sample_cols - 1, n_samples, dtype=int)
    return bc_cols + cols_strip


def xi_of_vs(vs: float) -> float:
    return float(compute_damping_from_Q(compute_quality_factor(float(vs))))


def merge_soil_layers(
    vs_soil: np.ndarray,
    dz: float = DZ,
    rel_tol: float = LAYER_REL_TOL,
) -> list[Layer]:
    """Collapse contiguous cells with relative Vs jump ≤ *rel_tol*."""
    vs = np.asarray(vs_soil, dtype=float).ravel()
    if vs.size == 0:
        raise ValueError("empty soil column")
    thicknesses: list[float] = []
    values: list[float] = []
    t = dz
    v0 = float(vs[0])
    wsum = v0 * dz
    for v in vs[1:]:
        v = float(v)
        if abs(v - v0) / max(abs(v0), 1e-6) > rel_tol:
            thicknesses.append(t)
            values.append(wsum / t)
            t = dz
            v0 = v
            wsum = v * dz
        else:
            t += dz
            wsum += v * dz
    thicknesses.append(t)
    values.append(wsum / t)
    return [
        Layer(float(th), float(vv), RHO, xi_of_vs(vv))
        for th, vv in zip(thicknesses, values)
    ]


def column_tf(
    vs_col: np.ndarray,
    n_soil: int,
    freq: np.ndarray,
) -> np.ndarray:
    vs = np.asarray(vs_col, dtype=float).ravel()
    n_soil = int(np.clip(n_soil, 1, len(vs) - 1))
    layers = merge_soil_layers(vs[:n_soil])
    rock_vs = float(np.median(vs[n_soil:]))
    rock = RockHalfspace(rock_vs, RHO, xi_of_vs(rock_vs))
    _, aw, _ = layered_transfer_function(freq, layers, rock)
    return np.asarray(aw, dtype=np.float64)


def pretell_column_stats(
    vs_field: np.ndarray,
    n_soil: int,
    freq: np.ndarray,
    cols: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Geomean, p16, p84, and σ_ln of |TF| across Pretell strip columns."""
    stack = np.empty((len(cols), len(freq)), dtype=np.float64)
    for i, col in enumerate(cols):
        stack[i] = column_tf(vs_field[:, int(col)], n_soil, freq)
    clipped = np.clip(stack, EPS, None)
    log_tf = np.log(clipped)
    geo = np.exp(np.mean(log_tf, axis=0))
    if len(cols) > 1:
        sig = np.std(log_tf, axis=0, ddof=1)
        p16 = np.percentile(clipped, 16, axis=0)
        p84 = np.percentile(clipped, 84, axis=0)
    else:
        sig = np.zeros(len(freq))
        p16 = clipped[0].copy()
        p84 = clipped[0].copy()
    return geo, p16, p84, sig


def _process_one(
    index: int,
    n_soil: int,
    freq: np.ndarray,
    cols: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[int, None, None, None, None]:
    import hdf5plugin  # noqa: F401  — must import in each loky worker

    path = H5_DIR / f"run_{index}.h5"
    if not path.is_file():
        return index, None, None, None, None
    try:
        with h5py.File(path, "r") as f:
            vs_field = np.asarray(f["Vs_realization_2D"][:], dtype=np.float64)
            if "params" in f and "soil_layer_count" in f["params"].attrs:
                n_soil = int(f["params"].attrs["soil_layer_count"])
        geo, p16, p84, sig = pretell_column_stats(vs_field, n_soil, freq, cols)
        return (
            index,
            geo.astype(np.float32),
            p16.astype(np.float32),
            p84.astype(np.float32),
            sig.astype(np.float32),
        )
    except Exception as exc:
        print(f"[warn] index={index}: {exc}", flush=True)
        return index, None, None, None, None


def pearson_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.log(np.clip(np.asarray(x, dtype=np.float64), EPS, None))
    y = np.log(np.clip(np.asarray(y, dtype=np.float64), EPS, None))
    n = x.shape[1]
    y_c = y - y.mean()
    denom_y = np.sqrt(np.sum(y_c * y_c))
    if denom_y < 1e-15:
        return np.full(x.shape[0], np.nan)
    num = x @ y_c
    sx = x.sum(axis=1)
    sx2 = np.sum(x * x, axis=1)
    denom_x = np.sqrt(np.clip(sx2 - sx * sx / n, 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / (denom_x * denom_y)
    r[~np.isfinite(r)] = np.nan
    return r


def _open_ckpt(
    path: Path,
    *,
    n_runs: int,
    freq: np.ndarray,
    cols: np.ndarray,
    n_samp: int,
    force: bool,
) -> h5py.File:
    """Open local checkpoint H5 for r+; create if missing or *force*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n_freq = len(freq)
    if path.exists() and not force:
        f = h5py.File(path, "r+")
        n_valid = int(np.asarray(f["valid"][:]).sum())
        print(f"[resume] {path}  valid={n_valid}/{n_runs}")
        return f

    if path.exists():
        path.unlink()
    f = h5py.File(path, "w")
    f.create_dataset("freq", data=freq)
    shape = (n_runs, n_freq)
    chunks = (1, n_freq)
    for name in ("geomean", "p16", "p84", "sigma_ln"):
        f.create_dataset(
            name,
            shape=shape,
            dtype="float32",
            fillvalue=np.nan,
            compression="gzip",
            chunks=chunks,
        )
    f.create_dataset("valid", data=np.zeros(n_runs, dtype=bool))
    f.create_dataset("columns", data=cols.astype(np.int32))
    f.attrs["n_samples"] = n_samp
    f.attrs["layer_rel_tol"] = LAYER_REL_TOL
    f.attrs["damping"] = "taborda_bielak_elemental_Q_Vs"
    f.attrs["approach"] = "pretell_strip_columns"
    f.attrs["stats"] = "geomean,p16,p84,sigma_ln"
    f.flush()
    print(f"[init] {path}")
    return f


def _sync_to_box(local: Path, box: Path) -> None:
    box.parent.mkdir(parents=True, exist_ok=True)
    try:
        if local.resolve() == box.resolve():
            return  # ckpt already lives at the publish path (e.g. Savio scratch)
        shutil.copy2(local, box)
        print(f"[sync] → {box}", flush=True)
    except OSError as exc:
        print(f"[warn] Box sync failed ({exc}); local ckpt still at {local}", flush=True)


def run(*, smoke: bool = False, force: bool = False, n_jobs: int | None = None) -> None:
    _configure_blas()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_samp = _n_samples()
    jobs = _n_jobs() if n_jobs is None else n_jobs
    cols = pretell_column_indices(n_samp)
    batch = max(1, int(os.getenv("PRETELL_BATCH", "128")))
    print(
        f"[pretell] n_samples={n_samp}  cols[{cols[0]}..{cols[-1]}]  "
        f"n_jobs={jobs}  batch={batch}"
    )

    freq = np.asarray(np.load(TF_DIR / "freq.npy"), dtype=np.float64)
    with h5py.File(OUT_DIR / "tf_2d_center.h5", "r") as f:
        tf_2d = np.asarray(f["tf_center"][:], dtype=np.float64)

    manifest = build_manifest(
        sample_count=DEFAULT_SOBOL_SAMPLE_COUNT,
        seeds_per_sample=DEFAULT_RF_SEEDS_PER_SAMPLE,
    )
    if smoke:
        keep = {e.sample_id for e in manifest if e.sample_id < 2}
        entries = [e for e in manifest if e.sample_id in keep]
    else:
        entries = list(manifest)

    n_runs = len(manifest)
    name = "pretell_ensembles_smoke.h5" if smoke else "pretell_ensembles.h5"
    local_h5 = LOCAL_CKPT_DIR / name
    box_h5 = OUT_DIR / name

    ckpt = _open_ckpt(
        local_h5,
        n_runs=n_runs,
        freq=freq,
        cols=cols,
        n_samp=n_samp,
        force=force or smoke,
    )
    valid = np.asarray(ckpt["valid"][:], dtype=bool)
    pending = [e for e in entries if not valid[e.index]]
    print(f"[pretell] pending={len(pending)}/{len(entries)}")

    t0 = time.time()
    n_done = 0
    try:
        for start in range(0, len(pending), batch):
            chunk = pending[start : start + batch]
            results = Parallel(n_jobs=jobs, backend="loky", verbose=5)(
                delayed(_process_one)(e.index, e.soil_layer_count, freq, cols)
                for e in chunk
            )
            for idx, geo, p16, p84, sig in results:
                if geo is None or idx >= n_runs:
                    continue
                ckpt["geomean"][idx] = geo
                ckpt["p16"][idx] = p16
                ckpt["p84"][idx] = p84
                ckpt["sigma_ln"][idx] = sig
                ckpt["valid"][idx] = True
                valid[idx] = True
                n_done += 1
            ckpt.flush()
            _sync_to_box(local_h5, box_h5)
            print(
                f"[ckpt] valid={int(valid.sum())}/{n_runs}  "
                f"batch={n_done}/{len(pending)}  "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
    finally:
        ckpt.close()

    # reload arrays for Pearson
    with h5py.File(local_h5, "r") as f:
        geo_all = np.asarray(f["geomean"][:], dtype=np.float64)
        valid = np.asarray(f["valid"][:], dtype=bool)

    _sync_to_box(local_h5, box_h5)
    print(
        f"[pretell] ensembles ready  valid={int(valid.sum())}/{n_runs}  "
        f"in {time.time() - t0:.1f}s → {box_h5}"
    )

    if smoke:
        print("[smoke] skipping pearson_center update")
        print(f"[done] → {OUT_DIR}")
        return

    pearson_csv = OUT_DIR / "pearson_center.csv"
    pearson_h5 = OUT_DIR / "pearson_center.h5"
    df = pd.read_csv(pearson_csv)
    r_list = np.full(len(df), np.nan)
    for i, row in tqdm(df.iterrows(), total=len(df), desc="pearson pretell"):
        idx = int(row["index"])
        if idx >= len(valid) or not valid[idx]:
            continue
        r_list[i] = pearson_rows(tf_2d[idx : idx + 1], geo_all[idx])[0]
    df["r_pretell"] = r_list

    tmp_csv = Path("/tmp/pearson_center_pretell.csv")
    df.to_csv(tmp_csv, index=False)
    shutil.copy2(tmp_csv, pearson_csv)

    tmp_h5 = Path("/tmp/pearson_center_pretell.h5")
    with h5py.File(tmp_h5, "w") as f:
        for col in df.columns:
            data = df[col].to_numpy()
            if data.dtype == object:
                continue
            f.create_dataset(
                col,
                data=data.astype(np.float64 if data.dtype.kind == "f" else data.dtype),
            )
        f.attrs["n_rows"] = len(df)
        f.attrs["has_pretell"] = True
    shutil.copy2(tmp_h5, pearson_h5)

    v = df["r_pretell"].to_numpy()
    v = v[np.isfinite(v)]
    if len(v) == 0:
        print("[summary] r_pretell  n=0  (no valid Pretell TFs)")
    else:
        print(
            f"[summary] r_pretell  n={len(v)}  mean={np.mean(v):.3f}  "
            f"p50={np.median(v):.3f}  "
            f"p16–p84={np.percentile(v,16):.3f}–{np.percentile(v,84):.3f}"
        )
    print(f"[done] → {OUT_DIR}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true", help="wipe checkpoint and restart")
    p.add_argument("--n-jobs", type=int, default=None)
    args = p.parse_args()
    run(smoke=args.smoke, force=args.force, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
