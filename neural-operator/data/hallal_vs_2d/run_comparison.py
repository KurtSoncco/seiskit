"""Hallal Toro / Passeri / Dmult vs neural-operator two-layer 2D center TF.

Reads the Box two-layer campaign (256 Sobol × 30 RF seeds = 7680), uses the
precomputed ``transfer_function/tf_per_sample.npy`` center lateral as the 2D
reference, builds Hallal 1D ensembles (N=200 Toro/Passeri + 1 Dmult per Sobol
column), then writes geomean / σ_ln and Pearson r(ln|TF|) to Box.

Usage
-----
  # smoke (2 samples, N=4)
  HALLAL_N_REAL=4 python run_comparison.py --smoke

  # full production
  python run_comparison.py

  # force rebuild of cached artifacts
  python run_comparison.py --force
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths / imports
# ---------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
NO_DATA = THIS_DIR.parent
REPO = NO_DATA.parents[1]
if str(NO_DATA) not in sys.path:
    sys.path.insert(0, str(NO_DATA))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sobol import (  # noqa: E402
    DEFAULT_RF_SEEDS_PER_SAMPLE,
    DEFAULT_SOBOL_SAMPLE_COUNT,
    ManifestEntry,
    build_manifest,
)
from seiskit.damping import (  # noqa: E402
    compute_damping_from_Q,
    compute_quality_factor,
)
from seiskit.profile_randomization import (  # noqa: E402
    dmult_from_vs_contrast,
    get_method,
    hallal_profile_config,
)
from seiskit.theory.layered_1d_tf import Layer, RockHalfspace, layered_transfer_function  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BOX_ROOT = Path(
    os.getenv(
        "NO_BOX_ROOT",
        "/mnt/box/GIG Lab - UC Berkeley/Projects/Neural Operator/data",
    )
)
TF_DIR = BOX_ROOT / "transfer_function"
OUT_DIR = BOX_ROOT / "hallal_vs_2d"

CENTER_LATERAL = 10  # 21 laterals → center index
N_LATERALS = 21
N_FREQ = 1000
FREQ_LO = 0.1
FREQ_HI = 10.0
RHO = 2000.0
BEDROCK_THICKNESS = 10.0
DZ = 1.0
N_REAL_DEFAULT = 200
EPS = 1e-12
# Dmult scales small-strain damping 0.25 ξ_Q, not the full Campbell ξ_Q.
DMULT_XI_SCALE = 0.25

METHODS = ("toro", "passeri")


def _n_jobs() -> int:
    raw = os.getenv("HALLAL_N_JOBS", "").strip()
    if raw:
        return int(raw)
    slurm = os.getenv("SLURM_CPUS_PER_TASK", "").strip()
    if slurm:
        return int(slurm)
    return -1


def _n_real(smoke: bool) -> int:
    raw = os.getenv("HALLAL_N_REAL", "").strip()
    if raw:
        return max(1, int(raw))
    return 4 if smoke else N_REAL_DEFAULT


def _configure_blas() -> None:
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")


# ---------------------------------------------------------------------------
# 2D center TF cache (from existing Box tf_per_sample.npy)
# ---------------------------------------------------------------------------
def cache_2d_center_tf(*, force: bool = False) -> Path:
    """Extract center lateral |TF| from Box cache → ``tf_2d_center.h5``."""
    out = OUT_DIR / "tf_2d_center.h5"
    if out.exists() and not force:
        print(f"[skip] {out} exists (use --force to rebuild)")
        return out

    freq_path = TF_DIR / "freq.npy"
    tf_path = TF_DIR / "tf_per_sample.npy"
    if not freq_path.is_file() or not tf_path.is_file():
        raise FileNotFoundError(
            f"Expected precomputed TF under {TF_DIR} "
            "(freq.npy, tf_per_sample.npy). Recompute from H5s if missing."
        )

    freq = np.asarray(np.load(freq_path), dtype=np.float64)
    tf_all = np.load(tf_path, mmap_mode="r")
    if tf_all.ndim != 3 or tf_all.shape[1] < N_LATERALS:
        raise ValueError(f"Unexpected tf_per_sample shape {tf_all.shape}")
    if abs(float(freq[0]) - FREQ_LO) > 1e-6 or abs(float(freq[-1]) - FREQ_HI) > 1e-3:
        print(f"[warn] freq range {freq[0]}–{freq[-1]} (expected {FREQ_LO}–{FREQ_HI})")

    center = np.asarray(tf_all[:, CENTER_LATERAL, :], dtype=np.float32)
    print(
        f"[2d] center lateral={CENTER_LATERAL}  shape={center.shape}  "
        f"freq={freq.shape[0]}"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.create_dataset("freq", data=freq.astype(np.float64), compression="gzip")
        f.create_dataset("tf_center", data=center, compression="gzip", chunks=True)
        f.attrs["center_lateral"] = CENTER_LATERAL
        f.attrs["n_runs"] = center.shape[0]
        f.attrs["source"] = str(tf_path)
    print(f"[2d] wrote {out}")
    return out


def load_2d_center(path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    path = path or (OUT_DIR / "tf_2d_center.h5")
    with h5py.File(path, "r") as f:
        freq = np.asarray(f["freq"][:], dtype=np.float64)
        tf = np.asarray(f["tf_center"][:], dtype=np.float64)
    return freq, tf


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------
def unique_samples(manifest: list[ManifestEntry]) -> list[ManifestEntry]:
    """One representative entry per sample_id (replicate 0)."""
    seen: dict[int, ManifestEntry] = {}
    for e in manifest:
        if e.sample_id not in seen:
            seen[e.sample_id] = e
    return [seen[k] for k in sorted(seen)]


# ---------------------------------------------------------------------------
# Haskell TF for Hallal arms
# ---------------------------------------------------------------------------
def xi_of_vs(vs: float, scale: float = 1.0) -> float:
    return float(scale * compute_damping_from_Q(compute_quality_factor(float(vs))))


def two_layer_af(
    freq: np.ndarray,
    vs1: float,
    H: float,
    vs2: float,
    *,
    soil_scale: float = 1.0,
    rock_scale: float = 1.0,
) -> np.ndarray:
    layers = [Layer(float(H), float(vs1), RHO, xi_of_vs(vs1, soil_scale))]
    rock = RockHalfspace(float(vs2), RHO, xi_of_vs(vs2, rock_scale))
    _, aw, _ = layered_transfer_function(freq, layers, rock)
    return np.asarray(aw, dtype=np.float64)


def _rng_seed(sample_id: int, method: str, realization: int) -> int:
    """Deterministic 31-bit seed from (sample, method, realization)."""
    method_tag = {"toro": 1, "passeri": 2, "dmult": 3}[method]
    # mix bits; stay in uint32 range for Generator
    return int((sample_id * 1_000_003 + method_tag * 97_001 + realization * 13) % (2**31 - 1))


def ensemble_geomean_sigma(
    entry: ManifestEntry,
    method: str,
    freq: np.ndarray,
    n_real: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (geomean |TF|, σ_ln |TF|) over *n_real* Hallal realizations."""
    cfg = hallal_profile_config(
        vs1=entry.Vs1,
        H=entry.H_discretized,
        cov=entry.CoV,
        vs2=entry.Vs2,
        dz=DZ,
        bedrock_thickness=BEDROCK_THICKNESS,
    )
    meth = get_method(method)
    stack = np.empty((n_real, len(freq)), dtype=np.float64)
    for i in range(n_real):
        rng = np.random.default_rng(_rng_seed(entry.sample_id, method, i))
        prof = meth.generate_profile(cfg, rng)
        vs_s = float(prof.vs_depth[0])
        vs_b = float(prof.vs_depth[-1])
        # Hallal Vs/tts: interface fixed at H; bedrock Vs fixed under hallal_profile_config
        stack[i] = two_layer_af(freq, vs_s, entry.H_discretized, vs_b)
    log_tf = np.log(np.clip(stack, EPS, None))
    geo = np.exp(np.mean(log_tf, axis=0))
    sig = np.std(log_tf, axis=0, ddof=1) if n_real > 1 else np.zeros(len(freq))
    return geo.astype(np.float64), sig.astype(np.float64)


def dmult_tf(entry: ManifestEntry, freq: np.ndarray) -> tuple[np.ndarray, float]:
    """Base-column |TF| with ξ = D_mult × 0.25 ξ_Q on soil and rock."""
    dmult = dmult_from_vs_contrast(entry.Vs1, entry.Vs2)
    scale = dmult * DMULT_XI_SCALE
    af = two_layer_af(
        freq,
        entry.Vs1,
        entry.H_discretized,
        entry.Vs2,
        soil_scale=scale,
        rock_scale=scale,
    )
    return af, float(dmult)


def _run_one_sample_ensembles(
    entry: ManifestEntry,
    freq: np.ndarray,
    n_real: int,
) -> dict:
    """Worker: Toro + Passeri geomean/σ_ln + Dmult for one Sobol sample."""
    out: dict = {
        "sample_id": entry.sample_id,
        "Vs1": entry.Vs1,
        "H": entry.H_discretized,
        "CoV": entry.CoV,
        "rH": entry.rH,
        "aHV": entry.aHV,
        "Vs2": entry.Vs2,
    }
    for method in METHODS:
        geo, sig = ensemble_geomean_sigma(entry, method, freq, n_real)
        out[f"{method}_geomean"] = geo
        out[f"{method}_sigma_ln"] = sig
    af_d, dmult = dmult_tf(entry, freq)
    out["dmult_tf"] = af_d
    out["dmult"] = dmult
    return out


# ---------------------------------------------------------------------------
# Pearson
# ---------------------------------------------------------------------------
def pearson_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson r of each row of *x* against fixed vector *y* (log space)."""
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


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run(
    *,
    smoke: bool = False,
    force: bool = False,
    n_jobs: int | None = None,
) -> None:
    _configure_blas()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_real = _n_real(smoke)
    jobs = _n_jobs() if n_jobs is None else n_jobs
    print(f"[config] BOX={BOX_ROOT}")
    print(f"[config] OUT={OUT_DIR}")
    print(f"[config] n_real={n_real}  n_jobs={jobs}  smoke={smoke}")

    # --- 2D center TF ---
    t0 = time.time()
    cache_2d_center_tf(force=force)
    freq, tf_2d = load_2d_center()
    print(f"[2d] loaded tf_center {tf_2d.shape} in {time.time() - t0:.1f}s")

    # Shared frequency axis for Haskell (match Box TF grid)
    if len(freq) != N_FREQ:
        print(f"[warn] freq length {len(freq)} != {N_FREQ}; using loaded axis")

    # --- Manifest ---
    manifest = build_manifest(
        sample_count=DEFAULT_SOBOL_SAMPLE_COUNT,
        seeds_per_sample=DEFAULT_RF_SEEDS_PER_SAMPLE,
    )
    if len(manifest) != tf_2d.shape[0]:
        raise RuntimeError(
            f"Manifest length {len(manifest)} != tf_2d runs {tf_2d.shape[0]}"
        )
    samples = unique_samples(manifest)
    if smoke:
        samples = samples[:2]
        # keep only matching 2D rows for summary; full pearson still uses all
        # unless smoke — then only indices belonging to those samples
    print(f"[manifest] {len(manifest)} runs, {len(samples)} unique Sobol samples")

    # --- Ensembles ---
    ens_path = OUT_DIR / "ensembles.h5"
    if ens_path.exists() and not force and not smoke:
        print(f"[skip] {ens_path} exists")
        with h5py.File(ens_path, "r") as f:
            sample_ids = np.asarray(f["sample_id"][:], dtype=int)
            toro_geo = np.asarray(f["toro_geomean"][:], dtype=np.float64)
            passeri_geo = np.asarray(f["passeri_geomean"][:], dtype=np.float64)
            dmult_afs = np.asarray(f["dmult_tf"][:], dtype=np.float64)
            dmult_vals = np.asarray(f["dmult"][:], dtype=np.float64)
            meta = {
                int(sid): {
                    "Vs1": float(f["Vs1"][i]),
                    "H": float(f["H"][i]),
                    "CoV": float(f["CoV"][i]),
                    "rH": float(f["rH"][i]),
                    "aHV": float(f["aHV"][i]),
                    "Vs2": float(f["Vs2"][i]),
                    "dmult": float(dmult_vals[i]),
                }
                for i, sid in enumerate(sample_ids)
            }
    else:
        t1 = time.time()
        print(f"[ensembles] computing {len(samples)} samples × "
              f"(toro+passeri)×{n_real} + dmult …")
        results = Parallel(n_jobs=jobs, backend="loky", verbose=10)(
            delayed(_run_one_sample_ensembles)(e, freq, n_real) for e in samples
        )
        # sort by sample_id
        results = sorted(results, key=lambda r: r["sample_id"])
        n_s = len(results)
        sample_ids = np.array([r["sample_id"] for r in results], dtype=np.int32)
        toro_geo = np.stack([r["toro_geomean"] for r in results])
        toro_sig = np.stack([r["toro_sigma_ln"] for r in results])
        passeri_geo = np.stack([r["passeri_geomean"] for r in results])
        passeri_sig = np.stack([r["passeri_sigma_ln"] for r in results])
        dmult_afs = np.stack([r["dmult_tf"] for r in results])
        dmult_vals = np.array([r["dmult"] for r in results], dtype=np.float64)
        meta = {
            int(r["sample_id"]): {
                "Vs1": r["Vs1"],
                "H": r["H"],
                "CoV": r["CoV"],
                "rH": r["rH"],
                "aHV": r["aHV"],
                "Vs2": r["Vs2"],
                "dmult": r["dmult"],
            }
            for r in results
        }

        with h5py.File(ens_path, "w") as f:
            f.create_dataset("freq", data=freq.astype(np.float64))
            f.create_dataset("sample_id", data=sample_ids)
            f.create_dataset("Vs1", data=np.array([r["Vs1"] for r in results]))
            f.create_dataset("H", data=np.array([r["H"] for r in results]))
            f.create_dataset("CoV", data=np.array([r["CoV"] for r in results]))
            f.create_dataset("rH", data=np.array([r["rH"] for r in results]))
            f.create_dataset("aHV", data=np.array([r["aHV"] for r in results]))
            f.create_dataset("Vs2", data=np.array([r["Vs2"] for r in results]))
            f.create_dataset("dmult", data=dmult_vals)
            f.create_dataset("toro_geomean", data=toro_geo.astype(np.float32), compression="gzip")
            f.create_dataset("toro_sigma_ln", data=toro_sig.astype(np.float32), compression="gzip")
            f.create_dataset(
                "passeri_geomean", data=passeri_geo.astype(np.float32), compression="gzip"
            )
            f.create_dataset(
                "passeri_sigma_ln", data=passeri_sig.astype(np.float32), compression="gzip"
            )
            f.create_dataset("dmult_tf", data=dmult_afs.astype(np.float32), compression="gzip")
            f.attrs["n_real"] = n_real
            f.attrs["n_samples"] = n_s
            f.attrs["rho"] = RHO
            f.attrs["dz"] = DZ
            f.attrs["bedrock_thickness"] = BEDROCK_THICKNESS
            f.attrs["dmult_xi_scale"] = DMULT_XI_SCALE
        print(f"[ensembles] wrote {ens_path} in {time.time() - t1:.1f}s")

    # Map sample_id → row in ensemble arrays
    sid_to_row = {int(s): i for i, s in enumerate(sample_ids)}

    # --- Pearson for every 2D run ---
    pearson_csv = OUT_DIR / "pearson_center.csv"
    pearson_h5 = OUT_DIR / "pearson_center.h5"
    if pearson_csv.exists() and pearson_h5.exists() and not force and not smoke:
        print(f"[skip] {pearson_csv} exists")
        return

    t2 = time.time()
    if smoke:
        keep_sids = {e.sample_id for e in samples}
        run_entries = [e for e in manifest if e.sample_id in keep_sids]
    else:
        run_entries = manifest

    rows: list[dict] = []
    # Process in chunks by sample for vectorized Pearson over the 30 seeds
    by_sample: dict[int, list[ManifestEntry]] = {}
    for e in run_entries:
        by_sample.setdefault(e.sample_id, []).append(e)

    for sid, entries in tqdm(by_sample.items(), desc="pearson"):
        if sid not in sid_to_row:
            continue
        i_ens = sid_to_row[sid]
        indices = [e.index for e in entries]
        tf_block = tf_2d[indices]  # (n_seed, n_freq)
        r_toro = pearson_rows(tf_block, toro_geo[i_ens])
        r_pas = pearson_rows(tf_block, passeri_geo[i_ens])
        r_dm = pearson_rows(tf_block, dmult_afs[i_ens])
        m = meta[sid]
        for j, e in enumerate(entries):
            rows.append(
                {
                    "index": e.index,
                    "sample_id": e.sample_id,
                    "replicate_id": e.replicate_id,
                    "rf_seed": e.rf_seed,
                    "Vs1": m["Vs1"],
                    "H": m["H"],
                    "CoV": m["CoV"],
                    "rH": m["rH"],
                    "aHV": m["aHV"],
                    "Vs2": m["Vs2"],
                    "dmult": m["dmult"],
                    "r_toro": float(r_toro[j]),
                    "r_passeri": float(r_pas[j]),
                    "r_dmult": float(r_dm[j]),
                }
            )

    df = pd.DataFrame(rows).sort_values("index").reset_index(drop=True)
    df.to_csv(pearson_csv, index=False)
    with h5py.File(pearson_h5, "w") as f:
        for col in df.columns:
            data = df[col].to_numpy()
            if data.dtype == object:
                continue
            f.create_dataset(col, data=data.astype(np.float64 if data.dtype.kind == "f" else data.dtype))
        f.attrs["n_rows"] = len(df)
        f.attrs["n_real_ensemble"] = n_real
        f.attrs["center_lateral"] = CENTER_LATERAL
    print(f"[pearson] wrote {pearson_csv} and {pearson_h5} ({len(df)} rows) in {time.time() - t2:.1f}s")

    # Summary
    for col in ("r_toro", "r_passeri", "r_dmult"):
        v = df[col].to_numpy()
        v = v[np.isfinite(v)]
        print(
            f"[summary] {col:12s}  n={len(v)}  mean={np.mean(v):.3f}  "
            f"p50={np.median(v):.3f}  p16–p84={np.percentile(v,16):.3f}–{np.percentile(v,84):.3f}"
        )
    print(f"[done] total wall {time.time() - t0:.1f}s → {OUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="2 samples, small N")
    parser.add_argument("--force", action="store_true", help="Rebuild even if outputs exist")
    parser.add_argument("--n-jobs", type=int, default=None, help="Override HALLAL_N_JOBS")
    args = parser.parse_args()
    run(smoke=args.smoke, force=args.force, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
