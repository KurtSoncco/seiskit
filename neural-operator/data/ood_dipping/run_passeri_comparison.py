"""Passeri fixed-H vs Passeri dip-depth on ood_dipping Sobol points.

Same Hallal-style workflow as ``run_toro_comparison.py``, Passeri tts arm:

1. Reuse / cache 2D center |TF| from Box ``ood_dipping/h5/run_*.h5``.
2. For each of 32 Sobol physics points, build N Passeri 1D ensembles:
   - ``fixed``: tts randomization, bedrock depth fixed at H
   - ``dip``: tts + depth ``H + x tan(θ_Sobol)``,
     ``x ~ Unif[-250, 250]`` m (θ fixed to that case's dip angle)
3. Geomean ± σ_ln of |TF|; Pearson r(ln|TF|_2D, ln|TF|_geomean) per RF seed.
4. Figures comparing the two Passeri arms.

Usage
-----
  python run_passeri_comparison.py --smoke
  python run_passeri_comparison.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
NO_DATA = THIS_DIR.parent
REPO = NO_DATA.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from seiskit.damping import compute_damping_from_Q, compute_quality_factor  # noqa: E402
from seiskit.profile_randomization import (  # noqa: E402
    ProfileRandomizationConfig,
    generate_passeri_profile,
    hallal_profile_config,
)
from seiskit.theory.layered_1d_tf import (  # noqa: E402
    Layer,
    RockHalfspace,
    layered_transfer_function,
)
from seiskit.ttf.TTF import TTF  # noqa: E402

try:
    import hdf5plugin  # noqa: F401
except ImportError:
    pass

BOX_ROOT = Path(
    os.getenv(
        "NO_BOX_ROOT",
        "/mnt/box/GIG Lab - UC Berkeley/Projects/Neural Operator/data",
    )
)
DIP_BOX = BOX_ROOT / "ood_dipping"
H5_DIR = Path(os.getenv("OOD_H5_DIR", str(DIP_BOX / "h5")))
MANIFEST_PATH = Path(os.getenv("OOD_MANIFEST", str(THIS_DIR / "manifest.csv")))
OUT_DIR = Path(os.getenv("OOD_PASSERI_OUT", str(THIS_DIR / "passeri_comparison")))
FIG_DIR = OUT_DIR / "figures"

N_LATERALS = 21
CENTER_LATERAL = 10  # within each depth row
BASE_COL = CENTER_LATERAL  # y=2 m row first
SURF_COL = N_LATERALS + CENTER_LATERAL  # y=Lz row second
N_FREQ = 1000
FREQ_LO = 0.1
FREQ_HI = 10.0
RHO = 2000.0
DZ = 1.0
N_REAL_DEFAULT = 200
DIP_HALF_SPAN = 250.0
EPS = 1e-12


@dataclass(frozen=True)
class PhysEntry:
    sobol_id: int
    Vs1: float
    Vs2: float
    H: float
    CoV: float
    rH: float
    aHV: float
    dip_angle_deg: float
    bedrock_thickness: float


@dataclass(frozen=True)
class RunEntry:
    index: int
    sobol_id: int
    replicate_id: int
    rf_seed: int


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
    try:
        import hdf5plugin

        os.environ.setdefault("HDF5_PLUGIN_PATH", str(hdf5plugin.PLUGIN_PATH))
    except ImportError:
        pass


def load_manifest(path: Path) -> tuple[list[PhysEntry], list[RunEntry]]:
    rows = list(csv.DictReader(path.open()))
    phys: dict[int, PhysEntry] = {}
    runs: list[RunEntry] = []
    for row in rows:
        sid = int(row["sobol_id"])
        runs.append(
            RunEntry(
                index=int(row["index"]),
                sobol_id=sid,
                replicate_id=int(row["replicate_id"]),
                rf_seed=int(row["rf_seed"]),
            )
        )
        if sid not in phys:
            phys[sid] = PhysEntry(
                sobol_id=sid,
                Vs1=float(row["Vs1"]),
                Vs2=float(row["Vs2"]),
                H=float(row["H_discretized"]),
                CoV=float(row["CoV"]),
                rH=float(row["rH"]),
                aHV=float(row["aHV"]),
                dip_angle_deg=float(row["dip_angle_deg"]),
                bedrock_thickness=float(row["bedrock_thickness_discretized"]),
            )
    return [phys[k] for k in sorted(phys)], runs


def xi_of_vs(vs: float) -> float:
    return float(compute_damping_from_Q(compute_quality_factor(float(vs))))


def two_layer_af(freq: np.ndarray, vs1: float, H: float, vs2: float) -> np.ndarray:
    layers = [Layer(float(H), float(vs1), RHO, xi_of_vs(vs1))]
    rock = RockHalfspace(float(vs2), RHO, xi_of_vs(vs2))
    _, aw, _ = layered_transfer_function(freq, layers, rock)
    return np.asarray(aw, dtype=np.float64)


def _rng_seed(sobol_id: int, arm: str, realization: int) -> int:
    tag = {"fixed": 3, "dip": 4}[arm]  # distinct from Toro tags (1, 2)
    return int((sobol_id * 1_000_003 + tag * 97_001 + realization * 13) % (2**31 - 1))


def _passeri_config(phys: PhysEntry, arm: str) -> ProfileRandomizationConfig:
    if arm == "fixed":
        return hallal_profile_config(
            vs1=phys.Vs1,
            H=phys.H,
            cov=phys.CoV,
            vs2=phys.Vs2,
            dz=DZ,
            bedrock_thickness=max(phys.bedrock_thickness, 20.0),
        )
    theta = float(phys.dip_angle_deg)
    return ProfileRandomizationConfig(
        vs_mean=float(phys.Vs1),
        thickness=float(phys.H),
        dz=DZ,
        cov=float(phys.CoV),
        vs_bedrock=float(phys.Vs2),
        bedrock_thickness=max(float(phys.bedrock_thickness), 20.0),
        sigma_ln_vs=float(phys.CoV),
        sigma_ln_tts=float(phys.CoV),
        use_full_model=True,
        randomize_layer_thickness=False,
        randomize_bedrock_depth=True,
        bedrock_depth_model="dip",
        dip_angle_min_deg=theta,
        dip_angle_max_deg=theta,
        dip_half_span_m=DIP_HALF_SPAN,
        vary_bedrock_vs=False,
    )


def ensemble_passeri(
    phys: PhysEntry,
    arm: str,
    freq: np.ndarray,
    n_real: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (geomean |TF|, σ_ln |TF|) over *n_real* Passeri realizations."""
    cfg = _passeri_config(phys, arm)
    stack = np.empty((n_real, len(freq)), dtype=np.float64)
    for i in range(n_real):
        rng = np.random.default_rng(_rng_seed(phys.sobol_id, arm, i))
        prof = generate_passeri_profile(cfg, rng)
        vs_s = float(prof.vs_depth[0])
        vs_b = float(prof.vs_depth[-1])
        H_use = float(phys.H) if arm == "fixed" else float(prof.interface_depth)
        stack[i] = two_layer_af(freq, vs_s, H_use, vs_b)
    log_tf = np.log(np.clip(stack, EPS, None))
    geo = np.exp(np.mean(log_tf, axis=0))
    sig = np.std(log_tf, axis=0, ddof=1) if n_real > 1 else np.zeros(len(freq))
    return geo.astype(np.float64), sig.astype(np.float64)


def _run_one_physics(phys: PhysEntry, freq: np.ndarray, n_real: int) -> dict:
    out: dict = {
        "sobol_id": phys.sobol_id,
        "Vs1": phys.Vs1,
        "H": phys.H,
        "CoV": phys.CoV,
        "rH": phys.rH,
        "aHV": phys.aHV,
        "Vs2": phys.Vs2,
        "dip_angle_deg": phys.dip_angle_deg,
    }
    for arm in ("fixed", "dip"):
        geo, sig = ensemble_passeri(phys, arm, freq, n_real)
        out[f"passeri_{arm}_geomean"] = geo
        out[f"passeri_{arm}_sigma_ln"] = sig
    return out


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


def _tf_one_h5(index: int, h5_dir: Path) -> tuple[int, np.ndarray, np.ndarray] | None:
    try:
        import hdf5plugin  # noqa: F401
    except ImportError:
        pass
    path = h5_dir / f"run_{index}.h5"
    if not path.is_file():
        return None
    with h5py.File(path, "r") as f:
        data = np.asarray(f["recorders/accel/data"][:], dtype=np.float64)
        dt = float(f["grid"].attrs["dt"])
    if data.ndim != 2 or data.shape[1] < SURF_COL + 1:
        raise ValueError(f"{path}: unexpected accel shape {data.shape}")
    base = data[:, BASE_COL]
    surf = data[:, SURF_COL]
    freq, af = TTF(surf, base, dt=dt, n_points=N_FREQ)
    return index, np.asarray(freq, dtype=np.float64), np.asarray(af, dtype=np.float64)


def cache_2d_center_tf(
    runs: list[RunEntry],
    *,
    force: bool = False,
    n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (n_runs, n_freq) center |TF| cache aligned to manifest indices."""
    # Prefer shared cache from the Toro run when present.
    shared = THIS_DIR / "toro_comparison" / "tf_2d_center.h5"
    out = shared if shared.exists() else (OUT_DIR / "tf_2d_center.h5")
    n_needed = max(r.index for r in runs) + 1
    if out.exists() and not force:
        with h5py.File(out, "r") as f:
            tf = np.asarray(f["tf_center"][:], dtype=np.float64)
            freq = np.asarray(f["freq"][:], dtype=np.float64)
        if tf.shape[0] >= n_needed and np.isfinite(tf[[r.index for r in runs]]).all():
            print(f"[skip] {out} exists ({tf.shape[0]} runs)")
            return freq, tf
        print(f"[2d] cache incomplete ({tf.shape[0]} < {n_needed}); rebuilding")

    indices = [r.index for r in runs]
    print(f"[2d] computing TTF for {len(indices)} H5s under {H5_DIR}")
    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_tf_one_h5)(idx, H5_DIR) for idx in indices
    )
    missing = [idx for idx, res in zip(indices, results) if res is None]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} H5s (e.g. run_{missing[0]}.h5) under {H5_DIR}"
        )

    # Common log-frequency axis (TTF default 0.1–10 Hz); verify and stack.
    freq0 = results[0][1]
    n_runs = max(indices) + 1
    tf = np.full((n_runs, len(freq0)), np.nan, dtype=np.float64)
    for idx, freq, af in results:
        if not np.allclose(freq, freq0):
            # interpolate onto freq0 if needed
            af = np.interp(freq0, freq, af)
        tf[idx] = af
    if abs(float(freq0[0]) - FREQ_LO) > 1e-6 or abs(float(freq0[-1]) - FREQ_HI) > 1e-3:
        print(f"[warn] freq range {freq0[0]}–{freq0[-1]} (expected {FREQ_LO}–{FREQ_HI})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.create_dataset("freq", data=freq0.astype(np.float64), compression="gzip")
        f.create_dataset("tf_center", data=tf.astype(np.float32), compression="gzip", chunks=True)
        f.attrs["center_lateral"] = CENTER_LATERAL
        f.attrs["base_col"] = BASE_COL
        f.attrs["surf_col"] = SURF_COL
        f.attrs["n_runs"] = n_runs
        f.attrs["h5_dir"] = str(H5_DIR)
    print(f"[2d] wrote {out} in {time.time() - t0:.1f}s  shape={tf.shape}")
    return freq0, tf


def plot_results(
    freq: np.ndarray,
    phys_list: list[PhysEntry],
    runs: list[RunEntry],
    tf_2d: np.ndarray,
    sample_ids: np.ndarray,
    geo_fixed: np.ndarray,
    sig_fixed: np.ndarray,
    geo_dip: np.ndarray,
    sig_dip: np.ndarray,
    pearson_df: pd.DataFrame,
    *,
    n_panel: int = 6,
) -> None:
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sid_to_row = {int(s): i for i, s in enumerate(sample_ids)}
    C_FIXED = "#0072B2"
    C_DIP = "#D55E00"
    C_2D = "#000000"

    # --- Pearson boxplot ---
    fig, ax = plt.subplots(figsize=(5.2, 4.2), constrained_layout=True)
    data = [pearson_df["r_passeri_fixed"].dropna(), pearson_df["r_passeri_dip"].dropna()]
    bp = ax.boxplot(
        data,
        tick_labels=["Passeri fixed H", "Passeri dip depth"],
        patch_artist=True,
        widths=0.55,
        showfliers=False,
    )
    for patch, color in zip(bp["boxes"], (C_FIXED, C_DIP)):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    ax.set_ylabel(r"Pearson $r(\ln|\mathrm{TF}|_{\mathrm{2D}},\ \ln|\mathrm{TF}|_{\mathrm{geo}})$")
    ax.set_title("ood_dipping — Passeri vs 2D center TF")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    med_f = float(np.nanmedian(data[0]))
    med_d = float(np.nanmedian(data[1]))
    ax.text(
        0.98,
        0.04,
        f"median fixed={med_f:.3f}\nmedian dip={med_d:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="0.8"),
    )
    fig.savefig(FIG_DIR / "pearson_passeri_fixed_vs_dip.png", dpi=160)
    plt.close(fig)
    print(f"[fig] {FIG_DIR / 'pearson_passeri_fixed_vs_dip.png'}")

    # --- Geomean panels for a stratified subset of Sobol points ---
    # pick evenly spaced sobol ids
    sids = [p.sobol_id for p in phys_list]
    if len(sids) > n_panel:
        pick = np.linspace(0, len(sids) - 1, n_panel).astype(int)
        show_sids = [sids[i] for i in pick]
    else:
        show_sids = sids

    by_sid: dict[int, list[RunEntry]] = {}
    for r in runs:
        by_sid.setdefault(r.sobol_id, []).append(r)

    n = len(show_sids)
    fig, axes = plt.subplots(n, 2, figsize=(10.5, 2.4 * n), sharex=True, constrained_layout=True)
    if n == 1:
        axes = np.asarray([axes])
    phys_map = {p.sobol_id: p for p in phys_list}

    for row, sid in enumerate(show_sids):
        i_ens = sid_to_row[sid]
        p = phys_map[sid]
        entries = sorted(by_sid[sid], key=lambda e: e.replicate_id)
        # One 2D center realization (replicate 0) vs the Passeri ensemble geomean.
        e0 = entries[0]
        tf0 = tf_2d[e0.index]
        ax0, ax1 = axes[row, 0], axes[row, 1]
        for ax in (ax0, ax1):
            ax.loglog(
                freq,
                tf0,
                color=C_2D,
                lw=1.6,
                zorder=3,
                label=f"2D center (seed {e0.replicate_id})",
            )

        g_f, s_f = geo_fixed[i_ens], sig_fixed[i_ens]
        g_d, s_d = geo_dip[i_ens], sig_dip[i_ens]
        lo_f, hi_f = g_f * np.exp(-s_f), g_f * np.exp(s_f)
        lo_d, hi_d = g_d * np.exp(-s_d), g_d * np.exp(s_d)

        ax0.loglog(freq, g_f, color=C_FIXED, lw=1.8, label="Passeri fixed geomean")
        ax0.fill_between(freq, lo_f, hi_f, color=C_FIXED, alpha=0.2, label=r"±$\sigma_{\ln}$")
        ax1.loglog(freq, g_d, color=C_DIP, lw=1.8, label="Passeri dip geomean")
        ax1.fill_between(freq, lo_d, hi_d, color=C_DIP, alpha=0.2, label=r"±$\sigma_{\ln}$")

        # also overlay the other arm faintly for direct comparison
        ax0.loglog(freq, g_d, color=C_DIP, lw=1.2, ls="--", alpha=0.85, label="Passeri dip geomean")
        ax1.loglog(
            freq, g_f, color=C_FIXED, lw=1.2, ls="--", alpha=0.85, label="Passeri fixed geomean"
        )

        for ax in (ax0, ax1):
            ax.set_xlim(FREQ_LO, FREQ_HI)
            ax.grid(True, which="both", alpha=0.25)
            ax.set_ylabel(r"$|\mathrm{TF}|$")
        title = (
            rf"sobol {sid}: $V_{{s1}}$={p.Vs1:.0f}, $H$={p.H:.0f} m, "
            rf"CoV={p.CoV:.2f}, $\theta$={p.dip_angle_deg:+.2f}°"
        )
        ax0.set_title(title + "  |  fixed H", fontsize=9)
        ax1.set_title("dip depth", fontsize=9)
        if row == 0:
            ax0.legend(fontsize=7, loc="lower left")
            ax1.legend(fontsize=7, loc="lower left")
        if row == n - 1:
            ax0.set_xlabel("Frequency (Hz)")
            ax1.set_xlabel("Frequency (Hz)")

    fig.suptitle(
        "ood_dipping: one 2D center TF vs Passeri geomean — fixed H vs dip depth",
        fontsize=11,
    )
    out = FIG_DIR / "geomean_panels_passeri_fixed_vs_dip.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[fig] {out}")


def run(*, smoke: bool = False, force: bool = False, n_jobs: int | None = None) -> None:
    _configure_blas()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_real = _n_real(smoke)
    jobs = _n_jobs() if n_jobs is None else n_jobs
    print(f"[config] H5_DIR={H5_DIR}")
    print(f"[config] MANIFEST={MANIFEST_PATH}")
    print(f"[config] OUT={OUT_DIR}")
    print(f"[config] n_real={n_real}  n_jobs={jobs}  smoke={smoke}")

    phys_list, runs = load_manifest(MANIFEST_PATH)
    if smoke:
        keep = {p.sobol_id for p in phys_list[:2]}
        phys_list = [p for p in phys_list if p.sobol_id in keep]
        runs = [r for r in runs if r.sobol_id in keep]
    print(f"[manifest] {len(phys_list)} Sobol physics, {len(runs)} runs")

    freq, tf_2d = cache_2d_center_tf(runs, force=force, n_jobs=jobs)

    ens_path = OUT_DIR / ("ensembles_smoke.h5" if smoke else "ensembles.h5")
    if ens_path.exists() and not force:
        print(f"[skip] {ens_path} exists")
        with h5py.File(ens_path, "r") as f:
            sample_ids = np.asarray(f["sobol_id"][:], dtype=int)
            geo_fixed = np.asarray(f["passeri_fixed_geomean"][:], dtype=np.float64)
            sig_fixed = np.asarray(f["passeri_fixed_sigma_ln"][:], dtype=np.float64)
            geo_dip = np.asarray(f["passeri_dip_geomean"][:], dtype=np.float64)
            sig_dip = np.asarray(f["passeri_dip_sigma_ln"][:], dtype=np.float64)
    else:
        t1 = time.time()
        print(f"[ensembles] {len(phys_list)} physics × passeri(fixed+dip)×{n_real}")
        results = Parallel(n_jobs=jobs, backend="loky", verbose=10)(
            delayed(_run_one_physics)(p, freq, n_real) for p in phys_list
        )
        results = sorted(results, key=lambda r: r["sobol_id"])
        sample_ids = np.array([r["sobol_id"] for r in results], dtype=np.int32)
        geo_fixed = np.stack([r["passeri_fixed_geomean"] for r in results])
        sig_fixed = np.stack([r["passeri_fixed_sigma_ln"] for r in results])
        geo_dip = np.stack([r["passeri_dip_geomean"] for r in results])
        sig_dip = np.stack([r["passeri_dip_sigma_ln"] for r in results])
        with h5py.File(ens_path, "w") as f:
            f.create_dataset("freq", data=freq.astype(np.float64))
            f.create_dataset("sobol_id", data=sample_ids)
            f.create_dataset("Vs1", data=np.array([r["Vs1"] for r in results]))
            f.create_dataset("H", data=np.array([r["H"] for r in results]))
            f.create_dataset("CoV", data=np.array([r["CoV"] for r in results]))
            f.create_dataset("rH", data=np.array([r["rH"] for r in results]))
            f.create_dataset("aHV", data=np.array([r["aHV"] for r in results]))
            f.create_dataset("Vs2", data=np.array([r["Vs2"] for r in results]))
            f.create_dataset("dip_angle_deg", data=np.array([r["dip_angle_deg"] for r in results]))
            f.create_dataset(
                "passeri_fixed_geomean", data=geo_fixed.astype(np.float32), compression="gzip"
            )
            f.create_dataset(
                "passeri_fixed_sigma_ln", data=sig_fixed.astype(np.float32), compression="gzip"
            )
            f.create_dataset(
                "passeri_dip_geomean", data=geo_dip.astype(np.float32), compression="gzip"
            )
            f.create_dataset(
                "passeri_dip_sigma_ln", data=sig_dip.astype(np.float32), compression="gzip"
            )
            f.attrs["n_real"] = n_real
            f.attrs["n_physics"] = len(results)
            f.attrs["dip_half_span_m"] = DIP_HALF_SPAN
        print(f"[ensembles] wrote {ens_path} in {time.time() - t1:.1f}s")

    # Pearson
    pearson_csv = OUT_DIR / ("pearson_smoke.csv" if smoke else "pearson.csv")
    sid_to_row = {int(s): i for i, s in enumerate(sample_ids)}
    by_sample: dict[int, list[RunEntry]] = {}
    for e in runs:
        by_sample.setdefault(e.sobol_id, []).append(e)

    rows: list[dict] = []
    for sid, entries in tqdm(by_sample.items(), desc="pearson"):
        if sid not in sid_to_row:
            continue
        i_ens = sid_to_row[sid]
        indices = [e.index for e in entries]
        tf_block = tf_2d[indices]
        r_fixed = pearson_rows(tf_block, geo_fixed[i_ens])
        r_dip = pearson_rows(tf_block, geo_dip[i_ens])
        # geomean absolute relative L1 difference between arms
        g0, g1 = geo_fixed[i_ens], geo_dip[i_ens]
        rel_l1_arms = float(np.mean(np.abs(g0 - g1) / np.clip(0.5 * (g0 + g1), EPS, None)))
        for j, e in enumerate(entries):
            rows.append(
                {
                    "index": e.index,
                    "sobol_id": e.sobol_id,
                    "replicate_id": e.replicate_id,
                    "rf_seed": e.rf_seed,
                    "r_passeri_fixed": float(r_fixed[j]),
                    "r_passeri_dip": float(r_dip[j]),
                    "delta_r": float(r_dip[j] - r_fixed[j]),
                    "rel_l1_geomean_arms": rel_l1_arms,
                }
            )
    df = pd.DataFrame(rows)
    # attach physics
    phys_map = {p.sobol_id: p for p in phys_list}
    for col in ("Vs1", "H", "CoV", "rH", "aHV", "Vs2", "dip_angle_deg"):
        df[col] = [getattr(phys_map[sid], col) for sid in df["sobol_id"]]
    df.to_csv(pearson_csv, index=False)
    with h5py.File(pearson_csv.with_suffix(".h5"), "w") as f:
        f.create_dataset("index", data=df["index"].to_numpy())
        f.create_dataset("sobol_id", data=df["sobol_id"].to_numpy())
        f.create_dataset("r_passeri_fixed", data=df["r_passeri_fixed"].to_numpy())
        f.create_dataset("r_passeri_dip", data=df["r_passeri_dip"].to_numpy())
        f.create_dataset("delta_r", data=df["delta_r"].to_numpy())
    print(f"[pearson] wrote {pearson_csv}  n={len(df)}")
    print(
        f"  median r fixed={df['r_passeri_fixed'].median():.4f}  "
        f"dip={df['r_passeri_dip'].median():.4f}  "
        f"Δr median={df['delta_r'].median():+.4f}"
    )
    print(f"  mean rel L1(geomean fixed vs dip)={df['rel_l1_geomean_arms'].mean():.4f}")

    plot_results(
        freq,
        phys_list,
        runs,
        tf_2d,
        sample_ids,
        geo_fixed,
        sig_fixed,
        geo_dip,
        sig_dip,
        df,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=None)
    args = parser.parse_args()
    run(smoke=args.smoke, force=args.force, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
