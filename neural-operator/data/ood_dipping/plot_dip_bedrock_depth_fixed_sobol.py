"""Toro / Passeri interface depth at one fixed Sobol dip angle.

The triangular depth histograms mix angles. This figure locks θ to a single
Sobol point and draws only x ~ Unif[-250, 250] m, which is the law in
``run_toro_comparison`` / ``run_passeri_comparison``. Depth is then uniform
on [H ± 250|tan θ|].

Also scores the one-layer geomean |TF| under that fixed-θ law against the
mixed law θ ~ Unif[-3°, 3°], using the cached 2D center |TF|.

Writes
    figures/toro_passeri_dip_depth_fixed_sobol.png
    figures/fixed_sobol_depth_ks.csv
    figures/fixed_sobol_tf_delta.csv
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
REPO = THIS_DIR.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from plot_dip_bedrock_depth import _one_layer_column  # noqa: E402
from run_passeri_comparison import (  # noqa: E402
    _one_layer_passeri,
    _passeri_config,
    _rng_seed as _passeri_seed,
)
from run_toro_comparison import (  # noqa: E402
    DIP_HALF_SPAN,
    EPS,
    MANIFEST_PATH,
    OUT_DIR as TORO_OUT,
    PhysEntry,
    _one_layer_toro,
    _rng_seed as _toro_seed,
    _toro_config,
    load_manifest,
    pearson_rows,
    two_layer_af,
)
from seiskit.profile_randomization.common import _total_column_depth  # noqa: E402
from seiskit.profile_randomization.nhpp import _sample_interface_depth  # noqa: E402

FIG_DIR = THIS_DIR / "figures"
OUT_PATH = FIG_DIR / "toro_passeri_dip_depth_fixed_sobol.png"
KS_PATH = FIG_DIR / "fixed_sobol_depth_ks.csv"
TF_PATH = FIG_DIR / "fixed_sobol_tf_delta.csv"

N_DEPTH = 8000
N_PROFILE = 40
N_REAL = 200
SEED = 0
THETA_MIX_MIN = -3.0
THETA_MIX_MAX = 3.0
C_TORO = "#D55E00"
C_PASSERI = "#009E73"
C_MIX = "#0072B2"


def _clip_bounds(cfg) -> tuple[float, float]:
    min_t = max(cfg.min_layer_thickness, cfg.dz)
    z_total = _total_column_depth(cfg)
    return float(min_t), float(z_total - min_t)


def sample_dip(cfg, rng: np.random.Generator) -> tuple[float, bool]:
    """Dip-branch of ``_sample_interface_depth``, with a clip flag."""
    lo, hi = _clip_bounds(cfg)
    theta_deg = float(rng.uniform(cfg.dip_angle_min_deg, cfg.dip_angle_max_deg))
    x_m = float(rng.uniform(-cfg.dip_half_span_m, cfg.dip_half_span_m))
    raw = float(cfg.thickness) + x_m * np.tan(np.radians(theta_deg))
    return float(np.clip(raw, lo, hi)), bool(raw < lo or raw > hi)


def _check_sampler_matches(cfg) -> None:
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    for _ in range(32):
        got, _clipped = sample_dip(cfg, rng_a)
        ref = float(_sample_interface_depth(cfg, rng_b))
        if abs(got - ref) > 1e-9:
            raise RuntimeError(f"dip sampler diverged: {got} vs {ref}")


def ks_uniform(sample: np.ndarray, lo: float, hi: float) -> float:
    """Two-sided Kolmogorov–Smirnov distance to Uniform[lo, hi]."""
    x = np.sort(np.asarray(sample, dtype=np.float64))
    n = x.size
    if n == 0 or hi <= lo:
        return float("nan")
    u = (x - lo) / (hi - lo)
    ecdf = np.arange(1, n + 1) / n
    ecdf_left = np.arange(0, n) / n
    return float(max(np.max(ecdf - u), np.max(u - ecdf_left)))


def half_width(theta_deg: float) -> float:
    return DIP_HALF_SPAN * abs(np.tan(np.radians(float(theta_deg))))


def _draw_depths(cfg, n: int, seed: int) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    depths = np.empty(n, dtype=np.float64)
    n_clip = 0
    for i in range(n):
        depths[i], clipped = sample_dip(cfg, rng)
        n_clip += int(clipped)
    return depths, n_clip / n


def pick_points(phys: list[PhysEntry]) -> list[tuple[str, PhysEntry]]:
    abs_th = np.array([abs(p.dip_angle_deg) for p in phys])
    med = float(np.median(abs_th))
    choices = {
        "smallest |θ|": int(np.argmin(abs_th)),
        "median |θ|": int(np.argmin(np.abs(abs_th - med))),
        "steepest |θ|": int(np.argmax(abs_th)),
    }
    return [(label, phys[i]) for label, i in choices.items()]


def xi_of_vs(vs: np.ndarray) -> np.ndarray:
    v = np.asarray(vs, dtype=np.float64) / 1000.0
    q = (
        10.5
        - 16 * v
        + 153 * v**2
        - 103 * v**3
        + 34.7 * v**4
        - 5.29 * v**5
        + 0.31 * v**6
    )
    return 1.0 / (2.0 * q)


def af_within_batch(freq: np.ndarray, vs: np.ndarray, thickness: np.ndarray) -> np.ndarray:
    """One-layer |u_surface / u_interface|. Shape (n_real, n_freq).

    Matches ``layered_transfer_function`` af_within for a single soil layer:
    bedrock impedance does not enter that magnitude.
    """
    omega = 2.0 * np.pi * np.asarray(freq, dtype=np.float64)
    vs_c = np.asarray(vs, dtype=np.float64) * np.sqrt(1.0 + 2.0j * xi_of_vs(vs))
    kh = omega[:, None] * (np.asarray(thickness, dtype=np.float64) / vs_c)[None, :]
    af = np.abs(1.0 / np.cos(kh))
    af = np.where(omega[:, None] == 0.0, 1.0, af)
    return np.asarray(af.T, dtype=np.float64)


def _check_af(freq: np.ndarray) -> None:
    vs = np.array([180.0, 250.0, 320.0])
    thick = np.array([30.0, 45.0, 55.0])
    vs2 = np.array([900.0, 1200.0, 1400.0])
    batch = af_within_batch(freq, vs, thick)
    for i in range(vs.size):
        ref = two_layer_af(freq, float(vs[i]), float(thick[i]), float(vs2[i]))
        if not np.allclose(batch[i], ref, rtol=1e-8, atol=1e-8):
            err = float(np.max(np.abs(batch[i] - ref)))
            raise RuntimeError(f"vectorized |TF| disagrees with seiskit (max abs {err})")


def geomean_tf(stack: np.ndarray) -> np.ndarray:
    return np.exp(np.mean(np.log(np.clip(stack, EPS, None)), axis=0))


def rel_l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b) / np.clip(0.5 * (a + b), EPS, None)))


def _realizations(
    phys: PhysEntry,
    method: str,
    *,
    random_theta: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Soil Vs, bedrock Vs, and interface depth for N_REAL paired draws."""
    if method == "toro":
        cfg = _toro_config(phys, "dip")
        seed_of = _toro_seed
        draw = _one_layer_toro
    elif method == "passeri":
        cfg = _passeri_config(phys, "dip")
        seed_of = _passeri_seed
        draw = _one_layer_passeri
    else:
        raise ValueError(method)
    if random_theta:
        cfg = replace(
            cfg,
            dip_angle_min_deg=THETA_MIX_MIN,
            dip_angle_max_deg=THETA_MIX_MAX,
        )
    vs_s = np.empty(N_REAL, dtype=np.float64)
    vs_b = np.empty(N_REAL, dtype=np.float64)
    depth = np.empty(N_REAL, dtype=np.float64)
    for i in range(N_REAL):
        rng = np.random.default_rng(seed_of(phys.sobol_id, "dip", i))
        vs_s[i], vs_b[i], depth[i] = draw(cfg, rng)
    return vs_s, vs_b, depth


def _plot_hist(ax, toro: np.ndarray, pas: np.ndarray, phys: PhysEntry, title: str) -> None:
    half = half_width(phys.dip_angle_deg)
    pad = max(0.2 * half, 0.05)
    lo, hi = phys.H - half - pad, phys.H + half + pad
    bins = np.linspace(lo, hi, 31)
    ax.hist(
        toro,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.45,
        color=C_TORO,
        label="Toro",
    )
    ax.hist(
        pas,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.35,
        color=C_PASSERI,
        label="Passeri",
    )
    if half > 0:
        height = 1.0 / (2.0 * half)
        ax.plot(
            [phys.H - half, phys.H - half, phys.H + half, phys.H + half],
            [0.0, height, height, 0.0],
            color="0.1",
            lw=1.3,
            label="uniform",
        )
    ax.axvline(phys.H, color="0.25", ls="--", lw=1.0)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("Interface depth (m)")
    ax.set_title(
        rf"{title}" "\n" rf"$\theta={phys.dip_angle_deg:+.2f}^\circ$, $H={phys.H:.0f}$ m",
        fontsize=9,
    )
    ax.grid(True, alpha=0.25)


def _plot_profiles(ax, store: list[tuple[np.ndarray, np.ndarray]], phys: PhysEntry, title: str, color: str) -> None:
    half = half_width(phys.dip_angle_deg)
    for column, z in store:
        ax.step(column, z, where="pre", color=color, alpha=0.35, lw=0.8)
    ax.axhline(phys.H, color="0.2", ls="--", lw=1.0)
    ax.axhspan(phys.H - half, phys.H + half, color=color, alpha=0.08, zorder=0)
    ax.set_xlabel(r"$V_s$ (m/s)")
    ax.set_title(title, fontsize=9)
    z_max = max(float(z[-1]) for _col, z in store)
    ax.set_ylim(z_max + 0.5, 0.0)
    ax.set_xlim(50, 1800)
    ax.grid(True, alpha=0.25)


def plot_figure(
    picked: list[tuple[str, PhysEntry]],
    depth_samples: dict[int, dict[str, np.ndarray]],
    pooled_delta: np.ndarray,
    mix_delta: np.ndarray,
    steep: PhysEntry,
) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.4), constrained_layout=True)

    for ax, (label, phys) in zip(axes[0], picked):
        samples = depth_samples[phys.sobol_id]
        _plot_hist(ax, samples["toro"], samples["passeri"], phys, label)
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend(fontsize=7, loc="upper right")

    ax = axes[1, 0]
    half_max = half_width(3.0)
    bins = np.linspace(-half_max - 1.0, half_max + 1.0, 51)
    ax.hist(
        pooled_delta,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.45,
        color=C_TORO,
        label=r"32 Sobol $\theta$, stacked",
    )
    ax.hist(
        mix_delta,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.4,
        color=C_MIX,
        label=r"$\theta\sim\mathrm{Unif}[-3^\circ,3^\circ]$",
    )
    ax.axvline(0.0, color="0.2", ls="--", lw=1.0)
    ax.set_xlabel(r"Interface depth $-$ $H$ (m)")
    ax.set_ylabel("Density")
    ax.set_title("Angle mixtures", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.25)

    def _profiles(method: str) -> list[tuple[np.ndarray, np.ndarray]]:
        if method == "toro":
            cfg = _toro_config(steep, "dip")
        else:
            cfg = _passeri_config(steep, "dip")
        rows = []
        for k in range(N_PROFILE):
            _col, _iface = _one_layer_column(
                cfg, method, np.random.default_rng(SEED + 50 + k)
            )
            z = (np.arange(len(_col)) + 0.5) * float(cfg.dz)
            rows.append((_col, z))
        return rows

    _plot_profiles(
        axes[1, 1],
        _profiles("toro"),
        steep,
        rf"Toro, steepest Sobol ($\theta={steep.dip_angle_deg:+.2f}^\circ$)",
        C_TORO,
    )
    axes[1, 1].set_ylabel("Depth (m)")
    _plot_profiles(
        axes[1, 2],
        _profiles("passeri"),
        steep,
        rf"Passeri, steepest Sobol ($\theta={steep.dip_angle_deg:+.2f}^\circ$)",
        C_PASSERI,
    )

    fig.suptitle(
        r"Fixed Sobol dip: $\theta$ locked, $x\sim\mathrm{Unif}[-250,250]$ m"
        "\n"
        r"Top row is one physics point. Bottom-left stacks angles.",
        fontsize=11,
    )
    fig.savefig(OUT_PATH, dpi=160)
    print(f"Saved {OUT_PATH}")


def depth_table(phys_list: list[PhysEntry]) -> pd.DataFrame:
    rows = []
    for phys in phys_list:
        half = half_width(phys.dip_angle_deg)
        lo, hi = phys.H - half, phys.H + half
        row = {
            "sobol_id": phys.sobol_id,
            "dip_angle_deg": phys.dip_angle_deg,
            "H": phys.H,
            "half_width_m": half,
        }
        for method, cfg in (
            ("toro", _toro_config(phys, "dip")),
            ("passeri", _passeri_config(phys, "dip")),
        ):
            depths, clip_frac = _draw_depths(
                cfg, N_DEPTH, seed=SEED + 17 * phys.sobol_id + (0 if method == "toro" else 1)
            )
            row[f"ks_{method}"] = ks_uniform(depths, lo, hi)
            row[f"clipped_fraction_{method}"] = clip_frac
            row[f"sample_std_{method}"] = float(depths.std(ddof=0))
        row["uniform_std"] = (2.0 * half) / np.sqrt(12.0) if half > 0 else 0.0
        rows.append(row)
        print(
            f"  sobol {phys.sobol_id:>3} θ={phys.dip_angle_deg:+.3f}° "
            f"half={half:.2f} m  "
            f"KS toro/passeri={row['ks_toro']:.4f}/{row['ks_passeri']:.4f}  "
            f"clip={row['clipped_fraction_toro']:.4f}/{row['clipped_fraction_passeri']:.4f}"
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(KS_PATH, index=False)
    print(f"Wrote {KS_PATH}")
    return frame


def _pooled_and_mix(phys_list: list[PhysEntry]) -> tuple[np.ndarray, np.ndarray]:
    pooled = []
    per = max(1, N_DEPTH // len(phys_list))
    for phys in phys_list:
        cfg = _toro_config(phys, "dip")
        depths, _clip = _draw_depths(cfg, per, seed=SEED + 1000 + phys.sobol_id)
        pooled.append(depths - phys.H)
    ref = phys_list[0]
    mix_cfg = replace(
        _toro_config(ref, "dip"),
        thickness=40.0,
        bedrock_thickness=30.0,
        dip_angle_min_deg=THETA_MIX_MIN,
        dip_angle_max_deg=THETA_MIX_MAX,
    )
    mix_depths, _clip = _draw_depths(mix_cfg, N_DEPTH, seed=SEED + 7)
    return np.concatenate(pooled), mix_depths - 40.0


def score_tf(phys_list: list[PhysEntry], runs, tf_2d: np.ndarray, freq: np.ndarray) -> pd.DataFrame:
    import h5py

    _check_af(freq)
    stored = h5py.File(TORO_OUT / "ensembles.h5", "r")
    stored_sid = np.asarray(stored["sobol_id"][:], dtype=int)
    stored_geo = np.asarray(stored["toro_dip_geomean"][:], dtype=np.float64)
    sid_to_stored = {int(s): i for i, s in enumerate(stored_sid)}

    by_sid: dict[int, list] = {}
    for run in runs:
        by_sid.setdefault(run.sobol_id, []).append(run)

    rows = []
    rep_scores: dict[str, list[np.ndarray]] = {
        "toro_fixed": [],
        "toro_random": [],
        "passeri_fixed": [],
        "passeri_random": [],
    }
    for phys in phys_list:
        entries = by_sid[phys.sobol_id]
        indices = [e.index for e in entries]
        block = tf_2d[indices]
        row: dict = {
            "sobol_id": phys.sobol_id,
            "dip_angle_deg": phys.dip_angle_deg,
            "abs_theta": abs(phys.dip_angle_deg),
            "H": phys.H,
            "half_width_m": half_width(phys.dip_angle_deg),
            "n_reps": len(entries),
        }
        for method in ("toro", "passeri"):
            vs_a, _vb_a, depth_a = _realizations(phys, method, random_theta=False)
            vs_b, _vb_b, depth_b = _realizations(phys, method, random_theta=True)
            geo_a = geomean_tf(af_within_batch(freq, vs_a, depth_a))
            geo_b = geomean_tf(af_within_batch(freq, vs_b, depth_b))
            r_a = pearson_rows(block, geo_a)
            r_b = pearson_rows(block, geo_b)
            row[f"r_{method}_fixed_theta"] = float(np.nanmedian(r_a))
            row[f"r_{method}_random_theta"] = float(np.nanmedian(r_b))
            row[f"delta_r_{method}"] = float(np.nanmedian(r_b - r_a))
            row[f"rel_l1_{method}"] = rel_l1(geo_a, geo_b)
            rep_scores[f"{method}_fixed"].append(r_a)
            rep_scores[f"{method}_random"].append(r_b)
            if method == "toro":
                ref = stored_geo[sid_to_stored[phys.sobol_id]]
                row["rel_l1_toro_vs_stored_dip"] = rel_l1(geo_a, ref)
        rows.append(row)
        print(
            f"  sobol {phys.sobol_id:>3} |θ|={row['abs_theta']:.2f}°  "
            f"Toro r {row['r_toro_fixed_theta']:.3f}→{row['r_toro_random_theta']:.3f}  "
            f"Passeri r {row['r_passeri_fixed_theta']:.3f}→{row['r_passeri_random_theta']:.3f}"
        )
    stored.close()

    frame = pd.DataFrame(rows)
    r_fixed = np.concatenate(rep_scores["toro_fixed"])
    r_rand = np.concatenate(rep_scores["toro_random"])
    p_fixed = np.concatenate(rep_scores["passeri_fixed"])
    p_rand = np.concatenate(rep_scores["passeri_random"])
    print(
        f"Toro replicate median r  fixed-θ={np.nanmedian(r_fixed):.4f}  "
        f"random-θ={np.nanmedian(r_rand):.4f}  "
        f"Δ={np.nanmedian(r_rand - r_fixed):+.4f}"
    )
    print(
        f"Passeri replicate median r  fixed-θ={np.nanmedian(p_fixed):.4f}  "
        f"random-θ={np.nanmedian(p_rand):.4f}  "
        f"Δ={np.nanmedian(p_rand - p_fixed):+.4f}"
    )
    print(
        f"Toro law A vs stored dip geomean: "
        f"median rel L1={frame['rel_l1_toro_vs_stored_dip'].median():.6f}"
    )

    med = float(frame["abs_theta"].median())
    for name, part in (
        ("shallow", frame[frame["abs_theta"] <= med]),
        ("steep", frame[frame["abs_theta"] > med]),
    ):
        print(
            f"  {name} n={len(part)}  "
            f"median Δr Toro={part['delta_r_toro'].median():+.4f}  "
            f"Passeri={part['delta_r_passeri'].median():+.4f}  "
            f"rel L1 Toro={part['rel_l1_toro'].median():.4f}  "
            f"Passeri={part['rel_l1_passeri'].median():.4f}"
        )

    frame.to_csv(TF_PATH, index=False)
    print(f"Wrote {TF_PATH}")
    return frame


def main() -> None:
    phys_list, runs = load_manifest(MANIFEST_PATH)
    print(f"Loaded {len(phys_list)} Sobol points from {MANIFEST_PATH}")
    _check_sampler_matches(_toro_config(phys_list[0], "dip"))

    print("Depth uniformity")
    table = depth_table(phys_list)
    print(
        f"  max KS toro={table['ks_toro'].max():.4f}  "
        f"passeri={table['ks_passeri'].max():.4f}  "
        f"max clip={table['clipped_fraction_toro'].max():.4f}"
    )

    picked = pick_points(phys_list)
    depth_samples: dict[int, dict[str, np.ndarray]] = {}
    for _label, phys in picked:
        depth_samples[phys.sobol_id] = {}
        for method, cfg in (
            ("toro", _toro_config(phys, "dip")),
            ("passeri", _passeri_config(phys, "dip")),
        ):
            depths, _clip = _draw_depths(
                cfg, N_DEPTH, seed=SEED + 17 * phys.sobol_id + (0 if method == "toro" else 1)
            )
            depth_samples[phys.sobol_id][method] = depths
    pooled, mix = _pooled_and_mix(phys_list)
    steep = picked[-1][1]
    plot_figure(picked, depth_samples, pooled, mix, steep)

    import h5py

    with h5py.File(TORO_OUT / "tf_2d_center.h5", "r") as f:
        freq = np.asarray(f["freq"][:], dtype=np.float64)
        tf_2d = np.asarray(f["tf_center"][:], dtype=np.float64)
    print("Transfer functions, fixed θ vs θ~Unif[-3, 3]")
    score_tf(phys_list, runs, tf_2d, freq)


if __name__ == "__main__":
    main()
