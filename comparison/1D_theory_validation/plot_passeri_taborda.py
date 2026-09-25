"""Passeri tts-only (fixed H, no NHPP): Taborda–Bielak ξ_Q vs Darendeli Dmin (3 Hz), plus Pearson box plot."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from seiskit.plot_config import apply_style
from seiskit.profile_randomization import (
    ProfileRandomizationConfig,
    build_base_case_profile,
    generate_passeri_profile,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "rayleigh_equivalent"
N_SWARM = 80
N_PEARSON = 200
COV = 0.20
SIGMA_LN_TTS = COV  # same rule as Toro: use specified CoV as the ln-sigma


def load_toro_mod():
    spec = importlib.util.spec_from_file_location("plot_toro_taborda", ROOT / "plot_toro_taborda.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["plot_toro_taborda"] = mod
    spec.loader.exec_module(mod)
    return mod


def pearson_ln(a: np.ndarray, b: np.ndarray) -> float:
    x = np.log(np.clip(np.asarray(a, float), 1e-12, None))
    y = np.log(np.clip(np.asarray(b, float), 1e-12, None))
    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def travel_time_profile(vs_depth: np.ndarray, dz: float) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative one-way t* from the free surface, including (0, 0)."""
    vs = np.asarray(vs_depth, dtype=float).ravel()
    dtt = dz / np.clip(vs, 1e-6, None)
    tts = np.concatenate([[0.0], np.cumsum(dtt)])
    z = np.concatenate([[0.0], (np.arange(len(vs)) + 1) * dz])
    return tts, z


def summarize(name: str, r: np.ndarray) -> None:
    r = r[np.isfinite(r)]
    print(
        f"{name:12s}  n={len(r)}  mean={np.mean(r):.3f}  "
        f"p50={np.median(r):.3f}  p16–p84={np.percentile(r,16):.3f}–{np.percentile(r,84):.3f}  "
        f"min={np.min(r):.3f}  max={np.max(r):.3f}"
    )


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    t = load_toro_mod()
    vs_mean = t.travel_time_vs()
    cfg = ProfileRandomizationConfig(
        vs_mean=vs_mean,
        thickness=t.H_SOIL,
        dz=t.DZ,
        cov=COV,
        vs_bedrock=t.VS_ROCK,
        bedrock_thickness=t.H_ROCK,
        sigma_ln_vs=COV,
        sigma_ln_tts=SIGMA_LN_TTS,
        use_full_model=True,
        randomize_layer_thickness=False,
        randomize_bedrock_depth=False,
        vary_bedrock_vs=False,
    )
    freq = np.logspace(np.log10(0.2), np.log10(12.0), 700)
    base = build_base_case_profile(cfg)
    n_soil = max(1, int(round(t.H_SOIL / t.DZ)))
    af_base_q = t.tf_profile(base, n_soil, t.DZ, freq, t.BASE_Q)
    af_base_d = t.tf_profile(base, n_soil, t.DZ, freq, t.BASE_DMIN)

    n = N_PEARSON
    af_q = np.empty((n, len(freq)))
    af_d = np.empty((n, len(freq)))
    profiles = []
    vs1 = np.empty(n)
    tts = np.empty(n)
    n_layers = []
    interfaces = []

    for k in range(n):
        rng = np.random.default_rng(t.SEED0 + k)
        prof = generate_passeri_profile(cfg, rng)
        profiles.append(prof)
        ns = int(prof.n_soil_samples)
        vs1[k] = float(prof.vs_depth[0])
        tts[k] = float(prof.interface_depth) / vs1[k]
        n_layers.append(len({round(float(v), 3) for v in prof.vs_depth[:ns]}))
        interfaces.append(float(prof.interface_depth))
        af_q[k] = t.tf_profile(prof.vs_depth, ns, t.DZ, freq, t.BASE_Q)
        af_d[k] = t.tf_profile(prof.vs_depth, ns, t.DZ, freq, t.BASE_DMIN)

    print(
        f"Passeri tts-only  n_swarm={N_SWARM}  n_pearson={N_PEARSON}  "
        f"sigma_ln_tts=CoV={SIGMA_LN_TTS:.3f}  vs_mean={vs_mean:.1f} m/s"
    )
    print(
        f"  σ_ln(tts)={float(np.std(np.log(tts), ddof=1)):.3f}  "
        f"σ_ln(Vs1)={float(np.std(np.log(vs1), ddof=1)):.3f}  "
        f"CoV(Vs1)={float(np.std(vs1, ddof=1)/np.mean(vs1)):.3f}  "
        f"(target {SIGMA_LN_TTS:.3f})\n"
        f"  soil layers={set(n_layers)}  H={sorted(set(np.round(interfaces, 2)))}"
    )
    peaks_q = np.max(af_q[:N_SWARM], axis=1)
    peaks_d = np.max(af_d[:N_SWARM], axis=1)
    print(
        f"  A1 ξ_Q     p50={np.median(peaks_q):.1f}  "
        f"p16–p84={np.percentile(peaks_q,16):.1f}–{np.percentile(peaks_q,84):.1f}\n"
        f"  A1 Dmin    p50={np.median(peaks_d):.1f}  "
        f"p16–p84={np.percentile(peaks_d,16):.1f}–{np.percentile(peaks_d,84):.1f}"
    )

    r_q = np.array([pearson_ln(af_q[k], af_base_q) for k in range(n)])
    r_d = np.array([pearson_ln(af_d[k], af_base_d) for k in range(n)])
    summarize("ξ_Q", r_q)
    summarize("Darendeli Dmin", r_d)
    csv_path = OUT / "passeri_vs_1d_pearson.csv"
    np.savetxt(
        csv_path,
        np.column_stack([np.arange(n), vs1, tts, r_q, r_d]),
        delimiter=",",
        header="seed,vs1,tts,r_xiQ,r_0p25xiQ",
        comments="",
    )
    print(csv_path)

    # --- swarm figure (same layout as Toro Vs-only) ---
    swarm = slice(0, N_SWARM)
    geo_q = np.exp(np.mean(np.log(np.clip(af_q[swarm], 1e-12, None)), axis=0))
    geo_d = np.exp(np.mean(np.log(np.clip(af_d[swarm], 1e-12, None)), axis=0))
    p16_q, p84_q = np.percentile(af_q[swarm], [16, 84], axis=0)
    p16_d, p84_d = np.percentile(af_d[swarm], [16, 84], axis=0)

    fig, axes = plt.subplots(1, 4, figsize=(16.6, 5.3), constrained_layout=True)
    ax = axes[0]
    for prof in profiles[:N_SWARM]:
        vs = prof.vs_depth
        z = (np.arange(len(vs)) + 0.5) * t.DZ
        ax.plot(vs, z, color="0.75", lw=0.7, alpha=0.85)
    z_b = (np.arange(len(base)) + 0.5) * t.DZ
    ax.plot(base, z_b, color="#0072B2", lw=2.0, label="median column")
    ax.set_ylim(t.H_SOIL + t.H_ROCK, 0.0)
    ax.set_xlim(0.0, 2800.0)
    ax.set_xlabel(r"$V_s$ (m/s)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(r"$V_s$ profiles")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1]
    tts_iface = []
    for prof in profiles[:N_SWARM]:
        tts_z, z_tt = travel_time_profile(prof.vs_depth, t.DZ)
        ax.plot(tts_z, z_tt, color="0.75", lw=0.7, alpha=0.85)
        n_s = int(prof.n_soil_samples)
        tts_iface.append(float(tts_z[n_s]))
    tts_base, z_tt_b = travel_time_profile(base, t.DZ)
    ax.plot(tts_base, z_tt_b, color="#0072B2", lw=2.0, label="median column")
    ax.axhline(t.H_SOIL, color="0.5", ls=":", lw=0.8)
    ax.set_ylim(t.H_SOIL + t.H_ROCK, 0.0)
    ax.set_xlabel(r"Travel time $t^*$ (s)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(fr"$t^*(z)$  ($\sigma_{{\ln t}}$=CoV={SIGMA_LN_TTS:.2f})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    print(
        f"  interface t*  p50={np.median(tts_iface):.3f} s  "
        f"range {min(tts_iface):.3f}–{max(tts_iface):.3f}  "
        f"(base {float(tts_base[n_soil]):.3f} s)"
    )

    t.tf_panel(axes[2], freq, af_q[swarm], geo_q, p16_q, p84_q, af_base_q, r"Taborda–Bielak $\xi_Q=1/(2Q)$", "#0072B2")
    t.tf_panel(
        axes[3],
        freq,
        af_d[swarm],
        geo_d,
        p16_d,
        p84_d,
        af_base_d,
        r"Darendeli $D_{\min}$ (3 Hz)",
        "#D55E00",
    )
    fig.suptitle(
        fr"Passeri travel-time only (fixed $H$, one soil layer)  "
        fr"$V_{{s,\mathrm{{med}}}}$={vs_mean:.0f} m/s,  $n$={N_SWARM}  |  "
        fr"ensemble $\sigma_{{\ln t}}$={float(np.std(np.log(tts[:N_SWARM]), ddof=1)):.3f}  "
        fr"(target {SIGMA_LN_TTS:.3f})",
        fontsize=11,
    )
    out_swarm = OUT / "af_passeri_taborda_vs.png"
    fig.savefig(out_swarm, dpi=160)
    plt.close(fig)
    print(out_swarm)

    # --- Pearson box plot ---
    fig, ax = plt.subplots(figsize=(5.6, 4.8), constrained_layout=True)
    data = [r_q[np.isfinite(r_q)], r_d[np.isfinite(r_d)]]
    bp = ax.boxplot(
        data,
        tick_labels=[r"Taborda–Bielak $\xi_Q$", r"Darendeli $D_{\min}$"],
        patch_artist=True,
        widths=0.55,
        medianprops=dict(color="0.15", lw=1.6),
        whiskerprops=dict(color="0.25"),
        capprops=dict(color="0.25"),
        flierprops=dict(marker="o", ms=3.5, markerfacecolor="0.5", markeredgecolor="none", alpha=0.7),
    )
    colors = ["#0072B2", "#D55E00"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
        patch.set_edgecolor("0.2")
    ax.set_ylim(-0.15, 1.05)
    ax.set_ylabel(r"Pearson $r(\ln|AF|_{\mathrm{Passeri}},\ \ln|AF|_{1\mathrm{D}})$")
    ax.set_title(fr"Passeri vs 1D base  ($n$={N_PEARSON}, fixed $H$)")
    ax.grid(True, axis="y", alpha=0.3)
    for i, r in enumerate(data, start=1):
        ax.annotate(
            fr"p50={np.median(r):.2f}",
            xy=(i, np.median(r)),
            xytext=(i + 0.28, np.median(r)),
            fontsize=8,
            va="center",
            color="0.2",
        )
    out_box = OUT / "passeri_vs_1d_pearson_box.png"
    fig.savefig(out_box, dpi=160)
    plt.close(fig)
    print(out_box)


if __name__ == "__main__":
    main()
