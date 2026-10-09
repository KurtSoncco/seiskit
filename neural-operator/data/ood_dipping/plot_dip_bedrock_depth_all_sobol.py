"""Toro / Passeri lognormal vs dip depth at one fixed Sobol angle.

The depth histogram is that point only. θ is locked to its dip angle and
x ~ Unif[-250, 250] m, so the dip depth is uniform on H ± 250|tan θ|.
Angles are not pooled.

Writes ``figures/toro_passeri_dip_depth_all_sobol.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from seiskit.profile_randomization import ProfileRandomizationConfig
from plot_dip_bedrock_depth import _one_layer_column, median_abs_theta_point
from seiskit.profile_randomization.nhpp import _sample_interface_depth

THIS_DIR = Path(__file__).resolve().parent
FIG_DIR = THIS_DIR / "figures"
OUT_PATH = FIG_DIR / "toro_passeri_dip_depth_all_sobol.png"
MANIFEST_PATH = THIS_DIR / "manifest.csv"

DZ = 0.5
DIP_HALF_SPAN = 250.0
N_DEPTH = 8000
N_PROFILE = 60
SEED = 0


def _cfg_for_point(phys: dict, model: str) -> ProfileRandomizationConfig:
    cov = float(phys["CoV"])
    theta = float(phys["dip_angle_deg"])
    # Fix θ to the Sobol angle so only x is random under the dip law.
    return ProfileRandomizationConfig(
        vs_mean=float(phys["Vs1"]),
        thickness=float(phys["H"]),
        bedrock_thickness=float(phys["bedrock_thickness"]),
        dz=DZ,
        vs_bedrock=float(phys["Vs2"]),
        cov=cov,
        sigma_ln_vs=cov,
        sigma_ln_tts=cov,
        randomize_layer_thickness=False,
        randomize_bedrock_depth=True,
        vary_bedrock_vs=False,
        use_full_model=True,
        bedrock_depth_model=model,
        dip_angle_min_deg=theta,
        dip_angle_max_deg=theta,
        dip_half_span_m=DIP_HALF_SPAN,
    )


def main() -> None:
    if not MANIFEST_PATH.is_file():
        raise SystemExit(f"Missing manifest: {MANIFEST_PATH}")

    point = median_abs_theta_point(MANIFEST_PATH)
    phys = {
        "sobol_id": point["sobol_id"],
        "Vs1": point["Vs1"],
        "Vs2": point["Vs2"],
        "H": point["H"],
        "CoV": point["CoV"],
        "dip_angle_deg": point["dip_angle_deg"],
        "bedrock_thickness": point["bedrock_thickness"],
        "Lz": point["H"] + point["bedrock_thickness"],
    }
    H = float(phys["H"])
    theta = float(phys["dip_angle_deg"])
    half = DIP_HALF_SPAN * abs(np.tan(np.radians(theta)))
    print(
        f"Fixed Sobol {phys['sobol_id']}: H={H:.1f} m, θ={theta:+.3f}°, half={half:.2f} m"
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg_ln = _cfg_for_point(phys, "lognormal")
    cfg_dip = _cfg_for_point(phys, "dip")
    rng = np.random.default_rng(SEED)
    depths_ln = np.array([_sample_interface_depth(cfg_ln, rng) for _ in range(N_DEPTH)])
    depths_dip = np.array([_sample_interface_depth(cfg_dip, rng) for _ in range(N_DEPTH)])

    toro_ln_prof: list[tuple[np.ndarray, np.ndarray]] = []
    toro_dip_prof: list[tuple[np.ndarray, np.ndarray]] = []
    pass_ln_prof: list[tuple[np.ndarray, np.ndarray]] = []
    pass_dip_prof: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(N_PROFILE):
        seed_k = SEED + 1000 * phys["sobol_id"] + k
        for model, store_t, store_p in (
            ("lognormal", toro_ln_prof, pass_ln_prof),
            ("dip", toro_dip_prof, pass_dip_prof),
        ):
            cfg = _cfg_for_point(phys, model)
            col_t, _iface_t = _one_layer_column(cfg, "toro", np.random.default_rng(seed_k))
            col_p, _iface_p = _one_layer_column(
                cfg, "passeri", np.random.default_rng(seed_k + 17)
            )
            z_t = (np.arange(len(col_t)) + 0.5) * float(cfg.dz)
            z_p = (np.arange(len(col_p)) + 0.5) * float(cfg.dz)
            store_t.append((col_t, z_t))
            store_p.append((col_p, z_p))

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.4), constrained_layout=True)

    ax = axes[0]
    span = max(half, 0.25 * H) + 1.0
    bins = np.linspace(H - span, H + span, 41)
    ax.hist(
        depths_ln,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.45,
        color="#0072B2",
        label=r"lognormal ($\sigma_{\ln}=0.05$)",
    )
    ax.hist(
        depths_dip,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.45,
        color="#D55E00",
        label=r"$H + x\tan\theta$",
    )
    if half > 0:
        height = 1.0 / (2.0 * half)
        ax.plot(
            [H - half, H - half, H + half, H + half],
            [0.0, height, height, 0.0],
            color="0.1",
            lw=1.3,
            label="uniform",
        )
    ax.axvline(H, color="0.2", ls="--", lw=1.2, label=rf"$H={H:.0f}$ m")
    ax.set_xlabel("Interface depth (m)")
    ax.set_ylabel("Density")
    ax.set_title(rf"Bedrock-depth law (Sobol {phys['sobol_id']})")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)

    def _plot_profiles(ax, store_ln, store_dip, title: str):
        for vs, z in store_ln:
            ax.step(vs, z, where="pre", color="#0072B2", alpha=0.35, lw=0.8)
        for vs, z in store_dip:
            ax.step(vs, z, where="pre", color="#D55E00", alpha=0.35, lw=0.8)
        ax.axhline(H, color="0.2", ls="--", lw=1.0)
        ax.axhspan(H - half, H + half, color="#D55E00", alpha=0.08, zorder=0)
        ax.set_xlabel(r"$V_s$ (m/s)")
        ax.set_title(title)
        z_max = max(float(z[-1]) for _vs, z in store_ln + store_dip)
        ax.set_ylim(z_max + DZ, 0)
        ax.set_xlim(50, 1800)
        ax.grid(True, alpha=0.25)

    _plot_profiles(axes[1], toro_ln_prof, toro_dip_prof, "Toro (Vs randomization)")
    axes[1].set_ylabel("Depth (m)")
    axes[1].plot([], [], color="#0072B2", lw=2, label="lognormal depth")
    axes[1].plot([], [], color="#D55E00", lw=2, label="dip depth")
    axes[1].legend(fontsize=8, loc="lower right")

    _plot_profiles(axes[2], pass_ln_prof, pass_dip_prof, "Passeri (tts randomization)")

    fig.suptitle(
        rf"Toro / Passeri, Sobol {phys['sobol_id']}  "
        rf"($H={H:.0f}$ m, $\theta={theta:+.2f}^\circ$ fixed, "
        rf"$x\sim\mathrm{{Unif}}[-250,250]$ m)",
        fontsize=11,
    )
    fig.savefig(OUT_PATH, dpi=160)
    print(f"Saved {OUT_PATH}")
    print(
        f"  lognormal depth: mean={depths_ln.mean():.2f}  std={depths_ln.std():.2f}  "
        f"range=[{depths_ln.min():.2f}, {depths_ln.max():.2f}]"
    )
    print(
        f"  dip depth:       mean={depths_dip.mean():.2f}  std={depths_dip.std():.2f}  "
        f"range=[{depths_dip.min():.2f}, {depths_dip.max():.2f}]"
    )


if __name__ == "__main__":
    main()
