"""Compare Toro / Passeri under lognormal vs dip bedrock-depth laws.

Default depth remains lognormal. For the dipping case, interface depth is
sampled as H + x tan(θ) with θ ~ Unif[-3°, 3°] and x ~ Unif[-250, 250] m,
matching Box ood_dipping geometry.

Writes ``figures/toro_passeri_dip_depth.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from seiskit.profile_randomization import (
    ProfileRandomizationConfig,
    generate_passeri_profile,
    generate_toro_profile,
)

THIS_DIR = Path(__file__).resolve().parent
FIG_DIR = THIS_DIR / "figures"
OUT_PATH = FIG_DIR / "toro_passeri_dip_depth.png"

H = 40.0
BEDROCK = 30.0
DZ = 0.5
VS1 = 230.0
VS2 = 1500.0
COV = 0.20
N_DEPTH = 8000
N_PROFILE = 60
SEED = 0


def _base_kwargs() -> dict:
    return dict(
        vs_mean=VS1,
        thickness=H,
        bedrock_thickness=BEDROCK,
        dz=DZ,
        vs_bedrock=VS2,
        cov=COV,
        sigma_ln_vs=COV,
        sigma_ln_tts=COV,
        randomize_layer_thickness=False,
        randomize_bedrock_depth=True,
        vary_bedrock_vs=False,
        use_full_model=True,
    )


def _config(model: str) -> ProfileRandomizationConfig:
    return ProfileRandomizationConfig(bedrock_depth_model=model, **_base_kwargs())


def _ensemble(model: str, method: str, n: int, seed: int):
    cfg = _config(model)
    gen = generate_toro_profile if method == "toro" else generate_passeri_profile
    rng = np.random.default_rng(seed)
    profiles = []
    depths = []
    for _ in range(n):
        prof = gen(cfg, rng)
        profiles.append(prof.vs_depth)
        depths.append(prof.interface_depth)
    return np.asarray(profiles), np.asarray(depths), cfg


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    max_shift = 250.0 * np.tan(np.radians(3.0))

    # Shared seeds across depth models so profile differences are depth-driven.
    toro_ln, d_toro_ln, cfg_ln = _ensemble("lognormal", "toro", N_PROFILE, SEED)
    toro_dip, d_toro_dip, _ = _ensemble("dip", "toro", N_PROFILE, SEED)
    pass_ln, d_pass_ln, _ = _ensemble("lognormal", "passeri", N_PROFILE, SEED + 1)
    pass_dip, d_pass_dip, _ = _ensemble("dip", "passeri", N_PROFILE, SEED + 1)

    # Larger sample for the depth histogram only.
    rng = np.random.default_rng(SEED)
    from seiskit.profile_randomization.nhpp import _sample_interface_depth

    depths_ln = np.array([_sample_interface_depth(cfg_ln, rng) for _ in range(N_DEPTH)])
    cfg_dip = _config("dip")
    depths_dip = np.array([_sample_interface_depth(cfg_dip, rng) for _ in range(N_DEPTH)])

    depth_axis = (np.arange(toro_ln.shape[1]) + 0.5) * DZ

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.2), constrained_layout=True)

    ax = axes[0]
    bins = np.linspace(H - max_shift - 1.0, H + max_shift + 1.0, 41)
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
    ax.axvline(H, color="0.2", ls="--", lw=1.2, label=rf"$H={H:.0f}$ m")
    ax.set_xlabel("Interface depth (m)")
    ax.set_ylabel("Density")
    ax.set_title("Bedrock-depth law")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)

    def _plot_profiles(ax, profiles_ln, profiles_dip, title: str):
        for i in range(len(profiles_ln)):
            ax.step(
                profiles_ln[i],
                depth_axis,
                where="pre",
                color="#0072B2",
                alpha=0.35,
                lw=0.8,
            )
            ax.step(
                profiles_dip[i],
                depth_axis,
                where="pre",
                color="#D55E00",
                alpha=0.35,
                lw=0.8,
            )
        ax.axhline(H, color="0.2", ls="--", lw=1.0)
        ax.axhspan(H - max_shift, H + max_shift, color="#D55E00", alpha=0.08, zorder=0)
        ax.set_xlabel(r"$V_s$ (m/s)")
        ax.set_title(title)
        ax.set_ylim(depth_axis[-1] + DZ, 0)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(50, 1800)

    _plot_profiles(axes[1], toro_ln, toro_dip, "Toro (Vs randomization)")
    axes[1].set_ylabel("Depth (m)")
    # Legend proxies
    axes[1].plot([], [], color="#0072B2", lw=2, label="lognormal depth")
    axes[1].plot([], [], color="#D55E00", lw=2, label="dip depth")
    axes[1].legend(fontsize=8, loc="lower right")

    _plot_profiles(axes[2], pass_ln, pass_dip, "Passeri (tts randomization)")

    fig.suptitle(
        rf"Toro / Passeri with dip bedrock depth  "
        rf"($H={H:.0f}$ m, $\theta\sim\mathrm{{Unif}}[-3^\circ,3^\circ]$, "
        rf"$x\sim\mathrm{{Unif}}[-250,250]$ m)",
        fontsize=11,
    )
    fig.savefig(OUT_PATH, dpi=160)
    print(f"Saved {OUT_PATH}")
    print(
        f"  lognormal depth: mean={depths_ln.mean():.2f}  "
        f"std={depths_ln.std():.2f}  "
        f"range=[{depths_ln.min():.2f}, {depths_ln.max():.2f}]"
    )
    print(
        f"  dip depth:       mean={depths_dip.mean():.2f}  "
        f"std={depths_dip.std():.2f}  "
        f"range=[{depths_dip.min():.2f}, {depths_dip.max():.2f}]"
    )
    print(
        f"  toro iface (ln/dip)  mean={d_toro_ln.mean():.2f}/{d_toro_dip.mean():.2f}  "
        f"std={d_toro_ln.std():.2f}/{d_toro_dip.std():.2f}"
    )
    print(
        f"  passeri iface (ln/dip) mean={d_pass_ln.mean():.2f}/{d_pass_dip.mean():.2f}  "
        f"std={d_pass_ln.std():.2f}/{d_pass_dip.std():.2f}"
    )


if __name__ == "__main__":
    main()
