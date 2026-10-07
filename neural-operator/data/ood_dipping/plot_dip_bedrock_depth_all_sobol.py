"""Toro / Passeri lognormal vs dip depth over all ood_dipping Sobol physics.

Uses the 32 unique physics rows from ``manifest.csv`` (Vs1, H, CoV, Vs2,
dip_angle_deg). For each Sobol point the dip interface is the campaign
geometry at a random column:

    H + x tan(θ_sobol),   x ~ Unif[-250, 250] m

so θ is fixed to that point's Sobol angle (not redrawn). Lognormal depth
still draws around each point's H with σ_ln = 0.05.

Writes ``figures/toro_passeri_dip_depth_all_sobol.png``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from seiskit.profile_randomization import ProfileRandomizationConfig
from plot_dip_bedrock_depth import _one_layer_column
from seiskit.profile_randomization.nhpp import _sample_interface_depth

THIS_DIR = Path(__file__).resolve().parent
FIG_DIR = THIS_DIR / "figures"
OUT_PATH = FIG_DIR / "toro_passeri_dip_depth_all_sobol.png"
MANIFEST_PATH = THIS_DIR / "manifest.csv"

DZ = 0.5
DIP_HALF_SPAN = 250.0
N_DEPTH_PER_POINT = 400
N_PROFILE_PER_POINT = 4
SEED = 0


def _load_unique_physics(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open()))
    by_id: dict[int, dict] = {}
    for row in rows:
        sid = int(row["sobol_id"])
        if sid in by_id:
            continue
        by_id[sid] = {
            "sobol_id": sid,
            "Vs1": float(row["Vs1"]),
            "Vs2": float(row["Vs2"]),
            "H": float(row["H_discretized"]),
            "CoV": float(row["CoV"]),
            "dip_angle_deg": float(row["dip_angle_deg"]),
            "bedrock_thickness": float(row["bedrock_thickness_discretized"]),
            "Lz": float(row["Lz_discretized"]),
        }
    return [by_id[k] for k in sorted(by_id)]


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

    physics = _load_unique_physics(MANIFEST_PATH)
    n_phys = len(physics)
    print(f"Loaded {n_phys} unique Sobol physics from {MANIFEST_PATH}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    max_shift = DIP_HALF_SPAN * np.tan(np.radians(3.0))

    depths_ln: list[float] = []
    depths_dip: list[float] = []
    delta_ln: list[float] = []
    delta_dip: list[float] = []

    # Profiles: list of (vs, depth_axis, H) for overlay.
    toro_ln_prof: list[tuple[np.ndarray, np.ndarray, float]] = []
    toro_dip_prof: list[tuple[np.ndarray, np.ndarray, float]] = []
    pass_ln_prof: list[tuple[np.ndarray, np.ndarray, float]] = []
    pass_dip_prof: list[tuple[np.ndarray, np.ndarray, float]] = []

    rng = np.random.default_rng(SEED)
    for phys in physics:
        H = float(phys["H"])
        cfg_ln = _cfg_for_point(phys, "lognormal")
        cfg_dip = _cfg_for_point(phys, "dip")

        for _ in range(N_DEPTH_PER_POINT):
            d_ln = _sample_interface_depth(cfg_ln, rng)
            d_dip = _sample_interface_depth(cfg_dip, rng)
            depths_ln.append(d_ln)
            depths_dip.append(d_dip)
            delta_ln.append(d_ln - H)
            delta_dip.append(d_dip - H)

        for k in range(N_PROFILE_PER_POINT):
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
                store_t.append((col_t, z_t, H))
                store_p.append((col_p, z_p, H))

    depths_ln = np.asarray(depths_ln)
    depths_dip = np.asarray(depths_dip)
    delta_ln = np.asarray(delta_ln)
    delta_dip = np.asarray(delta_dip)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.4), constrained_layout=True)

    # --- depth: ΔH = interface − H (comparable across Sobol H values) ---
    ax = axes[0]
    bins = np.linspace(-max_shift - 1.0, max_shift + 1.0, 51)
    ax.hist(
        delta_ln,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.45,
        color="#0072B2",
        label=r"lognormal ($\sigma_{\ln}=0.05$)",
    )
    ax.hist(
        delta_dip,
        bins=bins,
        density=True,
        histtype="stepfilled",
        alpha=0.45,
        color="#D55E00",
        label=r"$x\tan\theta_{\mathrm{Sobol}}$",
    )
    ax.axvline(0.0, color="0.2", ls="--", lw=1.2, label=r"$\Delta H=0$")
    ax.set_xlabel(r"Interface depth $-$ $H$ (m)")
    ax.set_ylabel("Density")
    ax.set_title(rf"Bedrock-depth law ({n_phys} Sobol $H$, $\theta$)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)

    def _plot_profiles(ax, store_ln, store_dip, title: str):
        for vs, z, _H in store_ln:
            ax.step(vs, z, where="pre", color="#0072B2", alpha=0.22, lw=0.7)
        for vs, z, _H in store_dip:
            ax.step(vs, z, where="pre", color="#D55E00", alpha=0.22, lw=0.7)
        # Mark each Sobol H
        for phys in physics:
            ax.axhline(phys["H"], color="0.55", ls=":", lw=0.4, alpha=0.5)
        ax.set_xlabel(r"$V_s$ (m/s)")
        ax.set_title(title)
        z_max = max(float(phys["Lz"]) for phys in physics)
        ax.set_ylim(z_max, 0)
        ax.set_xlim(50, 1800)
        ax.grid(True, alpha=0.25)

    _plot_profiles(
        axes[1],
        toro_ln_prof,
        toro_dip_prof,
        rf"Toro ({n_phys}×{N_PROFILE_PER_POINT} profiles)",
    )
    axes[1].set_ylabel("Depth (m)")
    axes[1].plot([], [], color="#0072B2", lw=2, label="lognormal depth")
    axes[1].plot([], [], color="#D55E00", lw=2, label="dip depth")
    axes[1].legend(fontsize=8, loc="lower right")

    _plot_profiles(
        axes[2],
        pass_ln_prof,
        pass_dip_prof,
        rf"Passeri ({n_phys}×{N_PROFILE_PER_POINT} profiles)",
    )

    H_vals = np.array([p["H"] for p in physics])
    th_vals = np.array([p["dip_angle_deg"] for p in physics])
    fig.suptitle(
        rf"Toro / Passeri over ood_dipping Sobol physics  "
        rf"($n={n_phys}$, $H\in[{H_vals.min():.0f},{H_vals.max():.0f}]$ m, "
        rf"$\theta\in[{th_vals.min():+.1f},{th_vals.max():+.1f}]^\circ$)",
        fontsize=11,
    )
    fig.savefig(OUT_PATH, dpi=160)
    print(f"Saved {OUT_PATH}")
    print(
        f"  ΔH lognormal: mean={delta_ln.mean():+.3f}  std={delta_ln.std():.2f}  "
        f"range=[{delta_ln.min():+.2f}, {delta_ln.max():+.2f}]"
    )
    print(
        f"  ΔH dip:       mean={delta_dip.mean():+.3f}  std={delta_dip.std():.2f}  "
        f"range=[{delta_dip.min():+.2f}, {delta_dip.max():+.2f}]"
    )
    print(f"  interface lognormal: [{depths_ln.min():.1f}, {depths_ln.max():.1f}] m")
    print(f"  interface dip:       [{depths_dip.min():.1f}, {depths_dip.max():.1f}] m")


if __name__ == "__main__":
    main()
