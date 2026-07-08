"""Compatibility facade for profile randomization utilities.

Implementation has been split into `seiskit/profile_randomization/` modules so each
component (Toro, Passeri, NHPP, simplified path, analysis) can be reviewed and tested
independently.
"""

from __future__ import annotations

import numpy as np

from .profile_randomization import (
    ProfileRandomizationConfig,
    RandomizedProfile,
    acf_rmse,
    build_base_case_profile,
    build_layered_profile,
    calibrate_rho_to_target_acf,
    generate_nhpp_layer_thicknesses,
    generate_passeri_profile,
    generate_toro_profile,
    generate_tts_randomized_profile,
    generate_tts_randomized_profile_full,
    generate_vs_randomized_profile,
    generate_vs_randomized_profile_full,
    geological_layer_starts,
    profile_cov,
    profile_to_opensees_column,
    toro_adjacent_correlation,
    toro_rho,
    vertical_acf_from_2d_field,
    vertical_acf_ln_vs,
)

__all__ = [
    "ProfileRandomizationConfig",
    "RandomizedProfile",
    "acf_rmse",
    "build_base_case_profile",
    "build_layered_profile",
    "calibrate_rho_to_target_acf",
    "generate_nhpp_layer_thicknesses",
    "generate_passeri_profile",
    "generate_toro_profile",
    "generate_tts_randomized_profile",
    "generate_tts_randomized_profile_full",
    "generate_vs_randomized_profile",
    "generate_vs_randomized_profile_full",
    "geological_layer_starts",
    "profile_cov",
    "profile_to_opensees_column",
    "toro_adjacent_correlation",
    "toro_rho",
    "vertical_acf_from_2d_field",
    "vertical_acf_ln_vs",
]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    config = ProfileRandomizationConfig(
        vs_mean=230.0,
        thickness=15.0,
        bedrock_thickness=10.0,
        dz=0.5,
        vs_bedrock=1500.0,
        cov=0.2,
        use_full_model=True,
    )
    base_vs = build_base_case_profile(config)
    n_seeds = 50
    depth = (np.arange(len(base_vs)) + 0.5) * config.dz

    toro = np.zeros((n_seeds, len(base_vs)))
    passeri = np.zeros((n_seeds, len(base_vs)))
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        toro[seed] = generate_vs_randomized_profile(config, rng)
        passeri[seed] = generate_tts_randomized_profile(config, rng)

    fig, axes = plt.subplots(1, 2, figsize=(9, 6), sharey=True)
    for ax, (title, profiles) in zip(
        axes,
        (("Toro (full)", toro), ("Passeri (full)", passeri)),
    ):
        ax.step(base_vs, depth, "k-", lw=2, label="Base", where="pre")
        for i in range(n_seeds):
            ax.step(profiles[i], depth, "-", alpha=0.55, lw=1.0, where="pre")
        ax.axhline(config.thickness, color="0.45", ls=":", lw=1.0)
        ax.set_xlabel(r"$V_s$ (m/s)")
        ax.set_title(title)
        ax.set_ylim(depth[-1] + config.dz, 0)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Depth (m)")
    fig.suptitle("Full Toro / Passeri - NHPP layers + bedrock depth/Vs", fontsize=11)
    fig.tight_layout()
    out = "profile_randomization_check.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.show()

    n_layers_toro = np.zeros(n_seeds)
    n_layers_passeri = np.zeros(n_seeds)
    for i in range(n_seeds):
        n_layers_toro[i] = len(geological_layer_starts(toro[i], rel_tol=config.layer_jump_rel))
        n_layers_passeri[i] = len(
            geological_layer_starts(passeri[i], rel_tol=config.layer_jump_rel)
        )
    print(f"Number of layers in Toro: {n_layers_toro}")
    print(f"Number of layers in Passeri: {n_layers_passeri}")
