"""Publication plots for Response_Variability comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze_response import collect_rows, reference_curves
from manifest import METHODS, THICKNESS, VS2, active_bc_width, active_dz, active_lx_var

from seiskit.intensity_measures import sigma_ln
from seiskit.plot_config import apply_style, get_crameri_cmap
from seiskit.profile_randomization import (
    ProfileRandomizationConfig,
    generate_tts_randomized_profile,
    generate_vs_randomized_profile,
)

apply_style()

METHOD_LABELS = {
    "grf_2d": "2D GRF (reference)",
    "delatorre_2d": "de la Torre 2D",
    "hallal_vs": "Hallal VsRand",
    "hallal_tts": "Hallal ttsRand",
    "hallal_dmin": "Hallal Dmin",
}


def _method_curves(df: pd.DataFrame, method: str, vs1: float, motion_id: str) -> dict | None:
    sub = df[(df["method"] == method) & (df["vs1"] == vs1) & (df["motion_id"] == motion_id)]
    if sub.empty:
        return None
    sa_stack = np.vstack(sub["sa"].tolist())
    af_stack = np.vstack(sub["af"].tolist())
    periods = sub.iloc[0]["periods"]
    freq = sub.iloc[0]["freq"]

    # Prefer across-node spatial bands for 2D (available even with 1 seed)
    spatial_rows = (
        sub[sub["af_spatial_p16"].notna()] if "af_spatial_p16" in sub.columns else sub.iloc[0:0]
    )
    if len(spatial_rows):
        p16_af = np.median(np.vstack(spatial_rows["af_spatial_p16"].tolist()), axis=0)
        p84_af = np.median(np.vstack(spatial_rows["af_spatial_p84"].tolist()), axis=0)
        med_spatial = np.median(np.vstack(spatial_rows["af_spatial_median"].tolist()), axis=0)
        sig_spatial = np.median(np.vstack(spatial_rows["af_spatial_sigma_ln"].tolist()), axis=0)
        n_nodes = int(spatial_rows["n_spatial_nodes"].max())
        band_source = f"spatial n={n_nodes}"
    else:
        p16_af = np.percentile(af_stack, 16, axis=0)
        p84_af = np.percentile(af_stack, 84, axis=0)
        med_spatial = np.median(af_stack, axis=0)
        sig_spatial = np.array([sigma_ln(af_stack[:, j]) for j in range(af_stack.shape[1])])
        n_nodes = 0
        band_source = f"seeds n={len(sub)}"

    return {
        "periods": periods,
        "median_sa": np.median(sa_stack, axis=0),
        "p16_sa": np.percentile(sa_stack, 16, axis=0),
        "p84_sa": np.percentile(sa_stack, 84, axis=0),
        "sigma_ln_sa": np.array([sigma_ln(sa_stack[:, j]) for j in range(sa_stack.shape[1])]),
        "freq": freq,
        "median_af": np.median(af_stack, axis=0),
        "p16_af": p16_af,
        "p84_af": p84_af,
        "median_af_spatial": med_spatial,
        "sigma_ln_af": sig_spatial,
        "sigma_ln_af_seeds": np.array([sigma_ln(af_stack[:, j]) for j in range(af_stack.shape[1])]),
        "n": len(sub),
        "n_spatial_nodes": n_nodes,
        "band_source": band_source,
    }


def plot_af_method_subplots(
    df: pd.DataFrame, out_dir: Path, vs1: float = 230.0, motion_id: str = "M1"
) -> Path | None:
    """Primary comparison: one row per method — median AF(f) and σ_ln AF vs 2D GRF."""
    ref = reference_curves(df, vs1, motion_id)
    if not ref:
        return None

    methods = [m for m in METHODS if m in set(df["method"].unique())]
    if not methods:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(methods)
    fig, axes = plt.subplots(n, 2, figsize=(10, 2.35 * n), sharex="col")
    if n == 1:
        axes = np.asarray([axes])

    freq = ref["freq"]
    ref_sig = ref.get("sigma_ln_af")
    if ref_sig is None:
        # Older analyze_response caches; recompute from dataframe rows.
        ref_rows = df[
            (df["method"] == "grf_2d") & (df["vs1"] == vs1) & (df["motion_id"] == motion_id)
        ]
        af_stack = np.vstack(ref_rows["af"].tolist())
        ref_sig = np.array([sigma_ln(af_stack[:, j]) for j in range(af_stack.shape[1])])

    for i, method in enumerate(methods):
        ax_af, ax_sig = axes[i]
        curves = _method_curves(df, method, vs1, motion_id)
        if curves is None:
            ax_af.set_visible(False)
            ax_sig.set_visible(False)
            continue

        label = METHOD_LABELS.get(method, method)
        is_ref = method == "grf_2d"

        ax_af.plot(freq, ref["median_af"], color="0.35", lw=1.8, label="grf_2d ref")
        if is_ref:
            ax_af.fill_between(
                freq,
                curves["p16_af"],
                curves["p84_af"],
                color="0.5",
                alpha=0.25,
                label=f"16–84% ({curves['band_source']})",
            )
            ax_af.plot(freq, curves["median_af"], color="k", lw=2.0, label=label)
        else:
            ax_af.fill_between(
                freq,
                curves["p16_af"],
                curves["p84_af"],
                color="C0",
                alpha=0.2,
                label=f"16–84% ({curves['band_source']})",
            )
            ax_af.plot(freq, curves["median_af"], color="C0", lw=2.0, label=label)
            delta = float(
                np.mean(
                    np.log(np.clip(curves["median_af"], 1e-12, None))
                    - np.log(np.clip(ref["median_af"], 1e-12, None))
                )
            )
            ax_af.text(
                0.98,
                0.95,
                f"Δμ_ln AF = {delta:+.3f}",
                transform=ax_af.transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )
        ax_af.set_xscale("log")
        ax_af.set_ylabel("|AF|")
        ax_af.set_title(f"{label}  (seeds={curves['n']})", loc="left", fontsize=10)
        ax_af.legend(fontsize=7, loc="upper left")

        # Right: spatial σ_ln AF when available; else seed σ_ln; else ratio
        has_spatial = curves["n_spatial_nodes"] >= 2
        if has_spatial or curves["n"] >= 2:
            if is_ref:
                ax_sig.plot(freq, curves["sigma_ln_af"], color="k", lw=2.0, label="grf_2d")
            else:
                # Prefer matching metric: spatial if both have it
                ref_curves = _method_curves(df, "grf_2d", vs1, motion_id)
                ref_line = (
                    ref_curves["sigma_ln_af"]
                    if ref_curves is not None and ref_curves["n_spatial_nodes"] >= 2
                    else ref_sig
                )
                ax_sig.plot(freq, ref_line, color="0.35", lw=1.8, label="grf_2d ref")
                ax_sig.plot(freq, curves["sigma_ln_af"], color="C1", lw=2.0, label=label)
                dsig = float(np.mean(curves["sigma_ln_af"] - ref_line))
                ax_sig.text(
                    0.98,
                    0.95,
                    f"Δσ_ln = {dsig:+.3f}",
                    transform=ax_sig.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                )
            ylabel = "σ_ln AF (spatial)" if has_spatial else "σ_ln AF (seeds)"
            ax_sig.set_ylabel(ylabel)
        else:
            ratio = curves["median_af"] / np.clip(ref["median_af"], 1e-12, None)
            ax_sig.axhline(1.0, color="0.35", lw=1.2, label="ref")
            ax_sig.plot(freq, ratio, color="C1", lw=2.0, label=label)
            ax_sig.set_ylabel("AF / AF_ref")
        ax_sig.set_xscale("log")
        ax_sig.legend(fontsize=7, loc="upper left")

    axes[-1, 0].set_xlabel("Frequency (Hz)")
    axes[-1, 1].set_xlabel("Frequency (Hz)")
    fig.suptitle(
        f"Transfer-function comparison vs 2D GRF — Vs1={vs1:g}, drive={motion_id}",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    out_path = out_dir / f"af_method_subplots_Vs1{vs1:.0f}_{motion_id}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_method_subplots(
    df: pd.DataFrame, out_dir: Path, vs1: float = 230.0, motion_id: str = "M1"
) -> Path | None:
    """One subplot per method: Sa (with Δln) and σ_ln Sa vs 2D GRF reference."""
    ref = reference_curves(df, vs1, motion_id)
    if not ref:
        return None

    methods = [m for m in METHODS if m in set(df["method"].unique())]
    if not methods:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(methods)
    fig, axes = plt.subplots(n, 2, figsize=(10, 2.4 * n), sharex="col")
    if n == 1:
        axes = np.asarray([axes])

    periods = ref["periods"]
    ref_sig = ref["sigma_ln_sa"]

    for i, method in enumerate(methods):
        ax_sa, ax_sig = axes[i]
        curves = _method_curves(df, method, vs1, motion_id)
        if curves is None:
            ax_sa.set_visible(False)
            ax_sig.set_visible(False)
            continue

        label = METHOD_LABELS.get(method, method)
        is_ref = method == "grf_2d"

        # Left: median Sa (+ optional 16–84 band); reference always overlaid
        ax_sa.plot(periods, ref["median_sa"], color="0.35", lw=1.8, label="grf_2d ref")
        if not is_ref:
            ax_sa.fill_between(
                periods,
                curves["p16_sa"],
                curves["p84_sa"],
                color="C0",
                alpha=0.18,
                label="16–84%",
            )
            ax_sa.plot(periods, curves["median_sa"], color="C0", lw=2.0, label=label)
            delta = float(
                np.mean(
                    np.log(np.clip(curves["median_sa"], 1e-12, None))
                    - np.log(np.clip(ref["median_sa"], 1e-12, None))
                )
            )
            ax_sa.text(
                0.98,
                0.95,
                f"Δμ_ln Sa = {delta:+.3f}",
                transform=ax_sa.transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )
        else:
            ax_sa.fill_between(
                periods,
                curves["p16_sa"],
                curves["p84_sa"],
                color="0.5",
                alpha=0.2,
                label="16–84%",
            )
            ax_sa.plot(periods, curves["median_sa"], color="k", lw=2.0, label=label)

        ax_sa.set_ylabel("Sa (m/s²)")
        ax_sa.set_title(f"{label}  (n={curves['n']})", loc="left", fontsize=10)
        ax_sa.legend(fontsize=7, loc="upper left")

        # Right: σ_ln Sa vs period (needs ≥2 seeds; otherwise show Sa ratio vs ref)
        if curves["n"] >= 2:
            if is_ref:
                ax_sig.plot(periods, curves["sigma_ln_sa"], color="k", lw=2.0, label="grf_2d")
            else:
                ax_sig.plot(periods, ref_sig, color="0.35", lw=1.8, label="grf_2d ref")
                ax_sig.plot(periods, curves["sigma_ln_sa"], color="C1", lw=2.0, label=label)
                dsig = float(np.mean(curves["sigma_ln_sa"] - ref_sig))
                ax_sig.text(
                    0.98,
                    0.95,
                    f"Δσ_ln = {dsig:+.3f}",
                    transform=ax_sig.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                )
            ax_sig.set_ylabel("σ_ln Sa")
        else:
            ratio = curves["median_sa"] / np.clip(ref["median_sa"], 1e-12, None)
            ax_sig.axhline(1.0, color="0.35", lw=1.2, label="ref")
            ax_sig.plot(periods, ratio, color="C1", lw=2.0, label=label)
            ax_sig.set_ylabel("Sa / Sa_ref")
        ax_sig.legend(fontsize=7, loc="upper left")

    axes[-1, 0].set_xlabel("Period (s)")
    axes[-1, 1].set_xlabel("Period (s)")
    fig.suptitle(
        f"Method comparison vs 2D GRF reference — Vs1={vs1:g}, {motion_id}",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    out_path = out_dir / f"method_subplots_Vs1{vs1:.0f}_{motion_id}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _profile_config(vs1: float) -> ProfileRandomizationConfig:
    dz = active_dz()
    return ProfileRandomizationConfig(
        vs_mean=vs1,
        thickness=THICKNESS,
        dz=dz,
        vs_bedrock=VS2,
        bedrock_thickness=BEDROCK_DEPTH,
        cov=0.2,
    )


def _depth_axis(n_layers: int, dz: float) -> np.ndarray:
    return (np.arange(n_layers) + 0.5) * dz


def _base_soil_profile(vs1: float) -> tuple[np.ndarray, np.ndarray]:
    dz = active_dz()
    n_soil = max(1, int(round(THICKNESS / dz)))
    return _depth_axis(n_soil, dz), np.full(n_soil, vs1, dtype=float)


def plot_hallal_profile_realizations(
    out_dir: Path,
    *,
    vs1: float = 230.0,
    seeds: list[int] | None = None,
    n_show: int = 5,
) -> Path:
    """Vs vs depth: base case and full Toro / Passeri realizations (soil + bedrock)."""
    from seiskit.profile_randomization import build_base_case_profile

    seeds = seeds or list(range(1, n_show + 1))
    cfg = _profile_config(vs1)
    dz = active_dz()
    base_full = build_base_case_profile(cfg)
    depth = (np.arange(len(base_full)) + 0.5) * dz
    int(round(THICKNESS / dz))

    fig, axes = plt.subplots(1, 2, figsize=(9, 5.5), sharey=True)
    panels = (
        ("Hallal VsRand (Toro full)", generate_vs_randomized_profile),
        ("Hallal ttsRand (Passeri full)", generate_tts_randomized_profile),
    )

    cmap = plt.get_cmap("tab10")
    for ax, (title, gen_fn) in zip(axes, panels):
        ax.step(base_full, depth, color="k", lw=2.0, ls="--", label="Base case", where="pre")
        for i, seed in enumerate(seeds):
            rng = np.random.default_rng(seed)
            prof = gen_fn(cfg, rng)
            ax.step(
                prof,
                depth,
                color=cmap(i % 10),
                lw=1.4,
                alpha=0.85,
                label=f"seed {seed}",
                where="pre",
            )
        ax.axhline(THICKNESS, color="0.5", ls=":", lw=1.0)
        ax.set_xlabel(r"$V_s$ (m/s)")
        ax.set_title(title)
        ax.set_ylim(depth[-1] + dz, 0)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="lower right")

    axes[0].set_ylabel("Depth (m)")
    fig.suptitle(
        f"1D profile randomization — Vs1={vs1:g} m/s, H={THICKNESS:g} m (+ bedrock)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"hallal_profiles_Vs1{vs1:.0f}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _load_2d_from_h5(h5_path: Path) -> dict:
    import h5py

    with h5py.File(h5_path, "r") as f:
        vs_field = f["Vs_field"][:]
        vs_profile = f["Vs_profile_1D"][:]
        dx = float(f["grid"].attrs["dx"])
        dz = float(f["grid"].attrs["dz"])
        vs1 = float(f["params"].attrs["Vs1"])
        seed = int(f["params"].attrs["seed"])
    return {
        "vs_field": vs_field,
        "vs_profile_1d": vs_profile,
        "dx": dx,
        "dz": dz,
        "vs1": vs1,
        "seed": seed,
    }


def _variable_domain_slice(vs_field: np.ndarray, dx: float) -> tuple[np.ndarray, int]:
    bc_cols = int(round(active_bc_width() / dx))
    var_cols = int(round(active_lx_var() / dx))
    i0 = bc_cols
    i1 = min(vs_field.shape[1], i0 + var_cols)
    vs_var = vs_field[:, i0:i1]
    i_center = vs_var.shape[1] // 2
    return vs_var, i_center


def plot_grf_2d_explainability(
    out_dir: Path,
    *,
    h5_path: Path | None = None,
    vs1: float = 230.0,
) -> Path | None:
    """2D GRF field with center-column extraction vs base template."""
    h5_path = h5_path or Path("results/h5/run_0.h5")
    if not h5_path.exists():
        return None

    data = _load_2d_from_h5(h5_path)
    vs_var, i_center = _variable_domain_slice(data["vs_field"], data["dx"])
    nz, nx = vs_var.shape
    depth = (np.arange(nz) + 0.5) * data["dz"]
    x = (np.arange(nx) + 0.5) * data["dx"]
    x_extract = x[i_center]

    center_col = vs_var[:, i_center]
    base_profile = np.asarray(data["vs_profile_1d"], dtype=float)
    depth_base = _depth_axis(len(base_profile), data["dz"])
    p16 = np.percentile(vs_var, 16, axis=1)
    p84 = np.percentile(vs_var, 84, axis=1)

    soil_mask = vs_var < VS2 * 0.99
    vmin = float(np.min(vs_var[soil_mask])) if np.any(soil_mask) else float(data["vs1"]) * 0.5
    vmax = float(np.max(vs_var[soil_mask])) if np.any(soil_mask) else float(data["vs1"]) * 1.5

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1.35, 1.0]})
    ax0 = axes[0]
    im = ax0.imshow(
        vs_var,
        extent=(0, nx * data["dx"], nz * data["dz"], 0),
        aspect="auto",
        cmap=get_crameri_cmap("batlow", reverse=True),
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    ax0.axvline(x_extract, color="white", lw=2.0, ls="-", label="Extracted 1D column")
    ax0.axvline(x_extract, color="crimson", lw=1.0, ls="--")
    ax0.axhline(THICKNESS, color="white", lw=0.8, ls=":", alpha=0.8)
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.02)
    cbar.set_label(r"$V_s$ (m/s)")
    ax0.set_xlabel("Distance x (m)")
    ax0.set_ylabel("Depth (m)")
    ax0.set_title(
        f"2D GRF realization (seed={data['seed']})\nsame field for grf_2d & de la Torre 2D",
        fontsize=10,
    )
    ax0.legend(loc="upper right", fontsize=8)

    ax1 = axes[1]
    ax1.fill_betweenx(depth, p16, p84, color="C0", alpha=0.2, label="16–84% across x")
    ax1.plot(base_profile, depth_base, color="k", lw=2.0, ls="--", label="Base 1D template")
    ax1.plot(center_col, depth, color="crimson", lw=2.2, label="Center column (extracted)")
    ax1.axhline(THICKNESS, color="0.5", ls=":", lw=1.0)
    ax1.set_xlabel(r"$V_s$ (m/s)")
    ax1.set_ylabel("Depth (m)")
    ax1.set_ylim(nz * data["dz"], 0)
    ax1.set_title("de la Torre 1D diagnostic\n(center column vs mean template)", fontsize=10)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(True, alpha=0.25)

    fig.suptitle(
        f"2D GRF vs extracted 1D profile — Vs1={data['vs1']:g} m/s",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"grf2d_explainability_Vs1{data['vs1']:.0f}_s{data['seed']}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-dir", type=Path, default=Path("results/h5"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--vs1", type=float, default=230.0)
    parser.add_argument("--motion", type=str, default="M1")
    args = parser.parse_args()

    df = collect_rows(args.h5_dir)
    if df.empty:
        print("No data to plot.")
        return
    af_out = plot_af_method_subplots(df, args.out_dir, vs1=args.vs1, motion_id=args.motion)
    if af_out is not None:
        print(f"TF subplot figure: {af_out}")
    sa_out = plot_method_subplots(df, args.out_dir, vs1=args.vs1, motion_id=args.motion)
    if sa_out is not None:
        print(f"Sa subplot figure: {sa_out}")
    hallal_fig = plot_hallal_profile_realizations(args.out_dir, vs1=args.vs1, seeds=[1, 2, 3, 4, 5])
    print(f"Hallal profile figure: {hallal_fig}")
    h5_2d = args.h5_dir / "run_0.h5"
    grf_fig = plot_grf_2d_explainability(args.out_dir, h5_path=h5_2d, vs1=args.vs1)
    if grf_fig is not None:
        print(f"2D GRF explainability figure: {grf_fig}")
    print(f"Figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
