"""Publication plots for Response_Variability comparison."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze_response import (
    _align_af_stack,
    _geomean_af,
    _interp_af,
    _sigma_ln_af_stack,
    collect_rows,
    reference_curves,
)
from manifest import (
    BEDROCK_DEPTH,
    METHODS,
    NO_NX_FULL,
    PRETELL_CENTRAL_WIDTH_M,
    active_bc_width,
    active_dz,
    active_lx_var,
    active_rf_bc_width,
    active_rf_lx_var,
    pretell_column_indices,
)

from seiskit.intensity_measures import sigma_ln
from seiskit.plot_config import apply_style, get_crameri_cmap
from seiskit.profile_randomization import (
    ProfileRandomizationConfig,
    generate_tts_randomized_profile,
    generate_vs_randomized_profile,
)
from seiskit.theory import Layer, RockHalfspace, layered_transfer_function


def _load_rv_run_experiment():
    """Load this campaign's run_experiment (not neural-operator/data)."""
    path = Path(__file__).resolve().parent / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("rv_run_experiment", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_rv_run = _load_rv_run_experiment()
theoretical_f0 = _rv_run.theoretical_f0

apply_style()

METHOD_LABELS = {
    "grf_2d": "2D GRF (reference)",
    "pretell": "Pretell (1D geomean, central 100 m)",
    "hallal_vs": "Hallal VsRand",
    "hallal_tts": "Hallal ttsRand",
    "hallal_dmin": "Hallal Dmin sweep (base Vs)",
}

METHOD_COLORS = {
    "grf_2d": "0.2",
    "pretell": "C3",
    "hallal_vs": "C0",
    "hallal_tts": "C1",
    "hallal_dmin": "C4",
}

# Match AnalysisConfig.damping_zeta used for Hallal / base 1D runs.
_BASE_THEORY_XI = 0.025
_BASE_THEORY_RHO = 2000.0


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _common_plot_freq(df: pd.DataFrame, sobol_id: int, motion_id: str) -> np.ndarray:
    """Shared 0.1–10 Hz axis (prefer grf_2d / surrogate grid)."""
    for method in ("grf_2d", "pretell"):
        sub = df[
            (df["method"] == method) & (df["sobol_id"] == sobol_id) & (df["motion_id"] == motion_id)
        ]
        if not sub.empty:
            freq = np.asarray(sub.iloc[0]["freq"], dtype=float)
            return freq[(freq >= 0.1) & (freq <= 10.0)]
    return np.logspace(-1, 1, 1000)


def _base_1d_theory_tf(
    *,
    vs1: float,
    H: float,
    vs2: float,
    freq: np.ndarray,
    xi: float = _BASE_THEORY_XI,
    rho: float = _BASE_THEORY_RHO,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Closed-form within AF from ``seiskit.theory`` (surface / soil–rock interface)."""
    f0 = theoretical_f0(vs1, H)
    freq = np.asarray(freq, dtype=float)
    _, af_within, _ = layered_transfer_function(
        freq,
        [Layer(float(H), float(vs1), float(rho), float(xi))],
        RockHalfspace(float(vs2), float(rho), 0.0),
    )
    return freq, np.asarray(af_within, dtype=float), f0


def _load_base_1d_tf(
    df: pd.DataFrame,
    *,
    sobol_id: int,
    motion_id: str,
    freq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    sub = df[(df["sobol_id"] == sobol_id) & (df["motion_id"] == motion_id)]
    if sub.empty:
        return None
    row = sub.iloc[0]
    return _base_1d_theory_tf(
        vs1=float(row["vs1"]),
        H=float(row["H"]),
        vs2=float(row["vs2"]),
        freq=freq,
    )


def _overlay_base_1d_and_f0(
    ax,
    *,
    base: tuple[np.ndarray, np.ndarray, float] | None,
    freq_plot: np.ndarray | None = None,
    fmax_hz: float = 10.0,
) -> None:
    """Plot closed-form 1D base-Vs AF and mark quarter-wave f₀."""
    if base is None:
        return
    freq_b, af_b, f0 = base
    if freq_plot is not None and len(freq_plot) and len(freq_b):
        af_plot = _interp_af(freq_b, af_b, np.asarray(freq_plot, dtype=float))
        mask = np.asarray(freq_plot) <= fmax_hz
        ax.plot(
            freq_plot[mask],
            af_plot[mask],
            color="0.45",
            ls="-.",
            lw=2.0,
            label=rf"1D base theory (within, $\xi$={_BASE_THEORY_XI:g})",
            zorder=3,
        )
    else:
        mask = np.asarray(freq_b) <= fmax_hz
        ax.plot(
            freq_b[mask],
            af_b[mask],
            color="0.45",
            ls="-.",
            lw=2.0,
            label=rf"1D base theory (within, $\xi$={_BASE_THEORY_XI:g})",
            zorder=3,
        )
    ax.axvline(
        f0,
        color="0.2",
        ls=":",
        lw=1.6,
        label=rf"$f_0=V_{{s1}}/(4H)$={f0:.3f} Hz",
        zorder=4,
    )


def _method_curves(
    df: pd.DataFrame,
    method: str,
    sobol_id: int,
    motion_id: str,
    *,
    freq_dst: np.ndarray | None = None,
) -> dict | None:
    sub = df[
        (df["method"] == method) & (df["sobol_id"] == sobol_id) & (df["motion_id"] == motion_id)
    ]
    if sub.empty:
        return None
    sa_stack = np.vstack(sub["sa"].tolist())
    af_list = [np.asarray(row["af"], dtype=float) for _, row in sub.iterrows()]
    freqs = [np.asarray(row["freq"], dtype=float) for _, row in sub.iterrows()]
    periods = sub.iloc[0]["periods"]
    freq = freqs[0] if freq_dst is None else np.asarray(freq_dst, dtype=float)
    af_stack = _align_af_stack(af_list, freqs, freq)

    # Prefer across-node spatial bands for 2D (available even with 1 seed)
    spatial_rows = (
        sub[sub["af_spatial_p16"].notna()] if "af_spatial_p16" in sub.columns else sub.iloc[0:0]
    )
    if len(spatial_rows):
        freq_sp = np.asarray(spatial_rows.iloc[0]["freq"], dtype=float)
        p16_af = np.median(np.vstack(spatial_rows["af_spatial_p16"].tolist()), axis=0)
        p84_af = np.median(np.vstack(spatial_rows["af_spatial_p84"].tolist()), axis=0)
        med_spatial = np.median(np.vstack(spatial_rows["af_spatial_median"].tolist()), axis=0)
        sig_spatial = np.median(np.vstack(spatial_rows["af_spatial_sigma_ln"].tolist()), axis=0)
        if freq_dst is not None:
            p16_af = _interp_af(freq_sp, p16_af, freq)
            p84_af = _interp_af(freq_sp, p84_af, freq)
            med_spatial = _interp_af(freq_sp, med_spatial, freq)
            sig_spatial = _interp_af(freq_sp, sig_spatial, freq)
        n_nodes = int(spatial_rows["n_spatial_nodes"].max())
        band_source = f"spatial n={n_nodes}"
    else:
        p16_af = np.nanpercentile(af_stack, 16, axis=0)
        p84_af = np.nanpercentile(af_stack, 84, axis=0)
        med_spatial = np.nanmedian(af_stack, axis=0)

        def _col_sigma(j: int) -> float:
            col = af_stack[:, j]
            col = col[np.isfinite(col)]
            return float(sigma_ln(col)) if len(col) else float("nan")

        sig_spatial = np.array([_col_sigma(j) for j in range(af_stack.shape[1])])
        n_nodes = 0
        band_source = f"seeds n={len(sub)}"

    return {
        "periods": periods,
        "median_sa": np.median(sa_stack, axis=0),
        "p16_sa": np.percentile(sa_stack, 16, axis=0),
        "p84_sa": np.percentile(sa_stack, 84, axis=0),
        "sigma_ln_sa": np.array([sigma_ln(sa_stack[:, j]) for j in range(sa_stack.shape[1])]),
        "freq": freq,
        "median_af": np.nanmedian(af_stack, axis=0),
        "p16_af": p16_af,
        "p84_af": p84_af,
        "median_af_spatial": med_spatial,
        "sigma_ln_af": sig_spatial,
        "sigma_ln_af_seeds": np.array(
            [
                float(sigma_ln(c[np.isfinite(c)])) if np.any(np.isfinite(c)) else float("nan")
                for c in af_stack.T
            ]
        ),
        "n": len(sub),
        "n_spatial_nodes": n_nodes,
        "band_source": band_source,
    }


def plot_tf_method_comparison(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    sobol_id: int,
    motion_id: str = "M1",
) -> Path | None:
    """All methods on one axis: median AF(f) with ±1σ_ln band per method."""
    sub = df[(df["sobol_id"] == sobol_id) & (df["motion_id"] == motion_id)]
    if sub.empty:
        return None

    methods = [m for m in METHODS if m in set(sub["method"].unique())]
    if not methods:
        return None

    meta = sub.iloc[0]
    freq = _common_plot_freq(df, sobol_id, motion_id)

    fig, axes = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]}
    )
    ax_af, ax_sig = axes

    for method in methods:
        msub = sub[sub["method"] == method]
        if msub.empty:
            continue
        freqs = [np.asarray(row["freq"], dtype=float) for _, row in msub.iterrows()]
        af_list = [np.asarray(row["af"], dtype=float) for _, row in msub.iterrows()]
        af_stack = _align_af_stack(af_list, freqs, freq)
        med = np.nanmedian(af_stack, axis=0)
        sig = np.array(
            [sigma_ln(af_stack[:, j][np.isfinite(af_stack[:, j])]) for j in range(af_stack.shape[1])]
        )
        lo = med * np.exp(-sig)
        hi = med * np.exp(sig)
        color = METHOD_COLORS.get(method, "C0")
        label = _method_label(method)
        lw = 2.4 if method == "grf_2d" else 1.8
        ax_af.fill_between(freq, lo, hi, color=color, alpha=0.18, linewidth=0)
        ax_af.plot(freq, med, color=color, lw=lw, label=f"{label} (n={len(msub)})")
        ax_sig.plot(freq, sig, color=color, lw=lw, label=label)

    base = _load_base_1d_tf(df, sobol_id=sobol_id, motion_id=motion_id, freq=freq)
    _overlay_base_1d_and_f0(ax_af, base=base, freq_plot=freq)
    if base is not None:
        ax_sig.axvline(base[2], color="0.2", ls=":", lw=1.4, alpha=0.8)

    ax_af.set_xscale("log")
    ax_af.set_xlim(0.1, 10.0)
    ax_af.set_ylabel("|AF|")
    ax_af.set_title(
        f"Transfer functions — Sobol #{sobol_id}  "
        f"Vs1={meta['vs1']:.0f} H={meta['H']:.0f} CoV={meta['cov']:.2f} Vs2={meta['vs2']:.0f}",
        loc="left",
        fontsize=10,
    )
    ax_af.legend(fontsize=8, loc="upper left")
    ax_af.grid(True, alpha=0.25)

    ax_sig.set_xscale("log")
    ax_sig.set_yscale("log")
    ax_sig.set_xlim(0.1, 10.0)
    ax_sig.set_xlabel("Frequency (Hz)")
    ax_sig.set_ylabel(r"$\sigma_{\ln}$ AF")
    ax_sig.legend(fontsize=7, loc="upper right", ncol=2)
    ax_sig.grid(True, alpha=0.25)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tf_methods_sobol{sobol_id:02d}_{motion_id}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_af_method_subplots(
    df: pd.DataFrame, out_dir: Path, sobol_id: int = 0, motion_id: str = "M1"
) -> Path | None:
    """Primary comparison: one row per method — median AF(f) and σ_ln AF vs 2D GRF."""
    ref = reference_curves(df, sobol_id, motion_id)
    if not ref:
        return None

    ref_method = ref["ref_method"]
    ref_label = _method_label(ref_method)

    methods = [m for m in METHODS if m in set(df["method"].unique())]
    if not methods:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(methods)
    fig, axes = plt.subplots(n, 2, figsize=(10, 2.35 * n), sharex="col")
    if n == 1:
        axes = np.asarray([axes])

    freq = ref["freq"]
    base_overlay = _load_base_1d_tf(df, sobol_id=sobol_id, motion_id=motion_id, freq=freq)
    ref_sig = ref.get("sigma_ln_af")
    if ref_sig is None:
        ref_rows = df[
            (df["method"] == ref_method)
            & (df["sobol_id"] == sobol_id)
            & (df["motion_id"] == motion_id)
        ]
        af_stack = np.vstack(ref_rows["af"].tolist())
        ref_sig = np.array([sigma_ln(af_stack[:, j]) for j in range(af_stack.shape[1])])

    for i, method in enumerate(methods):
        ax_af, ax_sig = axes[i]
        curves = _method_curves(df, method, sobol_id, motion_id, freq_dst=freq)
        if curves is None:
            ax_af.set_visible(False)
            ax_sig.set_visible(False)
            continue

        label = _method_label(method)
        is_ref = method == ref_method

        ax_af.plot(freq, ref["median_af"], color="0.35", lw=1.8, label=f"{ref_label} ref")
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
        if i == 0 and base_overlay is not None:
            _overlay_base_1d_and_f0(ax_af, base=base_overlay, freq_plot=freq)
        elif base_overlay is not None:
            ax_af.axvline(base_overlay[2], color="0.2", ls=":", lw=1.2, alpha=0.7)
        ax_af.set_xscale("log")
        ax_af.set_ylabel("|AF|")
        ax_af.set_yscale("log")
        ax_af.set_title(f"{label}  (seeds={curves['n']})", loc="left", fontsize=10)
        ax_af.legend(fontsize=7, loc="upper left")

        # Right: spatial σ_ln AF when available; else seed σ_ln; else ratio
        has_spatial = curves["n_spatial_nodes"] >= 2
        if has_spatial or curves["n"] >= 2:
            if is_ref:
                ax_sig.plot(freq, curves["sigma_ln_af"], color="k", lw=2.0, label=ref_label)
            else:
                ref_curves = _method_curves(df, ref_method, sobol_id, motion_id, freq_dst=freq)
                ref_line = (
                    ref_curves["sigma_ln_af"]
                    if ref_curves is not None and ref_curves["n_spatial_nodes"] >= 2
                    else ref_sig
                )
                ax_sig.plot(freq, ref_line, color="0.35", lw=1.8, label=f"{ref_label} ref")
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
        ax_sig.set_yscale("log")
        ax_sig.legend(fontsize=7, loc="upper left")

    axes[-1, 0].set_xlabel("Frequency (Hz)")
    axes[-1, 1].set_xlabel("Frequency (Hz)")
    fig.suptitle(
        f"TF subplots vs {ref_label} — Sobol #{sobol_id} Vs1={ref['vs1']:g} H={ref['H']:g} {motion_id}",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    out_path = out_dir / f"af_method_subplots_sobol{sobol_id:02d}_{motion_id}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_method_subplots(
    df: pd.DataFrame, out_dir: Path, sobol_id: int = 0, motion_id: str = "M1"
) -> Path | None:
    """One subplot per method: Sa (with Δln) and σ_ln Sa vs 2D GRF reference."""
    ref = reference_curves(df, sobol_id, motion_id)
    if not ref:
        return None

    ref_method = ref["ref_method"]
    ref_label = _method_label(ref_method)

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
        curves = _method_curves(df, method, sobol_id, motion_id)
        if curves is None:
            ax_sa.set_visible(False)
            ax_sig.set_visible(False)
            continue

        label = _method_label(method)
        is_ref = method == ref_method

        # Left: median Sa (+ optional 16–84 band); reference always overlaid
        ax_sa.plot(periods, ref["median_sa"], color="0.35", lw=1.8, label=f"{ref_label} ref")
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
        ax_sa.set_xscale("log")
        ax_sa.set_yscale("log")
        ax_sa.set_title(f"{label}  (n={curves['n']})", loc="left", fontsize=10)
        ax_sa.legend(fontsize=7, loc="upper left")

        # Right: σ_ln Sa vs period (needs ≥2 seeds; otherwise show Sa ratio vs ref)
        if curves["n"] >= 2:
            if is_ref:
                ax_sig.plot(periods, curves["sigma_ln_sa"], color="k", lw=2.0, label=ref_label)
            else:
                ax_sig.plot(periods, ref_sig, color="0.35", lw=1.8, label=f"{ref_label} ref")
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
            ax_sig.set_yscale("log")
        else:
            ratio = curves["median_sa"] / np.clip(ref["median_sa"], 1e-12, None)
            ax_sig.axhline(1.0, color="0.35", lw=1.2, label="ref")
            ax_sig.plot(periods, ratio, color="C1", lw=2.0, label=label)
            ax_sig.set_ylabel("Sa / Sa_ref")
            ax_sig.set_yscale("log")
        ax_sig.legend(fontsize=7, loc="upper left")

    axes[-1, 0].set_xlabel("Period (s)")
    axes[-1, 1].set_xlabel("Period (s)")
    fig.suptitle(
        f"Method comparison vs {ref_label} — Sobol #{sobol_id} Vs1={ref['vs1']:g} H={ref['H']:g} {motion_id}",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    out_path = out_dir / f"method_subplots_sobol{sobol_id:02d}_{motion_id}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _profile_config(
    vs1: float,
    *,
    H: float,
    cov: float,
    vs2: float,
) -> ProfileRandomizationConfig:
    dz = active_dz()
    return ProfileRandomizationConfig(
        vs_mean=vs1,
        thickness=H,
        dz=dz,
        vs_bedrock=vs2,
        bedrock_thickness=BEDROCK_DEPTH,
        cov=cov,
        use_full_model=False,
        randomize_layer_thickness=False,
        randomize_bedrock_depth=False,
        vary_bedrock_vs=False,
    )


def _depth_axis(n_layers: int, dz: float) -> np.ndarray:
    return (np.arange(n_layers) + 0.5) * dz


def _sobol_case(sobol_id: int):
    from manifest import active_base_cases

    for case in active_base_cases():
        if case.sobol_id == sobol_id:
            return case
    raise ValueError(f"Unknown sobol_id={sobol_id}")


def plot_hallal_profile_realizations(
    out_dir: Path,
    *,
    sobol_id: int = 0,
    seeds: list[int] | None = None,
    n_show: int = 5,
) -> Path:
    """Vs vs depth: base case and full Toro / Passeri realizations (soil + bedrock)."""
    from seiskit.profile_randomization import build_base_case_profile

    case = _sobol_case(sobol_id)
    seeds = seeds or list(range(1, n_show + 1))
    cfg = _profile_config(case.vs1, H=case.H, cov=case.cov, vs2=case.vs2)
    dz = active_dz()
    base_full = build_base_case_profile(cfg)
    depth = (np.arange(len(base_full)) + 0.5) * dz

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
        ax.axhline(case.H, color="0.5", ls=":", lw=1.0)
        ax.set_xlabel(r"$V_s$ (m/s)")
        ax.set_title(title)
        ax.set_ylim(depth[-1] + dz, 0)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="lower right")

    axes[0].set_ylabel("Depth (m)")
    fig.suptitle(
        f"1D profile randomization — Sobol #{sobol_id}  "
        f"Vs1={case.vs1:g} m/s, H={case.H:g} m, CoV={case.cov:.2f} (+ bedrock)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"hallal_profiles_sobol{sobol_id:02d}.png"
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
        H = float(f["params"].attrs["H"])
        vs2 = float(f["params"].attrs["Vs2"])
        seed = int(f["params"].attrs["seed"])
    return {
        "vs_field": vs_field,
        "vs_profile_1d": vs_profile,
        "dx": dx,
        "dz": dz,
        "vs1": vs1,
        "H": H,
        "vs2": vs2,
        "seed": seed,
    }


def _variable_domain_slice(vs_field: np.ndarray, dx: float) -> tuple[np.ndarray, int, np.ndarray]:
    """Return (variability strip, center index in strip, pretell column indices in strip)."""
    if vs_field.shape[1] == NO_NX_FULL:
        bc_cols = int(round(active_rf_bc_width() / dx))
        var_cols = int(round(active_rf_lx_var() / dx))
    else:
        bc_cols = int(round(active_bc_width() / dx))
        var_cols = int(round(active_lx_var() / dx))
    i0 = bc_cols
    i1 = min(vs_field.shape[1], i0 + var_cols)
    vs_var = vs_field[:, i0:i1]
    i_center = vs_var.shape[1] // 2
    full_cols = pretell_column_indices()
    strip_cols = full_cols - bc_cols
    return vs_var, i_center, strip_cols


def plot_grf_2d_explainability(
    out_dir: Path,
    *,
    h5_path: Path | None = None,
) -> Path | None:
    """2D GRF on NO grid with Pretell central-region 1D extractions."""
    h5_path = h5_path or Path("results/h5/run_0.h5")
    if not h5_path.exists():
        return None

    data = _load_2d_from_h5(h5_path)
    H = float(data["H"])
    vs2 = float(data["vs2"])
    vs_var, i_center, pretell_cols = _variable_domain_slice(data["vs_field"], data["dx"])
    nz, nx = vs_var.shape
    depth = (np.arange(nz) + 0.5) * data["dz"]
    x = (np.arange(nx) + 0.5) * data["dx"]
    x_center = x[i_center]
    x_pretell_lo = x[pretell_cols[0]] if len(pretell_cols) else x_center
    x_pretell_hi = x[pretell_cols[-1]] if len(pretell_cols) else x_center

    pretell_profiles = [vs_var[:, j] for j in pretell_cols]
    pretell_geo = np.exp(np.mean(np.log(np.clip(np.vstack(pretell_profiles), 1e-6, None)), axis=0))
    base_profile = np.asarray(data["vs_profile_1d"], dtype=float)
    depth_base = _depth_axis(len(base_profile), data["dz"])
    p16 = np.percentile(vs_var, 16, axis=1)
    p84 = np.percentile(vs_var, 84, axis=1)

    soil_mask = vs_var < vs2 * 0.99
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
    ax0.axvspan(
        x_pretell_lo,
        x_pretell_hi,
        color="white",
        alpha=0.12,
        label=f"Pretell {PRETELL_CENTRAL_WIDTH_M:.0f} m",
    )
    for j in pretell_cols[:: max(1, len(pretell_cols) // 8)]:
        ax0.axvline(x[j], color="crimson", lw=0.6, alpha=0.5)
    ax0.axvline(x_center, color="white", lw=1.5, ls="--", label="Strip center")
    ax0.axhline(H, color="white", lw=0.8, ls=":", alpha=0.8)
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.02)
    cbar.set_label(r"$V_s$ (m/s)")
    ax0.set_xlabel("Distance x within variability strip (m)")
    ax0.set_ylabel("Depth (m)")
    ax0.set_title(
        f"2D GRF (NO grid, seed={data['seed']})\n"
        f"shared by grf_2d surrogate & Pretell ({len(pretell_cols)} 1D profiles)",
        fontsize=10,
    )
    ax0.legend(loc="upper right", fontsize=8)

    ax1 = axes[1]
    ax1.fill_betweenx(depth, p16, p84, color="C0", alpha=0.2, label="16–84% across strip")
    ax1.plot(base_profile, depth_base, color="k", lw=2.0, ls="--", label="Base 1D template")
    ax1.plot(pretell_geo, depth, color="crimson", lw=2.2, label="Pretell geomean Vs")
    ax1.axhline(H, color="0.5", ls=":", lw=1.0)
    ax1.set_xlabel(r"$V_s$ (m/s)")
    ax1.set_ylabel("Depth (m)")
    ax1.set_ylim(nz * data["dz"], 0)
    ax1.set_title(
        f"Pretell 1D diagnostic\n(geomean of {len(pretell_cols)} profiles in central 100 m)",
        fontsize=10,
    )
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(True, alpha=0.25)

    fig.suptitle(
        f"2D GRF vs extracted 1D profile — Vs1={data['vs1']:g} H={H:g} m/s",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"grf2d_explainability_Vs1{data['vs1']:.0f}_s{data['seed']}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_grf2d_vs_pretell_geomean(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    sobol_id: int,
    motion_id: str = "M1",
) -> Path | None:
    """Center-recorder AF: geomean ± 1σ_ln across RF seeds — surrogate vs OpenSees pretell."""
    methods = ("grf_2d", "pretell")
    curves: dict[str, dict] = {}
    freq_ref: np.ndarray | None = None
    for method in methods:
        sub = df[
            (df["method"] == method) & (df["sobol_id"] == sobol_id) & (df["motion_id"] == motion_id)
        ]
        if sub.empty:
            return None
        if freq_ref is None:
            freq_ref = np.asarray(sub.iloc[0]["freq"], dtype=float)
        freqs = [np.asarray(row["freq"], dtype=float) for _, row in sub.iterrows()]
        af_list = [np.asarray(row["af"], dtype=float) for _, row in sub.iterrows()]
        af_stack = _align_af_stack(af_list, freqs, freq_ref)
        geo = _geomean_af(af_stack)
        sig = _sigma_ln_af_stack(af_stack)
        curves[method] = {
            "freq": freq_ref,
            "geo": geo,
            "lo": geo * np.exp(-sig),
            "hi": geo * np.exp(sig),
            "n": len(sub),
            "af_stack": af_stack,
        }

    freq = freq_ref
    meta = df[(df["sobol_id"] == sobol_id) & (df["motion_id"] == motion_id)].iloc[0]
    n_seeds = curves["grf_2d"]["n"]

    fig, axes = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]}
    )
    ax_af, ax_ratio = axes

    styles = {
        "grf_2d": {"color": "k", "ls": "-", "lw": 2.4, "label": "grf_2d surrogate (geomean)"},
        "pretell": {"color": "C3", "ls": "--", "lw": 2.0, "label": "pretell OpenSees (geomean)"},
    }
    for method in methods:
        c = curves[method]
        st = styles[method]
        ax_af.fill_between(freq, c["lo"], c["hi"], color=st["color"], alpha=0.18, linewidth=0)
        ax_af.plot(freq, c["geo"], color=st["color"], ls=st["ls"], lw=st["lw"], label=st["label"])
        if n_seeds <= 10:
            for row_af in c["af_stack"]:
                ax_af.plot(freq, row_af, color=st["color"], alpha=0.12, lw=0.6)

    base = _load_base_1d_tf(df, sobol_id=sobol_id, motion_id=motion_id, freq=freq)
    _overlay_base_1d_and_f0(ax_af, base=base, freq_plot=freq)
    if base is not None:
        ax_ratio.axvline(base[2], color="0.2", ls=":", lw=1.4, alpha=0.8)

    ax_af.set_xscale("log")
    ax_af.set_xlim(0.1, 10.0)
    ax_af.set_ylabel("|AF| (center recorder)")
    ax_af.set_title(
        f"Center TF geomean ± 1σ_ln — Sobol #{sobol_id}  N={n_seeds} seeds  "
        f"Vs1={meta['vs1']:.0f} H={meta['H']:.0f} CoV={meta['cov']:.2f}",
        loc="left",
        fontsize=10,
    )
    ax_af.legend(fontsize=8, loc="upper left")
    ax_af.grid(True, alpha=0.25)

    ratio = curves["grf_2d"]["geo"] / np.clip(curves["pretell"]["geo"], 1e-12, None)
    ax_ratio.axhline(1.0, color="0.35", lw=1.2)
    ax_ratio.plot(freq, ratio, color="C0", lw=2.0, label="surrogate / pretell")
    ax_ratio.set_xscale("log")
    ax_ratio.set_xlabel("Frequency (Hz)")
    ax_ratio.set_ylabel("AF ratio")
    ax_ratio.legend(fontsize=8)
    ax_ratio.grid(True, alpha=0.25)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tf_grf2d_vs_pretell_geomean_sobol{sobol_id:02d}_{motion_id}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-dir", type=Path, default=Path("results/h5"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--sobol-id", type=int, default=0)
    parser.add_argument("--motion", type=str, default="M1")
    args = parser.parse_args()

    df = collect_rows(args.h5_dir)
    if df.empty:
        print("No data to plot.")
        return

    tf_out = plot_tf_method_comparison(
        df, args.out_dir, sobol_id=args.sobol_id, motion_id=args.motion
    )
    if tf_out is not None:
        print(f"TF method comparison: {tf_out}")

    af_out = plot_af_method_subplots(
        df, args.out_dir, sobol_id=args.sobol_id, motion_id=args.motion
    )
    if af_out is not None:
        print(f"TF subplot figure: {af_out}")

    geo_out = plot_grf2d_vs_pretell_geomean(
        df, args.out_dir, sobol_id=args.sobol_id, motion_id=args.motion
    )
    if geo_out is not None:
        print(f"Geomean grf_2d vs pretell: {geo_out}")

    sa_out = plot_method_subplots(df, args.out_dir, sobol_id=args.sobol_id, motion_id=args.motion)
    if sa_out is not None:
        print(f"Sa subplot figure: {sa_out}")

    hallal_fig = plot_hallal_profile_realizations(
        args.out_dir, sobol_id=args.sobol_id, seeds=[1, 2, 3, 4, 5]
    )
    print(f"Hallal profile figure: {hallal_fig}")

    grf_rows = df[(df["method"] == "grf_2d") & (df["sobol_id"] == args.sobol_id)]
    if not grf_rows.empty:
        h5_2d = args.h5_dir / f"run_{int(grf_rows.iloc[0]['index'])}.h5"
        grf_fig = plot_grf_2d_explainability(args.out_dir, h5_path=h5_2d)
        if grf_fig is not None:
            print(f"2D GRF explainability figure: {grf_fig}")
    print(f"Figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
