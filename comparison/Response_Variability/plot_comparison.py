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
    AHV_FIXED,
    BEDROCK_DEPTH,
    METHODS,
    NO_NX_FULL,
    PRETELL_SAMPLE_WIDTH_M,
    RH_FIXED,
    active_bc_width,
    active_dz,
    active_lx_var,
    active_rf_bc_width,
    active_rf_lx_var,
    pretell_column_indices,
)

from seiskit.intensity_measures import sigma_ln
from seiskit.plot_config import (
    COLORBLIND_COLORS,
    apply_style,
    get_crameri_cmap,
    panel_letter,
    place_legend,
)
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
    "grf_2d": "2D GIFNO",
    "opensees_2d": "2D OpenSees",
    "pretell": "Pretell",
    "hallal_vs": "Toro's Vs",
    "hallal_tts": "Passeri tts",
    "hallal_dmin": "Dmult sweep",
}

# High-contrast method colors (Wong-inspired; OpenSees = black reference).
METHOD_COLORS = {
    "opensees_2d": "#000000",
    "grf_2d": "#0072B2",  # blue
    "pretell": "#E69F00",  # amber
    "hallal_vs": "#009E73",  # green
    "hallal_tts": "#D55E00",  # vermillion
    "hallal_dmin": "#CC79A7",  # magenta
}
METHOD_LINESTYLES = {
    "opensees_2d": "-",
    "grf_2d": "-",
    "pretell": "-.",
    "hallal_vs": "--",
    "hallal_tts": ":",
    "hallal_dmin": (0, (3, 1, 1, 1)),
}

_SUMMARY_METHODS = (
    "grf_2d",
    "pretell",
    "hallal_vs",
    "hallal_tts",
    "hallal_dmin",
)

DEFAULT_PANEL_SOBOL_IDS = (19, 37, 36, 10, 44)

# Match AnalysisConfig.damping_zeta used for Hallal / base 1D runs.
_BASE_THEORY_XI = 0.025
_BASE_THEORY_RHO = 2000.0


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _vs_axis_on_top(ax) -> None:
    """Place $V_s$ ticks and axis label on the top of a depth–Vs profile axis."""
    ax.set_xlabel(r"$V_s$ (m/s)")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", which="both", top=True, labeltop=True, bottom=False, labelbottom=False)


def _common_plot_freq(df: pd.DataFrame, sobol_id: int, motion_id: str) -> np.ndarray:
    """Shared 0.1–10 Hz axis (prefer OpenSees 2D / GIFNO / pretell grid)."""
    for method in ("opensees_2d", "grf_2d", "pretell"):
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
    """Closed-form within TF from ``seiskit.theory`` (surface / soil–rock interface)."""
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
    """Plot closed-form 1D base-Vs TF and mark quarter-wave f₀."""
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
    """All methods on one axis: median |TF|(f) with ±1σ_ln band per method."""
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
    ax_tf, ax_sig = axes

    for method in methods:
        msub = sub[sub["method"] == method]
        if msub.empty:
            continue
        freqs = [np.asarray(row["freq"], dtype=float) for _, row in msub.iterrows()]
        af_list = [np.asarray(row["af"], dtype=float) for _, row in msub.iterrows()]
        af_stack = _align_af_stack(af_list, freqs, freq)
        med = np.nanmedian(af_stack, axis=0)
        sig = np.array(
            [
                sigma_ln(af_stack[:, j][np.isfinite(af_stack[:, j])])
                for j in range(af_stack.shape[1])
            ]
        )
        lo = med * np.exp(-sig)
        hi = med * np.exp(sig)
        color = METHOD_COLORS.get(method, COLORBLIND_COLORS[0])
        label = _method_label(method)
        lw = 2.4 if method in ("opensees_2d", "grf_2d") else 1.8
        ax_tf.fill_between(freq, lo, hi, color=color, alpha=0.18, linewidth=0)
        ax_tf.plot(freq, med, color=color, lw=lw, label=f"{label} (n={len(msub)})")
        ax_sig.plot(freq, sig, color=color, lw=lw, label=label)

    base = _load_base_1d_tf(df, sobol_id=sobol_id, motion_id=motion_id, freq=freq)
    _overlay_base_1d_and_f0(ax_tf, base=base, freq_plot=freq)
    if base is not None:
        ax_sig.axvline(base[2], color="0.2", ls=":", lw=1.4, alpha=0.8)

    ax_tf.set_xscale("log")
    ax_tf.set_xlim(0.1, 10.0)
    ax_tf.set_ylabel("|TF|")
    ax_tf.set_title(
        f"Transfer functions — Sobol #{sobol_id}  "
        f"Vs1={meta['vs1']:.0f} H={meta['H']:.0f} CoV={meta['cov']:.2f} Vs2={meta['vs2']:.0f}",
        loc="left",
        fontsize=10,
    )
    place_legend(ax_tf, position="bottom", fontsize=8)
    panel_letter(ax_tf, "a")

    ax_sig.set_xscale("log")
    ax_sig.set_yscale("log")
    ax_sig.set_xlim(0.1, 10.0)
    ax_sig.set_xlabel("Frequency (Hz)")
    ax_sig.set_ylabel(r"$\sigma_{\ln}$ TF")
    place_legend(ax_sig, position="bottom", fontsize=7, ncol=2)
    panel_letter(ax_sig, "b")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tf_methods_sobol{sobol_id:02d}_{motion_id}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_tf_method_subplots(
    df: pd.DataFrame, out_dir: Path, sobol_id: int = 0, motion_id: str = "M1"
) -> Path | None:
    """Primary comparison: one row per method — median |TF|(f) and σ_ln TF vs reference."""
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

    accent = COLORBLIND_COLORS[0]
    accent2 = COLORBLIND_COLORS[1]

    for i, method in enumerate(methods):
        ax_tf, ax_sig = axes[i]
        curves = _method_curves(df, method, sobol_id, motion_id, freq_dst=freq)
        if curves is None:
            ax_tf.set_visible(False)
            ax_sig.set_visible(False)
            continue

        label = _method_label(method)
        is_ref = method == ref_method
        mcolor = METHOD_COLORS.get(method, accent)

        ax_tf.plot(freq, ref["median_af"], color="0.35", lw=1.8, label=f"{ref_label} ref")
        if is_ref:
            ax_tf.fill_between(
                freq,
                curves["p16_af"],
                curves["p84_af"],
                color="0.5",
                alpha=0.25,
                label=f"16–84% ({curves['band_source']})",
            )
            ax_tf.plot(freq, curves["median_af"], color="k", lw=2.0, label=label)
        else:
            ax_tf.fill_between(
                freq,
                curves["p16_af"],
                curves["p84_af"],
                color=mcolor,
                alpha=0.2,
                label=f"16–84% ({curves['band_source']})",
            )
            ax_tf.plot(freq, curves["median_af"], color=mcolor, lw=2.0, label=label)
            delta = float(
                np.mean(
                    np.log(np.clip(curves["median_af"], 1e-12, None))
                    - np.log(np.clip(ref["median_af"], 1e-12, None))
                )
            )
            ax_tf.text(
                0.98,
                0.95,
                rf"$\Delta\mu_{{\ln}}$ TF = {delta:+.3f}",
                transform=ax_tf.transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )
        if i == 0 and base_overlay is not None:
            _overlay_base_1d_and_f0(ax_tf, base=base_overlay, freq_plot=freq)
        elif base_overlay is not None:
            ax_tf.axvline(base_overlay[2], color="0.2", ls=":", lw=1.2, alpha=0.7)
        ax_tf.set_xscale("log")
        ax_tf.set_ylabel("|TF|")
        ax_tf.set_yscale("log")
        ax_tf.set_title(f"{label}  (seeds={curves['n']})", loc="left", fontsize=10)
        place_legend(ax_tf, position="bottom", fontsize=7)
        panel_letter(ax_tf, chr(ord("a") + 2 * i))

        # Right: spatial σ_ln TF when available; else seed σ_ln; else ratio
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
                ax_sig.plot(freq, curves["sigma_ln_af"], color=accent2, lw=2.0, label=label)
                dsig = float(np.mean(curves["sigma_ln_af"] - ref_line))
                ax_sig.text(
                    0.98,
                    0.95,
                    rf"$\Delta\sigma_{{\ln}}$ = {dsig:+.3f}",
                    transform=ax_sig.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                )
            ylabel = r"$\sigma_{\ln}$ TF (spatial)" if has_spatial else r"$\sigma_{\ln}$ TF (seeds)"
            ax_sig.set_ylabel(ylabel)
        else:
            ratio = curves["median_af"] / np.clip(ref["median_af"], 1e-12, None)
            ax_sig.axhline(1.0, color="0.35", lw=1.2, label="ref")
            ax_sig.plot(freq, ratio, color=accent2, lw=2.0, label=label)
            ax_sig.set_ylabel(r"TF / TF$_{\mathrm{ref}}$")
        ax_sig.set_xscale("log")
        ax_sig.set_yscale("log")
        place_legend(ax_sig, position="bottom", fontsize=7)
        panel_letter(ax_sig, chr(ord("a") + 2 * i + 1))

    axes[-1, 0].set_xlabel("Frequency (Hz)")
    axes[-1, 1].set_xlabel("Frequency (Hz)")
    fig.suptitle(
        f"TF subplots vs {ref_label} — Sobol #{sobol_id} "
        f"Vs1={ref['vs1']:g} H={ref['H']:g} {motion_id}",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout()
    out_path = out_dir / f"tf_method_subplots_sobol{sobol_id:02d}_{motion_id}.png"
    fig.savefig(out_path, bbox_inches="tight")
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

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 5.0), sharey=True)
    panels = (
        ("Hallal VsRand (Toro full)", generate_vs_randomized_profile),
        ("Hallal ttsRand (Passeri full)", generate_tts_randomized_profile),
    )

    for ax, (title, gen_fn) in zip(axes, panels):
        ax.step(base_full, depth, color="k", lw=2.0, ls="--", label="Base case", where="pre")
        for i, seed in enumerate(seeds):
            rng = np.random.default_rng(seed)
            prof = gen_fn(cfg, rng)
            ax.step(
                prof,
                depth,
                color=COLORBLIND_COLORS[i % len(COLORBLIND_COLORS)],
                lw=1.4,
                alpha=0.85,
                label=f"seed {seed}",
                where="pre",
            )
        ax.axhline(case.H, color="0.5", ls=":", lw=1.0)
        _vs_axis_on_top(ax)
        ax.set_title(title, pad=10)
        ax.set_ylim(depth[-1] + dz, 0)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9)

    axes[0].set_ylabel("Depth (m)")
    panel_letter(axes[0], "a")
    panel_letter(axes[1], "b")
    fig.suptitle(
        f"1D profile randomization — Sobol #{sobol_id}  "
        f"Vs1={case.vs1:g} m/s, H={case.H:g} m, CoV={case.cov:.2f} (+ bedrock)",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout(pad=0.4, w_pad=0.6)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"hallal_profiles_sobol{sobol_id:02d}.png"
    fig.savefig(out_path, bbox_inches="tight")
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

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), gridspec_kw={"width_ratios": [1.35, 1.0]})
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
        label=f"Pretell {PRETELL_SAMPLE_WIDTH_M:.0f} m",
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
    place_legend(ax0, position="top", fontsize=8)

    ax1 = axes[1]
    ax1.fill_betweenx(depth, p16, p84, color="C0", alpha=0.2, label="16–84% across strip")
    ax1.plot(base_profile, depth_base, color="k", lw=2.0, ls="--", label="Base 1D template")
    ax1.plot(pretell_geo, depth, color="crimson", lw=2.2, label="Pretell geomean Vs")
    ax1.axhline(H, color="0.5", ls=":", lw=1.0)
    _vs_axis_on_top(ax1)
    ax1.set_ylabel("Depth (m)")
    ax1.set_ylim(nz * data["dz"], 0)
    ax1.set_title(
        f"Pretell 1D diagnostic\n(geomean of {len(pretell_cols)} profiles over full "
        f"{PRETELL_SAMPLE_WIDTH_M:.0f} m strip)",
        fontsize=10,
        pad=10,
    )
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax1.grid(True, alpha=0.25)

    fig.suptitle(
        f"2D GRF vs extracted 1D profile — Vs1={data['vs1']:g} H={H:g} m/s",
        fontsize=12,
        y=1.01,
    )
    fig.tight_layout(pad=0.4, w_pad=0.6)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"grf2d_explainability_Vs1{data['vs1']:.0f}_s{data['seed']}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_grf2d_vs_pretell_geomean(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    sobol_id: int,
    motion_id: str = "M1",
) -> Path | None:
    """Center-recorder TF geomean ± 1σ_ln — OpenSees 2D / GIFNO / Pretell."""
    methods = tuple(
        m
        for m in ("opensees_2d", "grf_2d", "pretell")
        if not df[
            (df["method"] == m) & (df["sobol_id"] == sobol_id) & (df["motion_id"] == motion_id)
        ].empty
    )
    if len(methods) < 2:
        return None
    curves: dict[str, dict] = {}
    freq_ref: np.ndarray | None = None
    for method in methods:
        sub = df[
            (df["method"] == method) & (df["sobol_id"] == sobol_id) & (df["motion_id"] == motion_id)
        ]
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
    n_seeds = min(c["n"] for c in curves.values())

    fig, axes = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]}
    )
    ax_tf, ax_ratio = axes

    styles = {
        "opensees_2d": {
            "color": METHOD_COLORS["opensees_2d"],
            "ls": "-",
            "lw": 2.4,
            "label": "opensees_2d baseline (geomean)",
        },
        "grf_2d": {
            "color": METHOD_COLORS["grf_2d"],
            "ls": "-",
            "lw": 2.2,
            "label": "grf_2d GIFNO (geomean)",
        },
        "pretell": {
            "color": METHOD_COLORS["pretell"],
            "ls": "--",
            "lw": 2.0,
            "label": "pretell OpenSees (geomean)",
        },
    }
    for method in methods:
        c = curves[method]
        st = styles[method]
        ax_tf.fill_between(freq, c["lo"], c["hi"], color=st["color"], alpha=0.18, linewidth=0)
        ax_tf.plot(freq, c["geo"], color=st["color"], ls=st["ls"], lw=st["lw"], label=st["label"])
        if n_seeds <= 10:
            for row_af in c["af_stack"]:
                ax_tf.plot(freq, row_af, color=st["color"], alpha=0.12, lw=0.6)

    base = _load_base_1d_tf(df, sobol_id=sobol_id, motion_id=motion_id, freq=freq)
    _overlay_base_1d_and_f0(ax_tf, base=base, freq_plot=freq)
    if base is not None:
        ax_ratio.axvline(base[2], color="0.2", ls=":", lw=1.4, alpha=0.8)

    ax_tf.set_xscale("log")
    ax_tf.set_xlim(0.1, 10.0)
    ax_tf.set_ylabel("|TF| (center recorder)")
    ax_tf.set_title(
        f"Center TF geomean ± 1σ_ln — Sobol #{sobol_id}  N={n_seeds} seeds  "
        f"Vs1={meta['vs1']:.0f} H={meta['H']:.0f} CoV={meta['cov']:.2f}",
        loc="left",
        fontsize=10,
    )
    place_legend(ax_tf, position="bottom", fontsize=8)
    panel_letter(ax_tf, "a")

    ref_key = "opensees_2d" if "opensees_2d" in curves else methods[0]
    ax_ratio.axhline(1.0, color="0.35", lw=1.2)
    for method in methods:
        if method == ref_key:
            continue
        ratio = curves[method]["geo"] / np.clip(curves[ref_key]["geo"], 1e-12, None)
        ax_ratio.plot(
            freq,
            ratio,
            color=styles[method]["color"],
            lw=2.0,
            label=f"{method} / {ref_key}",
        )
    ax_ratio.set_xscale("log")
    ax_ratio.set_xlabel("Frequency (Hz)")
    ax_ratio.set_ylabel("TF ratio")
    place_legend(ax_ratio, position="bottom", fontsize=8)
    panel_letter(ax_ratio, "b")

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tf_grf2d_vs_pretell_geomean_sobol{sobol_id:02d}_{motion_id}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _find_2d_h5_paths_for_sobol(df: pd.DataFrame, h5_dir: Path, sobol_id: int) -> list[Path]:
    """All 2D field H5s for a Sobol case (prefer OpenSees; fall back to GIFNO)."""
    paths: list[Path] = []
    for method in ("opensees_2d", "grf_2d"):
        sub = df[(df["method"] == method) & (df["sobol_id"] == sobol_id)]
        if sub.empty:
            continue
        for idx in sorted(int(i) for i in sub["index"].unique()):
            path = h5_dir / f"run_{idx}.h5"
            if path.is_file():
                paths.append(path)
        if paths:
            return paths
    return paths


def _load_1d_vs_profiles_from_h5(
    df: pd.DataFrame,
    h5_dir: Path,
    *,
    method: str,
    sobol_id: int,
    motion_id: str = "M1",
) -> tuple[list[np.ndarray], float]:
    """Load every Vs_profile_1D for method×sobol (all seeds). Returns (profiles, dz)."""
    import h5py

    sub = df[
        (df["method"] == method) & (df["sobol_id"] == sobol_id) & (df["motion_id"] == motion_id)
    ]
    if sub.empty:
        sub = df[(df["method"] == method) & (df["sobol_id"] == sobol_id)]
    profiles: list[np.ndarray] = []
    dz = float(active_dz())
    for idx in sorted(int(i) for i in sub["index"].unique()):
        path = h5_dir / f"run_{idx}.h5"
        if not path.is_file():
            continue
        with h5py.File(path, "r") as f:
            profiles.append(np.asarray(f["Vs_profile_1D"][:], dtype=float))
            if "grid" in f and "dz" in f["grid"].attrs:
                dz = float(f["grid"].attrs["dz"])
    return profiles, dz


def _load_pretell_column_profiles(
    df: pd.DataFrame,
    h5_dir: Path,
    *,
    sobol_id: int,
) -> tuple[list[np.ndarray], float]:
    """All Pretell-sampled columns from every 2D RF realization for this Sobol."""
    paths = _find_2d_h5_paths_for_sobol(df, h5_dir, sobol_id)
    profiles: list[np.ndarray] = []
    dz = 1.0
    for path in paths:
        data = _load_2d_from_h5(path)
        dz = float(data["dz"])
        vs_var, _, strip_cols = _variable_domain_slice(data["vs_field"], data["dx"])
        for j in strip_cols:
            j = int(j)
            if 0 <= j < vs_var.shape[1]:
                profiles.append(np.asarray(vs_var[:, j], dtype=float))
    return profiles, dz


def _geomean_vs_profiles(profiles: list[np.ndarray]) -> np.ndarray | None:
    if not profiles:
        return None
    n = min(len(p) for p in profiles)
    stack = np.vstack([np.asarray(p[:n], dtype=float) for p in profiles])
    return np.exp(np.mean(np.log(np.clip(stack, 1e-6, None)), axis=0))


def _draw_vs_swarm(
    ax,
    *,
    base: np.ndarray,
    depth_base: np.ndarray,
    profiles: list[np.ndarray],
    depth_prof: np.ndarray,
    color: str,
    swarm_label: str,
    H: float,
    share_ylabel: bool = False,
) -> None:
    from matplotlib.collections import LineCollection

    ax.step(base, depth_base, color="k", lw=2.4, ls="--", label="Base Vs", where="pre", zorder=6)
    if profiles:
        segs = []
        for prof in profiles:
            n = min(len(prof), len(depth_prof))
            # steps-pre: repeat each Vs at layer top/bottom for a stair look
            x = np.asarray(prof[:n], dtype=float)
            y = np.asarray(depth_prof[:n], dtype=float)
            xs = np.repeat(x, 2)
            ys = np.empty_like(xs)
            ys[0::2] = y
            ys[1::2] = np.concatenate([y[1:], y[-1:]])
            segs.append(np.column_stack([xs, ys]))
        lc = LineCollection(
            segs,
            colors=color,
            linewidths=0.4,
            alpha=0.6,
            rasterized=True,
            zorder=2,
        )
        ax.add_collection(lc)
        geo = _geomean_vs_profiles(profiles)
        if geo is not None:
            ax.step(
                geo,
                depth_prof[: len(geo)],
                color=color,
                lw=2.4,
                alpha=1.0,
                label=f"Geomean (n={len(profiles)})",
                where="pre",
                zorder=5,
            )
        ax.plot([], [], color=color, lw=0.8, alpha=0.6, label=swarm_label)
        # Autoscale after LineCollection (collections are ignored by some autoscalers).
        xmax = max(float(np.max(p)) for p in profiles)
        ax.set_xlim(0.0, xmax * 1.05)
    ax.axhline(H, color="0.45", ls=":", lw=1.0)
    _vs_axis_on_top(ax)
    if share_ylabel:
        ax.set_ylabel("Depth (m)")
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.9)


def plot_profile_tf_panel(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    h5_dir: Path,
    sobol_id: int,
    motion_id: str = "M1",
) -> Path | None:
    """Top row: Vs profiles (shared depth); bottom: full-width TF (all seeds)."""
    from seiskit.profile_randomization import build_base_case_profile

    case = _sobol_case(sobol_id)
    cfg = _profile_config(case.vs1, H=case.H, cov=case.cov, vs2=case.vs2)
    dz_1d = active_dz()
    base_full = build_base_case_profile(cfg)
    depth_base = (np.arange(len(base_full)) + 0.5) * dz_1d

    toro_profs, dz_toro = _load_1d_vs_profiles_from_h5(
        df, h5_dir, method="hallal_vs", sobol_id=sobol_id, motion_id=motion_id
    )
    passeri_profs, dz_pas = _load_1d_vs_profiles_from_h5(
        df, h5_dir, method="hallal_tts", sobol_id=sobol_id, motion_id=motion_id
    )
    # Fall back to regenerating the production seed list if H5s are missing.
    if not toro_profs:
        from manifest import active_hallal_seeds

        toro_profs = [
            generate_vs_randomized_profile(cfg, np.random.default_rng(s))
            for s in active_hallal_seeds()
        ]
        dz_toro = dz_1d
    if not passeri_profs:
        from manifest import active_hallal_seeds

        passeri_profs = [
            generate_tts_randomized_profile(cfg, np.random.default_rng(s))
            for s in active_hallal_seeds()
        ]
        dz_pas = dz_1d

    pretell_profs, dz_pt = _load_pretell_column_profiles(df, h5_dir, sobol_id=sobol_id)

    depth_toro = (
        np.arange(max((len(p) for p in toro_profs), default=len(base_full))) + 0.5
    ) * dz_toro
    depth_pas = (
        np.arange(max((len(p) for p in passeri_profs), default=len(base_full))) + 0.5
    ) * dz_pas
    depth_pt = (
        (np.arange(max((len(p) for p in pretell_profs), default=len(base_full))) + 0.5) * dz_pt
        if pretell_profs
        else depth_base
    )
    z_max = max(
        float(depth_base[-1]),
        float(depth_toro[-1]) if len(depth_toro) else 0.0,
        float(depth_pas[-1]) if len(depth_pas) else 0.0,
        float(depth_pt[-1]) if len(depth_pt) else 0.0,
    )

    fig = plt.figure(figsize=(12.0, 7.6))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 1.2],
        hspace=0.10,
        wspace=0.10,
        left=0.055,
        right=0.995,
        top=0.88,
        bottom=0.08,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
    ax_c = fig.add_subplot(gs[0, 2], sharey=ax_a)
    ax_d = fig.add_subplot(gs[1, :])

    _draw_vs_swarm(
        ax_a,
        base=base_full,
        depth_base=depth_base,
        profiles=toro_profs,
        depth_prof=depth_toro,
        color=METHOD_COLORS["hallal_vs"],
        swarm_label=f"Toro seeds (n={len(toro_profs)})",
        H=case.H,
        share_ylabel=True,
    )
    ax_a.set_title("Toro's Vs", loc="left", fontsize=10, pad=10)
    panel_letter(ax_a, "a")

    _draw_vs_swarm(
        ax_b,
        base=base_full,
        depth_base=depth_base,
        profiles=passeri_profs,
        depth_prof=depth_pas,
        color=METHOD_COLORS["hallal_tts"],
        swarm_label=f"Passeri seeds (n={len(passeri_profs)})",
        H=case.H,
    )
    ax_b.set_title("Passeri tts", loc="left", fontsize=10, pad=10)
    panel_letter(ax_b, "b")
    plt.setp(ax_b.get_yticklabels(), visible=False)

    _draw_vs_swarm(
        ax_c,
        base=base_full,
        depth_base=depth_base,
        profiles=pretell_profs,
        depth_prof=depth_pt,
        color=METHOD_COLORS["pretell"],
        swarm_label=f"Pretell cols (n={len(pretell_profs)})",
        H=case.H,
    )
    ax_c.set_title("Pretell columns vs base Vs", loc="left", fontsize=10, pad=10)
    panel_letter(ax_c, "c")
    plt.setp(ax_c.get_yticklabels(), visible=False)
    if not pretell_profs:
        ax_c.text(0.5, 0.5, "No 2D H5 for Pretell columns", transform=ax_c.transAxes, ha="center")

    ax_a.set_ylim(z_max, 0)

    freq = _common_plot_freq(df, sobol_id, motion_id)
    # Draw OpenSees last so the reference sits on top.
    plot_order = [m for m in METHODS if m != "opensees_2d"] + ["opensees_2d"]
    plotted = False
    for method in plot_order:
        sub = df[
            (df["method"] == method) & (df["sobol_id"] == sobol_id) & (df["motion_id"] == motion_id)
        ]
        if sub.empty:
            continue
        freqs = [np.asarray(row["freq"], dtype=float) for _, row in sub.iterrows()]
        af_list = [np.asarray(row["af"], dtype=float) for _, row in sub.iterrows()]
        stack = _align_af_stack(af_list, freqs, freq)
        geo = np.clip(_geomean_af(stack), 1e-3, None)
        sig = _sigma_ln_af_stack(stack)
        color = METHOD_COLORS.get(method, COLORBLIND_COLORS[0])
        lw = 3.0 if method == "opensees_2d" else 2.0
        ls = METHOD_LINESTYLES.get(method, "-")
        ax_d.fill_between(
            freq,
            np.clip(geo * np.exp(-sig), 1e-3, None),
            geo * np.exp(sig),
            color=color,
            alpha=0.18 if method == "opensees_2d" else 0.10,
            linewidth=0,
            zorder=2 if method == "opensees_2d" else 1,
        )
        ax_d.plot(
            freq,
            geo,
            color=color,
            lw=lw,
            ls=ls,
            label=f"{_method_label(method)} (n={len(sub)})",
            zorder=6 if method == "opensees_2d" else 3,
        )
        plotted = True

    if plotted:
        ax_d.set_xscale("log")
        ax_d.set_yscale("log")
        ax_d.set_xlim(0.1, 10.0)
        ax_d.set_ylabel("|TF|")
        ax_d.set_xlabel("Frequency (Hz)")
        ax_d.set_title("TF geomean ± 1σ_ln (ref: 2D OpenSees)", loc="left", fontsize=11)
        ax_d.legend(fontsize=7, loc="lower left", framealpha=0.9, ncol=3)
    else:
        ax_d.text(0.5, 0.5, "No TF data", transform=ax_d.transAxes, ha="center")
    panel_letter(ax_d, "d")

    fig.suptitle(
        f"Sobol #{sobol_id}  Vs1={case.vs1:.0f} m/s  H={case.H:.1f} m  "
        f"CoV={case.cov:.2f}  Vs2={case.vs2:.0f} m/s  "
        f"(aHV={AHV_FIXED:g}, rH={RH_FIXED:g} m)  {motion_id}",
        fontsize=12,
        y=0.98,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"profile_tf_panel_sobol{sobol_id:02d}_{motion_id}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return float("nan")
    denom = np.linalg.norm(b[mask])
    if denom < 1e-12:
        return float("nan")
    return float(np.linalg.norm(a[mask] - b[mask]) / denom)


def compute_tf_band_misfit(
    df: pd.DataFrame,
    *,
    motion_id: str = "M1",
    ref_method: str = "opensees_2d",
) -> pd.DataFrame:
    """Per Sobol × method geomean / σ_ln band misfit vs OpenSees 2D."""
    rows: list[dict] = []
    sobols = sorted(int(s) for s in df["sobol_id"].unique())
    for sobol_id in sobols:
        ref_sub = df[
            (df["method"] == ref_method)
            & (df["sobol_id"] == sobol_id)
            & (df["motion_id"] == motion_id)
        ]
        if ref_sub.empty:
            continue
        freq = _common_plot_freq(df, sobol_id, motion_id)
        ref_freqs = [np.asarray(r["freq"], dtype=float) for _, r in ref_sub.iterrows()]
        ref_afs = [np.asarray(r["af"], dtype=float) for _, r in ref_sub.iterrows()]
        ref_stack = _align_af_stack(ref_afs, ref_freqs, freq)
        ref_geo = _geomean_af(ref_stack)
        ref_sig = _sigma_ln_af_stack(ref_stack)
        meta = ref_sub.iloc[0]
        for method in _SUMMARY_METHODS:
            msub = df[
                (df["method"] == method)
                & (df["sobol_id"] == sobol_id)
                & (df["motion_id"] == motion_id)
            ]
            if msub.empty:
                continue
            freqs = [np.asarray(r["freq"], dtype=float) for _, r in msub.iterrows()]
            afs = [np.asarray(r["af"], dtype=float) for _, r in msub.iterrows()]
            stack = _align_af_stack(afs, freqs, freq)
            geo = _geomean_af(stack)
            sig = _sigma_ln_af_stack(stack)
            rows.append(
                {
                    "sobol_id": sobol_id,
                    "vs1": float(meta["vs1"]),
                    "H": float(meta["H"]),
                    "cov": float(meta["cov"]) if "cov" in meta.index else float("nan"),
                    "motion_id": motion_id,
                    "method": method,
                    "ref_method": ref_method,
                    "n_realizations": len(msub),
                    "rel_l2_geomean": _rel_l2(geo, ref_geo),
                    "rel_l2_sigma_ln": _rel_l2(sig, ref_sig),
                }
            )
    return pd.DataFrame(rows)


def _load_method_summary(analysis_dir: Path, motion_id: str = "M1") -> pd.DataFrame | None:
    path = analysis_dir / "method_comparison_summary.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if "motion_id" in df.columns:
        df = df[df["motion_id"] == motion_id]
    if {"method", "ref_method"}.issubset(df.columns):
        df = df[df["method"] != df["ref_method"]]
    if "ref_method" in df.columns and (df["ref_method"] == "opensees_2d").any():
        df = df[df["ref_method"] == "opensees_2d"]
    return df


def _attach_sobol_params(df: pd.DataFrame) -> pd.DataFrame:
    """Join CoV / Vs2 / aHV / rH from the Sobol catalog onto a summary table."""
    from manifest import active_base_cases

    meta = pd.DataFrame(
        [
            {
                "sobol_id": int(c.sobol_id),
                "cov": float(c.cov),
                "vs2": float(c.vs2),
                "aHV": float(getattr(c, "aHV", AHV_FIXED)),
                "rH": float(getattr(c, "rH", RH_FIXED)),
            }
            for c in active_base_cases()
        ]
    )
    keep = df.drop(
        columns=[c for c in ("cov", "vs2", "aHV", "rH") if c in df.columns], errors="ignore"
    )
    return keep.merge(meta, on="sobol_id", how="left")


def _summary_methods_present(df: pd.DataFrame) -> list[str]:
    present = set(df["method"].unique())
    return [m for m in _SUMMARY_METHODS if m in present]


def _box_by_method(
    df: pd.DataFrame,
    *,
    value_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
) -> Path | None:
    methods = _summary_methods_present(df)
    if not methods or value_col not in df.columns:
        return None
    data = [df.loc[df["method"] == m, value_col].dropna().to_numpy(dtype=float) for m in methods]
    if not any(len(d) for d in data):
        return None

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    bp = ax.boxplot(
        data,
        tick_labels=[_method_label(m) for m in methods],
        patch_artist=True,
        showfliers=False,
    )
    for patch, method in zip(bp["boxes"], methods):
        patch.set_facecolor(METHOD_COLORS.get(method, COLORBLIND_COLORS[0]))
        patch.set_alpha(0.55)
    rng = np.random.default_rng(0)
    for i, (method, vals) in enumerate(zip(methods, data), start=1):
        if vals.size == 0:
            continue
        x = i + 0.08 * rng.standard_normal(vals.size)
        ax.scatter(
            x,
            vals,
            s=14,
            alpha=0.55,
            color=METHOD_COLORS.get(method, COLORBLIND_COLORS[0]),
            zorder=3,
            edgecolors="none",
        )
    ax.axhline(0.0, color="0.35", lw=1.0, ls=":")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.tick_params(axis="x", labelrotation=15)
    panel_letter(ax, "a")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_tf_peak_freq_bias_all_sobol(summary: pd.DataFrame, out_dir: Path) -> Path | None:
    return _box_by_method(
        summary,
        value_col="delta_f_peak",
        ylabel=r"$\Delta f_{\mathrm{peak}}$ (Hz) vs 2D OpenSees",
        title="Fundamental-mode frequency bias — all Sobol cases",
        out_path=out_dir / "tf_peak_freq_bias_all_sobol.png",
    )


def plot_tf_peak_amp_bias_all_sobol(summary: pd.DataFrame, out_dir: Path) -> Path | None:
    return _box_by_method(
        summary,
        value_col="delta_ln_A_peak",
        ylabel=r"$\Delta\ln$ |TF|$_{\mathrm{peak}}$ vs 2D OpenSees",
        title="Fundamental-mode amplitude bias — all Sobol cases",
        out_path=out_dir / "tf_peak_amp_bias_all_sobol.png",
    )


def plot_tf_band_misfit_all_sobol(misfit: pd.DataFrame, out_dir: Path) -> Path | None:
    methods = _summary_methods_present(misfit)
    if not methods:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    specs = (
        ("rel_l2_geomean", r"rel. $L_2$ geomean TF", axes[0], "a"),
        ("rel_l2_sigma_ln", r"rel. $L_2$ $\sigma_{\ln}$ TF", axes[1], "b"),
    )
    for col, ylabel, ax, letter in specs:
        data = [
            misfit.loc[misfit["method"] == m, col].dropna().to_numpy(dtype=float) for m in methods
        ]
        bp = ax.boxplot(
            data,
            tick_labels=[_method_label(m) for m in methods],
            patch_artist=True,
            showfliers=False,
        )
        for patch, method in zip(bp["boxes"], methods):
            patch.set_facecolor(METHOD_COLORS.get(method, COLORBLIND_COLORS[0]))
            patch.set_alpha(0.55)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " vs 2D OpenSees", loc="left", fontsize=10)
        ax.tick_params(axis="x", labelrotation=15)
        panel_letter(ax, letter)
    fig.suptitle("TF band misfit vs 2D OpenSees — all Sobol cases", fontsize=12, y=1.02)
    fig.tight_layout()
    out_path = out_dir / "tf_band_misfit_all_sobol.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_tf_error_vs_sobol_params(
    summary: pd.DataFrame,
    misfit: pd.DataFrame | None,
    out_dir: Path,
) -> Path | None:
    """Two param planes: Vs1–H (|δf|) and CoV–Vs2 (|δln A|); colorbars outside right."""
    methods = _summary_methods_present(summary)
    needed = {"vs1", "H", "delta_f_peak", "delta_ln_A_peak"}
    if not methods or not needed.issubset(summary.columns):
        return None

    summary = _attach_sobol_params(summary)

    n = len(methods)
    fig = plt.figure(figsize=(3.0 * n + 1.2, 6.6))
    # Extra column on the right for colorbars (outside data axes).
    gs = fig.add_gridspec(
        2,
        n + 1,
        width_ratios=[*[1.0] * n, 0.055],
        left=0.07,
        right=0.97,
        top=0.90,
        bottom=0.08,
        wspace=0.18,
        hspace=0.28,
    )
    axes = np.empty((2, n), dtype=object)
    for r in range(2):
        for c in range(n):
            axes[r, c] = fig.add_subplot(gs[r, c])
    cax0 = fig.add_subplot(gs[0, -1])
    cax1 = fig.add_subplot(gs[1, -1])

    cmap = get_crameri_cmap("batlow")

    abs_df = summary["delta_f_peak"].abs().to_numpy(dtype=float)
    abs_df = abs_df[np.isfinite(abs_df)]
    vmin0 = float(np.nanpercentile(abs_df, 5)) if abs_df.size else 0.0
    vmax0 = float(np.nanpercentile(abs_df, 95)) if abs_df.size else 1.0
    if vmin0 == vmax0:
        vmax0 = vmin0 + 1e-3
    last_sc0 = None
    for i, method in enumerate(methods):
        ax = axes[0, i]
        sub = summary[summary["method"] == method]
        last_sc0 = ax.scatter(
            sub["vs1"],
            sub["H"],
            c=sub["delta_f_peak"].abs(),
            cmap=cmap,
            vmin=vmin0,
            vmax=vmax0,
            s=40,
            edgecolors="0.2",
            linewidths=0.3,
        )
        ax.set_title(_method_label(method), loc="left", fontsize=9)
        panel_letter(ax, chr(ord("a") + i))
        if i == 0:
            ax.set_ylabel(r"$H$ (m)")
        ax.set_xlabel(r"$V_{s1}$ (m/s)")

    abs_a = summary["delta_ln_A_peak"].abs().to_numpy(dtype=float)
    abs_a = abs_a[np.isfinite(abs_a)]
    vmin1 = float(np.nanpercentile(abs_a, 5)) if abs_a.size else 0.0
    vmax1 = float(np.nanpercentile(abs_a, 95)) if abs_a.size else 1.0
    if vmin1 == vmax1:
        vmax1 = vmin1 + 1e-3
    last_sc1 = None
    for i, method in enumerate(methods):
        ax = axes[1, i]
        sub = summary[summary["method"] == method]
        last_sc1 = ax.scatter(
            sub["cov"],
            sub["vs2"],
            c=sub["delta_ln_A_peak"].abs(),
            cmap=cmap,
            vmin=vmin1,
            vmax=vmax1,
            s=40,
            edgecolors="0.2",
            linewidths=0.3,
        )
        ax.set_title(_method_label(method), loc="left", fontsize=9)
        panel_letter(ax, chr(ord("a") + n + i))
        if i == 0:
            ax.set_ylabel(r"$V_{s2}$ (m/s)")
        ax.set_xlabel(r"CoV")

    if last_sc0 is not None:
        cbar0 = fig.colorbar(last_sc0, cax=cax0)
        cbar0.set_label(r"$|\Delta f_{\mathrm{peak}}|$ (Hz)")
    if last_sc1 is not None:
        cbar1 = fig.colorbar(last_sc1, cax=cax1)
        cbar1.set_label(r"$|\Delta\ln$ |TF|$_{\mathrm{peak}}|$")

    fig.text(
        0.015,
        0.72,
        "Row 1: Vs1–H  (freq. bias)",
        rotation=90,
        va="center",
        ha="center",
        fontsize=9,
        color="0.25",
    )
    fig.text(
        0.015,
        0.30,
        "Row 2: CoV–Vs2  (amp. bias)",
        rotation=90,
        va="center",
        ha="center",
        fontsize=9,
        color="0.25",
    )
    fig.suptitle(
        f"Error vs Sobol params (GT: 2D OpenSees); fixed aHV={AHV_FIXED:g}, rH={RH_FIXED:g} m",
        fontsize=12,
        y=0.98,
    )
    out_path = out_dir / "tf_error_vs_sobol_params.png"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_all_sobol_summaries(
    analysis_dir: Path,
    out_dir: Path,
    *,
    motion_id: str = "M1",
    misfit: pd.DataFrame | None = None,
) -> list[Path]:
    summary = _load_method_summary(analysis_dir, motion_id=motion_id)
    if summary is None or summary.empty:
        print(f"No method_comparison_summary.csv in {analysis_dir}; skipping summaries.")
        return []
    written: list[Path] = []
    for fn in (
        plot_tf_peak_freq_bias_all_sobol,
        plot_tf_peak_amp_bias_all_sobol,
    ):
        path = fn(summary, out_dir)
        if path is not None:
            written.append(path)
            print(f"Sobol summary: {path}")
    if misfit is not None and not misfit.empty:
        path = plot_tf_band_misfit_all_sobol(misfit, out_dir)
        if path is not None:
            written.append(path)
            print(f"Sobol summary: {path}")
    path = plot_tf_error_vs_sobol_params(summary, misfit, out_dir)
    if path is not None:
        written.append(path)
        print(f"Sobol summary: {path}")
    # Drop legacy GOF heatmap if present.
    legacy_gof = out_dir / "tf_gof_heatmap_all_sobol.png"
    if legacy_gof.is_file():
        legacy_gof.unlink()
        print(f"Removed legacy GOF heatmap: {legacy_gof}")
    return written


def _parse_sobol_ids(raw: str | None, single: int | None) -> list[int]:
    if raw:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]
    if single is not None:
        return [int(single)]
    return list(DEFAULT_PANEL_SOBOL_IDS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-dir", type=Path, default=Path("results/h5"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--analysis-dir", type=Path, default=Path("results/analysis"))
    parser.add_argument("--sobol-id", type=int, default=None, help="Single Sobol id (optional)")
    parser.add_argument(
        "--sobol-ids",
        type=str,
        default=None,
        help="Comma-separated Sobol ids for profile/TF panels (default: 19,37,36,10,44)",
    )
    parser.add_argument("--motion", type=str, default="M1")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only write all-Sobol metric figures (skip per-Sobol panels).",
    )
    args = parser.parse_args()
    sobol_ids = _parse_sobol_ids(args.sobol_ids, args.sobol_id)

    df = collect_rows(args.h5_dir)
    if df.empty:
        print("No data to plot.")
        return

    misfit = compute_tf_band_misfit(df, motion_id=args.motion)
    if not misfit.empty:
        args.analysis_dir.mkdir(parents=True, exist_ok=True)
        misfit_path = args.analysis_dir / "tf_band_misfit_vs_opensees.csv"
        misfit.to_csv(misfit_path, index=False)
        print(f"Wrote {misfit_path}")

    if not args.summary_only:
        for sid in sobol_ids:
            out = plot_profile_tf_panel(
                df,
                args.out_dir,
                h5_dir=args.h5_dir,
                sobol_id=sid,
                motion_id=args.motion,
            )
            if out is not None:
                print(f"Profile/TF panel: {out}")

    plot_all_sobol_summaries(
        args.analysis_dir,
        args.out_dir,
        motion_id=args.motion,
        misfit=misfit if not misfit.empty else None,
    )
    print(f"Figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
