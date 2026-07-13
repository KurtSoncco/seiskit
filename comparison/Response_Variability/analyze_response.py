"""
Aggregate Response_Variability HDF5 outputs and compare methods vs 2D GRF reference.

Primary comparison is transfer-function based: AF(f) is an intrinsic site property
under linear viscoelasticity.

Usage (from comparison/Response_Variability):
  python analyze_response.py
  python analyze_response.py --h5-dir results/h5 --out-dir results/analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from seiskit.gof import anderson_frequency_domain, log_residual_bias
from seiskit.intensity_measures import sigma_ln


def _load_h5(path: Path) -> dict:
    import h5py

    with h5py.File(path, "r") as f:
        params = f["params"]
        method = f.attrs.get("method", "")
        motion_id = f.attrs.get("motion_id", "")
        pga = float(f["ims"].attrs.get("PGA_surface", 0.0))
        periods = f["ims"]["Sa_periods"][:]
        sa = f["ims"]["Sa_surface"][:]
        freq = f["transfer_function"]["freq"][:]
        af = f["transfer_function"]["AF"][:]
        out = {
            "method": method,
            "motion_id": motion_id,
            "sobol_id": int(params.attrs.get("sobol_id", 0)),
            "vs1": float(params.attrs["Vs1"]),
            "H": float(params.attrs.get("H", 15.0)),
            "vs2": float(params.attrs.get("Vs2", 1500.0)),
            "cov": float(params.attrs.get("CoV", 0.2)),
            "seed": int(params.attrs["seed"]),
            "seed_kind": str(params.attrs.get("seed_kind", "realization")),
            "pga": pga,
            "periods": periods,
            "sa": sa,
            "freq": freq,
            "af": af,
            "af_spatial_p16": None,
            "af_spatial_p84": None,
            "af_spatial_median": None,
            "af_spatial_sigma_ln": None,
            "n_spatial_nodes": 0,
        }
        tf = f["transfer_function"]
        if "AF_spatial_p16" in tf:
            out["af_spatial_p16"] = tf["AF_spatial_p16"][:]
            out["af_spatial_p84"] = tf["AF_spatial_p84"][:]
            out["af_spatial_median"] = tf["AF_spatial_median"][:]
            out["af_spatial_sigma_ln"] = tf["AF_spatial_sigma_ln"][:]
            out["n_spatial_nodes"] = int(tf.attrs.get("n_spatial_nodes", 0))
    return out


def collect_rows(h5_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(h5_dir.glob("run_*.h5")):
        try:
            idx = int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        try:
            d = _load_h5(path)
            d["index"] = idx
            rows.append(d)
        except Exception as e:
            print(f"skip {path}: {e}")
    return pd.DataFrame(rows)


def _interp_af(freq_src: np.ndarray, af: np.ndarray, freq_dst: np.ndarray) -> np.ndarray:
    """Linearly interpolate |AF| onto a common frequency grid."""
    fn = interp1d(freq_src, af, kind="linear", bounds_error=False, fill_value=np.nan)
    return fn(freq_dst).astype(np.float32)


def _align_af_stack(
    af_stack: list[np.ndarray], freqs: list[np.ndarray], freq_ref: np.ndarray
) -> np.ndarray:
    rows = [_interp_af(f, a, freq_ref) for f, a in zip(freqs, af_stack)]
    return np.vstack(rows)


def _peak_af(
    freq: np.ndarray,
    af: np.ndarray,
    *,
    fmin: float = 0.1,
    fmax: float = 10.0,
) -> tuple[float, float]:
    """Peak |AF| within the comparison band (default 0.1–10 Hz)."""
    freq = np.asarray(freq, dtype=float)
    af = np.asarray(af, dtype=float)
    mask = (freq >= fmin) & (freq <= fmax) & np.isfinite(af)
    if not np.any(mask):
        i = int(np.nanargmax(af))
        return float(freq[i]), float(af[i])
    i = int(np.argmax(af[mask]))
    return float(freq[mask][i]), float(af[mask][i])


def _geomean_af(af_stack: np.ndarray) -> np.ndarray:
    """Geometric mean of AF across seeds at each frequency."""
    clipped = np.clip(af_stack, 1e-12, None)
    return np.exp(np.nanmean(np.log(clipped), axis=0))


def _sigma_ln_af_stack(af_stack: np.ndarray) -> np.ndarray:
    return np.array([sigma_ln(af_stack[:, j]) for j in range(af_stack.shape[1])])


def _geomean_band(geo: np.ndarray, sig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return geo * np.exp(-sig), geo * np.exp(sig)


REF_METHOD_PRIORITY = ("grf_2d", "hallal_vs")


def pick_reference_method(
    df: pd.DataFrame,
    sobol_id: int,
    motion_id: str,
) -> str | None:
    """Prefer 2D GRF reference; fall back to hallal_vs for 1D-only smoke."""
    for method in REF_METHOD_PRIORITY:
        sub = df[
            (df["method"] == method) & (df["sobol_id"] == sobol_id) & (df["motion_id"] == motion_id)
        ]
        if not sub.empty:
            return method
    return None


def reference_curves(
    df: pd.DataFrame,
    sobol_id: int,
    motion_id: str,
    *,
    ref_method: str | None = None,
) -> dict:
    ref_method = ref_method or pick_reference_method(df, sobol_id, motion_id)
    if ref_method is None:
        return {}
    ref = df[
        (df["method"] == ref_method) & (df["sobol_id"] == sobol_id) & (df["motion_id"] == motion_id)
    ]
    if ref.empty:
        return {}
    sa_stack = np.vstack(ref["sa"].tolist())
    af_stack = np.vstack(ref["af"].tolist())
    med_sa = np.median(sa_stack, axis=0)
    sig_sa = np.array([sigma_ln(sa_stack[:, j]) for j in range(sa_stack.shape[1])])
    med_af = np.median(af_stack, axis=0)
    sig_af = np.array([sigma_ln(af_stack[:, j]) for j in range(af_stack.shape[1])])
    freq = ref.iloc[0]["freq"]
    periods = ref.iloc[0]["periods"]
    f_peak, a_peak = _peak_af(freq, med_af)
    return {
        "ref_method": ref_method,
        "sobol_id": sobol_id,
        "vs1": float(ref.iloc[0]["vs1"]),
        "H": float(ref.iloc[0]["H"]),
        "periods": periods,
        "median_sa": med_sa,
        "sigma_ln_sa": sig_sa,
        "freq": freq,
        "median_af": med_af,
        "sigma_ln_af": sig_af,
        "f_peak": f_peak,
        "A_peak": a_peak,
        "pga_median": float(np.median(ref["pga"])),
        "pga_sigma_ln": sigma_ln(ref["pga"].values),
    }


def summarize_methods(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for sobol_id in sorted(df["sobol_id"].unique()):
        for motion in sorted(df["motion_id"].unique()):
            ref = reference_curves(df, sobol_id, motion)
            if not ref:
                continue
            ref_method = ref["ref_method"]
            for method in sorted(df["method"].unique()):
                sub = df[
                    (df["method"] == method)
                    & (df["sobol_id"] == sobol_id)
                    & (df["motion_id"] == motion)
                ]
                if sub.empty:
                    continue
                sa_stack = np.vstack(sub["sa"].tolist())
                med_sa = np.median(sa_stack, axis=0)
                sig_sa = np.array([sigma_ln(sa_stack[:, j]) for j in range(sa_stack.shape[1])])
                af_stack = np.vstack(sub["af"].tolist())
                med_af = np.median(af_stack, axis=0)
                sig_af = np.array([sigma_ln(af_stack[:, j]) for j in range(af_stack.shape[1])])
                freq_m = np.asarray(sub.iloc[0]["freq"], dtype=float)
                f_peak, a_peak = _peak_af(freq_m, med_af)
                # Align to reference frequency axis for residual / GoF metrics.
                med_af_ref = _interp_af(freq_m, med_af, ref["freq"])
                sig_af_ref = _interp_af(freq_m, sig_af, ref["freq"])

                records.append(
                    {
                        "sobol_id": sobol_id,
                        "vs1": ref["vs1"],
                        "H": ref["H"],
                        "motion_id": motion,
                        "method": method,
                        "ref_method": ref_method,
                        "n_realizations": len(sub),
                        "delta_mu_ln_af_mean": log_residual_bias(ref["median_af"], med_af_ref),
                        "delta_sigma_ln_af_mean": float(
                            np.nanmean(sig_af_ref - ref["sigma_ln_af"])
                        ),
                        "gof_af": anderson_frequency_domain(
                            ref["freq"],
                            ref["median_af"],
                            med_af_ref,
                            f_weight_center=float(ref["f_peak"]),
                            f_weight_width=1.5,
                        ),
                        "f_peak": f_peak,
                        "A_peak": a_peak,
                        "delta_f_peak": f_peak - ref["f_peak"],
                        "delta_ln_A_peak": float(
                            np.log(max(a_peak, 1e-12) / max(ref["A_peak"], 1e-12))
                        ),
                        "delta_mu_ln_sa_mean": log_residual_bias(ref["median_sa"], med_sa),
                        "delta_sigma_ln_sa_mean": float(np.mean(sig_sa - ref["sigma_ln_sa"])),
                        "pga_median": float(np.median(sub["pga"])),
                        "pga_bias_ln": log_residual_bias(
                            np.array([ref["pga_median"]]),
                            np.array([np.median(sub["pga"])]),
                        ),
                    }
                )

    summary = pd.DataFrame(records)
    summary.to_csv(out_dir / "method_comparison_summary.csv", index=False)
    return summary


def summarize_geomean_grf2d_vs_pretell(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Center-recorder TF: geomean ± 1σ_ln across RF seeds for grf_2d vs pretell."""
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for sobol_id in sorted(df["sobol_id"].unique()):
        for motion in sorted(df["motion_id"].unique()):
            method_rows: dict[str, dict] = {}
            freq_ref: np.ndarray | None = None
            for method in ("grf_2d", "pretell"):
                sub = df[
                    (df["method"] == method)
                    & (df["sobol_id"] == sobol_id)
                    & (df["motion_id"] == motion)
                ]
                if sub.empty:
                    continue
                if freq_ref is None:
                    freq_ref = np.asarray(sub.iloc[0]["freq"], dtype=float)
                freqs = [np.asarray(row["freq"], dtype=float) for _, row in sub.iterrows()]
                af_list = [np.asarray(row["af"], dtype=float) for _, row in sub.iterrows()]
                af_stack = _align_af_stack(af_list, freqs, freq_ref)
                geo = _geomean_af(af_stack)
                sig = _sigma_ln_af_stack(af_stack)
                f_peak, a_peak = _peak_af(freq_ref, geo)
                row = {
                    "sobol_id": sobol_id,
                    "motion_id": motion,
                    "method": method,
                    "n_seeds": len(sub),
                    "vs1": float(sub.iloc[0]["vs1"]),
                    "H": float(sub.iloc[0]["H"]),
                    "f_peak": f_peak,
                    "A_peak_geomean": a_peak,
                    "mean_sigma_ln_af": float(np.mean(sig)),
                }
                method_rows[method] = {"row": row, "geo": geo, "sig": sig, "freq": freq_ref}
                records.append(row)

            if "grf_2d" not in method_rows or "pretell" not in method_rows:
                continue

            geo_grf = method_rows["grf_2d"]["geo"]
            geo_del = method_rows["pretell"]["geo"]
            freq = method_rows["grf_2d"]["freq"]
            records.append(
                {
                    "sobol_id": sobol_id,
                    "motion_id": motion,
                    "method": "grf_2d_vs_pretell",
                    "n_seeds": min(
                        method_rows["grf_2d"]["row"]["n_seeds"],
                        method_rows["pretell"]["row"]["n_seeds"],
                    ),
                    "vs1": method_rows["grf_2d"]["row"]["vs1"],
                    "H": method_rows["grf_2d"]["row"]["H"],
                    "delta_mu_ln_af_geomean": log_residual_bias(geo_del, geo_grf),
                    "gof_af_geomean": anderson_frequency_domain(
                        freq,
                        geo_del,
                        geo_grf,
                        f_weight_center=float(method_rows["pretell"]["row"]["f_peak"]),
                        f_weight_width=1.5,
                    ),
                    "delta_ln_A_peak_geomean": float(
                        np.log(max(method_rows["grf_2d"]["row"]["A_peak_geomean"], 1e-12))
                        - np.log(max(method_rows["pretell"]["row"]["A_peak_geomean"], 1e-12))
                    ),
                }
            )

    summary = pd.DataFrame(records)
    summary.to_csv(out_dir / "method_comparison_geomean_summary.csv", index=False)
    return summary


def _rel_l2(pred: np.ndarray, true: np.ndarray) -> float:
    num = float(np.linalg.norm(pred - true))
    den = float(np.linalg.norm(true)) + 1e-12
    return num / den


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.size < 2 or b.size < 2:
        return float("nan")
    if np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def summarize_surrogate_vs_pretell_rigorous(
    df: pd.DataFrame, out_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired RF-seed metrics: GIFNO surrogate (grf_2d) vs OpenSees Pretell 1D.

    This is a *method* comparison on the same 2D GRF (not OpenSees-2D ground
    truth). Reports linear |AF| rel_L2, Pearson, peak errors, and σ_ln ratios.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed: list[dict] = []
    ensemble: list[dict] = []

    for sobol_id in sorted(df["sobol_id"].unique()):
        for motion in sorted(df["motion_id"].unique()):
            g = df[
                (df["method"] == "grf_2d")
                & (df["sobol_id"] == sobol_id)
                & (df["motion_id"] == motion)
            ]
            p = df[
                (df["method"] == "pretell")
                & (df["sobol_id"] == sobol_id)
                & (df["motion_id"] == motion)
            ]
            if g.empty or p.empty:
                continue

            # Pair by RF seed (same 2D realization).
            pretell_by_seed = {int(r["seed"]): r for _, r in p.iterrows()}
            paired_g: list[np.ndarray] = []
            paired_p: list[np.ndarray] = []
            freqs_g: list[np.ndarray] = []
            freqs_p: list[np.ndarray] = []
            seeds_used: list[int] = []

            for _, row in g.iterrows():
                seed = int(row["seed"])
                if seed not in pretell_by_seed:
                    continue
                pr = pretell_by_seed[seed]
                fg = np.asarray(row["freq"], dtype=float)
                fp = np.asarray(pr["freq"], dtype=float)
                # Common frequency grid = surrogate (GIFNO) axis.
                af_g = np.asarray(row["af"], dtype=float)
                af_p = _interp_af(fp, np.asarray(pr["af"], dtype=float), fg)
                if np.any(~np.isfinite(af_p)):
                    continue
                rel = _rel_l2(af_g, af_p)
                pear = _pearson(af_g, af_p)
                f_g, a_g = _peak_af(fg, af_g)
                f_p, a_p = _peak_af(fg, af_p)
                per_seed.append(
                    {
                        "sobol_id": sobol_id,
                        "motion_id": motion,
                        "seed": seed,
                        "vs1": float(row["vs1"]),
                        "H": float(row["H"]),
                        "rel_l2_surrogate_vs_pretell": rel,
                        "pearson": pear,
                        "f_peak_surrogate": f_g,
                        "f_peak_pretell": f_p,
                        "abs_df_peak_hz": abs(f_g - f_p),
                        "A_peak_surrogate": a_g,
                        "A_peak_pretell": a_p,
                        "delta_ln_A_peak": float(np.log(max(a_g, 1e-12) / max(a_p, 1e-12))),
                        "checkpoint_note": "grf_2d=GIFNO xt_lat128_d128; pretell=OpenSees 1D",
                    }
                )
                paired_g.append(af_g)
                paired_p.append(af_p)
                freqs_g.append(fg)
                freqs_p.append(fg)
                seeds_used.append(seed)

            if len(paired_g) < 2:
                continue

            freq_ref = freqs_g[0]
            stack_g = _align_af_stack(paired_g, freqs_g, freq_ref)
            stack_p = _align_af_stack(paired_p, freqs_p, freq_ref)
            geo_g = _geomean_af(stack_g)
            geo_p = _geomean_af(stack_p)
            sig_g = _sigma_ln_af_stack(stack_g)
            sig_p = _sigma_ln_af_stack(stack_p)
            mean_sig_g = float(np.mean(sig_g))
            mean_sig_p = float(np.mean(sig_p))
            # Per-seed metrics for this (sobol, motion) only (just appended).
            batch = [
                r
                for r in per_seed
                if r["sobol_id"] == sobol_id
                and r["motion_id"] == motion
                and r["seed"] in seeds_used
            ]
            ensemble.append(
                {
                    "sobol_id": sobol_id,
                    "motion_id": motion,
                    "n_paired_seeds": len(seeds_used),
                    "seeds": ",".join(str(s) for s in seeds_used),
                    "vs1": float(g.iloc[0]["vs1"]),
                    "H": float(g.iloc[0]["H"]),
                    "rel_l2_geomean": _rel_l2(geo_g, geo_p),
                    "pearson_geomean": _pearson(geo_g, geo_p),
                    "rel_l2_per_seed_mean": float(
                        np.mean([r["rel_l2_surrogate_vs_pretell"] for r in batch])
                    ),
                    "pearson_per_seed_mean": float(np.mean([r["pearson"] for r in batch])),
                    "sigma_ln_mean_surrogate": mean_sig_g,
                    "sigma_ln_mean_pretell": mean_sig_p,
                    "sigma_ln_ratio_surrogate_over_pretell": (
                        mean_sig_g / mean_sig_p if mean_sig_p > 1e-12 else float("nan")
                    ),
                    "delta_mu_ln_af_geomean": log_residual_bias(geo_p, geo_g),
                    "gof_af_geomean": anderson_frequency_domain(
                        freq_ref,
                        geo_p,
                        geo_g,
                        f_weight_center=float(_peak_af(freq_ref, geo_p)[0]),
                        f_weight_width=1.5,
                    ),
                    "abs_df_peak_hz_geomean": abs(
                        _peak_af(freq_ref, geo_g)[0] - _peak_af(freq_ref, geo_p)[0]
                    ),
                    "note": (
                        "Pretell is 1D OpenSees on central profiles of the same 2D GRF; "
                        "not OpenSees-2D ground truth. For OS-2D fidelity use "
                        "seed_robustness on the GIFNO corpus."
                    ),
                }
            )

    per_df = pd.DataFrame(per_seed)
    ens_df = pd.DataFrame(ensemble)
    per_path = out_dir / "surrogate_vs_pretell_per_seed.csv"
    ens_path = out_dir / "surrogate_vs_pretell_ensemble.csv"
    per_df.to_csv(per_path, index=False)
    ens_df.to_csv(ens_path, index=False)
    return per_df, ens_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-dir", type=Path, default=Path("results/h5"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/analysis"))
    args = parser.parse_args()

    df = collect_rows(args.h5_dir)
    print(f"Loaded {len(df)} HDF5 files from {args.h5_dir}")
    if df.empty:
        print("No data to analyze.")
        return

    summary = summarize_methods(df, args.out_dir)
    geo_summary = summarize_geomean_grf2d_vs_pretell(df, args.out_dir)
    per_seed, ensemble = summarize_surrogate_vs_pretell_rigorous(df, args.out_dir)
    print(summary.to_string(index=False))
    print(f"Wrote {args.out_dir / 'method_comparison_summary.csv'}")
    if not geo_summary.empty:
        print("\n--- geomean grf_2d vs pretell ---")
        print(geo_summary[geo_summary["method"] == "grf_2d_vs_pretell"].to_string(index=False))
        print(f"Wrote {args.out_dir / 'method_comparison_geomean_summary.csv'}")
    if not ensemble.empty:
        print("\n=== Rigorous surrogate (xt_lat128) vs Pretell ===")
        print(ensemble.to_string(index=False))
        print(f"Wrote {args.out_dir / 'surrogate_vs_pretell_ensemble.csv'}")
        print(f"Wrote {args.out_dir / 'surrogate_vs_pretell_per_seed.csv'}")
        if not per_seed.empty:
            print("\nper-seed rel_l2 / pearson:")
            print(
                per_seed[
                    ["seed", "rel_l2_surrogate_vs_pretell", "pearson", "abs_df_peak_hz"]
                ].to_string(index=False)
            )


if __name__ == "__main__":
    main()
