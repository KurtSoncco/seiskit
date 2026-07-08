"""
Aggregate Response_Variability HDF5 outputs and compare methods vs 2D GRF reference.

Primary comparison is transfer-function based (Hallal / de la Torre style): for linear
viscoelastic small-strain analyses, AF(f) is an intrinsic site property and does not
depend on the input ground motion (beyond estimation quality of the FAS ratio).

Usage (from comparison/Response_Variability):
  python analyze_response.py
  python analyze_response.py --h5-dir results/h5 --out-dir results/analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from seiskit.gof import anderson_frequency_domain, log_residual_bias
from seiskit.intensity_measures import sigma_ln


def _load_h5(path: Path) -> dict:
    import h5py

    with h5py.File(path, "r") as f:
        method = f.attrs.get("method", "")
        motion_id = f.attrs.get("motion_id", "")
        pga = float(f["ims"].attrs.get("PGA_surface", 0.0))
        periods = f["ims"]["Sa_periods"][:]
        sa = f["ims"]["Sa_surface"][:]
        freq = f["transfer_function"]["freq"][:]
        af = f["transfer_function"]["AF"][:]
        vs1 = float(f["params"].attrs["Vs1"])
        seed = int(f["params"].attrs["seed"])
        out = {
            "method": method,
            "motion_id": motion_id,
            "vs1": vs1,
            "seed": seed,
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


def _peak_af(freq: np.ndarray, af: np.ndarray) -> tuple[float, float]:
    i = int(np.argmax(af))
    return float(freq[i]), float(af[i])


def reference_curves(df: pd.DataFrame, vs1: float, motion_id: str) -> dict:
    ref = df[(df["method"] == "grf_2d") & (df["vs1"] == vs1) & (df["motion_id"] == motion_id)]
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

    for vs1 in sorted(df["vs1"].unique()):
        for motion in sorted(df["motion_id"].unique()):
            ref = reference_curves(df, vs1, motion)
            if not ref:
                continue
            for method in sorted(df["method"].unique()):
                sub = df[
                    (df["method"] == method) & (df["vs1"] == vs1) & (df["motion_id"] == motion)
                ]
                if sub.empty:
                    continue
                sa_stack = np.vstack(sub["sa"].tolist())
                med_sa = np.median(sa_stack, axis=0)
                sig_sa = np.array([sigma_ln(sa_stack[:, j]) for j in range(sa_stack.shape[1])])
                af_stack = np.vstack(sub["af"].tolist())
                med_af = np.median(af_stack, axis=0)
                sig_af = np.array([sigma_ln(af_stack[:, j]) for j in range(af_stack.shape[1])])
                f_peak, a_peak = _peak_af(ref["freq"], med_af)

                delta_mu_sa = log_residual_bias(ref["median_sa"], med_sa)
                delta_sigma_sa = float(np.mean(sig_sa - ref["sigma_ln_sa"]))
                delta_mu_af = log_residual_bias(ref["median_af"], med_af)
                delta_sigma_af = float(np.mean(sig_af - ref["sigma_ln_af"]))
                gof_af = anderson_frequency_domain(
                    ref["freq"],
                    ref["median_af"],
                    med_af,
                    f_weight_center=float(ref["f_peak"]),
                    f_weight_width=1.5,
                )
                records.append(
                    {
                        "vs1": vs1,
                        "motion_id": motion,
                        "method": method,
                        "n_realizations": len(sub),
                        # Primary TF metrics (motion-independent under linear viscoelasticity)
                        "delta_mu_ln_af_mean": delta_mu_af,
                        "delta_sigma_ln_af_mean": delta_sigma_af,
                        "gof_af": gof_af,
                        "f_peak": f_peak,
                        "A_peak": a_peak,
                        "delta_f_peak": f_peak - ref["f_peak"],
                        "delta_ln_A_peak": float(
                            np.log(max(a_peak, 1e-12) / max(ref["A_peak"], 1e-12))
                        ),
                        # Secondary IM checks (motion-dependent envelopes of AF × FAS)
                        "delta_mu_ln_sa_mean": delta_mu_sa,
                        "delta_sigma_ln_sa_mean": delta_sigma_sa,
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
    print(summary.to_string(index=False))
    print(f"Wrote {args.out_dir / 'method_comparison_summary.csv'}")


if __name__ == "__main__":
    main()
