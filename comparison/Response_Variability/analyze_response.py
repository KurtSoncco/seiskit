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


def _peak_af(freq: np.ndarray, af: np.ndarray) -> tuple[float, float]:
    i = int(np.argmax(af))
    return float(freq[i]), float(af[i])


def reference_curves(
    df: pd.DataFrame,
    sobol_id: int,
    motion_id: str,
) -> dict:
    ref = df[(df["method"] == "grf_2d") & (df["sobol_id"] == sobol_id) & (df["motion_id"] == motion_id)]
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
                f_peak, a_peak = _peak_af(ref["freq"], med_af)

                records.append(
                    {
                        "sobol_id": sobol_id,
                        "vs1": ref["vs1"],
                        "H": ref["H"],
                        "motion_id": motion,
                        "method": method,
                        "n_realizations": len(sub),
                        "delta_mu_ln_af_mean": log_residual_bias(ref["median_af"], med_af),
                        "delta_sigma_ln_af_mean": float(np.mean(sig_af - ref["sigma_ln_af"])),
                        "gof_af": anderson_frequency_domain(
                            ref["freq"],
                            ref["median_af"],
                            med_af,
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
