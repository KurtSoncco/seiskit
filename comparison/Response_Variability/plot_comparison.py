"""Publication plots for Response_Variability comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze_response import collect_rows, reference_curves

from seiskit.plot_config import apply_style

apply_style()


def plot_sa_comparison(
    df: pd.DataFrame, out_dir: Path, vs1: float = 230.0, motion_id: str = "M1"
) -> None:
    ref = reference_curves(df, vs1, motion_id)
    if not ref:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    periods = ref["periods"]
    ax.plot(periods, ref["median_sa"], "k-", lw=2, label="grf_2d ref median")
    for method in sorted(df["method"].unique()):
        if method == "grf_2d":
            continue
        sub = df[(df["method"] == method) & (df["vs1"] == vs1) & (df["motion_id"] == motion_id)]
        if sub.empty:
            continue
        med = np.median(np.vstack(sub["sa"].tolist()), axis=0)
        ax.plot(periods, med, "--", label=method)
    ax.set_xlabel("Period (s)")
    ax.set_ylabel("Sa (m/s²)")
    ax.set_title(f"Median Sa — Vs1={vs1}, {motion_id}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"sa_comparison_Vs1{vs1:.0f}_{motion_id}.png", dpi=150)
    plt.close(fig)


def plot_sigma_ln(df: pd.DataFrame, out_dir: Path) -> None:
    from seiskit.intensity_measures import sigma_ln

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for vs1 in sorted(df["vs1"].unique()):
        for motion in sorted(df["motion_id"].unique()):
            for method in sorted(df["method"].unique()):
                sub = df[
                    (df["method"] == method) & (df["vs1"] == vs1) & (df["motion_id"] == motion)
                ]
                if len(sub) < 2:
                    continue
                sa_stack = np.vstack(sub["sa"].tolist())
                sig = np.mean([sigma_ln(sa_stack[:, j]) for j in range(sa_stack.shape[1])])
                summary_rows.append(
                    {"vs1": vs1, "motion_id": motion, "method": method, "mean_sigma_ln_sa": sig}
                )
    if not summary_rows:
        return
    s = pd.DataFrame(summary_rows)
    fig, ax = plt.subplots(figsize=(8, 4))
    methods = s["method"].unique()
    x = np.arange(len(methods))
    for i, (_, g) in enumerate(s.groupby(["vs1", "motion_id"])):
        vals = [
            g[g["method"] == m]["mean_sigma_ln_sa"].values[0] if m in g["method"].values else 0
            for m in methods
        ]
        ax.bar(
            x + i * 0.15, vals, width=0.15, label=f"Vs1={g['vs1'].iloc[0]} {g['motion_id'].iloc[0]}"
        )
    ax.set_xticks(x + 0.15)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Mean σ_ln Sa")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "sigma_ln_sa_by_method.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-dir", type=Path, default=Path("results/h5"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()

    df = collect_rows(args.h5_dir)
    if df.empty:
        print("No data to plot.")
        return
    plot_sa_comparison(df, args.out_dir)
    plot_sigma_ln(df, args.out_dir)
    print(f"Figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
