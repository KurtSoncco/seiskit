"""Dmult arms (Darendeli Dmin at 3 Hz and 1 Hz) vs the 2D OpenSees reference.

Reads ``method_comparison_summary.csv`` from ``analyze_response.py`` (run with
``--extra-h5-dir results/h5_dmult_1hz``) and plots, per Sobol column, the f0
peak-amplitude error, the peak-frequency error, and the mean ln-TF bias of the
Dmult arms next to the Toro / Passeri randomization arms, against Vs2/Vs1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from seiskit.plot_config import apply_style

ROOT = Path(__file__).resolve().parent
ARMS = {
    "hallal_vs": ("Toro Vs", "#009E73", "o"),
    "hallal_tts": ("Passeri tts", "#D55E00", "s"),
    "hallal_dmin_1hz": ("Dmult × Dmin(1 Hz)", "#56B4E9", "v"),
    "hallal_dmin": ("Dmult × Dmin(3 Hz)", "#CC79A7", "D"),
}
METRICS = (
    ("delta_ln_A_peak", r"$\ln(A_{f_0} / A_{f_0,\mathrm{2D}})$", "(a) f0 peak amplitude"),
    ("delta_f_peak", r"$f_0 - f_{0,\mathrm{2D}}$ (Hz)", "(b) f0 peak frequency"),
    ("delta_mu_ln_af_mean", r"mean $\ln(TF / TF_\mathrm{2D})$", "(c) mean ln-TF bias"),
)


def load(analysis_dir: Path) -> pd.DataFrame:
    summ = pd.read_csv(analysis_dir / "method_comparison_summary.csv")
    cases = pd.read_csv(ROOT / "rv_sobol_base_cases.csv")[["sobol_id", "Vs2"]]
    df = summ[(summ["ref_method"] == "opensees_2d") & summ["method"].isin(ARMS)]
    df = df.merge(cases, on="sobol_id", how="left")
    df["contrast"] = df["Vs2"] / df["vs1"]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "results" / "analysis")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "figures")
    args = parser.parse_args()

    apply_style()
    df = load(args.analysis_dir)
    if df.empty:
        print("No arms with an opensees_2d reference in the summary.")
        return
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.4), constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.4, 1]})
    arms = [m for m in ARMS if m in set(df["method"])]
    for j, (col, ylabel, title) in enumerate(METRICS):
        ax = axes[0, j]
        for m in arms:
            label, color, mk = ARMS[m]
            sub = df[df["method"] == m].sort_values("contrast")
            ax.scatter(sub["contrast"], sub[col], color=color, marker=mk, s=24, edgecolor="k", lw=0.3,
                       label=label, zorder=3)
        ax.axhline(0.0, color="0.2", lw=1)
        ax.set_xlabel(r"$V_{s2}/V_{s1}$")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=8)

        ax = axes[1, j]
        data = [df.loc[df["method"] == m, col].dropna().values for m in arms]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=True)
        for patch, m in zip(bp["boxes"], arms):
            patch.set_facecolor(ARMS[m][1])
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(arms) + 1), [ARMS[m][0] for m in arms], rotation=15, fontsize=8)
        ax.axhline(0.0, color="0.2", lw=1)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Dmult arms vs 2D OpenSees (64 Sobol columns, M1)", fontsize=11)
    out = args.out_dir / "dmult_arms_vs_opensees2d.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)

    stats = (
        df.groupby("method")[[c for c, _, _ in METRICS]]
        .agg(lambda v: f"{np.median(v):+.3f} [{np.percentile(v, 16):+.3f}, {np.percentile(v, 84):+.3f}]")
        .reindex(arms)
    )
    print("median [p16, p84] vs opensees_2d:")
    print(stats.to_string())
    abs_a = df.assign(abs_a=df["delta_ln_A_peak"].abs()).groupby("method")["abs_a"].median().reindex(arms)
    print("\nmedian |ln A_f0 error|:")
    print(abs_a.round(3).to_string())
    print(out)


if __name__ == "__main__":
    main()
