"""Compare NGBoost holdout metrics to chi_qbm mean GBM / PI coverage.

Reads train_ngboost outputs and chi_qbm compare_models CSVs (read-only).
Writes under figure_dir("chi_ngboost", "evaluate_ngboost").
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CHI_OLS_CEILING, CHI_QBM_COMPARE, METRICS, out_dir  # noqa: E402


def main() -> None:
    out = out_dir("evaluate_ngboost")
    train_dir = out_dir("train_ngboost")
    hold = pd.read_csv(train_dir / "holdout_metrics.csv")
    lag = pd.read_csv(train_dir / "residual_spatial_acf.csv")

    qbm = None
    cmp_path = CHI_QBM_COMPARE / "comparison_metrics.csv"
    if cmp_path.is_file():
        qbm = pd.read_csv(cmp_path)

    ceiling = None
    if CHI_OLS_CEILING.is_file():
        ceiling = pd.read_csv(CHI_OLS_CEILING)

    rows = []
    for metric in METRICS:
        h = hold.loc[hold["metric"] == metric].iloc[0]
        row = {
            "metric": metric,
            "ngboost_r2": float(h["r2_mean"]),
            "ngboost_rmse": float(h["rmse"]),
            "ngboost_nll": float(h["nll"]),
            "ngboost_pi90": float(h["pi90_coverage"]),
        }
        if qbm is not None and {"model", "r2"}.issubset(qbm.columns):
            sub = qbm[qbm["metric"] == metric]
            pick = sub[sub["model"] == "mean_gbm"]
            if len(pick):
                row["qbm_mean_r2"] = float(pick.iloc[0]["r2"])
                if "r2_ceiling" in pick.columns:
                    row["r2_ceiling"] = float(pick.iloc[0]["r2_ceiling"])
                if "efficiency" in pick.columns:
                    row["qbm_efficiency"] = float(pick.iloc[0]["efficiency"])
            pi_path = CHI_QBM_COMPARE / "pi_hetero.csv"
            if pi_path.is_file():
                pi = pd.read_csv(pi_path)
                if "metric" in pi.columns:
                    pis = pi[pi["metric"] == metric]
                    for col in ("coverage_90_qbm", "pi90_qbm", "coverage_qbm"):
                        if len(pis) and col in pis.columns:
                            row["qbm_pi90"] = float(pis.iloc[0][col])
                            break
        if "r2_ceiling" not in row and ceiling is not None and "metric" in ceiling.columns:
            csub = ceiling[ceiling["metric"] == metric]
            for col in ("r2_ceiling", "ceiling", "R2_ceiling"):
                if len(csub) and col in csub.columns:
                    row["r2_ceiling"] = float(csub.iloc[0][col])
                    break
        if "r2_ceiling" in row and row["r2_ceiling"] > 0:
            row["ngboost_efficiency"] = row["ngboost_r2"] / row["r2_ceiling"]
        lsub = lag[lag["metric"] == metric]
        if len(lsub):
            row["mean_abs_lag1"] = float(lsub.iloc[0]["mean_abs_lag1"])
        rows.append(row)

    tab = pd.DataFrame(rows)
    tab.to_csv(out / "ngboost_vs_qbm.csv", index=False)

    lines = [
        "# NGBoost evaluation vs QBM",
        "",
        "## Definitions",
        "",
        r"- NGBoost metrics from `train_ngboost/holdout_metrics.csv` (same seed holdout).",
        r"- QBM / mean GBM columns read from `chi_qbm/compare_models` when present.",
        r"- Efficiency = NGBoost mean \(R^2\) / \(R^2_{\mathrm{ceiling}}\) when ceiling CSV exists.",
        "",
        "## Comparison table",
        "",
        tab.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Conclusions",
        "",
        r"- Compare NGBoost and QBM on the **stated target**: mean \(R^2\) vs distributional scores (NLL, PI coverage, pinball).",
        "- High residual lag-1 for NGBoost (as for QBM) means neither model whitened spatial dependence; both remain conditional emulators.",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(tab.to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
