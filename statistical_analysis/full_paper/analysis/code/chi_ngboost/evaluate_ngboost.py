"""Summarize single-partition NGBoost holdout metrics against the matching R² ceiling.

Reads ``<partition>/train_ngboost/holdout_metrics.csv`` for the between-seed and
within-seed models and ``chi_ols/r2_ceiling/reliability_ceiling.csv`` (scopes
``between`` / ``within``). Writes under figure_dir("chi_ngboost", "evaluate_ngboost").
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CHI_OLS_CEILING, METRICS, PARTITION_LABELS, PARTITIONS, out_dir  # noqa: E402


def main() -> None:
    out = out_dir("evaluate_ngboost")
    ceiling = pd.read_csv(CHI_OLS_CEILING) if CHI_OLS_CEILING.is_file() else None

    rows = []
    for partition in PARTITIONS:
        train_dir = out_dir("train_ngboost", partition)
        hold = pd.read_csv(train_dir / "holdout_metrics.csv").set_index("metric")
        lag_path = train_dir / "residual_spatial_acf.csv"
        lag = pd.read_csv(lag_path).set_index("metric") if lag_path.is_file() else None
        for metric in METRICS:
            h = hold.loc[metric]
            row = {
                "partition": partition,
                "metric": metric,
                "ngboost_r2": float(h["r2_mean"]),
                "ngboost_rmse": float(h["rmse"]),
                "ngboost_nll": float(h["nll"]),
                "ngboost_pi90": float(h["pi90_coverage"]),
                "median_sigma_over_s": float(h["median_sigma_over_s"]),
            }
            if ceiling is not None and "scope" in ceiling.columns:
                c = ceiling[(ceiling["scope"] == partition) & (ceiling["metric"] == metric)]
                if len(c):
                    row["r2_ceiling"] = float(c.iloc[0]["reliability_ceiling"])
                    if row["r2_ceiling"] > 0:
                        row["ngboost_efficiency"] = row["ngboost_r2"] / row["r2_ceiling"]
            if lag is not None and metric in lag.index:
                row["mean_abs_lag1"] = float(lag.loc[metric, "mean_abs_lag1"])
            rows.append(row)

    tab = pd.DataFrame(rows)
    tab.to_csv(out / "ngboost_partitions.csv", index=False)

    lines = [
        "# NGBoost evaluation (between-seed and within-seed models)",
        "",
        "## Definitions",
        "",
        *(f"- `{p}`: {PARTITION_LABELS[p]}" for p in PARTITIONS),
        r"- Metrics from `<partition>/train_ngboost/holdout_metrics.csv`.",
        r"- Efficiency = NGBoost holdout \(R^2\) / \(R^2_{\mathrm{ceiling}}\) of the **same** scope "
        "(`chi_ols/r2_ceiling`, scope = partition).",
        r"- `median_sigma_over_s`: predicted σ vs the empirical SD of that partition "
        "(≈ 1 when σ tracks the intended variance).",
        "",
        "## Table",
        "",
        tab.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Notes",
        "",
        "- Rows from the two partitions are not pooled: between-seed σ and within-seed σ are "
        "different variances of \\(Y\\).",
        "- Residual lag-1 is reported for the within-seed model only (nodes along one profile).",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(tab.to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
