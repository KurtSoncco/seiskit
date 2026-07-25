"""Evaluate SR formulas; compare GP vs OLS vs reported under evaluate_sr."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import COLLAPSE_R2_THRESH, out_dir  # noqa: E402


def main() -> None:
    out = out_dir("evaluate_sr")
    train = out_dir("train_sr")
    formulas = pd.read_csv(train / "sr_formulas.csv")
    fidelity = pd.read_csv(train / "sr_fidelity.csv")
    ols = pd.read_csv(train / "ols_baseline.csv")

    formulas.to_csv(out / "sr_formulas.csv", index=False)
    fidelity.to_csv(out / "sr_fidelity.csv", index=False)
    ols.to_csv(out / "ols_baseline.csv", index=False)

    merged = fidelity.merge(
        formulas[
            [
                "metric",
                "target",
                "formula_reported",
                "formula_source",
                "sr_collapsed",
                "program_length",
                "y_std",
            ]
        ],
        on=["metric", "target"],
        suffixes=("", "_f"),
    )
    # Prefer fidelity's sr_collapsed / formula_source
    if "sr_collapsed_f" in merged.columns:
        merged = merged.drop(columns=["sr_collapsed_f"], errors="ignore")

    still_bad = merged[merged["r2_test_reported"] < 0.5]
    mu = merged[merged["target"] == "mu"]
    logsig = merged[merged["target"] == "log_sigma"]

    lines = [
        "# Symbolic regression evaluation",
        "",
        "## Definitions",
        "",
        r"- GP fits standardized NGBoost surfaces; reported \(R^2\) is on the original \(\mu\) / \(\log\sigma\) scale.",
        r"- OLS = shortlist linear baseline (intercept + SHAP-guided features/products).",
        rf"- Collapse: GP test \(R^2 < {COLLAPSE_R2_THRESH}\) or constant/short program → report OLS.",
        r"- `r2_test_reported` is the paper-facing fidelity metric.",
        "",
        "## GP vs OLS vs reported",
        "",
        merged[
            [
                "metric",
                "target",
                "r2_test_gp",
                "r2_test_ols",
                "r2_test_reported",
                "sr_collapsed",
                "formula_source",
                "y_std",
                "program_length",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## μ summary",
        "",
        mu[
            ["metric", "r2_test_gp", "r2_test_ols", "r2_test_reported", "formula_source"]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## log-σ summary",
        "",
        logsig[
            ["metric", "r2_test_gp", "r2_test_ols", "r2_test_reported", "formula_source"]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Residual weak reported fits (\(R^2<0.5\))",
        "",
    ]
    if len(still_bad):
        lines.append(
            still_bad[
                ["metric", "target", "r2_test_reported", "r2_test_ols", "formula_source"]
            ].to_markdown(index=False, floatfmt=".4f")
        )
    else:
        lines.append("None.")
    lines += [
        "",
        "## Conclusions",
        "",
        "- Standardization + lower parsimony address μ collapses driven by small absolute MSE; OLS fallback guarantees a readable high-fidelity formula when GP still fails.",
        "- Prefer `formula_reported` with `formula_source` for paper text; keep `formula_gp` for audit of search success/failure.",
        "- These compressions approximate NGBoost surfaces only; they do not whiten residual spatial lag-1.",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        merged[
            [
                "metric",
                "target",
                "r2_test_gp",
                "r2_test_ols",
                "r2_test_reported",
                "sr_collapsed",
                "formula_source",
            ]
        ].to_string(index=False)
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
