"""Spatial ACF role in OLS inference for χ ratios.

Loads CosWM / Exp / Gaussian ACF fits from ``chi_spatial/spatial_acf``,
computes \(n_{\mathrm{eff}}\) per cell × metric, and documents how spatial
structure enters Stage-1 inference (SE correction / clustering — not mean
covariates).

Writes CSV + summary.md under ``figure_dir("chi_ols", "spatial_in_ols")``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ACF_FIT_PATH,
    DX_M,
    FACTORS,
    METRICS,
    N_CELLS,
    N_NODES,
    N_SEEDS,
    fmt,
    out_dir,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chi_spatial"))
from spatial_acf import rho_coswm, rho_exp, rho_gauss  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chi_variables"))
# n_eff helpers live next to adequacy; reimplement minimally to avoid heavy import


def _rho_from_fit_row(row: pd.Series, h: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Evaluate preferred ACF (CosWM → best_model → gauss → exp)."""
    if bool(row.get("fit_ok_coswm", False)):
        return (
            rho_coswm(
                h,
                float(row["c0_coswm"]),
                float(row["nu_coswm"]),
                float(row["scale_s_m_coswm"]),
                float(row["period_b_m_coswm"]),
            ),
            "coswm",
        )
    best = str(row.get("best_model", "none"))
    if best == "gauss" and bool(row.get("fit_ok_gauss", False)):
        return rho_gauss(h, float(row["c0_gauss"]), float(row["a_m_gauss"])), "gauss"
    if best == "exp" and bool(row.get("fit_ok_exp", False)):
        return rho_exp(h, float(row["c0_exp"]), float(row["a_m_exp"])), "exp"
    if bool(row.get("fit_ok_gauss", False)):
        return rho_gauss(h, float(row["c0_gauss"]), float(row["a_m_gauss"])), "gauss"
    if bool(row.get("fit_ok_exp", False)):
        return rho_exp(h, float(row["c0_exp"]), float(row["a_m_exp"])), "exp"
    return None, "none"


def n_eff_from_rho(rho_k: np.ndarray, n: int = N_NODES) -> float:
    rho_k = np.asarray(rho_k, dtype=float)
    if rho_k.size != n - 1:
        raise ValueError(f"expected {n - 1} lag correlations, got {rho_k.size}")
    k = np.arange(1, n, dtype=float)
    weights = 1.0 - k / float(n)
    rho_use = np.clip(np.where(np.isfinite(rho_k), rho_k, 0.0), -1.0, 1.0)
    denom = 1.0 + 2.0 * float(np.sum(weights * rho_use))
    denom = max(denom, 1.0 / float(n))
    return float(n) / denom


def neff_row(row: pd.Series) -> dict:
    h = DX_M * np.arange(1, N_NODES, dtype=float)
    rho, model = _rho_from_fit_row(row, h)
    if rho is None:
        ne = np.nan
        rho1 = np.nan
    else:
        ne = n_eff_from_rho(rho, n=N_NODES)
        rho1 = float(rho[0]) if rho.size else np.nan
    return dict(
        Vs1=float(row["Vs1"]),
        Height=float(row["Height"]),
        CoV=float(row["CoV"]),
        rH=float(row["rH"]),
        aHV=float(row["aHV"]),
        cell=int(row["cell"]) if "cell" in row and pd.notna(row["cell"]) else -1,
        metric=str(row["metric"]),
        acf_model_used=model,
        best_model=str(row.get("best_model", "none")),
        rho_lag2_m=float(row["rho_lag2_m"]) if pd.notna(row.get("rho_lag2_m")) else np.nan,
        rho_model_lag2_m=rho1,
        n_eff=ne,
        n_nodes=N_NODES,
        n_eff_frac=ne / N_NODES if np.isfinite(ne) else np.nan,
        se_infl_spatial=np.sqrt(N_NODES / ne) if np.isfinite(ne) and ne > 0 else np.nan,
    )


def build_summary_md(neff: pd.DataFrame, summary: pd.DataFrame) -> str:
    lines = [
        "# Spatial ACF in OLS inference",
        "",
        "How lateral spatial autocorrelation of \(\\ln\\chi\) along the "
        f"{N_NODES}-node array enters Stage-1 OLS **uncertainty**, not the mean "
        "design matrix.",
        "",
        f"- ACF fits loaded from `{ACF_FIT_PATH}` "
        "(Exp / Gaussian / CosWM; CosWM preferred when available)",
        f"- Node spacing \(\\Delta x = {DX_M:g}\) m; "
        f"\(N_x = {N_NODES}\), \(N_s = {N_SEEDS}\), \(K = {N_CELLS}\) cells",
        "",
        "## Output files",
        "",
        "| File | Contents |",
        "|------|----------|",
        "| `neff_by_cell.csv` | Per cell × metric \(n_{\\mathrm{eff}}\), "
        "model used, spatial SE inflation \(\\sqrt{N_x / n_{\\mathrm{eff}}}\) |",
        "| `spatial_inference_summary.csv` | Metric-level medians / IQR of "
        "\(n_{\\mathrm{eff}}\) and lag-1 ρ |",
        "| `summary.md` | this file |",
        "",
        "## Notation",
        "",
        "Cosine Whittle–Matérn (CosWM) correlation:",
        "",
        r"$$",
        r"\rho(h) = (1-c_0)\,\rho_{\mathrm{WM}}(h;\nu,s)\,\cos(h/b).",
        r"$$",
        "",
        "Effective sample size for the sample mean of \(N_x\) nodes:",
        "",
        r"$$",
        r"n_{\mathrm{eff}}"
        r" = \frac{N_x}"
        r"{1 + 2\sum_{k=1}^{N_x-1}"
        r"\bigl(1 - k/N_x\bigr)\rho(k\,\Delta x)}.",
        r"$$",
        "",
        "Naive SE of a node-average that treats nodes as iid understates "
        "uncertainty by about \(\\sqrt{N_x / n_{\\mathrm{eff}}}\) when "
        "\(\\rho > 0\).",
        "",
        "### Roles in OLS (do use / do not use)",
        "",
        "| Role | Use spatial ACF? |",
        "|------|------------------|",
        "| Mean design matrix \(\\mathbf{x}_k\) | **No** — do not add lag "
        "features as predictors of \(E[Y\\mid\\mathrm{design}]\) |",
        "| SE of cell / seed geomean across nodes | **Yes** — \(n_{\\mathrm{eff}}\) |",
        "| Stage-2 residual whitening before variance model | **Yes** — CosWM "
        "\(R(\\hat\\phi_k)\) (Box `mixed_model`) |",
        "| Cluster-robust SE (seed / cell) | **Yes** — robust alternative that "
        "does not require estimating \(\\phi\) inside the mean |",
        "",
        "## Metric summary",
        "",
        "| Metric | Median \(n_{\\mathrm{eff}}\) | IQR | Median "
        "\(\\sqrt{N_x/n_{\\mathrm{eff}}}\) | Median ρ̂(2 m) | "
        "Frac CosWM |",
        "|--------|---------------------------:|----:|------------------------------:|"
        "----------------:|-----------:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['metric']} | {fmt(r['n_eff_median'], 1)} | "
            f"[{fmt(r['n_eff_q25'], 1)}, {fmt(r['n_eff_q75'], 1)}] | "
            f"{fmt(r['se_infl_spatial_median'])} | "
            f"{fmt(r['rho_lag2_median'])} | "
            f"{fmt(r['frac_coswm'])} |"
        )

    lines.extend(["", "## Conclusions", ""])
    for _, r in summary.iterrows():
        lines.append(
            f"- **{r['metric']}**: median \(n_{{\\mathrm{{eff}}}} = "
            f"{fmt(r['n_eff_median'], 1)}\) "
            f"(of {N_NODES} nodes) ⇒ spatial SE inflation "
            f"~{fmt(r['se_infl_spatial_median'])}× for node-averages; "
            f"CosWM used in {fmt(100 * r['frac_coswm'], 0)}% of cells."
        )
    lines.extend(
        [
            "",
            "Spatial ACF therefore belongs in **uncertainty** "
            "(\(n_{\\mathrm{eff}}\), CosWM whitening, cluster SE), not as "
            "extra Stage-1 mean covariates. Pair these CSVs with "
            "`stage1_mean_ols` cluster SEs and `naive_vs_hetero` variance "
            "diagnostics.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    if not ACF_FIT_PATH.is_file():
        raise FileNotFoundError(
            f"Missing ACF fits at {ACF_FIT_PATH}. "
            "Run chi_spatial/spatial_acf.py first."
        )
    print(f"Loading {ACF_FIT_PATH} …")
    fits = pd.read_csv(ACF_FIT_PATH)
    print(f"  rows={len(fits):,}")

    rows = [neff_row(r) for _, r in fits.iterrows()]
    neff = pd.DataFrame(rows)

    sum_rows = []
    for metric in METRICS:
        sub = neff[neff["metric"] == metric]
        sum_rows.append(
            dict(
                metric=metric,
                n_cells=int(sub["cell"].nunique()) if "cell" in sub else len(sub),
                n_eff_median=float(sub["n_eff"].median()),
                n_eff_q25=float(sub["n_eff"].quantile(0.25)),
                n_eff_q75=float(sub["n_eff"].quantile(0.75)),
                n_eff_mean=float(sub["n_eff"].mean()),
                se_infl_spatial_median=float(sub["se_infl_spatial"].median()),
                rho_lag2_median=float(sub["rho_lag2_m"].median()),
                frac_coswm=float((sub["acf_model_used"] == "coswm").mean()),
                frac_fit_ok=float(sub["n_eff"].notna().mean()),
            )
        )
    summary = pd.DataFrame(sum_rows)

    dest = out_dir("spatial_in_ols")
    neff.to_csv(dest / "neff_by_cell.csv", index=False)
    summary.to_csv(dest / "spatial_inference_summary.csv", index=False)
    (dest / "summary.md").write_text(build_summary_md(neff, summary), encoding="utf-8")
    print(f"Wrote {dest}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
