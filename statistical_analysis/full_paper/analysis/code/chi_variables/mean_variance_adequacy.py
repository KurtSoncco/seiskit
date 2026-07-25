"""Per-design-cell mean / variance sample-size adequacy for χ ratios.

For each of 243 design cells × 5 metrics on ln(χ):

- Seed-blocked SE of the cell mean (geomean) vs number of seeds S
- Seed-blocked SE of the cell variance under seed heteroscedasticity
- CosWM spatial n_eff and naive vs corrected analytical SE cross-checks
- Seeds required for relative-precision targets; adequacy at S=100

Writes CSV + markdown + one 2-panel figure under
``figure_dir("chi_variables", "mean_variance_adequacy")``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import (
    BOX_ROOT,
    FACTORS,
    METRICS,
    add_panel_label,
    apply_full_paper_style,
    figsize,
    figure_dir,
    metric_color,
    metric_label,
    save_figure,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "chi_spatial"))
from spatial_acf import DX_M, N_NODES, rho_coswm, rho_exp, rho_gauss

DATA_PATH = BOX_ROOT / "peak_analysis" / "join_master.h5"
ACF_FIT_PATH = (
    BOX_ROOT / "full_paper" / "figures" / "chi_spatial" / "spatial_acf" / "acf_fit_params.csv"
)
N_SEEDS = 100
N_CELLS = 243
S_GRID = np.array([5, 10, 20, 30, 50, 75, 100], dtype=int)
EPS_MEAN = (0.05, 0.10)  # relative SE targets for geomean (≡ SE(μ))
EPS_VAR = (0.05, 0.10)  # relative SE targets for σ²
ADEQ_MEAN_RSE = 0.05  # SE(μ) ≤ this at S=100 (RSE of G)
ADEQ_VAR_RSE = 0.10  # SE(σ²)/σ² ≤ this at S=100
MU_ABS_FLOOR = 0.05  # |μ| below this → classical RSE(μ) ill-defined


def load_ratios(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load joined ratio table; rename channel → node."""
    cols = [
        "Vs1",
        "Height",
        "CoV",
        "rH",
        "aHV",
        "channel",
        "seed",
        *METRICS,
    ]
    with h5py.File(path, "r") as f:
        g = f["master"]
        df = pd.DataFrame({c: g[c][:] for c in cols})
    return df.rename(columns={"channel": "node"})


def _rho_from_fit_row(row: pd.Series, h: np.ndarray) -> np.ndarray | None:
    """Evaluate best available ACF model on lag distances *h* (metres)."""
    if bool(row.get("fit_ok_coswm", False)):
        return rho_coswm(
            h,
            float(row["c0_coswm"]),
            float(row["nu_coswm"]),
            float(row["scale_s_m_coswm"]),
            float(row["period_b_m_coswm"]),
        )
    if bool(row.get("fit_ok_gauss", False)):
        return rho_gauss(h, float(row["c0_gauss"]), float(row["a_m_gauss"]))
    if bool(row.get("fit_ok_exp", False)):
        return rho_exp(h, float(row["c0_exp"]), float(row["a_m_exp"]))
    return None


def n_eff_from_rho(rho_k: np.ndarray, n: int = N_NODES) -> float:
    """Effective sample size for the sample mean under lag correlations.

    Parameters
    ----------
    rho_k
        Correlations at lags k = 1, …, n−1 (same units as node spacing).
    """
    rho_k = np.asarray(rho_k, dtype=float)
    if rho_k.size != n - 1:
        raise ValueError(f"expected {n - 1} lag correlations, got {rho_k.size}")
    k = np.arange(1, n, dtype=float)
    weights = 1.0 - k / float(n)
    rho_use = np.where(np.isfinite(rho_k), rho_k, 0.0)
    # Clamp ρ to [-1, 1] for numerical safety of CosWM oscillations
    rho_use = np.clip(rho_use, -1.0, 1.0)
    denom = 1.0 + 2.0 * float(np.sum(weights * rho_use))
    denom = max(denom, 1.0 / float(n))  # keep n_eff ≤ n
    return float(n) / denom


def n_eff_from_fit_row(row: pd.Series, n: int = N_NODES, dx: float = DX_M) -> float:
    """CosWM (or fallback) n_eff for an ACF fit row."""
    h = dx * np.arange(1, n, dtype=float)
    rho = _rho_from_fit_row(row, h)
    if rho is None:
        return np.nan
    return n_eff_from_rho(rho, n=n)


def _s_req(scale: float, eps: float) -> float:
    """Seeds required so scale/√S ≤ eps → S ≥ (scale/eps)²."""
    if not np.isfinite(scale) or not np.isfinite(eps) or eps <= 0:
        return np.nan
    if scale <= 0:
        return 1.0
    return float(np.ceil((scale / eps) ** 2))


def assess_cell_metric(
    Y: np.ndarray,
    cell_keys: dict,
    metric: str,
    n_eff: float,
) -> dict:
    """Compute adequacy statistics for one cell × metric on ln(χ) array Y."""
    # Y: (n_nodes, n_seeds)
    finite = np.isfinite(Y)
    n_finite = int(finite.sum())
    seed_ok = finite.any(axis=0)
    n_seeds_used = int(seed_ok.sum())

    empty = {
        **cell_keys,
        "metric": metric,
        "n_nodes": int(Y.shape[0]),
        "n_seeds": int(Y.shape[1]),
        "n_seeds_used": n_seeds_used,
        "n_finite": n_finite,
        "mu_hat": np.nan,
        "G_hat": np.nan,
        "s_mu": np.nan,
        "sigma2_hat": np.nan,
        "s_sigma2": np.nan,
        "n_eff": n_eff,
        "se_mu_S100": np.nan,
        "se_G_S100": np.nan,
        "rse_mu_S100": np.nan,
        "rse_G_S100": np.nan,
        "se_sigma2_S100": np.nan,
        "rse_sigma2_S100": np.nan,
        "se_sigma2_naive_S100": np.nan,
        "se_sigma2_neff_S100": np.nan,
        "ratio_naive_over_block": np.nan,
        "ratio_neff_over_block": np.nan,
        "S_req_mean_rse05": np.nan,
        "S_req_mean_rse10": np.nan,
        "S_req_mu_classical_rse05": np.nan,
        "S_req_mu_classical_rse10": np.nan,
        "S_req_var_rse05": np.nan,
        "S_req_var_rse10": np.nan,
        "adequate_mean_S100": False,
        "adequate_var_S100": False,
        "adequate_both_S100": False,
        "mu_near_zero": False,
    }
    if n_seeds_used < 2:
        return empty

    Yj = Y[:, seed_ok]
    # Seed spatial means (nanmean over nodes)
    mu_j = np.nanmean(Yj, axis=0)
    mu_hat = float(np.nanmean(mu_j))
    s_mu = float(np.nanstd(mu_j, ddof=1))
    G_hat = float(np.exp(mu_hat)) if np.isfinite(mu_hat) else np.nan

    # Per-seed sample variances (ddof=1); skip seeds with < 2 finite nodes
    s2_j = []
    for j in range(Yj.shape[1]):
        col = Yj[:, j]
        col = col[np.isfinite(col)]
        if col.size >= 2:
            s2_j.append(float(np.var(col, ddof=1)))
    s2_j = np.asarray(s2_j, dtype=float)
    if s2_j.size < 2:
        return empty

    sigma2_hat = float(np.mean(s2_j))
    s_sigma2 = float(np.std(s2_j, ddof=1))

    S_ref = float(N_SEEDS)
    se_mu = s_mu / np.sqrt(S_ref)
    se_G = G_hat * se_mu if np.isfinite(G_hat) else np.nan
    rse_G = se_mu  # delta-method RSE(G) = SE(μ)
    mu_near_zero = bool(np.isfinite(mu_hat) and abs(mu_hat) < MU_ABS_FLOOR)
    if mu_near_zero or not np.isfinite(mu_hat) or abs(mu_hat) < 1e-15:
        rse_mu = np.nan
    else:
        rse_mu = se_mu / abs(mu_hat)

    se_sigma2 = s_sigma2 / np.sqrt(S_ref)
    rse_sigma2 = se_sigma2 / sigma2_hat if np.isfinite(sigma2_hat) and sigma2_hat > 0 else np.nan

    # Analytical cross-checks for Var estimator SE (Gaussian, iid / n_eff)
    N = float(Y.shape[0])
    se_naive = (
        sigma2_hat * np.sqrt(2.0 / (N * S_ref - 1.0))
        if np.isfinite(sigma2_hat) and sigma2_hat > 0
        else np.nan
    )
    if np.isfinite(n_eff) and n_eff > 1 and np.isfinite(sigma2_hat) and sigma2_hat > 0:
        se_neff = sigma2_hat * np.sqrt(2.0 / (S_ref * n_eff - 1.0))
    else:
        se_neff = np.nan

    ratio_naive = se_naive / se_sigma2 if se_sigma2 > 0 and np.isfinite(se_naive) else np.nan
    ratio_neff = se_neff / se_sigma2 if se_sigma2 > 0 and np.isfinite(se_neff) else np.nan

    # S required: geomean relative (SE(μ) ≤ ε) and classical RSE(μ)
    S_req_mean_05 = _s_req(s_mu, EPS_MEAN[0])
    S_req_mean_10 = _s_req(s_mu, EPS_MEAN[1])
    if mu_near_zero or not np.isfinite(mu_hat) or abs(mu_hat) < 1e-15:
        S_req_mu_cl_05 = np.nan
        S_req_mu_cl_10 = np.nan
    else:
        S_req_mu_cl_05 = _s_req(s_mu / abs(mu_hat), EPS_MEAN[0])
        S_req_mu_cl_10 = _s_req(s_mu / abs(mu_hat), EPS_MEAN[1])

    S_req_var_05 = (
        _s_req(s_sigma2 / sigma2_hat, EPS_VAR[0])
        if sigma2_hat > 0 and np.isfinite(sigma2_hat)
        else np.nan
    )
    S_req_var_10 = (
        _s_req(s_sigma2 / sigma2_hat, EPS_VAR[1])
        if sigma2_hat > 0 and np.isfinite(sigma2_hat)
        else np.nan
    )

    adequate_mean = bool(np.isfinite(rse_G) and rse_G <= ADEQ_MEAN_RSE)
    adequate_var = bool(np.isfinite(rse_sigma2) and rse_sigma2 <= ADEQ_VAR_RSE)

    return {
        **cell_keys,
        "metric": metric,
        "n_nodes": int(Y.shape[0]),
        "n_seeds": int(Y.shape[1]),
        "n_seeds_used": n_seeds_used,
        "n_finite": n_finite,
        "mu_hat": mu_hat,
        "G_hat": G_hat,
        "s_mu": s_mu,
        "sigma2_hat": sigma2_hat,
        "s_sigma2": s_sigma2,
        "n_eff": float(n_eff) if np.isfinite(n_eff) else np.nan,
        "se_mu_S100": se_mu,
        "se_G_S100": se_G,
        "rse_mu_S100": rse_mu,
        "rse_G_S100": rse_G,
        "se_sigma2_S100": se_sigma2,
        "rse_sigma2_S100": rse_sigma2,
        "se_sigma2_naive_S100": se_naive,
        "se_sigma2_neff_S100": se_neff,
        "ratio_naive_over_block": ratio_naive,
        "ratio_neff_over_block": ratio_neff,
        "S_req_mean_rse05": S_req_mean_05,
        "S_req_mean_rse10": S_req_mean_10,
        "S_req_mu_classical_rse05": S_req_mu_cl_05,
        "S_req_mu_classical_rse10": S_req_mu_cl_10,
        "S_req_var_rse05": S_req_var_05,
        "S_req_var_rse10": S_req_var_10,
        "adequate_mean_S100": adequate_mean,
        "adequate_var_S100": adequate_var,
        "adequate_both_S100": adequate_mean and adequate_var,
        "mu_near_zero": mu_near_zero,
    }


def build_se_curves(cell_df: pd.DataFrame) -> pd.DataFrame:
    """Long-form SE vs S: per metric, percentiles across cells."""
    rows: list[dict] = []
    for metric in METRICS:
        m = cell_df[cell_df["metric"] == metric]
        s_mu = m["s_mu"].to_numpy(dtype=float)
        G = m["G_hat"].to_numpy(dtype=float)
        s_s2 = m["s_sigma2"].to_numpy(dtype=float)
        sig2 = m["sigma2_hat"].to_numpy(dtype=float)
        for S in S_GRID:
            se_mu = s_mu / np.sqrt(float(S))
            se_G = G * se_mu
            se_s2 = s_s2 / np.sqrt(float(S))
            rse_G = se_mu  # = SE(μ)
            with np.errstate(divide="ignore", invalid="ignore"):
                rse_s2 = se_s2 / sig2

            def _pct(a: np.ndarray, q: float) -> float:
                a = a[np.isfinite(a)]
                return float(np.percentile(a, q)) if a.size else np.nan

            rows.append(
                dict(
                    metric=metric,
                    n_seeds=int(S),
                    se_mu_p10=_pct(se_mu, 10),
                    se_mu_median=_pct(se_mu, 50),
                    se_mu_p90=_pct(se_mu, 90),
                    se_G_p10=_pct(se_G, 10),
                    se_G_median=_pct(se_G, 50),
                    se_G_p90=_pct(se_G, 90),
                    rse_G_p10=_pct(rse_G, 10),
                    rse_G_median=_pct(rse_G, 50),
                    rse_G_p90=_pct(rse_G, 90),
                    se_sigma2_p10=_pct(se_s2, 10),
                    se_sigma2_median=_pct(se_s2, 50),
                    se_sigma2_p90=_pct(se_s2, 90),
                    rse_sigma2_p10=_pct(rse_s2, 10),
                    rse_sigma2_median=_pct(rse_s2, 50),
                    rse_sigma2_p90=_pct(rse_s2, 90),
                    n_cells=int(np.isfinite(s_mu).sum()),
                )
            )
    return pd.DataFrame.from_records(rows)


def build_summary_md(cell_df: pd.DataFrame, curves: pd.DataFrame) -> str:
    """Derivation + numeric conclusions markdown."""
    lines: list[str] = [
        "# Mean / variance sample-size adequacy",
        "",
        "Per design cell × metric on the 1D-normalized ratios χ from "
        "`join_master.h5`, with working scale \\(Y = \\ln\\chi\\) (lognormal "
        "model for χ).",
        "",
        "Canonical symbols: "
        "`statistical_analysis/full_paper/analysis/NOTATION.md`.",
        "",
        "## Setup",
        "",
        "Each design cell fixes `(Vs1, Height, CoV, rH, aHV)`. Within a cell "
        "the array is \\(Y_{ij}\\) for nodes \\(i = 1,\\ldots,N_x\\) "
        f"(\\(N_x = {N_NODES}\\), \\(\\Delta x = {DX_M:g}\\,\\mathrm{{m}}\\)) and "
        f"seeds \\(j = 1,\\ldots,N_s\\) (\\(N_s = {N_SEEDS}\\) in the experiment).",
        "",
        "Prior diagnostics imply:",
        "",
        "- Seeds at a fixed node are approximately exchangeable.",
        "- Nodes within a seed are strongly spatially correlated "
        "(CosWM ACF; \\(\\hat\\rho(2\\,\\mathrm{m}) \\approx 0.96\\)–\\(0.99\\)).",
        "- Within-cell seed variances are strongly heteroscedastic.",
        "",
        r"Therefore \(N_x N_s\) must **not** be treated as iid. Precision of "
        "cell-level \\(\\mu = \\mathbb{E}[Y]\\) and \\(\\sigma^2 = \\mathrm{Var}(Y)\\) "
        "is assessed with seed-blocked estimators; spatial correlation enters "
        "via seed aggregation (mean) and CosWM \\(n_{\\mathrm{eff}}\\) (variance "
        "cross-check).",
        "",
        "## A. Cell mean and geomean",
        "",
        "Seed spatial means",
        "",
        r"$$",
        r"\hat\mu_j = N_x^{-1}\sum_{i=1}^{N_x} Y_{ij},"
        r"\qquad"
        r"\hat\mu = N_s^{-1}\sum_{j=1}^{N_s}\hat\mu_j,"
        r"\qquad"
        r"\hat G = e^{\hat\mu}.",
        r"$$",
        "",
        "Under seed independence (supported by prior IID checks),",
        "",
        r"$$",
        r"\mathrm{SE}(\hat\mu) = \frac{s_\mu}{\sqrt{N_s}},"
        r"\qquad "
        r"s_\mu^2 = \frac{1}{N_s-1}\sum_{j=1}^{N_s}(\hat\mu_j-\hat\mu)^2.",
        r"$$",
        "",
        "Spatial correlation is absorbed into \\(\\{\\hat\\mu_j\\}\\): "
        "high within-seed \\(\\rho\\) reduces the precision of each "
        "\\(\\hat\\mu_j\\) but does not add a false factor of \\(N_x\\) to the "
        "degrees of freedom for \\(\\hat\\mu\\).",
        "",
        "Delta method for the geomean:",
        "",
        r"$$",
        r"\mathrm{SE}(\hat G)\approx \hat G\,\mathrm{SE}(\hat\mu),"
        r"\qquad"
        r"\mathrm{RSE}(\hat G)=\mathrm{SE}(\hat\mu).",
        r"$$",
        "",
        "Classical \\(\\mathrm{RSE}(\\hat\\mu)=\\mathrm{SE}(\\hat\\mu)/|\\hat\\mu|\\) "
        f"is reported when \\(|\\hat\\mu| \\ge {MU_ABS_FLOOR:g}\\); otherwise it "
        "is ill-defined (e.g. \\(f\\_ratio\\) near 1 on the raw scale).",
        "",
        "Seeds required so \\(\\mathrm{RSE}(\\hat G)\\le\\varepsilon\\):",
        "",
        r"$$",
        r"N_{s,\mu}(\varepsilon)=\left\lceil(s_\mu/\varepsilon)^2\right\rceil.",
        r"$$",
        "",
        rf"**Sufficiency (mean):** at \(N_s={N_SEEDS}\), declare adequate if "
        f"\\(\\mathrm{{RSE}}(\\hat G)\\le {ADEQ_MEAN_RSE:g}\\).",
        "",
        "## B. Cell variance",
        "",
        "### Within-seed effective size",
        "",
        "From the cell’s CosWM (or fallback Exp/Gauss) ACF \\(\\rho(h)\\),",
        "",
        r"$$",
        r"n_{\mathrm{eff}}"
        r"= \frac{N_x}{1+2\sum_{k=1}^{N_x-1}\bigl(1-k/N_x\bigr)\rho(k\,\Delta x)}.",
        r"$$",
        "",
        "Under a Gaussian dependent sample, "
        r"\(\mathrm{Var}(s_j^2)\approx 2\sigma_j^4/(n_{\mathrm{eff}}-1)\). "
        "Typically \\(n_{\\mathrm{eff}}\\ll N_x\\), so the 101-node aperture does "
        "not provide 101 independent variance degrees of freedom.",
        "",
        "### Heteroscedastic seed pooling",
        "",
        "Per-seed sample variances \\(s_j^2=\\mathrm{Var}_i(Y_{ij})\\) (ddof=1), then",
        "",
        r"$$",
        r"\hat\sigma^2 = N_s^{-1}\sum_{j=1}^{N_s} s_j^2,"
        r"\qquad"
        r"\mathrm{SE}(\hat\sigma^2)=\frac{s_{s^2}}{\sqrt{N_s}},",
        r"$$",
        "",
        r"with \(s_{s^2}\) the sample SD of \(\{s_j^2\}\). This "
        "**seed-block** SE automatically reflects seed heteroscedasticity "
        r"and within-seed spatial dependence (no iid \(N_x N_s\) assumption).",
        "",
        "Relative SE \\(\\mathrm{RSE}(\\hat\\sigma^2)"
        "=\\mathrm{SE}(\\hat\\sigma^2)/\\hat\\sigma^2\\). Seeds required for "
        "target \\(\\varepsilon\\):",
        "",
        r"$$",
        r"N_{s,\sigma^2}(\varepsilon)"
        r"=\left\lceil\bigl(s_{s^2}/(\varepsilon\,\hat\sigma^2)\bigr)^2\right\rceil.",
        r"$$",
        "",
        rf"**Sufficiency (variance):** at \(N_s={N_SEEDS}\), declare adequate if "
        f"\\(\\mathrm{{RSE}}(\\hat\\sigma^2)\\le {ADEQ_VAR_RSE:g}\\).",
        "",
        "### Analytical cross-checks",
        "",
        r"For a Gaussian iid sample of size \(n\), "
        r"\(\mathrm{SE}(s^2)=\sigma^2\sqrt{2/(n-1)}\). We report:",
        "",
        r"- Naive: \(n = N_x N_s\) (overstates precision under spatial correlation).",
        r"- CosWM-corrected: \(n = N_s\,n_{\mathrm{eff}}\).",
        r"- Seed-block empirical: \(s_{s^2}/\sqrt{N_s}\) (primary).",
        "",
        "CSV columns `ratio_naive_over_block` and `ratio_neff_over_block` "
        rf"compare the first two to the seed-block SE at \(N_s={N_SEEDS}\).",
        "",
        "## SE curves",
        "",
        f"Plug-in curves use \\(s_\\mu\\) and \\(s_{{s^2}}\\) estimated at full "
        f"\\(N_s={N_SEEDS}\\), then \\(\\mathrm{{SE}}(N_s)=s/\\sqrt{{N_s}}\\) for "
        f"\\(N_s \\in \\{{{', '.join(str(int(s)) for s in S_GRID)}\\}}\\). "
        "Bands in the figure are p10–p90 across the 243 cells.",
        "",
        "## Results by metric",
        "",
        "| Metric | med \\(n_{\\mathrm{eff}}\\) | med \\(s_\\mu\\) | med "
        "RSE(\\(G\\)) @100 | med RSE(\\(\\sigma^2\\)) @100 | "
        "frac adeq. mean | frac adeq. var | frac both | "
        "med \\(N_{s,\\mathrm{req}}\\) mean 5% | med \\(N_{s,\\mathrm{req}}\\) var 10% |",
        "|--------|--------------------------:|-------------:|--------------------:|"
        "------------------------:|----------------:|---------------:|----------:|"
        "--------------------------:|-------------------------:|",
    ]

    for metric in METRICS:
        m = cell_df[cell_df["metric"] == metric]
        med = m.median(numeric_only=True)
        n = len(m)
        frac_mean = float(m["adequate_mean_S100"].mean()) if n else np.nan
        frac_var = float(m["adequate_var_S100"].mean()) if n else np.nan
        frac_both = float(m["adequate_both_S100"].mean()) if n else np.nan
        lines.append(
            "| {} | {:.2f} | {:.4g} | {:.4g} | {:.4g} | {:.1%} | {:.1%} | {:.1%} | "
            "{:.0f} | {:.0f} |".format(
                metric,
                med.get("n_eff", np.nan),
                med.get("s_mu", np.nan),
                med.get("rse_G_S100", np.nan),
                med.get("rse_sigma2_S100", np.nan),
                frac_mean,
                frac_var,
                frac_both,
                med.get("S_req_mean_rse05", np.nan),
                med.get("S_req_var_rse10", np.nan),
            )
        )

    # Overall n_eff vs N
    neff_all = cell_df["n_eff"].to_numpy(dtype=float)
    neff_all = neff_all[np.isfinite(neff_all)]
    lines.extend(
        [
            "",
            "## Spatial \\(n_{\\mathrm{eff}}\\) vs \\(N_x=101\\)",
            "",
            f"Across all cell×metric rows with a valid ACF fit "
            f"(n={len(neff_all)}), median \\(n_{{\\mathrm{{eff}}}}="
            rf"{np.median(neff_all):.2f}\) "
            f"(p10={np.percentile(neff_all, 10):.2f}, "
            f"p90={np.percentile(neff_all, 90):.2f}) versus nominal "
            rf"\(N_x={N_NODES}\). The aperture therefore contributes far fewer "
            "independent samples than a naive count suggests.",
            "",
        ]
    )

    # Naive SE inflation
    ratio = cell_df["ratio_naive_over_block"].to_numpy(dtype=float)
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    if ratio.size:
        lines.extend(
            [
                r"## Naive \(N_x N_s\) SE vs seed-block SE",
                "",
                f"Median ratio "
                f"`se_sigma2_naive_S100 / se_sigma2_S100` = "
                f"**{np.median(ratio):.3g}** "
                f"(p10={np.percentile(ratio, 10):.3g}, "
                f"p90={np.percentile(ratio, 90):.3g}). "
                r"Values ≪ 1 mean the iid \(N_x N_s\) formula understates the "
                "true seed-block uncertainty (overconfident).",
                "",
            ]
        )

    lines.extend(["## Conclusions", ""])
    for metric in METRICS:
        m = cell_df[cell_df["metric"] == metric]
        med = m.median(numeric_only=True)
        frac_mean = float(m["adequate_mean_S100"].mean())
        frac_var = float(m["adequate_var_S100"].mean())
        frac_both = float(m["adequate_both_S100"].mean())
        n_near0 = int(m["mu_near_zero"].sum())
        c100 = curves[(curves["metric"] == metric) & (curves["n_seeds"] == 100)]
        se_mu_med = float(c100["se_mu_median"].iloc[0]) if len(c100) else float("nan")
        se_s2_med = float(c100["se_sigma2_median"].iloc[0]) if len(c100) else float("nan")
        mean_ok = "yes" if frac_mean >= 0.9 else ("partial" if frac_mean >= 0.5 else "no")
        var_ok = "yes" if frac_var >= 0.9 else ("partial" if frac_var >= 0.5 else "no")
        lines.append(
            f"- **`{metric}`**: median \\(n_{{\\mathrm{{eff}}}}="
            rf"{med.get('n_eff', float('nan')):.2f}\)≪{N_NODES}. "
            f"At \\(N_s=100\\), median \\(\\mathrm{{SE}}(\\hat\\mu)={se_mu_med:.4g}\\), "
            f"median \\(\\mathrm{{RSE}}(\\hat G)="
            rf"{med.get('rse_G_S100', float('nan')):.4g}\) "
            f"({frac_mean:.0%} cells meet ≤{ADEQ_MEAN_RSE:g}; "
            f"mean adequacy: **{mean_ok}**"
            + (f"; {n_near0} cells with \\(|\\hat\\mu|<{MU_ABS_FLOOR:g}\\)" if n_near0 else "")
            + "). "
            f"Median \\(\\mathrm{{RSE}}(\\hat\\sigma^2)="
            rf"{med.get('rse_sigma2_S100', float('nan')):.4g}\) "
            f"({frac_var:.0%} meet ≤{ADEQ_VAR_RSE:g}; variance adequacy: "
            f"**{var_ok}**). Both: {frac_both:.0%}. "
            f"Median seeds for \\(\\mathrm{{RSE}}(G)\\le 0.05\\): "
            f"{med.get('S_req_mean_rse05', float('nan')):.0f}; "
            f"for \\(\\mathrm{{RSE}}(\\sigma^2)\\le 0.10\\): "
            f"{med.get('S_req_var_rse10', float('nan')):.0f}. "
            f"Median seed-block \\(\\mathrm{{SE}}(\\hat\\sigma^2)={se_s2_med:.4g}\\)."
        )

    lines.extend(
        [
            "",
            "### Design-cell takeaway",
            "",
            "Fixing factors inside a design cell isolates \\(\\mu\\) and "
            "\\(\\sigma^2\\) as within-cell parameters. With the present "
            f"\\(N_x\\times N_s = {N_NODES}\\times{N_SEEDS}\\) layout:",
            "",
            "1. **Mean / geomean** precision is governed by seed count and "
            "\\(s_\\mu\\); spatial replication mainly stabilises each "
            "\\(\\hat\\mu_j\\), not the cell-level SE beyond that.",
            "2. **Variance** precision is limited by seed heteroscedasticity "
            "(\\(s_{s^2}\\)) and by \\(n_{\\mathrm{eff}}\\ll N_x\\); treating all "
            "node×seed samples as iid grossly overstates precision.",
            r"3. Whether \(N_s=100\) is sufficient depends on the metric "
            "(see table and bullets above) under the stated 5% / 10% RSE rules.",
            "",
            "## Files",
            "",
            "- `adequacy_by_cell.csv` — one row per cell×metric",
            r"- `se_curves.csv` — SE vs \(N_s\) percentiles by metric",
            "- `mean_variance_adequacy.pdf` — panels (a) \\(\\mathrm{SE}(\\hat\\mu)\\) "
            "vs \\(N_s\\), (b) \\(\\mathrm{SE}(\\hat\\sigma^2)\\) vs \\(N_s\\)",
            "",
        ]
    )
    return "\n".join(lines)


def plot_se_curves(curves: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Two-panel Nature figure: SE(μ) and SE(σ²) vs N_s."""
    apply_full_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize(aspect=0.45))

    for ax, kind, ylab in [
        (axes[0], "se_mu", r"$\mathrm{SE}(\hat\mu)$"),
        (axes[1], "se_sigma2", r"$\mathrm{SE}(\hat\sigma^2)$"),
    ]:
        for metric in METRICS:
            m = curves[curves["metric"] == metric].sort_values("n_seeds")
            if m.empty:
                continue
            S = m["n_seeds"].to_numpy(dtype=float)
            med = m[f"{kind}_median"].to_numpy(dtype=float)
            lo = m[f"{kind}_p10"].to_numpy(dtype=float)
            hi = m[f"{kind}_p90"].to_numpy(dtype=float)
            color = metric_color(metric)
            ax.fill_between(S, lo, hi, color=color, alpha=0.15, linewidth=0)
            ax.plot(
                S,
                med,
                color=color,
                lw=1.0,
                label=metric_label(metric),
            )
        ax.axvline(N_SEEDS, color="0.5", ls="--", lw=0.6, zorder=0)
        ax.set_xlabel(r"Number of seeds $N_s$")
        ax.set_ylabel(ylab)
        ax.set_xlim(0, 105)
        if kind == "se_mu":
            ax.set_ylim(1e-3, 1e0)
        else:
            ax.set_ylim(1e-7, 1e0)
        ax.set_yscale("log")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(METRICS),
        frameon=False,
        fontsize=6,
    )
    add_panel_label(axes[0], 0)
    add_panel_label(axes[1], 1)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return save_figure(fig, "mean_variance_adequacy", out_dir=out_dir)


def main() -> None:
    print(f"Loading {DATA_PATH} …")
    df = load_ratios()
    print(f"Loaded {len(df):,} rows")

    print(f"Loading ACF fits {ACF_FIT_PATH} …")
    acf = pd.read_csv(ACF_FIT_PATH)
    # Index ACF by factor tuple + metric
    acf_idx = acf.set_index([*FACTORS, "metric"], drop=False)

    # Precompute n_eff per ACF row
    n_eff_map: dict[tuple, float] = {}
    for key, row in acf_idx.iterrows():
        n_eff_map[key] = n_eff_from_fit_row(row)

    cell_rows: list[dict] = []
    grouped = df.groupby(FACTORS, sort=True)
    n_groups = grouped.ngroups
    assert n_groups == N_CELLS, f"expected {N_CELLS} cells, got {n_groups}"

    for i, (keys, df_cell) in enumerate(grouped):
        keys_t = keys if isinstance(keys, tuple) else (keys,)
        cell_keys = dict(zip(FACTORS, keys_t))
        for metric in METRICS:
            chi = df_cell.pivot(index="node", columns="seed", values=metric)
            arr = chi.to_numpy(dtype=float)
            with np.errstate(invalid="ignore", divide="ignore"):
                Y = np.where(np.isfinite(arr) & (arr > 0), np.log(arr), np.nan)
            key = (*keys_t, metric)
            n_eff = n_eff_map.get(key, np.nan)
            cell_rows.append(assess_cell_metric(Y, cell_keys, metric, n_eff))
        if (i + 1) % 25 == 0 or i == 0 or (i + 1) == n_groups:
            print(f"  assessed cell {i + 1}/{n_groups}")

    cell_df = pd.DataFrame.from_records(cell_rows)
    curves = build_se_curves(cell_df)

    out_dir = figure_dir("chi_variables", "mean_variance_adequacy")
    cell_path = out_dir / "adequacy_by_cell.csv"
    curves_path = out_dir / "se_curves.csv"
    md_path = out_dir / "mean_variance_adequacy.md"

    cell_df.to_csv(cell_path, index=False)
    curves.to_csv(curves_path, index=False)

    summary = build_summary_md(cell_df, curves)
    md_path.write_text(summary, encoding="utf-8")

    paths = plot_se_curves(curves, out_dir)

    print()
    print(summary)
    print(f"Wrote {cell_path} ({len(cell_df):,} rows)")
    print(f"Wrote {curves_path} ({len(curves):,} rows)")
    print(f"Wrote {md_path}")
    for p in paths:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
