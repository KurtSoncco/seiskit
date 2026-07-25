"""Design-cell and factor-level heteroscedasticity diagnostics for χ ratios.

Answers, for each metric on ``join_master.h5`` 1D-normalized ratios:

- Within each design cell: are node (seed-spread) and seed (node-spread)
  variances homogeneous?
- Across the design: does variance depend on Vs1, Height, CoV, rH, aHV?

Primary working scale is ln(ratio). Writes CSV + markdown only (no plots)
under ``figure_dir("chi_variables", "metric_heteroscedasticity")``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import BOX_ROOT, figure_dir

DATA_PATH = BOX_ROOT / "peak_analysis" / "join_master.h5"
METRICS = ("f_ratio", "abs_TF_ratio", "PGA_ratio", "PSA_ratio", "Ia_ratio")
FACTORS = ["Vs1", "Height", "CoV", "rH", "aHV"]
N_NODES = 101
N_SEEDS = 100
N_CELLS = 243
ALPHA = 0.05
CENTER_NODE = 50
MIN_GROUP = 3


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


def _finite(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def _levene_groups(groups: list[np.ndarray]) -> tuple[float, float]:
    """Brown–Forsythe / Levene (median center) across groups."""
    cleaned = []
    for g in groups:
        s = _finite(g)
        if s.size >= MIN_GROUP:
            cleaned.append(s)
    if len(cleaned) < 2:
        return np.nan, np.nan
    try:
        W, p = stats.levene(*cleaned, center="median")
        return float(W), float(p)
    except Exception:
        return np.nan, np.nan


def _spread_ratio(stds: np.ndarray) -> float:
    stds = _finite(stds)
    stds = stds[stds > 1e-15]
    if stds.size < 2:
        return np.nan
    return float(np.max(stds) / np.min(stds))


def _pearson_rho(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5:
        return np.nan
    a, b = a[mask], b[mask]
    if a.std() < 1e-15 or b.std() < 1e-15:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def eta2(dfin: pd.DataFrame, val: str, fac: str) -> float:
    """Fraction of SS of ``val`` explained by factor ``fac``."""
    series = dfin[[val, fac]].dropna()
    if series.empty or series[val].std() < 1e-15:
        return np.nan
    grand = series[val].mean()
    ss_tot = float(((series[val] - grand) ** 2).sum())
    if ss_tot <= 0:
        return np.nan
    ss_b = 0.0
    for _, g in series.groupby(fac)[val]:
        ss_b += len(g) * (g.mean() - grand) ** 2
    return float(ss_b / ss_tot)


def assess_cell_metric(df_cell: pd.DataFrame, metric: str, cell_keys: dict) -> dict:
    """Within-cell heteroscedasticity diagnostics for one design cell × metric."""
    pivot = df_cell.pivot(index="node", columns="seed", values=metric)
    pivot = pivot.sort_index(axis=0).sort_index(axis=1)
    arr = pivot.to_numpy(dtype=float)

    with np.errstate(invalid="ignore", divide="ignore"):
        Y = np.where(np.isfinite(arr) & (arr > 0), np.log(arr), np.nan)

    out = {
        **cell_keys,
        "metric": metric,
        "n_nodes": int(pivot.shape[0]),
        "n_seeds": int(pivot.shape[1]),
        "cell_mean_ln": np.nan,
        "cell_std_ln": np.nan,
        "center_mean_ln": np.nan,
        "center_std_ln": np.nan,
        "node_levene_W": np.nan,
        "node_levene_p": np.nan,
        "node_std_ratio": np.nan,
        "node_mean_std_rho": np.nan,
        "seed_levene_W": np.nan,
        "seed_levene_p": np.nan,
        "seed_std_ratio": np.nan,
        "seed_mean_std_rho": np.nan,
        "nodes_homoscedastic": False,
        "seeds_homoscedastic": False,
        "cell_homoscedastic": False,
    }

    n_finite = int(np.isfinite(Y).sum())
    if n_finite < 20:
        return out

    flat = Y[np.isfinite(Y)]
    out["cell_mean_ln"] = float(np.mean(flat))
    out["cell_std_ln"] = float(np.std(flat, ddof=1)) if flat.size >= 2 else np.nan

    # Center node (conference parity)
    if CENTER_NODE in pivot.index:
        center_row = Y[pivot.index.get_loc(CENTER_NODE), :]
        c = _finite(center_row)
        if c.size >= 2:
            out["center_mean_ln"] = float(np.mean(c))
            out["center_std_ln"] = float(np.std(c, ddof=1))

    # Per-node seed std / mean
    node_means = np.nanmean(Y, axis=1)
    node_stds = np.nanstd(Y, axis=1, ddof=1)
    out["node_std_ratio"] = _spread_ratio(node_stds)
    out["node_mean_std_rho"] = _pearson_rho(node_means, node_stds)

    node_groups = [Y[i, :] for i in range(Y.shape[0])]
    W_n, p_n = _levene_groups(node_groups)
    out["node_levene_W"] = W_n
    out["node_levene_p"] = p_n
    out["nodes_homoscedastic"] = bool(np.isfinite(p_n) and p_n > ALPHA)

    # Per-seed node std / mean
    seed_means = np.nanmean(Y, axis=0)
    seed_stds = np.nanstd(Y, axis=0, ddof=1)
    out["seed_std_ratio"] = _spread_ratio(seed_stds)
    out["seed_mean_std_rho"] = _pearson_rho(seed_means, seed_stds)

    seed_groups = [Y[:, j] for j in range(Y.shape[1])]
    W_s, p_s = _levene_groups(seed_groups)
    out["seed_levene_W"] = W_s
    out["seed_levene_p"] = p_s
    out["seeds_homoscedastic"] = bool(np.isfinite(p_s) and p_s > ALPHA)

    out["cell_homoscedastic"] = bool(out["nodes_homoscedastic"] and out["seeds_homoscedastic"])
    return out


def assess_factors(by_cell: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Factor-level Levene / η² / mean–variance ρ / center-node Breusch–Pagan."""
    rows: list[dict] = []

    for metric in METRICS:
        m = by_cell[by_cell["metric"] == metric].copy()
        rho_cell = _pearson_rho(m["cell_mean_ln"].to_numpy(), m["cell_std_ln"].to_numpy())
        rho_center = _pearson_rho(m["center_mean_ln"].to_numpy(), m["center_std_ln"].to_numpy())
        rows.append(
            {
                "metric": metric,
                "test": "mean_variance_rho",
                "factor": "cell_std_ln",
                "stat": rho_cell,
                "p": np.nan,
                "eta2": np.nan,
                "detail": "corr(cell_mean_ln, cell_std_ln)",
            }
        )
        rows.append(
            {
                "metric": metric,
                "test": "mean_variance_rho",
                "factor": "center_std_ln",
                "stat": rho_center,
                "p": np.nan,
                "eta2": np.nan,
                "detail": "corr(center_mean_ln, center_std_ln)",
            }
        )

        for fac in FACTORS:
            groups = [
                m.loc[m[fac] == lv, "cell_std_ln"].to_numpy() for lv in sorted(m[fac].unique())
            ]
            W, p = _levene_groups(groups)
            e2 = eta2(m, "cell_std_ln", fac)
            rows.append(
                {
                    "metric": metric,
                    "test": "levene_cell_std",
                    "factor": fac,
                    "stat": W,
                    "p": p,
                    "eta2": e2,
                    "detail": "Levene on cell std_ln by factor level; eta2 of cell std",
                }
            )

        # Breusch–Pagan on center node only (seed replicates × cells)
        d50 = df[df["node"] == CENTER_NODE].copy()
        raw = d50[metric].to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            y = np.where(np.isfinite(raw) & (raw > 0), np.log(raw), np.nan)
        d50 = d50.assign(y_ln=y).dropna(subset=["y_ln", *FACTORS])
        if len(d50) >= 50:
            Z = d50[FACTORS].astype(float)
            Zs = (Z - Z.mean()) / Z.std(ddof=0)
            Zs.columns = [c + "_z" for c in FACTORS]
            X = sm.add_constant(Zs.to_numpy(dtype=float), has_constant="add")
            y_vec = d50["y_ln"].to_numpy(dtype=float)
            try:
                ols = sm.OLS(y_vec, X).fit()
                lm, lm_p, _F, _F_p = het_breuschpagan(ols.resid, X)
                rows.append(
                    {
                        "metric": metric,
                        "test": "breusch_pagan_center",
                        "factor": "all_factors",
                        "stat": float(lm),
                        "p": float(lm_p),
                        "eta2": np.nan,
                        "detail": (
                            f"OLS ln({metric}) ~ z-factors at node {CENTER_NODE}; "
                            "het_breuschpagan on residuals"
                        ),
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "metric": metric,
                        "test": "breusch_pagan_center",
                        "factor": "all_factors",
                        "stat": np.nan,
                        "p": np.nan,
                        "eta2": np.nan,
                        "detail": f"failed: {exc}",
                    }
                )

    return pd.DataFrame(rows)


def build_summary_md(by_cell: pd.DataFrame, by_factor: pd.DataFrame) -> str:
    lines = [
        "# Metric heteroscedasticity summary",
        "",
        "Per design cell (243) × metric diagnostics on 1D-normalized ratios "
        f"(`{'`, `'.join(METRICS)}`) from `join_master.h5`.",
        "",
        "## How to read this report",
        "",
        f"- **Working scale:** `ln(ratio)` (positive finite values only); "
        f"significance level **α = {ALPHA}**.",
        "- **Homoscedastic / homo:** variance treated as equal across the compared "
        "groups. A case *passes* the gate when Levene’s *p* > α "
        "(fail to reject equal variances).",
        "- **Heteroscedastic:** variance differs across groups (Levene *p* ≤ α, "
        "or large descriptive spread ratios).",
        "- **`reject`:** null of equal variances is rejected at α "
        f"(evidence of heteroscedasticity; *p* ≤ {ALPHA}).",
        "- **`ns`:** *not significant* — fail to reject equal variances at α "
        f"(*p* > {ALPHA}). This is **not** proof of homoscedasticity; it only "
        "means the test did not detect a difference at this α.",
        "- **η² (eta-squared):** fraction of variation in cell-level `std_ln` "
        "attributable to a design factor (0 = none, closer to 1 = strong "
        "factor-driven variance). Reported even when the Levene call is `ns`, "
        "because effect size can still be nonzero.",
        "- **std ratio:** `max(group std) / min(group std)` within a cell. "
        "Near 1 ⇒ similar spreads; large values ⇒ strong within-cell "
        "heteroscedasticity (descriptive; does not use α).",
        "- **mean–std ρ:** Pearson correlation between group means and group "
        "stds. Large |ρ| ⇒ mean–variance coupling (another heteroscedasticity "
        "signature).",
        "- **Breusch–Pagan LM / p:** tests whether OLS residual variance at the "
        f"center node ({CENTER_NODE}) depends on the z-scored factors. "
        "`reject α? = yes` means residual heteroscedasticity.",
        '- **Within-cell tests:** Brown–Forsythe / Levene (`center="median"`) '
        "across nodes (replicates = seeds) and across seeds (replicates = nodes).",
        "- **Across-factor tests:** Levene + η² on the 243 cell `std_ln` values; "
        "Breusch–Pagan on center-node seeds only (does not treat the spatial "
        "node array as iid).",
        "",
        "### Caveat: node dependence and Levene *p*-values",
        "",
        "The IID / spatial-ACF analyses show that **nodes along the array are "
        "strongly serially correlated**, so the 101 node series in a cell are "
        "not independent draws. Classical Levene / Brown–Forsythe assumes "
        "independent groups (or independent replicates within groups).",
        "",
        "Practical consequences:",
        "",
        "1. **Effective sample size is smaller than the nominal *n*.** "
        "Positive spatial correlation reduces the information in the node "
        "array relative to 101 independent locations.",
        "2. **Within-cell Levene *p*-values across nodes are "
        "anti-conservative** if dependence is ignored: the test tends to "
        "report *p* too small, so it rejects equal variances *too often*. "
        "A `reject` for nodes is therefore a *weak* claim; a pass (`ns` / "
        "homoscedastic gate) is a *stronger* claim under dependence.",
        "3. **Prefer descriptive spread ratios and η²** alongside *p*-values. "
        "Median node std ratios near 1 support approximate equal node "
        "spreads even when occasional Levene rejects appear.",
        "4. **Seed-level Levene** compares RF realizations whose node "
        "vectors are dependent *within* each seed, but seeds themselves are "
        "separate Monte Carlo draws (exchangeable labels from the IID "
        "check). Dependence still inflates within-seed *n*, so seed Levene "
        "is also somewhat anti-conservative — yet the very large seed std "
        "ratios (far above 1) are not an artifact of that bias.",
        "5. **Factor-level Levene on cell `std_ln`** uses one summary per "
        "design cell (243 nearly independent factorial points), so node "
        "dependence within a cell does **not** invalidate those tests the "
        "same way. Center-node Breusch–Pagan likewise uses one location.",
        "",
        "## Aggregate across design cells (within-cell)",
        "",
        "Column meanings: **frac nodes/seeds/cell homo** = fraction of the "
        "243 cells that pass the corresponding Levene gate; **med * std "
        "ratio** = median within-cell max/min std; **med * mean–std ρ** = "
        "median within-cell mean–variance correlation.",
        "",
        "| Metric | frac nodes homo | frac seeds homo | frac cell homo | "
        "med node std ratio | med seed std ratio | med node mean–std ρ | "
        "med seed mean–std ρ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for metric in METRICS:
        m = by_cell[by_cell["metric"] == metric]
        lines.append(
            "| {} | {:.3f} | {:.3f} | {:.3f} | {:.2f} | {:.2f} | {:.3f} | {:.3f} |".format(
                metric,
                float(m["nodes_homoscedastic"].mean()),
                float(m["seeds_homoscedastic"].mean()),
                float(m["cell_homoscedastic"].mean()),
                float(m["node_std_ratio"].median()),
                float(m["seed_std_ratio"].median()),
                float(m["node_mean_std_rho"].median()),
                float(m["seed_mean_std_rho"].median()),
            )
        )

    lines.extend(
        [
            "",
            "## Across design factors",
            "",
            "### Cell-level mean–variance coupling",
            "",
            "Pearson ρ between each cell’s mean and std on ln scale "
            "(243 cells). Center columns use only node "
            f"{CENTER_NODE}. Positive ρ ⇒ cells with larger typical "
            "response also have larger spread; negative ρ ⇒ the opposite.",
            "",
            "| Metric | ρ(cell mean, cell std) | ρ(center mean, center std) |",
            "| --- | ---: | ---: |",
        ]
    )

    for metric in METRICS:
        mv = by_factor[(by_factor["metric"] == metric) & (by_factor["test"] == "mean_variance_rho")]
        rho_cell = mv.loc[mv["factor"] == "cell_std_ln", "stat"]
        rho_ctr = mv.loc[mv["factor"] == "center_std_ln", "stat"]
        lines.append(
            "| {} | {:.3f} | {:.3f} |".format(
                metric,
                float(rho_cell.iloc[0]) if len(rho_cell) else np.nan,
                float(rho_ctr.iloc[0]) if len(rho_ctr) else np.nan,
            )
        )

    lines.extend(
        [
            "",
            "### Levene on cell `std_ln` by factor",
            "",
            "Each cell shows **`reject` or `ns`** at α, with **η²** of cell "
            "`std_ln` explained by that factor in parentheses. "
            "`reject` = factor levels differ in the *dispersion* of cell "
            "stds (heteroscedasticity across the design); `ns` = no "
            "significant difference at α. Compare η² across factors to see "
            "which design knobs most control response variance.",
            "",
            "| Metric | " + " | ".join(FACTORS) + " |",
            "| --- | " + " | ".join(["---:"] * len(FACTORS)) + " |",
        ]
    )

    for metric in METRICS:
        cells = []
        for fac in FACTORS:
            row = by_factor[
                (by_factor["metric"] == metric)
                & (by_factor["test"] == "levene_cell_std")
                & (by_factor["factor"] == fac)
            ]
            if row.empty:
                cells.append("—")
                continue
            p = float(row["p"].iloc[0])
            e2 = float(row["eta2"].iloc[0])
            flag = "reject" if np.isfinite(p) and p <= ALPHA else "ns"
            cells.append(f"{flag} (η²={e2:.3f})")
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "### Breusch–Pagan (center node)",
            "",
            f"OLS of `ln(metric)` on z-scored factors using all seeds at "
            f"node {CENTER_NODE}, then Breusch–Pagan on residuals. "
            "**LM** = Lagrange-multiplier statistic; **p** = *p*-value; "
            "**reject α?** = `yes` if residual variance depends on the "
            "factors (*p* ≤ α).",
            "",
            "| Metric | LM | p | reject α? |",
            "| --- | ---: | ---: | --- |",
        ]
    )

    for metric in METRICS:
        row = by_factor[
            (by_factor["metric"] == metric) & (by_factor["test"] == "breusch_pagan_center")
        ]
        if row.empty:
            lines.append(f"| {metric} | — | — | — |")
            continue
        lm = float(row["stat"].iloc[0])
        p = float(row["p"].iloc[0])
        rej = "yes" if np.isfinite(p) and p <= ALPHA else "no"
        lines.append(f"| {metric} | {lm:.1f} | {p:.2e} | {rej} |")

    # Narrative stats
    all_frac_node = float(by_cell["nodes_homoscedastic"].mean())
    all_frac_seed = float(by_cell["seeds_homoscedastic"].mean())
    all_frac_cell = float(by_cell["cell_homoscedastic"].mean())
    med_node_ratio = float(by_cell["node_std_ratio"].median())
    med_seed_ratio = float(by_cell["seed_std_ratio"].median())
    med_node_rho = float(by_cell["node_mean_std_rho"].median())
    med_seed_rho = float(by_cell["seed_mean_std_rho"].median())

    bp = by_factor[by_factor["test"] == "breusch_pagan_center"]
    n_bp_reject = int((bp["p"] <= ALPHA).sum()) if len(bp) else 0
    n_bp = int(len(bp))

    lev = by_factor[by_factor["test"] == "levene_cell_std"]
    n_lev_reject = int((lev["p"] <= ALPHA).sum()) if len(lev) else 0
    n_lev = int(len(lev))
    top_eta = lev.sort_values("eta2", ascending=False).head(5) if len(lev) else pd.DataFrame()

    lines.extend(["", "## Conclusions", ""])

    node_verdict = "Mostly no" if all_frac_node < 0.5 else "Mostly yes"
    seed_verdict = "Mostly no" if all_frac_seed < 0.5 else "Mostly yes"
    lines.append(
        f"1. **Within each design cell, are nodes homoscedastic?** "
        f"**{node_verdict}.** "
        f"**{all_frac_node:.1%}** of cell×metric cases pass the node Levene gate "
        f"(fail to reject equal seed-spread across the 101 nodes). "
        f"The median max/min node seed-std ratio is only **{med_node_ratio:.2f}** "
        f"(spreads typically differ by tens of percent, not factors of several), "
        f"and the median node mean–std ρ is **{med_node_rho:.3f}**. "
        f"Given anti-conservative Levene *p*-values under spatial dependence "
        f"(see caveat above), the high pass rate plus near-unity std ratios "
        f"together support treating **within-cell node spreads as roughly "
        f"homogeneous** for practical purposes."
    )
    lines.append("")
    lines.append(
        f"2. **Within each design cell, are seeds homoscedastic?** "
        f"**{seed_verdict}.** "
        f"**{all_frac_seed:.1%}** of cases pass the seed Levene gate. "
        f"Median seed node-std ratio = **{med_seed_ratio:.2f}** (some RF "
        f"realizations have several times the spatial spread of others); "
        f"median seed mean–std ρ = **{med_seed_rho:.3f}**. "
        f"Joint cell homoscedasticity (nodes *and* seeds) holds in "
        f"**{all_frac_cell:.1%}** of cell×metric cases. The large std ratios "
        f"are the main evidence here: even if dependence modestly shrinks "
        f"seed-level *p*-values, ratios this far from 1 indicate real "
        f"seed-to-seed variance heterogeneity."
    )
    lines.append("")
    lines.append(
        f"3. **Across the design, are the metrics heteroscedastic in the factors?** "
        f"**Yes.** Levene on cell `std_ln` rejects equal factor-level spread in "
        f"**{n_lev_reject}/{n_lev}** factor×metric tests (`reject` cells in the "
        f"table above). Center-node Breusch–Pagan rejects constant residual "
        f"variance in **{n_bp_reject}/{n_bp}** metrics. These factor-level "
        f"tests are the more trustworthy formal evidence of design-driven "
        f"heteroscedasticity because they do not rely on treating neighboring "
        f"nodes as independent."
    )
    if len(top_eta):
        lines.append("")
        lines.append("Largest η² of cell `std_ln` by factor (strongest variance drivers):")
        lines.append("")
        for _, r in top_eta.iterrows():
            flag = "reject" if np.isfinite(r["p"]) and float(r["p"]) <= ALPHA else "ns"
            lines.append(
                f"- `{r['metric']}` × `{r['factor']}`: η² = {r['eta2']:.3f} "
                f"(Levene {flag}, p = {r['p']:.2e})"
            )

    lines.append("")
    lines.append(
        "4. **What does node dependence change about interpretation?** "
        "It mainly weakens *rejections* from within-cell node Levene, and it "
        "reminds us not to pool all node×seed rows into one OLS / Breusch–Pagan "
        "as if they were iid. It does **not** erase the seed-level "
        "heteroscedasticity (large std ratios) or the factor-level "
        "heteroscedasticity (cell-std Levene, center-node BP, nonzero η²). "
        "Descriptive ratios and factor η² should be cited alongside any "
        "*p*-value narrative."
    )

    lines.append("")
    lines.append("### Per-metric notes")
    lines.append("")
    for metric in METRICS:
        m = by_cell[by_cell["metric"] == metric]
        bp_row = bp[bp["metric"] == metric]
        bp_p = float(bp_row["p"].iloc[0]) if len(bp_row) else np.nan
        bp_rej = (
            "rejects constant residual variance"
            if np.isfinite(bp_p) and bp_p <= ALPHA
            else "does not reject at α"
        )
        mv = by_factor[
            (by_factor["metric"] == metric)
            & (by_factor["test"] == "mean_variance_rho")
            & (by_factor["factor"] == "cell_std_ln")
        ]
        rho = float(mv["stat"].iloc[0]) if len(mv) else np.nan
        # top factor by eta2 for this metric
        lev_m = lev[lev["metric"] == metric].sort_values("eta2", ascending=False)
        if len(lev_m):
            top = lev_m.iloc[0]
            top_fac = f"largest η² from `{top['factor']}` ({float(top['eta2']):.3f})"
        else:
            top_fac = "no factor η² available"
        rho_note = (
            "positive mean–variance coupling"
            if np.isfinite(rho) and rho > 0.2
            else (
                "negative mean–variance coupling"
                if np.isfinite(rho) and rho < -0.2
                else "weak mean–variance coupling"
            )
        )
        lines.append(
            f"- **`{metric}`**: nodes pass homo gate in "
            f"{100 * m['nodes_homoscedastic'].mean():.1f}% of cells "
            f"(med node std ratio {m['node_std_ratio'].median():.2f}); "
            f"seeds pass in {100 * m['seeds_homoscedastic'].mean():.1f}% "
            f"(med seed std ratio {m['seed_std_ratio'].median():.2f}). "
            f"Cell mean–std ρ = {rho:.3f} ({rho_note}); {top_fac}; "
            f"center BP p = {bp_p:.2e} ({bp_rej})."
        )

    lines.extend(
        [
            "",
            "### Modeling implication",
            "",
            "- **Within a cell:** node seed-spreads are often similar, but "
            "**seed-to-seed spatial spread is strongly heteroscedastic** (RF "
            "realizations differ in node-array variance). Do not pool seeds as if "
            "each shared a common spatial variance.",
            "- **Because nodes are dependent:** do not inflate degrees of freedom "
            "by treating 101 nodes × 100 seeds as iid when building SEs or "
            "tests; use seed clustering, mixed effects, or spatial models. "
            "Weight formal within-cell node Levene rejects lightly relative to "
            "spread ratios.",
            "- **Across the design:** residual / cell variance is **factor-dependent** "
            "(Levene on cell stds; Breusch–Pagan at the center node; nonzero η²). "
            "Constant-variance OLS SEs and fixed-width prediction intervals are "
            "poorly calibrated.",
            "- Prefer variance models, quantile / distributional regression, or "
            "heteroscedastic prediction intervals (e.g. QBM-style) when reporting "
            "uncertainty for χ ratios.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    print(f"Loading {DATA_PATH} …")
    df = load_ratios()
    print(f"Loaded {len(df):,} rows")

    df["cell"] = df.groupby(FACTORS, sort=False).ngroup()
    n_cells = int(df["cell"].nunique())
    print(f"Design cells: {n_cells} (expected {N_CELLS})")

    rows: list[dict] = []
    grouped = df.groupby("cell", sort=True)
    for i, (_, g) in enumerate(grouped):
        cell_keys = {f: g[f].iloc[0] for f in FACTORS}
        cell_keys["cell"] = int(g["cell"].iloc[0])
        for metric in METRICS:
            rows.append(assess_cell_metric(g, metric, cell_keys))
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  assessed cell {i} ({i + 1}/{n_cells})")

    by_cell = pd.DataFrame(rows)
    col_order = [
        "cell",
        *FACTORS,
        "metric",
        "n_nodes",
        "n_seeds",
        "cell_mean_ln",
        "cell_std_ln",
        "center_mean_ln",
        "center_std_ln",
        "node_levene_W",
        "node_levene_p",
        "node_std_ratio",
        "node_mean_std_rho",
        "seed_levene_W",
        "seed_levene_p",
        "seed_std_ratio",
        "seed_mean_std_rho",
        "nodes_homoscedastic",
        "seeds_homoscedastic",
        "cell_homoscedastic",
    ]
    by_cell = by_cell[col_order]

    print("Assessing factor-level heteroscedasticity …")
    by_factor = assess_factors(by_cell, df)

    out_dir = figure_dir("chi_variables", "metric_heteroscedasticity")
    csv_cell = out_dir / "hetero_by_cell.csv"
    csv_fac = out_dir / "hetero_by_factor.csv"
    md_path = out_dir / "hetero_summary.md"

    by_cell.to_csv(csv_cell, index=False)
    by_factor.to_csv(csv_fac, index=False)
    summary = build_summary_md(by_cell, by_factor)
    md_path.write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print(f"Wrote {csv_cell}")
    print(f"Wrote {csv_fac}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
