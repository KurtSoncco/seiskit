"""Per-design-cell geometric means and seed/node variance splits for χ ratios.

Loads ``join_master.h5`` 1D-normalized ratios, and for each of 243 design cells
× 5 metrics computes:

- Geomeans: per seed (across nodes), per node (across seeds), overall
- Law-of-total-variance splits (log scale): seed-conditioned and node-conditioned
- Descriptive within/between averages and naive-sum diagnostic (Box parity)

Writes CSV + markdown under ``figure_dir("chi_variables", "central_variability")``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import BOX_ROOT, figure_dir

DATA_PATH = BOX_ROOT / "peak_analysis" / "join_master.h5"
METRICS = ("f_ratio", "abs_TF_ratio", "PGA_ratio", "PSA_ratio", "Ia_ratio")
FACTORS = ["Vs1", "Height", "CoV", "rH", "aHV"]
N_NODES = 101
N_SEEDS = 100
N_CELLS = 243
GAP_TOL = 1e-9


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


def _pop_var(x: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """Population variance (ddof=0), ignoring NaNs."""
    return np.nanvar(x, axis=axis, ddof=0)


def _samp_var(x: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    """Sample variance (ddof=1), ignoring NaNs."""
    return np.nanvar(x, axis=axis, ddof=1)


def assess_cell_metric(
    df_cell: pd.DataFrame,
    metric: str,
    cell_keys: dict,
) -> tuple[dict, list[dict], list[dict]]:
    """Return (cell_summary_row, within_per_seed_rows, between_per_node_rows)."""
    chi = df_cell.pivot(index="node", columns="seed", values=metric)
    arr = chi.to_numpy(dtype=float)
    nodes = chi.index.to_numpy()
    seeds = chi.columns.to_numpy()

    # Log scale; non-positive / non-finite → NaN
    with np.errstate(invalid="ignore", divide="ignore"):
        Y = np.where(np.isfinite(arr) & (arr > 0), np.log(arr), np.nan)

    n_finite = int(np.isfinite(Y).sum())
    flat = Y[np.isfinite(Y)]
    chi_flat = arr[np.isfinite(arr) & (arr > 0)]

    empty = {
        **cell_keys,
        "metric": metric,
        "n_nodes": int(len(nodes)),
        "n_seeds": int(len(seeds)),
        "n_finite": n_finite,
        "chi_geom": np.nan,
        "ln_chi_bar": np.nan,
        "chi_mean": np.nan,
        "chi_median": np.nan,
        "s_W_bar": np.nan,
        "s_mu": np.nan,
        "s_B_bar": np.nan,
        "s_nu": np.nan,
        "s_total": np.nan,
        "frac_W_seed": np.nan,
        "frac_mu": np.nan,
        "frac_B_node": np.nan,
        "frac_nu": np.nan,
        "s_T_naive": np.nan,
        "s_T_gap": np.nan,
        "s_interaction": np.nan,
        "frac_interaction": np.nan,
        "ratio_W_over_mu": np.nan,
        "ratio_B_over_nu": np.nan,
        "s_W_bar_ddof1": np.nan,
        "s_mu_ddof1": np.nan,
        "s_B_bar_ddof1": np.nan,
        "s_nu_ddof1": np.nan,
        "s_total_ddof1": np.nan,
        "G_seed_p10": np.nan,
        "G_seed_p90": np.nan,
        "G_node_p10": np.nan,
        "G_node_p90": np.nan,
        "decomp_ok": False,
    }
    if flat.size < 8:
        return empty, [], []

    # Means: axis 0 = node, axis 1 = seed
    mu_j = np.nanmean(Y, axis=0)  # (m,) per seed
    nu_i = np.nanmean(Y, axis=1)  # (n,) per node
    Y_bar = float(np.nanmean(Y))

    s2_W_j = _pop_var(Y, axis=0)  # within seed (across nodes)
    s2_B_i = _pop_var(Y, axis=1)  # across seeds (at fixed node)
    s2_W_bar = float(np.nanmean(s2_W_j))
    s2_B_bar = float(np.nanmean(s2_B_i))
    s2_mu = float(_pop_var(mu_j[np.isfinite(mu_j)]))
    s2_nu = float(_pop_var(nu_i[np.isfinite(nu_i)]))
    s2_total = float(_pop_var(flat))

    # Additive two-way fit residual: Y_ij - mu_j - nu_i + Y_bar
    mu_grid = mu_j[np.newaxis, :]
    nu_grid = nu_i[:, np.newaxis]
    resid = Y - mu_grid - nu_grid + Y_bar
    s2_interaction = float(_pop_var(resid[np.isfinite(resid)]))

    s2_naive = s2_W_bar + s2_B_bar
    s_total = float(np.sqrt(s2_total)) if s2_total >= 0 else np.nan
    s_T_naive = float(np.sqrt(s2_naive)) if s2_naive >= 0 else np.nan
    s_T_gap = (
        float(s_T_naive - s_total) if np.isfinite(s_T_naive) and np.isfinite(s_total) else np.nan
    )

    seed_recon = s2_W_bar + s2_mu
    node_recon = s2_B_bar + s2_nu
    decomp_ok = abs(seed_recon - s2_total) < max(GAP_TOL, 1e-6 * max(s2_total, 1e-30)) and abs(
        node_recon - s2_total
    ) < max(GAP_TOL, 1e-6 * max(s2_total, 1e-30))

    def _frac(num: float) -> float:
        if not np.isfinite(s2_total) or s2_total <= 0:
            return np.nan
        return float(num / s2_total)

    # ddof=1 counterparts (sensitivity)
    s2_W_j_1 = _samp_var(Y, axis=0)
    s2_B_i_1 = _samp_var(Y, axis=1)
    s2_W_bar_1 = float(np.nanmean(s2_W_j_1))
    s2_B_bar_1 = float(np.nanmean(s2_B_i_1))
    mu_ok = mu_j[np.isfinite(mu_j)]
    nu_ok = nu_i[np.isfinite(nu_i)]
    s2_mu_1 = float(_samp_var(mu_ok)) if mu_ok.size > 1 else np.nan
    s2_nu_1 = float(_samp_var(nu_ok)) if nu_ok.size > 1 else np.nan
    s2_total_1 = float(_samp_var(flat)) if flat.size > 1 else np.nan

    G_seed = np.exp(mu_j)
    G_node = np.exp(nu_i)
    G_seed_f = G_seed[np.isfinite(G_seed)]
    G_node_f = G_node[np.isfinite(G_node)]

    summary = {
        **cell_keys,
        "metric": metric,
        "n_nodes": int(np.isfinite(nu_i).sum()),
        "n_seeds": int(np.isfinite(mu_j).sum()),
        "n_finite": n_finite,
        "chi_geom": float(np.exp(Y_bar)),
        "ln_chi_bar": Y_bar,
        "chi_mean": float(np.mean(chi_flat)) if chi_flat.size else np.nan,
        "chi_median": float(np.median(chi_flat)) if chi_flat.size else np.nan,
        "s_W_bar": float(np.sqrt(s2_W_bar)),
        "s_mu": float(np.sqrt(s2_mu)),
        "s_B_bar": float(np.sqrt(s2_B_bar)),
        "s_nu": float(np.sqrt(s2_nu)),
        "s_total": s_total,
        "frac_W_seed": _frac(s2_W_bar),
        "frac_mu": _frac(s2_mu),
        "frac_B_node": _frac(s2_B_bar),
        "frac_nu": _frac(s2_nu),
        "s_T_naive": s_T_naive,
        "s_T_gap": s_T_gap,
        "s_interaction": float(np.sqrt(s2_interaction)),
        "frac_interaction": _frac(s2_interaction),
        "ratio_W_over_mu": float(s2_W_bar / s2_mu) if s2_mu > 0 else np.nan,
        "ratio_B_over_nu": float(s2_B_bar / s2_nu) if s2_nu > 0 else np.nan,
        "s_W_bar_ddof1": float(np.sqrt(s2_W_bar_1)),
        "s_mu_ddof1": float(np.sqrt(s2_mu_1)) if np.isfinite(s2_mu_1) else np.nan,
        "s_B_bar_ddof1": float(np.sqrt(s2_B_bar_1)),
        "s_nu_ddof1": float(np.sqrt(s2_nu_1)) if np.isfinite(s2_nu_1) else np.nan,
        "s_total_ddof1": float(np.sqrt(s2_total_1)) if np.isfinite(s2_total_1) else np.nan,
        "G_seed_p10": float(np.percentile(G_seed_f, 10)) if G_seed_f.size else np.nan,
        "G_seed_p90": float(np.percentile(G_seed_f, 90)) if G_seed_f.size else np.nan,
        "G_node_p10": float(np.percentile(G_node_f, 10)) if G_node_f.size else np.nan,
        "G_node_p90": float(np.percentile(G_node_f, 90)) if G_node_f.size else np.nan,
        "decomp_ok": bool(decomp_ok),
    }

    within_rows: list[dict] = []
    for j, seed in enumerate(seeds):
        if not np.isfinite(mu_j[j]):
            continue
        within_rows.append(
            {
                **cell_keys,
                "metric": metric,
                "seed": int(seed) if np.isfinite(seed) else seed,
                "mu_j": float(mu_j[j]),
                "chi_geom_j": float(np.exp(mu_j[j])),
                "s_W_sq": float(s2_W_j[j]),
                "s_W": float(np.sqrt(s2_W_j[j])),
                "n_nodes": int(np.isfinite(Y[:, j]).sum()),
            }
        )

    between_rows: list[dict] = []
    for i, node in enumerate(nodes):
        if not np.isfinite(nu_i[i]):
            continue
        between_rows.append(
            {
                **cell_keys,
                "metric": metric,
                "node": int(node) if np.isfinite(node) else node,
                "nu_i": float(nu_i[i]),
                "chi_geom_i": float(np.exp(nu_i[i])),
                "s_B_sq": float(s2_B_i[i]),
                "s_B": float(np.sqrt(s2_B_i[i])),
                "n_seeds": int(np.isfinite(Y[i, :]).sum()),
            }
        )

    return summary, within_rows, between_rows


def build_summary_md(cell_df: pd.DataFrame) -> str:
    lines = [
        "# Central tendency and variability summary",
        "",
        "Per-design-cell log-χ geometric means and variance components "
        f"(`{'`, `'.join(METRICS)}`) from `join_master.h5`.",
        "",
        f"- Design cells: **{N_CELLS}** (`{'`, `'.join(FACTORS)}`)",
        f"- Within cell: **{N_NODES}** nodes × **{N_SEEDS}** seeds",
        "- Non-positive / non-finite χ values are dropped before taking logs",
        "- Primary variances use population moments (ddof = 0); "
        "`*_ddof1` columns in `cell_summary.csv` use sample moments (ddof = 1)",
        "",
        "## Output files",
        "",
        "| File | Rows | Contents |",
        "|------|-----:|----------|",
        "| `cell_summary.csv` | one per cell × metric "
        f"({N_CELLS} × {len(METRICS)} = {N_CELLS * len(METRICS)}) | "
        "Design factors, overall geomean $G$, log-mean, linear mean/median, "
        "rms components $s_W$, $s_\\mu$, $s_B$, $s_\\nu$, $s_{\\mathrm{total}}$, "
        "fractions, naive-sum diagnostic, interaction residual, "
        "P10–P90 of seed/node geomeans, `decomp_ok` |",
        "| `within_per_seed.csv` | one per cell × metric × seed | "
        "Per-seed log-mean $\\mu_j$, geomean $G^{\\mathrm{seed}}_j = e^{\\mu_j}$, "
        "within-seed variance $s^{2}_{W,j}$ and $s_{W,j}$ |",
        "| `between_per_node.csv` | one per cell × metric × node | "
        "Per-node log-mean $\\nu_i$, geomean $G^{\\mathrm{node}}_i = e^{\\nu_i}$, "
        "across-seed variance $s^{2}_{B,i}$ and $s_{B,i}$ |",
        "| `summary.md` | this file | Equations, CSV guide, "
        "median summaries and conclusions across cells |",
        "",
        "## Notation",
        "",
        "Fix one design cell and one metric $\\chi$. Index **nodes** by "
        "$i = 1,\\ldots,n$ ($n = 101$) and **seeds** by $j = 1,\\ldots,m$ "
        "($m = 100$). Work on the log scale",
        "",
        r"$$",
        r"Y_{ij} = \ln \chi_{ij}.",
        r"$$",
        "",
        "All sums below are over finite $Y_{ij}$ only. Population (ddof = 0) "
        "means and variances divide by the count of terms (not $n-1$).",
        "",
        "### Central tendency",
        "",
        "**Seed log-mean** (average across nodes at fixed seed) and **seed geomean**:",
        "",
        r"$$",
        r"\mu_j = \frac{1}{n}\sum_{i=1}^{n} Y_{ij},"
        r"\qquad "
        r"G^{\mathrm{seed}}_j = e^{\mu_j}.",
        r"$$",
        "",
        "**Node log-mean** (average across seeds at fixed node) and **node geomean**:",
        "",
        r"$$",
        r"\nu_i = \frac{1}{m}\sum_{j=1}^{m} Y_{ij},"
        r"\qquad "
        r"G^{\mathrm{node}}_i = e^{\nu_i}.",
        r"$$",
        "",
        "**Overall log-mean** and **overall geomean** (CSV: `ln_chi_bar`, `chi_geom`):",
        "",
        r"$$",
        r"\bar{Y} = \frac{1}{nm}\sum_{i=1}^{n}\sum_{j=1}^{m} Y_{ij},"
        r"\qquad "
        r"G = e^{\bar{Y}}.",
        r"$$",
        "",
        "On a complete balanced grid,",
        "",
        r"$$",
        r"G ="
        r"\Bigl(\prod_{j=1}^{m} G^{\mathrm{seed}}_j\Bigr)^{1/m} ="
        r"\Bigl(\prod_{i=1}^{n} G^{\mathrm{node}}_i\Bigr)^{1/n}.",
        r"$$",
        "",
        "Also reported in `cell_summary.csv`:",
        "",
        r"- Arithmetic mean: $\overline{\chi} = \mathrm{mean}_{ij}(\chi_{ij})$ (`chi_mean`)",
        r"- Median: $\mathrm{median}_{ij}(\chi_{ij})$ (`chi_median`)",
        r"- Percentiles of seed/node geomeans: "
        r"$G^{\mathrm{seed}}_{P10}, G^{\mathrm{seed}}_{P90}$, "
        r"$G^{\mathrm{node}}_{P10}, G^{\mathrm{node}}_{P90}$",
        "",
        "### Within-seed and across-seed conditional variances",
        "",
        "**Within-seed** variance (node scatter at fixed seed $j$):",
        "",
        r"$$",
        r"s^{2}_{W,j} ="
        r"\frac{1}{n}\sum_{i=1}^{n} (Y_{ij} - \mu_j)^2.",
        r"$$",
        "",
        "Design-level summary (average over seeds; CSV rms form "
        "`s_W_bar` $= \\sqrt{\\overline{s^{2}_W}}$):",
        "",
        r"$$",
        r"\overline{s^{2}_W} ="
        r"\frac{1}{m}\sum_{j=1}^{m} s^{2}_{W,j}.",
        r"$$",
        "",
        "**Across-seed** variance at fixed node $i$ (seed-to-seed scatter at one depth):",
        "",
        r"$$",
        r"s^{2}_{B,i} ="
        r"\frac{1}{m}\sum_{j=1}^{m} (Y_{ij} - \nu_i)^2.",
        r"$$",
        "",
        "Design-level summary (average over nodes; CSV `s_B_bar`):",
        "",
        r"$$",
        r"\overline{s^{2}_B} ="
        r"\frac{1}{n}\sum_{i=1}^{n} s^{2}_{B,i}.",
        r"$$",
        "",
        "### Law of total variance (primary splits)",
        "",
        "**Direct total variance** of all log entries "
        "(CSV `s_total` $= \\sqrt{\\sigma^{2}_{\\mathrm{total}}}$):",
        "",
        r"$$",
        r"\sigma^{2}_{\mathrm{total}} ="
        r"\frac{1}{nm}"
        r"\sum_{i=1}^{n}\sum_{j=1}^{m} (Y_{ij} - \bar{Y})^2.",
        r"$$",
        "",
        "**Variance of seed means** (CSV `s_mu` $= \\sqrt{\\sigma^{2}_{\\mu}}$):",
        "",
        r"$$",
        r"\sigma^{2}_{\mu} ="
        r"\frac{1}{m}\sum_{j=1}^{m} (\mu_j - \bar{Y})^2.",
        r"$$",
        "",
        "**Variance of node means** (CSV `s_nu` $= \\sqrt{\\sigma^{2}_{\\nu}}$):",
        "",
        r"$$",
        r"\sigma^{2}_{\nu} ="
        r"\frac{1}{n}\sum_{i=1}^{n} (\nu_i - \bar{Y})^2.",
        r"$$",
        "",
        "**Seed-conditioned decomposition** (within a realization vs across realizations):",
        "",
        r"$$",
        r"\sigma^{2}_{\mathrm{total}}"
        r"= \overline{s^{2}_W} + \sigma^{2}_{\mu}.",
        r"$$",
        "",
        "**Node-conditioned decomposition** (seed noise at a depth vs depth-to-depth structure):",
        "",
        r"$$",
        r"\sigma^{2}_{\mathrm{total}}"
        r"= \overline{s^{2}_B} + \sigma^{2}_{\nu}.",
        r"$$",
        "",
        "Both identities hold (up to floating point) on a complete grid; "
        "`decomp_ok` is `True` when both reconstructions match "
        r"$\sigma^{2}_{\mathrm{total}}$.",
        "",
        "### Fractions",
        "",
        r"$$",
        r"f_{W\mid\mathrm{seed}} ="
        r"\frac{\overline{s^{2}_W}}{\sigma^{2}_{\mathrm{total}}},"
        r"\qquad "
        r"f_{\mu} ="
        r"\frac{\sigma^{2}_{\mu}}{\sigma^{2}_{\mathrm{total}}},",
        r"$$",
        "",
        r"$$",
        r"f_{B\mid\mathrm{node}} ="
        r"\frac{\overline{s^{2}_B}}{\sigma^{2}_{\mathrm{total}}},"
        r"\qquad "
        r"f_{\nu} ="
        r"\frac{\sigma^{2}_{\nu}}{\sigma^{2}_{\mathrm{total}}}.",
        r"$$",
        "",
        "CSV columns: `frac_W_seed`, `frac_mu`, `frac_B_node`, `frac_nu`. "
        "Within each split the two fractions sum to 1.",
        "",
        "### Naive sum (diagnostic only)",
        "",
        "Averaging the two conditional variances and adding them does "
        "**not** recover $\\sigma^{2}_{\\mathrm{total}}$ in general:",
        "",
        r"$$",
        r"s^{2}_{T,\mathrm{naive}}"
        r"= \overline{s^{2}_W} + \overline{s^{2}_B},"
        r"\qquad "
        r"s_{T,\mathrm{naive}} = \sqrt{s^{2}_{T,\mathrm{naive}}}.",
        r"$$",
        "",
        r"$$",
        r"s_{T,\mathrm{gap}}"
        r"= s_{T,\mathrm{naive}} - s_{\mathrm{total}}.",
        r"$$",
        "",
        "CSV: `s_T_naive`, `s_T_gap`. Prefer the law-of-total-variance "
        "fractions above for interpretation.",
        "",
        "### Interaction residual",
        "",
        "Additive two-way fit residual",
        "",
        r"$$",
        r"R_{ij} = Y_{ij} - \mu_j - \nu_i + \bar{Y},",
        r"$$",
        "",
        r"with $\sigma^{2}_{R} = \mathrm{Var}_{ij}(R_{ij})$ "
        r"(CSV `s_interaction` $= \sqrt{\sigma^{2}_{R}}$, "
        r"`frac_interaction` $= \sigma^{2}_{R}/\sigma^{2}_{\mathrm{total}}$). "
        "Large $f_R$ flags seed×node structure beyond additive main effects.",
        "",
        "### Dominance ratios",
        "",
        r"$$",
        r"\frac{\overline{s^{2}_W}}{\sigma^{2}_{\mu}}"
        r"\quad(\texttt{ratio\_W\_over\_mu}),"
        r"\qquad "
        r"\frac{\overline{s^{2}_B}}{\sigma^{2}_{\nu}}"
        r"\quad(\texttt{ratio\_B\_over\_nu}).",
        r"$$",
        "",
        "## Per-metric medians across cells",
        "",
        "| Metric | median $G$ | median $s_{\\mathrm{total}}$ | "
        "median $f_\\mu$ | median $f_W$ | "
        "median $f_\\nu$ | median $f_B$ | decomp_ok |",
        "|--------|---------:|---------------:|--------------:|--------------:|"
        "--------------:|--------------:|----------:|",
    ]

    conclusions: list[str] = []
    for metric in METRICS:
        m = cell_df[cell_df["metric"] == metric]
        n = len(m)
        med = m.median(numeric_only=True)
        n_ok = int(m["decomp_ok"].sum()) if "decomp_ok" in m.columns else 0
        lines.append(
            "| {} | {:.4g} | {:.4g} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {} / {} |".format(
                metric,
                med.get("chi_geom", np.nan),
                med.get("s_total", np.nan),
                med.get("frac_mu", np.nan),
                med.get("frac_W_seed", np.nan),
                med.get("frac_nu", np.nan),
                med.get("frac_B_node", np.nan),
                n_ok,
                n,
            )
        )

        f_mu = float(med.get("frac_mu", np.nan))
        f_W = float(med.get("frac_W_seed", np.nan))
        f_nu = float(med.get("frac_nu", np.nan))
        f_B = float(med.get("frac_B_node", np.nan))
        G = float(med.get("chi_geom", np.nan))
        sT = float(med.get("s_total", np.nan))

        if np.isfinite(f_mu) and np.isfinite(f_W):
            if f_mu >= f_W:
                seed_dom = (
                    f"seed-to-seed shifts of the depth-averaged level "
                    f"($f_\\mu={f_mu:.2f} \\ge f_W={f_W:.2f}$)"
                )
            else:
                seed_dom = f"within-seed depth scatter ($f_W={f_W:.2f} > f_\\mu={f_mu:.2f}$)"
        else:
            seed_dom = "seed split fractions unavailable"

        if np.isfinite(f_nu) and np.isfinite(f_B):
            if f_B >= f_nu:
                node_dom = f"seed noise at fixed depth ($f_B={f_B:.2f} \\ge f_\\nu={f_nu:.2f}$)"
            else:
                node_dom = (
                    f"systematic node-to-node structure ($f_\\nu={f_nu:.2f} > f_B={f_B:.2f}$)"
                )
        else:
            node_dom = "node split fractions unavailable"

        conclusions.append(
            f"- **`{metric}`**: median geomean $G \\approx {G:.4g}$, "
            f"median $\\sigma_{{\\ln}} \\approx {sT:.4g}$. "
            f"Seed-conditioned variability is dominated by {seed_dom}. "
            f"Node-conditioned variability is dominated by {node_dom}."
        )

    lines.extend(["", "## Conclusions", ""])
    lines.extend(conclusions)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    print(f"Loading {DATA_PATH} …")
    df = load_ratios()
    print(f"Loaded {len(df):,} rows")

    cell_rows: list[dict] = []
    within_rows: list[dict] = []
    between_rows: list[dict] = []

    grouped = df.groupby(FACTORS, sort=True)
    n_groups = grouped.ngroups
    for i, (keys, df_cell) in enumerate(grouped):
        cell_keys = dict(zip(FACTORS, keys if isinstance(keys, tuple) else (keys,)))
        for metric in METRICS:
            summary, within, between = assess_cell_metric(df_cell, metric, cell_keys)
            cell_rows.append(summary)
            within_rows.extend(within)
            between_rows.extend(between)
        if (i + 1) % 25 == 0 or i == 0 or (i + 1) == n_groups:
            print(f"  assessed cell {i + 1}/{n_groups}")

    cell_df = pd.DataFrame.from_records(cell_rows)
    within_df = pd.DataFrame.from_records(within_rows)
    between_df = pd.DataFrame.from_records(between_rows)

    col_order = [
        *FACTORS,
        "metric",
        "n_nodes",
        "n_seeds",
        "n_finite",
        "chi_geom",
        "ln_chi_bar",
        "chi_mean",
        "chi_median",
        "s_W_bar",
        "s_mu",
        "s_B_bar",
        "s_nu",
        "s_total",
        "frac_W_seed",
        "frac_mu",
        "frac_B_node",
        "frac_nu",
        "s_T_naive",
        "s_T_gap",
        "s_interaction",
        "frac_interaction",
        "ratio_W_over_mu",
        "ratio_B_over_nu",
        "s_W_bar_ddof1",
        "s_mu_ddof1",
        "s_B_bar_ddof1",
        "s_nu_ddof1",
        "s_total_ddof1",
        "G_seed_p10",
        "G_seed_p90",
        "G_node_p10",
        "G_node_p90",
        "decomp_ok",
    ]
    cell_df = cell_df[col_order]

    out_dir = figure_dir("chi_variables", "central_variability")
    cell_path = out_dir / "cell_summary.csv"
    within_path = out_dir / "within_per_seed.csv"
    between_path = out_dir / "between_per_node.csv"
    md_path = out_dir / "summary.md"

    cell_df.to_csv(cell_path, index=False)
    within_df.to_csv(within_path, index=False)
    between_df.to_csv(between_path, index=False)

    summary = build_summary_md(cell_df)
    md_path.write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print(f"Wrote {cell_path} ({len(cell_df):,} rows)")
    print(f"Wrote {within_path} ({len(within_df):,} rows)")
    print(f"Wrote {between_path} ({len(between_df):,} rows)")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
