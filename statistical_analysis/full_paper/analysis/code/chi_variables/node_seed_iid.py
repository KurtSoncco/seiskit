"""Design-cell node/seed independence (IID) diagnostics for χ ratios.

Answers, per design cell × metric:

- Are nodes independent along the array?
- Are nodes identically distributed (hence iid)?
- Are seeds exchangeable by label, and do they share a common seed effect?

Loads ``join_master.h5`` 1D-normalized ratios. Working tests use ln(ratio);
raw-scale lag-1 is also reported for Box parity. Writes CSV + markdown only
(no plots) under ``figure_dir("chi_variables", "node_seed_iid")``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import BOX_ROOT, figure_dir

DATA_PATH = BOX_ROOT / "peak_analysis" / "join_master.h5"
METRICS = ("f_ratio", "abs_TF_ratio", "PGA_ratio", "PSA_ratio", "Ia_ratio")
FACTORS = ["Vs1", "Height", "CoV", "rH", "aHV"]
N_NODES = 101
N_SEEDS = 100
N_CELLS = 243
ALPHA = 0.05
LB_LAGS = 10
CHANNEL_SPACING_M = 2.0


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


def _lag1_pearson(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.nan
    a, b = x[:-1], x[1:]
    if a.std() < 1e-15 or b.std() < 1e-15:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _durbin_watson(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return np.nan
    d = np.diff(x)
    denom = float(np.dot(x, x))
    if denom <= 0:
        return np.nan
    return float(np.dot(d, d) / denom)


def _ljungbox_p(x: np.ndarray, lags: int = LB_LAGS) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < lags + 5:
        return np.nan
    if np.nanstd(x) < 1e-15:
        return np.nan
    try:
        res = acorr_ljungbox(x, lags=[lags], return_df=True)
        return float(res["lb_pvalue"].iloc[0])
    except Exception:
        return np.nan


def _series_stats(series_list: list[np.ndarray]) -> dict:
    """Aggregate lag-1 / DW / LB over many 1-D series."""
    lag1 = np.array([_lag1_pearson(s) for s in series_list], dtype=float)
    dw = np.array([_durbin_watson(s) for s in series_list], dtype=float)
    lb_p = np.array([_ljungbox_p(s) for s in series_list], dtype=float)
    finite_lag = lag1[np.isfinite(lag1)]
    finite_dw = dw[np.isfinite(dw)]
    finite_lb = lb_p[np.isfinite(lb_p)]
    return {
        "n_series": int(len(series_list)),
        "median_lag1_r": float(np.median(finite_lag)) if finite_lag.size else np.nan,
        "mean_lag1_r": float(np.mean(finite_lag)) if finite_lag.size else np.nan,
        "median_abs_lag1_r": float(np.median(np.abs(finite_lag))) if finite_lag.size else np.nan,
        "median_durbin_watson": float(np.median(finite_dw)) if finite_dw.size else np.nan,
        "median_ljungbox_p": float(np.median(finite_lb)) if finite_lb.size else np.nan,
        "frac_reject_ljungbox": (float(np.mean(finite_lb < ALPHA)) if finite_lb.size else np.nan),
    }


def _kruskal_nodes(Y: np.ndarray) -> tuple[float, float]:
    """Kruskal–Wallis across nodes; each node has seed replicates (rows of Y)."""
    samples = []
    for i in range(Y.shape[0]):
        s = Y[i, np.isfinite(Y[i])]
        if s.size >= 2:
            samples.append(s)
    if len(samples) < 3:
        return np.nan, np.nan
    try:
        stat, p = stats.kruskal(*samples)
        return float(stat), float(p)
    except Exception:
        return np.nan, np.nan


def _icc_seed(Y: np.ndarray) -> float:
    """One-way ICC for seed random effect on (node × seed) matrix Y.

    ICC = (MSB - MSW) / (MSB + (n - 1) MSW), n = nodes per seed.
    """
    # Drop columns/rows that are all-NaN
    ok_col = np.isfinite(Y).any(axis=0)
    ok_row = np.isfinite(Y).any(axis=1)
    Y = Y[np.ix_(ok_row, ok_col)]
    if Y.size == 0 or Y.shape[0] < 2 or Y.shape[1] < 2:
        return np.nan

    # Impute column-wise for incomplete? Prefer complete cases only per seed
    # Use nanmean / nanvar with equal weight approximation
    n_i = np.sum(np.isfinite(Y), axis=0)  # nodes observed per seed
    if np.any(n_i < 2):
        # keep seeds with ≥2 nodes
        keep = n_i >= 2
        Y = Y[:, keep]
        n_i = n_i[keep]
    if Y.shape[1] < 2:
        return np.nan

    mu_j = np.nanmean(Y, axis=0)
    grand = float(np.nanmean(Y))
    n_bar = float(np.mean(n_i))
    k = int(Y.shape[1])

    ssb = float(np.sum(n_i * (mu_j - grand) ** 2))
    ssw = float(np.nansum((Y - mu_j[np.newaxis, :]) ** 2))
    dfb = k - 1
    dfw = float(np.sum(n_i) - k)
    if dfb <= 0 or dfw <= 0:
        return np.nan
    msb = ssb / dfb
    msw = ssw / dfw
    denom = msb + (n_bar - 1.0) * msw
    if denom <= 0:
        return np.nan
    return float((msb - msw) / denom)


def _mean_pairwise_node_corr(Y: np.ndarray) -> float:
    """Mean off-diagonal correlation of node seed-vectors (rowvar=True)."""
    ok_row = np.sum(np.isfinite(Y), axis=1) >= 3
    Y = Y[ok_row]
    if Y.shape[0] < 2:
        return np.nan
    # Fill NaN with row mean so corrcoef works
    Yf = Y.copy()
    for i in range(Yf.shape[0]):
        row = Yf[i]
        m = np.nanmean(row)
        row[~np.isfinite(row)] = m if np.isfinite(m) else 0.0
    # Drop constant rows
    std = Yf.std(axis=1)
    Yf = Yf[std > 1e-15]
    if Yf.shape[0] < 2:
        return np.nan
    C = np.corrcoef(Yf)
    iu = np.triu_indices(C.shape[0], k=1)
    vals = C[iu]
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else np.nan


def assess_cell_metric(df_cell: pd.DataFrame, metric: str, cell_keys: dict) -> dict:
    """Compute node/seed IID diagnostics for one design cell × metric."""
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
        "channel_spacing_m": CHANNEL_SPACING_M,
        # Node serial dependence (ln), per seed along ordered nodes
        "node_median_lag1_r_ln": np.nan,
        "node_median_abs_lag1_r_ln": np.nan,
        "node_median_durbin_watson_ln": np.nan,
        "node_median_ljungbox_p_ln": np.nan,
        "node_frac_reject_ljungbox_ln": np.nan,
        # Raw-scale lag-1 (Box parity)
        "node_median_lag1_r_raw": np.nan,
        "node_frac_reject_ljungbox_raw": np.nan,
        # Identical distribution across nodes
        "node_kruskal_stat": np.nan,
        "node_kruskal_p": np.nan,
        "nodes_identical_dist": False,
        # Seed-index serial dependence (ln), per node along seed id
        "seed_median_lag1_r_ln": np.nan,
        "seed_median_abs_lag1_r_ln": np.nan,
        "seed_median_ljungbox_p_ln": np.nan,
        "seed_frac_reject_ljungbox_ln": np.nan,
        # Shared seed effect
        "seed_icc": np.nan,
        "mean_pairwise_node_corr": np.nan,
        # Verdicts
        "nodes_independent": False,
        "nodes_iid": False,
        "seeds_label_exchangeable": False,
        "seeds_no_shared_effect": False,
    }

    n_finite = int(np.isfinite(Y).sum())
    if n_finite < 20:
        return out

    # --- Node dependence along array (columns = seeds) ---
    node_series_ln = [Y[:, j] for j in range(Y.shape[1]) if np.isfinite(Y[:, j]).sum() >= 8]
    node_agg_ln = _series_stats(node_series_ln)
    out["node_median_lag1_r_ln"] = node_agg_ln["median_lag1_r"]
    out["node_median_abs_lag1_r_ln"] = node_agg_ln["median_abs_lag1_r"]
    out["node_median_durbin_watson_ln"] = node_agg_ln["median_durbin_watson"]
    out["node_median_ljungbox_p_ln"] = node_agg_ln["median_ljungbox_p"]
    out["node_frac_reject_ljungbox_ln"] = node_agg_ln["frac_reject_ljungbox"]

    node_series_raw = [arr[:, j] for j in range(arr.shape[1]) if np.isfinite(arr[:, j]).sum() >= 8]
    node_agg_raw = _series_stats(node_series_raw)
    out["node_median_lag1_r_raw"] = node_agg_raw["median_lag1_r"]
    out["node_frac_reject_ljungbox_raw"] = node_agg_raw["frac_reject_ljungbox"]

    # Independence: require low lag-1 and low LB rejection rate
    out["nodes_independent"] = bool(
        np.isfinite(out["node_median_abs_lag1_r_ln"])
        and out["node_median_abs_lag1_r_ln"] < 0.2
        and np.isfinite(out["node_frac_reject_ljungbox_ln"])
        and out["node_frac_reject_ljungbox_ln"] < 0.2
    )

    # Identical distribution
    kw_stat, kw_p = _kruskal_nodes(Y)
    out["node_kruskal_stat"] = kw_stat
    out["node_kruskal_p"] = kw_p
    out["nodes_identical_dist"] = bool(np.isfinite(kw_p) and kw_p > ALPHA)
    out["nodes_iid"] = bool(out["nodes_independent"] and out["nodes_identical_dist"])

    # --- Seed-index dependence (rows = nodes) ---
    seed_series = [Y[i, :] for i in range(Y.shape[0]) if np.isfinite(Y[i, :]).sum() >= 8]
    seed_agg = _series_stats(seed_series)
    out["seed_median_lag1_r_ln"] = seed_agg["median_lag1_r"]
    out["seed_median_abs_lag1_r_ln"] = seed_agg["median_abs_lag1_r"]
    out["seed_median_ljungbox_p_ln"] = seed_agg["median_ljungbox_p"]
    out["seed_frac_reject_ljungbox_ln"] = seed_agg["frac_reject_ljungbox"]

    # Label exchangeable if seed-index serial dependence is weak
    out["seeds_label_exchangeable"] = bool(
        np.isfinite(out["seed_median_abs_lag1_r_ln"])
        and out["seed_median_abs_lag1_r_ln"] < 0.2
        and np.isfinite(out["seed_frac_reject_ljungbox_ln"])
        and out["seed_frac_reject_ljungbox_ln"] < 0.2
    )

    out["seed_icc"] = _icc_seed(Y)
    out["mean_pairwise_node_corr"] = _mean_pairwise_node_corr(Y)
    out["seeds_no_shared_effect"] = bool(
        np.isfinite(out["seed_icc"]) and abs(out["seed_icc"]) < 0.05
    )

    return out


def build_summary_md(by_cell: pd.DataFrame) -> str:
    lines = [
        "# Node / seed independence (IID) summary",
        "",
        "Per design cell (243) × metric diagnostics on 1D-normalized ratios "
        f"(`{'`, `'.join(METRICS)}`) from `join_master.h5`.",
        "",
        f"- Node spacing: **{CHANNEL_SPACING_M} m**; Ljung–Box lag = {LB_LAGS}; α = {ALPHA}",
        "- Primary scale: **ln(ratio)**; raw lag-1 reported for Box parity",
        "- Independence gate (nodes / seed labels): median abs lag-1 < 0.2 and "
        "Ljung–Box reject fraction < 0.2",
        "",
        "## Aggregate across design cells",
        "",
        "### Node dependence and identical distribution",
        "",
        "| Metric | med node lag1 (ln) | frac cells with node LB reject ≥ 50% | "
        "frac cells KW reject | frac cells nodes independent | frac cells nodes iid |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for metric in METRICS:
        m = by_cell[by_cell["metric"] == metric]
        lines.append(
            "| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                metric,
                float(m["node_median_lag1_r_ln"].median()),
                float((m["node_frac_reject_ljungbox_ln"] >= 0.5).mean()),
                float((m["node_kruskal_p"] <= ALPHA).mean()),
                float(m["nodes_independent"].mean()),
                float(m["nodes_iid"].mean()),
            )
        )

    lines.extend(
        [
            "",
            "### Seed diagnostics (fixed-node series vs shared seed effect)",
            "",
            "| Metric | med seed abs lag1 (ln) | med seed LB reject frac | "
            "frac cells seed labels exchangeable | med seed ICC | "
            "med pairwise node corr | frac cells abs ICC < 0.05 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for metric in METRICS:
        m = by_cell[by_cell["metric"] == metric]
        lines.append(
            "| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                metric,
                float(m["seed_median_abs_lag1_r_ln"].median()),
                float(m["seed_frac_reject_ljungbox_ln"].median()),
                float(m["seeds_label_exchangeable"].mean()),
                float(m["seed_icc"].median()),
                float(m["mean_pairwise_node_corr"].median()),
                float(m["seeds_no_shared_effect"].mean()),
            )
        )

    # Overall narrative stats
    all_med_lag = float(by_cell["node_median_lag1_r_ln"].median())
    all_frac_node_indep = float(by_cell["nodes_independent"].mean())
    all_frac_node_iid = float(by_cell["nodes_iid"].mean())
    all_frac_seed_exch = float(by_cell["seeds_label_exchangeable"].mean())
    all_frac_no_shared = float(by_cell["seeds_no_shared_effect"].mean())
    all_med_icc = float(by_cell["seed_icc"].median())
    all_med_seed_abs_lag = float(by_cell["seed_median_abs_lag1_r_ln"].median())
    all_med_seed_lb_frac = float(by_cell["seed_frac_reject_ljungbox_ln"].median())
    all_med_pair = float(by_cell["mean_pairwise_node_corr"].median())
    frac_kw_reject = float((by_cell["node_kruskal_p"] <= ALPHA).mean())

    lines.extend(["", "## Conclusions", ""])

    lines.append(
        f"1. **Are nodes independent of each other?** **No.** "
        f"Median node lag-1 correlation (ln) across cells/metrics is **{all_med_lag:.3f}**; "
        f"only {all_frac_node_indep:.1%} of cell×metric cases meet the independence gate. "
        f"Strong spatial serial dependence along the 2 m node array matches prior Box "
        f"`channel_iid` findings."
    )
    lines.append("")
    lines.append(
        f"2. **Within each design cell, are nodes iid?** **No.** "
        f"IID requires independence *and* identical distributions. "
        f"Only {all_frac_node_iid:.1%} of cell×metric cases pass both gates. "
        f"Lag-1 dependence alone rules out iid even when marginals look similar "
        f"(Kruskal–Wallis rejects equal node distributions in {frac_kw_reject:.1%} of "
        f"cell×metric cases overall; rejection is metric-dependent)."
    )
    lines.append("")
    lines.append(
        "3. **Within each design cell, are seeds iid?** "
        "**Yes at a fixed node; no for the joint (node × seed) sample.**"
    )
    lines.append("")
    lines.append("#### Clarification: fixed node vs joint array")
    lines.append("")
    lines.append("| Scope | Treat as iid? | Evidence |")
    lines.append("| --- | --- | --- |")
    lines.append(
        f"| Seeds at **one fixed node** in one cell | **Yes** | "
        f"seed-index median abs lag-1 = **{all_med_seed_abs_lag:.3f}**; "
        f"median LB reject fraction = **{all_med_seed_lb_frac:.3f}**; "
        f"**{all_frac_seed_exch:.1%}** of cell×metric cases pass the exchangeability gate |"
    )
    lines.append(
        f"| Nodes along the array (same seed) | **No** | "
        f"median node lag-1 = **{all_med_lag:.3f}** |"
    )
    lines.append(
        f"| Joint (node × seed) as if all draws were iid | **No** | "
        f"median seed ICC = **{all_med_icc:.3f}**; "
        f"median pairwise node corr = **{all_med_pair:.3f}**; "
        f"only {all_frac_no_shared:.1%} of cases have abs(ICC) < 0.05 |"
    )
    lines.append("")
    lines.append(
        "The large seed ICC means: *given the same seed*, nodes move together "
        "(shared random-field realization). It does **not** mean seed 3 and seed 17 "
        "at a fixed node are dependent — those are separate Monte Carlo draws. "
        f"Overall, seed labels look exchangeable in **{all_frac_seed_exch:.1%}** of "
        f"cell×metric cases, with overall median abs seed lag-1 "
        f"**{all_med_seed_abs_lag:.3f}**."
    )
    lines.append("")
    lines.append("### Per-metric notes")
    lines.append("")
    for metric in METRICS:
        m = by_cell[by_cell["metric"] == metric]
        lines.append(
            f"- **`{metric}`**: med node lag1={m['node_median_lag1_r_ln'].median():.3f}; "
            f"med seed abs lag1={m['seed_median_abs_lag1_r_ln'].median():.3f}; "
            f"med seed ICC={m['seed_icc'].median():.3f}; "
            f"nodes_iid in {100 * m['nodes_iid'].mean():.1f}% of cells; "
            f"seed labels exchangeable in {100 * m['seeds_label_exchangeable'].mean():.1f}% of cells."
        )

    lines.extend(
        [
            "",
            "### Modeling implication",
            "",
            "- **Fixed node** (e.g. center recorder): seed averages and SEs under an "
            "iid-seed model are appropriate.",
            "- **Across nodes** within a cell: do **not** assume nodes are iid. Prefer "
            "spatial models (semivariogram / correlation length).",
            "- **Pooling node × seed**: do **not** treat all cells of the matrix as iid; "
            "use seed clustering or mixed effects when the same seed couples nodes.",
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
        "channel_spacing_m",
        "node_median_lag1_r_ln",
        "node_median_abs_lag1_r_ln",
        "node_median_durbin_watson_ln",
        "node_median_ljungbox_p_ln",
        "node_frac_reject_ljungbox_ln",
        "node_median_lag1_r_raw",
        "node_frac_reject_ljungbox_raw",
        "node_kruskal_stat",
        "node_kruskal_p",
        "nodes_identical_dist",
        "seed_median_lag1_r_ln",
        "seed_median_abs_lag1_r_ln",
        "seed_median_ljungbox_p_ln",
        "seed_frac_reject_ljungbox_ln",
        "seed_icc",
        "mean_pairwise_node_corr",
        "nodes_independent",
        "nodes_iid",
        "seeds_label_exchangeable",
        "seeds_no_shared_effect",
    ]
    by_cell = by_cell[col_order]

    out_dir = figure_dir("chi_variables", "node_seed_iid")
    csv_path = out_dir / "iid_by_cell.csv"
    md_path = out_dir / "iid_summary.md"
    by_cell.to_csv(csv_path, index=False)
    summary = build_summary_md(by_cell)
    md_path.write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
