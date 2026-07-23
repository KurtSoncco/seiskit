"""Per-node normality / lognormality assessment for 1D-normalized IM ratios.

Loads ``join_master.h5`` peak and IM ratios, renames ``channel`` → ``node``,
and for each of 101 nodes tests raw, ln, and Box–Cox transforms (pooled and
within-cell residuals). Writes CSV + markdown conclusions (no plots).
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import boxcox, boxcox_normmax

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import BOX_ROOT, figure_dir

DATA_PATH = BOX_ROOT / "peak_analysis" / "join_master.h5"
METRICS = ("f_ratio", "abs_TF_ratio", "PGA_ratio", "PSA_ratio", "Ia_ratio")
FACTORS = ["Vs1", "Height", "CoV", "rH", "aHV"]
N_NODES = 101
SHAPIRO_CAP = 5000
ALPHA = 0.05
SKEW_SOFT = 0.5
RNG = np.random.default_rng(42)


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


def _subsample(x: np.ndarray, n_max: int = SHAPIRO_CAP) -> np.ndarray:
    if len(x) <= n_max:
        return x
    idx = RNG.choice(len(x), size=n_max, replace=False)
    return x[idx]


def _passes(shapiro_p: float | None, skew: float | None) -> bool:
    if shapiro_p is None or skew is None or not np.isfinite(shapiro_p):
        return False
    return shapiro_p > ALPHA and abs(skew) < SKEW_SOFT


def assess_series(x: np.ndarray, *, transform: str, boxcox_lambda: float | None = None) -> dict:
    """Skew, kurtosis, Shapiro/KS/Anderson for one transformed series."""
    x = _finite(x)
    n = int(x.size)
    out = {
        "n": n,
        "skew": np.nan,
        "kurtosis": np.nan,
        "boxcox_lambda": np.nan if boxcox_lambda is None else float(boxcox_lambda),
        "shapiro_stat": np.nan,
        "shapiro_p": np.nan,
        "ks_stat": np.nan,
        "ks_p": np.nan,
        "anderson_stat": np.nan,
        "passes": False,
    }
    if n < 8:
        return out

    out["skew"] = float(stats.skew(x))
    out["kurtosis"] = float(stats.kurtosis(x))  # excess kurtosis

    xs = _subsample(x)
    if xs.std() < 1e-15:
        return out

    sh_stat, sh_p = stats.shapiro(xs)
    out["shapiro_stat"] = float(sh_stat)
    out["shapiro_p"] = float(sh_p)

    z = (xs - xs.mean()) / xs.std(ddof=1)
    ks_stat, ks_p = stats.kstest(z, "norm")
    out["ks_stat"] = float(ks_stat)
    out["ks_p"] = float(ks_p)

    out["anderson_stat"] = float(stats.anderson(xs, dist="norm").statistic)
    out["passes"] = _passes(out["shapiro_p"], out["skew"])
    return out


def _verdict(transform: str, passes: bool) -> str:
    if not passes:
        return "neither"
    if transform == "raw":
        return "normal"
    if transform == "ln":
        return "lognormal"
    if transform == "boxcox":
        return "boxcox"
    return "neither"


def transform_values(raw: np.ndarray) -> list[tuple[str, np.ndarray, float | None]]:
    """Return (name, values, lambda_or_None) for raw / ln / Box–Cox."""
    raw = _finite(raw)
    results: list[tuple[str, np.ndarray, float | None]] = [("raw", raw, None)]

    pos = raw[raw > 0]
    if pos.size >= 8:
        results.append(("ln", np.log(pos), None))
        try:
            lam = float(boxcox_normmax(pos, method="mle"))
            results.append(("boxcox", boxcox(pos, lmbda=lam), lam))
        except Exception:
            results.append(("boxcox", np.array([]), np.nan))
    else:
        results.append(("ln", np.array([]), None))
        results.append(("boxcox", np.array([]), np.nan))
    return results


def within_cell_residual(values: np.ndarray, factor_frame: pd.DataFrame) -> np.ndarray:
    """Value minus design-cell mean (rows aligned with ``factor_frame``)."""
    s = pd.Series(np.asarray(values, dtype=float), index=factor_frame.index)
    cell_mean = s.groupby([factor_frame[c] for c in FACTORS], sort=False).transform("mean")
    return (s - cell_mean).to_numpy()


def _row(node: int, metric: str, residual: bool, transform: str, stats_d: dict) -> dict:
    return {
        "node": node,
        "metric": metric,
        "residual": residual,
        "transform": transform,
        **{k: stats_d[k] for k in stats_d if k != "passes"},
        "verdict": _verdict(transform, stats_d["passes"]),
    }


def assess_node_metric(df_node: pd.DataFrame, metric: str) -> list[dict]:
    """Pooled + residual assessments for one node × metric."""
    node = int(df_node["node"].iloc[0])
    rows: list[dict] = []
    raw = df_node[metric].to_numpy()
    factors = df_node[FACTORS]

    for name, vals, lam in transform_values(raw):
        rows.append(
            _row(node, metric, False, name, assess_series(vals, transform=name, boxcox_lambda=lam))
        )

    rows.append(
        _row(
            node,
            metric,
            True,
            "raw",
            assess_series(within_cell_residual(raw, factors), transform="raw"),
        )
    )

    pos_mask = raw > 0
    if int(pos_mask.sum()) >= 8:
        factors_pos = factors.loc[pos_mask].reset_index(drop=True)
        pos = raw[pos_mask]

        resid_ln = within_cell_residual(np.log(pos), factors_pos)
        rows.append(_row(node, metric, True, "ln", assess_series(resid_ln, transform="ln")))

        try:
            lam = float(boxcox_normmax(pos, method="mle"))
            resid_bc = within_cell_residual(boxcox(pos, lmbda=lam), factors_pos)
            stats_d = assess_series(resid_bc, transform="boxcox", boxcox_lambda=lam)
        except Exception:
            stats_d = assess_series(np.array([]), transform="boxcox", boxcox_lambda=np.nan)
        rows.append(_row(node, metric, True, "boxcox", stats_d))
    else:
        empty = assess_series(np.array([]), transform="ln")
        rows.append(_row(node, metric, True, "ln", empty))
        rows.append(
            _row(node, metric, True, "boxcox", assess_series(np.array([]), transform="boxcox"))
        )

    return rows


def _best_by_abs_skew(
    block: pd.DataFrame, transforms: tuple[str, ...] = ("raw", "ln", "boxcox")
) -> str:
    """Transform with smallest |skew| for one node×metric slice (one residual flag)."""
    cands = []
    for t in transforms:
        row = block[block["transform"] == t]
        if row.empty or not np.isfinite(row["skew"].iloc[0]):
            continue
        cands.append((abs(float(row["skew"].iloc[0])), t))
    if not cands:
        return "neither"
    return min(cands)[1]


def _median_abs_skew(df: pd.DataFrame, metric: str, residual: bool, transform: str) -> float:
    s = df[
        (df["metric"] == metric) & (df["residual"] == residual) & (df["transform"] == transform)
    ]["skew"]
    return float(s.abs().median()) if len(s) else float("nan")


def _count_best(df: pd.DataFrame, metric: str, residual: bool) -> dict[str, int]:
    counts = {"raw": 0, "ln": 0, "boxcox": 0, "neither": 0}
    sub = df[(df["metric"] == metric) & (df["residual"] == residual)]
    for node, g in sub.groupby("node"):
        counts[_best_by_abs_skew(g)] += 1
    return counts


def _count_ln_vs_raw(df: pd.DataFrame, metric: str, residual: bool) -> tuple[int, int]:
    """Return (n_nodes ln wins, n_nodes raw wins) by |skew|."""
    sub = df[(df["metric"] == metric) & (df["residual"] == residual)]
    ln_wins = raw_wins = 0
    for _, g in sub.groupby("node"):
        best = _best_by_abs_skew(g, transforms=("raw", "ln"))
        if best == "ln":
            ln_wins += 1
        elif best == "raw":
            raw_wins += 1
    return ln_wins, raw_wins


def build_summary_md(results: pd.DataFrame) -> str:
    pooled = results[not results["residual"]]
    results[results["residual"]]

    lines = [
        "# Normality check summary",
        "",
        "Per-node assessment of 1D-normalized ratios "
        f"(`{'`, `'.join(METRICS)}`) from `join_master.h5`.",
        "",
        f"- Nodes: **{N_NODES}** (0–100)",
        f"- Formal gate (CSV `verdict`): Shapiro–Wilk p > {ALPHA} and |skew| < {SKEW_SOFT} "
        f"(Shapiro on ≤ {SHAPIRO_CAP} samples). With ~24k points per node this gate almost "
        "always fails; **relative |skew| / Anderson rankings** are used for conclusions.",
        "- Transforms: raw (= normal candidate), ln (= lognormal), Box–Cox",
        "- Blocks: pooled values and within-design-cell residuals",
        "",
        "## Median |skew| by transform (pooled)",
        "",
        "| Metric | raw | ln | boxcox | median Box–Cox λ |",
        "|--------|----:|---:|-------:|-----------------:|",
    ]
    for metric in METRICS:
        lam = pooled[(pooled["metric"] == metric) & (pooled["transform"] == "boxcox")][
            "boxcox_lambda"
        ].median()
        lines.append(
            "| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |".format(
                metric,
                _median_abs_skew(results, metric, False, "raw"),
                _median_abs_skew(results, metric, False, "ln"),
                _median_abs_skew(results, metric, False, "boxcox"),
                float(lam) if np.isfinite(lam) else float("nan"),
            )
        )

    lines.extend(
        [
            "",
            "## Median |skew| by transform (within-cell residuals)",
            "",
            "| Metric | raw | ln | boxcox |",
            "|--------|----:|---:|-------:|",
        ]
    )
    for metric in METRICS:
        lines.append(
            "| {} | {:.3f} | {:.3f} | {:.3f} |".format(
                metric,
                _median_abs_skew(results, metric, True, "raw"),
                _median_abs_skew(results, metric, True, "ln"),
                _median_abs_skew(results, metric, True, "boxcox"),
            )
        )

    lines.extend(
        [
            "",
            "## Normal vs lognormal head-to-head (lower |skew| wins)",
            "",
            "Ignores Box–Cox; counts nodes where ln beats raw or vice versa.",
            "",
            "| Metric | pooled: ln wins | pooled: raw wins | residual: ln wins | residual: raw wins |",
            "|--------|----------------:|-----------------:|------------------:|-------------------:|",
        ]
    )
    for metric in METRICS:
        pl, pr = _count_ln_vs_raw(results, metric, False)
        rl, rr = _count_ln_vs_raw(results, metric, True)
        lines.append(f"| {metric} | {pl} | {pr} | {rl} | {rr} |")

    lines.extend(
        [
            "",
            "## Best of raw / ln / Box–Cox by |skew| (node counts)",
            "",
            "### Pooled",
            "",
            "| Metric | raw | ln | boxcox |",
            "|--------|----:|---:|-------:|",
        ]
    )
    for metric in METRICS:
        c = _count_best(results, metric, False)
        lines.append(f"| {metric} | {c['raw']} | {c['ln']} | {c['boxcox']} |")

    lines.extend(
        [
            "",
            "### Within-cell residuals",
            "",
            "| Metric | raw | ln | boxcox |",
            "|--------|----:|---:|-------:|",
        ]
    )
    for metric in METRICS:
        c = _count_best(results, metric, True)
        lines.append(f"| {metric} | {c['raw']} | {c['ln']} | {c['boxcox']} |")

    lines.extend(["", "## Conclusions", ""])
    for metric in METRICS:
        pl, pr = _count_ln_vs_raw(results, metric, False)
        rl, rr = _count_ln_vs_raw(results, metric, True)
        best_pooled = _count_best(results, metric, False)
        best_resid = _count_best(results, metric, True)
        fam = "lognormal" if pl >= pr else "normal"
        if pl == pr:
            fam = "mixed"

        med_raw = _median_abs_skew(results, metric, False, "raw")
        med_ln = _median_abs_skew(results, metric, False, "ln")
        med_bc = _median_abs_skew(results, metric, False, "boxcox")

        lines.append(
            f"- **`{metric}`**: normal-vs-lognormal → **{fam}** "
            f"(pooled ln wins {pl}/{N_NODES}, residual ln wins {rl}/{N_NODES}). "
            f"Median |skew| pooled: raw={med_raw:.3f}, ln={med_ln:.3f}, boxcox={med_bc:.3f}. "
            f"Including Box–Cox, best transform is usually "
            f"{max(best_pooled, key=best_pooled.get)} (pooled) / "
            f"{max(best_resid, key=best_resid.get)} (residuals)."
        )

    lines.extend(
        [
            "",
            "### Overall",
            "",
            "- Formal Shapiro/KS tests reject normality for essentially all node×metric×transform "
            "combinations at this sample size; use them only as continuous scores (see CSV).",
            "- For χ / ratio modeling: prefer **ln** when it clearly reduces |skew| vs raw "
            "(`f_ratio`, `abs_TF_ratio`, `Ia_ratio` in the pooled table); prefer **raw** when "
            "ln increases skew (`PGA_ratio`, often `PSA_ratio`).",
            "- Box–Cox further reduces skew for most metrics but λ is not 0 (log) or 1 (identity), "
            "so it is a diagnostic power transform rather than a simple normal/lognormal label.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    print(f"Loading {DATA_PATH} …")
    df = load_ratios()
    print(f"Loaded {len(df):,} rows; nodes {df['node'].min()}–{df['node'].max()}")

    records: list[dict] = []
    grouped = df.groupby("node", sort=True)
    for i, (node, df_node) in enumerate(grouped):
        for metric in METRICS:
            records.extend(assess_node_metric(df_node, metric))
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  assessed node {int(node)} ({i + 1}/{N_NODES})")

    results = pd.DataFrame.from_records(records)
    col_order = [
        "node",
        "metric",
        "residual",
        "transform",
        "n",
        "skew",
        "kurtosis",
        "boxcox_lambda",
        "shapiro_stat",
        "shapiro_p",
        "ks_stat",
        "ks_p",
        "anderson_stat",
        "verdict",
    ]
    results = results[col_order]

    out_dir = figure_dir("chi_variables")
    csv_path = out_dir / "normality_results.csv"
    md_path = out_dir / "normality_summary.md"
    results.to_csv(csv_path, index=False)
    summary = build_summary_md(results)
    md_path.write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
