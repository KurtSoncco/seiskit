"""Node robustness of the between-seed Y NGBoost (center node vs other nodes).

Refits the single-node, all-seeds Normal NGBoost on \\(Y=\\ln\\chi\\) at nodes
10, 30, 50, 70 and 90 with the same features, learner, seed holdout and
early stopping as ``train_ngboost.py --partition between``, then compares
holdout \\(R^2\\), efficiency, PI90 and the permutation-SHAP ranks of
\\(\\mu\\). Writes under ``figure_dir("chi_ngboost", "node_robustness")``.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CENTER_NODE,
    METRICS,
    N_SEEDS,
    NGB_FEATURES,
    VAL_FRAC,
    VAL_SEED,
    add_design_columns,
    load_or_make_split,
    load_ratios,
    log_response,
    out_dir,
    r2_score,
)
from train_ngboost import _finite_mask, fit_one, pi_coverage, predict_params  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import (  # noqa: E402
    add_panel_label,
    apply_full_paper_style,
    factor_color,
    figsize,
    metric_color,
    metric_label,
    save_figure,
)

warnings.filterwarnings("ignore")
apply_full_paper_style(auto_format=True, frame="open", grid=False)

NODES = (10, 30, CENTER_NODE, 70, 90)
EXPLAIN_N = 400
BG_N = 100
SHAP_SAMPLE_SEED = 3
FEATURE_DISPLAY = {
    "Vs1_z": r"$V_{s1}$",
    "Height_z": r"$H$",
    "CoV_z": "CoV",
    "rH_z": r"$r_h$",
    "aHV_z": r"$a_{hv}$",
}


def seed_ceiling(df: pd.DataFrame, y: np.ndarray) -> float:
    """Signal-to-total ceiling of Y at one node with the seeds as noise."""
    work = pd.DataFrame({"cell": df["cell"].to_numpy(), "y": y})
    g = work[np.isfinite(work["y"])].groupby("cell")["y"]
    signal = float(g.mean().var(ddof=1))
    noise = float(g.var(ddof=1).mean())
    return signal / (signal + noise)


def mu_shap(model, X_bg: np.ndarray, X_ex: np.ndarray) -> np.ndarray:
    def f(X):
        return predict_params(model, np.asarray(X, dtype=float))[0]

    explainer = shap.Explainer(f, X_bg, algorithm="permutation")
    return np.asarray(explainer(X_ex, max_evals=2 * X_ex.shape[1] + 1).values, dtype=float)


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Loading join_master …")
    full = add_design_columns(load_ratios(), include_node_z=False)
    rows, shap_rows = [], []
    for node in NODES:
        df = full.loc[full["node"] == node].reset_index(drop=True)
        tr, te = load_or_make_split(df)
        groups = df["seed"].to_numpy()
        gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=VAL_SEED)
        fit_rel, val_rel = next(gss.split(tr, groups=groups[tr]))
        fit_idx, val_idx = tr[fit_rel], tr[val_rel]
        X = df[NGB_FEATURES].to_numpy(dtype=float)
        rng = np.random.default_rng(SHAP_SAMPLE_SEED)
        pick = rng.choice(te, size=BG_N + EXPLAIN_N, replace=False)
        X_bg, X_ex = X[pick[:BG_N]], X[pick[BG_N:]]
        for metric in METRICS:
            y = log_response(df, metric)
            m_fit = _finite_mask(y[fit_idx], X[fit_idx])
            m_val = _finite_mask(y[val_idx], X[val_idx])
            m_te = _finite_mask(y[te], X[te])
            t0 = time.perf_counter()
            model, n_trees = fit_one(
                X[fit_idx][m_fit], y[fit_idx][m_fit], X[val_idx][m_val], y[val_idx][m_val]
            )
            mu, sig = predict_params(model, X[te][m_te])
            y_te = y[te][m_te]
            r2 = r2_score(y_te, mu)
            ceiling = seed_ceiling(df, y)
            sv = mu_shap(model, X_bg, X_ex)
            mean_abs = np.mean(np.abs(sv), axis=0)
            rank = stats.rankdata(-mean_abs, method="min").astype(int)
            for j, feat in enumerate(NGB_FEATURES):
                shap_rows.append(
                    {
                        "node": node,
                        "metric": metric,
                        "feature": feat,
                        "mean_abs_shap": float(mean_abs[j]),
                        "rank": int(rank[j]),
                    }
                )
            rows.append(
                {
                    "node": node,
                    "metric": metric,
                    "n_trees": n_trees,
                    "n_test": int(m_te.sum()),
                    "r2_mean": r2,
                    "ceiling": ceiling,
                    "efficiency": r2 / ceiling,
                    "pi90_coverage": pi_coverage(y_te, mu, sig, alpha=0.10),
                    "median_sigma": float(np.median(sig)),
                }
            )
            print(
                f"  node {node:3d} {metric:13s} R2={r2:.3f} eff={r2 / ceiling:.3f} "
                f"PI90={rows[-1]['pi90_coverage']:.3f} trees={n_trees} "
                f"{time.perf_counter() - t0:.1f}s"
            )
    res = pd.DataFrame(rows)
    ranks = pd.DataFrame(shap_rows)

    ref = ranks[ranks["node"] == CENTER_NODE].set_index(["metric", "feature"])["mean_abs_shap"]
    rho = []
    for (node, metric), g in ranks.groupby(["node", "metric"], sort=False):
        a = g.set_index("feature")["mean_abs_shap"]
        b = ref.loc[metric].loc[a.index]
        rho.append({"node": node, "metric": metric, "spearman_vs_center": stats.spearmanr(a, b)[0]})
    res = res.merge(pd.DataFrame(rho), on=["node", "metric"])
    return res, ranks


def plot(res: pd.DataFrame, ranks: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=figsize(height=2.6))
    offsets = np.linspace(-2.4, 2.4, len(METRICS))
    for k, metric in enumerate(METRICS):
        r = res[res["metric"] == metric]
        x = r["node"].to_numpy() + offsets[k]
        kw = dict(color=metric_color(metric), marker="o", ms=3.5, ls="none")
        axes[0].plot(x, r["efficiency"], label=metric_label(metric), **kw)
        axes[1].plot(x, r["pi90_coverage"], **kw)
    axes[0].set_ylabel(r"Efficiency $R^2/\mathrm{ceiling}$")
    axes[1].axhline(0.90, color="0.4", lw=0.6, ls="--")
    axes[1].set_ylabel("PI90 coverage")
    offsets = np.linspace(-2.4, 2.4, len(NGB_FEATURES))
    mean_rank = ranks.groupby(["node", "feature"])["rank"].mean().reset_index()
    for k, feat in enumerate(NGB_FEATURES):
        r = mean_rank[mean_rank["feature"] == feat]
        axes[2].plot(
            r["node"].to_numpy() + offsets[k],
            r["rank"],
            color=factor_color(feat),
            marker="s",
            ms=3.5,
            ls="none",
            label=FEATURE_DISPLAY[feat],
        )
    axes[2].set_ylabel(r"Mean SHAP rank of $\mu$ (1 = top)")
    axes[2].invert_yaxis()
    for i, ax in enumerate(axes):
        ax.set_xticks(NODES)
        ax.set_xlabel("Node")
        ax.axvline(CENTER_NODE, color="0.8", lw=0.6, zorder=0)
        add_panel_label(ax, i)
    for ax in (axes[0], axes[2]):
        ax.legend(
            fontsize=6,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=3,
            handletextpad=0.2,
            columnspacing=0.8,
        )
    fig.tight_layout(pad=0.4)
    save_figure(fig, "node_robustness", out_dir=out)
    plt.close(fig)


def main() -> None:
    out = out_dir("node_robustness")
    if "--plot-only" in sys.argv:
        res = pd.read_csv(out / "node_robustness.csv")
        ranks = pd.read_csv(out / "node_robustness_shap_ranks.csv")
    else:
        res, ranks = run()
        res.to_csv(out / "node_robustness.csv", index=False)
        ranks.to_csv(out / "node_robustness_shap_ranks.csv", index=False)
    plot(res, ranks, out)

    spread = (
        res.groupby("metric", sort=False)
        .agg(
            r2_min=("r2_mean", "min"),
            r2_max=("r2_mean", "max"),
            eff_min=("efficiency", "min"),
            eff_max=("efficiency", "max"),
            pi90_min=("pi90_coverage", "min"),
            pi90_max=("pi90_coverage", "max"),
            spearman_min=("spearman_vs_center", "min"),
        )
        .reset_index()
    )
    rank_wide = ranks.pivot_table(
        index=["metric", "feature"], columns="node", values="rank"
    ).reset_index()
    lines = [
        "# Node robustness of the between-seed Y NGBoost",
        "",
        "## Definitions",
        "",
        rf"- At each node in {list(NODES)}: all \(N_s={N_SEEDS}\) seeds × 243 cells, "
        r"Normal NGBoost on \(Y=\ln\chi\) with the five z-scored factors.",
        "- Holdout: the same 25 held-out seeds as the center-node model; inner 20% seed "
        "validation for early stopping.",
        r"- `ceiling`: signal-to-total ceiling of \(Y\) at that node with the seeds as noise; "
        r"`efficiency` = holdout \(R^2(\hat\mu)\) / ceiling.",
        rf"- SHAP: permutation SHAP of \(\hat\mu\) ({EXPLAIN_N} explained / {BG_N} background "
        "held-out rows); `spearman_vs_center` = rank correlation of mean |SHAP| with node "
        f"{CENTER_NODE}.",
        "",
        "## Range across nodes",
        "",
        spread.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## All nodes",
        "",
        res.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## SHAP rank of μ by node",
        "",
        rank_wide.to_markdown(index=False, floatfmt=".0f"),
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
