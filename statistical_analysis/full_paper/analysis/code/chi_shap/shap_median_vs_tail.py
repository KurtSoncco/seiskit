"""Median vs tail SHAP deltas for NGBoost, per partition (between-seed, within-seed).

Contrasts are τ=0.05−τ=0.50 and τ=0.95−τ=0.50, each as a 2×5 figure (abs /
signed). NGBoost explains the Normal quantile \\(q_\tau=\\mu+z_\tau\\sigma\\)
with permutation SHAP composed from μ and σ attributions. Pass ``--force`` to
recompute instead of loading cached CSVs.

Writes under figure_dir("chi_shap", "shap_median_vs_tail", <partition>).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.ticker import MaxNLocator
from ngboost import NGBRegressor
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FEATURE_DISPLAY,
    METRICS,
    NGB_FEATURES,
    PARTITION_LABELS,
    PARTITIONS,
    importance_table,
    load_partition,
    ngboost_model_path,
    out_dir,
    partition_shap_sample,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import (  # noqa: E402
    LABEL_FONTSIZE,
    TICK_LABELSIZE,
    add_panel_label,
    apply_full_paper_style,
    factor_color,
    figsize,
    metric_label,
    save_figure,
)

warnings.filterwarnings("ignore")
apply_full_paper_style(auto_format=True, frame="open", grid=False)

FORCE = "--force" in sys.argv
NGB_EXPLAIN_N = 400
NGB_BG_N = 100

# (id, tail target, row label) — Δ = tail − median
CONTRASTS = (
    ("q05_minus_q50", "q05", r"$q_{0.05}-q_{0.50}$"),
    ("q95_minus_q50", "q95", r"$q_{0.95}-q_{0.50}$"),
)
QUANTILE_TAUS = {"q05": 0.05, "q50": 0.50, "q95": 0.95}


def _explain_ngb(predict_fn, X_bg: np.ndarray, X_ex: np.ndarray) -> np.ndarray:
    explainer = shap.Explainer(predict_fn, X_bg, algorithm="permutation")
    explanation = explainer(X_ex, max_evals=2 * X_ex.shape[1] + 1)
    return np.asarray(explanation.values, dtype=float)


def _load_or_compute_quantiles(partition: str) -> dict[str, pd.DataFrame]:
    """Permutation SHAP of \\(q_\\tau=\\mu+z_\\tau\\sigma\\) via μ and σ composition.

    SHAP is linear, so \\(\\phi(q_\\tau)=\\phi(\\mu)+z_\\tau\\phi(\\sigma)\\). Caching
    writes ``shap_importance_q{05,50,95}.csv`` under ``shap_ngboost/<partition>``.
    """
    ngb_dir = out_dir("shap_ngboost", partition)
    paths = {t: ngb_dir / f"shap_importance_{t}.csv" for t in QUANTILE_TAUS}
    if not FORCE and all(p.is_file() for p in paths.values()):
        out = {}
        for t, p in paths.items():
            tab = pd.read_csv(p)
            if not len(tab) or not {"metric", "feature", "mean_abs_shap"}.issubset(tab.columns):
                break
            print(f"Loaded {p}")
            out[t] = tab
        else:
            return out

    print(f"Computing NGBoost permutation SHAP for μ and σ [{partition}] …")
    df = load_partition(partition)
    bg_idx, ex_idx, _ = partition_shap_sample(df, partition, explain_n=NGB_EXPLAIN_N, bg_n=NGB_BG_N)
    X_bg = df.iloc[bg_idx][NGB_FEATURES].to_numpy(dtype=float)
    X_ex = df.iloc[ex_idx][NGB_FEATURES].to_numpy(dtype=float)
    z05 = float(norm.ppf(0.05))
    z95 = float(norm.ppf(0.95))
    rows: dict[str, list[pd.DataFrame]] = {t: [] for t in QUANTILE_TAUS}

    for metric in METRICS:
        mpath = ngboost_model_path(partition, metric)
        if not mpath.is_file():
            raise FileNotFoundError(f"Missing NGBoost model: {mpath}")
        model: NGBRegressor = joblib.load(mpath)

        def predict_mu(X, m=model):
            dist = m.pred_dist(np.asarray(X, dtype=float))
            return np.asarray(dist.loc, dtype=float).ravel()

        def predict_sigma(X, m=model):
            dist = m.pred_dist(np.asarray(X, dtype=float))
            return np.maximum(np.asarray(dist.scale, dtype=float).ravel(), 1e-8)

        print(f"  SHAP NGBoost μ {metric} …")
        sv_mu = _explain_ngb(predict_mu, X_bg, X_ex)
        print(f"  SHAP NGBoost σ {metric} …")
        sv_sig = _explain_ngb(predict_sigma, X_bg, X_ex)
        composed = {
            "q50": sv_mu,
            "q05": sv_mu + z05 * sv_sig,
            "q95": sv_mu + z95 * sv_sig,
        }
        for t, sv in composed.items():
            rows[t].append(
                importance_table(sv, NGB_FEATURES, metric=metric, model="ngboost", target=t)
            )

    out = {}
    for t, chunks in rows.items():
        tab = pd.concat(chunks, ignore_index=True)
        tab.to_csv(paths[t], index=False)
        print(f"Wrote {paths[t]}")
        out[t] = tab
    return out


def _column_xlim(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (-0.05, 0.05)
    lo, hi = float(np.min(finite)), float(np.max(finite))
    span = max(hi - lo, max(abs(lo), abs(hi), 1e-12))
    pad = 0.10 * span
    lo, hi = min(lo - pad, 0.0), max(hi + pad, 0.0)
    if hi - lo < 1e-15:
        return (-0.05, 0.05)
    return lo, hi


def _plot_deltas(diff: pd.DataFrame, *, value_col: str, xlabel: str, stem: str, out: Path) -> None:
    """2×5: rows = (0.05−0.50, 0.95−0.50), columns = χ metrics."""
    n_metrics = len(METRICS)
    fig, axes = plt.subplots(2, n_metrics, figsize=figsize(height=4.85), sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.12, right=0.995, bottom=0.08, top=0.96, wspace=0.06, hspace=0.10)

    y = np.arange(len(NGB_FEATURES))
    yticklabels = [FEATURE_DISPLAY.get(f, f) for f in NGB_FEATURES]
    colors = [factor_color(f) for f in NGB_FEATURES]

    for col, metric in enumerate(METRICS):
        col_vals = []
        for cid, _tail, _lab in CONTRASTS:
            sub = diff[(diff["metric"] == metric) & (diff["contrast"] == cid)]
            sub = sub.set_index("feature").reindex(NGB_FEATURES)
            col_vals.append(sub[value_col].to_numpy(dtype=float))
        xlim = _column_xlim(np.concatenate(col_vals))

        for row, ((_cid, _tail, row_lab), vals) in enumerate(zip(CONTRASTS, col_vals)):
            ax = axes[row, col]
            ax.barh(y, vals, color=colors, height=0.7, edgecolor="none")
            ax.axvline(0.0, color="0.55", linewidth=0.5)
            ax.set_yticks(y)
            ax.set_xlim(*xlim)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune=None))
            ax.tick_params(labelsize=TICK_LABELSIZE, length=2.0)
            add_panel_label(ax, row * n_metrics + col, x=0.97, y=0.97, alpha=0.75)
            if row == 0:
                ax.set_title(metric_label(metric, log=True), fontsize=LABEL_FONTSIZE, pad=1)
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(
                    xlabel if col == n_metrics // 2 else "", fontsize=LABEL_FONTSIZE, labelpad=1
                )
            if col == 0:
                ax.set_yticklabels(yticklabels, fontsize=TICK_LABELSIZE)
                ax.set_ylabel(row_lab, fontsize=LABEL_FONTSIZE, labelpad=1)
            else:
                ax.tick_params(axis="y", left=False, labelleft=False, length=0)
                ax.set_ylabel("")

    axes[0, 0].invert_yaxis()
    save_figure(fig, stem, out_dir=out)
    plt.close(fig)


def _build_diff(by_target: dict[str, pd.DataFrame]) -> pd.DataFrame:
    q50 = by_target["q50"]
    rows = []
    for metric in METRICS:
        med = q50[q50["metric"] == metric].set_index("feature")
        for cid, tail_key, _lab in CONTRASTS:
            tail = by_target[tail_key]
            hi = tail[tail["metric"] == metric].set_index("feature")
            for feat in NGB_FEATURES:
                if feat not in med.index or feat not in hi.index:
                    continue
                abs_med = float(med.loc[feat, "mean_abs_shap"])
                abs_tail = float(hi.loc[feat, "mean_abs_shap"])
                signed_med = float(med.loc[feat, "mean_signed_shap"])
                signed_tail = float(hi.loc[feat, "mean_signed_shap"])
                rows.append(
                    {
                        "metric": metric,
                        "feature": feat,
                        "contrast": cid,
                        "tau": QUANTILE_TAUS[tail_key],
                        "mean_abs_shap_q50": abs_med,
                        "mean_abs_shap_tau": abs_tail,
                        "mean_signed_shap_q50": signed_med,
                        "mean_signed_shap_tau": signed_tail,
                        "delta_mean_abs_shap": abs_tail - abs_med,
                        "delta_mean_signed_shap": signed_tail - signed_med,
                        "abs_ratio_tau_over_q50": (abs_tail / abs_med if abs_med > 0 else np.nan),
                    }
                )
    diff = pd.DataFrame(rows)
    if not len(diff):
        raise RuntimeError("No overlapping q05/q50/q95 SHAP rows to compare.")
    diff["rank_abs_delta"] = (
        diff.groupby(["metric", "contrast"])["delta_mean_abs_shap"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    return diff.sort_values(["contrast", "metric", "rank_abs_delta"])


def run_partition(partition: str) -> None:
    out = out_dir("shap_median_vs_tail", partition)
    diff = _build_diff(_load_or_compute_quantiles(partition))
    diff.to_csv(out / "shap_median_vs_tail.csv", index=False)
    _plot_deltas(
        diff,
        value_col="delta_mean_abs_shap",
        xlabel="SHAP value",
        stem="shap_median_vs_tail_delta_abs",
        out=out,
    )
    _plot_deltas(
        diff,
        value_col="delta_mean_signed_shap",
        xlabel="SHAP value",
        stem="shap_median_vs_tail_delta_signed",
        out=out,
    )

    top = (
        diff.sort_values("delta_mean_abs_shap", ascending=False)
        .groupby(["contrast", "metric"], as_index=False)
        .head(3)
    )
    sigma_meaning = (
        "between-seed spread at the center node"
        if partition == "between"
        else "within-seed spread across nodes"
    )
    lines = [
        f"# NGBoost SHAP: median vs tails — {PARTITION_LABELS[partition]}",
        "",
        "## Definitions",
        "",
        r"- Source: NGBoost permutation SHAP of the Normal quantile "
        r"\(q_\tau(\mathbf{x})=\mu(\mathbf{x})+z_\tau\sigma(\mathbf{x})\), with "
        r"\(\phi(q_\tau)=\phi(\mu)+z_\tau\phi(\sigma)\). "
        f"Here σ is the {sigma_meaning}.",
        r"- Contrasts: \(\Delta\overline{|\phi|}=\overline{|\phi|}_\tau-\overline{|\phi|}_{0.50}\) "
        r"and \(\Delta\overline{\phi}=\overline{\phi}_\tau-\overline{\phi}_{0.50}\) "
        r"for \(\tau\in\{0.05,0.95\}\).",
        r"- Because \(q_{0.50}\equiv\mu\) and \(z_{0.05}=-z_{0.95}\), signed deltas for "
        "the two tails are exact opposites; absolute deltas are not.",
        "",
        "## Largest |SHAP| increases toward each tail",
        "",
        top.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Output files",
        "",
        "| File | Content |",
        "|------|---------|",
        "| `shap_median_vs_tail.csv` | q05/q50/q95 SHAP and both tail−median deltas |",
        "| `shap_median_vs_tail_delta_abs.pdf` | 2×5 Δ mean \\|SHAP\\| |",
        "| `shap_median_vs_tail_delta_signed.pdf` | 2×5 Δ signed SHAP |",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


def main() -> None:
    for partition in PARTITIONS:
        run_partition(partition)


if __name__ == "__main__":
    main()
