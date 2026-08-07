"""Difference of QBM TreeSHAP importance: median (τ=0.50) vs upper tail (τ=0.95).

Reuses ``chi_shap/shap_qbm`` CSVs when present; otherwise computes TreeSHAP from
QBM models. Writes figure + CSV under figure_dir("chi_shap", "shap_median_vs_tail").
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FEATURES,
    METRICS,
    add_design_columns,
    importance_table,
    load_or_make_split,
    load_ratios,
    make_shap_sample,
    out_dir,
    qbm_model_path,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import (  # noqa: E402
    add_panel_label,
    apply_full_paper_style,
    factor_color,
    figsize,
    metric_label,
    save_figure,
)

warnings.filterwarnings("ignore")
apply_full_paper_style(auto_format=True, frame="open", grid=False)


def _as_booster(obj) -> lgb.Booster:
    if isinstance(obj, lgb.Booster):
        return obj
    if hasattr(obj, "booster_"):
        return obj.booster_
    return obj


def _load_or_compute_importance(target: str) -> pd.DataFrame:
    """Load shap_qbm CSV for *target* (`q50` / `q95`), else compute TreeSHAP."""
    shap_dir = out_dir("shap_qbm")
    path = shap_dir / f"shap_importance_{target}.csv"
    if path.is_file():
        tab = pd.read_csv(path)
        if len(tab) and {"metric", "feature", "mean_abs_shap"}.issubset(tab.columns):
            print(f"Loaded {path}")
            return tab

    print(f"Computing TreeSHAP for {target} …")
    df = add_design_columns(load_ratios())
    tr, te = load_or_make_split(df)
    bg_idx, ex_idx, _ = make_shap_sample(df, te)
    X_ex = df.iloc[ex_idx][FEATURES].to_numpy(dtype=float)
    rows = []
    kind = target  # q50 / q95
    for metric in METRICS:
        mpath = qbm_model_path(kind, metric)
        if not mpath.is_file():
            print(f"  missing {mpath}")
            continue
        booster = _as_booster(joblib.load(mpath))
        explainer = shap.TreeExplainer(booster)
        sv = np.asarray(explainer.shap_values(X_ex), dtype=float)
        rows.append(
            importance_table(sv, FEATURES, metric=metric, model="qbm", target=target)
        )
    if not rows:
        raise FileNotFoundError(f"No QBM models / SHAP CSVs for target={target}")
    return pd.concat(rows, ignore_index=True)


def _feature_color(feat: str) -> str:
    if feat == "node_z":
        return "0.35"
    try:
        return factor_color(feat)
    except KeyError:
        return "0.45"


def _plot_deltas(diff: pd.DataFrame, out: Path) -> None:
    n = len(METRICS)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize(height=min(6.5, 2.0 * nrows)),
        squeeze=False,
    )
    for i, metric in enumerate(METRICS):
        ax = axes[i // ncols, i % ncols]
        sub = diff[diff["metric"] == metric].copy()
        sub = sub.set_index("feature").reindex(FEATURES).reset_index()
        y = np.arange(len(sub))
        colors = [_feature_color(f) for f in sub["feature"]]
        ax.barh(y, sub["delta_mean_abs_shap"], color=colors, height=0.7, edgecolor="none")
        ax.axvline(0.0, color="0.55", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["feature"], fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel(r"$\overline{|\phi|}_{0.95}-\overline{|\phi|}_{0.50}$")
        ax.set_title(metric_label(metric, log=True), fontsize=7)
        add_panel_label(ax, i)
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)
    fig.tight_layout(pad=0.4)
    save_figure(fig, "shap_median_vs_tail_delta_abs", out_dir=out)
    plt.close(fig)

    # Signed difference companion figure
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize(height=min(6.5, 2.0 * nrows)),
        squeeze=False,
    )
    for i, metric in enumerate(METRICS):
        ax = axes[i // ncols, i % ncols]
        sub = diff[diff["metric"] == metric].copy()
        sub = sub.set_index("feature").reindex(FEATURES).reset_index()
        y = np.arange(len(sub))
        colors = [_feature_color(f) for f in sub["feature"]]
        ax.barh(y, sub["delta_mean_signed_shap"], color=colors, height=0.7, edgecolor="none")
        ax.axvline(0.0, color="0.55", linewidth=0.5)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["feature"], fontsize=6)
        ax.invert_yaxis()
        ax.set_xlabel(r"$\overline{\phi}_{0.95}-\overline{\phi}_{0.50}$")
        ax.set_title(metric_label(metric, log=True), fontsize=7)
        add_panel_label(ax, i)
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)
    fig.tight_layout(pad=0.4)
    save_figure(fig, "shap_median_vs_tail_delta_signed", out_dir=out)
    plt.close(fig)


def main() -> None:
    out = out_dir("shap_median_vs_tail")
    q50 = _load_or_compute_importance("q50")
    q95 = _load_or_compute_importance("q95")

    rows = []
    for metric in METRICS:
        a = q50[q50["metric"] == metric].set_index("feature")
        b = q95[q95["metric"] == metric].set_index("feature")
        for feat in FEATURES:
            if feat not in a.index or feat not in b.index:
                continue
            rows.append(
                {
                    "metric": metric,
                    "feature": feat,
                    "mean_abs_shap_q50": float(a.loc[feat, "mean_abs_shap"]),
                    "mean_abs_shap_q95": float(b.loc[feat, "mean_abs_shap"]),
                    "mean_signed_shap_q50": float(a.loc[feat, "mean_signed_shap"]),
                    "mean_signed_shap_q95": float(b.loc[feat, "mean_signed_shap"]),
                    "delta_mean_abs_shap": float(
                        b.loc[feat, "mean_abs_shap"] - a.loc[feat, "mean_abs_shap"]
                    ),
                    "delta_mean_signed_shap": float(
                        b.loc[feat, "mean_signed_shap"] - a.loc[feat, "mean_signed_shap"]
                    ),
                    "abs_ratio_q95_over_q50": float(
                        b.loc[feat, "mean_abs_shap"]
                        / a.loc[feat, "mean_abs_shap"]
                        if float(a.loc[feat, "mean_abs_shap"]) > 0
                        else np.nan
                    ),
                }
            )

    diff = pd.DataFrame(rows)
    if not len(diff):
        raise RuntimeError("No overlapping q50/q95 SHAP rows to compare.")
    diff["rank_abs_delta"] = (
        diff.groupby("metric")["delta_mean_abs_shap"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    diff = diff.sort_values(["metric", "rank_abs_delta"])
    diff.to_csv(out / "shap_median_vs_tail.csv", index=False)
    _plot_deltas(diff, out)

    top = (
        diff.sort_values("delta_mean_abs_shap", ascending=False)
        .groupby("metric", as_index=False)
        .head(3)
    )

    lines = [
        "# QBM SHAP: median (τ=0.50) vs upper tail (τ=0.95)",
        "",
        "## Definitions",
        "",
        r"- Source: TreeSHAP mean \(|\phi|\) and signed mean \(\overline{\phi}\) from "
        r"`chi_shap/shap_qbm` (computed on demand if CSVs are missing).",
        r"- \(\Delta\overline{|\phi|}=\overline{|\phi|}_{\tau=0.95}-\overline{|\phi|}_{\tau=0.50}\).",
        r"- \(\Delta\overline{\phi}=\overline{\phi}_{0.95}-\overline{\phi}_{0.50}\).",
        r"- Positive \(\Delta\overline{|\phi|}\): feature is more important for the upper "
        r"tail quantile than for the median.",
        "",
        "## Largest |SHAP| increases toward the tail",
        "",
        top.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Conclusions",
        "",
        "- Features with large positive \(\\Delta\\overline{|\\phi|}\) are candidate "
        "tail drivers for SR / hazard narratives; near-zero deltas indicate "
        "rank-stable importance across the center and P95.",
        "- Signed deltas can flip when the tail model reverses the average direction "
        "of a feature's contribution relative to the median.",
        "",
        "## Output files",
        "",
        "| File | Content |",
        "|------|---------|",
        "| `shap_median_vs_tail.csv` | q50/q95 SHAP and deltas |",
        "| `shap_median_vs_tail_delta_abs.pdf` | bar charts of Δ mean |SHAP| |",
        "| `shap_median_vs_tail_delta_signed.pdf` | bar charts of Δ signed SHAP |",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(diff.to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
