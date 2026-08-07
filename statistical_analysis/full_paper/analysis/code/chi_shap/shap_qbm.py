"""TreeSHAP for full-array LightGBM mean GBM + QBM (τ=0.05, 0.50, 0.95).

Reads chi_qbm models (read-only). Writes CSV + summary.md under
figure_dir("chi_shap", "shap_qbm").
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FEATURES,
    METRICS,
    SHAP_TAUS,
    add_design_columns,
    importance_table,
    load_or_make_split,
    load_ratios,
    make_shap_sample,
    out_dir,
    qbm_model_path,
    shap_by_node_table,
    top_pairwise_interactions,
)

warnings.filterwarnings("ignore")


def _as_booster(obj) -> lgb.Booster:
    if isinstance(obj, lgb.Booster):
        return obj
    if hasattr(obj, "booster_"):
        return obj.booster_
    return obj


def _tree_shap(booster: lgb.Booster, X_bg: np.ndarray, X_ex: np.ndarray):
    explainer = shap.TreeExplainer(booster)
    sv = explainer.shap_values(X_ex)
    # interactions are expensive; compute on a subset
    n_int = min(400, len(X_ex))
    sint = explainer.shap_interaction_values(X_ex[:n_int])
    return np.asarray(sv), np.asarray(sint), n_int


def main() -> None:
    out = out_dir("shap_qbm")
    print("Loading data …")
    df = add_design_columns(load_ratios())
    tr, te = load_or_make_split(df)
    bg_idx, ex_idx, sample_meta = make_shap_sample(df, te)
    X_bg = df.iloc[bg_idx][FEATURES].to_numpy(dtype=float)
    X_ex = df.iloc[ex_idx][FEATURES].to_numpy(dtype=float)

    imp_rows = []
    int_rows = []
    node_rows = []
    meta = {"sample": sample_meta, "models": []}

    for metric in METRICS:
        # Mean GBM
        path = qbm_model_path("mean", metric)
        if not path.is_file():
            print(f"  missing {path}")
            continue
        booster = _as_booster(joblib.load(path))
        print(f"SHAP mean {metric} …")
        sv, sint, n_int = _tree_shap(booster, X_bg, X_ex)
        imp_rows.append(
            importance_table(sv, FEATURES, metric=metric, model="qbm_mean", target="mean")
        )
        int_rows.append(
            top_pairwise_interactions(
                sint, FEATURES, metric=metric, model="qbm_mean", target="mean", top_k=15
            )
        )
        node_rows.append(
            shap_by_node_table(
                df, ex_idx, sv, FEATURES, metric=metric, model="qbm_mean", target="mean"
            )
        )
        meta["models"].append(
            {"metric": metric, "kind": "mean", "path": path.name, "n_interaction": n_int}
        )

        for tau in SHAP_TAUS:
            kind = f"q{int(tau * 100):02d}"
            path = qbm_model_path(kind, metric)
            if not path.is_file():
                print(f"  missing {path}")
                continue
            booster = _as_booster(joblib.load(path))
            print(f"SHAP {kind} {metric} …")
            sv, sint, n_int = _tree_shap(booster, X_bg, X_ex)
            target = f"q{int(tau * 100):02d}"
            imp_rows.append(
                importance_table(sv, FEATURES, metric=metric, model="qbm", target=target)
            )
            if tau == 0.50:
                int_rows.append(
                    top_pairwise_interactions(
                        sint, FEATURES, metric=metric, model="qbm", target=target, top_k=15
                    )
                )
            node_rows.append(
                shap_by_node_table(
                    df, ex_idx, sv, FEATURES, metric=metric, model="qbm", target=target
                )
            )
            meta["models"].append(
                {"metric": metric, "kind": kind, "path": path.name, "n_interaction": n_int}
            )

    imp = pd.concat(imp_rows, ignore_index=True) if imp_rows else pd.DataFrame()
    inter = pd.concat(int_rows, ignore_index=True) if int_rows else pd.DataFrame()
    by_node = pd.concat(node_rows, ignore_index=True) if node_rows else pd.DataFrame()

    # Split importance by target for plan-named files
    imp.to_csv(out / "shap_importance_all.csv", index=False)
    for target, name in [
        ("mean", "shap_importance_mean.csv"),
        ("q05", "shap_importance_q05.csv"),
        ("q50", "shap_importance_q50.csv"),
        ("q95", "shap_importance_q95.csv"),
    ]:
        sub = imp[imp["target"] == target] if len(imp) else imp
        sub.to_csv(out / name, index=False)

    inter.to_csv(out / "shap_interactions_top.csv", index=False)
    by_node.to_csv(out / "shap_by_node.csv", index=False)
    (out / "shap_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Compact mean-abs table for summary
    pivot = (
        imp.groupby(["metric", "target", "feature"], as_index=False)["mean_abs_shap"].mean()
        if len(imp)
        else pd.DataFrame()
    )

    lines = [
        "# QBM / mean GBM TreeSHAP summary",
        "",
        "## Definitions",
        "",
        r"- Models: LightGBM mean GBM and QBM at \(\tau\in\{0.05,0.50,0.95\}\) from `chi_qbm/models` (seed split).",
        r"- Features: z-scored design factors + `node_z`.",
        r"- \(\phi_j\): TreeSHAP attribution for feature \(j\) on a shared seed-holdout explain sample.",
        r"- Importance: mean \(|\phi_j|\) over explain rows; interactions: mean \(|\phi_{jk}|\) on a subset.",
        r"- SHAP interprets the learned conditional mean/quantile — it does **not** remove residual spatial autocorrelation.",
        "",
        "## Output files",
        "",
        "| File | Content |",
        "|------|---------|",
        "| `shap_importance_*.csv` | mean |SHAP| and signed mean by feature |",
        "| `shap_interactions_top.csv` | top pairwise interactions (mean + τ=0.50) |",
        "| `shap_by_node.csv` | SHAP aggregated by node index |",
        "| `shap_meta.json` | sample sizes and model list |",
        "",
        "## Top features (mean |SHAP|)",
        "",
    ]
    if len(pivot):
        top = (
            pivot.sort_values("mean_abs_shap", ascending=False)
            .groupby(["metric", "target"], as_index=False)
            .head(3)
        )
        lines.append(top.to_markdown(index=False, floatfmt=".4f"))
    lines += [
        "",
        "## Conclusions",
        "",
        "- Quantile-specific SHAP ranks show how design and `node_z` reshape the center vs tails of \\(Y=\\ln\\chi\\).",
        "- Large `node_z` attributions indicate the model uses the spatial coordinate for broad trends; high residual lag-1 (from QBM/NGBoost diagnostics) still implies unmodelled short-range dependence.",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
