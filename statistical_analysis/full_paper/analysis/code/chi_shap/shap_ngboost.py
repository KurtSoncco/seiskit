"""SHAP for NGBoost predictive mean μ(x) and log-scale log σ(x), per partition.

Between-seed and within-seed models are explained separately (σ is a
different variance in each). Prediction wrapper + shap.Explainer
(permutation) on a holdout sample of that partition. Writes under
figure_dir("chi_shap", "shap_ngboost", <partition>).
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from ngboost import NGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
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

warnings.filterwarnings("ignore")

EXPLAIN_N = 400
BG_N = 100


class _MuModel:
    def __init__(self, model: NGBRegressor):
        self.model = model

    def predict(self, X):
        dist = self.model.pred_dist(np.asarray(X, dtype=float))
        return np.asarray(dist.loc, dtype=float).ravel()


class _LogSigmaModel:
    def __init__(self, model: NGBRegressor):
        self.model = model

    def predict(self, X):
        dist = self.model.pred_dist(np.asarray(X, dtype=float))
        sigma = np.maximum(np.asarray(dist.scale, dtype=float).ravel(), 1e-8)
        return np.log(sigma)


def _explain(predict_fn, X_bg: np.ndarray, X_ex: np.ndarray) -> np.ndarray:
    explainer = shap.Explainer(predict_fn, X_bg, algorithm="permutation")
    explanation = explainer(X_ex, max_evals=2 * X_ex.shape[1] + 1)
    return np.asarray(explanation.values, dtype=float)


def load_model(partition: str, metric: str) -> NGBRegressor:
    path = ngboost_model_path(partition, metric)
    if not path.is_file():
        raise FileNotFoundError(f"Missing NGBoost model: {path}. Run train_ngboost.py first.")
    return joblib.load(path)


def _proxy_interactions(imp_rows: list[pd.DataFrame]) -> pd.DataFrame:
    """Product of mean |SHAP| per feature pair (cheap proxy, not SHAP interactions)."""
    inter_rows = []
    for tab in imp_rows:
        metric, target = tab.iloc[0]["metric"], tab.iloc[0]["target"]
        vals = tab.set_index("feature")["mean_abs_shap"].to_dict()
        feats = tab.sort_values("mean_abs_shap", ascending=False)["feature"].tolist()
        pairs = [
            {
                "metric": metric,
                "model": "ngboost",
                "target": target,
                "feature_i": feats[i],
                "feature_j": feats[j],
                "mean_abs_interaction": float(vals[feats[i]] * vals[feats[j]]),
                "note": "product_proxy",
            }
            for i in range(len(feats))
            for j in range(i + 1, len(feats))
        ]
        pairs = sorted(pairs, key=lambda d: -d["mean_abs_interaction"])[:10]
        for rank, p in enumerate(pairs, 1):
            p["rank"] = rank
            inter_rows.append(p)
    return pd.DataFrame(inter_rows)


def run_partition(partition: str) -> None:
    out = out_dir("shap_ngboost", partition)
    print(f"Loading {PARTITION_LABELS[partition]} …")
    df = load_partition(partition)
    bg_idx, ex_idx, sample_meta = partition_shap_sample(
        df, partition, explain_n=EXPLAIN_N, bg_n=BG_N
    )
    sample_meta["algorithm"] = "permutation"
    X_bg = df.iloc[bg_idx][NGB_FEATURES].to_numpy(dtype=float)
    X_ex = df.iloc[ex_idx][NGB_FEATURES].to_numpy(dtype=float)

    imp_rows = []
    meta = {"sample": sample_meta, "models": []}
    for metric in METRICS:
        model = load_model(partition, metric)
        print(f"SHAP NGBoost [{partition}] μ {metric} …")
        sv_mu = _explain(_MuModel(model).predict, X_bg, X_ex)
        imp_rows.append(
            importance_table(sv_mu, NGB_FEATURES, metric=metric, model="ngboost", target="mu")
        )
        print(f"SHAP NGBoost [{partition}] logσ {metric} …")
        sv_ls = _explain(_LogSigmaModel(model).predict, X_bg, X_ex)
        imp_rows.append(
            importance_table(
                sv_ls, NGB_FEATURES, metric=metric, model="ngboost", target="log_sigma"
            )
        )
        meta["models"].append(
            {"metric": metric, "path": ngboost_model_path(partition, metric).name}
        )

    imp = pd.concat(imp_rows, ignore_index=True)
    inter = _proxy_interactions(imp_rows)

    imp.to_csv(out / "shap_importance_all.csv", index=False)
    imp[imp["target"] == "mu"].to_csv(out / "shap_importance_mean.csv", index=False)
    imp[imp["target"] == "log_sigma"].to_csv(out / "shap_importance_logscale.csv", index=False)
    inter.to_csv(out / "shap_interactions_top.csv", index=False)
    (out / "shap_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    sigma_meaning = (
        "between-seed dispersion at the center node"
        if partition == "between"
        else "within-seed (node-to-node) dispersion around the cell mean"
    )
    lines = [
        f"# NGBoost SHAP summary — {PARTITION_LABELS[partition]}",
        "",
        "## Definitions",
        "",
        r"- Targets: predictive mean \(\mu(\mathbf{x})\) and log-scale \(\log\sigma(\mathbf{x})\) "
        rf"from Normal NGBoost; here \(\sigma\) is {sigma_meaning}.",
        "- Features: five z-scored design factors (no `node_z`).",
        r"- Explainer: model-agnostic permutation SHAP on a holdout subsample of this partition "
        "(see `shap_meta.json`).",
        r"- `shap_interactions_top.csv` uses a **product proxy** of mean |SHAP| (not full SHAP "
        "interactions).",
        "",
        "## Output files",
        "",
        "| File | Content |",
        "|------|---------|",
        "| `shap_importance_mean.csv` | μ attributions |",
        "| `shap_importance_logscale.csv` | log-σ attributions |",
        "| `shap_interactions_top.csv` | top feature pairs (proxy) |",
        "",
        "## Importance (mean |SHAP|)",
        "",
        imp.sort_values(["metric", "target", "rank"]).to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


def main() -> None:
    for partition in PARTITIONS:
        run_partition(partition)


if __name__ == "__main__":
    main()
