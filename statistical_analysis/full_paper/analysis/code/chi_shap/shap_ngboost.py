"""SHAP for NGBoost predictive mean μ(x) and log-scale log σ(x).

Uses a prediction-wrapper + shap.Explainer (Permutation/auto) on a shared
seed-holdout sample. Writes under figure_dir("chi_shap", "shap_ngboost").
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
    FEATURES,
    METRICS,
    add_design_columns,
    importance_table,
    load_or_make_split,
    load_ratios,
    make_shap_sample,
    ngboost_model_path,
    out_dir,
    shap_by_node_table,
)

warnings.filterwarnings("ignore")


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
    # Permutation explainer is model-agnostic and stable for NGBoost wrappers
    explainer = shap.Explainer(predict_fn, X_bg, algorithm="permutation")
    explanation = explainer(X_ex, max_evals=2 * X_ex.shape[1] + 1)
    return np.asarray(explanation.values, dtype=float)


def main() -> None:
    out = out_dir("shap_ngboost")
    print("Loading data …")
    df = add_design_columns(load_ratios())
    tr, te = load_or_make_split(df)
    bg_idx, ex_idx, sample_meta = make_shap_sample(df, te)
    # Smaller explain set for permutation SHAP cost
    rng = np.random.default_rng(3)
    if len(ex_idx) > 400:
        ex_idx = rng.choice(ex_idx, size=400, replace=False)
    if len(bg_idx) > 100:
        bg_idx = rng.choice(bg_idx, size=100, replace=False)
    sample_meta["explain_n_used"] = int(len(ex_idx))
    sample_meta["bg_n_used"] = int(len(bg_idx))
    sample_meta["algorithm"] = "permutation"

    X_bg = df.iloc[bg_idx][FEATURES].to_numpy(dtype=float)
    X_ex = df.iloc[ex_idx][FEATURES].to_numpy(dtype=float)

    imp_rows = []
    node_rows = []
    meta = {"sample": sample_meta, "models": []}

    for metric in METRICS:
        path = ngboost_model_path(metric)
        if not path.is_file():
            raise FileNotFoundError(f"Missing NGBoost model: {path}. Run train_ngboost.py first.")
        model: NGBRegressor = joblib.load(path)
        print(f"SHAP NGBoost μ {metric} …")
        mu_m = _MuModel(model)
        sv_mu = _explain(mu_m.predict, X_bg, X_ex)
        imp_rows.append(
            importance_table(sv_mu, FEATURES, metric=metric, model="ngboost", target="mu")
        )
        node_rows.append(
            shap_by_node_table(
                df, ex_idx, sv_mu, FEATURES, metric=metric, model="ngboost", target="mu"
            )
        )

        print(f"SHAP NGBoost logσ {metric} …")
        ls_m = _LogSigmaModel(model)
        sv_ls = _explain(ls_m.predict, X_bg, X_ex)
        imp_rows.append(
            importance_table(sv_ls, FEATURES, metric=metric, model="ngboost", target="log_sigma")
        )
        node_rows.append(
            shap_by_node_table(
                df, ex_idx, sv_ls, FEATURES, metric=metric, model="ngboost", target="log_sigma"
            )
        )
        meta["models"].append({"metric": metric, "path": path.name})

    # Approximate pairwise interactions via product of |signed| co-importance (cheap proxy)
    # True interactions require KernelSHAP interaction — skip heavy compute; leave empty file note
    inter_rows = []
    for metric in METRICS:
        for target in ("mu", "log_sigma"):
            sub = [
                r
                for r in imp_rows
                if len(r) and r.iloc[0]["metric"] == metric and r.iloc[0]["target"] == target
            ]
            if not sub:
                continue
            tab = sub[0].sort_values("mean_abs_shap", ascending=False)
            feats = tab["feature"].tolist()
            vals = tab.set_index("feature")["mean_abs_shap"].to_dict()
            pairs = []
            for i in range(len(feats)):
                for j in range(i + 1, len(feats)):
                    pairs.append(
                        {
                            "metric": metric,
                            "model": "ngboost",
                            "target": target,
                            "feature_i": feats[i],
                            "feature_j": feats[j],
                            "mean_abs_interaction": float(vals[feats[i]] * vals[feats[j]]),
                            "note": "product_proxy",
                        }
                    )
            pairs = sorted(pairs, key=lambda d: -d["mean_abs_interaction"])[:10]
            for rank, p in enumerate(pairs, 1):
                p["rank"] = rank
                inter_rows.append(p)

    imp = pd.concat(imp_rows, ignore_index=True)
    by_node = pd.concat(node_rows, ignore_index=True)
    inter = pd.DataFrame(inter_rows)

    imp.to_csv(out / "shap_importance_all.csv", index=False)
    imp[imp["target"] == "mu"].to_csv(out / "shap_importance_mean.csv", index=False)
    imp[imp["target"] == "log_sigma"].to_csv(out / "shap_importance_logscale.csv", index=False)
    inter.to_csv(out / "shap_interactions_top.csv", index=False)
    by_node.to_csv(out / "shap_by_node.csv", index=False)
    (out / "shap_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    lines = [
        "# NGBoost SHAP summary",
        "",
        "## Definitions",
        "",
        r"- Targets: predictive mean \(\mu(\mathbf{x})\) and log-scale \(\log\sigma(\mathbf{x})\) from Normal NGBoost.",
        r"- Explainer: model-agnostic permutation SHAP on a seed-holdout subsample (see `shap_meta.json`).",
        r"- Pairwise `shap_interactions_top.csv` uses a **product proxy** of mean |SHAP| (not full SHAP interactions) for tractability.",
        r"- SHAP describes the learned conditional μ/σ; residual spatial lag-1 remains a separate diagnostic.",
        "",
        "## Output files",
        "",
        "| File | Content |",
        "|------|---------|",
        "| `shap_importance_mean.csv` | μ attributions |",
        "| `shap_importance_logscale.csv` | log-σ attributions |",
        "| `shap_interactions_top.csv` | top feature pairs (proxy) |",
        "| `shap_by_node.csv` | attributions by node |",
        "",
        "## Importance (mean |SHAP|)",
        "",
        imp.sort_values(["metric", "target", "rank"]).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Conclusions",
        "",
        "- Feature ranks for \\(\\mu\\) vs \\(\\log\\sigma\\) need not match: dispersion drivers can differ from mean drivers.",
        "- If `node_z` captures only a smooth spatial trend while residual lag-1 stays high, short-range RF/wave structure remains unmodelled.",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
