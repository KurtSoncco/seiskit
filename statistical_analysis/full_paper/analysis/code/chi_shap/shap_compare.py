"""Compare NGBoost vs QBM SHAP importance; write SR feature shortlist.

Writes under figure_dir("chi_shap", "shap_compare").
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FEATURES,
    METRICS,
    TOP_K_FEATURES,
    TOP_K_INTERACTIONS,
    out_dir,
)


def _load_imp(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _rank_map(tab: pd.DataFrame, metric: str, feature_col: str = "feature") -> dict[str, float]:
    sub = tab[tab["metric"] == metric]
    return {r[feature_col]: float(r["mean_abs_shap"]) for _, r in sub.iterrows()}


def _top_set(vals: dict[str, float], k: int) -> set[str]:
    return set(sorted(vals, key=vals.get, reverse=True)[:k])


def main() -> None:
    out = out_dir("shap_compare")
    ngb_dir = out_dir("shap_ngboost")
    qbm_dir = out_dir("shap_qbm")

    ngb_mu = _load_imp(ngb_dir / "shap_importance_mean.csv")
    ngb_ls = _load_imp(ngb_dir / "shap_importance_logscale.csv")
    qbm_mean = _load_imp(qbm_dir / "shap_importance_mean.csv")
    qbm_q05 = _load_imp(qbm_dir / "shap_importance_q05.csv")
    qbm_q95 = _load_imp(qbm_dir / "shap_importance_q95.csv")
    ngb_int = pd.read_csv(ngb_dir / "shap_interactions_top.csv") if (ngb_dir / "shap_interactions_top.csv").is_file() else pd.DataFrame()
    qbm_int = pd.read_csv(qbm_dir / "shap_interactions_top.csv") if (qbm_dir / "shap_interactions_top.csv").is_file() else pd.DataFrame()

    compare_rows = []
    short_rows = []

    for metric in METRICS:
        mu = _rank_map(ngb_mu, metric)
        ls = _rank_map(ngb_ls, metric)
        qm = _rank_map(qbm_mean, metric)
        q05 = _rank_map(qbm_q05, metric)
        q95 = _rank_map(qbm_q95, metric)

        feats = [f for f in FEATURES if f in mu and f in qm]
        if len(feats) >= 2:
            rho_mu, _ = spearmanr([mu[f] for f in feats], [qm[f] for f in feats])
        else:
            rho_mu = float("nan")

        # Tail: average q05+q95 importance
        qtail = {f: 0.5 * (q05.get(f, 0.0) + q95.get(f, 0.0)) for f in FEATURES}
        feats2 = [f for f in FEATURES if f in ls]
        if len(feats2) >= 2:
            rho_disp, _ = spearmanr([ls[f] for f in feats2], [qtail[f] for f in feats2])
        else:
            rho_disp = float("nan")

        top_mu = _top_set(mu, TOP_K_FEATURES)
        top_qm = _top_set(qm, TOP_K_FEATURES)
        top_ls = _top_set(ls, TOP_K_FEATURES)
        top_tail = _top_set(qtail, TOP_K_FEATURES)
        overlap_mean = len(top_mu & top_qm) / max(TOP_K_FEATURES, 1)
        overlap_disp = len(top_ls & top_tail) / max(TOP_K_FEATURES, 1)

        # Sign agreement for mean targets
        sign_agree = []
        ngb_signed = ngb_mu[ngb_mu["metric"] == metric].set_index("feature")["mean_signed_shap"]
        qbm_signed = qbm_mean[qbm_mean["metric"] == metric].set_index("feature")["mean_signed_shap"]
        for f in FEATURES:
            if f in ngb_signed.index and f in qbm_signed.index:
                sign_agree.append(float(np.sign(ngb_signed[f]) == np.sign(qbm_signed[f])))
        sign_frac = float(np.mean(sign_agree)) if sign_agree else float("nan")

        def node_share(vals: dict[str, float]) -> float:
            tot = sum(vals.values())
            return float(vals.get("node_z", 0.0) / tot) if tot > 0 else float("nan")

        compare_rows.append(
            {
                "metric": metric,
                "spearman_mu_vs_qbm_mean": float(rho_mu),
                "spearman_logsigma_vs_qbm_tails": float(rho_disp),
                "topk_overlap_mean": overlap_mean,
                "topk_overlap_dispersion": overlap_disp,
                "sign_agree_frac_mean": sign_frac,
                "node_z_share_ngboost_mu": node_share(mu),
                "node_z_share_qbm_mean": node_share(qm),
                "node_z_share_ngboost_logsigma": node_share(ls),
                "node_z_share_qbm_tails": node_share(qtail),
            }
        )

        # Shortlist: top mains from NGBoost mu and log_sigma
        for target, vals, qbm_top in [
            ("mu", mu, top_qm),
            ("log_sigma", ls, top_tail),
        ]:
            ranked = sorted(vals, key=vals.get, reverse=True)
            for rank, f in enumerate(ranked[:TOP_K_FEATURES], 1):
                short_rows.append(
                    {
                        "metric": metric,
                        "kind": "main",
                        "target": target,
                        "feature": f,
                        "rank": rank,
                        "mean_abs_shap": vals[f],
                        "also_top_in_qbm_mean": int(f in top_qm) if target == "mu" else "",
                        "also_top_in_qbm_tail": int(f in top_tail) if target == "log_sigma" else int(f in top_tail),
                    }
                )

        # Top interactions from NGBoost proxy for mu
        if len(ngb_int):
            sub = ngb_int[(ngb_int["metric"] == metric) & (ngb_int["target"] == "mu")].head(
                TOP_K_INTERACTIONS
            )
            qbm_pairs = set()
            if len(qbm_int):
                qsub = qbm_int[
                    (qbm_int["metric"] == metric)
                    & (qbm_int["target"].isin(["mean", "q50"]))
                ].head(10)
                for _, r in qsub.iterrows():
                    qbm_pairs.add(tuple(sorted([r["feature_i"], r["feature_j"]])))
            for rank, (_, r) in enumerate(sub.iterrows(), 1):
                pair = tuple(sorted([r["feature_i"], r["feature_j"]]))
                short_rows.append(
                    {
                        "metric": metric,
                        "kind": "interaction",
                        "target": "mu",
                        "feature": f"{pair[0]}*{pair[1]}",
                        "feature_i": pair[0],
                        "feature_j": pair[1],
                        "rank": rank,
                        "mean_abs_shap": float(r["mean_abs_interaction"]),
                        "also_top_in_qbm_mean": int(pair in qbm_pairs),
                        "also_top_in_qbm_tail": "",
                    }
                )

    compare = pd.DataFrame(compare_rows)
    short = pd.DataFrame(short_rows)
    compare.to_csv(out / "shap_agreement.csv", index=False)
    short.to_csv(out / "feature_shortlist.csv", index=False)

    # Per-feature side-by-side for mean
    side_rows = []
    for metric in METRICS:
        for f in FEATURES:
            nm = ngb_mu[(ngb_mu["metric"] == metric) & (ngb_mu["feature"] == f)]
            qm = qbm_mean[(qbm_mean["metric"] == metric) & (qbm_mean["feature"] == f)]
            if len(nm) and len(qm):
                side_rows.append(
                    {
                        "metric": metric,
                        "feature": f,
                        "ngboost_mu_mean_abs": float(nm.iloc[0]["mean_abs_shap"]),
                        "qbm_mean_mean_abs": float(qm.iloc[0]["mean_abs_shap"]),
                        "ngboost_mu_signed": float(nm.iloc[0]["mean_signed_shap"]),
                        "qbm_mean_signed": float(qm.iloc[0]["mean_signed_shap"]),
                        "same_sign": int(
                            np.sign(nm.iloc[0]["mean_signed_shap"])
                            == np.sign(qm.iloc[0]["mean_signed_shap"])
                        ),
                    }
                )
    side = pd.DataFrame(side_rows)
    side.to_csv(out / "shap_side_by_side_mean.csv", index=False)

    lines = [
        "# NGBoost vs QBM SHAP comparison",
        "",
        "## Definitions",
        "",
        r"- Spearman: rank correlation of mean \(|\phi|\) across features for NGBoost \(\mu\) vs QBM mean GBM, and NGBoost \(\log\sigma\) vs average QBM \(\tau=0.05/0.95\) importance.",
        rf"- Top-\(k\) overlap: fraction of shared features in top {TOP_K_FEATURES} by mean \(|\phi|\).",
        r"- Sign agree: fraction of features with matching sign of mean signed SHAP (mean targets).",
        r"- `node_z` share: `node_z` mean \(|\phi|\) / sum of mean \(|\phi|\).",
        r"- `feature_shortlist.csv`: NGBoost-led mains + interactions for SR, with QBM agreement flags.",
        "",
        "## Agreement summary",
        "",
        compare.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Shortlist (head)",
        "",
        short.head(40).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Conclusions",
        "",
        "- Agreement between NGBoost and QBM SHAP supports robust design/spatial drivers; disagreement highlights distributional target differences (mean vs quantile vs scale).",
        "- SR should prefer shortlisted NGBoost features; QBM flags indicate which drivers are shared across model families.",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(compare.to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
