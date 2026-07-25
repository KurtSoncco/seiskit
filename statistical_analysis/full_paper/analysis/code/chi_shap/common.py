"""Shared helpers for chi_shap (NGBoost + QBM attributions)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Reuse NGBoost common for data / splits / features (load by path to avoid
# shadowing this package's `common` when scripts insert chi_shap on sys.path).
import importlib.util

_ngb_common_path = Path(__file__).resolve().parent.parent / "chi_ngboost" / "common.py"
_spec = importlib.util.spec_from_file_location("chi_ngboost_common", _ngb_common_path)
assert _spec is not None and _spec.loader is not None
_ngb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ngb)

CHI_QBM_MODELS = _ngb.CHI_QBM_MODELS
FEATURES = _ngb.FEATURES
METRICS = _ngb.METRICS
N_NODES = _ngb.N_NODES
SPLIT_SEED = _ngb.SPLIT_SEED
TAUS = _ngb.TAUS
TEST_SIZE = _ngb.TEST_SIZE
ZCOLS = _ngb.ZCOLS
add_design_columns = _ngb.add_design_columns
fmt = _ngb.fmt
load_or_make_split = _ngb.load_or_make_split
load_ratios = _ngb.load_ratios
log_response = _ngb.log_response
ngboost_models_dir = _ngb.models_dir
r2_score = _ngb.r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import FACTORS, figure_dir  # noqa: E402

SHAP_BG_N = 200
SHAP_EXPLAIN_N = 1500
SHAP_SAMPLE_SEED = 3
SHAP_TAUS = [0.05, 0.50, 0.95]
TOP_K_FEATURES = 4
TOP_K_INTERACTIONS = 3


def out_dir(stem: str) -> Path:
    return figure_dir("chi_shap", stem)


def qbm_model_path(kind: str, metric: str) -> Path:
    """kind: 'mean' or 'q05'/'q50'/'q95'."""
    if kind == "mean":
        return CHI_QBM_MODELS / f"lgbm_mean_{metric}_seed.pkl"
    return CHI_QBM_MODELS / f"lgbm_{kind}_{metric}_seed.pkl"


def ngboost_model_path(metric: str) -> Path:
    return ngboost_models_dir() / f"ngboost_{metric}.pkl"


def make_shap_sample(df: pd.DataFrame, te: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Background + explain indices within the test set."""
    rng = np.random.default_rng(SHAP_SAMPLE_SEED)
    te = np.asarray(te)
    if len(te) < SHAP_BG_N + SHAP_EXPLAIN_N:
        bg = te
        ex = te
    else:
        pick = rng.choice(te, size=SHAP_BG_N + SHAP_EXPLAIN_N, replace=False)
        bg, ex = pick[:SHAP_BG_N], pick[SHAP_BG_N:]
    meta = {
        "bg_n": int(len(bg)),
        "explain_n": int(len(ex)),
        "sample_seed": SHAP_SAMPLE_SEED,
        "features": FEATURES,
    }
    return bg, ex, meta


def importance_table(
    shap_values: np.ndarray,
    feature_names: list[str],
    *,
    metric: str,
    model: str,
    target: str,
) -> pd.DataFrame:
    sv = np.asarray(shap_values, dtype=float)
    if sv.ndim == 3:  # interaction tensor unused here
        sv = sv.sum(axis=2)  # unlikely
    mean_abs = np.mean(np.abs(sv), axis=0)
    mean_signed = np.mean(sv, axis=0)
    rows = []
    for j, name in enumerate(feature_names):
        rows.append(
            {
                "metric": metric,
                "model": model,
                "target": target,
                "feature": name,
                "mean_abs_shap": float(mean_abs[j]),
                "mean_signed_shap": float(mean_signed[j]),
                "rank": 0,
            }
        )
    tab = pd.DataFrame(rows)
    tab["rank"] = tab["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
    return tab.sort_values("rank")


def top_pairwise_interactions(
    shap_interaction: np.ndarray,
    feature_names: list[str],
    *,
    metric: str,
    model: str,
    target: str,
    top_k: int = 10,
) -> pd.DataFrame:
    """Mean |φ_jk| for j < k from TreeSHAP interaction values."""
    S = np.asarray(shap_interaction, dtype=float)
    # S shape: (n, p, p)
    p = len(feature_names)
    rows = []
    for j in range(p):
        for k in range(j + 1, p):
            vals = S[:, j, k]
            rows.append(
                {
                    "metric": metric,
                    "model": model,
                    "target": target,
                    "feature_i": feature_names[j],
                    "feature_j": feature_names[k],
                    "mean_abs_interaction": float(np.mean(np.abs(vals))),
                }
            )
    tab = pd.DataFrame(rows).sort_values("mean_abs_interaction", ascending=False)
    tab["rank"] = np.arange(1, len(tab) + 1)
    return tab.head(top_k)


def shap_by_node_table(
    df: pd.DataFrame,
    explain_idx: np.ndarray,
    shap_values: np.ndarray,
    feature_names: list[str],
    *,
    metric: str,
    model: str,
    target: str,
) -> pd.DataFrame:
    nodes = df.iloc[explain_idx]["node"].to_numpy()
    sv = np.asarray(shap_values, dtype=float)
    rows = []
    for node in sorted(np.unique(nodes)):
        m = nodes == node
        if not np.any(m):
            continue
        for j, name in enumerate(feature_names):
            rows.append(
                {
                    "metric": metric,
                    "model": model,
                    "target": target,
                    "node": int(node),
                    "feature": name,
                    "mean_abs_shap": float(np.mean(np.abs(sv[m, j]))),
                    "mean_signed_shap": float(np.mean(sv[m, j])),
                }
            )
    return pd.DataFrame(rows)
