"""Shared helpers for chi_shap.

Manuscript figures use the single-partition NGBoost models (between-seed /
within-seed, design factors only). QBM helpers remain for the legacy
``shap_qbm`` / ``shap_compare`` tables.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from _shared import (  # noqa: E402,F401
    CHI_QBM_MODELS,
    FACTORS,
    FEATURES,
    METRICS,
    N_NODES,
    NGB_FEATURES,
    PARTITION_LABELS,
    PARTITIONS,
    SPLIT_SEED,
    SPREAD_KINDS,
    SPREAD_LABELS,
    TAUS,
    TEST_SIZE,
    ZCOLS,
    add_design_columns,
    fmt,
    load_or_make_split,
    load_partition,
    load_ratios,
    load_spread,
    log_response,
    partition_split,
    r2_score,
    spread_split,
)
from config import figure_dir  # noqa: E402

FEATURE_DISPLAY = {
    "Vs1_z": r"$V_{s1}$",
    "Height_z": r"$H$",
    "CoV_z": "CoV",
    "rH_z": r"$r_h$",
    "aHV_z": r"$a_{hv}$",
}

SHAP_BG_N = 200
SHAP_EXPLAIN_N = 1500
SHAP_SAMPLE_SEED = 3
SHAP_TAUS = [0.05, 0.50, 0.95]
TOP_K_FEATURES = 4
TOP_K_INTERACTIONS = 3


def out_dir(stem: str, partition: str | None = None) -> Path:
    if partition is None:
        return figure_dir("chi_shap", stem)
    return figure_dir("chi_shap", stem, partition)


def ngboost_models_dir(partition: str) -> Path:
    path = figure_dir("chi_ngboost", partition, "models")
    path.mkdir(parents=True, exist_ok=True)
    return path


def ngboost_model_path(partition: str, metric: str) -> Path:
    return ngboost_models_dir(partition) / f"ngboost_{metric}.pkl"


def spread_model_path(kind: str, metric: str, part: str = "normal") -> Path:
    """Spread NGBoost from chi_ngboost/train_spread.py (part: 'normal' or 'zero')."""
    stem = "spread" if part == "normal" else "spread_zero"
    return figure_dir("chi_ngboost", "spread", kind, "models") / f"{stem}_{metric}.pkl"


def factor_levels(df: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{feature_z: (z_levels, raw_levels)}`` for the three-level design factors."""
    out = {}
    for f in FACTORS:
        pairs = df[[f"{f}_z", f]].drop_duplicates().sort_values(f"{f}_z")
        out[f"{f}_z"] = (pairs[f"{f}_z"].to_numpy(dtype=float), pairs[f].to_numpy(dtype=float))
    return out


def format_level(value: float) -> str:
    return f"{int(value)}" if float(value).is_integer() else f"{value:g}"


def partition_shap_sample(
    df: pd.DataFrame, partition: str, *, explain_n: int, bg_n: int, seed: int = SHAP_SAMPLE_SEED
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Background + explain rows from the partition holdout (disjoint when possible)."""
    _, te = partition_split(df, partition)
    rng = np.random.default_rng(seed)
    te = np.asarray(te)
    if len(te) >= bg_n + explain_n:
        pick = rng.choice(te, size=bg_n + explain_n, replace=False)
        bg, ex = pick[:bg_n], pick[bg_n:]
    else:
        bg = rng.choice(te, size=min(bg_n, len(te)), replace=False)
        ex = te
    meta = {
        "partition": partition,
        "bg_n": int(len(bg)),
        "explain_n": int(len(ex)),
        "sample_seed": seed,
        "features": NGB_FEATURES,
    }
    return bg, ex, meta


def qbm_model_path(kind: str, metric: str) -> Path:
    """kind: 'mean' or 'q05'/'q50'/'q95'."""
    if kind == "mean":
        return CHI_QBM_MODELS / f"lgbm_mean_{metric}_seed.pkl"
    return CHI_QBM_MODELS / f"lgbm_{kind}_{metric}_seed.pkl"


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
    if sv.ndim == 3:
        sv = sv.sum(axis=2)
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
