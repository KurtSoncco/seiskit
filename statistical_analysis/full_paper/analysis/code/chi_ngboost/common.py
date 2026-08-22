"""Shared loaders, splits, and helpers for chi_ngboost."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from _shared import (  # noqa: E402,F401
    CENTER_NODE,
    CHI_OLS_CEILING,
    CHI_QBM_COMPARE,
    CHI_QBM_MODELS,
    CHI_QBM_ROOT,
    CHI_QBM_TRAIN,
    DATA_PATH,
    DX_M,
    FACTORS,
    FEATURES,
    LAG1_FAIL_THRESH,
    METRICS,
    N_CELLS,
    N_NODES,
    N_SEEDS,
    SPLIT_SEED,
    TAUS,
    TEST_SIZE,
    VAL_FRAC,
    VAL_SEED,
    ZCOLS,
    add_design_columns,
    fmt,
    lag1_pearson,
    load_or_make_split,
    load_ratios,
    log_response,
    pinball_loss,
    r2_score,
    rmse,
    seed_grouped_split_indices,
)
from _shared import (
    save_split as _save_split,
)
from config import figure_dir  # noqa: E402

# Subsample for NGBoost fit speed (full holdout still evaluated).
TRAIN_SUBSAMPLE_FRAC = 0.10
TRAIN_SUBSAMPLE_SEED = 2
MAX_ESTIMATORS = 200
EARLY_STOPPING_ROUNDS = 20


def out_dir(stem: str) -> Path:
    return figure_dir("chi_ngboost", stem)


def models_dir() -> Path:
    path = figure_dir("chi_ngboost", "models")
    path.mkdir(parents=True, exist_ok=True)
    return path


def surfaces_dir() -> Path:
    path = figure_dir("chi_ngboost", "surfaces")
    path.mkdir(parents=True, exist_ok=True)
    return path


def factorial_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Unique design-cell × node grid (no seed replication)."""
    keys = list(FACTORS) + ["node"]
    keep = ["cell", "node", *FACTORS, *FEATURES]
    keep = [c for c in keep if c in df.columns]
    out = df.drop_duplicates(subset=keys, keep="first")[keep].copy()
    return out.reset_index(drop=True)


def save_split(path: Path, tr: np.ndarray, te: np.ndarray, df: pd.DataFrame) -> None:
    _save_split(
        path,
        tr,
        te,
        df,
        extra={"train_subsample_frac": TRAIN_SUBSAMPLE_FRAC},
    )


def mean_abs_lag1_residuals(
    df: pd.DataFrame,
    resid: np.ndarray,
    *,
    max_blocks: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Mean |lag-1| of residuals over a sample of (cell, seed) node blocks."""
    work = df[["cell", "seed", "node"]].copy()
    work["resid"] = resid
    keys = work[["cell", "seed"]].drop_duplicates()
    rng = np.random.default_rng(seed)
    if len(keys) > max_blocks:
        idx = rng.choice(len(keys), size=max_blocks, replace=False)
        keys = keys.iloc[idx]
    lags = []
    for cell, seed_id in keys.itertuples(index=False):
        block = work[(work["cell"] == cell) & (work["seed"] == seed_id)].sort_values("node")
        r = block["resid"].to_numpy(dtype=float)
        if r.size == N_NODES:
            lags.append(lag1_pearson(r))
    lags = np.asarray(lags, dtype=float)
    lags = lags[np.isfinite(lags)]
    if lags.size == 0:
        return {
            "mean_abs_lag1": float("nan"),
            "frac_abs_lag1_gt_thresh": float("nan"),
            "n_blocks": 0,
        }
    return {
        "mean_abs_lag1": float(np.mean(np.abs(lags))),
        "frac_abs_lag1_gt_thresh": float(np.mean(np.abs(lags) > LAG1_FAIL_THRESH)),
        "n_blocks": int(lags.size),
    }
