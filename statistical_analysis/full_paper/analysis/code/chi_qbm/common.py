"""Shared loaders, splits, and helpers for chi_qbm (spatial OLS + QBM)."""

from __future__ import annotations

import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parent.parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from _shared import (  # noqa: E402,F401
    ACF_FIT_PATH,
    CENTER_NODE,
    CHI_OLS_CEILING,
    CHI_OLS_STAGE1,
    DATA_PATH,
    DX_M,
    FACTORS,
    FEATURES,
    LAG1_FAIL_THRESH,
    METRICS,
    N_CELLS,
    N_NODES,
    N_SEEDS,
    QBM_FEATURES,
    SPLIT_SEED,
    TAUS,
    TEST_SIZE,
    VAL_FRAC,
    VAL_SEED,
    ZCOLS,
    add_design_columns,
    fmt,
    koenker_pseudo_r2,
    lag1_pearson,
    load_or_make_split,
    load_ratios,
    log_response,
    pinball_loss,
    r2_score,
    rmse,
    save_split,
    seed_grouped_split_indices,
)
from config import figure_dir  # noqa: E402


def out_dir(stem: str) -> Path:
    return figure_dir("chi_qbm", stem)


def models_dir() -> Path:
    path = figure_dir("chi_qbm", "models")
    path.mkdir(parents=True, exist_ok=True)
    return path
