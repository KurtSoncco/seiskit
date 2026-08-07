"""Shared loaders and helpers for chi_ols Stage-1 / ceiling / hetero / spatial."""

from __future__ import annotations

import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parent.parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from _shared import (  # noqa: E402,F401
    ACF_FIT_PATH,
    CENTER_NODE,
    DATA_PATH,
    DX_M,
    FACTORS,
    METRICS,
    N_CELLS,
    N_NODES,
    N_SEEDS,
    ZCOLS,
    add_design_columns as _add_design_columns,
    formula_rhs_main,
    fmt,
    load_ratios,
    log_response,
)
from config import figure_dir  # noqa: E402


def add_design_columns(df):
    """OLS Stage-1: z-scored factors only (no node_z)."""
    return _add_design_columns(df, include_node_z=False)


def out_dir(stem: str) -> Path:
    return figure_dir("chi_ols", stem)
