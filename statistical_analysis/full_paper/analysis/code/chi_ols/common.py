"""Shared loaders and helpers for chi_ols Stage-1 / ceiling / hetero / spatial."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import BOX_ROOT, FACTORS, METRICS, figure_dir  # noqa: E402

DATA_PATH = BOX_ROOT / "peak_analysis" / "join_master.h5"
ACF_FIT_PATH = (
    BOX_ROOT / "full_paper" / "figures" / "chi_spatial" / "spatial_acf" / "acf_fit_params.csv"
)
N_NODES = 101
N_SEEDS = 100
N_CELLS = 243
CENTER_NODE = 50
DX_M = 2.0

ZCOLS = [f"{c}_z" for c in FACTORS]


def load_ratios(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load joined ratio table; rename channel → node."""
    cols = [
        "Vs1",
        "Height",
        "CoV",
        "rH",
        "aHV",
        "channel",
        "seed",
        *METRICS,
    ]
    with h5py.File(path, "r") as f:
        g = f["master"]
        df = pd.DataFrame({c: g[c][:] for c in cols})
    return df.rename(columns={"channel": "node"})


def add_design_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add cell id and z-scored factor columns (population z over rows)."""
    out = df.copy()
    out["cell"] = out.groupby(list(FACTORS), sort=False).ngroup()
    for c in FACTORS:
        mu = float(out[c].mean())
        sd = float(out[c].std(ddof=0))
        out[f"{c}_z"] = (out[c] - mu) / sd if sd > 0 else 0.0
    return out


def log_response(df: pd.DataFrame, metric: str) -> np.ndarray:
    """Return ln(χ); non-positive / non-finite → NaN."""
    chi = df[metric].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.where(np.isfinite(chi) & (chi > 0), np.log(chi), np.nan)
    return y


def formula_rhs_main() -> str:
    return " + ".join(ZCOLS)


def out_dir(stem: str) -> Path:
    return figure_dir("chi_ols", stem)


def fmt(x: float, digits: int = 4) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{digits}f}"
