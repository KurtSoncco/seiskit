"""Shared loaders, splits, and helpers for chi_ngboost."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import BOX_ROOT, FACTORS, METRICS, figure_dir  # noqa: E402

DATA_PATH = BOX_ROOT / "peak_analysis" / "join_master.h5"
CHI_QBM_ROOT = BOX_ROOT / "full_paper" / "figures" / "chi_qbm"
CHI_QBM_MODELS = CHI_QBM_ROOT / "models"
CHI_QBM_TRAIN = CHI_QBM_ROOT / "train_qbm"
CHI_QBM_COMPARE = CHI_QBM_ROOT / "compare_models"
CHI_OLS_CEILING = (
    BOX_ROOT / "full_paper" / "figures" / "chi_ols" / "r2_ceiling" / "reliability_ceiling.csv"
)

N_NODES = 101
N_SEEDS = 100
N_CELLS = 243
CENTER_NODE = 50
DX_M = 2.0
ZCOLS = [f"{c}_z" for c in FACTORS]
FEATURES = ZCOLS + ["node_z"]
TAUS = [0.05, 0.25, 0.50, 0.75, 0.95]
TEST_SIZE = 0.25
VAL_FRAC = 0.20
SPLIT_SEED = 0
VAL_SEED = 1
LAG1_FAIL_THRESH = 0.2

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
    """Unique design-cell × node grid (no seed replication).

    Expects ``add_design_columns`` already applied so z-scored FEATURES match
    the fitted NGBoost models (z-scoring uses the full-table means/sds).
    """
    keys = list(FACTORS) + ["node"]
    keep = ["cell", "node", *FACTORS, *FEATURES]
    keep = [c for c in keep if c in df.columns]
    out = df.drop_duplicates(subset=keys, keep="first")[keep].copy()
    return out.reset_index(drop=True)


def fmt(x: float, digits: int = 4) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{digits}f}"


def load_ratios(path: Path = DATA_PATH) -> pd.DataFrame:
    cols = ["Vs1", "Height", "CoV", "rH", "aHV", "channel", "seed", *METRICS]
    with h5py.File(path, "r") as f:
        g = f["master"]
        df = pd.DataFrame({c: g[c][:] for c in cols})
    return df.rename(columns={"channel": "node"})


def add_design_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["cell"] = out.groupby(list(FACTORS), sort=False).ngroup()
    for c in FACTORS:
        mu = float(out[c].mean())
        sd = float(out[c].std(ddof=0))
        out[f"{c}_z"] = (out[c] - mu) / sd if sd > 0 else 0.0
    node = out["node"].to_numpy(dtype=float)
    nmu, nsd = float(node.mean()), float(node.std(ddof=0))
    out["node_z"] = (node - nmu) / nsd if nsd > 0 else 0.0
    out["x_m"] = (out["node"].to_numpy(dtype=float) - CENTER_NODE) * DX_M
    return out


def log_response(df: pd.DataFrame, metric: str) -> np.ndarray:
    chi = df[metric].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(chi) & (chi > 0), np.log(chi), np.nan)


def seed_grouped_split_indices(
    df: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    seed: int = SPLIT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(df, groups=df["seed"].to_numpy()))
    return tr, te


def load_or_make_split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Reuse chi_qbm seed_split.json seed lists when available."""
    split_path = CHI_QBM_TRAIN / "seed_split.json"
    if split_path.is_file():
        meta = json.loads(split_path.read_text(encoding="utf-8"))
        train_seeds = set(meta["train_seeds"])
        test_seeds = set(meta["test_seeds"])
        seeds = df["seed"].to_numpy()
        tr = np.flatnonzero(np.isin(seeds, list(train_seeds)))
        te = np.flatnonzero(np.isin(seeds, list(test_seeds)))
        if len(tr) and len(te):
            return tr, te
    return seed_grouped_split_indices(df)


def save_split(path: Path, tr: np.ndarray, te: np.ndarray, df: pd.DataFrame) -> None:
    payload = {
        "test_size": TEST_SIZE,
        "split_seed": SPLIT_SEED,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "train_seeds": sorted(int(s) for s in df.iloc[tr]["seed"].unique()),
        "test_seeds": sorted(int(s) for s in df.iloc[te]["seed"].unique()),
        "train_subsample_frac": TRAIN_SUBSAMPLE_FRAC,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    e = y_true - y_pred
    return float(np.mean(np.where(e >= 0, tau * e, (tau - 1) * e)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def lag1_pearson(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return float("nan")
    a, b = y[:-1], y[1:]
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    a, b = a[m], b[m]
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if den <= 0:
        return float("nan")
    return float(np.sum(a * b) / den)


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
