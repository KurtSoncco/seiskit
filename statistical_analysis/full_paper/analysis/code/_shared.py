"""Shared loaders, constants, and metrics for full-paper χ analysis packages.

Import via ``sys.path`` root at ``full_paper/`` then::

    from analysis.code._shared import load_ratios, log_response, ...

Or from a package ``common.py`` that inserts ``full_paper`` and this directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

_FULL_PAPER = Path(__file__).resolve().parents[2]
if str(_FULL_PAPER) not in sys.path:
    sys.path.insert(0, str(_FULL_PAPER))

from config import BOX_ROOT, FACTORS, METRICS, figure_dir  # noqa: E402

DATA_PATH = BOX_ROOT / "peak_analysis" / "join_master.h5"
ACF_FIT_PATH = (
    BOX_ROOT / "full_paper" / "figures" / "chi_spatial" / "spatial_acf" / "acf_fit_params.csv"
)
CHI_OLS_STAGE1 = BOX_ROOT / "full_paper" / "figures" / "chi_ols" / "stage1_mean_ols"
CHI_OLS_CEILING = (
    BOX_ROOT / "full_paper" / "figures" / "chi_ols" / "r2_ceiling" / "reliability_ceiling.csv"
)
CHI_QBM_ROOT = BOX_ROOT / "full_paper" / "figures" / "chi_qbm"
CHI_QBM_MODELS = CHI_QBM_ROOT / "models"
CHI_QBM_TRAIN = CHI_QBM_ROOT / "train_qbm"
CHI_QBM_COMPARE = CHI_QBM_ROOT / "compare_models"

N_NODES = 101
N_SEEDS = 100
N_CELLS = 243
CENTER_NODE = 50
DX_M = 2.0

ZCOLS = [f"{c}_z" for c in FACTORS]
FEATURES = ZCOLS + ["node_z"]
QBM_FEATURES = FEATURES  # alias; prefer FEATURES in new code
TAUS = [0.05, 0.25, 0.50, 0.75, 0.95]
TEST_SIZE = 0.25
VAL_FRAC = 0.20
SPLIT_SEED = 0
VAL_SEED = 1
LAG1_FAIL_THRESH = 0.2


def fmt(x: float, digits: int = 4) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{digits}f}"


def load_ratios(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load joined ratio table; rename channel → node."""
    cols = ["Vs1", "Height", "CoV", "rH", "aHV", "channel", "seed", *METRICS]
    with h5py.File(path, "r") as f:
        g = f["master"]
        df = pd.DataFrame({c: g[c][:] for c in cols})
    return df.rename(columns={"channel": "node"})


def add_design_columns(
    df: pd.DataFrame,
    *,
    include_node_z: bool = True,
) -> pd.DataFrame:
    """Cell id, z-scored factors; optionally node_z and x_m."""
    out = df.copy()
    out["cell"] = out.groupby(list(FACTORS), sort=False).ngroup()
    for c in FACTORS:
        mu = float(out[c].mean())
        sd = float(out[c].std(ddof=0))
        out[f"{c}_z"] = (out[c] - mu) / sd if sd > 0 else 0.0
    if include_node_z:
        node = out["node"].to_numpy(dtype=float)
        nmu, nsd = float(node.mean()), float(node.std(ddof=0))
        out["node_z"] = (node - nmu) / nsd if nsd > 0 else 0.0
        out["x_m"] = (out["node"].to_numpy(dtype=float) - CENTER_NODE) * DX_M
    return out


def log_response(df: pd.DataFrame, metric: str) -> np.ndarray:
    """Return ln(χ); non-positive / non-finite → NaN."""
    chi = df[metric].to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(chi) & (chi > 0), np.log(chi), np.nan)


def seed_grouped_split_indices(
    df: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    seed: int = SPLIT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) with groups = seed."""
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


def save_split(
    path: Path,
    tr: np.ndarray,
    te: np.ndarray,
    df: pd.DataFrame,
    *,
    extra: dict | None = None,
) -> None:
    payload = {
        "test_size": TEST_SIZE,
        "split_seed": SPLIT_SEED,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "train_seeds": sorted(int(s) for s in df.iloc[tr]["seed"].unique()),
        "test_seeds": sorted(int(s) for s in df.iloc[te]["seed"].unique()),
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    e = y_true - y_pred
    return float(np.mean(np.where(e >= 0, tau * e, (tau - 1) * e)))


def koenker_pseudo_r2(
    y_true: np.ndarray, y_pred: np.ndarray, tau: float, y_null: float
) -> float:
    v_model = pinball_loss(y_true, y_pred, tau)
    v_null = pinball_loss(y_true, np.full_like(y_true, y_null), tau)
    if v_null <= 0:
        return float("nan")
    return 1.0 - v_model / v_null


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
    """Lag-1 Pearson correlation along a 1-D node series."""
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


def formula_rhs_main() -> str:
    return " + ".join(ZCOLS)


def code_dir() -> Path:
    return Path(__file__).resolve().parent


def ensure_code_on_path() -> Path:
    """Insert ``analysis/code`` on sys.path so ``import _shared`` works from scripts."""
    d = code_dir()
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
    return d


def n_jobs(default: int = -1) -> int:
    """Parallel workers from ``FULL_PAPER_N_JOBS`` (default: all cores)."""
    import os

    raw = os.environ.get("FULL_PAPER_N_JOBS", "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def parallel_map(fn, items, *, desc: str, n_jobs_: int | None = None):
    """``joblib.Parallel`` over *items* with a ``tqdm`` bar.

    Returns a list of ``fn(item)`` results in input order.
    """
    from joblib import Parallel, delayed
    from tqdm import tqdm

    items = list(items)
    jobs = n_jobs() if n_jobs_ is None else n_jobs_
    if not items:
        return []
    if jobs == 1 or len(items) == 1:
        return [fn(item) for item in tqdm(items, desc=desc)]
    return Parallel(n_jobs=jobs, prefer="threads")(
        delayed(fn)(item) for item in tqdm(items, desc=desc)
    )
