"""Centralized configuration and data loading for the conference paper analysis.

All scripts import paths, constants, and data loaders from here instead of
duplicating HDF5 loading logic.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BOX_ROOT = Path("/mnt/box/GIG Lab - UC Berkeley/Projects/Statistical Analysis/complete")
DATA_PATH = BOX_ROOT / "peak_analysis" / "tf_peak_ratios_mode0.h5"
RESULTS_ROOT = BOX_ROOT

CONF_ROOT = Path(__file__).resolve().parent
MODELS_DIR = CONF_ROOT / "models"
CACHE_DIR = CONF_ROOT / ".cache"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FACTORS = ["Vs1", "Height", "CoV", "rH", "aHV"]
TARGETS = ["f_ratio", "abs_TF_ratio"]
COLS = FACTORS + ["channel", "seed"] + TARGETS

N_CELLS = 243  # 3^5 factorial design
N_CHANNELS = 101  # recorders 0–100
N_SEEDS = 100  # random-field seeds per cell
N_ROWS = N_CELLS * N_CHANNELS * N_SEEDS  # 2,454,300


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
_MASTER_CACHE: pd.DataFrame | None = None


def load_master(*, cached: bool = True) -> pd.DataFrame:
    """Load the full 2,454,300-row master dataset from HDF5.

    Parameters
    ----------
    cached:
        If ``True`` (default), cache the DataFrame in memory so repeated
        calls within the same process don't re-read the file.
    """
    global _MASTER_CACHE
    if cached and _MASTER_CACHE is not None:
        return _MASTER_CACHE

    with h5py.File(DATA_PATH, "r") as f:
        g = f["master"]
        df = pd.DataFrame({c: g[c][:] for c in COLS})

    df["log_abs"] = np.log(df["abs_TF_ratio"])

    if cached:
        _MASTER_CACHE = df
    return df


def load_channel50(*, cached: bool = True) -> pd.DataFrame:
    """Load and filter to the center recorder only (24,300 rows)."""
    df = load_master(cached=cached)
    return df[df["channel"] == 50].drop(columns=["channel"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(name: str):
    """Load a pre-trained model from the models directory.

    Accepted *name* formats (all resolve to pkl files)::

        "lgbm_mean_log_abs"
        "lgbm_q05_f_ratio"
    """
    path = MODELS_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def load_mean_models() -> dict:
    """Load both pre-trained mean LightGBM models."""
    return {
        "log_abs": load_model("lgbm_mean_log_abs"),
        "f_ratio": load_model("lgbm_mean_f_ratio"),
    }


def load_quantile_models(taus: list[float] | None = None) -> dict[str, dict]:
    """Load pre-trained quantile LightGBM models for both targets.

    Returns ``{target: {tau: model}}``.
    """
    if taus is None:
        taus = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    result: dict[str, dict] = {}
    for tgt in ["log_abs", "f_ratio"]:
        result[tgt] = {}
        for tau in taus:
            tag = f"lgbm_q{int(tau * 100):02d}_{tgt}"
            path = MODELS_DIR / f"{tag}.pkl"
            if path.exists():
                result[tgt][tau] = joblib.load(path)
    return result


# ---------------------------------------------------------------------------
# SHAP value caching — avoid recomputing expensive TreeSHAP every run
# ---------------------------------------------------------------------------


def cached_shap(key: str, compute_fn, *, force: bool = False) -> np.ndarray:
    """Return SHAP values from cache or compute and store them.

    Parameters
    ----------
    key:
        Unique identifier (e.g. ``"shap_log_abs_ch50_te"``).
    compute_fn:
        Zero-argument callable that returns an ndarray of SHAP values.
    force:
        Recompute even if a cache file exists.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.npy"
    if not force and path.exists():
        return np.load(path)
    vals = compute_fn()
    np.save(path, vals)
    return vals


def seed_grouped_split(df: pd.DataFrame, test_size: float = 0.25, seed: int = 0):
    """Return (train_idx, test_idx) with seeds as groups."""
    from sklearn.model_selection import GroupShuffleSplit

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(df, groups=df["seed"].values))
    return tr, te
