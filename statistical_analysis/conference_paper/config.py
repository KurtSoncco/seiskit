"""Centralized configuration for center-recorder conference-paper analysis.

Scope
-----
All analyses in this package use the **center recorder** (channel 50) of the
Monte Carlo peak-ratio dataset. Full-array / 101-recorder workflows are out of
scope and have been removed.

Scripts import paths, constants, factor styles, target metadata, and loaders
from here instead of duplicating HDF5 / model / color logic.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd

from seiskit.plot_config.paths import set_results_root

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BOX_ROOT = Path("/mnt/box/GIG Lab - UC Berkeley/Projects/Statistical Analysis/complete")
DATA_PATH = BOX_ROOT / "peak_analysis" / "tf_peak_ratios_mode0.h5"

CONF_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = CONF_ROOT  # local results/<topic>/{plots,data}/
MODELS_DIR = CONF_ROOT / "models"
CACHE_DIR = CONF_ROOT / ".cache"

# Route seiskit.plot_config.result_path() to this package root.
set_results_root(RESULTS_ROOT)

# ---------------------------------------------------------------------------
# Design constants (center recorder)
# ---------------------------------------------------------------------------
CENTER_CHANNEL = 50
FACTORS = ["Vs1", "Height", "CoV", "rH", "aHV"]
RAW_TARGETS = ["f_ratio", "abs_TF_ratio"]
MODEL_TARGETS = ["abs_TF_ratio", "log_abs", "f_ratio"]
PRIMARY_AMPLITUDE_TARGET = "log_abs"  # primary SHAP / QBM amplitude view
COLS = FACTORS + ["channel", "seed"] + RAW_TARGETS

N_CELLS = 243  # 3^5 factorial design
N_CHANNELS = 101  # recorders 0–100 in the source HDF5
N_SEEDS = 100  # random-field seeds per cell
N_ROWS_FULL = N_CELLS * N_CHANNELS * N_SEEDS  # 2,454,300
N_ROWS_CENTER = N_CELLS * N_SEEDS  # 24,300

DEFAULT_SPLIT_BY = "seed"  # seed-grouped hold-out is the package default
DEFAULT_TAUS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
FIG_DPI = 150

# ---------------------------------------------------------------------------
# Paul Tol Bright palette (factor + reference colors)
# ---------------------------------------------------------------------------
# Factor order matches FACTORS. Purple/Gray reserved for reference overlays.
TOL_BRIGHT = {
    "blue": "#4477AA",
    "red": "#EE6677",
    "green": "#228833",
    "yellow": "#CCBB44",
    "cyan": "#66CCEE",
    "purple": "#AA3377",
    "gray": "#BBBBBB",
}

FACTOR_COLORS = {
    "Vs1": TOL_BRIGHT["blue"],
    "Height": TOL_BRIGHT["red"],
    "CoV": TOL_BRIGHT["green"],
    "rH": TOL_BRIGHT["yellow"],
    "aHV": TOL_BRIGHT["cyan"],
}

FACTOR_MARKERS = {
    "Vs1": "o",
    "Height": "v",
    "CoV": "s",
    "rH": "d",
    "aHV": "x",
}

FACTOR_PLOT = {f: {"color": FACTOR_COLORS[f], "marker": FACTOR_MARKERS[f]} for f in FACTORS}

# Target colors: amplitude family uses red/cyan; frequency uses blue; gray = ref.
TARGET_COLORS = {
    "abs_TF_ratio": TOL_BRIGHT["red"],
    "log_abs": TOL_BRIGHT["cyan"],
    "f_ratio": TOL_BRIGHT["blue"],
}

REF_COLOR = TOL_BRIGHT["gray"]
COMPARE_COLOR = TOL_BRIGHT["purple"]

# Display labels (natural log is explicit; auto-format uses LABEL_MAP keys)
TARGET_LABELS = {
    "abs_TF_ratio": r"$|TF|_0^N$",
    "log_abs": r"$\ln(|TF|_0^N)$",
    "f_ratio": r"$f_0^N$",
}


def factor_color(name: str) -> str:
    """Return Paul Tol Bright color for a factor (strips ``_z`` suffix)."""
    key = name[:-2] if name.endswith("_z") else name
    return FACTOR_COLORS[key]


def target_color(name: str) -> str:
    """Return the canonical color for a modeled target."""
    return TARGET_COLORS[name]


def target_label(name: str) -> str:
    """Return the display label for a modeled target."""
    return TARGET_LABELS[name]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
_MASTER_CACHE: pd.DataFrame | None = None
_CENTER_CACHE: pd.DataFrame | None = None


def load_master(*, cached: bool = True) -> pd.DataFrame:
    """Load the full HDF5 master table (used only to derive center recorder).

    Adds ``log_abs = ln(abs_TF_ratio)`` (natural logarithm).
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
    """Load center-recorder rows only (24,300 = 243 cells × 100 seeds).

    Drops the ``channel`` column (always 50). Includes raw amplitude,
    natural-log amplitude (``log_abs``), and frequency ratio.
    """
    global _CENTER_CACHE
    if cached and _CENTER_CACHE is not None:
        return _CENTER_CACHE

    df = load_master(cached=cached)
    out = df[df["channel"] == CENTER_CHANNEL].drop(columns=["channel"]).reset_index(drop=True)
    if cached:
        _CENTER_CACHE = out
    return out


# ---------------------------------------------------------------------------
# Model loading (split-specific filenames)
# ---------------------------------------------------------------------------
# Canonical naming so seed- and cell-grouped fits cannot overwrite each other:
#   lgbm_mean_{target}_{seed|cell}.pkl
#   lgbm_q{tau:02d}_{target}_{seed|cell}.pkl


def _validate_split(split_by: str) -> str:
    if split_by not in {"seed", "cell"}:
        raise ValueError("split_by must be 'seed' or 'cell'")
    return split_by


def mean_model_stem(target: str, *, split_by: str = DEFAULT_SPLIT_BY) -> str:
    """Return stem for a mean-model pickle (no ``.pkl``)."""
    return f"lgbm_mean_{target}_{_validate_split(split_by)}"


def quantile_model_stem(
    target: str,
    tau: float,
    *,
    split_by: str = DEFAULT_SPLIT_BY,
) -> str:
    """Return stem for a quantile-model pickle (no ``.pkl``)."""
    return f"lgbm_q{int(tau * 100):02d}_{target}_{_validate_split(split_by)}"


def model_path(stem: str) -> Path:
    """Resolve a model stem to ``models/<stem>.pkl``."""
    return MODELS_DIR / f"{stem}.pkl"


def load_model(name: str):
    """Load a pre-trained model from ``models/`` by stem name."""
    path = model_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def load_mean_models(
    targets: list[str] | None = None,
    *,
    split_by: str = DEFAULT_SPLIT_BY,
) -> dict:
    """Load pre-trained mean LightGBM models for the requested split."""
    if targets is None:
        targets = list(MODEL_TARGETS)
    out = {}
    for tgt in targets:
        path = model_path(mean_model_stem(tgt, split_by=split_by))
        if path.exists():
            out[tgt] = joblib.load(path)
    return out


def load_quantile_models(
    taus: list[float] | None = None,
    targets: list[str] | None = None,
    *,
    split_by: str = DEFAULT_SPLIT_BY,
) -> dict[str, dict]:
    """Load pre-trained quantile LightGBM models for the requested split.

    Returns ``{target: {tau: model}}`` for models that exist on disk.
    """
    if taus is None:
        taus = list(DEFAULT_TAUS)
    if targets is None:
        targets = list(MODEL_TARGETS)
    result: dict[str, dict] = {}
    for tgt in targets:
        result[tgt] = {}
        for tau in taus:
            path = model_path(quantile_model_stem(tgt, tau, split_by=split_by))
            if path.exists():
                result[tgt][tau] = joblib.load(path)
    return result


# ---------------------------------------------------------------------------
# SHAP value caching
# ---------------------------------------------------------------------------


def cached_shap(key: str, compute_fn, *, force: bool = False) -> np.ndarray:
    """Return SHAP values from cache or compute and store them."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.npy"
    if not force and path.exists():
        return np.load(path)
    vals = compute_fn()
    np.save(path, vals)
    return vals


def grouped_split(
    df: pd.DataFrame,
    *,
    split_by: str = DEFAULT_SPLIT_BY,
    test_size: float = 0.25,
    seed: int = 0,
):
    """Return (train_idx, test_idx) grouped by seed or design cell.

    Parameters
    ----------
    split_by:
        ``"seed"`` (default) — hold out unseen random-field seeds.
        ``"cell"`` — hold out entirely unseen design cells.
    """
    from sklearn.model_selection import GroupShuffleSplit

    if split_by == "cell":
        groups = df.groupby(FACTORS).ngroup().values
    elif split_by == "seed":
        groups = df["seed"].values
    else:
        raise ValueError("split_by must be 'seed' or 'cell'")

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(df, groups=groups))
    return tr, te


def seed_grouped_split(df: pd.DataFrame, test_size: float = 0.25, seed: int = 0):
    """Return (train_idx, test_idx) with seeds as groups (package default)."""
    return grouped_split(df, split_by="seed", test_size=test_size, seed=seed)
