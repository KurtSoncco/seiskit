"""Shared loaders, paths, and cross-package imports for chi_joint.

Sibling packages (``chi_ngboost``, ``chi_shap``) each ship a top-level
``common`` module and import it as ``from common import ...``. To reuse their
functions without the two ``common`` modules shadowing each other, they are
loaded here with :func:`load_sibling`, which isolates ``sys.modules['common']``
during the import.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_CODE = Path(__file__).resolve().parent.parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from _shared import (  # noqa: E402,F401
    DX_M,
    FACTORS,
    FEATURES,
    N_CELLS,
    N_NODES,
    N_SEEDS,
    ZCOLS,
    add_design_columns,
    fmt,
    load_or_make_split,
    load_ratios,
    log_response,
    parallel_map,
    r2_score,
)
from config import figure_dir  # noqa: E402

METRIC = "f_ratio"
N_OOF_FOLDS = 5
OOF_SEED = 11
H_FIT_MAX_M = 100.0  # half-aperture; derived distances beyond this are censored


def out_dir(stem: str) -> Path:
    return figure_dir("chi_joint", stem)


def models_dir() -> Path:
    return figure_dir("chi_joint", "models")


def residuals_path(metric: str = METRIC) -> Path:
    return out_dir("oof_residuals") / f"residual_cubes_{metric}.npz"


def cov_cell_path(metric: str = METRIC) -> Path:
    return out_dir("fit_joint_cov") / f"cov_params_cell_{metric}.csv"


def cov_global_path(metric: str = METRIC) -> Path:
    return out_dir("fit_joint_cov") / f"cov_params_global_{metric}.csv"


def load_sibling(package: str, module: str):
    """Import ``<package>/<module>.py`` with its own ``common`` module."""
    pkg_dir = _CODE / package
    saved = sys.modules.pop("common", None)
    sys.path.insert(0, str(pkg_dir))
    try:
        name = f"_chi_joint_{package}_{module}"
        spec = importlib.util.spec_from_file_location(name, pkg_dir / f"{module}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(pkg_dir))
        sys.modules.pop("common", None)
        if saved is not None:
            sys.modules["common"] = saved
    return mod


def cell_design_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per cell: raw factors + z-scored factors (sorted by cell)."""
    cols = ["cell", *FACTORS, *ZCOLS]
    return df.drop_duplicates("cell")[cols].sort_values("cell").reset_index(drop=True)


def response_cube(df: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (Y[cell, seed_idx, node], seeds) with seeds sorted ascending."""
    y = log_response(df, metric)
    seeds = np.sort(df["seed"].unique())
    seed_pos = np.searchsorted(seeds, df["seed"].to_numpy())
    n_cells = int(df["cell"].max()) + 1
    Y = np.full((n_cells, seeds.size, N_NODES), np.nan)
    Y[df["cell"].to_numpy(), seed_pos, df["node"].to_numpy()] = y
    return Y, seeds


def grid_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unique (cell, node) grid → (X[FEATURES], cell, node)."""
    g = df.drop_duplicates(["cell", "node"])[["cell", "node", *FEATURES]]
    return (
        g[FEATURES].to_numpy(dtype=float),
        g["cell"].to_numpy(),
        g["node"].to_numpy(),
    )


def params_to_grid(
    values: np.ndarray, cell: np.ndarray, node: np.ndarray, n_cells: int
) -> np.ndarray:
    out = np.full((n_cells, N_NODES), np.nan)
    out[cell, node] = values
    return out
