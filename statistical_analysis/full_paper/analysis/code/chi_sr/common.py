"""Shared helpers for chi_sr symbolic regression."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Load chi_ngboost/common by path to avoid shadowing this package's `common`.
_ngb_common_path = Path(__file__).resolve().parent.parent / "chi_ngboost" / "common.py"
_spec = importlib.util.spec_from_file_location("chi_ngboost_common", _ngb_common_path)
assert _spec is not None and _spec.loader is not None
_ngb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ngb)

FEATURES = _ngb.FEATURES
METRICS = _ngb.METRICS
add_design_columns = _ngb.add_design_columns
load_or_make_split = _ngb.load_or_make_split
load_ratios = _ngb.load_ratios
ngboost_models_dir = _ngb.models_dir
r2_score = _ngb.r2_score
rmse = _ngb.rmse

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import figure_dir  # noqa: E402

SHAP_COMPARE = figure_dir("chi_shap", "shap_compare")
SR_SAMPLE_N = 5000
SR_SAMPLE_SEED = 4
COLLAPSE_R2_THRESH = 0.05
GP_PARAMS = dict(
    population_size=1000,
    generations=40,
    tournament_size=20,
    stopping_criteria=0.01,
    const_range=(-3.0, 3.0),
    init_depth=(2, 5),
    p_crossover=0.7,
    p_subtree_mutation=0.1,
    p_hoist_mutation=0.05,
    p_point_mutation=0.1,
    max_samples=0.9,
    verbose=0,
    parsimony_coefficient=1e-4,
    random_state=0,
    n_jobs=1,
    metric="mse",
)


def out_dir(stem: str) -> Path:
    return figure_dir("chi_sr", stem)


def load_shortlist() -> pd.DataFrame:
    path = SHAP_COMPARE / "feature_shortlist.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing shortlist: {path}. Run shap_compare.py first.")
    return pd.read_csv(path)


def shortlist_features(short: pd.DataFrame, metric: str, target: str) -> list[str]:
    """Return base feature names (expand interactions into constituents)."""
    sub = short[(short["metric"] == metric) & (short["target"] == target)]
    feats: list[str] = []
    for _, r in sub.iterrows():
        if r["kind"] == "main":
            feats.append(str(r["feature"]))
        else:
            for key in ("feature_i", "feature_j"):
                if (
                    key in r
                    and isinstance(r[key], str)
                    and r[key]
                    and not (isinstance(r[key], float) and np.isnan(r[key]))
                ):
                    feats.append(str(r[key]))
            if "*" in str(r["feature"]):
                a, b = str(r["feature"]).split("*", 1)
                feats.extend([a, b])
    out_feats: list[str] = []
    for f in feats:
        if f in FEATURES and f not in out_feats:
            out_feats.append(f)
    if not out_feats:
        out_feats = list(FEATURES)
    return out_feats


def build_design_matrix(
    df: pd.DataFrame,
    feat_names: list[str],
    short: pd.DataFrame,
    metric: str,
    target: str,
):
    """X columns = shortlisted mains + explicit interaction products from shortlist."""
    X = df[feat_names].to_numpy(dtype=float)
    colnames = list(feat_names)
    sub = short[
        (short["metric"] == metric)
        & (short["target"] == target)
        & (short["kind"] == "interaction")
    ]
    extras = []
    for _, r in sub.iterrows():
        a = r.get("feature_i")
        b = r.get("feature_j")
        if not (isinstance(a, str) and isinstance(b, str) and a and b):
            feat = str(r["feature"])
            if "*" not in feat:
                continue
            a, b = feat.split("*", 1)
        if a in df.columns and b in df.columns:
            extras.append(
                (df[a].to_numpy(dtype=float) * df[b].to_numpy(dtype=float)).reshape(-1, 1)
            )
            colnames.append(f"{a}*{b}")
    if extras:
        X = np.hstack([X, *extras])
    return X, colnames


def standardize_y(y_train: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Return (z_train, mean, std) with std floored for stability."""
    y_train = np.asarray(y_train, dtype=float)
    mu = float(np.mean(y_train))
    sd = float(np.std(y_train))
    if not np.isfinite(sd) or sd < 1e-12:
        sd = 1.0
    z = (y_train - mu) / sd
    return z, mu, sd


def destandardize(z: np.ndarray, y_mean: float, y_std: float) -> np.ndarray:
    return np.asarray(z, dtype=float) * y_std + y_mean


def ols_shortlist_fit(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    colnames: list[str],
) -> dict:
    """OLS with intercept on shortlist design; return coefs, formula, predictions, R²."""
    A_tr = np.column_stack([np.ones(len(X_tr)), X_tr])
    A_te = np.column_stack([np.ones(len(X_te)), X_te])
    coef, *_ = np.linalg.lstsq(A_tr, y_tr, rcond=None)
    yhat_tr = A_tr @ coef
    yhat_te = A_te @ coef
    terms = [f"({coef[0]:.6g})"]
    for c, name in zip(coef[1:], colnames):
        terms.append(f"({c:.6g})*{name}")
    formula = " + ".join(terms)
    return {
        "coef": coef,
        "formula": formula,
        "yhat_tr": yhat_tr,
        "yhat_te": yhat_te,
        "r2_train": r2_score(y_tr, yhat_tr),
        "r2_test": r2_score(y_te, yhat_te),
        "rmse_train": rmse(y_tr, yhat_tr),
        "rmse_test": rmse(y_te, yhat_te),
    }


def is_collapsed_program(formula: str, program_length: int, r2_test: float) -> bool:
    """True if GP failed to recover useful structure."""
    if not np.isfinite(r2_test) or r2_test < COLLAPSE_R2_THRESH:
        return True
    if program_length <= 2:
        return True
    # Pure numeric constant (no feature tokens)
    s = formula.strip()
    if re.fullmatch(r"[-+]?\d*\.?\d+(e[-+]?\d+)?", s, flags=re.I):
        return True
    return False


def wrap_standardized_formula(formula_z: str, y_mean: float, y_std: float) -> str:
    """Map GP formula on z-scale back to original y: y = mean + std * f_z(x)."""
    return f"({y_mean:.6g}) + ({y_std:.6g})*({formula_z})"
