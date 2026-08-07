"""Shared helpers for chi_sr symbolic regression."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

_CODE = Path(__file__).resolve().parent.parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

from _shared import (  # noqa: E402,F401
    FEATURES,
    METRICS,
    TEST_SIZE,
    add_design_columns,
    load_or_make_split,
    load_ratios,
    log_response,
    pinball_loss,
    r2_score,
    rmse,
)
from config import figure_dir  # noqa: E402


def ngboost_models_dir() -> Path:
    path = figure_dir("chi_ngboost", "models")
    path.mkdir(parents=True, exist_ok=True)
    return path


def ngboost_surfaces_dir() -> Path:
    path = figure_dir("chi_ngboost", "surfaces")
    path.mkdir(parents=True, exist_ok=True)
    return path

SHAP_COMPARE = figure_dir("chi_shap", "shap_compare")
COLLAPSE_R2_THRESH = 0.05
CELL_SPLIT_SEED = 4
# Targets distilled from NGBoost surfaces (separate GP fits).
SR_TARGETS = ("mu", "log_sigma", "q05", "q50", "q95")
# Map SR target → SHAP shortlist target(s) for mains / interactions.
SHORTLIST_MAIN_SOURCES: dict[str, tuple[str, ...]] = {
    "mu": ("mu",),
    "q50": ("mu",),
    "log_sigma": ("log_sigma",),
    "q05": ("mu", "log_sigma"),
    "q95": ("mu", "log_sigma"),
}
SHORTLIST_INTERACTION_SOURCE: dict[str, str | None] = {
    "mu": "mu",
    "q50": "mu",
    "log_sigma": None,
    "q05": "mu",
    "q95": "mu",
}
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


def load_surface(metric: str) -> pd.DataFrame:
    path = ngboost_surfaces_dir() / f"ngboost_surface_{metric}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing surface: {path}. Run chi_ngboost/export_surfaces.py first."
        )
    return pd.read_csv(path)


def cell_grouped_split(
    df: pd.DataFrame,
    *,
    test_size: float = TEST_SIZE,
    seed: int = CELL_SPLIT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out a fraction of design cells for SR fidelity."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(df, groups=df["cell"].to_numpy()))
    return tr, te


def shortlist_features(short: pd.DataFrame, metric: str, target: str) -> list[str]:
    """Return base feature names (expand interactions into constituents).

    Quantile targets reuse SHAP shortlists: q50→mu; q05/q95→union(mu, log_sigma).
    """
    main_sources = SHORTLIST_MAIN_SOURCES.get(target, (target,))
    inter_source = SHORTLIST_INTERACTION_SOURCE.get(target, target)
    feats: list[str] = []
    for src in main_sources:
        sub = short[(short["metric"] == metric) & (short["target"] == src)]
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
    if inter_source is not None and inter_source not in main_sources:
        sub = short[
            (short["metric"] == metric)
            & (short["target"] == inter_source)
            & (short["kind"] == "interaction")
        ]
        for _, r in sub.iterrows():
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
    inter_source = SHORTLIST_INTERACTION_SOURCE.get(target, target)
    if inter_source is None:
        return X, colnames
    sub = short[
        (short["metric"] == metric)
        & (short["target"] == inter_source)
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


# gplearn protected division threshold
_DIV_EPS = 0.001
# Published formula_reported rounds y_mean/y_std with .6g; allow scale-aware slack.
INTEGRITY_MAX_ABS_TOL = 1e-4
INTEGRITY_REL_TOL = 1e-2  # vs |y_std|
INTEGRITY_SAMPLE_N = 2000
INTEGRITY_SAMPLE_SEED = 0


def integrity_abs_tol(y_std: float) -> float:
    return max(INTEGRITY_MAX_ABS_TOL, INTEGRITY_REL_TOL * abs(float(y_std)))


def ngboost_holdout_metrics_path() -> Path:
    return figure_dir("chi_ngboost", "train_ngboost") / "holdout_metrics.csv"


def parse_feature_list(features: str | list[str]) -> list[str]:
    if isinstance(features, list):
        return [str(f).strip() for f in features if str(f).strip()]
    return [p.strip() for p in str(features).split(",") if p.strip()]


def make_feature_dict(df: pd.DataFrame, colnames: list[str]) -> dict[str, np.ndarray]:
    """Map formula feature names (including ``a*b`` interactions) to arrays."""
    out: dict[str, np.ndarray] = {}
    n = len(df)
    for name in colnames:
        if name in df.columns:
            out[name] = df[name].to_numpy(dtype=float)
        elif "*" in name:
            a, b = name.split("*", 1)
            if a not in df.columns or b not in df.columns:
                raise KeyError(f"Cannot build interaction {name!r}: missing {a!r} or {b!r}")
            out[name] = df[a].to_numpy(dtype=float) * df[b].to_numpy(dtype=float)
        else:
            raise KeyError(f"Unknown formula feature {name!r}")
        if out[name].shape[0] != n:
            raise ValueError(f"Feature {name!r} length mismatch")
    return out


def design_matrix_from_colnames(df: pd.DataFrame, colnames: list[str]) -> np.ndarray:
    feat = make_feature_dict(df, colnames)
    return np.column_stack([feat[c] for c in colnames])


def _protected_div(a, b):
    """Match gplearn._operator.protected_division."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(b) > _DIV_EPS, np.divide(a, b), 1.0)


def eval_formula_reported(formula: str, feat: dict[str, np.ndarray]) -> np.ndarray:
    """Evaluate OLS or wrapped-GP ``formula_reported`` on feature arrays.

    Interaction tokens like ``CoV_z*aHV_z`` are aliased before ``eval`` so ``*``
    in names is not parsed as multiplication.
    """
    if not feat:
        raise ValueError("No features provided for formula evaluation")
    n = next(iter(feat.values())).shape[0]
    names = sorted(feat.keys(), key=len, reverse=True)
    expr = str(formula)
    local: dict = {
        "add": np.add,
        "sub": np.subtract,
        "mul": np.multiply,
        "div": _protected_div,
    }
    for i, name in enumerate(names):
        alias = f"_f{i}_"
        if name not in expr and name not in formula:
            # still bind for completeness
            pass
        expr = expr.replace(name, alias)
        local[alias] = np.asarray(feat[name], dtype=float)

    # Disallow anything except our aliases / ops / arithmetic
    if re.search(r"[^\w\s+\-*/().,eE]", expr):
        # allow only after stripping known safe chars; scientific notation ok
        cleaned = re.sub(r"[_a-zA-Z0-9\s+\-*/().,eE]", "", expr)
        if cleaned:
            raise ValueError(f"Unsafe formula tokens: {cleaned!r} in {expr[:120]!r}")

    try:
        val = eval(expr, {"__builtins__": {}}, local)  # noqa: S307 — restricted locals
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to evaluate formula: {exc}") from exc

    arr = np.asarray(val, dtype=float)
    if arr.ndim == 0:
        arr = np.full(n, float(arr))
    return arr.ravel()


def predict_from_pickle(
    pkl_path: Path,
    df: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Predict with saved GP (+ destandardize) or OLS coefs from train_sr pickle."""
    import joblib

    blob = joblib.load(pkl_path)
    colnames = list(blob["colnames"])
    X = design_matrix_from_colnames(df, colnames)
    source = blob.get("formula_source", "gp")
    if source == "ols" or blob.get("gp") is None:
        coef = np.asarray(blob["ols_coef"], dtype=float)
        A = np.column_stack([np.ones(len(X)), X])
        return (A @ coef).ravel(), colnames
    est = blob["gp"]
    zhat = np.asarray(est.predict(X), dtype=float).ravel()
    yhat = destandardize(zhat, float(blob["y_mean"]), float(blob["y_std"]))
    return yhat, colnames


def normal_nll(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-8)
    return float(np.mean(0.5 * np.log(2.0 * np.pi * sigma**2) + 0.5 * ((y - mu) / sigma) ** 2))
