"""Train Normal NGBoost on full-array χ ratios (Y = ln χ).

Features = z-scored factors + node_z. Seed-grouped holdout matches chi_qbm.
Training uses a stratified subsample of training rows for tractability;
holdout metrics use the full test set.

Writes models under figure_dir("chi_ngboost", "models") and CSV + summary.md
under figure_dir("chi_ngboost", "train_ngboost").
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    EARLY_STOPPING_ROUNDS,
    FEATURES,
    MAX_ESTIMATORS,
    METRICS,
    N_CELLS,
    N_NODES,
    N_SEEDS,
    TAUS,
    TRAIN_SUBSAMPLE_FRAC,
    TRAIN_SUBSAMPLE_SEED,
    VAL_FRAC,
    VAL_SEED,
    add_design_columns,
    load_or_make_split,
    load_ratios,
    log_response,
    mean_abs_lag1_residuals,
    models_dir,
    out_dir,
    pinball_loss,
    r2_score,
    rmse,
    save_split,
)

warnings.filterwarnings("ignore")

BASE_LEARNER = DecisionTreeRegressor(criterion="friedman_mse", max_depth=3, random_state=0)


def _finite_mask(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.isfinite(y) & np.all(np.isfinite(X), axis=1)


def _subsample_train(tr_idx: np.ndarray, groups: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Subsample training indices, stratified by seed group when possible."""
    if frac >= 1.0:
        return tr_idx
    rng = np.random.default_rng(seed)
    g = groups[tr_idx]
    keep = []
    for seed_id in np.unique(g):
        block = tr_idx[g == seed_id]
        n = max(1, int(round(len(block) * frac)))
        choose = rng.choice(block, size=min(n, len(block)), replace=False)
        keep.append(choose)
    return np.concatenate(keep)


def _ngb_params() -> dict:
    return dict(
        Dist=Normal,
        Score=LogScore,
        Base=BASE_LEARNER,
        n_estimators=MAX_ESTIMATORS,
        learning_rate=0.05,
        minibatch_frac=1.0,
        col_sample=1.0,
        verbose=False,
        random_state=0,
    )


def fit_one(X_fit, y_fit, X_val, y_val):
    model = NGBRegressor(**_ngb_params())
    model.fit(
        X_fit,
        y_fit,
        X_val=X_val,
        Y_val=y_val,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    n_trees = int(getattr(model, "best_val_loss_itr", None) or model.n_estimators)
    return model, n_trees


def predict_params(model: NGBRegressor, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dist = model.pred_dist(X)
    mu = np.asarray(dist.loc, dtype=float).ravel()
    # NGBoost Normal: scale is std; attribute name is .scale
    sigma = np.asarray(dist.scale, dtype=float).ravel()
    return mu, sigma


def normal_nll(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    sigma = np.maximum(sigma, 1e-8)
    return float(np.mean(0.5 * np.log(2 * np.pi * sigma**2) + 0.5 * ((y - mu) / sigma) ** 2))


def pi_coverage(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, alpha: float = 0.10) -> float:
    from scipy import stats

    z = stats.norm.ppf(1 - alpha / 2)
    lo, hi = mu - z * sigma, mu + z * sigma
    return float(np.mean((y >= lo) & (y <= hi)))


def main() -> None:
    out = out_dir("train_ngboost")
    mdir = models_dir()
    print("Loading join_master …")
    df = add_design_columns(load_ratios())
    tr, te = load_or_make_split(df)
    save_split(out / "seed_split.json", tr, te, df)

    X_all = df[FEATURES].to_numpy(dtype=float)
    groups = df["seed"].to_numpy()

    # Inner val from training seeds
    gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=VAL_SEED)
    fit_rel, val_rel = next(gss.split(tr, groups=groups[tr]))
    fit_idx, val_idx = tr[fit_rel], tr[val_rel]
    fit_sub = _subsample_train(fit_idx, groups, TRAIN_SUBSAMPLE_FRAC, TRAIN_SUBSAMPLE_SEED)
    val_sub = _subsample_train(
        val_idx, groups, min(1.0, TRAIN_SUBSAMPLE_FRAC * 2), TRAIN_SUBSAMPLE_SEED + 1
    )

    rows = []
    lag_rows = []
    meta = {
        "features": FEATURES,
        "max_estimators": MAX_ESTIMATORS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "train_subsample_frac": TRAIN_SUBSAMPLE_FRAC,
        "n_train_full": int(len(tr)),
        "n_test": int(len(te)),
        "n_fit_sub": int(len(fit_sub)),
        "n_val_sub": int(len(val_sub)),
        "n_nodes": N_NODES,
        "n_seeds": N_SEEDS,
        "n_cells": N_CELLS,
        "metrics": {},
    }

    for metric in METRICS:
        print(f"=== NGBoost {metric} ===")
        y = log_response(df, metric)
        m_fit = _finite_mask(y[fit_sub], X_all[fit_sub])
        m_val = _finite_mask(y[val_sub], X_all[val_sub])
        m_te = _finite_mask(y[te], X_all[te])

        t0 = time.perf_counter()
        model, n_trees = fit_one(
            X_all[fit_sub][m_fit],
            y[fit_sub][m_fit],
            X_all[val_sub][m_val],
            y[val_sub][m_val],
        )
        fit_s = time.perf_counter() - t0
        path = mdir / f"ngboost_{metric}.pkl"
        joblib.dump(model, path)
        print(f"  saved {path.name}  trees={n_trees}  {fit_s:.1f}s")

        mu_te, sig_te = predict_params(model, X_all[te][m_te])
        y_te = y[te][m_te]
        nll = normal_nll(y_te, mu_te, sig_te)
        r2 = r2_score(y_te, mu_te)
        rmse_te = rmse(y_te, mu_te)
        cov90 = pi_coverage(y_te, mu_te, sig_te, alpha=0.10)

        pinballs = {}
        for tau in TAUS:
            from scipy import stats

            q = mu_te + sig_te * stats.norm.ppf(tau)
            pinballs[f"pinball_q{int(tau * 100):02d}"] = pinball_loss(y_te, q, tau)

        # Residual lag-1 on a sample of test blocks
        resid_full = np.full(len(df), np.nan)
        mu_all_te, _ = predict_params(model, X_all[te])
        resid_full[te] = y[te] - mu_all_te
        lag = mean_abs_lag1_residuals(df.iloc[te].reset_index(drop=True), resid_full[te], seed=0)
        lag_rows.append({"metric": metric, **lag})

        row = {
            "metric": metric,
            "n_trees": n_trees,
            "fit_seconds": fit_s,
            "n_fit": int(m_fit.sum()),
            "n_test": int(m_te.sum()),
            "nll": nll,
            "r2_mean": r2,
            "rmse": rmse_te,
            "pi90_coverage": cov90,
            **pinballs,
            "model_path": str(path),
        }
        rows.append(row)
        meta["metrics"][metric] = {
            "n_trees": n_trees,
            "fit_seconds": fit_s,
            "model": path.name,
        }

    hold = pd.DataFrame(rows)
    hold.to_csv(out / "holdout_metrics.csv", index=False)
    pd.DataFrame(lag_rows).to_csv(out / "residual_spatial_acf.csv", index=False)
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    lines = [
        "# NGBoost training summary",
        "",
        "## Definitions",
        "",
        r"- Response: \(Y = \ln\chi\) on the full array (\(N_x=101\) nodes × \(N_s=100\) seeds × 243 cells).",
        r"- Model: Normal NGBoost \(Y\mid\mathbf{x}\sim\mathcal{N}(\mu(\mathbf{x}),\sigma^2(\mathbf{x}))\) with features = z-scored design factors + `node_z`.",
        r"- Score: logarithmic score (natural-gradient boosting).",
        r"- Holdout: seed-grouped 25% test (same seeds as `chi_qbm` when available).",
        f"- Fit subsample: {TRAIN_SUBSAMPLE_FRAC:.0%} of training rows (by seed); full test evaluated.",
        r"- NLL: mean Gaussian negative log-likelihood on test.",
        r"- \(R^2\) / RMSE: mean prediction \(\hat\mu\) vs \(Y\).",
        r"- PI90 coverage: fraction of test \(Y\) in \(\hat\mu\pm z_{0.95}\hat\sigma\).",
        r"- Pinball: Normal quantile \(\hat\mu + \hat\sigma\Phi^{-1}(\tau)\).",
        r"- Residual lag-1: mean \(|\rho_1|\) of \(Y-\hat\mu\) along nodes within sampled (cell, seed) blocks.",
        "",
        "## Output files",
        "",
        "| File | Content |",
        "|------|---------|",
        r"| `holdout_metrics.csv` | NLL, \(R^2\), RMSE, PI90, pinball, trees |",
        "| `residual_spatial_acf.csv` | residual spatial lag-1 diagnostics |",
        "| `train_meta.json` / `seed_split.json` | fit metadata and split |",
        "| `../models/ngboost_*.pkl` | fitted NGBoost models |",
        "",
        "## Holdout metrics",
        "",
        hold.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Residual spatial structure",
        "",
        pd.DataFrame(lag_rows).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Conclusions",
        "",
        "- NGBoost provides a full parametric predictive distribution at each (cell, node), enabling likelihood-based evaluation of aleatoric variability.",
        "- Because residuals remain spatially correlated along the profile, these distributions are conditional emulators validated via held-out seeds and lag-1 diagnostics, not joint spatial process models.",
        rf"- Training used a {TRAIN_SUBSAMPLE_FRAC:.0%} subsample for tractability on ~2.45M rows; interpret absolute NLL/\(R^2\) with that caveat, but the seed holdout protocol matches `chi_qbm`.",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
