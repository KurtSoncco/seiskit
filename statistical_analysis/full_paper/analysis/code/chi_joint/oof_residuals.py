"""Out-of-fold standardized NGBoost residual cubes for the joint layer.

Training seeds (the chi_qbm/chi_ngboost seed split) are divided into
N_OOF_FOLDS seed groups. For each fold, a Normal NGBoost with the same
settings as ``chi_ngboost/train_ngboost.py`` is refit on the other training
seeds and predicts μ, σ for the held-out fold, so every training-seed profile
gets residuals from a model that never saw that seed.

Test-seed residuals use the saved full model ``chi_ngboost/models/ngboost_<metric>.pkl``.

Writes ``residual_cubes_<metric>.npz`` + ``summary.md`` under
``figure_dir("chi_joint", "oof_residuals")``.
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
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FEATURES,
    METRIC,
    N_OOF_FOLDS,
    OOF_SEED,
    add_design_columns,
    grid_features,
    load_or_make_split,
    load_ratios,
    load_sibling,
    log_response,
    out_dir,
    params_to_grid,
    residuals_path,
    response_cube,
)

warnings.filterwarnings("ignore")

tng = load_sibling("chi_ngboost", "train_ngboost")
ngb_common = load_sibling("chi_ngboost", "common")


def _grid_mu_sigma(model, X_grid, cell, node, n_cells):
    mu, sig = tng.predict_params(model, X_grid)
    return params_to_grid(mu, cell, node, n_cells), params_to_grid(sig, cell, node, n_cells)


def main(metric: str = METRIC) -> None:
    out = out_dir("oof_residuals")
    print("Loading join_master …")
    df = add_design_columns(load_ratios())
    tr, te = load_or_make_split(df)
    y = log_response(df, metric)
    X_all = df[FEATURES].to_numpy(dtype=float)
    groups = df["seed"].to_numpy()

    Y, seeds = response_cube(df, metric)
    n_cells = Y.shape[0]
    X_grid, g_cell, g_node = grid_features(df)
    train_seeds = np.sort(df.iloc[tr]["seed"].unique())
    test_seeds = np.sort(df.iloc[te]["seed"].unique())
    tr_pos = np.searchsorted(seeds, train_seeds)
    te_pos = np.searchsorted(seeds, test_seeds)

    # --- OOF folds over training seeds -------------------------------------
    gkf = GroupKFold(n_splits=N_OOF_FOLDS, shuffle=True, random_state=OOF_SEED)
    fold_of_seed = np.full(seeds.size, -1)
    mu_oof = np.full((n_cells, train_seeds.size, Y.shape[2]), np.nan)
    sig_oof = np.full_like(mu_oof, np.nan)
    fold_rows = []
    for k, (fit_rel, hold_rel) in enumerate(gkf.split(tr, groups=groups[tr])):
        t0 = time.perf_counter()
        fit_pool, hold_idx = tr[fit_rel], tr[hold_rel]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=100 + k)
        f_rel, v_rel = next(gss.split(fit_pool, groups=groups[fit_pool]))
        fit_sub = tng._subsample_train(
            fit_pool[f_rel], groups, ngb_common.TRAIN_SUBSAMPLE_FRAC, 200 + k
        )
        val_sub = tng._subsample_train(
            fit_pool[v_rel], groups, min(1.0, ngb_common.TRAIN_SUBSAMPLE_FRAC * 2), 300 + k
        )
        model, n_trees = tng.fit_one(X_all[fit_sub], y[fit_sub], X_all[val_sub], y[val_sub])
        mu_g, sig_g = _grid_mu_sigma(model, X_grid, g_cell, g_node, n_cells)

        hold_seeds = np.unique(groups[hold_idx])
        fold_of_seed[np.searchsorted(seeds, hold_seeds)] = k
        cols = np.searchsorted(train_seeds, hold_seeds)
        mu_oof[:, cols, :] = mu_g[:, None, :]
        sig_oof[:, cols, :] = sig_g[:, None, :]

        y_h = Y[:, np.searchsorted(seeds, hold_seeds), :]
        z = (y_h - mu_g[:, None, :]) / sig_g[:, None, :]
        fold_rows.append(
            {
                "fold": k,
                "n_hold_seeds": int(hold_seeds.size),
                "n_trees": n_trees,
                "pi90_coverage": float(np.mean(np.abs(z) <= 1.6448536269514722)),
                "mean_z": float(np.mean(z)),
                "sd_z": float(np.std(z)),
                "fit_seconds": time.perf_counter() - t0,
            }
        )
        print(f"  fold {k}: trees={n_trees}  PI90={fold_rows[-1]['pi90_coverage']:.4f}")

    z_train = (Y[:, tr_pos, :] - mu_oof) / sig_oof

    # --- Test seeds from the saved full model ------------------------------
    full = joblib.load(ngb_common.models_dir() / f"ngboost_{metric}.pkl")
    mu_full, sig_full = _grid_mu_sigma(full, X_grid, g_cell, g_node, n_cells)
    z_test = (Y[:, te_pos, :] - mu_full[:, None, :]) / sig_full[:, None, :]
    test_pi90 = float(np.mean(np.abs(z_test) <= 1.6448536269514722))

    path = residuals_path(metric)
    np.savez_compressed(
        path,
        z_train=z_train,
        z_test=z_test,
        y_test=Y[:, te_pos, :],
        mu_test=mu_full,
        sigma_test=sig_full,
        train_seeds=train_seeds,
        test_seeds=test_seeds,
        fold_of_train_seed=fold_of_seed[tr_pos],
    )
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(out / f"oof_folds_{metric}.csv", index=False)
    meta = {
        "metric": metric,
        "n_folds": N_OOF_FOLDS,
        "oof_seed": OOF_SEED,
        "shape_z_train": list(z_train.shape),
        "shape_z_test": list(z_test.shape),
        "nan_frac_train": float(np.mean(~np.isfinite(z_train))),
        "nan_frac_test": float(np.mean(~np.isfinite(z_test))),
        "test_pi90_full_model": test_pi90,
    }
    (out / f"meta_{metric}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    lines = [
        f"# Out-of-fold NGBoost residuals ({metric})",
        "",
        "## Definitions",
        "",
        r"- \(z_{kji}=(Y_{kji}-\hat\mu_{ki})/\hat\sigma_{ki}\), \(Y=\ln\chi\); cube axes = (cell, seed, node).",
        f"- Training seeds split into {N_OOF_FOLDS} seed folds; each fold's residuals come from an NGBoost refit without those seeds (same settings as `chi_ngboost/train_ngboost.py`).",
        "- Test-seed residuals use the saved full model (`chi_ngboost/models`).",
        "",
        "## Folds",
        "",
        folds.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"- Test-seed pointwise PI90 (full model): **{test_pi90:.4f}**",
        f"- z_train shape {tuple(z_train.shape)}, z_test shape {tuple(z_test.shape)}",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
