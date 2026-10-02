"""NGBoost on the joint-layer kernel parameters θ_k as functions of the design.

One row per design cell (243). Features = the 5 z-scored factors (no node).
Targets = partially pooled per-cell θ_k from ``fit_joint_cov.py`` on the
unconstrained scale (a_b = log(w_b/w_n), a_s = log(w_s/w_n), log s, log ν,
ω = 100/b) plus derived log h95 of the kernel.

Runs only for the kernel chosen in ``evaluate_joint`` (CosWM) unless called
with ``--force <kernel>``.

- 5-fold cell CV → R², PI90 coverage; weighted (1/SE²) vs unweighted per target
- final models refit on all cells with the CV-median number of trees
- generalization: θ̂(d_k) from CV predictions → R_k → test-seed joint log score
  vs per-cell θ_k and global θ (seed bootstrap CI)

Writes under ``figure_dir("chi_joint", "train_kernel_ngboost")`` and
``figure_dir("chi_joint", "models")``.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore
from scipy import stats
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import covariance as cv  # noqa: E402
from common import (  # noqa: E402
    METRIC,
    ZCOLS,
    cov_cell_path,
    models_dir,
    out_dir,
    r2_score,
    residuals_path,
)

warnings.filterwarnings("ignore")

N_FOLDS = 5
CV_SEED = 7
MAX_TREES = 500
EARLY_STOP = 30
WEIGHT_CLIP = 10.0  # max weight relative to the median
B_LS = 1000
BOOT_SEED = 43
Z90 = float(stats.norm.ppf(0.95))


def ngb(n_estimators: int = MAX_TREES) -> NGBRegressor:
    return NGBRegressor(
        Dist=Normal,
        Score=LogScore,
        Base=DecisionTreeRegressor(criterion="friedman_mse", max_depth=2, random_state=0),
        n_estimators=n_estimators,
        learning_rate=0.03,
        minibatch_frac=1.0,
        verbose=False,
        random_state=0,
    )


def target_table(
    kernel: str, cell_tab: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Targets (cells × T), their bootstrap SEs, and the per-cell rows."""
    sub = cell_tab[cell_tab["model"] == kernel].sort_values("cell").reset_index(drop=True)
    names = cv.param_names(kernel)
    Y = pd.DataFrame({n: sub[f"theta_{n}"].to_numpy() for n in names})
    SE = pd.DataFrame({n: sub[f"se_theta_{n}"].to_numpy() for n in names})
    h95 = sub["h95_kernel_m"].to_numpy()
    Y["log_h95"] = np.log(h95)
    se_h = (
        sub["se_h95_kernel_m"].to_numpy() if "se_h95_kernel_m" in sub else np.full(len(sub), np.nan)
    )
    SE["log_h95"] = se_h / h95  # delta method
    return Y, SE, sub


def _weights(se: np.ndarray) -> np.ndarray:
    se = np.asarray(se, dtype=float)
    if not np.all(np.isfinite(se)) or np.nanmax(se) <= 0:
        return np.ones_like(se)
    floor = np.nanmedian(se[se > 0]) / np.sqrt(WEIGHT_CLIP)
    w = 1.0 / np.maximum(se, floor) ** 2
    return w / w.mean()


def fit_es(X, y, w, seed):
    """Fit with an inner 20% early-stopping split; returns (model, n_trees)."""
    X_f, X_v, y_f, y_v, w_f, _ = train_test_split(X, y, w, test_size=0.2, random_state=seed)
    m = ngb()
    m.fit(X_f, y_f, X_val=X_v, Y_val=y_v, sample_weight=w_f, early_stopping_rounds=EARLY_STOP)
    return m, int(getattr(m, "best_val_loss_itr", None) or m.n_estimators)


def cv_target(X, y, w):
    """Out-of-fold μ̂, σ̂ and per-fold tree counts."""
    mu, sig = np.empty_like(y), np.empty_like(y)
    trees = []
    for k, (tr, te) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=CV_SEED).split(X)):
        m, nt = fit_es(X[tr], y[tr], w[tr], seed=k)
        d = m.pred_dist(X[te], max_iter=nt)
        mu[te], sig[te] = d.loc, d.scale
        trees.append(nt)
    return mu, sig, trees


def _clip_theta(kernel: str, theta: np.ndarray) -> np.ndarray:
    lo = np.array([b[0] for b in cv.bounds(kernel)])
    hi = np.array([b[1] for b in cv.bounds(kernel)])
    return np.clip(theta, lo, hi)


def test_log_scores(
    kernel: str, thetas: np.ndarray, z_te: np.ndarray, log_sig: np.ndarray
) -> np.ndarray:
    p = z_te.shape[2]
    return np.stack(
        [
            cv.loglik_profiles(cv.build_R(kernel, thetas[c], p), z_te[c]) - log_sig[c]
            for c in range(z_te.shape[0])
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default=METRIC)
    ap.add_argument("--force", default=None, help="kernel to use regardless of the decision")
    args = ap.parse_args()
    metric = args.metric

    decision_path = out_dir("evaluate_joint") / f"decision_{metric}.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if args.force:
        kernel = args.force
    elif decision["coswm_better"]:
        kernel = "coswm"
    else:
        print(
            "CosWM did not beat Matérn-3/2 on held-out profiles "
            f"(see {decision_path}); not training. Use --force <kernel> to override."
        )
        return

    out = out_dir("train_kernel_ngboost")
    cell_tab = pd.read_csv(cov_cell_path(metric))
    Y, SE, sub = target_table(kernel, cell_tab)
    X = sub[ZCOLS].to_numpy(dtype=float)

    rows, cv_mu = [], {}
    mdir = models_dir()
    for t in Y.columns:
        y = Y[t].to_numpy(dtype=float)
        if np.nanstd(y) < 1e-8:
            print(f"  {t}: constant across cells (fully pooled) — skipped")
            rows.append({"target": t, "skipped": "constant"})
            continue
        best = None
        for wtag, w in (("unweighted", np.ones_like(y)), ("inv_se2", _weights(SE[t].to_numpy()))):
            mu, sig, trees = cv_target(X, y, w)
            res = {
                "target": t,
                "weighting": wtag,
                "cv_r2": r2_score(y, mu),
                "cv_rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
                "cv_pi90": float(np.mean(np.abs(y - mu) <= Z90 * sig)),
                "cv_median_sigma": float(np.median(sig)),
                "median_boot_se": float(np.nanmedian(SE[t])),
                "sd_target": float(np.std(y)),
                "cv_trees_median": int(np.median(trees)),
                "mu": mu,
            }
            if best is None or res["cv_r2"] > best["cv_r2"]:
                best = {**res, "w": w}
        cv_mu[t] = best.pop("mu")
        w = best.pop("w")
        final = ngb(max(best["cv_trees_median"], 10))
        final.fit(X, y, sample_weight=w)
        path = mdir / f"ngboost_kparam_{kernel}_{t}_{metric}.pkl"
        joblib.dump(final, path)
        best["model"] = path.name
        rows.append(best)
        print(f"  {t}: CV R²={best['cv_r2']:.3f} ({best['weighting']}), PI90={best['cv_pi90']:.3f}")
    skill = pd.DataFrame(rows)
    skill.to_csv(out / f"kernel_ngboost_cv_{metric}_{kernel}.csv", index=False)
    pd.DataFrame(
        {"cell": sub["cell"], **{f"cv_mu_{k}": v for k, v in cv_mu.items()}, **{k: Y[k] for k in Y}}
    ).to_csv(out / f"kernel_ngboost_cv_predictions_{metric}_{kernel}.csv", index=False)

    # --- generalization of design-dependent covariance --------------------
    cubes = np.load(residuals_path(metric))
    z_te, sig_te = cubes["z_test"], cubes["sigma_test"]
    log_sig = np.log(sig_te).sum(axis=1)
    fit_dir = out_dir("fit_joint_cov")
    theta_glob = np.load(fit_dir / f"theta_global_{metric}.npz")[kernel]
    theta_cell = np.load(fit_dir / f"theta_cell_{metric}.npz")[f"{kernel}_theta"]
    names = cv.param_names(kernel)
    theta_hat = theta_cell.copy()
    for j, n in enumerate(names):
        if n in cv_mu:
            theta_hat[:, j] = cv_mu[n]
    theta_hat = np.array([_clip_theta(kernel, t) for t in theta_hat])
    n_cells = z_te.shape[0]
    ls = {
        "per_cell_pooled": test_log_scores(kernel, theta_cell, z_te, log_sig),
        "ngboost_cv_predicted": test_log_scores(kernel, theta_hat, z_te, log_sig),
        "global": test_log_scores(kernel, np.tile(theta_glob, (n_cells, 1)), z_te, log_sig),
    }
    rng = np.random.default_rng(BOOT_SEED)
    n_te = z_te.shape[1]
    keys = list(ls)
    arr = np.stack([ls[k] for k in keys])
    boot = np.empty((B_LS, len(keys)))
    for b in range(B_LS):
        idx = rng.integers(0, n_te, n_te)
        boot[b] = arr[:, :, idx].mean(axis=(1, 2))
    gen_rows = []
    for i, k in enumerate(keys):
        d = boot[:, i] - boot[:, keys.index("global")]
        gen_rows.append(
            {
                "theta_source": k,
                "ls_per_profile": float(arr[i].mean()),
                "gain_vs_global": float(arr[i].mean() - arr[keys.index("global")].mean()),
                "gain_lo": float(np.quantile(d, 0.025)),
                "gain_hi": float(np.quantile(d, 0.975)),
            }
        )
    gen = pd.DataFrame(gen_rows)
    gen.to_csv(out / f"generalization_{metric}_{kernel}.csv", index=False)

    show = skill.drop(columns=[c for c in ("model",) if c in skill.columns])
    lines = [
        f"# NGBoost on joint-layer kernel parameters ({metric}, kernel = {kernel})",
        "",
        "## Definitions",
        "",
        r"- Rows: 243 design cells; features: 5 z-scored design factors; targets: partially pooled \(\theta_k\) (unconstrained scale) and \(\log h_{95}\) of the kernel.",
        "- Normal NGBoost (depth-2 trees, lr 0.03, early stopping on an inner 20% split).",
        f"- {N_FOLDS}-fold cell CV; weighting = unweighted or 1/SE² (bootstrap SE, clipped at {WEIGHT_CLIP:g}× median), whichever has higher CV R².",
        "- `cv_pi90`: fraction of cells with |θ_k − μ̂| ≤ 1.645 σ̂ (NGBoost σ̂ vs the observed spread of per-cell estimates).",
        "- Generalization: CV-predicted θ̂(d_k) plugged into R_k, scored on held-out test seeds (joint log score, Y scale).",
        "",
        "## CV skill",
        "",
        show.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Does the design-dependent covariance generalize?",
        "",
        gen.to_markdown(index=False, floatfmt=".4f"),
        "",
        "- A positive `gain_vs_global` for `ngboost_cv_predicted` means covariance parameters predicted from the design factors alone beat a single pooled covariance on unseen seeds, *for cells not used to fit the predictor*.",
        "- Targets whose CV R² is near zero are effectively design-independent at this aperture; their SHAP/PDP (in `compare_shap_pdp`) should not be interpreted.",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(gen.to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
