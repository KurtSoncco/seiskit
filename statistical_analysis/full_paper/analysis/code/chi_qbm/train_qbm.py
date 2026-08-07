r"""LightGBM QBM (+ mean GBM) on the full-array χ ratios.

Fits mean and quantile models for all five metrics on \(Y=\ln\chi\) with
features = z-scored factors + ``node_z``. Seed-grouped holdout matches
``spatial_ols`` (``SPLIT_SEED``).

Writes models under ``figure_dir("chi_qbm", "models")`` and training
CSV + summary.md under ``figure_dir("chi_qbm", "train_qbm")``.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    METRICS,
    N_CELLS,
    N_NODES,
    N_SEEDS,
    QBM_FEATURES,
    TAUS,
    TEST_SIZE,
    VAL_FRAC,
    VAL_SEED,
    add_design_columns,
    fmt,
    load_ratios,
    log_response,
    models_dir,
    out_dir,
    pinball_loss,
    r2_score,
    rmse,
    save_split,
    seed_grouped_split_indices,
)

warnings.filterwarnings("ignore")

LGB_PARAMS_BASE = dict(
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    verbose=-1,
    n_jobs=-1,
)
MAX_ROUNDS = 2000
EARLY_STOPPING_ROUNDS = 50


def _fit(params: dict, X_fit, y_fit, X_val, y_val):
    ds_fit = lgb.Dataset(X_fit, label=y_fit, free_raw_data=False)
    ds_val = lgb.Dataset(X_val, label=y_val, reference=ds_fit, free_raw_data=False)
    model = lgb.train(
        params,
        ds_fit,
        num_boost_round=MAX_ROUNDS,
        valid_sets=[ds_val],
        valid_names=["val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=0),
        ],
    )
    return model, int(model.best_iteration)


def fit_mean_model(X_fit, y_fit, X_val, y_val, seed: int = 42):
    params = {
        **LGB_PARAMS_BASE,
        "objective": "regression",
        "metric": "rmse",
        "seed": seed,
    }
    return _fit(params, X_fit, y_fit, X_val, y_val)


def fit_quantile_model(X_fit, y_fit, X_val, y_val, tau: float, seed: int = 42):
    params = {
        **LGB_PARAMS_BASE,
        "objective": "quantile",
        "metric": "quantile",
        "alpha": tau,
        "seed": seed,
    }
    return _fit(params, X_fit, y_fit, X_val, y_val)


def predict_sorted(models_tau: dict[float, lgb.Booster], X, taus=TAUS) -> np.ndarray:
    raw = np.column_stack([models_tau[t].predict(X) for t in taus])
    raw.sort(axis=1)
    return raw


def model_stem(metric: str, kind: str) -> str:
    """kind: 'mean' or 'q05' etc."""
    return f"lgbm_{kind}_{metric}_seed"


def build_summary_md(meta: pd.DataFrame) -> str:
    lines = [
        "# Full-array LightGBM QBM training",
        "",
        "Mean GBM and quantile boosting models for \\(Y = \\ln\\chi\\) on the full node×seed array.",
        "",
        rf"- Design cells: **{N_CELLS}**; \(N_x = {N_NODES}\); \(N_s = {N_SEEDS}\)",
        f"- Features: `{'`, `'.join(QBM_FEATURES)}` "
        "(z-scored factors + spatial `node_z`; seed is **not** a predictor)",
        f"- Quantiles \\(\\tau \\in \\{{{', '.join(str(t) for t in TAUS)}\\}}\\)",
        f"- Split: seed-grouped holdout ({TEST_SIZE:.0%}), "
        f"early-stop val = {VAL_FRAC:.0%} of train seeds",
        "- Trees encode **interactions**; separate τ models capture "
        "**heteroscedasticity / tails** without a parametric variance equation",
        "",
        "## Output files",
        "",
        "| File | Contents |",
        "|------|----------|",
        "| `qbm_train_meta.csv` | Best iteration, holdout RMSE / pinball / coverage per model |",
        "| `seed_split.json` | Holdout seed lists (aligned with spatial OLS) |",
        "| `../models/lgbm_*.pkl` | Serialized LightGBM boosters |",
        "| `summary.md` | this file |",
        "",
        "## Notation",
        "",
        "Conditional quantile \\(q_\\tau(Y\\mid \\mathbf{x}_k, x_{\\mathrm{node}})\\) "
        "estimated by LightGBM with pinball loss. Mean GBM minimizes RMSE for "
        "point-prediction parity with OLS.",
        "",
        "## Holdout metrics",
        "",
        "| Metric | Mean \\(R^2\\) | Mean RMSE | Pinball \\(\\tau=0.5\\) | 90% PI cov |",
        "|--------|-------------:|----------:|---------------------:|-----------:|",
    ]
    for metric in METRICS:
        sub = meta[meta["metric"] == metric]
        mean = sub[sub["kind"] == "mean"].iloc[0]
        med = sub[sub["kind"] == "q50"]
        pin = float(med["pinball"].iloc[0]) if len(med) else np.nan
        # 90% coverage from q05/q95 if present in a summary row
        cov_row = sub[sub["kind"] == "pi90"]
        cov = float(cov_row["coverage"].iloc[0]) if len(cov_row) else np.nan
        lines.append(
            f"| {metric} | {fmt(mean['r2'])} | {fmt(mean['rmse'])} | {fmt(pin)} | {fmt(cov)} |"
        )

    lines.extend(["", "## Conclusions", ""])
    for metric in METRICS:
        mean = meta[(meta["metric"] == metric) & (meta["kind"] == "mean")].iloc[0]
        lines.append(
            rf"- **{metric}**: mean GBM holdout \(R^2 = {fmt(mean['r2'])}\) "
            f"(RMSE {fmt(mean['rmse'])}); QBM models saved for all τ."
        )
    lines.extend(
        [
            "",
            "Use `compare_models.py` for the three-way comparison against "
            "naive OLS and CosWM spatial OLS (pinball, PI hetero, residual lag-1).",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    print("Loading join_master.h5 …")
    df = add_design_columns(load_ratios())
    print(f"  rows={len(df):,}")

    dest = out_dir("train_qbm")
    mdir = models_dir()

    # Prefer spatial_ols split if already written
    spatial_split = out_dir("spatial_ols") / "seed_split.json"
    tr, te = seed_grouped_split_indices(df)
    save_split(dest / "seed_split.json", tr, te, df)
    if spatial_split.is_file():
        # Verify same test seeds
        other = json.loads(spatial_split.read_text())
        mine = json.loads((dest / "seed_split.json").read_text())
        if other.get("test_seeds") != mine.get("test_seeds"):
            print("WARNING: spatial_ols seed_split differs; using common SPLIT_SEED anyway")
        else:
            print("  seed split matches spatial_ols")

    train = df.iloc[tr].copy()
    test = df.iloc[te].copy()
    print(f"  train={len(train):,} test={len(test):,}")

    gss_val = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=VAL_SEED)
    fit_idx, val_idx = next(gss_val.split(train, groups=train["seed"].to_numpy()))
    fit_df, val_df = train.iloc[fit_idx], train.iloc[val_idx]
    print(f"  fit={len(fit_df):,} val={len(val_df):,}")

    meta_rows: list[dict] = []

    for metric in METRICS:
        print(f"\n=== {metric} ===")
        y_fit = log_response(fit_df, metric)
        y_val = log_response(val_df, metric)
        y_te = log_response(test, metric)
        m_fit = np.isfinite(y_fit)
        m_val = np.isfinite(y_val)
        m_te = np.isfinite(y_te)

        X_fit = fit_df.loc[m_fit, QBM_FEATURES]
        X_val = val_df.loc[m_val, QBM_FEATURES]
        X_te = test.loc[m_te, QBM_FEATURES]
        yf, yv, yt = y_fit[m_fit], y_val[m_val], y_te[m_te]

        # Mean GBM
        m_mean, n_iter = fit_mean_model(X_fit, yf, X_val, yv)
        pred_mean = m_mean.predict(X_te)
        meta_rows.append(
            dict(
                metric=metric,
                kind="mean",
                tau=np.nan,
                best_iteration=n_iter,
                r2=r2_score(yt, pred_mean),
                rmse=rmse(yt, pred_mean),
                pinball=np.nan,
                coverage=np.nan,
                model_file=f"{model_stem(metric, 'mean')}.pkl",
            )
        )
        joblib.dump(m_mean, mdir / f"{model_stem(metric, 'mean')}.pkl")
        print(f"  mean  R²={meta_rows[-1]['r2']:.4f}  rounds={n_iter}")

        q_models: dict[float, lgb.Booster] = {}
        for tau in TAUS:
            m_q, n_iter = fit_quantile_model(X_fit, yf, X_val, yv, tau)
            q_models[tau] = m_q
            pred = m_q.predict(X_te)
            kind = f"q{int(round(tau * 100)):02d}"
            meta_rows.append(
                dict(
                    metric=metric,
                    kind=kind,
                    tau=tau,
                    best_iteration=n_iter,
                    r2=np.nan,
                    rmse=np.nan,
                    pinball=pinball_loss(yt, pred, tau),
                    coverage=float(np.mean(yt <= pred)),
                    model_file=f"{model_stem(metric, kind)}.pkl",
                )
            )
            joblib.dump(m_q, mdir / f"{model_stem(metric, kind)}.pkl")
            print(
                f"  τ={tau:.2f}  pinball={meta_rows[-1]['pinball']:.5f}  "
                f"cov={meta_rows[-1]['coverage']:.3f}  rounds={n_iter}"
            )

        preds = predict_sorted(q_models, X_te)
        cov90 = float(np.mean((yt >= preds[:, 0]) & (yt <= preds[:, -1])))
        width = preds[:, -1] - preds[:, 0]
        meta_rows.append(
            dict(
                metric=metric,
                kind="pi90",
                tau=np.nan,
                best_iteration=np.nan,
                r2=np.nan,
                rmse=np.nan,
                pinball=np.nan,
                coverage=cov90,
                model_file="",
                mean_pi_width=float(np.mean(width)),
                pi_width_min=float(np.min(width)),
                pi_width_max=float(np.max(width)),
                pi_width_ratio=float(np.max(width) / np.min(width))
                if np.min(width) > 0
                else np.nan,
            )
        )
        print(f"  90% PI coverage={cov90:.3f}  width_ratio={meta_rows[-1]['pi_width_ratio']:.2f}")

    meta = pd.DataFrame(meta_rows)
    meta.to_csv(dest / "qbm_train_meta.csv", index=False)
    (dest / "summary.md").write_text(build_summary_md(meta), encoding="utf-8")
    print(f"\nWrote {dest}")
    print(f"Models in {mdir}")


if __name__ == "__main__":
    main()
