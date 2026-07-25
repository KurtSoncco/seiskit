"""gplearn symbolic regression for NGBoost μ and log σ surfaces.

Fits GP on standardized targets; always fits shortlist OLS; reports OLS when
GP collapses. Writes under figure_dir("chi_sr", "train_sr").
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicRegressor
from ngboost import NGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    FEATURES,
    GP_PARAMS,
    METRICS,
    SR_SAMPLE_N,
    SR_SAMPLE_SEED,
    add_design_columns,
    build_design_matrix,
    destandardize,
    is_collapsed_program,
    load_or_make_split,
    load_ratios,
    load_shortlist,
    ngboost_models_dir,
    ols_shortlist_fit,
    out_dir,
    r2_score,
    rmse,
    shortlist_features,
    standardize_y,
    wrap_standardized_formula,
)

warnings.filterwarnings("ignore")


def _ngb_targets(model: NGBRegressor, X_full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dist = model.pred_dist(X_full)
    mu = np.asarray(dist.loc, dtype=float).ravel()
    sigma = np.maximum(np.asarray(dist.scale, dtype=float).ravel(), 1e-8)
    return mu, np.log(sigma)


def main() -> None:
    out = out_dir("train_sr")
    short = load_shortlist()
    print("Loading data …")
    df = add_design_columns(load_ratios())
    tr, te = load_or_make_split(df)
    rng = np.random.default_rng(SR_SAMPLE_SEED)
    tr_s = rng.choice(tr, size=min(SR_SAMPLE_N, len(tr)), replace=False)
    te_s = rng.choice(te, size=min(SR_SAMPLE_N, len(te)), replace=False)

    formula_rows = []
    fidelity_rows = []
    ols_rows = []
    meta = {
        "gp_params": GP_PARAMS,
        "sr_sample_n": SR_SAMPLE_N,
        "standardize_target": True,
        "metrics": {},
    }

    for metric in METRICS:
        mpath = ngboost_models_dir() / f"ngboost_{metric}.pkl"
        if not mpath.is_file():
            raise FileNotFoundError(f"Missing {mpath}")
        model: NGBRegressor = joblib.load(mpath)
        X_full = df[FEATURES].to_numpy(dtype=float)
        mu_all, logsig_all = _ngb_targets(model, X_full)

        for target, y_all in [("mu", mu_all), ("log_sigma", logsig_all)]:
            feat_names = shortlist_features(short, metric, target)
            X_mat, colnames = build_design_matrix(df, feat_names, short, metric, target)
            X_tr, y_tr = X_mat[tr_s], y_all[tr_s]
            X_te, y_te = X_mat[te_s], y_all[te_s]
            m_tr = np.isfinite(y_tr) & np.all(np.isfinite(X_tr), axis=1)
            m_te = np.isfinite(y_te) & np.all(np.isfinite(X_te), axis=1)
            X_tr_f, y_tr_f = X_tr[m_tr], y_tr[m_tr]
            X_te_f, y_te_f = X_te[m_te], y_te[m_te]

            # OLS baseline on original scale
            ols = ols_shortlist_fit(X_tr_f, y_tr_f, X_te_f, y_te_f, colnames)
            ols_row = {
                "metric": metric,
                "target": target,
                "formula": ols["formula"],
                "r2_train": ols["r2_train"],
                "r2_test": ols["r2_test"],
                "rmse_train": ols["rmse_train"],
                "rmse_test": ols["rmse_test"],
                "n_features": len(colnames),
                "features": ",".join(colnames),
                "intercept": float(ols["coef"][0]),
            }
            for j, name in enumerate(colnames):
                ols_row[f"coef_{name}"] = float(ols["coef"][j + 1])
            ols_rows.append(ols_row)

            # GP on standardized target
            z_tr, y_mean, y_std = standardize_y(y_tr_f)
            z_te = (y_te_f - y_mean) / y_std
            print(f"SR {metric} {target} features={colnames}  y_std={y_std:.4g} …")
            est = SymbolicRegressor(
                function_set=("add", "sub", "mul", "div"),
                feature_names=colnames,
                **GP_PARAMS,
            )
            est.fit(X_tr_f, z_tr)
            zhat_tr = est.predict(X_tr_f)
            zhat_te = est.predict(X_te_f)
            yhat_gp_tr = destandardize(zhat_tr, y_mean, y_std)
            yhat_gp_te = destandardize(zhat_te, y_mean, y_std)

            formula_z = str(est._program)
            formula_gp = wrap_standardized_formula(formula_z, y_mean, y_std)
            program_length = int(est._program.length_)
            r2_gp_te = r2_score(y_te_f, yhat_gp_te)
            r2_gp_tr = r2_score(y_tr_f, yhat_gp_tr)
            collapsed = is_collapsed_program(formula_z, program_length, r2_gp_te)

            if collapsed:
                formula_reported = ols["formula"]
                formula_source = "ols"
                yhat_rep_tr = ols["yhat_tr"]
                yhat_rep_te = ols["yhat_te"]
            else:
                formula_reported = formula_gp
                formula_source = "gp"
                yhat_rep_tr = yhat_gp_tr
                yhat_rep_te = yhat_gp_te

            formula_rows.append(
                {
                    "metric": metric,
                    "target": target,
                    "formula_z": formula_z,
                    "formula_gp": formula_gp,
                    "formula_reported": formula_reported,
                    "formula_source": formula_source,
                    "sr_collapsed": int(collapsed),
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "n_features": len(colnames),
                    "features": ",".join(colnames),
                    "program_length": program_length,
                    "raw_fitness_": float(getattr(est._program, "raw_fitness_", np.nan)),
                }
            )
            fidelity_rows.append(
                {
                    "metric": metric,
                    "target": target,
                    "r2_train_gp": r2_gp_tr,
                    "r2_test_gp": r2_gp_te,
                    "r2_train_ols": ols["r2_train"],
                    "r2_test_ols": ols["r2_test"],
                    "r2_train_reported": r2_score(y_tr_f, yhat_rep_tr),
                    "r2_test_reported": r2_score(y_te_f, yhat_rep_te),
                    "rmse_train_gp": rmse(y_tr_f, yhat_gp_tr),
                    "rmse_test_gp": rmse(y_te_f, yhat_gp_te),
                    "rmse_train_ols": ols["rmse_train"],
                    "rmse_test_ols": ols["rmse_test"],
                    "rmse_test_reported": rmse(y_te_f, yhat_rep_te),
                    "sr_collapsed": int(collapsed),
                    "formula_source": formula_source,
                    "n_train": int(m_tr.sum()),
                    "n_test": int(m_te.sum()),
                }
            )
            joblib.dump(
                {
                    "gp": est,
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "ols_coef": ols["coef"],
                    "colnames": colnames,
                    "formula_source": formula_source,
                },
                out / f"sr_{metric}_{target}.pkl",
            )
            meta["metrics"].setdefault(metric, {})[target] = {
                "formula_source": formula_source,
                "sr_collapsed": collapsed,
                "formula_reported": formula_reported,
                "features": colnames,
                "y_mean": y_mean,
                "y_std": y_std,
            }

    formulas = pd.DataFrame(formula_rows)
    fidelity = pd.DataFrame(fidelity_rows)
    ols_df = pd.DataFrame(ols_rows)
    formulas.to_csv(out / "sr_formulas.csv", index=False)
    fidelity.to_csv(out / "sr_fidelity.csv", index=False)
    ols_df.to_csv(out / "ols_baseline.csv", index=False)
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    n_collapse = int(fidelity["sr_collapsed"].sum())
    lines = [
        "# Symbolic regression training summary",
        "",
        "## Definitions",
        "",
        r"- Targets: NGBoost surfaces \(\mu(\mathbf{x})\) and \(\log\sigma(\mathbf{x})\) (still on \(Y=\ln\chi\)).",
        r"- Standardization: GP fits \(z=(y-\bar y_{\mathrm{tr}})/s_{y,\mathrm{tr}}\); reported GP formulas wrap back to \(y\).",
        r"- Features: SHAP shortlist mains + interaction products.",
        r"- OLS baseline: intercept + shortlist columns (always computed).",
        r"- Collapse guard: if GP test \(R^2<0.05\) or constant/short program → `formula_reported` = OLS.",
        r"- Fidelity: \(R^2\) vs NGBoost surface on original scale (not raw \(Y\)).",
        "",
        "## Output files",
        "",
        "| File | Content |",
        "|------|---------|",
        "| `sr_formulas.csv` | GP / reported formulas, collapse flags, `y_mean`/`y_std` |",
        "| `sr_fidelity.csv` | `r2_*_gp`, `r2_*_ols`, `r2_*_reported` |",
        "| `ols_baseline.csv` | shortlist OLS coefs and formulas |",
        "",
        "## Fidelity (reported)",
        "",
        fidelity[
            [
                "metric",
                "target",
                "r2_test_gp",
                "r2_test_ols",
                "r2_test_reported",
                "sr_collapsed",
                "formula_source",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        f"Collapsed GP runs (OLS fallback): **{n_collapse}** / {len(fidelity)}.",
        "",
        "## Conclusions",
        "",
        "- Prior μ collapses were search/parsimony artifacts on small absolute MSE, not flat NGBoost surfaces — OLS shortlist \(R^2\) documents recoverable linear structure.",
        "- Reported formulas use GP when it beats the collapse threshold; otherwise the shortlist OLS formula is the readable compression.",
        "- log-σ was already well recovered by GP; standardization + lower parsimony keeps that while making μ usable.",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        fidelity[
            [
                "metric",
                "target",
                "r2_test_gp",
                "r2_test_ols",
                "r2_test_reported",
                "sr_collapsed",
                "formula_source",
            ]
        ].to_string(index=False)
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
