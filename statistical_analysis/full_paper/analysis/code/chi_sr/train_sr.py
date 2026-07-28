"""gplearn symbolic regression for NGBoost factorial-grid surfaces.

Distills closed-form engineering approximations to NGBoost μ, log σ, and
quantiles τ ∈ {0.05, 0.50, 0.95} on the unique cell × node design grid.
GP collapses fall back to shortlist OLS. Writes under figure_dir("chi_sr", "train_sr").
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    CELL_SPLIT_SEED,
    FEATURES,
    GP_PARAMS,
    METRICS,
    SHORTLIST_INTERACTION_SOURCE,
    SHORTLIST_MAIN_SOURCES,
    SR_TARGETS,
    TEST_SIZE,
    build_design_matrix,
    cell_grouped_split,
    destandardize,
    is_collapsed_program,
    load_shortlist,
    load_surface,
    ols_shortlist_fit,
    out_dir,
    r2_score,
    rmse,
    shortlist_features,
    standardize_y,
    wrap_standardized_formula,
)

warnings.filterwarnings("ignore")


def main() -> None:
    out = out_dir("train_sr")
    short = load_shortlist()

    formula_rows = []
    fidelity_rows = []
    ols_rows = []
    meta = {
        "gp_params": GP_PARAMS,
        "standardize_target": True,
        "design": "factorial_grid_cell_holdout",
        "cell_test_size": TEST_SIZE,
        "cell_split_seed": CELL_SPLIT_SEED,
        "sr_targets": list(SR_TARGETS),
        "shortlist_main_sources": {k: list(v) for k, v in SHORTLIST_MAIN_SOURCES.items()},
        "shortlist_interaction_source": SHORTLIST_INTERACTION_SOURCE,
        "metrics": {},
        "grid": {},
    }

    for metric in METRICS:
        print(f"Loading surface {metric} …")
        df = load_surface(metric)
        tr, te = cell_grouped_split(df)
        meta["grid"][metric] = {
            "n_rows": int(len(df)),
            "n_cells": int(df["cell"].nunique()),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "n_train_cells": int(df.iloc[tr]["cell"].nunique()),
            "n_test_cells": int(df.iloc[te]["cell"].nunique()),
        }

        for target in SR_TARGETS:
            if target not in df.columns:
                raise KeyError(f"Surface missing column {target!r} for {metric}")
            y_all = df[target].to_numpy(dtype=float)
            feat_names = shortlist_features(short, metric, target)
            X_mat, colnames = build_design_matrix(df, feat_names, short, metric, target)
            X_tr, y_tr = X_mat[tr], y_all[tr]
            X_te, y_te = X_mat[te], y_all[te]
            m_tr = np.isfinite(y_tr) & np.all(np.isfinite(X_tr), axis=1)
            m_te = np.isfinite(y_te) & np.all(np.isfinite(X_te), axis=1)
            X_tr_f, y_tr_f = X_tr[m_tr], y_tr[m_tr]
            X_te_f, y_te_f = X_te[m_te], y_te[m_te]

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

            z_tr, y_mean, y_std = standardize_y(y_tr_f)
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
                    "shortlist_mains": ",".join(SHORTLIST_MAIN_SOURCES.get(target, (target,))),
                    "shortlist_interactions": SHORTLIST_INTERACTION_SOURCE.get(target) or "",
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
                    "features": FEATURES,
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
        "## Positioning",
        "",
        "Symbolic formulas are **engineering approximations distilled from the "
        "NGBoost probabilistic surrogate** "
        r"\(p(Y\mid\mathbf{x})=\mathcal{N}(\mu(\mathbf{x}),\sigma^2(\mathbf{x}))\). "
        r"They are **not** the primary model: fidelity is \(R^2\) versus NGBoost "
        r"surfaces on the factorial grid, not versus observed \(\ln\chi\). "
        "Primary inference remains NGBoost (and QBM where relevant).",
        "",
        "## Definitions",
        "",
        r"- Design: unique cell × node factorial grid from `chi_ngboost` surfaces "
        r"(seed axis dropped); train/test split holds out design **cells**.",
        r"- Targets: NGBoost surfaces \(\mu\), \(\log\sigma\), and separate "
        r"closed forms for \(q_{0.05}\), \(q_{0.50}\), \(q_{0.95}\) "
        r"(still on \(Y=\ln\chi\)).",
        r"- Under Normal NGBoost, median \(\equiv\mu\) and "
        r"\(q_\tau=\mu+\sigma\,z_\tau\); SR nonetheless fits **separate** "
        "quantile formulas for practitioner use.",
        r"- Shortlist: `q50`→μ SHAP; `q05`/`q95`→union(μ, logσ) mains + μ interactions; "
        r"`log_sigma`→logσ mains only.",
        r"- Standardization: GP fits \(z=(y-\bar y_{\mathrm{tr}})/s_{y,\mathrm{tr}}\); "
        r"reported GP formulas wrap back to \(y\).",
        r"- OLS baseline: intercept + shortlist columns (always computed).",
        r"- Collapse guard: if GP test \(R^2<0.05\) or constant/short program → "
        "`formula_reported` = OLS.",
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
        "- Reported formulas compress NGBoost surfaces for engineering use; "
        "they do not replace the probabilistic surrogate.",
        "- Quantile formulas (q05/q50/q95) are fitted separately even though "
        "Normal NGBoost implies \\(q_\\tau=\\mu+\\sigma z_\\tau\\).",
        "- Collapse → OLS shortlist remains the readable fallback when GP fails.",
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
