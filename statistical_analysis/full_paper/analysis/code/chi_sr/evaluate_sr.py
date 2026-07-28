"""Evaluate SR formulas on surface and observed holdouts.

Primary: re-score ``formula_reported`` vs NGBoost surfaces on held-out design cells
(with pickle integrity check). Secondary: score the same formulas vs observed
``Y = ln χ`` on the NGBoost seed holdout, with NGBoost metrics as ceiling.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    COLLAPSE_R2_THRESH,
    INTEGRITY_MAX_ABS_TOL,
    INTEGRITY_REL_TOL,
    INTEGRITY_SAMPLE_N,
    INTEGRITY_SAMPLE_SEED,
    METRICS,
    SR_TARGETS,
    add_design_columns,
    cell_grouped_split,
    eval_formula_reported,
    integrity_abs_tol,
    load_or_make_split,
    load_ratios,
    load_surface,
    log_response,
    make_feature_dict,
    ngboost_holdout_metrics_path,
    normal_nll,
    out_dir,
    parse_feature_list,
    pinball_loss,
    predict_from_pickle,
    r2_score,
    rmse,
)

warnings.filterwarnings("ignore")

TAU_BY_TARGET = {"q05": 0.05, "q50": 0.50, "q95": 0.95}


def _section(df: pd.DataFrame, title: str, cols: list[str]) -> list[str]:
    lines = [f"## {title}", ""]
    if len(df) == 0:
        lines.append("None.")
    else:
        use = [c for c in cols if c in df.columns]
        lines.append(df[use].to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    return lines


def _integrity_sample(n: int, seed: int = INTEGRITY_SAMPLE_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=min(INTEGRITY_SAMPLE_N, n), replace=False)


def score_surface_holdout(train: Path, formulas: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        df = load_surface(metric)
        _, te = cell_grouped_split(df)
        df_te = df.iloc[te].reset_index(drop=True)
        sample_idx = _integrity_sample(len(df_te))
        df_s = df_te.iloc[sample_idx].reset_index(drop=True)

        for target in SR_TARGETS:
            fr = formulas[(formulas["metric"] == metric) & (formulas["target"] == target)]
            if len(fr) != 1:
                raise ValueError(f"Expected one formula for {metric}/{target}")
            row = fr.iloc[0]
            colnames = parse_feature_list(row["features"])
            formula = str(row["formula_reported"])
            y_true = df_te[target].to_numpy(dtype=float)

            feat = make_feature_dict(df_te, colnames)
            try:
                y_hat = eval_formula_reported(formula, feat)
                eval_ok = True
                eval_err = ""
            except Exception as exc:  # noqa: BLE001
                y_hat = np.full(len(df_te), np.nan)
                eval_ok = False
                eval_err = str(exc)

            m = np.isfinite(y_true) & np.isfinite(y_hat)
            r2 = r2_score(y_true[m], y_hat[m]) if m.sum() > 2 else float("nan")
            rm = rmse(y_true[m], y_hat[m]) if m.sum() else float("nan")

            pkl_path = train / f"sr_{metric}_{target}.pkl"
            max_abs_pub = float("nan")
            max_abs_prog = float("nan")
            integrity_ok = False
            program_integrity_ok = False
            integrity_note = ""
            tol = integrity_abs_tol(float(row["y_std"]))
            if pkl_path.is_file() and eval_ok:
                try:
                    y_pkl, pkl_cols = predict_from_pickle(pkl_path, df_s)
                    feat_s = make_feature_dict(df_s, colnames)
                    y_pub = eval_formula_reported(formula, feat_s)
                    if str(row["formula_source"]) == "gp":
                        formula_prog = (
                            f"({float(row['y_mean'])}) + ({float(row['y_std'])})*"
                            f"({row['formula_z']})"
                        )
                        y_prog = eval_formula_reported(formula_prog, feat_s)
                    else:
                        # OLS reported: program ≡ published linear string
                        y_prog = y_pub
                    if list(pkl_cols) != colnames:
                        integrity_note = "colnames_mismatch"
                    max_abs_pub = float(np.nanmax(np.abs(y_pub - y_pkl)))
                    max_abs_prog = float(np.nanmax(np.abs(y_prog - y_pkl)))
                    program_integrity_ok = bool(np.isfinite(max_abs_prog) and max_abs_prog <= tol)
                    integrity_ok = bool(np.isfinite(max_abs_pub) and max_abs_pub <= tol)
                    if not program_integrity_ok and not integrity_note:
                        integrity_note = "program_pickle_gap"
                    elif not integrity_ok and not integrity_note:
                        integrity_note = "published_rounding_gap"
                except Exception as exc:  # noqa: BLE001
                    integrity_note = f"pickle_error:{exc}"
            elif not pkl_path.is_file():
                integrity_note = "missing_pickle"

            rows.append(
                {
                    "metric": metric,
                    "target": target,
                    "formula_source": row["formula_source"],
                    "n_test": int(m.sum()),
                    "r2_formula_vs_surface": r2,
                    "rmse_formula_vs_surface": rm,
                    "eval_ok": int(eval_ok),
                    "eval_error": eval_err,
                    "integrity_ok": int(integrity_ok),
                    "program_integrity_ok": int(program_integrity_ok),
                    "max_abs_formula_vs_pickle": max_abs_pub,
                    "max_abs_program_vs_pickle": max_abs_prog,
                    "integrity_tol": tol,
                    "integrity_note": integrity_note,
                    "program_length": int(row["program_length"]),
                }
            )
    return pd.DataFrame(rows)


def score_observed_holdout(formulas: pd.DataFrame) -> pd.DataFrame:
    print("Loading ratios for observed holdout …")
    df = add_design_columns(load_ratios())
    _, te = load_or_make_split(df)
    df_te = df.iloc[te].reset_index(drop=True)

    ceil_path = ngboost_holdout_metrics_path()
    if not ceil_path.is_file():
        raise FileNotFoundError(f"Missing NGBoost holdout metrics: {ceil_path}")
    ceiling = pd.read_csv(ceil_path).set_index("metric")

    # Precompute SR mu and log_sigma predictions per metric for NLL
    mu_hat: dict[str, np.ndarray] = {}
    logsig_hat: dict[str, np.ndarray] = {}
    rows = []

    for metric in METRICS:
        y = log_response(df_te, metric)
        for target in SR_TARGETS:
            fr = formulas[(formulas["metric"] == metric) & (formulas["target"] == target)]
            row = fr.iloc[0]
            colnames = parse_feature_list(row["features"])
            formula = str(row["formula_reported"])
            feat = make_feature_dict(df_te, colnames)
            try:
                y_hat = eval_formula_reported(formula, feat)
                eval_ok = True
                eval_err = ""
            except Exception as exc:  # noqa: BLE001
                y_hat = np.full(len(df_te), np.nan)
                eval_ok = False
                eval_err = str(exc)

            if target == "mu":
                mu_hat[metric] = y_hat
            elif target == "log_sigma":
                logsig_hat[metric] = y_hat

            m = np.isfinite(y) & np.isfinite(y_hat)
            rec = {
                "metric": metric,
                "target": target,
                "formula_source": row["formula_source"],
                "n_test": int(m.sum()),
                "eval_ok": int(eval_ok),
                "eval_error": eval_err,
                "r2_vs_y": float("nan"),
                "rmse_vs_y": float("nan"),
                "pinball": float("nan"),
                "nll_sr": float("nan"),
                "ngb_r2_mean": float(ceiling.loc[metric, "r2_mean"])
                if metric in ceiling.index
                else float("nan"),
                "ngb_rmse": float(ceiling.loc[metric, "rmse"])
                if metric in ceiling.index
                else float("nan"),
                "ngb_nll": float(ceiling.loc[metric, "nll"])
                if metric in ceiling.index
                else float("nan"),
                "ngb_pinball": float("nan"),
            }

            if target in ("mu", "q50") and m.sum() > 2:
                rec["r2_vs_y"] = r2_score(y[m], y_hat[m])
                rec["rmse_vs_y"] = rmse(y[m], y_hat[m])
            elif target in TAU_BY_TARGET and target != "q50" and m.sum():
                tau = TAU_BY_TARGET[target]
                rec["pinball"] = pinball_loss(y[m], y_hat[m], tau)
                col = f"pinball_q{int(tau * 100):02d}"
                if metric in ceiling.index and col in ceiling.columns:
                    rec["ngb_pinball"] = float(ceiling.loc[metric, col])
            # log_sigma: scored via paired NLL after both available

            rows.append(rec)

        # Paired NLL for this metric
        if metric in mu_hat and metric in logsig_hat:
            mu_s = mu_hat[metric]
            sig = np.exp(np.asarray(logsig_hat[metric], dtype=float))
            y = log_response(df_te, metric)
            m = np.isfinite(y) & np.isfinite(mu_s) & np.isfinite(sig)
            nll = normal_nll(y[m], mu_s[m], sig[m]) if m.sum() else float("nan")
            for rec in rows:
                if rec["metric"] == metric and rec["target"] == "log_sigma":
                    rec["nll_sr"] = nll
                    break
            # also attach nll_sr on mu row for convenience
            for rec in rows:
                if rec["metric"] == metric and rec["target"] == "mu":
                    rec["nll_sr"] = nll
                    break

    return pd.DataFrame(rows)


def main() -> None:
    out = out_dir("evaluate_sr")
    train = out_dir("train_sr")
    formulas = pd.read_csv(train / "sr_formulas.csv")
    fidelity = pd.read_csv(train / "sr_fidelity.csv")
    ols = pd.read_csv(train / "ols_baseline.csv")

    formulas.to_csv(out / "sr_formulas.csv", index=False)
    fidelity.to_csv(out / "sr_fidelity.csv", index=False)
    ols.to_csv(out / "ols_baseline.csv", index=False)

    merged = fidelity.merge(
        formulas[
            [
                "metric",
                "target",
                "formula_reported",
                "formula_source",
                "sr_collapsed",
                "program_length",
                "y_std",
            ]
        ],
        on=["metric", "target"],
        suffixes=("", "_f"),
    )
    if "sr_collapsed_f" in merged.columns:
        merged = merged.drop(columns=["sr_collapsed_f"], errors="ignore")

    print("Scoring formula_reported vs NGBoost surfaces (cell holdout) …")
    surface = score_surface_holdout(train, formulas)
    surface = surface.merge(
        fidelity[["metric", "target", "r2_test_reported"]].rename(
            columns={"r2_test_reported": "r2_train_reported"}
        ),
        on=["metric", "target"],
        how="left",
    )
    surface.to_csv(out / "surface_holdout.csv", index=False)

    print("Scoring formulas vs observed Y (seed holdout) …")
    observed = score_observed_holdout(formulas)
    observed.to_csv(out / "observed_holdout.csv", index=False)

    still_bad = merged[merged["r2_test_reported"] < 0.5]
    integrity_fail = surface[
        (surface["integrity_ok"] == 0) | (surface["program_integrity_ok"] == 0)
    ]
    surface_gap = surface[
        np.isfinite(surface["r2_formula_vs_surface"])
        & np.isfinite(surface["r2_train_reported"])
        & (np.abs(surface["r2_formula_vs_surface"] - surface["r2_train_reported"]) > 0.02)
    ]

    lines = [
        "# Symbolic regression evaluation",
        "",
        "## Positioning",
        "",
        "Reported formulas are **practitioner compressions distilled from the "
        "NGBoost surrogate**. They are engineering approximations of predictive "
        r"surfaces \(\mu\), \(\log\sigma\), and \(q_\tau\) on the factorial grid—"
        "**not** a replacement for NGBoost or QBM.",
        "",
        "**Primary correctness check:** does `formula_reported` match the NGBoost "
        "surface on held-out design cells?",
        "",
        "**Secondary check:** on held-out seeds, how do the same formulas score "
        r"against observed \(Y=\ln\chi\), relative to full NGBoost (ceiling)?",
        "",
        "## Definitions",
        "",
        r"- Surface holdout: unique cell × node grid; same cell GroupShuffleSplit "
        r"as `train_sr` (`CELL_SPLIT_SEED`).",
        r"- Observed holdout: seed-grouped split matching `chi_ngboost` / `chi_qbm`.",
        rf"- Collapse (training): GP test \(R^2 < {COLLAPSE_R2_THRESH}\) → OLS reported.",
        rf"- Integrity (published): \(\max|\hat y_{{\mathrm{{formula}}}}-"
        rf"\hat y_{{\mathrm{{pickle}}}}| \le \max({INTEGRITY_MAX_ABS_TOL:g},"
        rf" {INTEGRITY_REL_TOL:g}\,|y_{{\mathrm{{std}}}}|)\) (string rounding slack).",
        r"- Integrity (program): same tolerance on exact `y_mean`/`y_std` wrap of "
        r"`formula_z` (gplearn prints rounded constants).",
        r"- Observed: \(R^2\)/RMSE for \(\mu\)/\(q_{0.50}\); pinball for \(q_{0.05}\)/"
        r"\(q_{0.95}\); Normal NLL from paired \(\hat\mu_{\mathrm{SR}}\), "
        r"\(\hat\sigma=\exp(\widehat{\log\sigma}_{\mathrm{SR}})\).",
        "",
        "## GP vs OLS vs reported (from training)",
        "",
        merged[
            [
                "metric",
                "target",
                "r2_test_gp",
                "r2_test_ols",
                "r2_test_reported",
                "sr_collapsed",
                "formula_source",
                "y_std",
                "program_length",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    lines += _section(
        surface,
        "Primary: formula vs NGBoost surface (held-out cells)",
        [
            "metric",
            "target",
            "r2_formula_vs_surface",
            "r2_train_reported",
            "rmse_formula_vs_surface",
            "integrity_ok",
            "program_integrity_ok",
            "max_abs_formula_vs_pickle",
            "formula_source",
        ],
    )
    lines += [
        "### Formula ↔ pickle integrity failures",
        "",
    ]
    if len(integrity_fail):
        lines.append(
            integrity_fail[
                [
                    "metric",
                    "target",
                    "max_abs_formula_vs_pickle",
                    "max_abs_program_vs_pickle",
                    "integrity_tol",
                    "integrity_ok",
                    "program_integrity_ok",
                    "integrity_note",
                    "eval_ok",
                ]
            ].to_markdown(index=False, floatfmt=".4g")
        )
    else:
        lines.append(
            "None — published formulas match pickles within scale-aware tolerance; "
            "exact program wraps match within absolute tolerance."
        )
    lines += ["", r"### Surface \(R^2\) gaps vs training-reported (>0.02)", ""]
    if len(surface_gap):
        lines.append(
            surface_gap[
                [
                    "metric",
                    "target",
                    "r2_formula_vs_surface",
                    "r2_train_reported",
                ]
            ].to_markdown(index=False, floatfmt=".4f")
        )
    else:
        lines.append("None.")
    lines.append("")

    lines += _section(
        observed[observed["target"].isin(["mu", "q50"])],
        r"Secondary: observed \(Y\) — \(\mu\) / \(q_{0.50}\) (seed holdout)",
        [
            "metric",
            "target",
            "r2_vs_y",
            "rmse_vs_y",
            "ngb_r2_mean",
            "ngb_rmse",
            "nll_sr",
            "ngb_nll",
        ],
    )
    lines += _section(
        observed[observed["target"].isin(["q05", "q95"])],
        r"Secondary: observed \(Y\) — pinball \(q_{0.05}\) / \(q_{0.95}\)",
        ["metric", "target", "pinball", "ngb_pinball", "n_test"],
    )
    lines += _section(
        observed[observed["target"] == "log_sigma"],
        r"Secondary: paired Normal NLL from \(\hat\mu_{\mathrm{SR}}\) + \(\widehat{\log\sigma}_{\mathrm{SR}}\)",
        ["metric", "target", "nll_sr", "ngb_nll", "n_test"],
    )

    lines += [
        r"## Residual weak training fits (\(R^2<0.5\))",
        "",
    ]
    if len(still_bad):
        lines.append(
            still_bad[
                ["metric", "target", "r2_test_reported", "r2_test_ols", "formula_source"]
            ].to_markdown(index=False, floatfmt=".4f")
        )
    else:
        lines.append("None.")
    lines += [
        "",
        "## Conclusions",
        "",
        "- **Surface holdout** is the correctness check for the published equations: "
        "`r2_formula_vs_surface` should track `r2_train_reported`, and integrity "
        "should pass.",
        "- **Observed holdout** stacks distillation error with NGBoost model error; "
        r"compare SR metrics to the NGBoost ceiling columns, not to perfect \(R^2=1\).",
        "- Prefer `formula_reported` for paper text only where surface fidelity is "
        "adequate; primary inference remains NGBoost.",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(
        surface[
            [
                "metric",
                "target",
                "r2_formula_vs_surface",
                "r2_train_reported",
                "integrity_ok",
                "program_integrity_ok",
                "max_abs_formula_vs_pickle",
                "max_abs_program_vs_pickle",
            ]
        ].to_string(index=False)
    )
    print()
    print(
        observed[observed["target"].isin(["mu", "log_sigma", "q05", "q95"])][
            [
                "metric",
                "target",
                "r2_vs_y",
                "pinball",
                "nll_sr",
                "ngb_r2_mean",
                "ngb_nll",
                "ngb_pinball",
            ]
        ].to_string(index=False)
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
