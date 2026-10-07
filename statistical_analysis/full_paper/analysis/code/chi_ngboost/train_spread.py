"""Train NGBoost spread models over all replicates (Z = ln s).

- ``within``  — target \\(Z_W = \\ln s_W\\), one row per (cell, seed) over all
  100 seeds; \\(s_W\\) = SD of \\(Y\\) across the 101 nodes of that seed.
  Holdout = held-out seeds (same seed lists as the between-seed Y model).
  f0 is a two-part (hurdle) model: Bernoulli NGBoost for
  \\(p_0 = P(s_W < \\mathrm{tol})\\) plus the Normal model on nonzero rows.
- ``between`` — target \\(Z_B = \\ln s_B\\), one row per (cell, node) over all
  101 nodes; \\(s_B\\) = SD of \\(Y\\) across the 100 seeds at that node.
  Holdout = held-out contiguous node blocks.

Both use the five z-scored design factors and the learner settings of
``train_ngboost.py``. \\(\\mu\\) is the typical log spread and \\(\\sigma\\) its
replicate-to-replicate variability. Outputs go to
``figure_dir("chi_ngboost", "spread", <kind>)``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from ngboost import NGBClassifier
from ngboost.distns import Bernoulli
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration_crps_pit import crps_normal  # noqa: E402
from common import (  # noqa: E402
    CENTER_NODE,
    EARLY_STOPPING_ROUNDS,
    FACTORS,
    MAX_ESTIMATORS,
    METRICS,
    N_NODES,
    N_SEEDS,
    NGB_FEATURES,
    NODE_BLOCK,
    SPREAD_KINDS,
    SPREAD_LABELS,
    SPREAD_REPLICATE,
    SPREAD_ZERO_TOL,
    VAL_FRAC,
    VAL_SEED,
    load_spread,
    r2_score,
    rmse,
    spread_dir,
    spread_model_path,
    spread_split,
)
from train_ngboost import (  # noqa: E402
    BASE_LEARNER,
    fit_one,
    normal_nll,
    pi_coverage,
    predict_params,
)

warnings.filterwarnings("ignore")

# Fit the Bernoulli part only when zero spreads are common enough to matter.
MIN_ZERO_FRAC = 0.01


def representative_id(df: pd.DataFrame, kind: str) -> int:
    """Seed 1 (lowest seed) for within; center node for between."""
    if kind == "within":
        return int(df["seed"].min())
    return CENTER_NODE


def spread_ceiling(z: np.ndarray, cell: np.ndarray) -> float:
    """Signal-to-total ceiling of Z with the replicates (seeds or nodes) as noise."""
    ok = np.isfinite(z)
    work = pd.DataFrame({"cell": cell[ok], "z": z[ok]})
    g = work.groupby("cell")["z"]
    keep = g.transform("count") >= 2
    g = work.loc[keep].groupby("cell")["z"]
    signal = float(g.mean().var(ddof=1))
    noise = float(g.var(ddof=1).mean())
    return signal / (signal + noise)


def fit_zero(X_fit, z_fit, X_val, z_val) -> tuple[NGBClassifier, int]:
    model = NGBClassifier(
        Dist=Bernoulli,
        Base=BASE_LEARNER,
        n_estimators=MAX_ESTIMATORS,
        learning_rate=0.05,
        minibatch_frac=1.0,
        col_sample=1.0,
        verbose=False,
        random_state=0,
    )
    model.fit(X_fit, z_fit, X_val=X_val, Y_val=z_val, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    n_trees = int(getattr(model, "best_val_loss_itr", None) or model.n_estimators)
    return model, n_trees


def predict_p0(model: NGBClassifier | None, X: np.ndarray) -> np.ndarray:
    if model is None:
        return np.zeros(len(X))
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


def hurdle_pit(
    z: np.ndarray, zero: np.ndarray, mu: np.ndarray, sigma: np.ndarray, p0: np.ndarray
) -> np.ndarray:
    """Mid-distribution PIT of the two-part model (zero rows → p0/2)."""
    phi = stats.norm.cdf((z - mu) / np.maximum(sigma, 1e-8))
    return np.where(zero, 0.5 * p0, p0 + (1.0 - p0) * np.nan_to_num(phi))


def _empirical_sigma_ratio(df: pd.DataFrame, sigma_cell: pd.Series) -> float:
    """Median over cells of σ̂ / SD of Z across replicates (ddof=0, nonzero rows)."""
    s_emp = df.loc[np.isfinite(df["Z"])].groupby("cell")["Z"].std(ddof=0)
    s_emp = s_emp[s_emp > 0]
    return float(np.median(sigma_cell.loc[s_emp.index].to_numpy() / s_emp.to_numpy()))


def train_kind(kind: str) -> None:
    out = spread_dir(kind)
    rep_col = SPREAD_REPLICATE[kind]
    rows, pit_rows, pred_rows, cell_rows = [], [], [], []
    split_info: dict | None = None
    meta = {
        "kind": kind,
        "features": NGB_FEATURES,
        "max_estimators": MAX_ESTIMATORS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "zero_tol": SPREAD_ZERO_TOL,
        "metrics": {},
    }

    for metric in METRICS:
        print(f"=== spread [{kind}] {metric} ===")
        df = load_spread(kind, metric)
        tr, te = spread_split(df, kind)
        groups = df["group"].to_numpy()
        gss = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=VAL_SEED)
        fit_rel, val_rel = next(gss.split(tr, groups=groups[tr]))
        fit_idx, val_idx = tr[fit_rel], tr[val_rel]
        if split_info is None:
            split_info = {
                "kind": kind,
                "rows": f"cell x {rep_col}",
                "holdout_groups": "seed" if kind == "within" else f"node_block_{NODE_BLOCK}",
                "n_rows": int(len(df)),
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "test_groups": sorted(int(g) for g in np.unique(groups[te])),
                "val_groups": sorted(int(g) for g in np.unique(groups[val_idx])),
                "representative": {rep_col: representative_id(df, kind)},
            }

        X = df[NGB_FEATURES].to_numpy(dtype=float)
        z = df["Z"].to_numpy(dtype=float)
        zero = df["zero"].to_numpy(dtype=bool)
        nz = ~zero & np.isfinite(z)

        t0 = time.perf_counter()
        f_nz, v_nz = fit_idx[nz[fit_idx]], val_idx[nz[val_idx]]
        model, n_trees = fit_one(X[f_nz], z[f_nz], X[v_nz], z[v_nz])
        joblib.dump(model, spread_model_path(kind, metric))

        zero_model, n_trees_zero = None, 0
        zero_frac = float(zero.mean())
        if zero_frac >= MIN_ZERO_FRAC:
            zero_model, n_trees_zero = fit_zero(
                X[fit_idx], zero[fit_idx].astype(int), X[val_idx], zero[val_idx].astype(int)
            )
            joblib.dump(zero_model, spread_model_path(kind, metric, "zero"))
        fit_s = time.perf_counter() - t0
        print(
            f"  trees={n_trees}  zero_trees={n_trees_zero}  zero_frac={zero_frac:.3f}  {fit_s:.1f}s"
        )

        mu, sigma = predict_params(model, X)
        p0 = predict_p0(zero_model, X)
        pit = hurdle_pit(z, zero, mu, sigma, p0)

        te_nz = te[nz[te]]
        r2 = r2_score(z[te_nz], mu[te_nz])
        ceiling = spread_ceiling(z, df["cell"].to_numpy())
        cells = df.drop_duplicates("cell").set_index("cell")
        mu_c, sig_c = predict_params(model, cells[NGB_FEATURES].to_numpy(dtype=float))
        row = {
            "kind": kind,
            "metric": metric,
            "n_trees": n_trees,
            "fit_seconds": fit_s,
            "n_fit": int(len(f_nz)),
            "n_test": int(len(te_nz)),
            "nll": normal_nll(z[te_nz], mu[te_nz], sigma[te_nz]),
            "r2_mean": r2,
            "ceiling": ceiling,
            "efficiency": r2 / ceiling if ceiling > 0 else np.nan,
            "rmse": rmse(z[te_nz], mu[te_nz]),
            "pi90_coverage": pi_coverage(z[te_nz], mu[te_nz], sigma[te_nz], alpha=0.10),
            "crps": float(np.mean(crps_normal(z[te_nz], mu[te_nz], sigma[te_nz]))),
            "median_sigma_over_s": _empirical_sigma_ratio(df, pd.Series(sig_c, index=cells.index)),
            "zero_frac": zero_frac,
            "n_trees_zero": n_trees_zero,
            "brier_zero": np.nan,
            "auc_zero": np.nan,
            "zero_frac_test": float(zero[te].mean()),
            "p0_mean_test": float(p0[te].mean()),
        }
        if zero_model is not None:
            row["brier_zero"] = float(np.mean((p0[te] - zero[te]) ** 2))
            row["auc_zero"] = float(roc_auc_score(zero[te], p0[te]))
        rows.append(row)

        base = df[["cell", *FACTORS, rep_col]].copy()
        base.insert(0, "metric", metric)
        base["Z"], base["zero"], base["mu"], base["sigma"], base["p0"], base["pit"] = (
            z,
            zero,
            mu,
            sigma,
            p0,
            pit,
        )
        rep = base[rep_col] == representative_id(df, kind)
        pit_rows.append(base.loc[rep])
        test = base.iloc[te].copy()
        pred_rows.append(test)

        cell_tab = cells[list(FACTORS)].reset_index()
        cell_tab.insert(0, "metric", metric)
        cell_tab["mu"], cell_tab["sigma"] = mu_c, sig_c
        cell_tab["p0"] = predict_p0(zero_model, cells[NGB_FEATURES].to_numpy(dtype=float))
        cell_rows.append(cell_tab)
        meta["metrics"][metric] = {
            "n_trees": n_trees,
            "n_trees_zero": n_trees_zero,
            "fit_seconds": fit_s,
        }

    hold = pd.DataFrame(rows)
    hold.to_csv(out / "holdout_metrics.csv", index=False)
    pd.concat(pit_rows).to_csv(out / "representative_pit.csv", index=False)
    pd.concat(pred_rows).to_csv(out / "test_predictions.csv", index=False)
    pd.concat(cell_rows).to_csv(out / "cell_predictions.csv", index=False)
    (out / "split.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    pit_all = pd.concat(pit_rows)
    rep_id = split_info["representative"][rep_col]
    rep_stats = (
        pit_all.groupby("metric", sort=False)["pit"]
        .agg(
            median="median",
            frac_below_0p25=lambda p: float(np.mean(p < 0.25)),
            frac_above_0p75=lambda p: float(np.mean(p > 0.75)),
        )
        .reset_index()
    )
    if kind == "within":
        scope = (
            rf"- Rows: (cell, seed), all \(N_s={N_SEEDS}\) seeds. "
            rf"\(s_W\) = SD of \(Y\) across the {N_NODES} nodes of one seed (ddof=0)."
        )
        holdout = "- Holdout: 25 held-out seeds (same lists as the between-seed Y model)."
        sig_txt = (
            r"\(\hat\sigma_W\) = seed-to-seed variability of \(\ln s_W\) (realization uncertainty)."
        )
        zero_txt = (
            rf"- f0 hurdle: Bernoulli NGBoost for \(p_0 = P(s_W < {SPREAD_ZERO_TOL:g})\) "
            r"(every node has the same peak frequency); the Normal part is fitted on nonzero rows. "
            "Brier / AUC on the held-out seeds."
        )
    else:
        scope = (
            rf"- Rows: (cell, node), all \(N_x={N_NODES}\) nodes. "
            rf"\(s_B\) = SD of \(Y\) across the {N_SEEDS} seeds at one node (ddof=0)."
        )
        holdout = rf"- Holdout: 25% of contiguous {NODE_BLOCK}-node blocks."
        sig_txt = r"\(\hat\sigma_B\) = node-to-node variability of \(\ln s_B\)."
        zero_txt = "- No zero spreads; no hurdle part."
    lines = [
        f"# Spread model — {SPREAD_LABELS[kind]}",
        "",
        "## Definitions",
        "",
        scope,
        r"- Model: Normal NGBoost on \(Z=\ln s\) with the five z-scored design factors "
        "(no node feature); learner settings as in `train_ngboost.py`.",
        rf"- \(\exp\hat\mu\) = typical spread; {sig_txt}",
        holdout,
        zero_txt,
        r"- `ceiling`: signal-to-total ceiling of \(Z\) with the replicates as noise; "
        r"`efficiency` = holdout \(R^2(\hat\mu)\) / ceiling.",
        r"- `median_sigma_over_s`: median over cells of \(\hat\sigma\) / SD of \(Z\) across replicates.",
        rf"- `representative_pit.csv`: PIT of {rep_col} {rep_id} in each cell "
        r"(hurdle mid-PIT for f0: zero rows → \(p_0/2\)). Flat if that replicate is typical; "
        "it is a training replicate, so the check is in-sample.",
        "",
        "## Holdout metrics",
        "",
        hold.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"## Representative {rep_col} ({rep_col} {rep_id}) PIT across cells",
        "",
        rep_stats.to_markdown(index=False, floatfmt=".3f"),
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--kind", choices=[*SPREAD_KINDS, "all"], default="all")
    args = p.parse_args()
    for kind in SPREAD_KINDS if args.kind == "all" else (args.kind,):
        train_kind(kind)


if __name__ == "__main__":
    main()
