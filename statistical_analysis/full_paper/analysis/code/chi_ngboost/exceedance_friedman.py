"""Exceedance probabilities and Friedman H-statistics for the single-partition NGBoost.

For each partition (between-seed, within-seed):

- Exceedance: P(Y > t) from Normal NGBoost on the 243 design cells for
  thresholds Y>0 (χ>1) and Y>ln(1.5). Features carry no node position, so one
  prediction per cell; σ is that partition's dispersion only.
- Friedman H: pairwise interaction strength from residual variance of 2D vs
  1D PDPs on NGBoost μ, evaluated on a holdout subsample at the observed
  factor levels.

Writes under figure_dir("chi_ngboost", <partition>, "exceedance_friedman").
"""

from __future__ import annotations

import itertools
import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FACTORS,
    METRICS,
    NGB_FEATURES,
    PARTITION_LABELS,
    PARTITIONS,
    load_partition,
    model_path,
    out_dir,
    partition_split,
)
from train_ngboost import predict_params  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import apply_full_paper_style, metric_label, save_figure  # noqa: E402

FEATURE_DISPLAY = {
    "Vs1_z": r"$V_{s1}$",
    "Height_z": r"$H$",
    "CoV_z": "CoV",
    "rH_z": r"$r_h$",
    "aHV_z": r"$a_{hv}$",
}

warnings.filterwarnings("ignore")
apply_full_paper_style(auto_format=True, frame="open", grid=False)

THRESHOLDS = (
    ("Y_gt_0", 0.0, r"$P(Y>0)=P(\chi>1)$"),
    ("Y_gt_ln_1_5", float(np.log(1.5)), r"$P(Y>\ln 1.5)=P(\chi>1.5)$"),
)
H_SUBSAMPLE_N = 600
H_TOP_FEATURES = 4
H_SAMPLE_SEED = 5


def _load_model(partition: str, metric: str) -> NGBRegressor:
    mpath = model_path(partition, metric)
    if not mpath.is_file():
        raise FileNotFoundError(f"Missing NGBoost model: {mpath}")
    return joblib.load(mpath)


def cell_grid(df: pd.DataFrame) -> pd.DataFrame:
    """One row per design cell."""
    return df.drop_duplicates("cell")[["cell", *FACTORS, *NGB_FEATURES]].reset_index(drop=True)


def exceedance_table(mu: np.ndarray, sigma: np.ndarray, metric: str) -> pd.DataFrame:
    sigma = np.maximum(sigma, 1e-8)
    rows = []
    for key, thr, _label in THRESHOLDS:
        p = 1.0 - stats.norm.cdf((thr - mu) / sigma)
        rows.append(
            {
                "metric": metric,
                "threshold_key": key,
                "threshold_Y": thr,
                "mean_P": float(np.mean(p)),
                "median_P": float(np.median(p)),
                "p10_P": float(np.quantile(p, 0.10)),
                "p90_P": float(np.quantile(p, 0.90)),
                "frac_P_gt_0_5": float(np.mean(p > 0.5)),
                "n_cells": int(len(p)),
            }
        )
    return pd.DataFrame(rows)


def _predict_mu_fn(model: NGBRegressor):
    def _pred(X, m=model):
        dist = m.pred_dist(np.asarray(X, dtype=float))
        return np.asarray(dist.loc, dtype=float).ravel()

    return _pred


def _levels(X: np.ndarray, j: int) -> np.ndarray:
    return np.unique(X[:, j][np.isfinite(X[:, j])])


def pdp_1d_on_grid(predict_fn, X: np.ndarray, j: int, grid: np.ndarray) -> np.ndarray:
    vals = np.empty(grid.size, dtype=float)
    Xw = X.copy()
    for i, g in enumerate(grid):
        Xw[:, j] = g
        vals[i] = float(np.mean(predict_fn(Xw)))
    return vals - float(np.mean(vals))


def pdp_2d_on_grid(
    predict_fn, X: np.ndarray, j: int, k: int, grid_j: np.ndarray, grid_k: np.ndarray
) -> np.ndarray:
    vals = np.empty((grid_j.size, grid_k.size), dtype=float)
    Xw = X.copy()
    for a, gj in enumerate(grid_j):
        for b, gk in enumerate(grid_k):
            Xw[:, j] = gj
            Xw[:, k] = gk
            vals[a, b] = float(np.mean(predict_fn(Xw)))
    return vals - float(np.mean(vals))


def friedman_h_pair(predict_fn, X: np.ndarray, j: int, k: int) -> float:
    r"""Friedman \(H_{jk}\) from centered 1D/2D PDPs at the observed factor levels."""
    gj, gk = _levels(X, j), _levels(X, k)
    if gj.size < 2 or gk.size < 2:
        return float("nan")
    f_j = pdp_1d_on_grid(predict_fn, X, j, gj)
    f_k = pdp_1d_on_grid(predict_fn, X, k, gk)
    f_jk = pdp_2d_on_grid(predict_fn, X, j, k, gj, gk)
    resid = f_jk - f_j[:, None] - f_k[None, :]
    den = float(np.sum(f_jk**2))
    if den <= 1e-16:
        return float("nan")
    return float(np.sqrt(max(float(np.sum(resid**2)) / den, 0.0)))


def _feature_amplitude(predict_fn, X: np.ndarray, j: int) -> float:
    grid = _levels(X, j)
    if grid.size < 2:
        return 0.0
    f = pdp_1d_on_grid(predict_fn, X, j, grid)
    return float(np.max(f) - np.min(f))


def _plot_h_heatmap(h_tab: pd.DataFrame, metric: str, feats: list[str], out: Path) -> None:
    p = len(feats)
    mat = np.full((p, p), np.nan)
    for _, r in h_tab.iterrows():
        if r["feature_i"] not in feats or r["feature_j"] not in feats:
            continue
        a, b = feats.index(r["feature_i"]), feats.index(r["feature_j"])
        mat[a, b] = mat[b, a] = float(r["H"])
    np.fill_diagonal(mat, 0.0)

    labels = [FEATURE_DISPLAY.get(f, f) for f in feats]
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=max(0.5, float(np.nanmax(mat))))
    ax.set_xticks(range(p))
    ax.set_yticks(range(p))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(f"{metric_label(metric, log=True)}  Friedman $H$", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label(r"$H$", fontsize=7)
    fig.tight_layout(pad=0.35)
    save_figure(fig, f"friedman_H_{metric}", out_dir=out)
    plt.close(fig)


def run_partition(partition: str) -> None:
    out = out_dir("exceedance_friedman", partition)
    print(f"Loading {PARTITION_LABELS[partition]} …")
    df = load_partition(partition)
    grid = cell_grid(df)
    X_grid = grid[NGB_FEATURES].to_numpy(dtype=float)
    _, te = partition_split(df, partition)
    rng = np.random.default_rng(H_SAMPLE_SEED)
    te = np.asarray(te)
    te_h = rng.choice(te, size=H_SUBSAMPLE_N, replace=False) if len(te) > H_SUBSAMPLE_N else te
    X_h = df.iloc[te_h][NGB_FEATURES].to_numpy(dtype=float)

    exc_rows, h_rows, cell_rows = [], [], []
    meta = {
        "partition": partition,
        "thresholds": [{"key": k, "Y": t} for k, t, _ in THRESHOLDS],
        "h_subsample_n": int(len(te_h)),
        "h_top_features": H_TOP_FEATURES,
        "models": [],
    }

    for metric in METRICS:
        model = _load_model(partition, metric)
        print(f"Exceedance [{partition}] {metric} …")
        mu, sigma = predict_params(model, X_grid)
        exc_rows.append(exceedance_table(mu, sigma, metric))
        per_cell = grid[["cell", *FACTORS]].copy()
        per_cell.insert(0, "metric", metric)
        per_cell["mu"] = mu
        per_cell["sigma"] = np.maximum(sigma, 1e-8)
        for key, thr, _ in THRESHOLDS:
            per_cell[f"P_{key}"] = 1.0 - stats.norm.cdf((thr - mu) / per_cell["sigma"])
        cell_rows.append(per_cell)

        print(f"Friedman H [{partition}] {metric} …")
        predict_fn = _predict_mu_fn(model)
        amps = {f: _feature_amplitude(predict_fn, X_h, j) for j, f in enumerate(NGB_FEATURES)}
        top_feats = sorted(amps, key=amps.get, reverse=True)[:H_TOP_FEATURES]
        for fi, fj in itertools.combinations(top_feats, 2):
            j, k = NGB_FEATURES.index(fi), NGB_FEATURES.index(fj)
            h_rows.append(
                {
                    "metric": metric,
                    "model": "ngboost_mu",
                    "feature_i": fi,
                    "feature_j": fj,
                    "H": friedman_h_pair(predict_fn, X_h, j, k),
                    "amp_i": amps[fi],
                    "amp_j": amps[fj],
                }
            )
        meta["models"].append(
            {"metric": metric, "top_features": top_feats, "feature_amplitudes": amps}
        )
        h_metric = pd.DataFrame([r for r in h_rows if r["metric"] == metric])
        if len(h_metric):
            _plot_h_heatmap(h_metric, metric, top_feats, out)

    exc = pd.concat(exc_rows, ignore_index=True)
    htab = pd.DataFrame(h_rows)
    if len(htab):
        htab["rank"] = htab.groupby("metric")["H"].rank(ascending=False, method="min").astype(int)
        htab = htab.sort_values(["metric", "rank"])
    exc.to_csv(out / "exceedance_summary.csv", index=False)
    pd.concat(cell_rows, ignore_index=True).to_csv(out / "exceedance_by_cell.csv", index=False)
    htab.to_csv(out / "friedman_H_pairs.csv", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    lines = [
        f"# Exceedance probabilities and Friedman H — {PARTITION_LABELS[partition]}",
        "",
        "## Definitions",
        "",
        r"- Predictive law: Normal NGBoost on \(Y=\ln\chi\); "
        r"\(P(Y>t)=1-\Phi((t-\mu)/\sigma)\), one prediction per design cell.",
        r"- Thresholds: \(t=0\) (\(\chi>1\)) and \(t=\ln 1.5\) (\(\chi>1.5\)).",
        r"- Friedman \(H_{jk}=\sqrt{\sum(f_{jk}-f_j-f_k)^2/\sum f_{jk}^2}\) on centered "
        r"1D/2D PDPs of NGBoost \(\mu\) at the three observed levels of each factor, "
        rf"holdout subsample \(n={len(te_h)}\), top-{H_TOP_FEATURES} factors by 1D PDP amplitude.",
        "",
        "## Exceedance summary",
        "",
        exc.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Friedman H (top pairs)",
        "",
        (htab.head(30).to_markdown(index=False, floatfmt=".4f") if len(htab) else "_none_"),
        "",
        "## Output files",
        "",
        "| File | Content |",
        "|------|---------|",
        "| `exceedance_summary.csv` | mean/median/p10/p90 of P(Y>t) over cells |",
        "| `exceedance_by_cell.csv` | μ, σ, P(Y>t) per design cell |",
        "| `friedman_H_pairs.csv` | pairwise H on top factors |",
        "| `friedman_H_<metric>.pdf` | H heatmap |",
        "| `meta.json` | thresholds, subsample, amplitudes |",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(exc.to_string(index=False))
    print(f"Wrote {out}")


def main() -> None:
    for partition in PARTITIONS:
        run_partition(partition)


if __name__ == "__main__":
    main()
