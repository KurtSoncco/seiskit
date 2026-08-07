"""Exceedance probabilities and Friedman H-statistics for NGBoost / QBM.

Exceedance: P(Y > t) from Normal NGBoost surfaces (or live predictions) for
thresholds such as Y>0 (χ>1) and Y>ln(1.5). Friedman H: pairwise interaction
strength from residual variance of 2D vs 1D PDPs on NGBoost μ (fallback: QBM
mean), evaluated on a holdout subsample for tractability.

Writes under figure_dir("chi_ngboost", "exceedance_friedman").
"""

from __future__ import annotations

import itertools
import json
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CHI_QBM_MODELS,
    FEATURES,
    METRICS,
    add_design_columns,
    factorial_grid,
    load_or_make_split,
    load_ratios,
    models_dir,
    out_dir,
    surfaces_dir,
)
from train_ngboost import predict_params  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import (  # noqa: E402
    add_panel_label,
    apply_full_paper_style,
    figsize,
    metric_color,
    metric_label,
    save_figure,
)

warnings.filterwarnings("ignore")
apply_full_paper_style(auto_format=True, frame="open", grid=False)

THRESHOLDS = (
    ("Y_gt_0", 0.0, r"$P(Y>0)=P(\chi>1)$"),
    ("Y_gt_ln_1_5", float(np.log(1.5)), r"$P(Y>\ln 1.5)=P(\chi>1.5)$"),
)
H_SUBSAMPLE_N = 600
H_GRID_N = 8
H_TOP_FEATURES = 4
H_SAMPLE_SEED = 5


def _as_booster(obj) -> lgb.Booster:
    if isinstance(obj, lgb.Booster):
        return obj
    if hasattr(obj, "booster_"):
        return obj.booster_
    return obj


def _load_surface(metric: str) -> pd.DataFrame | None:
    path = surfaces_dir() / f"ngboost_surface_{metric}.csv"
    if path.is_file():
        return pd.read_csv(path)
    return None


def _surface_or_predict(metric: str, grid: pd.DataFrame) -> pd.DataFrame:
    surf = _load_surface(metric)
    if surf is not None and {"mu", "sigma"}.issubset(surf.columns):
        return surf
    mpath = models_dir() / f"ngboost_{metric}.pkl"
    if not mpath.is_file():
        raise FileNotFoundError(f"Need surface or model for {metric}")
    model: NGBRegressor = joblib.load(mpath)
    X = grid[FEATURES].to_numpy(dtype=float)
    mu, sigma = predict_params(model, X)
    out = grid.copy()
    out["mu"] = mu
    out["sigma"] = np.maximum(sigma, 1e-8)
    return out


def exceedance_table(surf: pd.DataFrame, metric: str) -> pd.DataFrame:
    mu = surf["mu"].to_numpy(dtype=float)
    sigma = np.maximum(surf["sigma"].to_numpy(dtype=float), 1e-8)
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
                "n_grid": int(len(p)),
            }
        )
    return pd.DataFrame(rows)


def _plot_exceedance_maps(surf: pd.DataFrame, metric: str, out: Path) -> None:
    """Mean exceedance vs node for each threshold (compact Nature panel)."""
    fig, axes = plt.subplots(1, len(THRESHOLDS), figsize=figsize(height=2.6), squeeze=False)
    mu = surf["mu"].to_numpy(dtype=float)
    sigma = np.maximum(surf["sigma"].to_numpy(dtype=float), 1e-8)
    nodes = surf["node"].to_numpy(dtype=float)
    for i, (key, thr, label) in enumerate(THRESHOLDS):
        ax = axes[0, i]
        p = 1.0 - stats.norm.cdf((thr - mu) / sigma)
        tmp = pd.DataFrame({"node": nodes, "p": p})
        by = tmp.groupby("node", as_index=False)["p"].mean().sort_values("node")
        ax.plot(by["node"], by["p"], color=metric_color(metric), linewidth=1.0)
        ax.set_xlabel("Node")
        ax.set_ylabel(r"Mean $P(Y>t)$")
        ax.set_title(label, fontsize=7)
        ax.set_ylim(0.0, 1.0)
        add_panel_label(ax, i)
    fig.suptitle(metric_label(metric, log=True), fontsize=7, y=1.02)
    fig.tight_layout(pad=0.35)
    save_figure(fig, f"exceedance_vs_node_{metric}", out_dir=out)
    plt.close(fig)


def _predict_mu_fn(metric: str):
    """Prefer NGBoost μ; fallback to QBM mean GBM."""
    npath = models_dir() / f"ngboost_{metric}.pkl"
    if npath.is_file():
        model: NGBRegressor = joblib.load(npath)

        def _pred(X, m=model):
            dist = m.pred_dist(np.asarray(X, dtype=float))
            return np.asarray(dist.loc, dtype=float).ravel()

        return _pred, "ngboost_mu"

    qpath = CHI_QBM_MODELS / f"lgbm_mean_{metric}_seed.pkl"
    if qpath.is_file():
        booster = _as_booster(joblib.load(qpath))

        def _pred(X, b=booster):
            return np.asarray(b.predict(X), dtype=float).ravel()

        return _pred, "qbm_mean"

    raise FileNotFoundError(f"No NGBoost or QBM mean model for {metric}")


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


def friedman_h_pair(
    predict_fn,
    X: np.ndarray,
    j: int,
    k: int,
    *,
    n_grid: int = H_GRID_N,
) -> float:
    """Friedman \(H_{jk}\) from centered 1D/2D PDPs on a quantile grid."""
    xj = X[:, j]
    xk = X[:, k]
    gj = np.unique(np.quantile(xj, np.linspace(0.05, 0.95, n_grid)))
    gk = np.unique(np.quantile(xk, np.linspace(0.05, 0.95, n_grid)))
    if gj.size < 2 or gk.size < 2:
        return float("nan")
    f_j = pdp_1d_on_grid(predict_fn, X, j, gj)
    f_k = pdp_1d_on_grid(predict_fn, X, k, gk)
    f_jk = pdp_2d_on_grid(predict_fn, X, j, k, gj, gk)
    # Broadcast mains onto the 2D grid
    resid = f_jk - f_j[:, None] - f_k[None, :]
    num = float(np.sum(resid**2))
    den = float(np.sum(f_jk**2))
    if den <= 1e-16:
        return float("nan")
    h2 = num / den
    return float(np.sqrt(max(h2, 0.0)))


def _feature_amplitude(predict_fn, X: np.ndarray, j: int, n_grid: int = H_GRID_N) -> float:
    grid = np.unique(np.quantile(X[:, j], np.linspace(0.05, 0.95, n_grid)))
    if grid.size < 2:
        return 0.0
    f = pdp_1d_on_grid(predict_fn, X, j, grid)
    return float(np.max(f) - np.min(f))


def _plot_h_heatmap(h_tab: pd.DataFrame, metric: str, feats: list[str], out: Path) -> None:
    p = len(feats)
    mat = np.full((p, p), np.nan)
    for _, r in h_tab.iterrows():
        if r["metric"] != metric:
            continue
        if r["feature_i"] not in feats or r["feature_j"] not in feats:
            continue
        a, b = feats.index(r["feature_i"]), feats.index(r["feature_j"])
        mat[a, b] = mat[b, a] = float(r["H"])
    np.fill_diagonal(mat, 0.0)

    fig, ax = plt.subplots(figsize=figsize(height=3.2))
    im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=max(0.5, float(np.nanmax(mat))))
    ax.set_xticks(range(p))
    ax.set_yticks(range(p))
    ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(feats, fontsize=6)
    ax.set_title(f"{metric_label(metric, log=True)}  Friedman $H$", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label(r"$H$", fontsize=7)
    fig.tight_layout(pad=0.35)
    save_figure(fig, f"friedman_H_{metric}", out_dir=out)
    plt.close(fig)


def main() -> None:
    out = out_dir("exceedance_friedman")
    print("Loading data …")
    df = add_design_columns(load_ratios())
    grid = factorial_grid(df)
    tr, te = load_or_make_split(df)
    rng = np.random.default_rng(H_SAMPLE_SEED)
    te = np.asarray(te)
    if len(te) > H_SUBSAMPLE_N:
        te_h = rng.choice(te, size=H_SUBSAMPLE_N, replace=False)
    else:
        te_h = te
    X_h = df.iloc[te_h][FEATURES].to_numpy(dtype=float)

    exc_rows = []
    h_rows = []
    meta = {
        "thresholds": [{"key": k, "Y": t} for k, t, _ in THRESHOLDS],
        "h_subsample_n": int(len(te_h)),
        "h_grid_n": H_GRID_N,
        "h_top_features": H_TOP_FEATURES,
        "models": [],
    }

    for metric in METRICS:
        print(f"Exceedance {metric} …")
        surf = _surface_or_predict(metric, grid)
        etab = exceedance_table(surf, metric)
        exc_rows.append(etab)
        _plot_exceedance_maps(surf, metric, out)

        # Node-level exceedance CSV (compact: mean P by node)
        mu = surf["mu"].to_numpy(dtype=float)
        sigma = np.maximum(surf["sigma"].to_numpy(dtype=float), 1e-8)
        node_rows = []
        for key, thr, _ in THRESHOLDS:
            p = 1.0 - stats.norm.cdf((thr - mu) / sigma)
            tmp = pd.DataFrame({"node": surf["node"].to_numpy(), "p": p})
            by = tmp.groupby("node")["p"].agg(mean_P="mean", std_P="std").reset_index()
            by.insert(0, "metric", metric)
            by.insert(1, "threshold_key", key)
            by.insert(2, "threshold_Y", thr)
            node_rows.append(by)
        pd.concat(node_rows, ignore_index=True).to_csv(
            out / f"exceedance_by_node_{metric}.csv", index=False
        )

        print(f"Friedman H {metric} …")
        predict_fn, tag = _predict_mu_fn(metric)
        amps = {f: _feature_amplitude(predict_fn, X_h, j) for j, f in enumerate(FEATURES)}
        top_feats = sorted(amps, key=amps.get, reverse=True)[:H_TOP_FEATURES]
        # Also include all pairs among top feats
        for fi, fj in itertools.combinations(top_feats, 2):
            j, k = FEATURES.index(fi), FEATURES.index(fj)
            h = friedman_h_pair(predict_fn, X_h, j, k, n_grid=H_GRID_N)
            h_rows.append(
                {
                    "metric": metric,
                    "model": tag,
                    "feature_i": fi,
                    "feature_j": fj,
                    "H": h,
                    "amp_i": amps[fi],
                    "amp_j": amps[fj],
                }
            )
        meta["models"].append(
            {
                "metric": metric,
                "h_model": tag,
                "top_features": top_feats,
                "feature_amplitudes": amps,
            }
        )
        h_metric = pd.DataFrame([r for r in h_rows if r["metric"] == metric])
        if len(h_metric):
            _plot_h_heatmap(h_metric, metric, top_feats, out)

    exc = pd.concat(exc_rows, ignore_index=True)
    htab = pd.DataFrame(h_rows)
    if len(htab):
        htab["rank"] = (
            htab.groupby("metric")["H"].rank(ascending=False, method="min").astype(int)
        )
        htab = htab.sort_values(["metric", "rank"])
    exc.to_csv(out / "exceedance_summary.csv", index=False)
    htab.to_csv(out / "friedman_H_pairs.csv", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    lines = [
        "# Exceedance probabilities and Friedman H",
        "",
        "## Definitions",
        "",
        r"- Predictive law: Normal NGBoost on \(Y=\ln\chi\); "
        r"\(P(Y>t)=1-\Phi((t-\mu)/\sigma)\).",
        r"- Thresholds: \(t=0\) (\(\chi>1\)) and \(t=\ln 1.5\) (\(\chi>1.5\)).",
        r"- Grid: factorial cell × node surfaces from `chi_ngboost/surfaces` when present.",
        r"- Friedman \(H_{jk}=\sqrt{\sum(f_{jk}-f_j-f_k)^2/\sum f_{jk}^2}\) on centered "
        r"1D/2D PDPs of NGBoost \(\mu\) (fallback: QBM mean), quantile grid, "
        rf"holdout subsample \(n={H_SUBSAMPLE_N}\), top-{H_TOP_FEATURES} features by 1D PDP amplitude.",
        "",
        "## Exceedance summary",
        "",
        exc.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Friedman H (top pairs)",
        "",
        (htab.head(30).to_markdown(index=False, floatfmt=".4f") if len(htab) else "_none_"),
        "",
        "## Conclusions",
        "",
        "- Exceedance maps summarize where the emulator assigns high probability of "
        "amplification (\(\\chi>1\) or \(>1.5\)) across the design × node grid.",
        "- Large Friedman \(H\) flags feature pairs whose joint PDP is not additive — "
        "candidates for interaction terms in SR / engineering approximations.",
        "- \(H\) is a PDP residual statistic on the learned mean surface; it is not a "
        "causal interaction test and does not address residual spatial lag-1.",
        "",
        "## Output files",
        "",
        "| File | Content |",
        "|------|---------|",
        "| `exceedance_summary.csv` | mean/median/p10/p90 of P(Y>t) |",
        "| `exceedance_by_node_<metric>.csv` | node-wise mean exceedance |",
        "| `exceedance_vs_node_<metric>.pdf` | mean P vs node |",
        "| `friedman_H_pairs.csv` | pairwise H on top features |",
        "| `friedman_H_<metric>.pdf` | H heatmap |",
        "| `meta.json` | thresholds, subsample, model tags |",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(exc.to_string(index=False))
    print(htab.head(20).to_string(index=False) if len(htab) else "no H rows")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
