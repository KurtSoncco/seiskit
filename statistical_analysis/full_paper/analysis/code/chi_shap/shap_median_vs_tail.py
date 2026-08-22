"""Median vs tail SHAP deltas for QBM and NGBoost.

Contrasts are τ=0.05−τ=0.50 and τ=0.95−τ=0.50, each as a 2×5 figure (abs /
signed). QBM reuses TreeSHAP CSVs from ``chi_shap/shap_qbm``. NGBoost explains
the Normal quantile \(q_\tau=\mu+z_\tau\sigma\) with permutation SHAP (same
subsample recipe as ``shap_ngboost.py``). Pass ``--force`` to recompute
NGBoost quantile SHAP instead of loading cached CSVs.

Writes under figure_dir("chi_shap", "shap_median_vs_tail").
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.ticker import MaxNLocator
from ngboost import NGBRegressor
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FEATURES,
    METRICS,
    add_design_columns,
    importance_table,
    load_or_make_split,
    load_ratios,
    make_shap_sample,
    ngboost_model_path,
    out_dir,
    qbm_model_path,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import (  # noqa: E402
    LABEL_FONTSIZE,
    TICK_LABELSIZE,
    add_panel_label,
    apply_full_paper_style,
    factor_color,
    figsize,
    metric_label,
    save_figure,
)

warnings.filterwarnings("ignore")
apply_full_paper_style(auto_format=True, frame="open", grid=False)

FORCE = "--force" in sys.argv
NGB_EXPLAIN_N = 400
NGB_BG_N = 100
NGB_SAMPLE_SEED = 3

FEATURE_DISPLAY = {
    "Vs1_z": r"$V_{s1}$",
    "Height_z": r"$H$",
    "CoV_z": "CoV",
    "rH_z": r"$r_{h}$",
    "aHV_z": r"$a_{hv}$",
    "node_z": r"$x_{\mathrm{node}}$",
}

# (id, tail target, row label) — Δ = tail − median
CONTRASTS = (
    ("q05_minus_q50", "q05", r"$q_{0.05}-q_{0.50}$"),
    ("q95_minus_q50", "q95", r"$q_{0.95}-q_{0.50}$"),
)
QUANTILE_TAUS = {"q05": 0.05, "q50": 0.50, "q95": 0.95}


def _as_booster(obj) -> lgb.Booster:
    if isinstance(obj, lgb.Booster):
        return obj
    if hasattr(obj, "booster_"):
        return obj.booster_
    return obj


def _load_or_compute_qbm_importance(target: str) -> pd.DataFrame:
    """Load shap_qbm CSV for *target* (`q05` / `q50` / `q95`), else compute TreeSHAP."""
    shap_dir = out_dir("shap_qbm")
    path = shap_dir / f"shap_importance_{target}.csv"
    if path.is_file():
        tab = pd.read_csv(path)
        if len(tab) and {"metric", "feature", "mean_abs_shap"}.issubset(tab.columns):
            print(f"Loaded {path}")
            return tab

    print(f"Computing TreeSHAP for {target} …")
    df = add_design_columns(load_ratios())
    _, te = load_or_make_split(df)
    _, ex_idx, _ = make_shap_sample(df, te)
    X_ex = df.iloc[ex_idx][FEATURES].to_numpy(dtype=float)
    rows = []
    for metric in METRICS:
        mpath = qbm_model_path(target, metric)
        if not mpath.is_file():
            print(f"  missing {mpath}")
            continue
        booster = _as_booster(joblib.load(mpath))
        explainer = shap.TreeExplainer(booster)
        sv = np.asarray(explainer.shap_values(X_ex), dtype=float)
        rows.append(importance_table(sv, FEATURES, metric=metric, model="qbm", target=target))
    if not rows:
        raise FileNotFoundError(f"No QBM models / SHAP CSVs for target={target}")
    return pd.concat(rows, ignore_index=True)


def _ngb_holdout_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Match ``shap_ngboost.py``: 100 background / 400 explain rows."""
    _, te = load_or_make_split(df)
    bg_idx, ex_idx, _ = make_shap_sample(df, te)
    rng = np.random.default_rng(NGB_SAMPLE_SEED)
    if len(ex_idx) > NGB_EXPLAIN_N:
        ex_idx = rng.choice(ex_idx, size=NGB_EXPLAIN_N, replace=False)
    if len(bg_idx) > NGB_BG_N:
        bg_idx = rng.choice(bg_idx, size=NGB_BG_N, replace=False)
    X_bg = df.iloc[bg_idx][FEATURES].to_numpy(dtype=float)
    X_ex = df.iloc[ex_idx][FEATURES].to_numpy(dtype=float)
    return X_bg, X_ex


def _explain_ngb(predict_fn, X_bg: np.ndarray, X_ex: np.ndarray) -> np.ndarray:
    explainer = shap.Explainer(predict_fn, X_bg, algorithm="permutation")
    explanation = explainer(X_ex, max_evals=2 * X_ex.shape[1] + 1)
    return np.asarray(explanation.values, dtype=float)


def _load_or_compute_ngboost_quantiles() -> dict[str, pd.DataFrame]:
    """Permutation SHAP of \(q_\\tau=\\mu+z_\\tau\\sigma\) via μ and σ composition.

    SHAP is linear, so \(\phi(q_\\tau)=\phi(\\mu)+z_\\tau\phi(\\sigma)\). Caching
    writes ``shap_importance_q{05,50,95}.csv`` under ``shap_ngboost``.
    """
    ngb_dir = out_dir("shap_ngboost")
    paths = {t: ngb_dir / f"shap_importance_{t}.csv" for t in QUANTILE_TAUS}
    if not FORCE and all(p.is_file() for p in paths.values()):
        out = {}
        for t, p in paths.items():
            tab = pd.read_csv(p)
            if not len(tab) or not {"metric", "feature", "mean_abs_shap"}.issubset(tab.columns):
                break
            print(f"Loaded {p}")
            out[t] = tab
        else:
            return out

    print("Computing NGBoost permutation SHAP for μ and σ (compose q05/q50/q95) …")
    df = add_design_columns(load_ratios())
    X_bg, X_ex = _ngb_holdout_xy(df)
    z05 = float(norm.ppf(0.05))
    z95 = float(norm.ppf(0.95))
    rows: dict[str, list[pd.DataFrame]] = {t: [] for t in QUANTILE_TAUS}

    for metric in METRICS:
        mpath = ngboost_model_path(metric)
        if not mpath.is_file():
            raise FileNotFoundError(f"Missing NGBoost model: {mpath}")
        model: NGBRegressor = joblib.load(mpath)

        def predict_mu(X, m=model):
            dist = m.pred_dist(np.asarray(X, dtype=float))
            return np.asarray(dist.loc, dtype=float).ravel()

        def predict_sigma(X, m=model):
            dist = m.pred_dist(np.asarray(X, dtype=float))
            return np.maximum(np.asarray(dist.scale, dtype=float).ravel(), 1e-8)

        print(f"  SHAP NGBoost μ {metric} …")
        sv_mu = _explain_ngb(predict_mu, X_bg, X_ex)
        print(f"  SHAP NGBoost σ {metric} …")
        sv_sig = _explain_ngb(predict_sigma, X_bg, X_ex)
        composed = {
            "q50": sv_mu,
            "q05": sv_mu + z05 * sv_sig,
            "q95": sv_mu + z95 * sv_sig,
        }
        for t, sv in composed.items():
            rows[t].append(
                importance_table(sv, FEATURES, metric=metric, model="ngboost", target=t)
            )

    out = {}
    for t, chunks in rows.items():
        tab = pd.concat(chunks, ignore_index=True)
        tab.to_csv(paths[t], index=False)
        print(f"Wrote {paths[t]}")
        out[t] = tab
    return out


def _feature_color(feat: str) -> str:
    if feat == "node_z":
        return "0.35"
    try:
        return factor_color(feat)
    except KeyError:
        return "0.45"


def _column_xlim(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (-0.05, 0.05)
    lo, hi = float(np.min(finite)), float(np.max(finite))
    span = max(hi - lo, max(abs(lo), abs(hi), 1e-12))
    pad = 0.10 * span
    lo, hi = lo - pad, hi + pad
    lo = min(lo, 0.0)
    hi = max(hi, 0.0)
    if hi - lo < 1e-15:
        return (-0.05, 0.05)
    return lo, hi


def _plot_deltas(
    diff: pd.DataFrame,
    *,
    value_col: str,
    xlabel: str,
    stem: str,
    out: Path,
) -> None:
    """2×5: rows = (0.05−0.50, 0.95−0.50), columns = χ metrics."""
    n_metrics = len(METRICS)
    fig, axes = plt.subplots(
        2,
        n_metrics,
        figsize=figsize(height=4.85),
        sharey=True,
        squeeze=False,
    )
    fig.subplots_adjust(left=0.12, right=0.995, bottom=0.08, top=0.96, wspace=0.06, hspace=0.10)

    y = np.arange(len(FEATURES))
    yticklabels = [FEATURE_DISPLAY.get(f, f) for f in FEATURES]
    colors = [_feature_color(f) for f in FEATURES]

    for col, metric in enumerate(METRICS):
        col_vals = []
        for _row, (cid, _tail, _lab) in enumerate(CONTRASTS):
            sub = diff[(diff["metric"] == metric) & (diff["contrast"] == cid)]
            sub = sub.set_index("feature").reindex(FEATURES)
            col_vals.append(sub[value_col].to_numpy(dtype=float))
        xlim = _column_xlim(np.concatenate(col_vals))

        for row, ((cid, _tail, row_lab), vals) in enumerate(zip(CONTRASTS, col_vals)):
            ax = axes[row, col]
            ax.barh(y, vals, color=colors, height=0.7, edgecolor="none")
            ax.axvline(0.0, color="0.55", linewidth=0.5)
            ax.set_yticks(y)
            ax.set_xlim(*xlim)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3, prune=None))
            ax.tick_params(labelsize=TICK_LABELSIZE, length=2.0)
            add_panel_label(ax, row * n_metrics + col, x=0.97, y=0.97, alpha=0.75)
            if row == 0:
                ax.set_title(metric_label(metric, log=True), fontsize=LABEL_FONTSIZE, pad=1)
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(
                    xlabel if col == n_metrics // 2 else "",
                    fontsize=LABEL_FONTSIZE,
                    labelpad=1,
                )
            if col == 0:
                ax.set_yticklabels(yticklabels, fontsize=TICK_LABELSIZE)
                ax.set_ylabel(row_lab, fontsize=LABEL_FONTSIZE, labelpad=1)
            else:
                ax.tick_params(axis="y", left=False, labelleft=False, length=0)
                ax.set_ylabel("")

    axes[0, 0].invert_yaxis()
    save_figure(fig, stem, out_dir=out)
    plt.close(fig)


def _build_diff(by_target: dict[str, pd.DataFrame]) -> pd.DataFrame:
    q50 = by_target["q50"]
    rows = []
    for metric in METRICS:
        med = q50[q50["metric"] == metric].set_index("feature")
        for cid, tail_key, _lab in CONTRASTS:
            tail = by_target[tail_key]
            hi = tail[tail["metric"] == metric].set_index("feature")
            for feat in FEATURES:
                if feat not in med.index or feat not in hi.index:
                    continue
                abs_med = float(med.loc[feat, "mean_abs_shap"])
                abs_tail = float(hi.loc[feat, "mean_abs_shap"])
                signed_med = float(med.loc[feat, "mean_signed_shap"])
                signed_tail = float(hi.loc[feat, "mean_signed_shap"])
                rows.append(
                    {
                        "metric": metric,
                        "feature": feat,
                        "contrast": cid,
                        "tau": QUANTILE_TAUS[tail_key],
                        "mean_abs_shap_q50": abs_med,
                        "mean_abs_shap_tau": abs_tail,
                        "mean_signed_shap_q50": signed_med,
                        "mean_signed_shap_tau": signed_tail,
                        "delta_mean_abs_shap": abs_tail - abs_med,
                        "delta_mean_signed_shap": signed_tail - signed_med,
                        "abs_ratio_tau_over_q50": (
                            abs_tail / abs_med if abs_med > 0 else np.nan
                        ),
                    }
                )
    diff = pd.DataFrame(rows)
    if not len(diff):
        raise RuntimeError("No overlapping q05/q50/q95 SHAP rows to compare.")
    diff["rank_abs_delta"] = (
        diff.groupby(["metric", "contrast"])["delta_mean_abs_shap"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    return diff.sort_values(["contrast", "metric", "rank_abs_delta"])


def _write_pair(
    diff: pd.DataFrame,
    out: Path,
    *,
    csv_name: str,
    stem_abs: str,
    stem_signed: str,
) -> None:
    diff.to_csv(out / csv_name, index=False)
    _plot_deltas(
        diff,
        value_col="delta_mean_abs_shap",
        xlabel="SHAP value",
        stem=stem_abs,
        out=out,
    )
    _plot_deltas(
        diff,
        value_col="delta_mean_signed_shap",
        xlabel="SHAP value",
        stem=stem_signed,
        out=out,
    )


def _summary_block(title: str, source: str, diff: pd.DataFrame, files: list[tuple[str, str]]) -> list[str]:
    top = (
        diff.sort_values("delta_mean_abs_shap", ascending=False)
        .groupby(["contrast", "metric"], as_index=False)
        .head(3)
    )
    file_rows = "\n".join(f"| `{name}` | {desc} |" for name, desc in files)
    return [
        f"# {title}",
        "",
        "## Definitions",
        "",
        source,
        r"- Contrasts: \(\Delta\overline{|\phi|}=\overline{|\phi|}_\tau-\overline{|\phi|}_{0.50}\) "
        r"and \(\Delta\overline{\phi}=\overline{\phi}_\tau-\overline{\phi}_{0.50}\) "
        r"for \(\tau\in\{0.05,0.95\}\).",
        r"- Layout: one 2×5 figure per aggregation (abs / signed); rows = "
        r"\(0.05-0.50\) and \(0.95-0.50\); columns = χ metrics.",
        r"- Positive \(\Delta\overline{|\phi|}\): feature is more important at that "
        r"tail quantile than at the median.",
        "",
        "## Largest |SHAP| increases toward each tail",
        "",
        top.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Output files",
        "",
        "| File | Content |",
        "|------|---------|",
        file_rows,
        "",
    ]


def main() -> None:
    out = out_dir("shap_median_vs_tail")

    qbm = {t: _load_or_compute_qbm_importance(t) for t in QUANTILE_TAUS}
    qbm_diff = _build_diff(qbm)
    _write_pair(
        qbm_diff,
        out,
        csv_name="shap_median_vs_tail.csv",
        stem_abs="shap_median_vs_tail_delta_abs",
        stem_signed="shap_median_vs_tail_delta_signed",
    )

    ngb = _load_or_compute_ngboost_quantiles()
    ngb_diff = _build_diff(ngb)
    _write_pair(
        ngb_diff,
        out,
        csv_name="shap_median_vs_tail_ngboost.csv",
        stem_abs="shap_median_vs_tail_ngboost_delta_abs",
        stem_signed="shap_median_vs_tail_ngboost_delta_signed",
    )

    lines = _summary_block(
        "QBM SHAP: median (τ=0.50) vs tails (τ=0.05, τ=0.95)",
        r"- Source: **QBM** TreeSHAP from `chi_shap/shap_qbm` (computed on demand if CSVs are missing).",
        qbm_diff,
        [
            ("shap_median_vs_tail.csv", "QBM q05/q50/q95 SHAP and both tail−median deltas"),
            ("shap_median_vs_tail_delta_abs.pdf", "QBM 2×5 Δ mean |SHAP|"),
            ("shap_median_vs_tail_delta_signed.pdf", "QBM 2×5 Δ signed SHAP"),
        ],
    )
    lines += [
        "## Conclusions (QBM)",
        "",
        "- Features with large positive \\(\\Delta\\overline{|\\phi|}\\) are candidate "
        "tail drivers; near-zero deltas indicate rank-stable importance.",
        "- Lower-tail (\\(0.05-0.50\\)) and upper-tail (\\(0.95-0.50\\)) rows need "
        "not agree.",
        "- Signed deltas can flip when the tail model reverses the average direction "
        "of a feature's contribution relative to the median.",
        "",
    ]
    lines += _summary_block(
        "NGBoost SHAP: median (τ=0.50) vs tails (τ=0.05, τ=0.95)",
        r"- Source: **NGBoost** permutation SHAP of the Normal quantile "
        r"\(q_\tau(\mathbf{x})=\mu(\mathbf{x})+z_\tau\sigma(\mathbf{x})\), with "
        r"\(\phi(q_\tau)=\phi(\mu)+z_\tau\phi(\sigma)\) (same holdout subsample as "
        r"`shap_ngboost.py`). Cached as `chi_shap/shap_ngboost/shap_importance_q*.csv`.",
        ngb_diff,
        [
            ("shap_median_vs_tail_ngboost.csv", "NGBoost q05/q50/q95 SHAP and both tail−median deltas"),
            ("shap_median_vs_tail_ngboost_delta_abs.pdf", "NGBoost 2×5 Δ mean |SHAP|"),
            ("shap_median_vs_tail_ngboost_delta_signed.pdf", "NGBoost 2×5 Δ signed SHAP"),
        ],
    )
    lines += [
        "## Conclusions (NGBoost)",
        "",
        r"- Because \(q_{0.50}\equiv\mu\) and \(z_{0.05}=-z_{0.95}\), signed "
        r"\(\Delta\overline{\phi}\) for the two tails are exact opposites "
        r"(\(\Delta_{0.05}=-\Delta_{0.95}=z_{0.05}\overline{\phi}(\sigma)\)).",
        r"- Absolute deltas are **not** mirrors: \(\overline{|\phi(\mu)+z\phi(\sigma)|}"
        r"-\overline{|\phi(\mu)|}\) depends on the per-row alignment of μ and σ "
        r"attributions.",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("QBM")
    print(qbm_diff.to_string(index=False))
    print("NGBoost")
    print(ngb_diff.to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
