"""1D ALE marginal effects of NGBoost μ on central IM (Y = ln χ), per partition (Fig16).

Every design factor has three experimental levels, so ALE is evaluated only at
those levels and drawn as markers (no connecting line; a line would imply a
continuous response between levels that the design never sampled).

Figures under ``figure_dir("chi_shap", "ale_effects", <partition>)``:
``ale_<metric>.pdf``, one panel per design factor.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ngboost import NGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FEATURE_DISPLAY,
    METRICS,
    NGB_FEATURES,
    PARTITION_LABELS,
    PARTITIONS,
    factor_levels,
    format_level,
    load_partition,
    ngboost_model_path,
    out_dir,
    partition_split,
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

ALE_SUBSAMPLE_N = 4000
ALE_SAMPLE_SEED = 4
MARKER_SIZE = 4.0


def _predict_ngb_mu(model: NGBRegressor, X: np.ndarray) -> np.ndarray:
    dist = model.pred_dist(np.asarray(X, dtype=float))
    return np.asarray(dist.loc, dtype=float).ravel()


def load_ngb(partition: str, metric: str) -> NGBRegressor:
    path = ngboost_model_path(partition, metric)
    if not path.is_file():
        raise FileNotFoundError(f"Missing NGBoost model: {path}. Run train_ngboost.py first.")
    return joblib.load(path)


def ale_1d(predict_fn, X: np.ndarray, j: int) -> tuple[np.ndarray, np.ndarray]:
    """Centered ALE of feature column *j* at its observed (discrete) levels.

    Local effect between consecutive levels uses the rows at the upper level;
    effects accumulate from the lowest level and are centered by level counts.
    """
    xj = np.asarray(X[:, j], dtype=float)
    levels = np.unique(xj[np.isfinite(xj)])
    counts = np.array([np.sum(xj == v) for v in levels], dtype=float)
    effects = [0.0]
    for k in range(1, levels.size):
        mask = xj == levels[k]
        X_lo = X[mask].copy()
        X_hi = X[mask].copy()
        X_lo[:, j] = levels[k - 1]
        X_hi[:, j] = levels[k]
        effects.append(float(np.mean(predict_fn(X_hi) - predict_fn(X_lo))))
    ale = np.cumsum(effects)
    ale = ale - float(np.sum(ale * counts) / max(counts.sum(), 1.0))
    return levels, ale


def ale_subsample(df: pd.DataFrame, partition: str) -> np.ndarray:
    _, te = partition_split(df, partition)
    rng = np.random.default_rng(ALE_SAMPLE_SEED)
    te = np.asarray(te)
    if len(te) > ALE_SUBSAMPLE_N:
        te = rng.choice(te, size=ALE_SUBSAMPLE_N, replace=False)
    return df.iloc[te][NGB_FEATURES].to_numpy(dtype=float)


def plot_ale_levels(
    curves: dict[str, np.ndarray],
    raw_levels: dict[str, np.ndarray],
    *,
    title: str,
    ylabel: str,
    stem: str,
    out: Path,
) -> None:
    """Marker-only ALE per factor at categorical positions labelled by raw level."""
    n = len(NGB_FEATURES)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize(height=min(6.5, 2.0 * nrows)), squeeze=False, sharey=True
    )
    for i, feat in enumerate(NGB_FEATURES):
        ax = axes[i // ncols, i % ncols]
        y = curves[feat]
        pos = np.arange(len(y))
        ax.plot(
            pos,
            y,
            linestyle="none",
            marker="o",
            markersize=MARKER_SIZE,
            color=factor_color(feat),
        )
        ax.axhline(0.0, color="0.55", linewidth=0.5, linestyle=":")
        ax.set_xticks(pos)
        ax.set_xticklabels([format_level(v) for v in raw_levels[feat]], fontsize=TICK_LABELSIZE)
        ax.set_xlim(-0.5, len(y) - 0.5)
        ax.set_xlabel(FEATURE_DISPLAY.get(feat, feat), fontsize=LABEL_FONTSIZE)
        if i % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
        add_panel_label(ax, i)
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)
    fig.suptitle(title, fontsize=LABEL_FONTSIZE, y=0.99)
    fig.tight_layout(pad=0.35, rect=(0, 0, 1, 0.95))
    save_figure(fig, stem, out_dir=out)
    plt.close(fig)


def amplitude_table(tab: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    return (
        tab.groupby(by, as_index=False)["effect"]
        .agg(effect_min="min", effect_max="max")
        .assign(effect_range=lambda d: d["effect_max"] - d["effect_min"])
        .sort_values([*by[:-1], "effect_range"], ascending=[True] * (len(by) - 1) + [False])
    )


def run_partition(partition: str) -> None:
    out = out_dir("ale_effects", partition)
    print(f"Loading {PARTITION_LABELS[partition]} …")
    df = load_partition(partition)
    levels = factor_levels(df)
    raw = {f: levels[f][1] for f in NGB_FEATURES}
    X = ale_subsample(df, partition)

    rows = []
    for metric in METRICS:
        model = load_ngb(partition, metric)
        predict_fn = lambda Xq, m=model: _predict_ngb_mu(m, Xq)  # noqa: E731
        print(f"ALE [{partition}] {metric} …")
        curves = {}
        for j, feat in enumerate(NGB_FEATURES):
            lv, ale = ale_1d(predict_fn, X, j)
            curves[feat] = ale
            for z, r, y in zip(lv, raw[feat], ale):
                rows.append(
                    {
                        "metric": metric,
                        "model": "ngboost_mu",
                        "feature": feat,
                        "level_z": float(z),
                        "level": float(r),
                        "effect": float(y),
                    }
                )
        plot_ale_levels(
            curves,
            raw,
            title=f"{metric_label(metric, log=True)} — NGBoost $\\mu$, {PARTITION_LABELS[partition]}",
            ylabel=r"ALE on $Y$",
            stem=f"ale_{metric}",
            out=out,
        )

    tab = pd.DataFrame(rows)
    tab.to_csv(out / "ale_curves.csv", index=False)
    amp = amplitude_table(tab, ["metric", "model", "feature"])
    amp.to_csv(out / "ale_effect_range.csv", index=False)
    meta = {"partition": partition, "subsample_n": int(len(X)), "model": "ngboost_mu"}
    (out / "ale_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    lines = [
        f"# ALE marginal effects (central IM) — {PARTITION_LABELS[partition]}",
        "",
        "## Definitions",
        "",
        r"- Response: \(Y=\ln\chi\); model: NGBoost \(\mu\) for this partition.",
        r"- Estimator: 1D accumulated local effects at the three observed levels of each "
        "design factor, centered by level counts. Drawn as markers only.",
        "- Features: five z-scored design factors (no `node_z`). Evaluation on a holdout subsample.",
        "",
        "## Effect amplitude (max − min over the three levels)",
        "",
        amp.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Output files",
        "",
        "| File | Content |",
        "|------|---------|",
        "| `ale_curves.csv` | ALE per metric × factor level |",
        "| `ale_effect_range.csv` | amplitude summary |",
        "| `ale_<metric>.pdf` | marker ALE per factor |",
        "| `ale_meta.json` | subsample size |",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


def main() -> None:
    for partition in PARTITIONS:
        run_partition(partition)


if __name__ == "__main__":
    main()
