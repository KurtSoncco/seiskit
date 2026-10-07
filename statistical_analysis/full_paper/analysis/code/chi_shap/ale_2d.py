"""2D ALE on dominant factor pairs at the observed 3×3 levels, per partition (Fig20).

Pairs: (rH_z, CoV_z), (Vs1_z, Height_z). Target: NGBoost μ for each partition
(between-seed, within-seed). Both factors have three experimental levels, so
the surface is a discrete 3×3 cell grid (no contour interpolation between
levels).

Writes under ``figure_dir("chi_shap", "ale_2d", <partition>)``.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ale_effects import _predict_ngb_mu, load_ngb  # noqa: E402
from common import (  # noqa: E402
    METRICS,
    NGB_FEATURES,
    PARTITION_LABELS,
    PARTITIONS,
    factor_levels,
    format_level,
    load_partition,
    out_dir,
    partition_split,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import (  # noqa: E402
    LABEL_FONTSIZE,
    TICK_LABELSIZE,
    add_panel_label,
    apply_full_paper_style,
    figsize,
    metric_label,
    save_figure,
)

from seiskit.plot_config import get_crameri_cmap  # noqa: E402

warnings.filterwarnings("ignore")
apply_full_paper_style(auto_format=True, frame="open", grid=False)

ALE_SUBSAMPLE_N = 3000
ALE_SAMPLE_SEED = 5
PAIRS = (
    ("rH_z", "CoV_z", r"$r_h$", "CoV"),
    ("Vs1_z", "Height_z", r"$V_{s1}$", r"$H$"),
)


def ale_2d(predict_fn, X: np.ndarray, j: int, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centered second-order ALE on the discrete level grid of columns *j*, *k*.

    Returns (levels_j, levels_k, effect) with effect shape (n_j, n_k); the
    lowest row/column accumulate from zero before count-weighted centering.
    """
    xj = np.asarray(X[:, j], dtype=float)
    xk = np.asarray(X[:, k], dtype=float)
    lj = np.unique(xj[np.isfinite(xj)])
    lk = np.unique(xk[np.isfinite(xk)])
    nj, nk = lj.size, lk.size
    counts = np.array([[np.sum((xj == a) & (xk == b)) for b in lk] for a in lj], dtype=float)
    local = np.zeros((nj, nk), dtype=float)
    for a in range(1, nj):
        for b in range(1, nk):
            mask = (xj == lj[a]) & (xk == lk[b])
            if not np.any(mask):
                continue
            Xm = X[mask]
            corners = {}
            for da in (0, 1):
                for db in (0, 1):
                    Xc = Xm.copy()
                    Xc[:, j] = lj[a - da]
                    Xc[:, k] = lk[b - db]
                    corners[(da, db)] = predict_fn(Xc)
            delta = corners[(0, 0)] - corners[(1, 0)] - corners[(0, 1)] + corners[(1, 1)]
            local[a, b] = float(np.mean(delta))
    ale = np.cumsum(np.cumsum(local, axis=0), axis=1)
    ale = ale - float(np.sum(ale * counts) / max(counts.sum(), 1.0))
    return lj, lk, ale


def _plot_metric(
    surfaces: dict, raw: dict[str, np.ndarray], *, metric: str, partition: str, out: Path
) -> None:
    fig, axes = plt.subplots(1, len(PAIRS), figsize=figsize(height=3.0), squeeze=False)
    cmap = get_crameri_cmap("vik")
    for col, (fj, fk, lab_j, lab_k) in enumerate(PAIRS):
        ax = axes[0, col]
        Z = surfaces[(fj, fk)]
        vmax = max(float(np.max(np.abs(Z))), 1e-6)
        im = ax.imshow(
            Z,
            cmap=cmap,
            norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )
        for a in range(Z.shape[0]):
            for b in range(Z.shape[1]):
                ax.text(
                    b,
                    a,
                    f"{Z[a, b]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=TICK_LABELSIZE,
                    color="white" if abs(Z[a, b]) > 0.6 * vmax else "0.1",
                )
        ax.set_xticks(range(Z.shape[1]))
        ax.set_yticks(range(Z.shape[0]))
        ax.set_xticklabels([format_level(v) for v in raw[fk]], fontsize=TICK_LABELSIZE)
        ax.set_yticklabels([format_level(v) for v in raw[fj]], fontsize=TICK_LABELSIZE)
        ax.set_xlabel(lab_k, fontsize=LABEL_FONTSIZE)
        ax.set_ylabel(lab_j, fontsize=LABEL_FONTSIZE)
        ax.set_title(f"{lab_j} × {lab_k}", fontsize=TICK_LABELSIZE)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=TICK_LABELSIZE)
        add_panel_label(ax, col)
    fig.suptitle(
        f"{metric_label(metric, log=True)} — NGBoost $\\mu$, {PARTITION_LABELS[partition]}",
        fontsize=LABEL_FONTSIZE,
        y=0.99,
    )
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.92))
    save_figure(fig, f"ale_2d_{metric}", out_dir=out)
    plt.close(fig)


def run_partition(partition: str) -> None:
    out = out_dir("ale_2d", partition)
    print(f"Loading {PARTITION_LABELS[partition]} …")
    df = load_partition(partition)
    raw = {f: v[1] for f, v in factor_levels(df).items()}
    _, te = partition_split(df, partition)
    rng = np.random.default_rng(ALE_SAMPLE_SEED)
    te = np.asarray(te)
    if len(te) > ALE_SUBSAMPLE_N:
        te = rng.choice(te, size=ALE_SUBSAMPLE_N, replace=False)
    X = df.iloc[te][NGB_FEATURES].to_numpy(dtype=float)
    feat_idx = {f: i for i, f in enumerate(NGB_FEATURES)}

    rows = []
    for metric in METRICS:
        print(f"2D ALE [{partition}] {metric} …")
        model = load_ngb(partition, metric)
        predict_fn = lambda Xq, m=model: _predict_ngb_mu(m, Xq)  # noqa: E731
        surfaces = {}
        for fj, fk, _, _ in PAIRS:
            _, _, Z = ale_2d(predict_fn, X, feat_idx[fj], feat_idx[fk])
            surfaces[(fj, fk)] = Z
            for a, vj in enumerate(raw[fj]):
                for b, vk in enumerate(raw[fk]):
                    rows.append(
                        {
                            "metric": metric,
                            "target": "ngboost_mu",
                            "feature_i": fj,
                            "feature_j": fk,
                            "level_i": float(vj),
                            "level_j": float(vk),
                            "effect": float(Z[a, b]),
                        }
                    )
        _plot_metric(surfaces, raw, metric=metric, partition=partition, out=out)

    pd.DataFrame(rows).to_csv(out / "ale_2d_surfaces.csv", index=False)
    meta = {
        "partition": partition,
        "subsample_n": int(len(X)),
        "pairs": [list(p[:2]) for p in PAIRS],
        "target": "ngboost_mu",
    }
    (out / "ale_2d_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(
        "\n".join(
            [
                f"# 2D ALE on the 3×3 level grid — {PARTITION_LABELS[partition]} (Fig20)",
                "",
                "Pairs: $r_h\\times CoV$, $V_{s1}\\times H$. Target: NGBoost $\\mu$.",
                "Each cell is one combination of observed levels; no interpolation between levels.",
                "Dispersion / tail ALE is in `ale_dispersion/` (Fig18).",
                "",
                "| File | Content |",
                "|------|---------|",
                "| `ale_2d_<metric>.pdf` | 1×2 discrete 3×3 grids |",
                "| `ale_2d_surfaces.csv` | grid values |",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out}")


def main() -> None:
    for partition in PARTITIONS:
        run_partition(partition)


if __name__ == "__main__":
    main()
