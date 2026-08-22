"""2D ALE contour plots for dominant factor pairs (Fig20).

Pairs: (rH_z, CoV_z), (Vs1_z, Height_z).
Targets: QBM τ=0.50 and NGBoost μ (central response; both models).

Writes under ``figure_dir("chi_shap", "ale_2d")``.
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
from ale_effects import _as_booster, _predict_ngb_mu, _predict_qbm  # noqa: E402
from common import (  # noqa: E402
    FEATURES,
    METRICS,
    add_design_columns,
    load_or_make_split,
    load_ratios,
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
    figsize,
    metric_label,
    save_figure,
)

warnings.filterwarnings("ignore")
apply_full_paper_style(auto_format=True, frame="open", grid=False)

ALE_SUBSAMPLE_N = 3000
ALE_SAMPLE_SEED = 5
N_BINS = 12
PAIRS = (
    ("rH_z", "CoV_z", r"$r_h$", "CoV"),
    ("Vs1_z", "Height_z", r"$V_{s1}$", r"$H$"),
)
TARGETS = (
    ("qbm_q50", r"QBM $\tau=0.50$"),
    ("ngboost_mu", r"NGBoost $\mu$"),
)


def _load_predict(metric: str, target: str):
    if target == "qbm_q50":
        path = qbm_model_path("q50", metric)
        booster = _as_booster(joblib.load(path))
        return lambda X, b=booster: _predict_qbm(b, X)
    if target == "ngboost_mu":
        model: NGBRegressor = joblib.load(ngboost_model_path(metric))
        return lambda X, m=model: _predict_ngb_mu(m, X)
    raise ValueError(target)


def ale_2d(
    predict_fn,
    X: np.ndarray,
    j: int,
    k: int,
    *,
    n_bins: int = N_BINS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centered 2D ALE grid for feature columns j, k.

    Returns (centers_j, centers_k, effect_jk) with shape (n_j, n_k).
    """
    xj = np.asarray(X[:, j], dtype=float)
    xk = np.asarray(X[:, k], dtype=float)
    edges_j = np.unique(np.quantile(xj, np.linspace(0, 1, n_bins + 1)))
    edges_k = np.unique(np.quantile(xk, np.linspace(0, 1, n_bins + 1)))
    if edges_j.size < 3 or edges_k.size < 3:
        gj = np.unique(np.quantile(xj, np.linspace(0, 1, min(n_bins, max(edges_j.size, 2)))))
        gk = np.unique(np.quantile(xk, np.linspace(0, 1, min(n_bins, max(edges_k.size, 2)))))
        Z = np.zeros((gj.size, gk.size), dtype=float)
        Xw = X.copy()
        for a, vj in enumerate(gj):
            for b, vk in enumerate(gk):
                Xw[:, j] = vj
                Xw[:, k] = vk
                Z[a, b] = float(np.mean(predict_fn(Xw)))
        Z = Z - float(np.mean(Z))
        return gj, gk, Z

    nj, nk = edges_j.size - 1, edges_k.size - 1
    local = np.zeros((nj, nk), dtype=float)
    counts = np.zeros((nj, nk), dtype=float)
    for a in range(nj):
        lo_j, hi_j = edges_j[a], edges_j[a + 1]
        mask_j = (xj >= lo_j) & (xj < hi_j if a < nj - 1 else xj <= hi_j)
        for b in range(nk):
            lo_k, hi_k = edges_k[b], edges_k[b + 1]
            mask_k = (xk >= lo_k) & (xk < hi_k if b < nk - 1 else xk <= hi_k)
            mask = mask_j & mask_k
            if not np.any(mask):
                continue
            Xm = X[mask]
            corners = []
            for vj in (lo_j, hi_j):
                for vk in (lo_k, hi_k):
                    Xc = Xm.copy()
                    Xc[:, j] = vj
                    Xc[:, k] = vk
                    corners.append(predict_fn(Xc))
            f00, f01, f10, f11 = corners
            delta = (f11 - f10) - (f01 - f00)
            local[a, b] = float(np.mean(delta))
            counts[a, b] = float(mask.sum())

    ale = np.cumsum(np.cumsum(local, axis=0), axis=1)
    if np.any(counts > 0):
        w = counts / counts.sum()
        ale = ale - float(np.sum(ale * w))
    else:
        ale = ale - float(np.mean(ale))

    cj = 0.5 * (edges_j[:-1] + edges_j[1:])
    ck = 0.5 * (edges_k[:-1] + edges_k[1:])
    return cj, ck, ale


def _plot_metric(surfaces: dict, *, metric: str, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=figsize(height=5.8))
    panel = 0
    for row, (target, ttl) in enumerate(TARGETS):
        for col, (fj, fk, lj, lk) in enumerate(PAIRS):
            ax = axes[row, col]
            cj, ck, Z = surfaces[target][(fj, fk)]
            cf = ax.contourf(ck, cj, Z, levels=12, cmap="RdBu_r")
            fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xlabel(lk, fontsize=LABEL_FONTSIZE)
            ax.set_ylabel(lj, fontsize=LABEL_FONTSIZE)
            ax.set_title(f"{ttl}: {lj}×{lk}", fontsize=TICK_LABELSIZE)
            ax.tick_params(labelsize=TICK_LABELSIZE)
            add_panel_label(ax, panel)
            panel += 1
    fig.suptitle(metric_label(metric, log=True), fontsize=LABEL_FONTSIZE, y=1.01)
    fig.tight_layout(pad=0.4)
    save_figure(fig, f"ale_2d_{metric}", out_dir=out)
    plt.close(fig)


def main() -> None:
    out = out_dir("ale_2d")
    print("Loading data …")
    df = add_design_columns(load_ratios())
    _, te = load_or_make_split(df)
    rng = np.random.default_rng(ALE_SAMPLE_SEED)
    te = np.asarray(te)
    if len(te) > ALE_SUBSAMPLE_N:
        te = rng.choice(te, size=ALE_SUBSAMPLE_N, replace=False)
    X = df.iloc[te][FEATURES].to_numpy(dtype=float)
    feat_idx = {f: i for i, f in enumerate(FEATURES)}

    rows = []
    meta = {
        "subsample_n": int(len(te)),
        "n_bins": N_BINS,
        "pairs": [list(p[:2]) for p in PAIRS],
        "targets": [t for t, _ in TARGETS],
    }

    for metric in METRICS:
        print(f"2D ALE {metric} …")
        surfaces = {}
        for target, _ in TARGETS:
            predict_fn = _load_predict(metric, target)
            surfaces[target] = {}
            for fj, fk, _, _ in PAIRS:
                j, k = feat_idx[fj], feat_idx[fk]
                cj, ck, Z = ale_2d(predict_fn, X, j, k, n_bins=N_BINS)
                surfaces[target][(fj, fk)] = (cj, ck, Z)
                for a, vj in enumerate(cj):
                    for b, vk in enumerate(ck):
                        rows.append(
                            {
                                "metric": metric,
                                "target": target,
                                "feature_i": fj,
                                "feature_j": fk,
                                "x_i": float(vj),
                                "x_j": float(vk),
                                "effect": float(Z[a, b]),
                            }
                        )
        _plot_metric(surfaces, metric=metric, out=out)

    pd.DataFrame(rows).to_csv(out / "ale_2d_surfaces.csv", index=False)
    (out / "ale_2d_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(
        "\n".join(
            [
                "# 2D ALE interaction surfaces (Fig20)",
                "",
                "Pairs: $r_h\\times CoV$, $V_{s1}\\times H$.",
                r"Targets: QBM $\tau=0.50$ and NGBoost $\mu$ (central response).",
                r"Dispersion / tail ALE is in `ale_dispersion/` (Fig18).",
                "",
                "| File | Content |",
                "|------|---------|",
                "| `ale_2d_<metric>.pdf` | 2×2 contour panels |",
                "| `ale_2d_surfaces.csv` | grid values |",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
