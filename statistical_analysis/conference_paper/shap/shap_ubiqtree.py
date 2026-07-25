"""UbiQTree epistemic / aleatoric / entanglement SHAP for seed LightGBM models.

Implements the full Dubey et al. (arXiv:2508.09639) pipeline via
``ubiqtree_lgbm`` on stored ``models/*_seed.pkl`` Boosters — no retraining.

Runs on mean-GBM and QBM tails τ ∈ {0.05, 0.50, 0.95} for log_abs and f_ratio.

Outputs (under ``results/shap/``)
--------------------------------
Data
    ubiqtree_summary.csv
Plots
    ubiq_bars_{tag}.png          — mean SHAP ± 2σ
    ubiq_decompose_{tag}.png     — aleatoric / epistemic / entanglement
    ubiq_metrics_{tag}.png       — σ, entropy, sign stability, DST ignorance
    ubiq_violin_{tag}.png        — Dirichlet-sample SHAP violins

Usage
-----
    python shap_ubiqtree.py
    python shap_ubiqtree.py --n-draws 100 --n-explain 150
    python shap_ubiqtree.py --force-shap
    python shap_ubiqtree.py --mean-only
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import UbiQTree core before putting conference_paper on sys.path — that path
# would shadow the real ``shap`` library with this directory's package name.
from ubiqtree_lgbm import e_shap, per_tree_shap, result_to_rows  # noqa: E402

from seiskit.plot_config import apply_style, panel_letter, result_path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    CACHE_DIR,
    FACTOR_COLORS,
    FACTORS,
    FIG_DPI,
    FIG_WIDTH,
    REF_COLOR,
    figsize,
    load_channel50,
    load_mean_models,
    load_quantile_models,
    seed_grouped_split,
    target_color,
    target_label,
)

warnings.filterwarnings("ignore")

FORCE = "--force-shap" in sys.argv
MEAN_ONLY = "--mean-only" in sys.argv


def _arg_int(flag: str, default: int) -> int:
    if flag in sys.argv:
        i = sys.argv.index(flag)
        return int(sys.argv[i + 1])
    return default


def _arg_float(flag: str, default: float) -> float:
    if flag in sys.argv:
        i = sys.argv.index(flag)
        return float(sys.argv[i + 1])
    return default


N_DRAWS = _arg_int("--n-draws", 200)
N_EXPLAIN = _arg_int("--n-explain", 250)
ALPHA = _arg_float("--alpha", 0.5)
BETA = _arg_float("--beta", 5.0)
TAIL_TAUS = [0.05, 0.50, 0.95]
TARGETS = ["log_abs", "f_ratio"]


def _save(fig, name: str) -> None:
    out = result_path("plots", name)
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot {name}")


def _cached_phi_trees(key: str, booster, X: np.ndarray) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.npy"
    if not FORCE and path.exists():
        arr = np.load(path)
        if arr.shape[1] == len(X) and arr.shape[2] == X.shape[1]:
            return arr
    print(f"  computing per-tree SHAP → {key}")
    arr = per_tree_shap(booster, X, progress=True)
    np.save(path, arr)
    return arr


def _plot_bars(result, tag: str, title: str) -> None:
    order = np.argsort(result.mean)
    y = np.arange(len(FACTORS))
    fig, ax = plt.subplots(figsize=figsize(height=3.8))
    colors = [FACTOR_COLORS[FACTORS[i]] for i in order]
    ax.barh(
        y,
        result.mean[order],
        xerr=2.0 * result.std[order],
        color=colors,
        ecolor=REF_COLOR,
        capsize=3,
        alpha=0.9,
    )
    ax.axvline(0, color=REF_COLOR, lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([FACTORS[i] for i in order])
    ax.set_xlabel(r"mean SHAP $\pm 2\sigma$ (Dirichlet sub-ensembles)")
    ax.set_title(title)
    _save(fig, f"ubiq_bars_{tag}.png")


def _plot_decompose(result, tag: str, title: str, tgt: str) -> None:
    x = np.arange(len(FACTORS))
    w = 0.25
    fig, ax = plt.subplots(figsize=figsize(height=3.8))
    ax.bar(x - w, result.aleatoric, width=w, color=REF_COLOR, label="aleatoric $A$")
    ax.bar(
        x,
        result.epistemic,
        width=w,
        color=target_color(tgt),
        alpha=0.75,
        label="epistemic $E$",
    )
    ax.bar(
        x + w,
        result.entanglement,
        width=w,
        color="#AA3377",
        alpha=0.85,
        label="entanglement $C$",
    )
    ax.axhline(0, color=REF_COLOR, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(FACTORS)
    ax.set_ylabel("variance component")
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=False)
    _save(fig, f"ubiq_decompose_{tag}.png")


def _plot_metrics(result, tag: str, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=figsize(height=FIG_WIDTH * 0.75), layout="constrained")
    order = np.argsort(np.abs(result.mean))
    names = [FACTORS[i] for i in order]
    panels = [
        (axes[0, 0], result.std[order], r"$\sigma$ (SHAP)", "a"),
        (axes[0, 1], result.entropy[order], "explanation entropy", "b"),
        (axes[1, 0], result.sign_stability[order], "sign stability", "c"),
        (axes[1, 1], result.ignorance[order], r"DST ignorance (Pl$-$Bel)", "d"),
    ]
    for ax, vals, ylab, letter in panels:
        ax.barh(names, vals, color=[FACTOR_COLORS[f] for f in names], alpha=0.9)
        ax.set_xlabel(ylab)
        panel_letter(ax, letter)
        if ylab.startswith("sign"):
            ax.set_xlim(0, 1)
            ax.axvline(0.9, color="green", ls="--", lw=0.9, alpha=0.8)
            ax.axvline(0.7, color="orange", ls="--", lw=0.9, alpha=0.8)
    _save(fig, f"ubiq_metrics_{tag}.png")


def _plot_violin(result, tag: str, title: str) -> None:
    if result.samples_adj is None:
        return
    flat = result.samples_adj.reshape(-1, result.samples_adj.shape[-1])
    order = np.argsort(np.abs(result.mean))
    data = [flat[:, j] for j in order]
    labels = [FACTORS[j] for j in order]
    fig, ax = plt.subplots(figsize=figsize(height=3.8))
    parts = ax.violinplot(data, showmeans=True, showextrema=True, vert=False)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(FACTOR_COLORS[labels[i]])
        body.set_alpha(0.7)
    ax.set_yticks(np.arange(1, len(labels) + 1))
    ax.set_yticklabels(labels)
    ax.axvline(0, color=REF_COLOR, lw=0.8)
    ax.set_xlabel("Dirichlet-sample SHAP")
    ax.set_title(title)
    _save(fig, f"ubiq_violin_{tag}.png")


def _run_one(booster, X, tag: str, title: str, rows: list, extra: dict) -> None:
    phi = _cached_phi_trees(f"ubiq_phi_trees_{tag}", booster, X)
    result = e_shap(
        booster,
        X,
        phi_trees=phi,
        n_draws=N_DRAWS,
        alpha=ALPHA,
        beta=BETA,
        random_state=0,
        store_samples=True,
        progress=True,
    )
    rows.extend(result_to_rows(result, FACTORS, extra=extra))
    print(
        f"  {tag}: mean |A|={np.mean(np.abs(result.aleatoric)):.4g} "
        f"|E|={np.mean(np.abs(result.epistemic)):.4g} "
        f"|C|={np.mean(np.abs(result.entanglement)):.4g}"
    )
    tgt = extra["target"]
    _plot_bars(result, tag, title)
    _plot_decompose(result, tag, f"SHAP variance decomposition — {title}", tgt)
    _plot_metrics(result, tag, f"UbiQTree uncertainty metrics — {title}")
    _plot_violin(result, tag, f"SHAP Dirichlet distribution — {title}")


def main() -> None:
    t0 = time.monotonic()
    apply_style(auto_format=True, font_size=10, frame="open")

    d50 = load_channel50()
    tr, te = seed_grouped_split(d50, test_size=0.25, seed=0)
    Xte = d50[FACTORS].iloc[te].reset_index(drop=True)

    rng = np.random.default_rng(0)
    n_ex = min(N_EXPLAIN, len(Xte))
    explain_idx = rng.choice(len(Xte), size=n_ex, replace=False)
    X_ex = Xte.iloc[explain_idx].to_numpy()

    mean_models = load_mean_models(targets=TARGETS, split_by="seed")
    quant_models = (
        {} if MEAN_ONLY else load_quantile_models(taus=TAIL_TAUS, targets=TARGETS, split_by="seed")
    )

    rows: list[dict] = []
    print(f"\n=== UbiQTree (n_explain={n_ex}, n_draws={N_DRAWS}, α={ALPHA}, β={BETA}) ===")

    for tgt in TARGETS:
        if tgt not in mean_models:
            raise FileNotFoundError(f"Missing mean model for {tgt}")
        tag = f"mean_{tgt}"
        title = f"{target_label(tgt)} mean GBM"
        print(f"\n--- {tag} ---")
        _run_one(
            mean_models[tgt],
            X_ex,
            tag,
            title,
            rows,
            extra=dict(model="mean", target=tgt, tau=np.nan),
        )

        if MEAN_ONLY:
            continue
        for tau in TAIL_TAUS:
            if tau not in quant_models.get(tgt, {}):
                raise FileNotFoundError(f"Missing quantile model {tgt}@τ={tau}")
            tag = f"q{int(tau * 100):02d}_{tgt}"
            title = rf"{target_label(tgt)} QBM $\tau={tau:g}$"
            print(f"\n--- {tag} ---")
            _run_one(
                quant_models[tgt][tau],
                X_ex,
                tag,
                title,
                rows,
                extra=dict(model="quantile", target=tgt, tau=tau),
            )

    out = result_path("data", "ubiqtree_summary.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("  data ubiqtree_summary.csv")
    print(f"\nDone in {time.monotonic() - t0:.0f}s. Cache: {CACHE_DIR}")


if __name__ == "__main__":
    main()
