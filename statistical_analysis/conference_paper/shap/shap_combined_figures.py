"""Combined paper figures from the seed-split SHAP suite.

Joins f_ratio and log_abs into three figures (no τ∈{0.10, 0.90}):

1. Quantile |SHAP| importance share vs τ (1×2)
2. Quantile beeswarm at τ∈{0.05, 0.50, 0.95} (2×3)
3. Interval ΔSHAP sensitivity + τ-dynamics (2×2)

Reuses ``shap_seed_q_*`` TreeSHAP caches from ``shap_seed_suite.py`` —
never retrains. Pass ``--force-shap`` to recompute.

Usage
-----
    python shap_combined_figures.py
    python shap_combined_figures.py --force-shap
"""

from __future__ import annotations

import string
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import shap
from seiskit.plot_config import apply_style, get_crameri_cmap, panel_letter, result_path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    DEFAULT_TAUS,
    FACTOR_COLORS,
    FACTORS,
    FIG_DPI,
    FIG_WIDTH,
    REF_COLOR,
    cached_shap,
    load_channel50,
    load_quantile_models,
    seed_grouped_split,
    target_label,
)

warnings.filterwarnings("ignore")

FORCE = "--force-shap" in sys.argv
TARGETS = ["f_ratio", "log_abs"]
TAUS = list(DEFAULT_TAUS)  # 0.05, 0.25, 0.50, 0.75, 0.95
TAIL_TAUS = [0.05, 0.50, 0.95]
FS = 10  # journal body text
FS_DENSE = 8  # multi-panel dense figures

# Fixed Δ-bar palette (same in both columns — not target_color).
DELTA_LOWER = REF_COLOR
DELTA_UPPER = "#4477AA"
DELTA_FULL = "#2C5A8A"


def _save(fig, name: str) -> str:
    """Save at exactly FIG_WIDTH (proceedings column width)."""
    w, h = fig.get_size_inches()
    if w > FIG_WIDTH + 1e-6:
        raise ValueError(f"{name}: figure width {w:.3f} > FIG_WIDTH={FIG_WIDTH}")
    out = result_path("plots", name)
    # No bbox_inches="tight" — it can expand past the column width.
    fig.savefig(out, dpi=FIG_DPI, pad_inches=0.01)
    plt.close(fig)
    print(f"  plot {name} ({w:.2f}×{h:.2f} in)")
    return out


def _style_axes(
    ax,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    fontsize: int = FS,
) -> None:
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=2)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=2)
    if title is not None:
        ax.set_title(title, fontsize=fontsize, pad=3)
    ax.tick_params(axis="both", labelsize=fontsize)


def load_quantile_shap(
    quant_models: dict,
    Xte: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, float], np.ndarray]]:
    """Compute / load quantile SHAP; return importance rows + tail SHAP dict."""
    rows: list[dict] = []
    q_shap: dict[tuple[str, float], np.ndarray] = {}
    for tgt in TARGETS:
        for tau in TAUS:
            model = quant_models[tgt][tau]
            sv = cached_shap(
                f"shap_seed_q_{tgt}_tau{int(tau * 100):02d}",
                lambda m=model: shap.TreeExplainer(m).shap_values(Xte),
                force=FORCE,
            )
            q_shap[(tgt, tau)] = sv
            mean_abs = np.abs(sv).mean(0)
            shares = 100 * mean_abs / mean_abs.sum()
            for f, ma, sh in zip(FACTORS, mean_abs, shares):
                rows.append(
                    dict(
                        target=tgt,
                        tau=tau,
                        factor=f,
                        mean_abs_shap=float(ma),
                        share_pct=float(sh),
                    )
                )
    return pd.DataFrame(rows), q_shap


def plot_importance_combined(q_imp: pd.DataFrame) -> None:
    """Fig 1: 1×2 quantile |SHAP| share vs τ for f_ratio | log_abs."""
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_WIDTH * 0.50), sharey=True)
    y_top = float(np.ceil(q_imp["share_pct"].max() / 10.0) * 10.0 + 5.0)
    y_top = float(np.clip(y_top, 40.0, 100.0))
    for col, tgt in enumerate(TARGETS):
        ax = axes[col]
        qdf = q_imp[q_imp.target == tgt]
        for f in FACTORS:
            sub = qdf[qdf.factor == f].sort_values("tau")
            ax.plot(
                sub["tau"],
                sub["share_pct"],
                "o-",
                color=FACTOR_COLORS[f],
                ms=4,
                lw=1.6,
                label=f,  # raw; Figure.legend auto_format → LABEL_MAP LaTeX
            )
        _style_axes(ax, xlabel=r"quantile $\tau$", title=target_label(tgt))
        ax.set_xticks(TAUS)
        ax.set_ylim(0, y_top)
        panel_letter(ax, string.ascii_lowercase[col], fontsize=FS)
    axes[0].set_ylabel("% of total mean |SHAP|", fontsize=FS, labelpad=2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(FACTORS),
        fontsize=FS,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
        columnspacing=1.0,
        handlelength=1.6,
    )
    fig.subplots_adjust(top=0.92, bottom=0.22, left=0.11, right=0.98, wspace=0.12)
    _save(fig, "quantile_importance_combined.png")


def plot_beeswarm_combined(
    q_shap: dict[tuple[str, float], np.ndarray],
    Xte: pd.DataFrame,
) -> None:
    """Fig 2: 2×3 beeswarm — rows = targets, cols = τ∈{0.05, 0.50, 0.95}."""
    beeswarm_cmap = get_crameri_cmap("managua").reversed()
    Xte_np = Xte.to_numpy()
    fig, axes = plt.subplots(2, 3, figsize=(FIG_WIDTH, FIG_WIDTH * 0.68))

    for row, tgt in enumerate(TARGETS):
        mean_abs = np.mean(
            [np.abs(q_shap[(tgt, tau)]).mean(0) for tau in TAIL_TAUS],
            axis=0,
        )
        feature_order = np.argsort(-mean_abs)
        for col, tau in enumerate(TAIL_TAUS):
            ax = axes[row, col]
            expl = shap.Explanation(
                values=q_shap[(tgt, tau)],
                data=Xte_np,
                feature_names=list(FACTORS),  # raw; ytick auto_format → LaTeX
            )
            shap.plots.beeswarm(
                expl,
                ax=ax,
                show=False,
                color=beeswarm_cmap,
                color_bar=False,
                plot_size=None,
                order=feature_order,
            )
            ax.tick_params(axis="both", labelsize=FS_DENSE)
            if row == 0:
                ax.set_title(rf"$\tau={tau:g}$", fontsize=FS, pad=2)
                ax.set_xlabel("")
            else:
                ax.set_xlabel("SHAP value" if col == 1 else "", fontsize=FS_DENSE, labelpad=1)
            if col > 0:
                ax.set_yticklabels([])
                ax.set_ylabel("")
            else:
                # shap.plots.beeswarm often sets Text directly (bypasses patch).
                texts = [t.get_text() for t in ax.get_yticklabels()]
                if texts and not str(texts[0]).startswith("$"):
                    ax.set_yticklabels(texts, fontsize=FS_DENSE)
                ax.set_ylabel(target_label(tgt), fontsize=FS, labelpad=2)
            panel_letter(ax, string.ascii_lowercase[row * 3 + col], fontsize=FS)

    # Re-assert row labels (beeswarm can overwrite ylabel).
    for row, tgt in enumerate(TARGETS):
        axes[row, 0].set_ylabel(target_label(tgt), fontsize=FS, labelpad=2)

    sm = mpl_cm.ScalarMappable(cmap=beeswarm_cmap)
    sm.set_array([0, 1])
    fig.subplots_adjust(top=0.94, bottom=0.09, left=0.13, right=0.88, hspace=0.12, wspace=0.05)
    cbar = fig.colorbar(sm, ax=axes, ticks=[0, 1], fraction=0.035, pad=0.015, shrink=0.90)
    cbar.set_ticklabels(["Low", "High"])
    cbar.set_label("Feature value", fontsize=FS_DENSE)
    cbar.ax.tick_params(length=0, labelsize=FS_DENSE)
    _save(fig, "quantile_beeswarm_combined.png")


def plot_interval_delta_combined(
    q_shap: dict[tuple[str, float], np.ndarray],
) -> None:
    """Fig 3: 2×2 — top mean|Δφ| bars; bottom Δ mean|SHAP| vs τ span."""
    fs = FS_DENSE
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH * 0.70))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.05, 1.0, 0.09],
        hspace=0.22,
        wspace=0.20,
        top=0.96,
        bottom=0.04,
        left=0.14,
        right=0.98,
    )
    axes = np.array(
        [
            [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
            [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])],
        ]
    )

    for col, tgt in enumerate(TARGETS):
        sv_lo = q_shap[(tgt, 0.05)]
        sv_med = q_shap[(tgt, 0.50)]
        sv_hi = q_shap[(tgt, 0.95)]
        d_full = sv_hi - sv_lo
        mean_abs_d = np.abs(d_full).mean(axis=0)
        ma_lo = np.abs(sv_lo).mean(0)
        ma_med = np.abs(sv_med).mean(0)
        ma_hi = np.abs(sv_hi).mean(0)

        ax_top = axes[0, col]
        order = np.argsort(mean_abs_d)
        ax_top.barh(
            np.arange(len(FACTORS)),
            mean_abs_d[order],
            color=[FACTOR_COLORS[FACTORS[i]] for i in order],
            height=0.7,
        )
        ax_top.set_yticks(np.arange(len(FACTORS)))
        ax_top.set_yticklabels([FACTORS[i] for i in order], fontsize=fs)
        ax_top.set_title(target_label(tgt), fontsize=fs, pad=2)
        ax_top.set_xlabel(r"mean $|\Delta\phi|$", fontsize=fs, labelpad=1)
        ax_top.tick_params(axis="both", labelsize=fs)
        panel_letter(ax_top, string.ascii_lowercase[col], fontsize=fs)

        ax_bot = axes[1, col]
        x = np.arange(len(FACTORS))
        w = 0.25
        ax_bot.bar(
            x - w,
            ma_med - ma_lo,
            width=w,
            color=DELTA_LOWER,
            label=r"$\Delta$ lower ($0.50{-}0.05$)",
        )
        ax_bot.bar(
            x,
            ma_hi - ma_med,
            width=w,
            color=DELTA_UPPER,
            alpha=0.75,
            label=r"$\Delta$ upper ($0.95{-}0.50$)",
        )
        ax_bot.bar(
            x + w,
            ma_hi - ma_lo,
            width=w,
            color=DELTA_FULL,
            edgecolor="black",
            linewidth=0.5,
            label=r"$\Delta$ full ($0.95{-}0.05$)",
        )
        ax_bot.axhline(0, color=REF_COLOR, lw=0.8)
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels(FACTORS, fontsize=fs, rotation=25, ha="right")
        if col == 0:
            ax_bot.set_ylabel(r"$\Delta$ mean $|$SHAP$|$", fontsize=fs, labelpad=2)
        ax_bot.tick_params(axis="both", labelsize=fs)
        panel_letter(ax_bot, string.ascii_lowercase[2 + col], fontsize=fs)

    axes[0, 0].set_ylabel(r"mean $|\Delta\phi|$ (PI drivers)", fontsize=fs, labelpad=2)
    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center",
        ncol=3,
        fontsize=fs,
        frameon=False,
        bbox_to_anchor=(0.555, 0.035),
        columnspacing=1.2,
        handlelength=1.4,
    )
    _save(fig, "shap_interval_delta_combined.png")


def main() -> None:
    t0 = time.monotonic()
    apply_style(auto_format=True, font_size=10, frame="open")

    d50 = load_channel50()
    _, te = seed_grouped_split(d50, test_size=0.25, seed=0)
    Xte = d50[FACTORS].iloc[te].reset_index(drop=True)

    quant_models = load_quantile_models(taus=TAUS, targets=TARGETS, split_by="seed")
    missing = [
        f"{t}@τ={tau}" for t in TARGETS for tau in TAUS if tau not in quant_models.get(t, {})
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing seed quantile models: {missing}. "
            "Train with: python quantile/quantile_channel_model.py"
        )

    print("=== Loading quantile SHAP (cache) ===")
    q_imp, q_shap = load_quantile_shap(quant_models, Xte)

    print("=== Figure 1: quantile importance (1×2) ===")
    plot_importance_combined(q_imp)

    print("=== Figure 2: quantile beeswarm (2×3) ===")
    plot_beeswarm_combined(q_shap, Xte)

    print("=== Figure 3: interval ΔSHAP + τ dynamics (2×2) ===")
    plot_interval_delta_combined(q_shap)

    elapsed = time.monotonic() - t0
    print(f"\nDone in {elapsed:.0f}s. Plots: {result_path('plots', '_').rsplit('/', 1)[0]}")


if __name__ == "__main__":
    main()
