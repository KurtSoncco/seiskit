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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from PIL import Image

import shap
from seiskit.plot_config import apply_style, get_crameri_cmap, panel_letter, result_path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    DEFAULT_TAUS,
    FACTOR_COLORS,
    FACTORS,
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
SAVE_DPI = 600  # journal raster DPI
# Tiny pad so antialiased strokes/text are not clipped by tight crop.
TIGHT_PAD_IN = 0.02

# Fixed Δ-bar palette (same in both columns — not target_color).
DELTA_LOWER = REF_COLOR
DELTA_UPPER = "#4477AA"
DELTA_FULL = "#2C5A8A"


def _save(fig, name: str, *, pad_inches: float | None = None) -> str:
    """Tight-crop at ``SAVE_DPI``, then snap width to exactly ``FIG_WIDTH`` in.

    ``bbox_inches='tight'`` removes empty margin but changes physical width.
    After the tight save we resize the raster to ``round(FIG_WIDTH * DPI)``
    pixels so the file is exactly 6.5 in wide at ``SAVE_DPI``.
    """
    out = result_path("plots", name)
    target_px = int(round(FIG_WIDTH * SAVE_DPI))

    fig.savefig(
        out,
        dpi=SAVE_DPI,
        bbox_inches="tight",
        pad_inches=TIGHT_PAD_IN if pad_inches is None else pad_inches,
    )
    plt.close(fig)

    img = Image.open(out)
    if img.width != target_px:
        target_h = max(1, int(round(img.height * (target_px / img.width))))
        img = img.resize((target_px, target_h), Image.Resampling.LANCZOS)
    img.save(out, dpi=(SAVE_DPI, SAVE_DPI))

    phys_w = target_px / SAVE_DPI
    phys_h = Image.open(out).height / SAVE_DPI
    print(f"  plot {name} ({phys_w:.4f}×{phys_h:.4f} in @ {SAVE_DPI} dpi, tight)")
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
        ax.set_title(title, fontsize=fontsize, pad=3, loc="center")
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
    # Dedicated legend row: fixed-canvas save cannot rely on bbox_inches=tight
    # or on loc="outside" (needs a layout engine).
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH * 0.50))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 0.14],
        hspace=0.30,
        wspace=0.18,
        top=0.88,
        bottom=0.02,
        left=0.12,
        right=0.98,
    )
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    axes[1].sharey(axes[0])
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
        _style_axes(ax, xlabel=r"Quantile $\tau$", title=target_label(tgt))
        ax.set_xticks(TAUS)
        ax.set_ylim(0, 70.0)
        panel_letter(ax, string.ascii_lowercase[col], fontsize=FS)
    axes[0].set_ylabel("Relative SHAP Importance (%)", fontsize=FS, labelpad=2)
    handles, labels = axes[0].get_legend_handles_labels()
    ax_leg = fig.add_subplot(gs[1, :])
    ax_leg.set_axis_off()
    ax_leg.legend(
        handles,
        labels,
        loc="center",
        ncol=len(FACTORS),
        fontsize=FS,
        frameon=False,
        columnspacing=1.0,
        handlelength=1.6,
    )
    _save(fig, "quantile_importance_combined.png")


def _factor_level_codes(X: pd.DataFrame) -> np.ndarray:
    """Map each factor column to {0, 1, 2} = low / middle / high."""
    out = np.empty(X.shape, dtype=np.float64)
    for j, col in enumerate(X.columns):
        levels = np.sort(pd.unique(X[col]))
        mapping = {v: float(i) for i, v in enumerate(levels)}
        out[:, j] = X[col].map(mapping).to_numpy(dtype=np.float64)
    return out


def plot_beeswarm_combined(
    q_shap: dict[tuple[str, float], np.ndarray],
    Xte: pd.DataFrame,
) -> None:
    """Fig 2: 2×3 beeswarm — rows = targets, cols = τ∈{0.05, 0.50, 0.95}."""
    # Three discrete colors (low / middle / high) from managua.
    base_cmap = get_crameri_cmap("managua").reversed()
    level_colors = [base_cmap(0.05), base_cmap(0.50), base_cmap(0.95)]
    beeswarm_cmap = ListedColormap(level_colors)
    Xte_levels = _factor_level_codes(Xte[FACTORS])

    # Per-row x-range from data (trim extreme tails) so panels are not padded empty.
    x_lims: dict[str, tuple[float, float]] = {}
    for tgt in TARGETS:
        row_sv = np.concatenate([q_shap[(tgt, tau)].ravel() for tau in TAIL_TAUS])
        lo, hi = np.quantile(row_sv, [0.002, 0.998])
        pad = 0.04 * (hi - lo)
        x_lims[tgt] = (float(lo - pad), float(hi + pad))

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH * 0.68))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.09, 1.0],
        hspace=0.08,
        top=0.99,
        bottom=0.06,
        left=0.10,
        right=0.995,
    )
    ax_leg = fig.add_subplot(gs[0])
    ax_leg.set_axis_off()
    gs_plots = gs[1].subgridspec(2, 3, hspace=0.05, wspace=0.02)
    axes = np.array([[fig.add_subplot(gs_plots[r, c]) for c in range(3)] for r in range(2)])

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
                data=Xte_levels,
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
            ax.tick_params(axis="both", labelsize=FS)
            ax.set_xticks([0])
            ax.set_xticklabels(["0"] if row == 1 else [])
            ax.xaxis.grid(False)
            ax.yaxis.grid(False)
            # beeswarm uses long y-ticks and padded ylim — reclaim that space.
            ax.tick_params("y", length=0)
            ax.set_ylim(-0.5, len(FACTORS) - 0.5)
            ax.set_xlim(*x_lims[tgt])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.0)
                spine.set_color("0.2")
            if row == 0:
                ax.set_title(rf"$\tau={tau:g}$", fontsize=FS, pad=2)
                ax.set_xlabel("")
            else:
                ax.set_xlabel("SHAP value" if col == 1 else "", fontsize=FS, labelpad=0)
            if col > 0:
                ax.tick_params(axis="y", left=False, labelleft=False, length=0)
                ax.set_yticklabels([])
                ax.set_ylabel("")
            else:
                texts = [t.get_text() for t in ax.get_yticklabels()]
                if texts and not str(texts[0]).startswith("$"):
                    ax.set_yticklabels(texts, fontsize=FS)
                ax.set_ylabel(target_label(tgt), fontsize=FS, labelpad=1)
            panel_letter(ax, string.ascii_lowercase[row * 3 + col], fontsize=FS)

    for row, tgt in enumerate(TARGETS):
        axes[row, 0].set_ylabel(target_label(tgt), fontsize=FS, labelpad=1)

    level_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=c,
            markeredgecolor="none",
            markersize=7,
            label=lab,
        )
        for c, lab in zip(level_colors, ["Low", "Middle", "High"])
    ]
    leg = ax_leg.legend(
        level_handles,
        ["Low", "Middle", "High"],
        loc="center",
        ncol=3,
        fontsize=FS,
        frameon=False,
        columnspacing=0.5,
        handletextpad=0.2,
        borderpad=0.0,
        labelspacing=0.0,
        handlelength=0.8,
        title="Feature value",
        title_fontsize=FS,
    )
    # Collapse default title ↔ handles gap.
    if hasattr(leg, "_legend_box") and leg._legend_box is not None:
        leg._legend_box.sep = 0.05
    _save(fig, "quantile_beeswarm_combined.png", pad_inches=0.04)


def plot_interval_delta_combined(
    q_shap: dict[tuple[str, float], np.ndarray],
) -> None:
    """Fig 3: 2×2 — top mean|Δφ| bars; bottom Δ mean|SHAP| vs τ span."""
    fs = FS
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH * 0.85))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.05, 1.0, 0.22],
        hspace=0.48,
        wspace=0.30,
        top=0.96,
        bottom=0.03,
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
        ax_top.set_title(target_label(tgt), fontsize=fs, pad=2, loc="left", ha="left")
        ax_top.set_xlabel(r"Mean $|\Delta\phi|$", fontsize=fs, labelpad=1)
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

    axes[0, 0].set_ylabel(r"Mean $|\Delta\phi|$ (PI drivers)", fontsize=fs, labelpad=2)
    handles, labels = axes[1, 0].get_legend_handles_labels()

    axes[1, 0].set_ylim(-0.04, 0.06)
    axes[1, 1].set_ylim(-0.2, 0.4)

    # Dedicated gridspec row (height_ratios[-1]) — stays inside fixed canvas.
    ax_leg = fig.add_subplot(gs[2, :])
    ax_leg.set_axis_off()
    ax_leg.legend(
        handles,
        labels,
        loc="center",
        ncol=3,
        fontsize=fs,
        frameon=False,
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
