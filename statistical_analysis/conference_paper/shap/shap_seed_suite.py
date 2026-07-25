"""Seed-split SHAP / xAI suite for center-recorder models.

Uses existing ``models/*_seed.pkl`` only — never retrains. TreeSHAP values are
cached under ``.cache/`` so re-runs regenerate plots/tables without recomputing
SHAP unless ``--force-shap`` is passed.

Outputs (all under ``results/shap/``)
------------------------------------
Data (always written when computed)
    mean_shap_*.csv          — mean-GBM xAI tables (kept; plots off by default)
    quantile_shap_*.csv
    shap_interval_delta.csv  — Δφ = φ(0.95)−φ(0.05) (aleatoric interval drivers)
    shap_tau_sensitivity.csv — how |SHAP| / φ change across τ
    pdp_by_quantile.csv
    h2_by_quantile.csv

Plots (QBM-focused by default)
    quantile_beeswarm_{target}.png   — τ=0.05 | 0.50 | 0.95 side-by-side
    quantile_importance_{target}.png
    quantile_dependence_{target}_{feat}_q{tau}.png
    shap_interval_sensitivity_{target}.png
    shap_interval_beeswarm_{target}.png
    shap_tau_delta_{target}.png
    quantile_interactions_{target}.png
    pdp_{target}.png              — all factors × τ∈{0.05,0.50,0.95}
    h2_heatmap_{target}_q{tau}.png

    Optional mean-GBM plots with ``--plot-mean``:
    mean_beeswarm_*.png, mean_importance_*.png, ...

Notes
-----
Interval ΔSHAP attributes features to conditional-quantile spread (aleatoric /
predictive-interval drivers), following He et al. (Chem. Eng. J. 2025).

For full UbiQTree epistemic/aleatoric/entanglement decomposition of SHAP
(Dubey et al., arXiv:2508.09639) see ``shap_ubiqtree.py``.

Usage
-----
    python shap_seed_suite.py
    python shap_seed_suite.py --force-shap
    python shap_seed_suite.py --plot-mean          # also emit mean-GBM SHAP plots
    python shap_seed_suite.py --skip-interactions
    python shap_seed_suite.py --skip-pdp
"""

from __future__ import annotations

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
SKIP_INTER = "--skip-interactions" in sys.argv
SKIP_PDP = "--skip-pdp" in sys.argv
PLOT_MEAN = "--plot-mean" in sys.argv  # off by default: paper xAI is QBM-only

MEAN_TARGETS = ["log_abs", "f_ratio"]
Q_TARGETS = ["log_abs", "f_ratio"]
TAUS = list(DEFAULT_TAUS)
TAIL_TAUS = [0.05, 0.50, 0.95]
N_INTER = 2500
N_PDP = 2000
PDP_GRID = 25

# Dependence partners for signed directionality plots
DEP_PAIRS = {
    "log_abs": [("Vs1", "Height"), ("Height", "Vs1"), ("CoV", "aHV")],
    "f_ratio": [("CoV", "aHV"), ("aHV", "CoV"), ("Height", "CoV")],
}
TAIL_DEPS = [
    ("log_abs", 0.95, "Vs1", "Height"),
    ("log_abs", 0.05, "Height", "Vs1"),
    ("f_ratio", 0.05, "CoV", "aHV"),
    ("f_ratio", 0.95, "aHV", "CoV"),
]


def _save(fig, name: str) -> str:
    out = result_path("plots", name)
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  plot {name}")
    return out


def _csv(df: pd.DataFrame, name: str) -> str:
    out = result_path("data", name)
    df.to_csv(out, index=False)
    print(f"  data {name}")
    return out


def friedman_h2(pd_ij: np.ndarray, pd_i: np.ndarray, pd_j: np.ndarray) -> float:
    """Centered Friedman H² on a PDP grid; clipped to [0, 1]."""
    pd_ij_c = pd_ij - pd_ij.mean()
    pd_i_c = pd_i - pd_i.mean()
    pd_j_c = pd_j - pd_j.mean()
    inter = pd_ij_c - pd_i_c[:, None] - pd_j_c[None, :]
    den = float(np.sum(pd_ij_c**2))
    if den < 1e-12:
        return 0.0
    return float(np.clip(np.sum(inter**2) / den, 0.0, 1.0))


def pd_1d(model, X_ref: np.ndarray, feat_idx: int, grid: np.ndarray) -> np.ndarray:
    X = X_ref.copy()
    out = np.empty(len(grid))
    for k, val in enumerate(grid):
        X[:, feat_idx] = val
        out[k] = float(np.mean(model.predict(X)))
    return out


def pd_2d(model, X_ref: np.ndarray, i: int, j: int, gi: np.ndarray, gj: np.ndarray) -> np.ndarray:
    X = X_ref.copy()
    out = np.zeros((len(gi), len(gj)))
    for ii, vi in enumerate(gi):
        for jj, vj in enumerate(gj):
            X[:, i] = vi
            X[:, j] = vj
            out[ii, jj] = float(np.mean(model.predict(X)))
    return out


def main() -> None:
    t_all = time.monotonic()
    apply_style(auto_format=True, font_size=10, frame="open")

    d50 = load_channel50()
    _, te = seed_grouped_split(d50, test_size=0.25, seed=0)
    Xte = d50[FACTORS].iloc[te].reset_index(drop=True)

    mean_models = load_mean_models(targets=MEAN_TARGETS, split_by="seed")
    quant_models = load_quantile_models(taus=TAUS, targets=Q_TARGETS, split_by="seed")
    missing_mean = [t for t in MEAN_TARGETS if t not in mean_models]
    missing_q = [
        f"{t}@τ={tau}" for t in Q_TARGETS for tau in TAUS if tau not in quant_models.get(t, {})
    ]
    if missing_mean or missing_q:
        raise FileNotFoundError(
            f"Missing seed models. mean={missing_mean} quantile={missing_q}. "
            "Train with: python quantile/quantile_channel_model.py"
        )

    rng = np.random.default_rng(0)
    inter_idx = rng.choice(len(Xte), size=min(N_INTER, len(Xte)), replace=False)
    X_inter = Xte.iloc[inter_idx].reset_index(drop=True)
    pdp_idx = rng.choice(len(Xte), size=min(N_PDP, len(Xte)), replace=False)
    X_pdp = Xte.iloc[pdp_idx].to_numpy()

    # ------------------------------------------------------------------
    # 1) Mean-model SHAP (data always; plots only with --plot-mean)
    # ------------------------------------------------------------------
    print("\n=== Mean-model SHAP (data only; plots off unless --plot-mean) ===")
    mean_shap = {}
    imp_rows, dir_rows = [], []
    for tgt in MEAN_TARGETS:
        model = mean_models[tgt]
        sv = cached_shap(
            f"shap_seed_mean_{tgt}_te",
            lambda m=model: shap.TreeExplainer(m).shap_values(Xte),
            force=FORCE,
        )
        mean_shap[tgt] = sv
        mean_abs = np.abs(sv).mean(0)
        shares = 100 * mean_abs / mean_abs.sum()
        for f, ma, sh in zip(FACTORS, mean_abs, shares):
            imp_rows.append(
                dict(
                    target=tgt,
                    factor=f,
                    mean_abs_shap=float(ma),
                    share_pct=float(sh),
                )
            )
            for lv in sorted(Xte[f].unique()):
                m = Xte[f].to_numpy() == lv
                dir_rows.append(
                    dict(
                        target=tgt,
                        factor=f,
                        level=float(lv),
                        mean_signed_shap=float(sv[m, FACTORS.index(f)].mean()),
                        n=int(m.sum()),
                    )
                )

        if PLOT_MEAN:
            plt.figure()
            shap.summary_plot(
                sv,
                Xte,
                show=False,
                plot_size=(6.4, 3.8),
                color_bar=True,
                sort=True,
            )
            fig = plt.gcf()
            fig.axes[0].set_xlabel(f"SHAP value (impact on {target_label(tgt)})")
            _save(fig, f"mean_beeswarm_{tgt}.png")

            order = np.argsort(mean_abs)
            fig, ax = plt.subplots(figsize=(5.5, 3.6))
            ax.barh(
                np.arange(len(FACTORS)),
                mean_abs[order],
                color=[FACTOR_COLORS[FACTORS[i]] for i in order],
            )
            ax.set_yticks(np.arange(len(FACTORS)))
            ax.set_yticklabels([FACTORS[i] for i in order])
            ax.set_xlabel("mean |SHAP|")
            ax.set_title(f"Mean |SHAP| importance — {target_label(tgt)}")
            _save(fig, f"mean_importance_{tgt}.png")

            fig, axes = plt.subplots(1, 5, figsize=figsize(height=2.8), sharey=True)
            for ax, f in zip(axes, FACTORS):
                sub = [r for r in dir_rows if r["target"] == tgt and r["factor"] == f]
                xs = [r["level"] for r in sub]
                ys = [r["mean_signed_shap"] for r in sub]
                ax.plot(xs, ys, "o-", color=FACTOR_COLORS[f], ms=6, lw=1.6)
                ax.axhline(0, color=REF_COLOR, lw=0.8)
                ax.set_title(f)
                ax.set_xlabel("level")
            axes[0].set_ylabel("mean signed SHAP")
            fig.tight_layout()
            _save(fig, f"mean_directionality_{tgt}.png")

    _csv(pd.DataFrame(imp_rows), "mean_shap_importance.csv")
    _csv(pd.DataFrame(dir_rows), "mean_shap_directionality.csv")

    # ------------------------------------------------------------------
    # 2) Mean-model interactions + dependence (data always; plots optional)
    # ------------------------------------------------------------------
    inter_rows = []
    if not SKIP_INTER:
        print("\n=== Mean-model interactions (data; plots only with --plot-mean) ===")
        for tgt in MEAN_TARGETS:
            model = mean_models[tgt]
            si = cached_shap(
                f"shap_seed_inter_{tgt}",
                lambda m=model: shap.TreeExplainer(m).shap_interaction_values(X_inter),
                force=FORCE,
            )
            M = np.abs(si).mean(0)
            for i, fi in enumerate(FACTORS):
                for j, fj in enumerate(FACTORS):
                    inter_rows.append(
                        dict(
                            target=tgt,
                            factor_i=fi,
                            factor_j=fj,
                            mean_abs_interaction=float(M[i, j]),
                            is_main=(i == j),
                        )
                    )

            if PLOT_MEAN:
                Moff = M.copy()
                np.fill_diagonal(Moff, np.nan)
                fig, ax = plt.subplots(figsize=(5.2, 4.4))
                im = ax.imshow(Moff, cmap=get_crameri_cmap("batlow"))
                ax.set_xticks(range(5))
                ax.set_yticks(range(5))
                ax.set_xticklabels(FACTORS, rotation=45, ha="right")
                ax.set_yticklabels(FACTORS)
                ax.set_title(f"Mean |interaction SHAP| — {target_label(tgt)}")
                for i in range(5):
                    for j in range(5):
                        if i != j:
                            ax.text(
                                j,
                                i,
                                f"{M[i, j]:.3f}",
                                ha="center",
                                va="center",
                                fontsize=7,
                                color="white" if M[i, j] < np.nanmax(Moff) * 0.55 else "black",
                            )
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                _save(fig, f"mean_interaction_heatmap_{tgt}.png")

                for fa, fb in DEP_PAIRS[tgt]:
                    fi = FACTORS.index(fa)
                    fig, ax = plt.subplots(figsize=(4.8, 3.6))
                    sc = ax.scatter(
                        Xte[fa].to_numpy(),
                        mean_shap[tgt][:, fi],
                        c=Xte[fb].to_numpy(),
                        cmap=get_crameri_cmap("vik"),
                        s=10,
                        alpha=0.65,
                        edgecolor="none",
                    )
                    ax.axhline(0, color=REF_COLOR, lw=0.8)
                    ax.set_xlabel(fa)
                    ax.set_ylabel(f"SHAP({fa})")
                    ax.set_title(f"{target_label(tgt)}: SHAP({fa}) colored by {fb}")
                    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                    cb.set_label(fb)
                    _save(fig, f"mean_dependence_{tgt}_{fa}_by_{fb}.png")

        _csv(pd.DataFrame(inter_rows), "mean_shap_interactions.csv")
    else:
        print("\n=== Skipping mean interactions (--skip-interactions) ===")

    # ------------------------------------------------------------------
    # 3) Quantile SHAP importance + tail dependence
    # ------------------------------------------------------------------
    print("\n=== Quantile SHAP ===")
    q_imp_rows = []
    q_shap_tail = {}
    for tgt in Q_TARGETS:
        for tau in TAUS:
            model = quant_models[tgt][tau]
            sv = cached_shap(
                f"shap_seed_q_{tgt}_tau{int(tau * 100):02d}",
                lambda m=model: shap.TreeExplainer(m).shap_values(Xte),
                force=FORCE,
            )
            if tau in TAIL_TAUS:
                q_shap_tail[(tgt, tau)] = sv
            mean_abs = np.abs(sv).mean(0)
            shares = 100 * mean_abs / mean_abs.sum()
            for f, ma, sh in zip(FACTORS, mean_abs, shares):
                q_imp_rows.append(
                    dict(
                        target=tgt,
                        tau=tau,
                        factor=f,
                        mean_abs_shap=float(ma),
                        share_pct=float(sh),
                    )
                )

        qdf = pd.DataFrame([r for r in q_imp_rows if r["target"] == tgt])
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        for f in FACTORS:
            sub = qdf[qdf.factor == f].sort_values("tau")
            ax.plot(
                sub["tau"],
                sub["share_pct"],
                "o-",
                color=FACTOR_COLORS[f],
                ms=5,
                lw=1.8,
                label=f,
            )
        ax.set_xlabel(r"quantile $\tau$")
        ax.set_ylabel("% of total mean |SHAP|")
        ax.set_title(f"Quantile |SHAP| share — {target_label(tgt)}")
        ax.set_xticks(TAUS)
        ax.legend(fontsize=8, frameon=False, ncol=3)
        _save(fig, f"quantile_importance_{tgt}.png")

    _csv(pd.DataFrame(q_imp_rows), "quantile_shap_importance.csv")

    # Side-by-side beeswarms at lower / median / upper quantiles so feature
    # impacts that matter only in the tails are visible against τ=0.50.
    beeswarm_cmap = get_crameri_cmap("managua").reversed()
    for tgt in Q_TARGETS:
        mean_abs = np.mean(
            [np.abs(q_shap_tail[(tgt, tau)]).mean(0) for tau in TAIL_TAUS],
            axis=0,
        )
        feature_order = np.argsort(-mean_abs)
        fig, axes = plt.subplots(1, 3, figsize=figsize(height=FIG_WIDTH * 0.45), layout="constrained")
        for col, tau in enumerate(TAIL_TAUS):
            expl = shap.Explanation(
                values=q_shap_tail[(tgt, tau)],
                data=Xte.to_numpy(),
                feature_names=FACTORS,
            )
            shap.plots.beeswarm(
                expl,
                ax=axes[col],
                show=False,
                color=beeswarm_cmap,
                color_bar=False,
                plot_size=None,
                order=feature_order,
            )
            axes[col].set_title(rf"$\tau={tau:g}$")
            if col > 0:
                axes[col].set_yticklabels([])
            panel_letter(axes[col], "abc"[col])
        sm = mpl_cm.ScalarMappable(cmap=beeswarm_cmap)
        sm.set_array([0, 1])
        cbar = fig.colorbar(sm, ax=axes, ticks=[0, 1], shrink=0.85, location="right")
        cbar.set_ticklabels(["Low", "High"])
        cbar.set_label("Feature value")
        cbar.ax.tick_params(length=0)
        _save(fig, f"quantile_beeswarm_{tgt}.png")

    for tgt, tau, feat, partner in TAIL_DEPS:
        sv = q_shap_tail[(tgt, tau)]
        fi = FACTORS.index(feat)
        fig, ax = plt.subplots(figsize=(4.8, 3.6))
        sc = ax.scatter(
            Xte[feat].to_numpy(),
            sv[:, fi],
            c=Xte[partner].to_numpy(),
            cmap=get_crameri_cmap("vik"),
            s=10,
            alpha=0.65,
            edgecolor="none",
        )
        ax.axhline(0, color=REF_COLOR, lw=0.8)
        ax.set_xlabel(feat)
        ax.set_ylabel(f"SHAP({feat}) @ τ={tau:g}")
        ax.set_title(f"{target_label(tgt)}: {feat} @ τ={tau:g} (colored by {partner})")
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label(partner)
        _save(fig, f"quantile_dependence_{tgt}_{feat}_q{int(tau * 100):02d}.png")

    # ------------------------------------------------------------------
    # 3b) Interval ΔSHAP (aleatoric drivers) + τ-attribution dynamics
    # ------------------------------------------------------------------
    # Δφ_j = φ_j(τ=0.95) − φ_j(τ=0.05) attributes feature j to moving the
    # prediction across the conditional quantile interval (He et al. 2025).
    # This is NOT an epistemic CI on φ; epistemic UQ needs multi-fit ensembles.
    print("\n=== Interval SHAP sensitivity + τ-dynamics ===")
    interval_rows: list[dict] = []
    tau_sens_rows: list[dict] = []
    Xte_np = Xte.to_numpy()
    for tgt in Q_TARGETS:
        sv_lo = q_shap_tail[(tgt, 0.05)]
        sv_med = q_shap_tail[(tgt, 0.50)]
        sv_hi = q_shap_tail[(tgt, 0.95)]
        d_full = sv_hi - sv_lo
        d_upper = sv_hi - sv_med
        d_lower = sv_med - sv_lo

        y_lo = np.asarray(quant_models[tgt][0.05].predict(Xte), dtype=float)
        y_hi = np.asarray(quant_models[tgt][0.95].predict(Xte), dtype=float)
        dy = y_hi - y_lo
        # TreeSHAP: f(x) = base + Σφ; bases differ across quantile models.
        base_lo = float(np.mean(y_lo - sv_lo.sum(axis=1)))
        base_hi = float(np.mean(y_hi - sv_hi.sum(axis=1)))
        d_base = base_hi - base_lo
        efficiency_resid = d_full.sum(axis=1) - (dy - d_base)
        efficiency_mae = float(np.mean(np.abs(efficiency_resid)))
        mean_width = float(np.mean(dy))
        print(
            f"  {tgt}: mean PI width={mean_width:.4g}, "
            f"ΣΔφ efficiency MAE={efficiency_mae:.3g} (vs Δŷ−Δbase)"
        )

        mean_abs_d = np.abs(d_full).mean(axis=0)
        mean_d = d_full.mean(axis=0)
        share_d = 100 * mean_abs_d / mean_abs_d.sum() if mean_abs_d.sum() > 0 else mean_abs_d
        for f, ma, md, sh in zip(FACTORS, mean_abs_d, mean_d, share_d):
            interval_rows.append(
                dict(
                    target=tgt,
                    factor=f,
                    mean_delta_shap=float(md),
                    mean_abs_delta_shap=float(ma),
                    share_pct=float(sh),
                    mean_interval_width=mean_width,
                    efficiency_mae=efficiency_mae,
                    delta_base=d_base,
                )
            )

        # Bar: mean |Δφ| by factor
        order = np.argsort(mean_abs_d)
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.barh(
            np.arange(len(FACTORS)),
            mean_abs_d[order],
            color=[FACTOR_COLORS[FACTORS[i]] for i in order],
        )
        ax.set_yticks(np.arange(len(FACTORS)))
        ax.set_yticklabels([FACTORS[i] for i in order])
        ax.set_xlabel(r"mean $|\phi_{0.95}-\phi_{0.05}|$")
        ax.set_title(
            f"Interval SHAP sensitivity — {target_label(tgt)}\n"
            r"(aleatoric PI drivers; $\tau=0.95$ vs $0.05$)"
        )
        _save(fig, f"shap_interval_sensitivity_{tgt}.png")

        # Beeswarm of local Δφ colored by feature value
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        expl = shap.Explanation(
            values=d_full,
            data=Xte_np,
            feature_names=FACTORS,
        )
        shap.plots.beeswarm(
            expl,
            ax=ax,
            show=False,
            color=beeswarm_cmap,
            color_bar=True,
            plot_size=None,
            order=np.argsort(-mean_abs_d),
        )
        ax.set_xlabel(r"$\Delta\phi=\phi_{0.95}-\phi_{0.05}$")
        _save(fig, f"shap_interval_beeswarm_{tgt}.png")

        # --- τ-attribution dynamics ---
        qdf = pd.DataFrame([r for r in q_imp_rows if r["target"] == tgt])
        ma_lo = np.abs(sv_lo).mean(0)
        ma_med = np.abs(sv_med).mean(0)
        ma_hi = np.abs(sv_hi).mean(0)
        mean_abs_d_lower = np.abs(d_lower).mean(0)
        mean_abs_d_upper = np.abs(d_upper).mean(0)
        for k, f in enumerate(FACTORS):
            sub = qdf[qdf.factor == f].sort_values("tau")
            taus_arr = sub["tau"].to_numpy(dtype=float)
            ma_arr = sub["mean_abs_shap"].to_numpy(dtype=float)
            slope = float(np.polyfit(taus_arr, ma_arr, 1)[0]) if len(taus_arr) >= 2 else 0.0
            sh_lo = float(sub.loc[np.isclose(sub["tau"], 0.05), "share_pct"].iloc[0])
            sh_med = float(sub.loc[np.isclose(sub["tau"], 0.50), "share_pct"].iloc[0])
            sh_hi = float(sub.loc[np.isclose(sub["tau"], 0.95), "share_pct"].iloc[0])
            tau_sens_rows.append(
                dict(
                    target=tgt,
                    factor=f,
                    mean_abs_shap_q05=float(ma_lo[k]),
                    mean_abs_shap_q50=float(ma_med[k]),
                    mean_abs_shap_q95=float(ma_hi[k]),
                    delta_mean_abs_full=float(ma_hi[k] - ma_lo[k]),
                    delta_mean_abs_lower=float(ma_med[k] - ma_lo[k]),
                    delta_mean_abs_upper=float(ma_hi[k] - ma_med[k]),
                    slope_mean_abs_vs_tau=slope,
                    share_q05=sh_lo,
                    share_q50=sh_med,
                    share_q95=sh_hi,
                    delta_share_full=sh_hi - sh_lo,
                    mean_abs_local_delta_full=float(mean_abs_d[k]),
                    mean_abs_local_delta_lower=float(mean_abs_d_lower[k]),
                    mean_abs_local_delta_upper=float(mean_abs_d_upper[k]),
                    mean_signed_local_delta_full=float(mean_d[k]),
                )
            )

        # Grouped bars: Δ mean|SHAP| lower / upper / full span
        x = np.arange(len(FACTORS))
        w = 0.25
        fig, ax = plt.subplots(figsize=figsize(height=3.8))
        d_lo_bar = ma_med - ma_lo
        d_hi_bar = ma_hi - ma_med
        d_span_bar = ma_hi - ma_lo
        ax.bar(
            x - w,
            d_lo_bar,
            width=w,
            color=REF_COLOR,
            label=r"$\Delta$ lower ($\tau$ 0.50−0.05)",
        )
        ax.bar(
            x,
            d_hi_bar,
            width=w,
            color=target_color(tgt),
            alpha=0.65,
            label=r"$\Delta$ upper ($\tau$ 0.95−0.50)",
        )
        ax.bar(
            x + w,
            d_span_bar,
            width=w,
            color=target_color(tgt),
            edgecolor="black",
            linewidth=0.6,
            label=r"$\Delta$ full ($\tau$ 0.95−0.05)",
        )
        ax.axhline(0, color=REF_COLOR, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(FACTORS)
        ax.set_ylabel(r"$\Delta$ mean $|$SHAP$|$")
        ax.set_title(f"SHAP change across τ — {target_label(tgt)}")
        ax.legend(fontsize=8, frameon=False, loc="best")
        _save(fig, f"shap_tau_delta_{tgt}.png")

    _csv(pd.DataFrame(interval_rows), "shap_interval_delta.csv")
    _csv(pd.DataFrame(tau_sens_rows), "shap_tau_sensitivity.csv")

    # ------------------------------------------------------------------
    # 4) Quantile interaction share vs τ
    # ------------------------------------------------------------------
    qinter_rows = []
    if not SKIP_INTER:
        print("\n=== Quantile interactions ===")
        key_pair = {"log_abs": ("Vs1", "Height"), "f_ratio": ("CoV", "aHV")}
        for tgt in Q_TARGETS:
            a, b = key_pair[tgt]
            ia, ib = FACTORS.index(a), FACTORS.index(b)
            for tau in TAUS:
                model = quant_models[tgt][tau]
                si = cached_shap(
                    f"shap_seed_qinter_{tgt}_tau{int(tau * 100):02d}",
                    lambda m=model: shap.TreeExplainer(m).shap_interaction_values(X_inter),
                    force=FORCE,
                )
                M = np.abs(si).mean(0)
                off = M.copy()
                np.fill_diagonal(off, 0)
                total_inter = float(off.sum())
                total_main = float(np.diag(M).sum())
                qinter_rows.append(
                    dict(
                        target=tgt,
                        tau=tau,
                        key_pair=f"{a}x{b}",
                        pair_abs=float(M[ia, ib]),
                        pair_pct_of_interactions=100 * 2 * M[ia, ib] / total_inter
                        if total_inter > 0
                        else 0.0,
                        interaction_fraction=100 * total_inter / (total_main + total_inter)
                        if (total_main + total_inter) > 0
                        else 0.0,
                    )
                )

            # Separate interaction-fraction plot per target
            s = pd.DataFrame([r for r in qinter_rows if r["target"] == tgt]).sort_values("tau")
            fig, ax = plt.subplots(figsize=(5.2, 3.6))
            ax.plot(
                s["tau"],
                s["interaction_fraction"],
                "o-",
                color=target_color(tgt),
                ms=6,
                lw=1.8,
                label="all off-diagonal",
            )
            ax.plot(
                s["tau"],
                s["pair_pct_of_interactions"],
                "s--",
                color=FACTOR_COLORS[a],
                ms=5,
                lw=1.4,
                label=f"{a}×{b} share of interactions",
            )
            ax.set_xlabel(r"quantile $\tau$")
            ax.set_ylabel("%")
            ax.set_title(f"Interaction structure vs τ — {target_label(tgt)}")
            ax.legend(fontsize=8, frameon=False)
            _save(fig, f"quantile_interactions_{tgt}.png")

        _csv(pd.DataFrame(qinter_rows), "quantile_shap_interactions.csv")
    else:
        print("\n=== Skipping quantile interactions (--skip-interactions) ===")

    # ------------------------------------------------------------------
    # 5) PDP by target (all factors as subplots; τ∈{0.05,0.50,0.95})
    # ------------------------------------------------------------------
    if not SKIP_PDP:
        print("\n=== PDP + centered H² ===")
        pdp_rows, h2_rows = [], []
        grids = {
            f: np.linspace(X_pdp[:, k].min(), X_pdp[:, k].max(), PDP_GRID)
            for k, f in enumerate(FACTORS)
        }
        tau_style = {
            0.05: dict(ls="--", lw=1.4, alpha=0.75),
            0.50: dict(ls="-", lw=1.8, alpha=0.95),
            0.95: dict(ls=":", lw=1.4, alpha=0.75),
        }
        for tgt in Q_TARGETS:
            pd1 = {tau: {} for tau in TAIL_TAUS}
            for tau in TAIL_TAUS:
                model = quant_models[tgt][tau]
                for k, f in enumerate(FACTORS):
                    curve = pd_1d(model, X_pdp, k, grids[f])
                    pd1[tau][f] = curve
                    for x, y in zip(grids[f], curve):
                        pdp_rows.append(
                            dict(target=tgt, tau=tau, factor=f, x=float(x), pdp=float(y))
                        )

            n_fac = len(FACTORS)
            fig, axes = plt.subplots(
                1,
                n_fac,
                figsize=figsize(height=3.4),
                sharey=True,
                layout="constrained",
            )
            for col, f in enumerate(FACTORS):
                ax = axes[col]
                for tau in TAIL_TAUS:
                    ax.plot(
                        grids[f],
                        pd1[tau][f],
                        color=FACTOR_COLORS[f],
                        label=rf"$\tau={tau:g}$",
                        **tau_style[tau],
                    )
                ax.set_xlabel(f)
                ax.set_title(f)
                panel_letter(ax, "abcde"[col])
            axes[0].set_ylabel(f"partial dependence ({target_label(tgt)})")
            # Single shared τ legend from the first panel
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper center",
                ncol=len(TAIL_TAUS),
                fontsize=8,
                frameon=False,
                bbox_to_anchor=(0.5, 1.02),
            )
            _save(fig, f"pdp_{tgt}.png")

        _csv(pd.DataFrame(pdp_rows), "pdp_by_quantile.csv")
        _csv(pd.DataFrame(h2_rows), "h2_by_quantile.csv")
    else:
        print("\n=== Skipping PDP/H² (--skip-pdp) ===")

    elapsed = time.monotonic() - t_all
    print(f"\nDone in {elapsed:.0f}s. Models were not refit; SHAP used cache where available.")
    print(f"Plots/data: {result_path('plots', '_').rsplit('/', 1)[0]}")


if __name__ == "__main__":
    main()
