"""SHAP + PDP: kernel-parameter NGBoost vs the marginal NGBoost (μ, log σ).

- Kernel-parameter NGBoost (``train_kernel_ngboost.py``): permutation SHAP of each
  target's predictive mean over all 243 cells (background = 100 cells) and a
  1-D PDP per design factor.
- Marginal NGBoost (``chi_ngboost/models/ngboost_<metric>.pkl``): permutation
  SHAP of μ and log σ on a seed-holdout sample (same sampler as
  ``chi_shap/shap_ngboost.py``) and 1-D PDPs averaged over nodes.

Importance shares are normalized over the 5 design factors (``node_z`` is
reported separately for the marginal model). PDPs are centered and divided by
the SD of the target's predictions over the design so curve shapes and relative
amplitudes are comparable across targets.

Writes CSV + PDFs + ``summary.md`` under ``figure_dir("chi_joint", "compare_shap_pdp")``.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FACTORS,
    FEATURES,
    METRIC,
    ZCOLS,
    add_design_columns,
    cell_design_table,
    load_or_make_split,
    load_ratios,
    load_sibling,
    models_dir,
    out_dir,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import (  # noqa: E402
    add_panel_label,
    apply_full_paper_style,
    factor_color,
    figsize,
    save_figure,
)

warnings.filterwarnings("ignore")

shap_ngb = load_sibling("chi_shap", "shap_ngboost")
shap_common = load_sibling("chi_shap", "common")
ngb_common = load_sibling("chi_ngboost", "common")
apply_full_paper_style(auto_format=True, frame="open", grid=False)

BG_N = 100
MARGINAL_EXPLAIN_N = 400
MARGINAL_PDP_N = 2000
SAMPLE_SEED = 5
PDP_GRID_N = 25

TARGET_LABELS = {
    "marginal_mu": r"$\mu$ (marginal)",
    "marginal_log_sigma": r"$\log\sigma$ (marginal)",
    "a_b": r"$\log(w_b/w_n)$",
    "a_s": r"$\log(w_s/w_n)$",
    "log_s": r"$\log s$",
    "log_nu": r"$\log\nu$",
    "omega100": r"$\omega=100/b$",
    "log_h95": r"$\log h_{95}$",
}


def pdp_1d(predict_fn, X: np.ndarray, j: int, *, n_grid: int = PDP_GRID_N):
    """Centered 1D partial dependence (same as ``chi_shap/ale_effects.pdp_1d``,
    copied because importing that module requires LightGBM/libgomp)."""
    xj = np.asarray(X[:, j], dtype=float)
    grid = np.unique(np.quantile(xj, np.linspace(0.0, 1.0, n_grid)))
    vals = np.empty(grid.size, dtype=float)
    X_work = X.copy()
    for i, g in enumerate(grid):
        X_work[:, j] = g
        vals[i] = float(np.mean(predict_fn(X_work)))
    return grid, vals - float(np.mean(vals))


def factor_label(z: str) -> str:
    # Raw factor name: apply_full_paper_style(auto_format=True) maps it to LaTeX.
    return z[:-2] if z.endswith("_z") else z


def shares(sv: np.ndarray, feats: list[str]) -> dict[str, float]:
    """mean |SHAP| per design factor, normalized over the 5 factors."""
    mabs = np.mean(np.abs(sv), axis=0)
    d = dict(zip(feats, mabs, strict=True))
    tot = sum(d[z] for z in ZCOLS)
    return {z: d[z] / tot for z in ZCOLS} | {
        "_node_mean_abs": d.get("node_z", np.nan),
        "_factor_total": tot,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default=METRIC)
    ap.add_argument("--kernel", default="coswm")
    args = ap.parse_args()
    metric, kernel = args.metric, args.kernel
    out = out_dir("compare_shap_pdp")
    rng = np.random.default_rng(SAMPLE_SEED)

    cv_tab = pd.read_csv(
        out_dir("train_kernel_ngboost") / f"kernel_ngboost_cv_{metric}_{kernel}.csv"
    )
    cv_tab = cv_tab[cv_tab.get("skipped", pd.Series(index=cv_tab.index)).isna()]
    cv_r2 = dict(zip(cv_tab["target"], cv_tab["cv_r2"], strict=True))

    df = add_design_columns(load_ratios())
    design = cell_design_table(df)
    _, te = load_or_make_split(df)
    Xd = design[ZCOLS].to_numpy(dtype=float)
    bg_cells = rng.choice(len(Xd), size=min(BG_N, len(Xd)), replace=False)

    predictors: dict[str, tuple[object, np.ndarray, np.ndarray, list[str]]] = {}
    # Marginal NGBoost (features include node_z)
    marg = joblib.load(ngb_common.models_dir() / f"ngboost_{metric}.pkl")
    bg_idx, ex_idx, _ = shap_common.make_shap_sample(df, te)
    ex_idx = rng.choice(ex_idx, size=min(MARGINAL_EXPLAIN_N, len(ex_idx)), replace=False)
    bg_idx = rng.choice(bg_idx, size=min(BG_N, len(bg_idx)), replace=False)
    X_bg_m = df.iloc[bg_idx][FEATURES].to_numpy(dtype=float)
    X_ex_m = df.iloc[ex_idx][FEATURES].to_numpy(dtype=float)
    X_pdp_m = df.iloc[rng.choice(te, size=MARGINAL_PDP_N, replace=False)][FEATURES].to_numpy(
        dtype=float
    )
    predictors["marginal_mu"] = (shap_ngb._MuModel(marg).predict, X_bg_m, X_ex_m, FEATURES)
    predictors["marginal_log_sigma"] = (
        shap_ngb._LogSigmaModel(marg).predict,
        X_bg_m,
        X_ex_m,
        FEATURES,
    )
    pdp_X = {"marginal_mu": X_pdp_m, "marginal_log_sigma": X_pdp_m}
    # Kernel-parameter NGBoosts (design features only)
    for t in cv_r2:
        m = joblib.load(models_dir() / f"ngboost_kparam_{kernel}_{t}_{metric}.pkl")
        predictors[t] = (shap_ngb._MuModel(m).predict, Xd[bg_cells], Xd, ZCOLS)
        pdp_X[t] = Xd

    imp_rows, pdp_rows = [], []
    for t, (fn, X_bg, X_ex, feats) in predictors.items():
        print(f"SHAP {t} …")
        sv = shap_ngb._explain(fn, X_bg, X_ex)
        sh = shares(sv, feats)
        signed = np.mean(sv, axis=0)
        for j, f in enumerate(feats):
            imp_rows.append(
                {
                    "target": t,
                    "feature": f,
                    "mean_abs_shap": float(np.mean(np.abs(sv[:, j]))),
                    "mean_signed_shap": float(signed[j]),
                    "share_of_factors": sh.get(f, np.nan),
                    "cv_r2": cv_r2.get(t, np.nan),
                }
            )
        Xp = pdp_X[t]
        pred_sd = float(np.std(fn(Xp if t.startswith("marginal") else Xd)))
        for f in ZCOLS:
            j = feats.index(f)
            grid, vals = pdp_1d(fn, Xp, j, n_grid=PDP_GRID_N)
            for g, v in zip(grid, vals, strict=True):
                pdp_rows.append(
                    {
                        "target": t,
                        "feature": f,
                        "x_z": float(g),
                        "pdp_centered": float(v),
                        "pdp_std": float(v / pred_sd),
                        "pred_sd": pred_sd,
                    }
                )

    imp = pd.DataFrame(imp_rows)
    pdp = pd.DataFrame(pdp_rows)
    # raw factor values for PDP x-axis
    for f in FACTORS:
        zc = f"{f}_z"
        mu_f, sd_f = float(df[f].mean()), float(df[f].std(ddof=0))
        pdp.loc[pdp["feature"] == zc, "x_raw"] = pdp.loc[pdp["feature"] == zc, "x_z"] * sd_f + mu_f
    imp.to_csv(out / f"shap_importance_compare_{metric}_{kernel}.csv", index=False)
    pdp.to_csv(out / f"pdp_compare_{metric}_{kernel}.csv", index=False)

    targets = list(predictors)
    share_mat = (
        imp[imp["feature"].isin(ZCOLS)]
        .pivot(index="target", columns="feature", values="share_of_factors")
        .loc[targets, ZCOLS]
    )
    rho = pd.DataFrame(
        [
            [spearmanr(share_mat.loc[a], share_mat.loc[b]).statistic for b in targets]
            for a in targets
        ],
        index=targets,
        columns=targets,
    )
    rho.to_csv(out / f"importance_rank_spearman_{metric}_{kernel}.csv")

    # --- figure: importance heatmap + Spearman ----------------------------
    ylabels = [
        TARGET_LABELS.get(t, t)
        + ("" if t.startswith("marginal") else f"  (CV $R^2$={cv_r2[t]:.2f})")
        for t in targets
    ]
    fig, (ax_h, ax_r) = plt.subplots(
        1, 2, figsize=figsize(aspect=0.45), gridspec_kw={"width_ratios": [1.25, 1]}
    )
    im = ax_h.imshow(share_mat.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(share_mat.shape[0]):
        for j in range(share_mat.shape[1]):
            v = share_mat.iat[i, j]
            ax_h.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=5.5,
                color="w" if v > 0.55 else "k",
            )
    ax_h.set_xticks(range(len(ZCOLS)))
    ax_h.set_xticklabels([factor_label(z) for z in ZCOLS], rotation=30, ha="right")
    ax_h.set_yticks(range(len(targets)))
    ax_h.set_yticklabels(ylabels, fontsize=5.5)
    ax_h.axhline(1.5, color="k", lw=0.6)
    fig.colorbar(im, ax=ax_h, fraction=0.04, pad=0.02, label="Share of mean |SHAP|")
    add_panel_label(ax_h, 0)
    im2 = ax_r.imshow(rho.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    short = [TARGET_LABELS.get(t, t).split(" (")[0] for t in targets]
    ax_r.set_xticks(range(len(targets)))
    ax_r.set_xticklabels(short, rotation=60, ha="right", fontsize=5.5)
    ax_r.set_yticks(range(len(targets)))
    ax_r.set_yticklabels(short, fontsize=5.5)
    fig.colorbar(im2, ax=ax_r, fraction=0.04, pad=0.02, label=r"Spearman $\rho$ of factor ranks")
    add_panel_label(ax_r, 1)
    fig.tight_layout()
    save_figure(fig, f"shap_importance_compare_{metric}_{kernel}", out_dir=out)
    plt.close(fig)

    # --- figure: PDP grid (rows = factors, cols = marginal vs kernel) -----
    kern_targets = [t for t in targets if not t.startswith("marginal")]
    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(
        2, len(ZCOLS), figsize=figsize(aspect=0.5), sharey="row", squeeze=False
    )
    for j, f in enumerate(ZCOLS):
        for r, group in enumerate((["marginal_mu", "marginal_log_sigma"], kern_targets)):
            ax = axes[r, j]
            for i, t in enumerate(group):
                s = pdp[(pdp["target"] == t) & (pdp["feature"] == f)]
                style = {"ls": "-" if t != "marginal_log_sigma" else "--"}
                color = factor_color(f) if r == 0 else cmap(i)
                ax.plot(
                    s["x_raw"],
                    s["pdp_std"],
                    marker="o",
                    ms=2,
                    lw=0.9,
                    color=color,
                    label=TARGET_LABELS.get(t, t),
                    **style,
                )
            ax.axhline(0, color="0.55", lw=0.5, ls=":")
            if r == 1:
                ax.set_xlabel(factor_label(f))
            if j == 0:
                ax.set_ylabel("PDP / SD(pred)" + ("\nmarginal" if r == 0 else "\nkernel params"))
            add_panel_label(ax, r * len(ZCOLS) + j)
    for r in range(2):
        axes[r, -1].legend(frameon=False, fontsize=5, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    save_figure(fig, f"pdp_compare_{metric}_{kernel}", out_dir=out)
    plt.close(fig)

    # --- summary ----------------------------------------------------------
    top = share_mat.idxmax(axis=1).map(factor_label)
    node_share = imp[(imp["feature"] == "node_z")].set_index("target")["mean_abs_shap"]
    fac_tot = imp[imp["feature"].isin(ZCOLS)].groupby("target")["mean_abs_shap"].sum()
    lines = [
        f"# SHAP / PDP comparison: kernel-parameter NGBoost vs marginal NGBoost ({metric}, {kernel})",
        "",
        "## Definitions",
        "",
        "- Marginal targets: NGBoost μ(d, x) and log σ(d, x) of Y = ln χ (permutation SHAP on a seed-holdout sample; PDP averaged over nodes).",
        "- Kernel targets: NGBoost-predicted mean of each partially pooled joint-layer parameter θ_k(d) (243 cells).",
        "- `share_of_factors`: mean |SHAP| of a factor ÷ sum over the 5 design factors (node_z excluded).",
        "- PDP curves are centered and divided by SD of the target's predictions over the design.",
        "",
        "## Factor importance shares",
        "",
        share_mat.rename(columns=factor_label)
        .assign(cv_r2=[cv_r2.get(t, np.nan) for t in targets])
        .to_markdown(floatfmt=".3f"),
        "",
        "Top factor per target: " + ", ".join(f"{t} → {v}" for t, v in top.items()),
        "",
        f"Marginal node_z mean |SHAP| relative to factor total: μ {node_share.get('marginal_mu', np.nan) / fac_tot['marginal_mu']:.3f}, "
        f"log σ {node_share.get('marginal_log_sigma', np.nan) / fac_tot['marginal_log_sigma']:.3f}.",
        "",
        "## Spearman rank agreement of factor importance",
        "",
        rho.to_markdown(floatfmt=".2f"),
        "",
        "## Reading guide / caveats",
        "",
        "- Only targets with clearly positive CV R² (see `train_kernel_ngboost/summary.md`) carry interpretable SHAP/PDP; for the rest the attributions describe noise in per-cell estimates.",
        "- Kernel parameters describe correlation along a 200 m array; they are descriptive of the model response, not calibrated soil correlation lengths.",
        "- Each design factor has 3 levels, so PDPs have 3 support points.",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "meta.json").write_text(
        json.dumps(
            {
                "metric": metric,
                "kernel": kernel,
                "targets": targets,
                "bg_n": BG_N,
                "marginal_explain_n": MARGINAL_EXPLAIN_N,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
