"""Heteroscedasticity diagnostics for the center recorder.

Shows residual funnels, mean–variance coupling, η² decompositions, and
per-factor spread. Documents why point summaries / OLS fail when variance
is factor-dependent (raw amplitude, natural-log amplitude, and f_ratio).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import levene, pearsonr
from statsmodels.stats.diagnostic import het_breuschpagan

from seiskit.plot_config import apply_style, panel_letter, result_path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    COMPARE_COLOR,
    FACTOR_COLORS,
    FACTORS,
    FIG_DPI,
    FIG_WIDTH,
    REF_COLOR,
    figsize,
    load_channel50,
    target_color,
    target_label,
)


def bp_test(d50: pd.DataFrame, target: str):
    X = sm.add_constant(d50[[c + "_z" for c in FACTORS]])
    y = d50[target]
    m = sm.OLS(y, X).fit()
    lm, lm_p, F, F_p = het_breuschpagan(m.resid, np.asarray(X))
    return m, lm, lm_p


def eta2(dfin: pd.DataFrame, val: str, fac: str) -> float:
    grand = dfin[val].mean()
    ss_tot = ((dfin[val] - grand) ** 2).sum()
    ss_b = sum(len(g) * (g.mean() - grand) ** 2 for _, g in dfin.groupby(fac)[val])
    return ss_b / ss_tot


def main() -> None:
    d50 = load_channel50()
    Z = d50[FACTORS].copy()
    Zs = (Z - Z.mean()) / Z.std()
    Zs.columns = [c + "_z" for c in FACTORS]
    d50 = pd.concat([d50, Zs], axis=1)

    ols_models = {}
    for tgt in ["f_ratio", "abs_TF_ratio", "log_abs"]:
        m, lm, lmp = bp_test(d50, tgt)
        ols_models[tgt] = m
        print(f"Breusch-Pagan {tgt}: LM={lm:.1f} p={lmp:.2e}")

    lev_rows = []
    for tgt in ["f_ratio", "log_abs", "abs_TF_ratio"]:
        for fc in FACTORS:
            gs = [d50.loc[d50[fc] == lv, tgt].values for lv in sorted(d50[fc].unique())]
            W, p = levene(*gs, center="median")
            lev_rows.append(dict(target=tgt, factor=fc, levene_W=W, p=p))
    print(pd.DataFrame(lev_rows).to_string(index=False))

    cell_stats = (
        d50.groupby(FACTORS)
        .agg(
            f_mean=("f_ratio", "mean"),
            f_std=("f_ratio", "std"),
            a_mean=("log_abs", "mean"),
            a_std=("log_abs", "std"),
            raw_mean=("abs_TF_ratio", "mean"),
            raw_std=("abs_TF_ratio", "std"),
        )
        .reset_index()
    )

    eta_rows = []
    for fc in FACTORS:
        eta_rows.append(
            dict(
                factor=fc,
                absTF_mean_eta2=eta2(cell_stats, "a_mean", fc),
                absTF_std_eta2=eta2(cell_stats, "a_std", fc),
                f_mean_eta2=eta2(cell_stats, "f_mean", fc),
                f_std_eta2=eta2(cell_stats, "f_std", fc),
            )
        )
    eta_df = pd.DataFrame(eta_rows)

    apply_style(auto_format=True, font_size=10, frame="open")
    fig, axes = plt.subplots(2, 3, figsize=figsize(height=FIG_WIDTH * 0.75))

    for ax, tgt in [
        (axes[0, 0], "f_ratio"),
        (axes[0, 1], "log_abs"),
    ]:
        m = ols_models[tgt]
        fit = m.fittedvalues.values
        res = m.resid.values
        idx = np.linspace(0, len(fit) - 1, 4000).astype(int)
        ax.plot(fit[idx], res[idx], "o", ms=2, color=target_color(tgt), alpha=0.3)
        ax.axhline(0, color=REF_COLOR, lw=1)
        ax.set_xlabel("OLS fitted value")
        ax.set_ylabel("residual")
        ax.set_title(f"{target_label(tgt)}: residual funnel", loc="left")
    panel_letter(axes[0, 0], "a")
    panel_letter(axes[0, 1], "b")
    axes[0, 0].set_ylim(-0.4, 0.8)
    axes[0, 1].set_ylim(-2, 3)
    axes[0, 0].set_xlim(0.875, 1.05)
    axes[0, 1].set_xlim(-3.0, -1.0)

    ax = axes[0, 2]
    ax.plot(
        cell_stats.a_mean,
        cell_stats.a_std,
        "o",
        ms=4,
        color=target_color("log_abs"),
        alpha=0.6,
        label=rf"{target_label('log_abs')} ($\rho$={pearsonr(cell_stats.a_mean, cell_stats.a_std)[0]:.2f})",
    )
    ax.plot(
        cell_stats.f_mean,
        cell_stats.f_std,
        "s",
        ms=4,
        color=target_color("f_ratio"),
        alpha=0.6,
        label=rf"{target_label('f_ratio')} ($\rho$={pearsonr(cell_stats.f_mean, cell_stats.f_std)[0]:.2f})",
    )
    ax.set_xlabel("design-cell mean")
    ax.set_ylabel("design-cell std")
    ax.set_ylim(0, 1.4)
    ax.set_xlim(-4, 2)
    ax.set_title("Mean-variance coupling (243 cells)", loc="left")
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    panel_letter(ax, "c")

    for ax, (mcol, scol, tgt, pl) in zip(
        [axes[1, 0], axes[1, 1]],
        [
            ("f_mean_eta2", "f_std_eta2", "f_ratio", "d"),
            ("absTF_mean_eta2", "absTF_std_eta2", "log_abs", "e"),
        ],
    ):
        x = np.arange(5)
        w = 0.38
        ax.bar(x - w / 2, eta_df[mcol], w, color=REF_COLOR, label="Explains Mean")
        ax.bar(x + w / 2, eta_df[scol], w, color=COMPARE_COLOR, label="Explains Std")
        ax.set_xticks(x)
        ax.set_xticklabels(FACTORS, fontsize=10)
        ax.set_ylabel(r"$\eta^2$ (variance explained)")
        ax.set_title(f"{target_label(tgt)}: mean vs variance drivers", loc="left")
        ax.legend(fontsize=9, frameon=False)
        panel_letter(ax, pl)

    axes[1, 0].set_ylim(0, 0.40)
    axes[1, 1].set_ylim(0, 0.8)

    ax = axes[1, 2]
    xpos = 0
    for fc in ["CoV", "aHV", "Height"]:
        g = d50.groupby(fc)["f_ratio"].std()
        xs = np.arange(len(g)) + xpos
        ax.bar(xs, g.values, color=FACTOR_COLORS[fc], label=fc, width=0.8)
        xpos += len(g) + 0.6
    ax.set_ylabel(f"std of {target_label('f_ratio')}")
    ax.set_xticks([])
    ax.set_title(f"{target_label('f_ratio')} spread by level", loc="left")
    ax.set_ylim(0, 0.14)
    ax.legend(fontsize=9, frameon=False)
    panel_letter(ax, "f")

    fig.tight_layout()
    out = result_path("plots", "heteroscedasticity_diagnostics.png")
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
