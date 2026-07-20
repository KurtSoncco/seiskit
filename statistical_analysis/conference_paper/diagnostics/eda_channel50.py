"""EDA for the center recorder (channel 50).

3×3 panel: raw vs natural-log amplitude and frequency-ratio distributions,
marginal factor means, seed structure, and a Height×Vs1 interaction heatmap.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from seiskit.plot_config import apply_style, get_crameri_cmap, panel_letter, result_path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    FACTORS,
    FIG_DPI,
    COMPARE_COLOR,
    REF_COLOR,
    load_channel50,
    target_color,
    target_label,
)


def main() -> None:
    d50 = load_channel50()
    apply_style(auto_format=True, font_size=10, frame="open")

    fig = plt.figure(figsize=(13, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.25, wspace=0.30)

    # a) f_ratio
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(d50["f_ratio"], bins=60, color=target_color("f_ratio"), edgecolor="none")
    ax.axvline(1.0, color=REF_COLOR, ls="--", lw=1)
    ax.set_xlabel(target_label("f_ratio"))
    ax.set_ylabel("count")
    ax.set_title(f"{target_label('f_ratio')} — near 1, right tail", loc="left")
    panel_letter(ax, "a")

    # b) raw amplitude
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(
        d50["abs_TF_ratio"], bins=80, color=target_color("abs_TF_ratio"), edgecolor="none"
    )
    ax.axvline(1.0, color=REF_COLOR, ls="--", lw=1)
    ax.set_xlabel(target_label("abs_TF_ratio"))
    ax.set_ylabel("count")
    ax.set_title(f"{target_label('abs_TF_ratio')} — strong right skew", loc="left")
    panel_letter(ax, "b")

    # c) natural-log amplitude (canonical transform)
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(d50["log_abs"], bins=70, color=target_color("log_abs"), edgecolor="none", alpha=0.9)
    ax.set_xlabel(target_label("log_abs"))
    ax.set_ylabel("count")
    ax.set_title(f"{target_label('log_abs')} — near symmetric", loc="left")
    panel_letter(ax, "c")

    for j, tgt in enumerate(["f_ratio", "abs_TF_ratio"]):
        ax = fig.add_subplot(gs[1, j])
        col = target_color(tgt)
        ttl = target_label(tgt)
        xpos = 0
        xticks, xlabs = [], []
        for fc in FACTORS:
            g = d50.groupby(fc)[tgt].mean()
            xs = np.arange(len(g)) + xpos
            ax.plot(xs, g.values, "o-", color=col, ms=5)
            for x, lv in zip(xs, g.index):
                xticks.append(x)
                xlabs.append(f"{lv:g}")
            xpos += len(g) + 0.8
            ax.axvline(xpos - 1.4, color="0.85", lw=0.8)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabs, fontsize=6)
        ax.set_ylabel(f"mean {ttl}")
        ax.set_title(f"{ttl}: marginal means", loc="left")
        panel_letter(ax, "de"[j])

    ax = fig.add_subplot(gs[1, 2])
    xpos = 0
    xticks, xlabs = [], []
    for fc in FACTORS:
        g = d50.groupby(fc)["f_ratio"].std()
        xs = np.arange(len(g)) + xpos
        ax.plot(xs, g.values, "s-", color=COMPARE_COLOR, ms=5)
        for x, lv in zip(xs, g.index):
            xticks.append(x)
            xlabs.append(f"{lv:g}")
        xpos += len(g) + 0.8
        ax.axvline(xpos - 1.4, color="0.85", lw=0.8)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabs, fontsize=6)
    ax.set_ylabel(f"std of {target_label('f_ratio')}")
    ax.set_title(f"{target_label('f_ratio')}: spread — CoV drives it", loc="left")
    panel_letter(ax, "f")

    ax = fig.add_subplot(gs[2, 0])
    cell = d50[
        (d50.Vs1 == 230)
        & (d50.Height == 50)
        & (d50.CoV == 0.2)
        & (d50.rH == 30)
        & (d50.aHV == 10)
    ]
    ax.plot(
        cell["seed"],
        cell["abs_TF_ratio"],
        "o",
        ms=3,
        color=COMPARE_COLOR,
        alpha=0.7,
    )
    ax.axhline(cell["abs_TF_ratio"].mean(), color=REF_COLOR, ls="--", lw=1)
    ax.set_xlabel("seed")
    ax.set_ylabel(target_label("abs_TF_ratio"))
    ax.set_title("Seed scatter, one design cell", loc="left")
    panel_letter(ax, "g")

    ax = fig.add_subplot(gs[2, 1])
    sm = d50.groupby("seed")["abs_TF_ratio"].mean()
    ax.plot(sm.index, sm.values, "o", ms=3, color=COMPARE_COLOR, alpha=0.7)
    ax.axhline(sm.mean(), color=REF_COLOR, ls="--", lw=1, label=f"grand mean {sm.mean():.2f}")
    ax.set_xlabel("seed")
    ax.set_ylabel(f"mean {target_label('abs_TF_ratio')}")
    ax.set_title("Per-seed mean varies (not flat)", loc="left")
    ax.legend(loc="upper right", fontsize=6)
    panel_letter(ax, "h")

    ax = fig.add_subplot(gs[2, 2])
    piv = d50.groupby(["Height", "Vs1"])["abs_TF_ratio"].mean().unstack()
    im = ax.imshow(piv.values, cmap=get_crameri_cmap("batlow"), aspect="auto", origin="lower")
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{v:g}" for v in piv.columns])
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"{v:g}" for v in piv.index])
    ax.set_xlabel("Vs1")
    ax.set_ylabel("Height")
    for i in range(3):
        for k in range(3):
            v = piv.values[i, k]
            ax.text(
                k,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                color="white" if v < 0.3 else "black",
                fontsize=7,
            )
    ax.set_title(f"mean {target_label('abs_TF_ratio')}: Vs1 × Height", loc="left")
    panel_letter(ax, "i")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "Center recorder: distributions, factor effects, seed structure  ·  "
        "n=24,300 (243 cells × 100 seeds)",
        fontsize=10,
    )
    out = result_path("plots", "eda_channel50.png")
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
