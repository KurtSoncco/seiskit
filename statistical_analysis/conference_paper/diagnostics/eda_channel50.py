"""EDA for the center recorder (channel 50).

3×3 panel: raw vs natural-log amplitude and frequency-ratio distributions,
marginal factor means, seed structure, and a Height×Vs1 interaction heatmap.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from seiskit.plot_config import apply_style, panel_letter, result_path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    FIG_DPI,
    FIG_WIDTH,
    REF_COLOR,
    load_channel50,
    target_color,
    target_label,
)


def main() -> None:
    d50 = load_channel50()
    apply_style(auto_format=True, font_size=10, frame="open")

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH * 0.5))
    gs = fig.add_gridspec(1, 3, hspace=0.25, wspace=0.30)

    # a) f_ratio
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(d50["f_ratio"], bins=60, color=target_color("f_ratio"), edgecolor="none")
    ax.axvline(1.0, color=REF_COLOR, ls="--", lw=1)
    ax.set_xlabel(target_label("f_ratio"))
    ax.set_ylabel("Count")
    ax.set_ylim(0, 4500)
    ax.set_xlim(0.5, 1.5)
    ax.set_title(f"{target_label('f_ratio')} — near 1, right tail", loc="left")
    panel_letter(ax, "a")

    # b) raw amplitude
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(d50["abs_TF_ratio"], bins=80, color=target_color("abs_TF_ratio"), edgecolor="none")
    ax.axvline(1.0, color=REF_COLOR, ls="--", lw=1)
    ax.set_xlabel(target_label("abs_TF_ratio"))
    ax.set_ylim(0, 7000)
    ax.set_xlim(0, 2)
    ax.set_ylabel("")  # Do not show y-axis label, but show ticks
    ax.set_title(f"{target_label('abs_TF_ratio')} — strong right skew", loc="left")
    panel_letter(ax, "b")

    # c) natural-log amplitude (canonical transform)
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(d50["log_abs"], bins=70, color=target_color("log_abs"), edgecolor="none", alpha=0.9)
    ax.set_xlabel(target_label("log_abs"))
    ax.set_ylim(0, 1200)
    ax.set_xlim(-4, 2)
    ax.set_ylabel("")  # Do not show y-axis label, but show ticks
    ax.set_title(f"{target_label('log_abs')} — near symmetric", loc="left")
    panel_letter(ax, "c")

    fig.suptitle(
        "Fundamental Normalized Peak Parameter Histograms.",
        fontsize=10,
    )
    # fig.tight_layout()
    out = result_path("plots", "eda_channel50.png")
    fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
