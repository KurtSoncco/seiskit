"""EDA for the center recorder (channel 50).

3×3 panel: raw vs natural-log amplitude and frequency-ratio distributions,
marginal factor means, seed structure, and a Height×Vs1 interaction heatmap.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from seiskit.plot_config import apply_style, panel_letter, result_path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    FIG_WIDTH,
    REF_COLOR,
    load_channel50,
    target_color,
    target_label,
)

SAVE_DPI = 600
TIGHT_PAD_IN = 0.02


def save_tight_exact_width(fig: plt.Figure, out: str, *, fig_width: float = FIG_WIDTH) -> None:
    """Save tight-cropped PNG at exactly fig_width inches and 600 dpi."""
    target_px = int(round(fig_width * SAVE_DPI))
    fig.savefig(out, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=TIGHT_PAD_IN)
    img = Image.open(out)
    if img.width != target_px:
        target_h = max(1, int(round(img.height * (target_px / img.width))))
        img = img.resize((target_px, target_h), Image.Resampling.LANCZOS)
    img.save(out, dpi=(SAVE_DPI, SAVE_DPI))


def main() -> None:
    d50 = load_channel50()
    apply_style(auto_format=True, font_size=10, frame="open")

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH * 0.20))
    gs = fig.add_gridspec(1, 3, hspace=0.25, wspace=0.35)

    # a) f_ratio
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(d50["f_ratio"], bins=60, color=target_color("f_ratio"), edgecolor="none")
    ax.axvline(1.0, color=REF_COLOR, ls="--", lw=1)
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_locator(plt.MultipleLocator(1500))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
    ax.grid(True, which="minor", axis="x", alpha=0.12, lw=0.4)
    ax.set_ylim(0, 4500)
    ax.set_xlim(0.5, 1.5)
    ax.set_xlabel(f"{target_label('f_ratio')}")
    panel_letter(ax, "a")

    # b) raw amplitude
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(d50["abs_TF_ratio"], bins=80, color=target_color("abs_TF_ratio"), edgecolor="none")
    ax.axvline(1.0, color=REF_COLOR, ls="--", lw=1)
    ax.set_xlabel("")
    ax.yaxis.set_major_locator(plt.MultipleLocator(3500))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.grid(True, which="minor", axis="x", alpha=0.12, lw=0.4)
    ax.set_ylim(0, 7000)
    ax.set_xlim(0, 2)
    ax.set_ylabel("")  # Do not show y-axis label, but show ticks
    ax.set_xlabel(f"{target_label('abs_TF_ratio')}")
    panel_letter(ax, "b")

    # c) natural-log amplitude (canonical transform)
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(d50["log_abs"], bins=70, color=target_color("log_abs"), edgecolor="none", alpha=0.9)
    ax.axvline(0.0, color=REF_COLOR, ls="--", lw=1)
    ax.set_xlabel("")
    ax.yaxis.set_major_locator(plt.MultipleLocator(500))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(0.5))
    ax.grid(True, which="minor", axis="x", alpha=0.12, lw=0.4)
    ax.set_ylim(0, 1500)
    ax.set_xlim(-4, 2)
    ax.set_ylabel("")  # Do not show y-axis label, but show ticks
    ax.set_xlabel(f"{target_label('log_abs')}")
    panel_letter(ax, "c")

    out = result_path("plots", "eda_channel50.png")
    save_tight_exact_width(fig, out, fig_width=FIG_WIDTH)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
