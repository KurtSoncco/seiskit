"""Appendix figures for peak-detection reliability (Appendix 2).

Reads Box ``peak_analysis/plots/stability`` metrics CSVs and writes Nature PDFs
under ``figure_dir("appendix_im")``. Does not re-implement ``window_max``;
documents found-rate / modal reliability for the appendix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (  # noqa: E402
    BOX_ROOT,
    LABEL_FONTSIZE,
    TICK_LABELSIZE,
    add_panel_label,
    apply_full_paper_style,
    figsize,
    figure_dir,
    save_figure,
)

apply_full_paper_style(auto_format=True, frame="open", grid=False)

STABILITY = BOX_ROOT / "peak_analysis" / "plots" / "stability"
OUT = figure_dir("appendix_im")


def _load_pooled() -> pd.DataFrame:
    path = STABILITY / "metrics_pooled_by_mode.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


def _load_by_height() -> pd.DataFrame | None:
    path = STABILITY / "metrics_by_height_mode.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path)


def plot_found_rate(pooled: pd.DataFrame, by_h: pd.DataFrame | None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=figsize(aspect=0.45))
    modes = pooled["mode"].to_numpy()
    rates = pooled["found_rate"].to_numpy(dtype=float)
    axes[0].bar(modes.astype(str), rates, color="#4477AA", width=0.65)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_xlabel("Mode")
    axes[0].set_ylabel("Found rate")
    axes[0].axhline(0.95, color="0.5", ls="--", lw=0.8)
    add_panel_label(axes[0], 0)

    if by_h is not None and {"height", "mode", "found_rate"}.issubset(by_h.columns):
        for mode, g in by_h.groupby("mode"):
            axes[1].plot(
                g["height"],
                g["found_rate"],
                marker="o",
                ms=3,
                lw=1.0,
                label=f"mode {mode}",
            )
        axes[1].set_xlabel(r"$H$ (m)")
        axes[1].set_ylabel("Found rate")
        axes[1].set_ylim(0.0, 1.05)
        axes[1].legend(fontsize=TICK_LABELSIZE, frameon=False)
    else:
        axes[1].axis("off")
        axes[1].text(0.1, 0.5, "Height×mode CSV not available", fontsize=LABEL_FONTSIZE)
    add_panel_label(axes[1], 1)
    fig.suptitle("Peak detection found rates (window_max)", fontsize=LABEL_FONTSIZE, y=1.02)
    save_figure(fig, OUT / "peak_found_rates")
    plt.close(fig)


def write_summary(pooled: pd.DataFrame) -> None:
    lines = [
        "# Appendix IM / peak detection",
        "",
        "Source: Box `peak_analysis/plots/stability` (campaign `window_max` peaks).",
        "",
        "## Found rates (pooled)",
        "",
        "| mode | found_rate |",
        "|------|------------|",
    ]
    for _, r in pooled.iterrows():
        lines.append(f"| {int(r['mode'])} | {float(r['found_rate']):.4f} |")
    lines += [
        "",
        "## Parameters (manifest)",
        "",
        "- `peak_method=window_max`, `window_policy=midpoint`",
        "- `window_frac_fixed=0.5`, `min_prominence=0.05`",
        "",
        "See also Box PNGs: `window_width_stability.png`, `modal_reliability.png`.",
        "",
    ]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print(f"Stability dir: {STABILITY}")
    pooled = _load_pooled()
    by_h = _load_by_height()
    plot_found_rate(pooled, by_h)
    write_summary(pooled)
    print(f"Wrote appendix figures → {OUT}")


if __name__ == "__main__":
    main()
