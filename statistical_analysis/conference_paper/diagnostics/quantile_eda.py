"""
EDA Quantile Plots (specially p5, p50, p95) for channel 50 (center recorder) — seiskit conference paper diagnostics.

Figure of 2x5 panel covering quantile plots for p5, p50, and p95 for each factor.

Each column represent a factor, and each row represent f_ratio anf log_abs_TF_ratio.
In each panel, three lines represent p5, p50, and p95, by changing the values of the factor in x axis.

"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image

from seiskit.plot_config import apply_style, panel_letter, result_path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FACTORS, FIG_WIDTH, load_channel50, target_color, target_label

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


# Load data
d50 = load_channel50()

# Apply style
apply_style(auto_format=True, font_size=10, frame="open")

# Row colors: frequency vs natural-log amplitude
lines_color = [target_color("f_ratio"), target_color("log_abs")]
markers = ["o", "s", "D"]
linestyles = ["-", "--", "-."]
LINE_LW = 1.4
MARKER_MS = 5

# X axis labels
x_labels = {
    "Vs1": "Vs1",
    "Height": "Height",
    "CoV": "CoV",
    "rH": "rH",
    "aHV": "aHV",
}

# Y axis labels
y_labels = {
    "f_ratio": target_label("f_ratio"),
    "log_abs": target_label("log_abs"),
}

# Y axis limits
y_limits = {
    "f_ratio": (0.80, 1.40),
    "log_abs": (-4, 2.0),
}

# Create figure and gridspec
fig = plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH * 0.5))
gs = fig.add_gridspec(2, 5, hspace=0.1, wspace=0.20)

TARGETS_PLOT = [
    ("f_ratio", target_label("f_ratio")),
    ("log_abs", target_label("log_abs")),
]
QUANTILES = [0.05, 0.50, 0.95]
Q_LABELS = ["p5", "p50", "p95"]
Q_LABELS_SHORT = [r"$p_5$", r"$p_{50}$", r"$p_{95}$"]
letters = "abcdefghij"

for row, (tgt, ylabel) in enumerate(TARGETS_PLOT):
    for col, fc in enumerate(FACTORS):
        ax = fig.add_subplot(gs[row, col])

        grouped = d50.groupby(fc)[tgt]
        x_levels = np.array(sorted(d50[fc].unique()))

        q_vals = {
            qlbl: grouped.quantile(q).reindex(x_levels).values
            for q, qlbl in zip(QUANTILES, Q_LABELS)
        }

        ax.fill_between(
            x_levels,
            q_vals["p5"],
            q_vals["p95"],
            color=lines_color[row],
            alpha=0.15,
        )
        for k, qlbl in enumerate(Q_LABELS):
            ax.plot(
                x_levels,
                q_vals[qlbl],
                color=lines_color[row],
                marker=markers[k],
                linestyle=linestyles[k],
                ms=MARKER_MS,
                lw=LINE_LW,
            )

        # Set limits on all axes
        ax.set_ylim(y_limits[tgt])
        ax.set_xticks(x_levels)

        if row == 1:
            # Only show x label on the last row
            ax.set_xlabel(x_labels[fc])
        else:
            ax.tick_params(labelbottom=False)
        if col == 0:
            ax.set_ylabel(y_labels[tgt])
        else:
            ax.tick_params(labelleft=False)

        panel_letter(ax, letters[row * 5 + col])

# Neutral black proxies: same marker/linestyle/size as the panels (shared across both metrics)
legend_handles = [
    Line2D(
        [0],
        [0],
        color="black",
        marker=markers[k],
        linestyle=linestyles[k],
        markersize=MARKER_MS,
        linewidth=LINE_LW,
    )
    for k in range(len(Q_LABELS_SHORT))
]
fig.legend(
    legend_handles,
    Q_LABELS_SHORT,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.0),
    ncol=3,
    fontsize=10,
    frameon=False,
    handlelength=3.5,
    handletextpad=0.6,
    markerscale=1.0,
)

save_tight_exact_width(fig, result_path("plots", "quantile_eda.png"), fig_width=FIG_WIDTH)
