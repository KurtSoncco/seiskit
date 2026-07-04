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

from seiskit.plot_config import apply_style, panel_letter, result_path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FACTORS, load_channel50

# Load data
d50 = load_channel50()

# Apply style
apply_style(auto_format=False, font_size=10, frame="open")

# Define lines color and markers to plot.
lines_color = ["#C44E52", "#4C72B0"]
markers = ["o", "s", "D"]
linestyles = ["-", "--", "-."]

# X axis labels
x_labels = {
    "Vs1": r"$V_{s1}$ (m/s)",
    "Height": r"$H$ (m)",
    "CoV": r"$\mathrm{CoV}$",
    "rH": r"$r_{h}$ (m)",
    "aHV": r"$a_{hv}$",
}

# Y axis labels
y_labels = {
    "f_ratio": r"$f_0^N$",
    "log_abs": r"$\log(|TF|_0^N)$",
}

# Y axis limits
y_limits = {
    "f_ratio": (0.80, 1.30),
    "log_abs": (-4, 1.0),
}

# Create figure and gridspec
fig = plt.figure(figsize=(13, 5))
gs = fig.add_gridspec(2, 5, hspace=0.1, wspace=0.10)

TARGETS_PLOT = [
    ("f_ratio", "$f$ ratio"),
    ("log_abs", r"$\log\left(\left|TF\right|^N_{0}\right)$"),
]
QUANTILES = [0.05, 0.50, 0.95]
Q_LABELS = ["p5", "p50", "p95"]
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
                ms=5,
                label=qlbl,
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

handles, labels = fig.axes[0].get_legend_handles_labels()
leg = fig.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.95),
    ncol=3,
    fontsize=8,
    frameon=False,
)

for line in leg.get_lines():
    line.set_linewidth(2.5)


fig.suptitle(
    "Center Recorder: Marginal Quantile Profiles (p5 / p50 / p95) by Factor",
    fontsize=10,
    fontweight="bold",
)
fig.savefig(result_path("plots", "quantile_eda.png"), dpi=150, bbox_inches="tight")
