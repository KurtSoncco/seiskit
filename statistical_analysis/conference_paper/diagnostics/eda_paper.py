"""EDA for channel 50 (center recorder) — seiskit conference paper diagnostics.

Generates a 1x3 panel figure covering distributions in panels a-c.

"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from seiskit.plot_config import apply_style, panel_letter, result_path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_channel50

# Load data
d50 = load_channel50()

# Apply style
apply_style(auto_format=True, font_size=10, frame="open")

# Create figure and gridspec
fig = plt.figure(figsize=(13, 5))
gs = fig.add_gridspec(1, 3, hspace=0.2, wspace=0.20)

# Plot f_ratio
ax = fig.add_subplot(gs[0, 0])
ax.hist(d50["f_ratio"], bins=60, color="#4C72B0", edgecolor="none")
ax.axvline(1.0, color="0.3", ls="--", lw=1)
ax.set_xlabel("$f$ ratio")
ax.set_ylabel("count")
ax.set_title("$f$ ratio — near 1, right tail", loc="left")
ax.set_xlim(0.50, 1.75)
ax.set_ylim(0, 4000)
panel_letter(ax, "a")

# Plot abs_TF_ratio
ax = fig.add_subplot(gs[0, 1])
ax.hist(d50["abs_TF_ratio"], bins=80, color="#C44E52", edgecolor="none")
ax.axvline(1.0, color="0.3", ls="--", lw=1)
ax.set_xlabel("abs TF ratio")
# ax.set_ylabel("count"), No required, since it is the same as the y-axis for the other panels
ax.set_title("abs TF ratio — strong right skew", loc="left")
ax.set_xlim(0, 2.5)
ax.set_ylim(0, 7000)
panel_letter(ax, "b")

# Plot log_abs_TF_ratio
ax = fig.add_subplot(gs[0, 2])
ax.hist(np.log10(d50["abs_TF_ratio"]), bins=70, color="#C44E52", edgecolor="none", alpha=0.85)
ax.set_xlabel(r"$\log\left(\left|TF\right|^N_{0}\right)$")
# ax.set_ylabel("count"), No required, since it is the same as the y-axis for the other panels
ax.set_title("log abs TF ratio — near symmetric", loc="left")
ax.set_xlim(-2, 1)
ax.set_ylim(0, 1200)
panel_letter(ax, "c")

# Add title
fig.suptitle(
    "Center Recorder: distributions · (243 cells x 100 seeds)",
    fontsize=10,
)
fig.savefig(result_path("plots", "eda_paper.png"), dpi=150, bbox_inches="tight")
