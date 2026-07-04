# env: python
"""Normality assessment via QQ plots — seiskit conference paper diagnostics.

Generates a 2x3 panel comparing raw vs log vs Box-Cox transformed QQ plots
for both target variables at channel 50.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy import stats
from scipy.stats import boxcox, boxcox_normmax

from seiskit.plot_config import apply_style, panel_letter, result_path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FACTORS, load_channel50

# Load data
d50 = load_channel50()

# Compute deviations
d50 = d50.copy()
d50["abs_dev"] = d50["abs_TF_ratio"] - d50.groupby(FACTORS)["abs_TF_ratio"].transform("mean")
d50["logabs_dev"] = np.log(d50["abs_TF_ratio"]) - d50.groupby(FACTORS)["abs_TF_ratio"].transform(
    lambda s: np.log(s).mean()
)
d50["f_dev"] = d50["f_ratio"] - d50.groupby(FACTORS)["f_ratio"].transform("mean")

# Compute Box-Cox transformations
## Box-Cox transformations are a way to transform non-normal data into a normal distribution.
## The lambda parameter is the optimal transformation parameter. To understand the lambda parameter, see:
lam_abs = boxcox_normmax(d50["abs_TF_ratio"].values)
lam_f = boxcox_normmax(d50["f_ratio"].values)


# Define Box-Cox transformation function
def bc(x, lam):
    """Box-Cox transformation function
    Args:
        x: numpy array of data
        lam: float lambda parameter
    Returns:
        numpy array of transformed data
    """
    return boxcox(x, lmbda=lam)


# Apply Box-Cox transformations
abs_bc = bc(d50["abs_TF_ratio"].values, lam_abs)
f_bc = bc(d50["f_ratio"].values, lam_f)

# Add transformed variables to data frame
d50["abs_bc"] = abs_bc
d50["f_bc"] = f_bc


# Apply style
apply_style(auto_format=True, font_size=10, frame="open")
fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))


# Define QQ plot function
## QQ plots are a way to visualize the distribution of a variable compared to a normal distribution.
def qq(ax, x, title, color, step=0.2):
    """QQ plot function
    Args:
        ax: matplotlib axis object
        x: numpy array of data
        title: string title of the plot
        color: string color of the plot
        step: float step size for the y-axis. Default is 0.2 for f_ratio and 0.5 for abs_TF_ratio.
    Returns:
        None
    """

    # Convert x to numpy array and remove infinite values
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    (osm, osr), (sl, ic, r) = stats.probplot(x, dist="norm")
    # Limit the number of points to 3000
    idx = np.linspace(0, len(osm) - 1, min(3000, len(osm))).astype(int)
    ax.plot(osm[idx], osr[idx], "o", ms=2, color=color, alpha=0.4)
    # Plot the line of best fit
    lim = [osm.min(), osm.max()]
    ax.plot(lim, sl * np.array(lim) + ic, "-", color="0.2", lw=1)
    # Set title
    ax.set_title(title, loc="left", fontsize=8.5)
    # Set x-axis and y-axis labels
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")

    # Limit on Theretical Quantiles
    ## Always from -4 to 4
    ax.set_xlim(-4, 4)
    # Limit on Sample Quantiles
    ## Always ending in multiples of 0.2 for f_ratio and 0.5 for abs_TF_ratio.
    ## For example, if the data spans from -4 to 1.2 for abs_TF _ratio, then y-axis would be
    ## set from -4.5 to 1.5.
    y_min = np.floor(osr.min() - step)
    y_max = np.ceil(osr.max() + step)
    ax.set_ylim(y_min, y_max)
    # Set y-axis to be multiples of step
    ax.yaxis.set_major_locator(ticker.MultipleLocator(step))
    # Add y-axis tick labels
    ax.set_yticks(np.arange(y_min, y_max + step, step))
    ax.set_yticklabels([f"{y:.1f}" for y in ax.get_yticks()])
    # Add skew and kurtosis text
    text = f"Skew = {stats.skew(x):.2f}\nKurt = {stats.kurtosis(x):.2f}"
    ax.text(
        0.85,
        0.92,
        text,
        transform=ax.transAxes,
        va="top",
        fontsize=6.5,
        bbox=dict(fc="white", ec="0.7", alpha=0.8, boxstyle="round,pad=0.3"),
    )


# Plot QQ plots
## Absolute TF ratio QQ plots
## Raw QQ plot
qq(axes[0, 0], d50["abs_TF_ratio"], "abs TF ratio — raw", "#C44E52", step=0.5)
## Log QQ plot
qq(axes[0, 1], np.log(d50["abs_TF_ratio"]), "abs TF ratio — log", "#C44E52", step=0.5)
## Box-Cox QQ plot
qq(
    axes[0, 2],
    d50["abs_TF_ratio"] ** 0 + bc(d50["abs_TF_ratio"].values, lam_abs),
    rf"abs TF ratio — Box-Cox $\lambda$ = {lam_abs:.2f}",
    "#C44E52",
    step=1.0,
)
## F ratio QQ plots
## Raw QQ plot
qq(axes[1, 0], d50["f_ratio"], "f ratio — raw", "#4C72B0", step=0.2)
## Log QQ plot
qq(axes[1, 1], np.log(d50["f_ratio"]), "f ratio — log", "#4C72B0", step=0.2)
## Box-Cox QQ plot
qq(
    axes[1, 2],
    bc(d50["f_ratio"].values, lam_f),
    rf"f ratio — Box-Cox $\lambda$ = {lam_f:.2f}",
    "#4C72B0",
    step=0.2,
)

# Add panel letters and title
for letter, ax in zip("abcdef", axes.ravel()):
    panel_letter(ax, letter)
fig.suptitle(
    "Normal QQ plots: Raw, Log, and Box-Cox transformed targets (Recorder 50) · Points on the line = Normal",
    fontsize=10,
)
fig.tight_layout()
# Save figure
fig.savefig(result_path("plots", "normality_assessment.png"), dpi=150, bbox_inches="tight")
