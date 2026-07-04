"""SHAP summary bee-swarm plot for f_ratio — mean LightGBM model."""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

import shap

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FACTORS, cached_shap, load_channel50, load_mean_models, seed_grouped_split

from seiskit.plot_config import apply_style, result_path

apply_style(auto_format=True, font_size=10, frame="open")

d50 = load_channel50()

Xdf = d50[FACTORS]
tr, te = seed_grouped_split(d50)
Xte_df = Xdf.iloc[te]

mean_models = load_mean_models()

shap_vals = {}
shap_vals["f_ratio"] = cached_shap(
    "shap_f_ratio_ch50_te",
    lambda: shap.TreeExplainer(mean_models["f_ratio"]).shap_values(Xte_df),
)

nice = {"log_abs": "log(abs_TF_ratio)", "f_ratio": "f_ratio"}

tgt = "f_ratio"
plt.figure()
shap.summary_plot(
    shap_vals[tgt],
    Xte_df,
    show=False,
    plot_size=(6.4, 3.8),
    color_bar=True,
    sort=True,
)
fig = plt.gcf()
fig.suptitle(f"SHAP value distribution — {nice[tgt]}", fontsize=11, y=1.02)
ax = fig.axes[0]
ax.set_xlabel(f"SHAP value (impact on {nice[tgt]} prediction)")
fig.savefig(
    result_path("plots", "shap_summary_f_ratio.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.close(fig)
print("saved shap_summary_f_ratio.png")
