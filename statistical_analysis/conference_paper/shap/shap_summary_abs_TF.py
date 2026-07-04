"""SHAP summary bee-swarm plot for log(abs_TF_ratio) — mean LightGBM model."""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import shap

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import load_channel50, FACTORS, load_mean_models, seed_grouped_split, cached_shap

from seiskit.plot_config import apply_style, panel_letter, result_path

apply_style(auto_format=True, font_size=10, frame="open")

d50 = load_channel50()

Xdf = d50[FACTORS]
tr, te = seed_grouped_split(d50)
Xte_df = Xdf.iloc[te]

mean_model_log_abs = load_mean_models()["log_abs"]

shap_vals_log_abs = cached_shap(
    "shap_log_abs_ch50_te",
    lambda: shap.TreeExplainer(mean_model_log_abs).shap_values(Xte_df),
)

nice = {"log_abs": "log(abs_TF_ratio)", "f_ratio": "f_ratio"}

tgt = "log_abs"
plt.figure()
shap.summary_plot(
    shap_vals_log_abs,
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
    result_path("plots", "shap_summary_abs_TF.png"),
    dpi=150,
    bbox_inches="tight",
)
plt.close(fig)
print("saved shap_summary_abs_TF.png")
