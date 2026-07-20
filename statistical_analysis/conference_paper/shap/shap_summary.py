"""Mean-model SHAP bee-swarm summaries for center-recorder LightGBM models.

Primary amplitude view is natural-log ``log_abs``; frequency ratio is included
for comparison. Magnitude (|SHAP|) importance is separate from signed
directionality shown in the bee-swarm color encoding.

Usage
-----
    python shap_summary.py
    python shap_summary.py --target log_abs
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from seiskit.plot_config import apply_style, result_path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    FACTORS,
    FIG_DPI,
    PRIMARY_AMPLITUDE_TARGET,
    cached_shap,
    load_channel50,
    load_mean_models,
    seed_grouped_split,
    target_label,
)

DEFAULT_TARGETS = [PRIMARY_AMPLITUDE_TARGET, "f_ratio"]


def _parse_targets() -> list[str]:
    if "--target" in sys.argv:
        i = sys.argv.index("--target")
        return [sys.argv[i + 1]]
    return list(DEFAULT_TARGETS)


def main() -> None:
    apply_style(auto_format=True, font_size=10, frame="open")
    d50 = load_channel50()
    Xdf = d50[FACTORS]
    _, te = seed_grouped_split(d50)
    Xte_df = Xdf.iloc[te]

    targets = _parse_targets()
    mean_models = load_mean_models(targets=targets, split_by="seed")
    for tgt in targets:
        if tgt not in mean_models:
            print(f"skip {tgt}: mean model not found")
            continue
        model = mean_models[tgt]
        shap_vals = cached_shap(
            f"shap_{tgt}_ch50_te",
            lambda m=model: shap.TreeExplainer(m).shap_values(Xte_df),
        )
        plt.figure()
        shap.summary_plot(
            shap_vals,
            Xte_df,
            show=False,
            plot_size=(6.4, 3.8),
            color_bar=True,
            sort=True,
        )
        fig = plt.gcf()
        label = target_label(tgt)
        fig.suptitle(f"Mean SHAP — {label} (center recorder)", fontsize=11, y=1.02)
        ax = fig.axes[0]
        ax.set_xlabel(f"SHAP value (signed impact on {label})")
        # Keep legacy filenames for paper figure wiring
        fname = {
            "log_abs": "shap_summary_abs_TF.png",
            "f_ratio": "shap_summary_f_ratio.png",
        }.get(tgt, f"shap_summary_{tgt}.png")
        out = result_path("plots", fname)
        fig.savefig(out, dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out}")


if __name__ == "__main__":
    main()
