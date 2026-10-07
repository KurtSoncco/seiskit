"""1D ALE on NGBoost dispersion / upper-tail targets, per partition (Fig18).

For each partition (between-seed, within-seed) and metric, two figures:

1. ``ale_scale_<metric>.pdf`` — conditional scale σ(x); between-seed or
   within-seed dispersion depending on the partition.
2. ``ale_q95_<metric>.pdf`` — upper quantile μ(x) + Φ^{−1}(0.95) σ(x).

σ and q95 are never overlaid (incommensurable). ALE is evaluated at the three
observed factor levels and drawn as markers.

Writes under ``figure_dir("chi_shap", "ale_dispersion", <partition>)``.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ale_effects import (  # noqa: E402
    ale_1d,
    ale_subsample,
    amplitude_table,
    load_ngb,
    plot_ale_levels,
)
from common import (  # noqa: E402
    METRICS,
    NGB_FEATURES,
    PARTITION_LABELS,
    PARTITIONS,
    factor_levels,
    load_partition,
    out_dir,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import metric_label  # noqa: E402

warnings.filterwarnings("ignore")

Z95 = float(norm.ppf(0.95))

# (target tag / figure stem, title quantity, y label)
TARGETS = (
    ("scale", r"conditional scale $\sigma$", r"ALE on $\sigma$"),
    ("q95", r"upper quantile $\mu+z_{0.95}\sigma$", r"ALE on $q_{0.95}$"),
)


def _predict_ngb_sigma(model: NGBRegressor, X: np.ndarray) -> np.ndarray:
    dist = model.pred_dist(np.asarray(X, dtype=float))
    return np.maximum(np.asarray(dist.scale, dtype=float).ravel(), 1e-8)


def _predict_ngb_q95(model: NGBRegressor, X: np.ndarray) -> np.ndarray:
    dist = model.pred_dist(np.asarray(X, dtype=float))
    mu = np.asarray(dist.loc, dtype=float).ravel()
    sig = np.maximum(np.asarray(dist.scale, dtype=float).ravel(), 1e-8)
    return mu + Z95 * sig


def run_partition(partition: str) -> None:
    out = out_dir("ale_dispersion", partition)
    print(f"Loading {PARTITION_LABELS[partition]} …")
    df = load_partition(partition)
    raw = {f: factor_levels(df)[f][1] for f in NGB_FEATURES}
    X = ale_subsample(df, partition)

    rows = []
    for metric in METRICS:
        model = load_ngb(partition, metric)
        predictors = {
            "scale": lambda Xq, m=model: _predict_ngb_sigma(m, Xq),
            "q95": lambda Xq, m=model: _predict_ngb_q95(m, Xq),
        }
        print(f"ALE dispersion/tail [{partition}] {metric} …")
        for tag, title_qty, ylabel in TARGETS:
            curves = {}
            for j, feat in enumerate(NGB_FEATURES):
                lv, ale = ale_1d(predictors[tag], X, j)
                curves[feat] = ale
                for z, r, y in zip(lv, raw[feat], ale):
                    rows.append(
                        {
                            "metric": metric,
                            "target": tag,
                            "feature": feat,
                            "level_z": float(z),
                            "level": float(r),
                            "effect": float(y),
                        }
                    )
            plot_ale_levels(
                curves,
                raw,
                title=f"{metric_label(metric, log=True)} — {title_qty}, {PARTITION_LABELS[partition]}",
                ylabel=ylabel,
                stem=f"ale_{tag}_{metric}",
                out=out,
            )

    tab = pd.DataFrame(rows)
    tab.to_csv(out / "ale_dispersion_curves.csv", index=False)
    amp = amplitude_table(tab, ["metric", "target", "feature"])
    amp.to_csv(out / "ale_dispersion_effect_range.csv", index=False)
    meta = {"partition": partition, "subsample_n": int(len(X)), "z95": Z95}
    (out / "ale_dispersion_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    sigma_meaning = (
        "between-seed dispersion at the center node"
        if partition == "between"
        else "within-seed dispersion around the cell mean"
    )
    lines = [
        f"# ALE on NGBoost dispersion and upper tail — {PARTITION_LABELS[partition]} (Fig18)",
        "",
        rf"1. **Scale** (`ale_scale_<metric>.pdf`): NGBoost $\sigma$ ({sigma_meaning}).",
        r"2. **Upper quantile** (`ale_q95_<metric>.pdf`): NGBoost $\mu+z_{0.95}\sigma$.",
        "",
        "σ and q95 are never overlaid. ALE at the three observed factor levels, markers only.",
        "",
        "## Effect amplitude",
        "",
        amp.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Output files",
        "",
        "| File | Content |",
        "|------|---------|",
        "| `ale_scale_<metric>.pdf` | ALE on σ |",
        "| `ale_q95_<metric>.pdf` | ALE on $q_{0.95}$ |",
        "| `ale_dispersion_curves.csv` | level points |",
        "| `ale_dispersion_effect_range.csv` | amplitudes |",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


def main() -> None:
    for partition in PARTITIONS:
        run_partition(partition)


if __name__ == "__main__":
    main()
