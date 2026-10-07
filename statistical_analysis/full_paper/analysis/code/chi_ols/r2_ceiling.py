"""Reliability (signal-to-total) R² ceiling for χ ratios.

Ceilings are reported per variance partition so between-seed and within-seed
noise are never pooled:

- ``between`` — center node × all seeds (replicates = seeds; noise = between-seed).
- ``within``  — first seed × all nodes (replicates = nodes; noise = within-seed).
- ``full``    — every node×seed as a replicate (legacy pooled scope, kept for
  the heatmap annotations and the pooled Stage-1 efficiency; not a within-seed
  noise fraction).

Writes CSV + summary.md under ``figure_dir("chi_ols", "r2_ceiling")``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    FACTORS,
    METRICS,
    N_CELLS,
    N_NODES,
    N_SEEDS,
    add_design_columns,
    fmt,
    load_ratios,
    log_response,
    out_dir,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared import CENTER_NODE, SAMPLE_LABELS, first_seed  # noqa: E402


def reliability_ceiling(
    df: pd.DataFrame,
    y: np.ndarray,
    design_cols: list[str],
) -> dict[str, float | int]:
    """Signal-to-total ceiling for a single-draw Y given design cells."""
    work = df.loc[np.isfinite(y)].copy()
    work["y"] = y[np.isfinite(y)]
    g = work.groupby(design_cols, sort=False)["y"]
    n_i = g.count()
    if (n_i < 2).any():
        raise ValueError(f"every design point needs ≥2 replicates; min n={int(n_i.min())}")

    cell_mean = g.mean()
    cell_var = g.var(ddof=1)
    n_cells = int(len(cell_mean))
    n_bar = float(n_i.mean())
    n_harm = float(n_cells / (1.0 / n_i).sum())

    sigma2_signal = float(cell_mean.var(ddof=1))
    sigma2_noise = float(cell_var.mean())
    reliability = sigma2_signal / (sigma2_signal + sigma2_noise)

    sigma2_signal_bc = max(0.0, sigma2_signal - float((cell_var / n_i).mean()))
    reliability_bc = sigma2_signal_bc / (sigma2_signal_bc + sigma2_noise)

    yv = work["y"].to_numpy(dtype=float)
    grand = float(yv.mean())
    ss_total = float(((yv - grand) ** 2).sum())
    y_cell = g.transform("mean").to_numpy(dtype=float)
    ss_within = float(((yv - y_cell) ** 2).sum())
    ss_between = ss_total - ss_within
    reliability_ss = ss_between / ss_total if ss_total > 0 else np.nan

    return dict(
        n_obs=int(len(work)),
        n_cells=n_cells,
        n_bar=n_bar,
        n_harm=n_harm,
        sigma2_signal=sigma2_signal,
        sigma2_noise=sigma2_noise,
        sigma2_total=sigma2_signal + sigma2_noise,
        reliability_ceiling=reliability,
        sigma2_signal_bc=sigma2_signal_bc,
        reliability_ceiling_bc=reliability_bc,
        reliability_ceiling_ss=reliability_ss,
        frac_noise=1.0 - reliability,
    )


def load_stage1_r2(path: Path) -> dict[str, float]:
    if not path.is_file():
        return {}
    fit = pd.read_csv(path)
    return {str(r["metric"]): float(r["r2_insample"]) for _, r in fit.iterrows()}


def build_summary_md(ceil: pd.DataFrame) -> str:
    lines = [
        "# Reliability R² ceiling",
        "",
        "Population-information bound on how much of the variance of a single "
        "draw \\(Y = \\ln\\chi\\) is attributable to design-cell means "
        f"(`{'`, `'.join(METRICS)}`).",
        "",
        f"- Design cells: **{N_CELLS}** (`{'`, `'.join(FACTORS)}`)",
        rf"- **between**: {SAMPLE_LABELS['center_node_all_seeds']} — noise = between-seed variance",
        rf"- **within**: {SAMPLE_LABELS['one_seed_all_nodes']} — noise = within-seed (spatial) variance",
        rf"- **full** (legacy): \(N_x = {N_NODES}\) × \(N_s = {N_SEEDS}\) pooled replicates; "
        "its noise mixes both partitions and is not \\(\\overline{s^2_W}/\\sigma^2_{total}\\)",
        r"- Efficiency uses the pooled Stage-1 in-sample \(R^2\) "
        "(`chi_ols/stage1_mean_ols/mean_fit_metrics.csv`) and is reported for **full** only",
        "",
        "## Output files",
        "",
        "| File | Contents |",
        "|------|----------|",
        "| `reliability_ceiling.csv` | Signal/noise variance, ceilings "
        r"(`_bc`, `_ss`), Stage-1 \(R^2\), efficiency |",
        "| `summary.md` | this file |",
        "",
        "## Notation",
        "",
        "For design cell \\(k\\) and replicate \\(\\ell\\) (seed, node, or node×seed by scope):",
        "",
        r"$$",
        r"Y_{k\ell} = \mu_k + \varepsilon_{k\ell}.",
        r"$$",
        "",
        r"$$",
        r"R^2_{\mathrm{ceiling}}"
        r" = \frac{\widehat{\mathrm{Var}}(\bar Y_k)}"
        r"{\widehat{\mathrm{Var}}(\bar Y_k) + \overline{s^2_k}},",
        r"$$",
        "",
        "with \\(\\widehat{\\mathrm{Var}}(\\bar Y_k)\\) = sample variance of "
        "cell means and \\(\\overline{s^2_k}\\) = average within-cell sample "
        "variance. Noise-corrected (`_bc`) subtracts "
        "\\(\\mathrm{mean}(s^2_k / n_k)\\) from the between-cell variance; "
        "`_ss` is \\(SS_{\\mathrm{between}} / SS_{\\mathrm{total}}\\). "
        "Efficiency:",
        "",
        r"$$",
        r"\mathrm{efficiency}"
        r" = \frac{R^2_{\mathrm{Stage\,1}}}{R^2_{\mathrm{ceiling}}}.",
        r"$$",
        "",
        r"Interpreting the ceiling as an upper bound on single-draw \(R^2\) "
        "requires seed/node noise to be irreducible and approximately "
        "additive (independent of design).",
        "",
        "## Results",
        "",
        r"| Metric | Scope | Ceiling | Ceiling (bc) | Ceiling (SS) | Stage-1 \(R^2\) | Efficiency |",
        "|--------|-------|--------:|-------------:|-------------:|---------------:|-----------:|",
    ]
    for _, r in ceil.iterrows():
        lines.append(
            f"| {r['metric']} | {r['scope']} | "
            f"{fmt(r['reliability_ceiling'])} | "
            f"{fmt(r['reliability_ceiling_bc'])} | "
            f"{fmt(r['reliability_ceiling_ss'])} | "
            f"{fmt(r['r2_stage1'])} | {fmt(r['efficiency'])} |"
        )

    lines.extend(["", "## Conclusions", ""])
    for metric in METRICS:
        by = ceil[ceil["metric"] == metric].set_index("scope")
        if not {"between", "within"}.issubset(by.index):
            continue
        b, w = by.loc["between"], by.loc["within"]
        lines.append(
            f"- **{metric}**: between-seed ceiling {fmt(b['reliability_ceiling'])} "
            f"(noise fraction {fmt(b['frac_noise'])}); within-seed ceiling "
            f"{fmt(w['reliability_ceiling'])} (noise fraction {fmt(w['frac_noise'])})."
        )
    lines.extend(
        [
            "",
            r"Report efficiency alongside absolute \(R^2\): a modest Stage-1 "
            r"\(R^2\) can still capture most of the **explainable design signal** "
            "when the ceiling is low.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    print("Loading join_master.h5 …")
    df = add_design_columns(load_ratios())
    stage1_path = out_dir("stage1_mean_ols") / "mean_fit_metrics.csv"
    stage1_r2 = load_stage1_r2(stage1_path)
    if stage1_r2:
        print(f"  Loaded Stage-1 R² from {stage1_path}")
    else:
        print("  Stage-1 metrics not found; efficiency will be NaN")

    rows: list[dict] = []
    for metric in METRICS:
        y = log_response(df, metric)
        scopes = {
            "between": (df["node"] == CENTER_NODE).to_numpy(),
            "within": (df["seed"] == first_seed(df)).to_numpy(),
            "full": np.ones(len(df), dtype=bool),
        }
        for scope in ("between", "within", "full"):
            mask = scopes[scope]
            sub = df.loc[mask].reset_index(drop=True)
            ys = y[mask]
            print(f"Ceiling {metric} / {scope} …")
            res = reliability_ceiling(sub, ys, list(FACTORS))
            ceil = float(res["reliability_ceiling"])
            r2_s1 = stage1_r2.get(metric, np.nan) if scope == "full" else np.nan
            eff = (r2_s1 / ceil) if np.isfinite(r2_s1) and ceil > 0 else np.nan
            rows.append(
                dict(
                    metric=metric,
                    scope=scope,
                    r2_stage1=r2_s1,
                    efficiency=eff,
                    **res,
                )
            )

    out = pd.DataFrame(rows)
    dest = out_dir("r2_ceiling")
    out.to_csv(dest / "reliability_ceiling.csv", index=False)
    (dest / "summary.md").write_text(build_summary_md(out), encoding="utf-8")
    print(f"Wrote {dest}")


if __name__ == "__main__":
    main()
