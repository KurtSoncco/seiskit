"""Sobol' global sensitivity analysis on the trained NGBoost χ emulators.

SALib's ``analyze.sobol`` needs a deterministic, 1-D array of model outputs.
NGBoost predicts a full Normal distribution per input, so each fitted
``ngboost_<metric>.pkl`` is treated as *two* deterministic emulators: the
predictive mean :math:`\\hat\\mu(\\mathbf{x})` (first-moment / "where the
response sits") and the predictive std :math:`\\hat\\sigma(\\mathbf{x})`
(aleatoric spread / "how uncertain the response is"). Sobol' indices are
computed separately for each, for all five χ metrics.

Design space
------------
The physical factorial design only has 3 discrete levels per factor
(``FACTORS`` = Vs1, Height, CoV, rH, aHV), so a classical Sobol' sampler
is run over the *convex hull* ``[min, max]`` of each factor, using the
NGBoost model as a continuous surrogate/emulator across that hull. This is
a standard emulator-based GSA approach, but the indices should be read as
"sensitivity of the fitted emulator over the explored range", not as an
exact decomposition of the physical simulator's variance. Node position is
held fixed at the array center (``CENTER_NODE``), matching the
``chi_qbm``/``chi_ngboost`` center-node convention used elsewhere.

Two extensions beyond the fixed-node, 5-factor analysis are also run:

1. **Node as a 6th Sobol' factor** — instead of holding node position fixed
   at the array center, node is added to the sampled input space
   (``node ∈ [0, N_NODES-1]``) so its own S1/ST and its
   interactions with the physical factors can be read off directly.
2. **Quarter-wavelength check** — the classical 1-D natural-frequency
   estimate :math:`\\hat f_0 = V_{s1} / (4H)` is computed on the same
   Vs1/Height samples used for the fixed-node run, and correlated against
   each metric's predicted mean/std to see how much of the model's
   Vs1+Height sensitivity is captured by that single physical ratio.

Writes CSVs + bar figures under ``figure_dir("chi_ngboost", "sobol_analysis")``.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from SALib.analyze import sobol as sobol_analyze
from SALib.sample import sobol as sobol_sample

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CENTER_NODE,
    FACTORS,
    METRICS,
    add_design_columns,
    load_ratios,
    models_dir,
    out_dir,
)
from train_ngboost import predict_params  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import (  # noqa: E402
    FACTOR_COLORS,
    REF_COLOR,
    apply_full_paper_style,
    figsize,
    metric_label,
    save_figure,
)

warnings.filterwarnings("ignore")

apply_full_paper_style(auto_format=True, frame="open", grid=False)

N_SALTELLI = 8192  # base sample size; total evals = N * (2*D + 2)
SALTELLI_SEED = 0
NODE_MIN, NODE_MAX = 0, 100  # channel index range (N_NODES = 101)

FACTOR_LABELS = {
    "Vs1": r"$V_{s1}$",
    "Height": r"$H$",
    "CoV": "CoV",
    "rH": r"$r_h$",
    "aHV": r"$a_{hv}$",
    "node": "node",
}

RESPONSES = ("mu", "sigma")
RESPONSE_LABELS = {"mu": r"predictive mean $\hat\mu(\ln\chi)$", "sigma": r"predictive std $\hat\sigma(\ln\chi)$"}


def zscore_stats(df: pd.DataFrame) -> tuple[dict[str, tuple[float, float]], tuple[float, float]]:
    """(mean, std) per factor and for node, matching ``add_design_columns``."""
    stats = {}
    for c in FACTORS:
        stats[c] = (float(df[c].mean()), float(df[c].std(ddof=0)))
    node = df["node"].to_numpy(dtype=float)
    node_stats = (float(node.mean()), float(node.std(ddof=0)))
    return stats, node_stats


def build_problem(df: pd.DataFrame, *, include_node: bool = False) -> dict:
    names = list(FACTORS) + (["node"] if include_node else [])
    bounds = [[float(df[f].min()), float(df[f].max())] for f in FACTORS]
    if include_node:
        bounds.append([float(NODE_MIN), float(NODE_MAX)])
    return {"num_vars": len(names), "names": names, "bounds": bounds}


def to_features(
    X_real: np.ndarray,
    stats: dict[str, tuple[float, float]],
    node_stats: tuple[float, float],
    node: int | np.ndarray,
) -> np.ndarray:
    """Real-unit factor samples (+ optional per-row node) -> z-scored feature matrix.

    ``node`` is either a fixed scalar (center-node analysis) or a per-row
    array taken from the last column of a 6-variable Sobol' sample.
    """
    cols = []
    for j, f in enumerate(FACTORS):
        mu, sd = stats[f]
        cols.append((X_real[:, j] - mu) / sd if sd > 0 else np.zeros(len(X_real)))
    nmu, nsd = node_stats
    node_arr = np.broadcast_to(node, (len(X_real),)).astype(float)
    node_z = (node_arr - nmu) / nsd if nsd > 0 else np.zeros(len(X_real))
    cols.append(node_z)
    return np.column_stack(cols)


def indices_to_frame(problem: dict, si: dict) -> pd.DataFrame:
    names = problem["names"]
    rows = []
    for i, name in enumerate(names):
        rows.append(
            {
                "factor": name,
                "S1": si["S1"][i],
                "S1_conf": si["S1_conf"][i],
                "ST": si["ST"][i],
                "ST_conf": si["ST_conf"][i],
            }
        )
    tab = pd.DataFrame(rows).sort_values("ST", ascending=False).reset_index(drop=True)
    return tab


def s2_to_frame(problem: dict, si: dict) -> pd.DataFrame:
    names = problem["names"]
    n = len(names)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            rows.append(
                {
                    "factor_a": names[i],
                    "factor_b": names[j],
                    "S2": si["S2"][i, j],
                    "S2_conf": si["S2_conf"][i, j],
                }
            )
    tab = pd.DataFrame(rows).sort_values("S2", ascending=False, key=abs).reset_index(drop=True)
    return tab


def bar_color(factor: str) -> str:
    return FACTOR_COLORS.get(factor, REF_COLOR)


def plot_st_bars(all_s1: dict[str, pd.DataFrame], response: str, out: Path, *, tag: str, names: list[str]) -> None:
    fig, ax = plt.subplots(figsize=figsize(aspect=0.55))
    metrics = list(all_s1.keys())
    n_f = len(names)
    width = 0.8 / n_f
    x = np.arange(len(metrics))
    for k, f in enumerate(names):
        vals = [all_s1[m].set_index("factor").loc[f, "ST"] for m in metrics]
        errs = [all_s1[m].set_index("factor").loc[f, "ST_conf"] for m in metrics]
        ax.bar(
            x + (k - (n_f - 1) / 2) * width,
            vals,
            width=width,
            yerr=errs,
            color=bar_color(f),
            label=FACTOR_LABELS[f],
            capsize=1.5,
            linewidth=0.3,
            edgecolor="white",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([metric_label(m) for m in metrics])
    ax.set_ylabel(r"Total-order Sobol' index $S_T$")
    ax.axhline(0.0, color="black", lw=0.5)
    ax.legend(frameon=False, ncol=n_f, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    save_figure(fig, f"sobol_ST_{response}{tag}", out_dir=out)
    plt.close(fig)


def run_sobol(
    problem: dict,
    X_feat_fn,
    out: Path,
    *,
    tag: str,
    keep_y: dict | None = None,
) -> tuple[dict, dict]:
    """Sample + predict + analyze Sobol' indices for all metrics/responses.

    ``X_feat_fn(X_real) -> X_feat`` maps a Saltelli real-unit sample to the
    z-scored NGBoost feature matrix. ``tag`` suffixes output filenames
    (empty for the primary fixed-node run). If ``keep_y`` is given, the raw
    ``{response: {metric: y}}`` predictions are stashed there for reuse
    (e.g. the f0_calc correlation check).
    """
    n = problem["num_vars"]
    print(f"Saltelli sampling ({tag or 'fixed-node'}): N={N_SALTELLI} -> {N_SALTELLI * (2 * n + 2)} evals/metric")
    X_real = sobol_sample.sample(problem, N_SALTELLI, calc_second_order=True, seed=SALTELLI_SEED)
    X_feat = X_feat_fn(X_real)

    s1_tables: dict[str, dict[str, pd.DataFrame]] = {r: {} for r in RESPONSES}
    s2_tables: dict[str, dict[str, pd.DataFrame]] = {r: {} for r in RESPONSES}

    for metric in METRICS:
        print(f"=== Sobol {metric} {tag} ===")
        model: NGBRegressor = joblib.load(models_dir() / f"ngboost_{metric}.pkl")
        mu, sigma = predict_params(model, X_feat)
        for response, y in (("mu", mu), ("sigma", sigma)):
            if keep_y is not None:
                keep_y.setdefault(response, {})[metric] = (X_real, y)
            si = sobol_analyze.analyze(
                problem, y, calc_second_order=True, print_to_console=False, seed=SALTELLI_SEED
            )
            s1 = indices_to_frame(problem, si)
            s2 = s2_to_frame(problem, si)
            s1["metric"] = metric
            s2["metric"] = metric
            s1_tables[response][metric] = s1
            s2_tables[response][metric] = s2
            s1.to_csv(out / f"sobol_S1_ST_{response}{tag}_{metric}.csv", index=False)
            s2.to_csv(out / f"sobol_S2_{response}{tag}_{metric}.csv", index=False)

    for response in RESPONSES:
        plot_st_bars(s1_tables[response], response, out, tag=tag, names=problem["names"])

    combined_s1 = {r: pd.concat(s1_tables[r].values(), ignore_index=True) for r in RESPONSES}
    combined_s2 = {r: pd.concat(s2_tables[r].values(), ignore_index=True) for r in RESPONSES}
    for response in RESPONSES:
        combined_s1[response].to_csv(out / f"sobol_S1_ST_{response}{tag}_all.csv", index=False)
        combined_s2[response].to_csv(out / f"sobol_S2_{response}{tag}_all.csv", index=False)
    return combined_s1, combined_s2


def f0_calc_diagnostic(fixed_node_y: dict) -> pd.DataFrame:
    """Correlate the quarter-wavelength estimate f0_calc = Vs1/(4H) against
    each metric's predicted mean/std, using the fixed-node Saltelli sample.

    Also reports S1(Vs1) + S1(Height) from that same run for comparison:
    if f0_calc alone explains close to that combined marginal share, the
    two design factors act on the response mainly through this one ratio.
    """
    rows = []
    vs1_idx, height_idx = FACTORS.index("Vs1"), FACTORS.index("Height")
    for response in RESPONSES:
        for metric in METRICS:
            X_real, y = fixed_node_y[response][metric]
            vs1, height = X_real[:, vs1_idx], X_real[:, height_idx]
            f0_calc = vs1 / (4.0 * height)
            ln_f0 = np.log(f0_calc)
            r = float(np.corrcoef(ln_f0, y)[0, 1])
            rows.append(
                {
                    "response": response,
                    "metric": metric,
                    "corr_ln_f0_calc": r,
                    "r2_ln_f0_calc": r**2,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# summary.md helpers
# ---------------------------------------------------------------------------
def st_pivot(combined_s1: dict, response: str, names: list[str]) -> pd.DataFrame:
    long = combined_s1[response]
    piv = long.pivot(index="factor", columns="metric", values="ST")
    piv = piv[list(METRICS)].loc[sorted(names, key=lambda f: -piv.loc[f].mean())]
    return piv


def rank_sentence(combined_s1: dict, response: str, names: list[str]) -> list[str]:
    piv = st_pivot(combined_s1, response, names)
    lines = []
    for metric in METRICS:
        order = piv[metric].sort_values(ascending=False)
        ranked = " > ".join(f"{f} ({order[f]:.3f})" for f in order.index)
        lines.append(f"- **{metric_label(metric)}**: {ranked}")
    return lines


def top_interactions(combined_s2: dict, response: str, n: int = 3, *, involving: str | None = None) -> pd.DataFrame:
    long = combined_s2[response].copy()
    if involving is not None:
        long = long[(long["factor_a"] == involving) | (long["factor_b"] == involving)]
    long["abs_S2"] = long["S2"].abs()
    return long.sort_values("abs_S2", ascending=False).head(n).drop(columns="abs_S2")


def main() -> None:
    out = out_dir("sobol_analysis")
    print("Loading ratios for z-score stats and factor bounds …")
    df = add_design_columns(load_ratios())
    stats, node_stats = zscore_stats(df)

    # -- (1) primary run: node fixed at array center -----------------------
    problem_fixed = build_problem(df, include_node=False)
    fixed_node_y: dict = {}
    combined_s1, combined_s2 = run_sobol(
        problem_fixed,
        lambda X_real: to_features(X_real, stats, node_stats, CENTER_NODE),
        out,
        tag="",
        keep_y=fixed_node_y,
    )

    # -- (2) extension: node itself as a 6th Sobol' factor ------------------
    problem_node = build_problem(df, include_node=True)
    combined_s1_n, combined_s2_n = run_sobol(
        problem_node,
        lambda X_real: to_features(X_real, stats, node_stats, X_real[:, -1]),
        out,
        tag="_with_node",
    )

    # -- (3) extension: quarter-wavelength check f0_calc = Vs1/(4H) --------
    f0_tab = f0_calc_diagnostic(fixed_node_y)
    f0_tab.to_csv(out / "f0_calc_correlation.csv", index=False)

    names_fixed = list(FACTORS)
    names_node = problem_node["names"]

    mu_piv = st_pivot(combined_s1, "mu", names_fixed)
    sigma_piv = st_pivot(combined_s1, "sigma", names_fixed)
    mu_piv_n = st_pivot(combined_s1_n, "mu", names_node)
    sigma_piv_n = st_pivot(combined_s1_n, "sigma", names_node)

    lines = [
        "# Sobol' global sensitivity analysis (NGBoost χ emulators)",
        "",
        "## Method",
        "",
        r"- Each trained `ngboost_<metric>.pkl` predicts a Normal distribution "
        r"\(Y=\ln\chi \mid \mathbf{x} \sim \mathcal N(\hat\mu(\mathbf{x}), \hat\sigma(\mathbf{x})^2)\). "
        r"Sobol' indices are computed on \(\hat\mu\) (where the response sits) and on "
        r"\(\hat\sigma\) (how uncertain/variable the response is) as two separate "
        "deterministic emulator outputs, for each of the five χ metrics.",
        r"- Inputs: the five design factors "
        f"({', '.join(FACTORS)}), sampled with Saltelli's scheme over the "
        r"**convex hull** \([\min,\max]\) of each factor's 3 discrete design levels "
        f"(N={N_SALTELLI}, {N_SALTELLI * (2 * len(FACTORS) + 2):,} evaluations/metric for the "
        "fixed-node run, first + total + second order).",
        f"- Primary run: node position held fixed at the array center (`node={CENTER_NODE}`), "
        "matching the `chi_qbm`/`chi_ngboost` center-node convention used elsewhere in the paper. "
        "A second run (below) adds node as a 6th Sobol' factor.",
        "- **Caveat**: the physical design is a 3-level factorial, not a continuous space. "
        "Treating the NGBoost model as a continuous surrogate across the factor range is "
        "a standard emulator-based GSA approach, but indices should be read as sensitivity "
        "of the *fitted emulator* over the explored range, not an exact variance "
        "decomposition of the physical simulator.",
        "",
        "## Output files",
        "",
        "| File | Content |",
        "|------|---------|",
        "| `sobol_S1_ST_<response>_<metric>.csv` | fixed-node: first-/total-order indices per factor |",
        "| `sobol_S2_<response>_<metric>.csv` | fixed-node: second-order (pairwise) indices |",
        "| `sobol_S1_ST_<response>_with_node_<metric>.csv` | node-as-factor: first-/total-order indices |",
        "| `sobol_S2_<response>_with_node_<metric>.csv` | node-as-factor: second-order indices |",
        "| `sobol_S1_ST_<response>[_with_node]_all.csv` / `sobol_S2_..._all.csv` | combined across metrics |",
        "| `sobol_ST_<response>[_with_node].pdf` | total-order \\(S_T\\) bar chart, factors × metrics |",
        "| `f0_calc_correlation.csv` | corr(\\(\\ln \\hat f_0\\), response) for \\(\\hat f_0=V_{s1}/(4H)\\) |",
        "",
        "## Total-order sensitivity \\(S_T\\) — predictive mean \\(\\hat\\mu\\) (node fixed at center)",
        "",
        mu_piv.to_markdown(floatfmt=".3f"),
        "",
        "**Ranking (mean, most to least influential):**",
        "",
        *rank_sentence(combined_s1, "mu", names_fixed),
        "",
        "## Total-order sensitivity \\(S_T\\) — predictive std \\(\\hat\\sigma\\) (aleatoric spread, node fixed)",
        "",
        sigma_piv.to_markdown(floatfmt=".3f"),
        "",
        "**Ranking (spread, most to least influential):**",
        "",
        *rank_sentence(combined_s1, "sigma", names_fixed),
        "",
        "## Strongest pairwise interactions (|S2|, top 3 per response, node fixed)",
        "",
        "### Mean response",
        "",
        top_interactions(combined_s2, "mu").to_markdown(index=False, floatfmt=".3f"),
        "",
        "### Std response",
        "",
        top_interactions(combined_s2, "sigma").to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Node position as a Sobol' factor",
        "",
        r"Node is now sampled jointly with the five design factors over "
        f"`node ∈ [{NODE_MIN}, {NODE_MAX}]` "
        r"(N=" + f"{N_SALTELLI}, {N_SALTELLI * (2 * len(names_node) + 2):,} evaluations/metric).",
        "",
        "### Total-order sensitivity \\(S_T\\) — predictive mean \\(\\hat\\mu\\)",
        "",
        mu_piv_n.to_markdown(floatfmt=".3f"),
        "",
        *rank_sentence(combined_s1_n, "mu", names_node),
        "",
        "### Total-order sensitivity \\(S_T\\) — predictive std \\(\\hat\\sigma\\)",
        "",
        sigma_piv_n.to_markdown(floatfmt=".3f"),
        "",
        *rank_sentence(combined_s1_n, "sigma", names_node),
        "",
        "### Strongest pairwise interactions involving node (|S2|, top 3 per response)",
        "",
        "**Mean response**",
        "",
        top_interactions(combined_s2_n, "mu", involving="node").to_markdown(index=False, floatfmt=".3f"),
        "",
        "**Std response**",
        "",
        top_interactions(combined_s2_n, "sigma", involving="node").to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Quarter-wavelength estimator \\(\\hat f_0 = V_{s1}/(4H)\\)",
        "",
        r"Pearson correlation of \(\ln\hat f_0\) against each metric's predicted mean/std, "
        "computed on the same fixed-node Saltelli sample used above (no extra model evaluations).",
        "",
        f0_tab.pivot(index="metric", columns="response", values="r2_ln_f0_calc")
        .loc[list(METRICS)]
        .rename(columns={"mu": "R2 (mean)", "sigma": "R2 (std)"})
        .to_markdown(floatfmt=".3f"),
        "",
        "## Conclusions",
        "",
    ]

    # Derive a couple of automatic conclusion bullets from the data.
    mean_of_means = mu_piv.mean(axis=1).sort_values(ascending=False)
    top_factor = mean_of_means.index[0]
    lines.append(
        f"- Averaged across all five χ metrics, **{FACTOR_LABELS[top_factor]}** "
        f"({top_factor}) has the largest total-order effect on the predictive mean "
        f"(mean \\(S_T\\) = {mean_of_means.iloc[0]:.3f}), followed by "
        f"{', '.join(mean_of_means.index[1:])} in decreasing order."
    )
    s1_sum = combined_s1["mu"].groupby("metric").apply(lambda d: d["S1"].sum(), include_groups=False)
    st_sum = combined_s1["mu"].groupby("metric").apply(lambda d: d["ST"].sum(), include_groups=False)
    interaction_gap = (st_sum - s1_sum).sort_values(ascending=False)
    lines.append(
        f"- Sum(\\(S_T\\)) exceeds Sum(\\(S_1\\)) most for **{metric_label(interaction_gap.index[0])}** "
        f"(gap = {interaction_gap.iloc[0]:.3f}), indicating factor interactions/higher-order "
        "effects matter most for that metric; where the gap is small, the five factors act "
        "close to additively."
    )
    sigma_mean_of_means = sigma_piv.mean(axis=1).sort_values(ascending=False)
    lines.append(
        f"- For the predictive **spread** \\(\\hat\\sigma\\), "
        f"**{FACTOR_LABELS[sigma_mean_of_means.index[0]]}** ({sigma_mean_of_means.index[0]}) "
        f"dominates on average (mean \\(S_T\\) = {sigma_mean_of_means.iloc[0]:.3f}); comparing this "
        "to the mean-response ranking above shows whether the same factor that shifts χ also "
        "governs how uncertain that shift is."
    )
    node_mean_mu = mu_piv_n.mean(axis=1)
    node_st_mu = float(node_mean_mu.get("node", float("nan")))
    node_rank_mu = int(node_mean_mu.rank(ascending=False)["node"]) if "node" in node_mean_mu else None
    lines.append(
        f"- Adding node position as a 6th factor gives it mean \\(S_T\\) = {node_st_mu:.3f} on the "
        f"predictive mean (rank {node_rank_mu} of {len(names_node)}); this quantifies how much the "
        "along-array position matters relative to the physical design factors, and the "
        "node-paired |S2| interactions above show which factor(s) node's effect is entangled with "
        "(e.g. a mode shape / quarter-wavelength pattern would show up as node × Height or node × Vs1)."
    )
    mean_r2 = f0_tab.groupby("response")["r2_ln_f0_calc"].mean()
    lines.append(
        f"- The single quarter-wavelength ratio \\(\\hat f_0=V_{{s1}}/(4H)\\) alone reaches "
        f"mean \\(R^2\\) = {mean_r2.get('mu', float('nan')):.3f} against the predictive mean and "
        f"{mean_r2.get('sigma', float('nan')):.3f} against the predictive std, averaged over the five "
        "metrics — compare per-metric values above to the individual Vs1/Height \\(S_T\\) rows: a high "
        "\\(R^2\\) here means Vs1 and Height act on that metric mainly *through* this one physical ratio "
        "rather than independently."
    )
    lines.append(
        "- These rankings are conditional on the emulator's extrapolation across each factor's "
        "observed range; see `../surfaces/` for the full (cell × node) predictive grid for "
        "spatial profiles rather than global sensitivity indices."
    )
    lines.append("")

    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
