"""Descriptive, calibration and representativeness figures for the spread models.

Reads the spread tables (``load_spread``) and the ``train_spread.py`` outputs.
Writes under ``figure_dir("chi_ngboost", "spread", "figures")``:

- ``spread_by_factor_within.pdf`` / ``spread_by_factor_between.pdf`` — per
  replicate geometric-mean spread at each factor level (box), model
  \\(\\exp\\hat\\mu\\) (marker) and seed 1 / node 50 (red cross).
- ``sB_node_profile.pdf`` — \\(s_B\\) relative to its cell average along the array.
- ``spread_calibration.pdf`` — holdout PIT, central-interval coverage and the
  \\(p_0\\) reliability curve.
- ``representativeness.pdf`` — per-seed / per-node Y ceilings and the PIT of
  seed 1 / node 50 under the spread models.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CENTER_NODE,
    FACTORS,
    METRICS,
    N_NODES,
    N_SEEDS,
    SPREAD_KINDS,
    SPREAD_REPLICATE,
    add_design_columns,
    load_ratios,
    load_spread,
    log_response,
    spread_dir,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import (  # noqa: E402
    TOL_BRIGHT,
    add_panel_label,
    apply_full_paper_style,
    factor_color,
    figsize,
    metric_color,
    metric_label,
    save_figure,
)

warnings.filterwarnings("ignore")
apply_full_paper_style(auto_format=True, frame="open", grid=False)

FACTOR_DISPLAY = {
    "Vs1": r"$V_{s1}$",
    "Height": r"$H$",
    "CoV": "CoV",
    "rH": r"$r_h$",
    "aHV": r"$a_{hv}$",
}
SPREAD_SYMBOL = {"within": "s_W", "between": "s_B"}
REP_MARK = dict(marker="x", color=TOL_BRIGHT["red"], ms=5, mew=1.0, ls="none", zorder=5)
MODEL_MARK = dict(marker="D", color="0.15", ms=3.2, ls="none", zorder=4)
LOG_TICK_FMT = mticker.FuncFormatter(lambda v, _: f"{v:g}")
N_PIT_BINS = 10
COVERAGE_LEVELS = (0.5, 0.8, 0.9)


def fig_dir() -> Path:
    return spread_dir("figures")


def representative_id(kind: str) -> int:
    return 1 if kind == "within" else CENTER_NODE


def _fmt_level(v: float) -> str:
    return f"{int(v)}" if float(v).is_integer() else f"{v:g}"


def level_geomeans(df: pd.DataFrame, factor: str) -> pd.DataFrame:
    """Geometric-mean spread over the cells at each level, per replicate (nonzero rows)."""
    ok = df[np.isfinite(df["Z"])]
    return ok.groupby([factor, "replicate"])["Z"].mean().unstack(factor).apply(np.exp)


def plot_spread_by_factor(kind: str) -> None:
    cells = pd.read_csv(spread_dir(kind) / "cell_predictions.csv")
    rep_id = representative_id(kind)
    rep = SPREAD_REPLICATE[kind]
    fig, axes = plt.subplots(
        len(METRICS), len(FACTORS), figsize=figsize(height=6.4), sharey="row", squeeze=False
    )
    for i, metric in enumerate(METRICS):
        df = load_spread(kind, metric)
        cm = cells[cells["metric"] == metric]
        for j, factor in enumerate(FACTORS):
            ax = axes[i, j]
            gm = level_geomeans(df, factor)
            levels = gm.columns.to_numpy(dtype=float)
            pos = np.arange(len(levels))
            bp = ax.boxplot(
                [gm[c].dropna().to_numpy() for c in gm.columns],
                positions=pos,
                widths=0.5,
                whis=(5, 95),
                showfliers=False,
                patch_artist=True,
                medianprops=dict(color="0.2", lw=0.8),
                whiskerprops=dict(lw=0.6),
                capprops=dict(lw=0.6),
                boxprops=dict(lw=0.6),
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(factor_color(factor))
                patch.set_alpha(0.45)
            model = np.exp(cm.groupby(factor)["mu"].mean().reindex(levels).to_numpy())
            ax.plot(pos, model, label=r"Model $\exp\hat\mu$" if i == j == 0 else None, **MODEL_MARK)
            if rep_id in gm.index:
                ax.plot(
                    pos,
                    gm.loc[rep_id].to_numpy(),
                    label=f"{rep} {rep_id}" if i == j == 0 else None,
                    **REP_MARK,
                )
            ax.set_yscale("log")
            ax.set_xticks(pos, [_fmt_level(v) for v in levels])
            ax.set_xlim(-0.6, len(levels) - 0.4)
            if i == 0:
                ax.set_title(FACTOR_DISPLAY[factor])
            if j == 0:
                ax.set_ylabel(f"{metric_label(metric)}\n${SPREAD_SYMBOL[kind]}$")
            ax.yaxis.set_major_locator(mticker.LogLocator(base=10, subs=(1.0, 2.0, 3.0, 5.0)))
            ax.yaxis.set_major_formatter(LOG_TICK_FMT)
            ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)))
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    n_rep = N_SEEDS if kind == "within" else N_NODES
    fig.legend(loc="upper center", ncol=3, frameon=False, fontsize=6, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.97))
    fig.text(
        0.995,
        0.995,
        f"boxes: {n_rep} {rep}s (5–95%)",
        ha="right",
        va="top",
        fontsize=6,
        color="0.35",
    )
    save_figure(fig, f"spread_by_factor_{kind}", out_dir=fig_dir())
    plt.close(fig)


def plot_sB_node_profile() -> pd.DataFrame:
    rows = []
    fig, axes = plt.subplots(1, len(METRICS), figsize=figsize(height=1.9), sharey=True)
    for k, metric in enumerate(METRICS):
        df = load_spread("between", metric)
        rel = df["spread"] / df.groupby("cell")["spread"].transform("mean")
        q = rel.groupby(df["node"]).quantile([0.05, 0.5, 0.95]).unstack()
        ax = axes[k]
        ax.fill_between(q.index, q[0.05], q[0.95], color=metric_color(metric), alpha=0.3, lw=0)
        ax.plot(q.index, q[0.5], color=metric_color(metric), lw=0.9)
        ax.axhline(1.0, color="0.5", lw=0.5, ls="--")
        ax.axvline(CENTER_NODE, color=TOL_BRIGHT["red"], lw=0.6, ls=":")
        ax.set_title(metric_label(metric))
        ax.set_xlabel("Node")
        add_panel_label(ax, k)
        rows.append(
            {
                "metric": metric,
                "q05_min": float(q[0.05].min()),
                "q95_max": float(q[0.95].max()),
                "center_median": float(q.loc[CENTER_NODE, 0.5]),
            }
        )
    axes[0].set_ylabel(r"$s_B$ / cell mean of $s_B$")
    fig.tight_layout(pad=0.4)
    save_figure(fig, "sB_node_profile", out_dir=fig_dir())
    plt.close(fig)
    return pd.DataFrame(rows)


def _central_coverage(z, mu, sigma, level):
    q = stats.norm.ppf(0.5 + level / 2)
    return float(np.mean(np.abs(z - mu) <= q * np.maximum(sigma, 1e-8)))


def plot_calibration() -> pd.DataFrame:
    rows = []
    fig, axes = plt.subplots(3, len(METRICS), figsize=figsize(height=5.2), squeeze=False)
    preds = {k: pd.read_csv(spread_dir(k) / "test_predictions.csv") for k in SPREAD_KINDS}
    for r, kind in enumerate(SPREAD_KINDS):
        p = preds[kind]
        for k, metric in enumerate(METRICS):
            d = p[p["metric"] == metric]
            ax = axes[r, k]
            ax.hist(
                d["pit"],
                bins=N_PIT_BINS,
                range=(0, 1),
                density=True,
                color=metric_color(metric),
                edgecolor="white",
                lw=0.4,
            )
            ax.axhline(1.0, color="0.35", lw=0.6, ls="--")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
            ax.set_title(f"{metric_label(metric)}  ${SPREAD_SYMBOL[kind]}$", fontsize=7)
            if k == 0:
                ax.set_ylabel(
                    f"{'Held-out seeds' if kind == 'within' else 'Held-out node blocks'}\nPIT density"
                )
            ax.set_xlabel("PIT")
            nz = ~d["zero"].to_numpy(dtype=bool)
            for lvl in COVERAGE_LEVELS:
                rows.append(
                    {
                        "kind": kind,
                        "metric": metric,
                        "level": lvl,
                        "coverage": _central_coverage(
                            d["Z"].to_numpy()[nz],
                            d["mu"].to_numpy()[nz],
                            d["sigma"].to_numpy()[nz],
                            lvl,
                        ),
                    }
                )
    cov = pd.DataFrame(rows)
    for r, kind in enumerate(SPREAD_KINDS):
        ax = axes[2, r]
        ax.plot([0.4, 1.0], [0.4, 1.0], color="0.5", lw=0.6, ls="--")
        for metric in METRICS:
            c = cov[(cov["kind"] == kind) & (cov["metric"] == metric)]
            ax.plot(
                c["level"],
                c["coverage"],
                marker="o",
                ms=3,
                ls="none",
                color=metric_color(metric),
                label=metric_label(metric),
            )
        ax.set_xlim(0.4, 1.0)
        ax.set_ylim(0.4, 1.0)
        ax.set_xlabel("Nominal coverage")
        ax.set_ylabel("Empirical coverage")
        ax.set_title(f"Central intervals, ${SPREAD_SYMBOL[kind]}$", fontsize=7)
    axes[2, 0].legend(fontsize=5.5, frameon=False, loc="upper left")

    ax = axes[2, 2]
    d = preds["within"]
    d = d[d["metric"] == "f_ratio"]
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(d["p0"], bins) - 1, 0, len(bins) - 2)
    rel = d.groupby(idx).agg(p=("p0", "mean"), obs=("zero", "mean"), n=("zero", "size"))
    rel = rel[rel["n"] >= 20]
    ax.plot([0, 1], [0, 1], color="0.5", lw=0.6, ls="--")
    ax.plot(rel["p"], rel["obs"], marker="o", ms=3, ls="none", color=metric_color("f_ratio"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"Predicted $p_0$")
    ax.set_ylabel("Observed zero fraction")
    ax.set_title(r"$f_0$ zero-spread reliability", fontsize=7)
    for ax in axes[2, 3:]:
        ax.set_visible(False)
    for n, ax in enumerate([a for a in axes.ravel() if a.get_visible()]):
        add_panel_label(ax, n, fontsize=7)
    fig.tight_layout(pad=0.4)
    save_figure(fig, "spread_calibration", out_dir=fig_dir())
    plt.close(fig)
    return cov


def replicate_ceilings() -> pd.DataFrame:
    """Y ceiling per seed (nodes as noise) and per node (seeds as noise), per metric."""
    cache = fig_dir() / "replicate_ceilings.csv"
    if cache.is_file():
        return pd.read_csv(cache)
    print("Loading join_master for replicate ceilings …")
    df = add_design_columns(load_ratios(), include_node_z=False)
    rows = []
    for metric in METRICS:
        y = log_response(df, metric)
        work = pd.DataFrame({"seed": df["seed"], "node": df["node"], "cell": df["cell"], "y": y})
        work = work[np.isfinite(work["y"])]
        for kind, rep in (("within", "seed"), ("between", "node")):
            g = work.groupby([rep, "cell"])["y"].agg(["mean", "var"])
            per = g.groupby(level=0).agg(signal=("mean", "var"), noise=("var", "mean"))
            ceil = per["signal"] / (per["signal"] + per["noise"])
            rows.extend(
                {"kind": kind, "metric": metric, "replicate": int(r), "ceiling": float(c)}
                for r, c in ceil.items()
            )
    out = pd.DataFrame(rows)
    out.to_csv(cache, index=False)
    return out


def plot_representativeness() -> pd.DataFrame:
    ceil = replicate_ceilings()
    fig, axes = plt.subplots(2, 2, figsize=figsize(height=4.4))
    titles = {
        "within": rf"Within-seed $Y$ ceiling per seed ({N_SEEDS} seeds)",
        "between": rf"Between-seed $Y$ ceiling per node ({N_NODES} nodes)",
    }
    rng = np.random.default_rng(0)
    stats_rows = []
    for c, kind in enumerate(SPREAD_KINDS):
        ax = axes[0, c]
        rep_id = representative_id(kind)
        for k, metric in enumerate(METRICS):
            v = ceil[(ceil["kind"] == kind) & (ceil["metric"] == metric)]
            x = k + rng.uniform(-0.18, 0.18, len(v))
            ax.plot(x, v["ceiling"], "o", ms=1.6, color=metric_color(metric), alpha=0.6, mew=0)
            rv = v.loc[v["replicate"] == rep_id, "ceiling"]
            ax.plot(
                [k], rv, label=f"{SPREAD_REPLICATE[kind]} {rep_id}" if k == 0 else None, **REP_MARK
            )
            stats_rows.append(
                {
                    "kind": kind,
                    "metric": metric,
                    "q05": float(v["ceiling"].quantile(0.05)),
                    "median": float(v["ceiling"].median()),
                    "q95": float(v["ceiling"].quantile(0.95)),
                    "representative": float(rv.iloc[0]),
                    "representative_pct": float(stats.percentileofscore(v["ceiling"], rv.iloc[0])),
                }
            )
        ax.set_xticks(range(len(METRICS)), [metric_label(m) for m in METRICS])
        ax.set_ylim(0, 1)
        ax.set_ylabel(r"Ceiling of $Y$")
        ax.set_title(titles[kind], fontsize=7)
        ax.legend(fontsize=6, frameon=False, loc="lower right")

        ax = axes[1, c]
        pit = pd.read_csv(spread_dir(kind) / "representative_pit.csv")
        for metric in METRICS:
            ax.hist(
                pit.loc[pit["metric"] == metric, "pit"],
                bins=N_PIT_BINS,
                range=(0, 1),
                density=True,
                histtype="step",
                lw=0.9,
                color=metric_color(metric),
                label=metric_label(metric),
            )
        ax.axhline(1.0, color="0.35", lw=0.6, ls="--")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 3.4)
        ax.set_xlabel(
            rf"PIT of {SPREAD_REPLICATE[kind]} {rep_id} under the ${SPREAD_SYMBOL[kind]}$ model"
        )
        ax.set_ylabel("Density (243 cells)")
    axes[1, 0].legend(fontsize=5.5, frameon=False, loc="upper center", ncol=3)
    for n, ax in enumerate(axes.ravel()):
        add_panel_label(ax, n)
    fig.tight_layout(pad=0.4)
    save_figure(fig, "representativeness", out_dir=fig_dir())
    plt.close(fig)
    return pd.DataFrame(stats_rows)


def main() -> None:
    out = fig_dir()
    for kind in SPREAD_KINDS:
        plot_spread_by_factor(kind)
    prof = plot_sB_node_profile()
    cov = plot_calibration()
    rep = plot_representativeness()
    prof.to_csv(out / "sB_node_profile.csv", index=False)
    cov.to_csv(out / "spread_coverage.csv", index=False)
    rep.to_csv(out / "representativeness.csv", index=False)
    cov_wide = cov.pivot_table(
        index=["kind", "metric"], columns="level", values="coverage"
    ).reset_index()
    lines = [
        "# Spread-model figures",
        "",
        "## Definitions",
        "",
        r"- `spread_by_factor_<kind>`: box = per-replicate geometric mean of the spread over "
        "the 81 cells at each factor level (5–95% whiskers); replicates are seeds (within) or "
        r"nodes (between). Diamond = model \(\exp\) of the mean \(\hat\mu\) over those cells; "
        "red cross = seed 1 / node 50. f0 within-seed uses nonzero cells only.",
        r"- `sB_node_profile`: \(s_B\) / cell mean of \(s_B\) along the array; median and 5–95% "
        "band over the 243 cells.",
        "- `spread_calibration`: holdout PIT (f0 within-seed uses the hurdle mid-PIT), "
        "central-interval coverage of the Normal part, and the reliability of the f0 zero-spread "
        r"probability \(p_0\).",
        r"- `representativeness`: Y ceiling per seed (nodes as noise) and per node (seeds as "
        "noise), and the PIT of seed 1 / node 50 under the spread models.",
        "",
        "## Node profile of s_B (relative to cell mean)",
        "",
        prof.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Holdout central-interval coverage",
        "",
        cov_wide.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Representativeness of seed 1 / node 50 (Y ceilings)",
        "",
        rep.to_markdown(index=False, floatfmt=".3f"),
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
