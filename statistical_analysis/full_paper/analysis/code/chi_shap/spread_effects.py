"""What drives within- vs between-seed spread: SHAP and ALE of the spread models.

Explains the two spread NGBoost models from ``chi_ngboost/train_spread.py``
(\\(Z_W=\\ln s_W\\) over all seeds, \\(Z_B=\\ln s_B\\) over all nodes):

- ``spread_beeswarm.pdf`` — permutation SHAP of \\(\\mu_W\\) (row 1) and
  \\(\\mu_B\\) (row 2) × metrics; ``spread_beeswarm_sigma.pdf`` the same for
  \\(\\log\\sigma_W\\) / \\(\\log\\sigma_B\\).
- ``ale_spread_<metric>.pdf`` — ALE markers at the three factor levels,
  within-seed row over between-seed row, shared \\(\\ln s\\) axis.
- ``f0_spread_hurdle.pdf`` — ALE of the f0 zero-spread probability \\(p_0\\)
  and of \\(\\mu_W\\) given nonzero spread.

Samples come from the held-out groups (seeds / node blocks). Writes under
``figure_dir("chi_shap", "spread_effects")``; ``--force`` recomputes SHAP.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ale_effects import (  # noqa: E402
    ALE_SAMPLE_SEED,
    ALE_SUBSAMPLE_N,
    MARKER_SIZE,
    ale_1d,
    amplitude_table,
)
from common import (  # noqa: E402
    FEATURE_DISPLAY,
    METRICS,
    NGB_FEATURES,
    SHAP_SAMPLE_SEED,
    SPREAD_KINDS,
    factor_levels,
    format_level,
    load_spread,
    out_dir,
    spread_model_path,
    spread_split,
)
from shap_beeswarm import _shap_values, plot_beeswarm  # noqa: E402
from shap_ngboost import _LogSigmaModel, _MuModel  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import (  # noqa: E402
    LABEL_FONTSIZE,
    TICK_LABELSIZE,
    add_panel_label,
    apply_full_paper_style,
    factor_color,
    figsize,
    metric_label,
    save_figure,
)

warnings.filterwarnings("ignore")
# shap_beeswarm switches the style to boxed frames on import; it boxes its own axes.
apply_full_paper_style(auto_format=True, frame="open", grid=False)

FORCE = "--force" in sys.argv
EXPLAIN_N = 400
BG_N = 100
# The design (and hence X) is identical for every metric of one spread table.
SAMPLE_METRIC = "PGA_ratio"
KIND_SYMBOL = {"within": "W", "between": "B"}
KIND_TITLE = {"within": "within-seed spread", "between": "between-seed spread"}
KIND_MARKER = {"within": "o", "between": "s"}
MU_ROWS = tuple((f"mu_{k}", rf"$\mu_{KIND_SYMBOL[k]}$ ({KIND_TITLE[k]})") for k in SPREAD_KINDS)
SIGMA_ROWS = tuple(
    (f"log_sigma_{k}", rf"$\log\sigma_{KIND_SYMBOL[k]}$ ({KIND_TITLE[k]})") for k in SPREAD_KINDS
)


def load_spread_model(kind: str, metric: str, part: str = "normal"):
    path = spread_model_path(kind, metric, part)
    if not path.is_file():
        raise FileNotFoundError(f"Missing spread model: {path}. Run train_spread.py first.")
    return joblib.load(path)


def heldout_rows(kind: str) -> tuple[pd.DataFrame, np.ndarray]:
    df = load_spread(kind, SAMPLE_METRIC)
    _, te = spread_split(df, kind)
    return df, np.asarray(te)


def compute_shap() -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    out = out_dir("spread_effects")
    cache = out / "shap_values_cache.npz"
    keys = [k for k, _ in (*MU_ROWS, *SIGMA_ROWS)]
    if not FORCE and cache.is_file():
        data = np.load(cache, allow_pickle=False)
        if all(f"{k}_{m}" in data.files for k in keys for m in METRICS):
            print(f"Loaded SHAP cache {cache}")
            store = {k: {m: np.asarray(data[f"{k}_{m}"]) for m in METRICS} for k in keys}
            X = {f"mu_{k}": np.asarray(data[f"X_{k}"]) for k in SPREAD_KINDS}
            X.update({f"log_sigma_{k}": X[f"mu_{k}"] for k in SPREAD_KINDS})
            return store, X
    store: dict[str, dict[str, np.ndarray]] = {k: {} for k in keys}
    X: dict[str, np.ndarray] = {}
    for kind in SPREAD_KINDS:
        df, te = heldout_rows(kind)
        pick = np.random.default_rng(SHAP_SAMPLE_SEED).choice(
            te, size=BG_N + EXPLAIN_N, replace=False
        )
        X_all = df[NGB_FEATURES].to_numpy(dtype=float)
        X_bg, X_ex = X_all[pick[:BG_N]], X_all[pick[BG_N:]]
        X[f"mu_{kind}"] = X[f"log_sigma_{kind}"] = X_ex
        for metric in METRICS:
            model = load_spread_model(kind, metric)
            print(f"SHAP spread [{kind}] {metric}: μ, log σ …")
            store[f"mu_{kind}"][metric] = _shap_values(_MuModel(model).predict, X_bg, X_ex)
            store[f"log_sigma_{kind}"][metric] = _shap_values(
                _LogSigmaModel(model).predict, X_bg, X_ex
            )
    np.savez_compressed(
        cache,
        **{f"X_{k}": X[f"mu_{k}"] for k in SPREAD_KINDS},
        **{f"{k}_{m}": store[k][m] for k in keys for m in METRICS},
    )
    print(f"Wrote {cache}")
    return store, X


def importance(store: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    rows = []
    for key, vals in store.items():
        target, kind = key.rsplit("_", 1)
        for metric, sv in vals.items():
            mean_abs = np.mean(np.abs(sv), axis=0)
            rank = stats.rankdata(-mean_abs, method="min").astype(int)
            rows.extend(
                {
                    "kind": kind,
                    "target": target,
                    "metric": metric,
                    "feature": feat,
                    "mean_abs_shap": float(mean_abs[j]),
                    "rank": int(rank[j]),
                }
                for j, feat in enumerate(NGB_FEATURES)
            )
    return pd.DataFrame(rows).sort_values(["kind", "target", "metric", "rank"])


def plot_ale_rows(
    curves: dict[str, dict[str, np.ndarray]],
    raw: dict[str, np.ndarray],
    rows: list[tuple[str, str, str]],
    *,
    title: str,
    sharey: bool,
    stem: str,
    out: Path,
) -> None:
    """Rows × five factors, markers only; *rows* = (curve key, marker, y-label)."""
    n = len(NGB_FEATURES)
    fig, axes = plt.subplots(
        len(rows),
        n,
        figsize=figsize(height=1.75 * len(rows) + 0.35),
        squeeze=False,
        sharey=True if sharey else "row",
    )
    for r, (key, marker, ylabel) in enumerate(rows):
        for j, feat in enumerate(NGB_FEATURES):
            ax = axes[r, j]
            y = curves[key][feat]
            pos = np.arange(len(y))
            ax.plot(pos, y, ls="none", marker=marker, ms=MARKER_SIZE, color=factor_color(feat))
            ax.axhline(0.0, color="0.55", lw=0.5, ls=":")
            ax.margins(y=0.22)
            ax.set_xticks(pos, [format_level(v) for v in raw[feat]], fontsize=TICK_LABELSIZE)
            ax.set_xlim(-0.5, len(y) - 0.5)
            if r == len(rows) - 1:
                ax.set_xlabel(FEATURE_DISPLAY[feat], fontsize=LABEL_FONTSIZE)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
            add_panel_label(ax, r * n + j, fontsize=7)
    fig.suptitle(title, fontsize=LABEL_FONTSIZE, y=0.99)
    fig.tight_layout(pad=0.35, rect=(0, 0, 1, 0.95))
    save_figure(fig, stem, out_dir=out)
    plt.close(fig)


def ale_sample(kind: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    df, te = heldout_rows(kind)
    if len(te) > ALE_SUBSAMPLE_N:
        te = np.random.default_rng(ALE_SAMPLE_SEED).choice(te, size=ALE_SUBSAMPLE_N, replace=False)
    lv = factor_levels(df)
    return df.iloc[te][NGB_FEATURES].to_numpy(dtype=float), {f: lv[f][1] for f in NGB_FEATURES}


def _ale_all(predict_fn, X: np.ndarray, raw, rows: list, **tags) -> dict[str, np.ndarray]:
    curves = {}
    for j, feat in enumerate(NGB_FEATURES):
        lv, ale = ale_1d(predict_fn, X, j)
        curves[feat] = ale
        rows.extend(
            {**tags, "feature": feat, "level_z": float(z), "level": float(r), "effect": float(e)}
            for z, r, e in zip(lv, raw[feat], ale)
        )
    return curves


def run_ale(out: Path) -> pd.DataFrame:
    samples = {k: ale_sample(k) for k in SPREAD_KINDS}
    raw = samples["within"][1]
    rows: list[dict] = []
    for metric in METRICS:
        curves = {}
        for kind in SPREAD_KINDS:
            model = load_spread_model(kind, metric)
            X = samples[kind][0]
            print(f"ALE spread [{kind}] {metric} …")
            curves[kind] = _ale_all(
                _MuModel(model).predict, X, raw, rows, kind=kind, target="mu", metric=metric
            )
        plot_ale_rows(
            curves,
            raw,
            [
                (k, KIND_MARKER[k], rf"ALE on $\mu_{KIND_SYMBOL[k]}=\ln s_{KIND_SYMBOL[k]}$")
                for k in SPREAD_KINDS
            ],
            title=f"{metric_label(metric)} — within-seed (top) vs between-seed (bottom) spread",
            sharey=True,
            stem=f"ale_spread_{metric}",
            out=out,
        )

    X = samples["within"][0]
    zero_model = load_spread_model("within", "f_ratio", "zero")
    print("ALE f0 hurdle p0 …")
    curves = {
        "p0": _ale_all(
            lambda Xq: zero_model.predict_proba(np.asarray(Xq, dtype=float))[:, 1],
            X,
            raw,
            rows,
            kind="within",
            target="p0",
            metric="f_ratio",
        ),
        "mu": {
            f: np.asarray(
                [
                    r["effect"]
                    for r in rows
                    if r["kind"] == "within"
                    and r["target"] == "mu"
                    and r["metric"] == "f_ratio"
                    and r["feature"] == f
                ]
            )
            for f in NGB_FEATURES
        },
    }
    plot_ale_rows(
        curves,
        raw,
        [
            ("p0", "^", r"ALE on $p_0=P(s_W=0)$"),
            ("mu", "o", r"ALE on $\mu_W$ ($s_W>0$)"),
        ],
        title=f"{metric_label('f_ratio')} within-seed spread — two-part (hurdle) model",
        sharey=False,
        stem="f0_spread_hurdle",
        out=out,
    )
    return pd.DataFrame(rows)


def main() -> None:
    out = out_dir("spread_effects")
    store, X = compute_shap()
    plot_beeswarm(
        store,
        X,
        out=out,
        rows=MU_ROWS,
        stem="spread_beeswarm",
        legend_title="Spread models over all replicates — feature value",
    )
    plot_beeswarm(
        store,
        X,
        out=out,
        rows=SIGMA_ROWS,
        stem="spread_beeswarm_sigma",
        legend_title="Spread models over all replicates — feature value",
    )
    imp = importance(store)
    imp.to_csv(out / "spread_shap_importance.csv", index=False)

    ale = run_ale(out)
    ale.to_csv(out / "ale_spread_curves.csv", index=False)
    amp = amplitude_table(ale, ["kind", "target", "metric", "feature"])
    amp.to_csv(out / "ale_spread_range.csv", index=False)
    meta = {
        "explain_n": EXPLAIN_N,
        "bg_n": BG_N,
        "shap_sample_seed": SHAP_SAMPLE_SEED,
        "ale_subsample_n": ALE_SUBSAMPLE_N,
        "sample": "held-out seeds (within) / held-out node blocks (between)",
        "forced": FORCE,
    }
    (out / "spread_effects_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    rank_wide = (
        imp[imp["target"] == "mu"]
        .pivot_table(index=["metric", "feature"], columns="kind", values="rank")
        .reset_index()
    )
    lines = [
        "# What drives within- vs between-seed spread",
        "",
        "## Definitions",
        "",
        r"- Models: Normal NGBoost on \(Z_W=\ln s_W\) (rows = cell × seed, all seeds) and "
        r"\(Z_B=\ln s_B\) (rows = cell × node, all nodes); see `chi_ngboost/spread`.",
        rf"- SHAP: permutation SHAP of \(\mu\) and \(\log\sigma\) ({EXPLAIN_N} explained / "
        f"{BG_N} background rows from held-out seeds or node blocks).",
        r"- ALE: accumulated local effects at the three factor levels, in \(\ln s\) units "
        r"(+0.69 = spread doubles). f0 hurdle: ALE of \(p_0\) (probability scale) and of "
        r"\(\mu_W\) given \(s_W>0\).",
        "",
        "## SHAP rank of μ (1 = most important)",
        "",
        rank_wide.to_markdown(index=False, floatfmt=".0f"),
        "",
        "## ALE amplitude (max − min over the three levels)",
        "",
        amp.to_markdown(index=False, floatfmt=".3f"),
        "",
        "| File | Content |",
        "|------|---------|",
        r"| `spread_beeswarm.pdf` | SHAP of \(\mu_W\) / \(\mu_B\) |",
        r"| `spread_beeswarm_sigma.pdf` | SHAP of \(\log\sigma_W\) / \(\log\sigma_B\) |",
        "| `ale_spread_<metric>.pdf` | ALE markers, within over between |",
        "| `f0_spread_hurdle.pdf` | ALE of f0 p0 and μ_W |",
        "| `spread_shap_importance.csv` / `ale_spread_range.csv` | tables |",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
