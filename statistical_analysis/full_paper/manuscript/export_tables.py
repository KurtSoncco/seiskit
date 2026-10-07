"""Export LaTeX table fragments Tables 2–6 + ARTIFACT_MAP.md.

Reads existing Box CSVs / summaries under complete/full_paper/figures and
writes manuscript-ready ``.tex`` fragments to
``statistical_analysis/full_paper/manuscript/tables/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_FULL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_FULL))

from config import BOX_ROOT, METRICS  # noqa: E402

FIG = BOX_ROOT / "full_paper" / "figures"
OUT = _FULL / "manuscript"
TABLES = OUT / "tables"

MECHANISM = {
    "Vs1": "Anchors 1D impedance / $f_0=V_{s1}/4H$; dominates median location shifts",
    "Height": "Sets resonant period with $V_{s1}$; baseline site-period normalization",
    "CoV": "Drives wave-phase randomization, within-seed variance, and upper-tail spread",
    "rH": "Sets interference scale; controls coherence decay and $r_h\\times CoV$ coupling",
    "aHV": "Directional focusing; modulates between- vs within-realization variance ratio",
}


def _tex_escape(s: str) -> str:
    return (
        str(s)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def _metric_tex(m: str) -> str:
    return {
        "f_ratio": r"$f_0^N$",
        "abs_TF_ratio": r"$\lvert TF\rvert_0^N$",
        "PGA_ratio": r"$PGA^N$",
        "PSA_ratio": r"$SA^N$",  # HDF5 column; paper nomenclature is SA
        "Ia_ratio": r"$I_a^N$",
    }.get(m, _tex_escape(m))


def _booktabs(header: list[str], rows: list[list[str]], caption: str, label: str) -> str:
    ncol = len(header)
    lines = [
        r"\begin{table*}[!ht]",
        rf"\caption{{{caption}\label{{{label}}}}}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}" + ("l" + "c" * (ncol - 1)) + r"@{}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular*}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


PARTITIONS = ("between", "within")
PARTITION_TEX = {"between": "Between-seed", "within": "Within-seed (seed 1)"}
SPREAD_KINDS = ("within", "between")
SPREAD_TEX = {"within": r"Spread $s_W$ (all seeds)", "between": r"Spread $s_B$ (all nodes)"}


def export_table2() -> Path:
    ceil = pd.read_csv(FIG / "chi_ols" / "r2_ceiling" / "reliability_ceiling.csv")
    by_scope = {p: ceil[ceil["scope"] == p].set_index("metric") for p in PARTITIONS}
    rows = []
    for m in METRICS:
        row = [_metric_tex(m)]
        for p in PARTITIONS:
            r = by_scope[p].loc[m]
            row += [
                f"{r['reliability_ceiling']:.3f}",
                f"{r['reliability_ceiling_bc']:.3f}",
                f"{r['frac_noise']:.3f}",
            ]
        rows.append(row)
    tex = _booktabs(
        [
            "Metric",
            r"$R^2_{\mathrm{ceil}}$ (B)",
            r"$R^2_{\mathrm{ceil,bc}}$ (B)",
            r"Noise frac.\ (B)",
            r"$R^2_{\mathrm{ceil}}$ (W)",
            r"$R^2_{\mathrm{ceil,bc}}$ (W)",
            r"Noise frac.\ (W)",
        ],
        rows,
        r"Reliability ceiling $R^2_{\mathrm{ceiling}}$ per variance partition. "
        r"(B) between-seed: center node, all $N_s$ seeds as replicates of each design cell; "
        r"noise is between-seed variance. "
        r"(W) within-seed: one seed, all $N_x$ nodes as replicates; noise is within-seed "
        r"(spatial) variance. Noise frac.\ $=1-R^2_{\mathrm{ceiling}}$ of that partition.",
        "tab:r_ceiling",
    )
    path = TABLES / "tab2_r_ceiling.tex"
    path.write_text(tex, encoding="utf-8")
    return path


def export_table3() -> Path:
    acf = pd.read_csv(FIG / "chi_spatial" / "spatial_acf" / "acf_fit_params.csv")
    rows = []
    for m in METRICS:
        sub = acf[acf["metric"] == m]
        # Prefer CosWM length when fit_ok
        h95 = sub["h95_m_coswm"].where(sub["fit_ok_coswm"], sub["h95_m_exp"])
        rows.append(
            [
                _metric_tex(m),
                f"{sub['rho_lag2_m'].median():.3f}",
                f"{sub['rho_lag2_m'].std():.3f}",
                f"{h95.median():.1f}",
                f"{h95.std():.1f}",
                sub["best_model"].mode().iloc[0] if len(sub) else "—",
            ]
        )
    tex = _booktabs(
        [
            "Metric",
            r"Median $\hat\rho(2\,\mathrm{m})$",
            r"SD $\hat\rho(2\,\mathrm{m})$",
            r"Median $h_{95}$ (m)",
            r"SD $h_{95}$ (m)",
            "Best ACF (mode)",
        ],
        rows,
        r"Spatial autocorrelation summary across design cells: short-lag correlation "
        r"and $h_{95}$ correlation length (CosWM when available, else Exponential).",
        "tab:acf_summary",
    )
    path = TABLES / "tab3_acf.tex"
    path.write_text(tex, encoding="utf-8")
    return path


def export_table4() -> Path:
    src = FIG / "chi_spatial" / "literature_coherence" / "table4_compact.csv"
    if not src.is_file():
        # Placeholder until literature_coherence.py is run
        path = TABLES / "tab4_literature_coherence.tex"
        path.write_text(
            "% Run analysis/code/chi_spatial/literature_coherence.py first.\n",
            encoding="utf-8",
        )
        return path
    tab = pd.read_csv(src)
    rows = []
    for _, r in tab.iterrows():
        rows.append(
            [
                _metric_tex(r["metric"]),
                f"{r['rho_c_10']:.3f}",
                f"{r['Abr_10']:.3f}",
                f"{r['rho_c_50']:.3f}",
                f"{r['Abr_50']:.3f}",
                f"{r['rho_c_100']:.3f}",
                f"{r['Abr_100']:.3f}",
                f"{r['exp_length_m']:.0f}",
            ]
        )
    tex = _booktabs(
        [
            "Metric",
            r"$\bar\rho(10)$ emp.",
            r"Abr.\ $(10)$",
            r"$\bar\rho(50)$ emp.",
            r"Abr.\ $(50)$",
            r"$\bar\rho(100)$ emp.",
            r"Abr.\ $(100)$",
            r"Exp.\ $\ell$ (m)",
        ],
        rows,
        r"Between-seed coherence at the center design cell versus an Abrahamson-type "
        r"lagged-coherency model (reference $f=2$ Hz). Separations in metres.",
        "tab:literature_coherence",
    )
    path = TABLES / "tab4_literature_coherence.tex"
    path.write_text(tex, encoding="utf-8")
    return path


def export_table5() -> Path:
    ceil = pd.read_csv(FIG / "chi_ols" / "r2_ceiling" / "reliability_ceiling.csv")
    rows = []
    csv_rows = []
    for p in PARTITIONS:
        base = FIG / "chi_ngboost" / p
        crps = pd.read_csv(base / "calibration" / "crps_pit_summary.csv").set_index("metric")
        ngb = pd.read_csv(base / "train_ngboost" / "holdout_metrics.csv").set_index("metric")
        c_p = ceil[ceil["scope"] == p].set_index("metric")
        for m in METRICS:
            c, n, r = crps.loc[m], ngb.loc[m], c_p.loc[m]
            eff = n["r2_mean"] / r["reliability_ceiling"]
            rows.append(
                [
                    _metric_tex(m),
                    PARTITION_TEX[p],
                    f"{r['reliability_ceiling']:.3f}",
                    f"{n['r2_mean']:.3f}",
                    f"{eff:.3f}",
                    f"{c['mean_crps']:.3f}",
                    f"{c['ks_stat']:.3f}",
                    f"{n['pi90_coverage']:.3f}",
                ]
            )
            csv_rows.append(
                {
                    "partition": p,
                    "metric": m,
                    "r2_ceiling": r["reliability_ceiling"],
                    "r2_ngboost": n["r2_mean"],
                    "efficiency": eff,
                    "mean_crps": c["mean_crps"],
                    "pit_ks": c["ks_stat"],
                    "pi90_coverage": n["pi90_coverage"],
                }
            )
    for kind in SPREAD_KINDS:
        base = FIG / "chi_ngboost" / "spread" / kind
        hold = pd.read_csv(base / "holdout_metrics.csv").set_index("metric")
        pit = pd.read_csv(base / "test_predictions.csv")
        for m in METRICS:
            h = hold.loc[m]
            ks = stats.kstest(pit.loc[pit["metric"] == m, "pit"], "uniform").statistic
            rows.append(
                [
                    _metric_tex(m),
                    SPREAD_TEX[kind],
                    f"{h['ceiling']:.3f}",
                    f"{h['r2_mean']:.3f}",
                    f"{h['efficiency']:.3f}",
                    f"{h['crps']:.3f}",
                    f"{ks:.3f}",
                    f"{h['pi90_coverage']:.3f}",
                ]
            )
            csv_rows.append(
                {
                    "partition": f"spread_{kind}",
                    "metric": m,
                    "r2_ceiling": h["ceiling"],
                    "r2_ngboost": h["r2_mean"],
                    "efficiency": h["efficiency"],
                    "mean_crps": h["crps"],
                    "pit_ks": ks,
                    "pi90_coverage": h["pi90_coverage"],
                }
            )
    tex = _booktabs(
        [
            "Metric",
            "Model",
            r"$R^2_{\mathrm{ceiling}}$",
            r"$R^2$ NGBoost",
            r"Eff.",
            "CRPS",
            r"PIT KS",
            r"PI90 cov.",
        ],
        rows,
        r"NGBoost adequacy: holdout $R^2$ of $\mu$ relative to the same-scope reliability "
        r"ceiling, CRPS, PIT Kolmogorov--Smirnov statistic, and nominal 90\% "
        r"prediction-interval coverage. Between-seed: $Y=\ln\chi$ at the center node, "
        r"seed-grouped holdout. Within-seed (seed 1): $Y$ for a single realization, holdout of "
        r"contiguous node blocks (supplementary example). Spread $s_W$: $\ln s_W$ per "
        r"(cell, seed) over all seeds, held-out seeds; $f_0$ uses a two-part model and its "
        r"$R^2$/PI90 refer to nonzero spreads. Spread $s_B$: $\ln s_B$ per (cell, node) over "
        r"all nodes, held-out node blocks. Spread ceilings use the seeds (resp.\ nodes) as "
        r"replicates.",
        "tab:model_adequacy",
    )
    path = TABLES / "tab5_model_adequacy.tex"
    path.write_text(tex, encoding="utf-8")
    pd.DataFrame(csv_rows).to_csv(TABLES / "tab5_model_adequacy.csv", index=False)
    return path


def export_table6() -> Path:
    """Synthesis matrix: variance role + spread-model μ SHAP rank / ALE amplitude (B, W)."""

    def factorize(feat: str) -> str:
        return feat.replace("_z", "") if feat.endswith("_z") else feat

    base = FIG / "chi_shap" / "spread_effects"
    imp = pd.read_csv(base / "spread_shap_importance.csv")
    ale = pd.read_csv(base / "ale_spread_range.csv")
    imp = imp[imp["target"] == "mu"].assign(factor=lambda d: d["feature"].map(factorize))
    ale = ale[ale["target"] == "mu"].assign(factor=lambda d: d["feature"].map(factorize))
    n_rank: dict[str, pd.Series] = {}
    ale_amp: dict[str, pd.Series] = {}
    for p in PARTITIONS:
        n_rank[p] = imp[imp["kind"] == p].groupby("factor")["rank"].mean()
        ale_amp[p] = ale[ale["kind"] == p].groupby("factor")["effect_range"].mean()

    var_note = {
        "Vs1": "Median / impedance (low within-seed share)",
        "Height": "Median / resonance anchor",
        "CoV": r"Primary $\bar s_W^2$ driver",
        "rH": r"Coherence / interaction scale",
        "aHV": r"Between/within variance ratio",
    }
    factor_tex = {
        "Vs1": r"$V_{s1}$",
        "Height": r"$H$",
        "CoV": r"$CoV$",
        "rH": r"$r_h$",
        "aHV": r"$a_{hv}$",
    }

    def _fmt(series: pd.Series, f: str, spec: str) -> str:
        return format(series[f], spec) if f in series.index else "—"

    factors = ["Vs1", "Height", "CoV", "rH", "aHV"]
    rows = []
    for f in factors:
        rows.append(
            [
                factor_tex[f],
                var_note.get(f, "—"),
                _fmt(n_rank["between"], f, ".1f"),
                _fmt(n_rank["within"], f, ".1f"),
                _fmt(ale_amp["between"], f, ".3f"),
                _fmt(ale_amp["within"], f, ".3f"),
                MECHANISM.get(f, ""),
            ]
        )
    tex = _booktabs(
        [
            "Parameter",
            "Variance role",
            r"SHAP rank $\mu_B$",
            r"SHAP rank $\mu_W$",
            r"ALE amp.\ (B)",
            r"ALE amp.\ (W)",
            "Physical mechanism (editable)",
        ],
        rows,
        r"Synthesis of variance decomposition roles and what drives each kind of spread: "
        r"SHAP ranks of the spread-model means (lower = more important; averaged over metrics) "
        r"and three-level ALE amplitudes in $\ln s$ units ($0.69$ = spread doubles), for the "
        r"between-seed spread $\mu_B=\mathrm{E}[\ln s_B]$ (B, all nodes) and the within-seed "
        r"spread $\mu_W=\mathrm{E}[\ln s_W]$ (W, all seeds), with validated wave-scattering "
        r"mechanisms.",
        "tab:synthesis",
    )
    path = TABLES / "tab6_synthesis.tex"
    path.write_text(tex, encoding="utf-8")
    pd.DataFrame(
        {
            "factor": factors,
            "variance_role": [var_note[f] for f in factors],
            "shap_rank_spread_mu_between": [n_rank["between"].get(f, np.nan) for f in factors],
            "shap_rank_spread_mu_within": [n_rank["within"].get(f, np.nan) for f in factors],
            "ale_amp_spread_between": [ale_amp["between"].get(f, np.nan) for f in factors],
            "ale_amp_spread_within": [ale_amp["within"].get(f, np.nan) for f in factors],
            "mechanism": [MECHANISM[f] for f in factors],
        }
    ).to_csv(TABLES / "tab6_synthesis.csv", index=False)
    return path


def write_artifact_map() -> Path:
    lines = [
        "# Paper artifact map",
        "",
        "Base: `complete/full_paper/figures/`",
        "",
        "| Paper item | Main-text path | Supplemental |",
        "|------------|----------------|--------------|",
        "| Fig1 | `model_scheme/model_scheme_array.pdf` | `model_scheme_center.pdf` |",
        "| Fig2 | `descriptions/ricker_wave.pdf` | — |",
        "| Fig3 | `vs_rh_realizations/vs_rh_realizations.pdf`, `vs_cov_realizations.pdf` | — |",
        "| Fig4 | `qualitative/center_node_one_seed/3x3/tf_raw_3x3_h50_vs1_230.pdf` | other `h*_vs1_*` |",
        "| Fig5 | `qualitative/one_seed_all_nodes/3x3/tf_raw_3x3_h50_vs1_230.pdf` | other 8 cases |",
        "| Fig6 | `qualitative/center_node_all_seeds/3x3/tf_raw_3x3_h50_vs1_230.pdf` | other 8 cases |",
        "| Fig7 | `qualitative/all_seeds_all_nodes/3x3/tf_raw_3x3_h50_vs1_230.pdf` | other 8 cases |",
        "| Fig9 | `chi_variables/factor_violins/chi_violins_{one_seed_all_nodes,center_node_all_seeds}.pdf` | `distribution_histograms/hist_*.pdf` (same two samples) |",
        "| Table2 | `manuscript/tables/tab2_r_ceiling.tex` ← `chi_ols/r2_ceiling/` | — |",
        "| Fig10 | `chi_variables/central_profiles/seed_profiles/abs_TF_seed_profile_h50_vs1_230.pdf` | other cases/metrics |",
        "| Fig11 | `chi_variables/central_profiles/node_profiles/abs_TF_node_profile_h50_vs1_230.pdf` | other cases |",
        "| Fig12 | `chi_variables/variance_heatmaps/heatmap_frac_*.pdf` | per-metric panels |",
        "| Fig13 | `chi_spatial/spatial_acf/spatial_acf_abs_TF_ratio.pdf` | ACF CSVs other metrics |",
        "| Table3 | `manuscript/tables/tab3_acf.tex` | — |",
        "| Fig14 | `chi_spatial/spatial_coherence/coherence_vs_lag_*.pdf` | — |",
        "| Table4 | `manuscript/tables/tab4_literature_coherence.tex` ← `chi_spatial/literature_coherence/` | compare PDF |",
        "| Table5 | `manuscript/tables/tab5_model_adequacy.tex` ← `chi_ngboost/{between,within}/`, `chi_ngboost/spread/{within,between}/` | PIT histograms per partition |",
        "| Fig15 | `chi_shap/shap_beeswarm/between/shap_beeswarm.pdf` | `shap_beeswarm/within/` (seed 1, single realization) |",
        "| Fig16 | `chi_shap/ale_effects/between/ale_<metric>.pdf` | `ale_effects/within/` (seed 1) |",
        "| Fig17 | `chi_shap/shap_median_vs_tail/between/shap_median_vs_tail_delta_{abs,signed}.pdf` | `shap_median_vs_tail/within/` (seed 1) |",
        "| Fig18 | `chi_shap/ale_dispersion/between/ale_{scale,q95}_<metric>.pdf` | `ale_dispersion/within/` (seed 1) |",
        "| Fig19 | `chi_shap/interactions/between/interactions.pdf` | `interactions/within/` (seed 1) |",
        "| Fig20 | `chi_shap/ale_2d/between/ale_2d_<metric>.pdf` | `ale_2d/within/` (seed 1) |",
        "| Spread-a | `chi_ngboost/spread/figures/spread_by_factor_{within,between}.pdf` | — |",
        "| Spread-b | `chi_shap/spread_effects/spread_beeswarm.pdf` | `spread_beeswarm_sigma.pdf` |",
        "| Spread-c | `chi_shap/spread_effects/ale_spread_<metric>.pdf` | `f0_spread_hurdle.pdf` |",
        "| Spread-supp | — | `chi_ngboost/spread/figures/{sB_node_profile,spread_calibration,representativeness}.pdf`, `chi_ngboost/node_robustness/node_robustness.pdf` |",
        "| Table6 | `manuscript/tables/tab6_synthesis.tex` ← `chi_shap/spread_effects/` | `tab6_synthesis.csv` |",
        "| App1 | `chi_variables/mean_variance_adequacy/mean_variance_adequacy.pdf` | — |",
        "| App2 | `appendix_im/peak_found_rates.pdf` | — |",
        "",
    ]
    path = OUT / "ARTIFACT_MAP.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    paths = [
        export_table2(),
        export_table3(),
        export_table4(),
        export_table5(),
        export_table6(),
        write_artifact_map(),
    ]
    for p in paths:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
