"""Held-out test-seed evaluation of the joint correlation layer + CosWM decision.

Variants: ``indep`` and, for shared / matern32 / wm / coswm, the global θ
(``*_global``) and the partially pooled per-cell θ_k (``*_pooled``).

On the 25 held-out test seeds (standardized with the full NGBoost model):

- joint log score per whole profile on the Y = ln χ scale
- simulated vs observed profile diagnostics (same estimators as ``chi_spatial``):
  across-seed correlation vs separation, within-seed demeaned ACF, variance of
  each seed's spatial mean, within-seed variance, increment variance, and 90%
  coverage of the whole-profile mean and max of χ
- seed-profile bootstrap CIs for model differences

Decision (primary pair = best CosWM variant vs best Matérn-3/2 variant):
CosWM is "better" iff the 95% CI of its joint log-score gain is > 0 **and** it is
significantly worse (CI of discrepancy difference > 0) on fewer than half of the
profile diagnostics.

Writes CSV + PDFs + ``summary.md`` + ``decision.json`` under
``figure_dir("chi_joint", "evaluate_joint")``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import covariance as cv  # noqa: E402
from common import DX_M, METRIC, out_dir, residuals_path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import add_panel_label, apply_full_paper_style, figsize, save_figure  # noqa: E402

apply_full_paper_style(auto_format=True, frame="open", grid=False)

N_REP = 20  # simulated replicate test sets per cell (each of n_test_seeds profiles)
SIM_SEED = 31
B_LS = 1000
B_DIAG = 200
BOOT_SEED = 41
DIAG_LAGS = np.array([1, 5, 25, 50])  # node lags → 2, 10, 50, 100 m
PI_LO, PI_HI = 0.05, 0.95
POOLED = ("shared", "matern32", "wm", "coswm")
PLOT_VARIANTS = ("indep", "shared_pooled", "matern32_pooled", "coswm_pooled")


def variant_thetas(fit_dir: Path, metric: str, n_cells: int) -> dict[str, tuple[str, np.ndarray]]:
    """variant → (model, θ per cell (cells, k))."""
    g = np.load(fit_dir / f"theta_global_{metric}.npz")
    c = np.load(fit_dir / f"theta_cell_{metric}.npz")
    out = {"indep": ("indep", np.zeros((n_cells, 0)))}
    for m in POOLED:
        out[f"{m}_global"] = (m, np.tile(g[m], (n_cells, 1)))
        out[f"{m}_pooled"] = (m, c[f"{m}_theta"])
    return out


def diag_stats(Z: np.ndarray, lags: np.ndarray) -> dict[str, np.ndarray]:
    """Diagnostics over leading dims of Z (..., n_profiles, p)."""
    d = {}
    ac = cv.across_seed_corr(Z, lags)
    wa = cv.demeaned_acf(Z, lags)
    for i, k in enumerate(lags):
        d[f"across_corr_{int(k * DX_M)}m"] = ac[..., i]
        d[f"within_acf_{int(k * DX_M)}m"] = wa[..., i]
    d.update(cv.profile_stats(Z, lags))
    return d


def coverage_indicators(y_obs, mu, sig, z_sim):
    """Per-profile indicators that observed mean / max of χ fall in the model 90% band.

    y_obs (cells, seeds, p); z_sim (cells, n_sim, p).
    """
    chi_sim = np.exp(mu[:, None, :] + sig[:, None, :] * z_sim)
    chi_obs = np.exp(y_obs)
    out = {}
    for name, fn in (("profile_mean", np.mean), ("profile_max", np.max)):
        s = fn(chi_sim, axis=-1)
        lo, hi = np.quantile(s, PI_LO, axis=1), np.quantile(s, PI_HI, axis=1)
        o = fn(chi_obs, axis=-1)
        out[f"cov90_{name}"] = (o >= lo[:, None]) & (o <= hi[:, None])
    return out


def main(metric: str = METRIC) -> None:
    out = out_dir("evaluate_joint")
    fit_dir = out_dir("fit_joint_cov")
    cubes = np.load(residuals_path(metric))
    z_te, y_te = cubes["z_test"], cubes["y_test"]
    mu, sig = cubes["mu_test"], cubes["sigma_test"]
    n_cells, n_te, p = z_te.shape
    variants = variant_thetas(fit_dir, metric, n_cells)
    rng = np.random.default_rng(SIM_SEED)
    log_sig = np.log(sig).sum(axis=1)  # (cells,)

    # --- log scores, simulations, model-side diagnostics ------------------
    ls = {}  # variant → (cells, seeds)
    model_diag = {}  # variant → {diag: (cells,)}
    cov_ind = {}  # variant → {name: (cells, seeds) bool}
    full_lags = np.arange(1, p - 1)
    curves = {}
    sim_sd = {}
    for name, (model, thetas) in variants.items():
        ls_v = np.empty((n_cells, n_te))
        sims = np.empty((n_cells, N_REP * n_te, p))
        for c in range(n_cells):
            R = cv.build_R(model, thetas[c], p)
            ls_v[c] = cv.loglik_profiles(R, z_te[c]) - log_sig[c]
            sims[c] = cv.simulate(R, N_REP * n_te, rng)
        ls[name] = ls_v
        sim_sd[name] = float(np.std(sims))
        rep = sims.reshape(n_cells, N_REP, n_te, p)
        model_diag[name] = {k: v.mean(axis=1) for k, v in diag_stats(rep, DIAG_LAGS).items()}
        cov_ind[name] = coverage_indicators(y_te, mu, sig, sims)
        if name in PLOT_VARIANTS:
            curves[name] = (
                cv.across_seed_corr(rep, full_lags).mean(axis=1),
                cv.demeaned_acf(rep, full_lags).mean(axis=1),
            )
        print(f"  {name}: LS/profile={ls_v.mean():.3f}  sim sd={sim_sd[name]:.3f}")
    curves["observed"] = (cv.across_seed_corr(z_te, full_lags), cv.demeaned_acf(z_te, full_lags))
    diag_names = list(model_diag["indep"].keys())

    # --- bootstrap over test seeds ----------------------------------------
    brng = np.random.default_rng(BOOT_SEED)
    names = list(variants)
    ls_arr = np.stack([ls[v] for v in names])  # (V, cells, seeds)
    ls_boot = np.empty((B_LS, len(names)))
    for b in range(B_LS):
        idx = brng.integers(0, n_te, n_te)
        ls_boot[b] = ls_arr[:, :, idx].mean(axis=(1, 2))

    obs_diag = diag_stats(z_te, DIAG_LAGS)
    disc_names = diag_names + ["cov90_profile_mean", "cov90_profile_max"]

    def discrepancies(obs: dict, idx: np.ndarray) -> np.ndarray:
        """(V, D) discrepancy: mean over cells |model − obs|; coverage → |cov − 0.90|."""
        out_ = np.empty((len(names), len(disc_names)))
        for i, v in enumerate(names):
            for j, d in enumerate(disc_names):
                if d.startswith("cov90_"):
                    out_[i, j] = abs(cov_ind[v][d][:, idx].mean() - 0.90)
                else:
                    out_[i, j] = np.mean(np.abs(model_diag[v][d] - obs[d]))
        return out_

    disc_hat = discrepancies(obs_diag, np.arange(n_te))
    disc_boot = np.empty((B_DIAG, len(names), len(disc_names)))
    for b in range(B_DIAG):
        idx = brng.integers(0, n_te, n_te)
        disc_boot[b] = discrepancies(diag_stats(z_te[:, idx, :], DIAG_LAGS), idx)

    # --- tables -----------------------------------------------------------
    ls_mean = ls_arr.mean(axis=(1, 2))
    ls_tab = pd.DataFrame(
        {
            "variant": names,
            "ls_per_profile": ls_mean,
            "ls_lo": np.quantile(ls_boot, 0.025, axis=0),
            "ls_hi": np.quantile(ls_boot, 0.975, axis=0),
            "sim_sd_z": [sim_sd[v] for v in names],
        }
    ).sort_values("ls_per_profile", ascending=False)
    ls_tab.to_csv(out / f"joint_log_score_{metric}.csv", index=False)

    diag_rows = []
    for i, v in enumerate(names):
        for j, d in enumerate(disc_names):
            if d.startswith("cov90_"):
                model_val, obs_val = float(cov_ind[v][d].mean()), 0.90
            else:
                model_val, obs_val = float(model_diag[v][d].mean()), float(obs_diag[d].mean())
            diag_rows.append(
                {
                    "variant": v,
                    "diagnostic": d,
                    "model_mean_over_cells": model_val,
                    "observed_mean_over_cells": obs_val,
                    "discrepancy": disc_hat[i, j],
                }
            )
    diag_tab = pd.DataFrame(diag_rows)
    diag_tab.to_csv(out / f"profile_diagnostics_{metric}.csv", index=False)

    def best_of(model: str) -> str:
        cands = [f"{model}_global", f"{model}_pooled"]
        return max(cands, key=lambda v: ls_mean[names.index(v)])

    def compare(a: str, b: str) -> pd.DataFrame:
        ia, ib = names.index(a), names.index(b)
        rows = [
            {
                "pair": f"{a} vs {b}",
                "quantity": "joint_log_score_gain",
                "estimate": ls_mean[ia] - ls_mean[ib],
                "lo": float(np.quantile(ls_boot[:, ia] - ls_boot[:, ib], 0.025)),
                "hi": float(np.quantile(ls_boot[:, ia] - ls_boot[:, ib], 0.975)),
            }
        ]
        for j, d in enumerate(disc_names):
            diff = disc_boot[:, ia, j] - disc_boot[:, ib, j]
            with np.errstate(invalid="ignore", divide="ignore"):
                rel = -diff / disc_boot[:, ib, j]  # relative reduction of a vs b
            rows.append(
                {
                    "pair": f"{a} vs {b}",
                    "quantity": f"disc_{d}",
                    "estimate": disc_hat[ia, j] - disc_hat[ib, j],
                    "lo": float(np.quantile(diff, 0.025)),
                    "hi": float(np.quantile(diff, 0.975)),
                    "rel_reduction": float(1.0 - disc_hat[ia, j] / disc_hat[ib, j]),
                    "rel_lo": float(np.nanquantile(rel, 0.025)),
                    "rel_hi": float(np.nanquantile(rel, 0.975)),
                }
            )
        return pd.DataFrame(rows)

    cos_v, mat_v, wm_v = best_of("coswm"), best_of("matern32"), best_of("wm")
    pairs = [
        (cos_v, mat_v),
        (wm_v, mat_v),
        (cos_v, wm_v),
        (mat_v, "shared_pooled"),
        ("shared_pooled", "indep"),
    ]
    comp = pd.concat([compare(a, b) for a, b in pairs], ignore_index=True)
    comp.to_csv(out / f"model_comparisons_{metric}.csv", index=False)

    prim = comp[comp["pair"] == f"{cos_v} vs {mat_v}"]
    gain = prim[prim["quantity"] == "joint_log_score_gain"].iloc[0]
    dis = prim[prim["quantity"].str.startswith("disc_")]
    n_worse = int((dis["lo"] > 0).sum())
    n_better = int((dis["hi"] < 0).sum())
    coswm_better = bool(gain["lo"] > 0 and n_worse < len(dis) / 2)
    decision = {
        "metric": metric,
        "coswm_variant": cos_v,
        "matern32_variant": mat_v,
        "ls_gain": float(gain["estimate"]),
        "ls_gain_ci": [float(gain["lo"]), float(gain["hi"])],
        "n_diagnostics": int(len(dis)),
        "n_significantly_worse": n_worse,
        "n_significantly_better": n_better,
        "coswm_better": coswm_better,
    }
    (out / f"decision_{metric}.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    # --- figures ----------------------------------------------------------
    h = full_lags * DX_M
    labels = {
        "observed": "Observed (test seeds)",
        "indep": "Independent",
        "shared_pooled": "Shared + nugget",
        "matern32_pooled": "Shared + Matérn-3/2",
        "coswm_pooled": "Shared + CosWM",
    }
    styles = {
        "observed": dict(color="k", lw=1.0),
        "indep": dict(color="0.6", lw=0.8, ls=":"),
        "shared_pooled": dict(color="#CCBB44", lw=0.8, ls="-."),
        "matern32_pooled": dict(color="#4477AA", lw=0.9, ls="--"),
        "coswm_pooled": dict(color="#EE6677", lw=0.9),
    }
    fig, axes = plt.subplots(1, 2, figsize=figsize(aspect=0.36))
    for i, (ax, ylabel) in enumerate(
        zip(axes, ("Across-seed correlation", "Within-seed demeaned ACF"), strict=True)
    ):
        for key in ("observed", *PLOT_VARIANTS):
            arr = curves[key][i]
            med = np.median(arr, axis=0)
            ax.plot(h, med, label=labels[key], **styles[key])
            if key == "observed":
                ax.fill_between(
                    h,
                    np.quantile(arr, 0.25, axis=0),
                    np.quantile(arr, 0.75, axis=0),
                    color="k",
                    alpha=0.12,
                    lw=0,
                )
        ax.axhline(0, color="0.55", lw=0.5, ls=":")
        ax.axvline(100, color="0.55", lw=0.5, ls="--")
        ax.set_xlabel("Separation $h$ (m)")
        ax.set_ylabel(ylabel)
        add_panel_label(ax, i)
    axes[0].legend(frameon=False, fontsize=6, loc="lower left")
    fig.tight_layout()
    save_figure(fig, f"corr_vs_lag_{metric}", out_dir=out)
    plt.close(fig)

    fig, (ax_ls, ax_d) = plt.subplots(
        1, 2, figsize=figsize(aspect=0.42), gridspec_kw={"width_ratios": [1, 2.6]}
    )

    def _color(lo, hi):
        return "#228833" if lo > 0 else ("#EE6677" if hi < 0 else "0.4")

    ax_ls.errorbar(
        [gain["estimate"]],
        [0],
        xerr=[[gain["estimate"] - gain["lo"]], [gain["hi"] - gain["estimate"]]],
        fmt="o",
        ms=3,
        lw=0.8,
        color=_color(gain["lo"], gain["hi"]),
    )
    ax_ls.axvline(0, color="0.55", lw=0.5)
    ax_ls.set_yticks([0])
    ax_ls.set_yticklabels(["joint log score"])
    ax_ls.set_xlabel("Gain (nats / profile)")
    add_panel_label(ax_ls, 0)
    rows = dis.reset_index(drop=True)
    y = np.arange(len(rows))[::-1]
    for yi, (_, r) in zip(y, rows.iterrows(), strict=True):
        est, lo, hi = 100 * r["rel_reduction"], 100 * r["rel_lo"], 100 * r["rel_hi"]
        ax_d.errorbar(
            est, yi, xerr=[[est - lo], [hi - est]], fmt="o", ms=2.5, lw=0.8, color=_color(lo, hi)
        )
    ax_d.axvline(0, color="0.55", lw=0.5)
    ax_d.set_yticks(y)
    ax_d.set_yticklabels([q.replace("disc_", "") for q in rows["quantity"]], fontsize=6)
    ax_d.set_xlabel("Discrepancy reduction vs Matérn-3/2 (%; right = CosWM closer)")
    add_panel_label(ax_d, 1)
    fig.tight_layout()
    save_figure(fig, f"coswm_vs_matern_forest_{metric}", out_dir=out)
    plt.close(fig)

    # --- summary ----------------------------------------------------------
    disc_wide = diag_tab.pivot(index="diagnostic", columns="variant", values="discrepancy")
    keep = ["indep", "shared_pooled", mat_v, wm_v, cos_v]
    lines = [
        f"# Joint correlation layer: held-out evaluation ({metric})",
        "",
        "## Definitions",
        "",
        r"- Test profiles: 25 held-out seeds × 243 cells, standardized with the full NGBoost \(\hat\mu,\hat\sigma\).",
        r"- Joint log score: \(\log\mathcal N_{101}(\mathbf Y\mid\hat{\boldsymbol\mu},DRD)\) per whole profile (higher is better).",
        f"- Diagnostics: model value = mean over {N_REP} simulated replicate test sets of the same estimator applied to observed data (`chi_spatial` estimators); discrepancy = mean over cells |model − observed|; `cov90_*` = |coverage − 0.90| of whole-profile mean/max of χ.",
        f"- CIs: seed-profile bootstrap (B = {B_LS} for log score, {B_DIAG} for diagnostics).",
        "",
        "## Joint log score",
        "",
        ls_tab.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Diagnostic discrepancy (lower is better)",
        "",
        disc_wide[[k for k in keep if k in disc_wide.columns]].to_markdown(floatfmt=".4f"),
        "",
        "## Model comparisons (estimate [95% CI]; disc_* < 0 means first model closer to data)",
        "",
        comp.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Decision",
        "",
        f"- Primary pair: **{cos_v}** vs **{mat_v}**.",
        f"- Joint log-score gain: **{gain['estimate']:.4f}** [{gain['lo']:.4f}, {gain['hi']:.4f}] nats/profile.",
        f"- Diagnostics significantly worse / better for CosWM: {n_worse} / {n_better} of {len(dis)}.",
        f"- **CosWM better: {coswm_better}.**",
        "",
        "## Caveats",
        "",
        "- The array spans 200 m and the kernel was fitted to OOF residual covariance across seeds; correlation at separations beyond the aperture is not identified.",
        "- The CosWM hole-effect parameter ω is weakly identified at this aperture (log-likelihood differences of order 10⁻³ nats/profile between ω = 0 and moderate ω in synthetic checks with 75 profiles).",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(decision, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:2])
