"""Replicate-based reliability (signal-to-total variance) ceiling.

For design point i and seed replicate j,

    Y_ij = μ_i + ε_ij

estimate:

* signal variance  — sample variance of the design-point means Ȳ_i
* noise variance   — average within-design sample variance across cells
* ceiling          — σ̂²_signal / (σ̂²_signal + σ̂²_noise)

This is a population-information bound (fraction of variance attributable to
design-point means), not a fitted-model R². It bounds how much of the variance
of a single draw Y can be explained by the design factors only under the
modeling assumptions that seed stochasticity is (i) irreducible system noise
and (ii) independent of the design factors (additive ε_ij ⊥ design). With one
observation per design point the bound is not identifiable.

Also reports the noise-corrected signal (subtracting mean(s²_i / n_i) from
Var(Ȳ_i)) and the oracle between-cell SS fraction as cross-checks; for a
balanced design with large n they agree closely.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FACTORS, MODEL_TARGETS, load_channel50, target_label  # noqa: E402


def reliability_ceiling_from_replicates(
    df: pd.DataFrame,
    target: str,
    *,
    design_cols: list[str] | None = None,
) -> dict[str, float | int]:
    """Estimate the replicate-based signal-to-total variance ceiling.

    Returns a population-information bound, not a fitted-model score. Interpreting
    it as an upper limit on achievable R² for predicting a single draw Y given
    the design factors requires that seed noise be irreducible and independent
    of the design (additive within-cell error).

    Args:
        df: Long table with design columns, a ``seed`` column, and ``target``.
        target: Response column name.
        design_cols: Factor columns defining a design point (default: ``FACTORS``).

    Returns:
        Variance components and reliability / signal-to-total ceilings, plus
        noise-corrected and SS-based cross-checks.
    """
    design_cols = list(design_cols or FACTORS)
    g = df.groupby(design_cols, sort=False)[target]
    n_i = g.count()
    if (n_i < 2).any():
        raise ValueError(
            f"{target}: every design point needs ≥2 replicates; found min n_i={int(n_i.min())}."
        )

    cell_mean = g.mean()
    cell_var = g.var(ddof=1)  # within-cell σ̂²_i
    n_cells = int(len(cell_mean))
    n_bar = float(n_i.mean())
    n_harm = float(n_cells / (1.0 / n_i).sum())  # harmonic mean replicate count

    # Signal-to-total: Var(Ȳ_i) / (Var(Ȳ_i) + mean within-cell variance).
    sigma2_signal = float(cell_mean.var(ddof=1))
    sigma2_noise = float(cell_var.mean())
    reliability = sigma2_signal / (sigma2_signal + sigma2_noise)

    # Method-of-moments: E[Var(Ȳ)] = Var(μ) + E[σ²_i / n_i].
    sigma2_signal_bc = max(0.0, sigma2_signal - float((cell_var / n_i).mean()))
    reliability_bc = sigma2_signal_bc / (sigma2_signal_bc + sigma2_noise)

    # Cross-check: fraction of SS captured by knowing the cell mean.
    y = df[target].to_numpy(dtype=float)
    grand = float(y.mean())
    ss_total = float(((y - grand) ** 2).sum())
    y_cell = g.transform("mean").to_numpy(dtype=float)
    ss_within = float(((y - y_cell) ** 2).sum())
    ss_between = ss_total - ss_within
    reliability_ss = ss_between / ss_total if ss_total > 0 else np.nan

    return dict(
        target=target,
        n_obs=int(len(df)),
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
        frac_seed_noise=1.0 - reliability,
    )


def _label(target: str) -> str:
    extras = {"log_f_ratio": r"$\ln(f_0^N)$"}
    if target in extras:
        return extras[target]
    return target_label(target)


def main() -> None:
    d50 = load_channel50()
    d50["log_f_ratio"] = np.log(d50["f_ratio"])
    # Compare raw vs log for both amplitude and frequency ratio.
    targets = list(MODEL_TARGETS) + ["log_f_ratio"]

    rows = [reliability_ceiling_from_replicates(d50, tgt) for tgt in targets]
    out = pd.DataFrame(rows)

    print(
        f"Design: {len(FACTORS)} factors → cells; "
        f"replicates = seeds per cell.\n"
        f"N = {len(d50):,} rows, "
        f"{out['n_cells'].iloc[0]} cells, "
        f"n̄ = {out['n_bar'].iloc[0]:.1f} seeds/cell.\n"
    )
    print(
        "Reliability (signal-to-total) ceiling = σ²_signal / (σ²_signal + σ²_noise)\n"
        "  Population-information bound, not a fitted-model R².\n"
        "  Interpretable as an R² upper bound for a single draw Y | design only if\n"
        "  seed noise is irreducible and independent of the design factors.\n"
        "  σ²_signal = Var_i(Ȳ_i)   (variance of design-point means)\n"
        "  σ²_noise  = mean_i(s²_i) (average within-cell variance)\n"
        "  _bc       = noise-corrected signal: Var(Ȳ) − mean(s²_i / n_i)\n"
        "  _ss       = SS_between / SS_total (oracle cell-mean variance fraction)\n"
    )

    show = out.assign(
        label=out["target"].map(_label),
        reliability_ceiling=lambda x: x["reliability_ceiling"].round(4),
        reliability_ceiling_bc=lambda x: x["reliability_ceiling_bc"].round(4),
        reliability_ceiling_ss=lambda x: x["reliability_ceiling_ss"].round(4),
        frac_seed_noise=lambda x: x["frac_seed_noise"].round(4),
        sigma2_signal=lambda x: x["sigma2_signal"].map(lambda v: f"{v:.6g}"),
        sigma2_noise=lambda x: x["sigma2_noise"].map(lambda v: f"{v:.6g}"),
    )[
        [
            "target",
            "label",
            "sigma2_signal",
            "sigma2_noise",
            "reliability_ceiling",
            "reliability_ceiling_bc",
            "reliability_ceiling_ss",
            "frac_seed_noise",
        ]
    ]
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
