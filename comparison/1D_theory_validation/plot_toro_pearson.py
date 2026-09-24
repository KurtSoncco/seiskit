"""Pearson r of each Toro Vs-only TF vs the deterministic 1D base TF."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from seiskit.plot_config import apply_style
from seiskit.profile_randomization import (
    ProfileRandomizationConfig,
    build_base_case_profile,
    generate_toro_profile,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "rayleigh_equivalent"
N_REAL = 200


def load_toro():
    spec = importlib.util.spec_from_file_location("plot_toro_taborda", ROOT / "plot_toro_taborda.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["plot_toro_taborda"] = mod
    spec.loader.exec_module(mod)
    return mod


def pearson_ln(a: np.ndarray, b: np.ndarray) -> float:
    x = np.log(np.clip(np.asarray(a, float), 1e-12, None))
    y = np.log(np.clip(np.asarray(b, float), 1e-12, None))
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        return np.nan
    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def summarize(name: str, r: np.ndarray) -> None:
    r = r[np.isfinite(r)]
    print(
        f"{name:12s}  n={len(r)}  mean={np.mean(r):.3f}  "
        f"p50={np.median(r):.3f}  p16–p84={np.percentile(r,16):.3f}–{np.percentile(r,84):.3f}  "
        f"min={np.min(r):.3f}  max={np.max(r):.3f}"
    )


def hist_panel(ax, r, color, title):
    r = r[np.isfinite(r)]
    bins = np.linspace(-0.1, 1.0, 23)
    ax.hist(r, bins=bins, color=color, edgecolor="white", alpha=0.9)
    ax.axvline(np.median(r), color="0.15", lw=1.6, ls="--", label=fr"median={np.median(r):.2f}")
    ax.axvline(np.mean(r), color="0.15", lw=1.2, ls=":", label=fr"mean={np.mean(r):.2f}")
    ax.set_xlim(-0.1, 1.0)
    ax.set_xlabel(r"Pearson $r(\ln|AF|_{\mathrm{Toro}},\ \ln|AF|_{1\mathrm{D}})$")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, frameon=True)


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    t = load_toro()
    vs_mean = t.travel_time_vs()
    cfg = ProfileRandomizationConfig(
        vs_mean=vs_mean,
        thickness=t.H_SOIL,
        dz=t.DZ,
        cov=t.COV,
        vs_bedrock=t.VS_ROCK,
        bedrock_thickness=t.H_ROCK,
        sigma_ln_vs=t.SIGMA_LN_VS,
        use_full_model=True,
        randomize_layer_thickness=False,
        randomize_bedrock_depth=False,
        vary_bedrock_vs=False,
    )
    freq = np.logspace(np.log10(0.2), np.log10(12.0), 700)
    base = build_base_case_profile(cfg)
    n_soil = max(1, int(round(t.H_SOIL / t.DZ)))
    af_base_q = t.tf_profile(base, n_soil, t.DZ, freq, t.XI_SCALE_Q)
    af_base_d = t.tf_profile(base, n_soil, t.DZ, freq, t.XI_SCALE_DMIN)

    r_q = np.empty(N_REAL)
    r_d = np.empty(N_REAL)
    vs1 = np.empty(N_REAL)
    for k in range(N_REAL):
        rng = np.random.default_rng(t.SEED0 + k)
        prof = generate_toro_profile(cfg, rng)
        ns = int(prof.n_soil_samples)
        vs1[k] = float(prof.vs_depth[0])
        af_q = t.tf_profile(prof.vs_depth, ns, t.DZ, freq, t.XI_SCALE_Q)
        af_d = t.tf_profile(prof.vs_depth, ns, t.DZ, freq, t.XI_SCALE_DMIN)
        r_q[k] = pearson_ln(af_q, af_base_q)
        r_d[k] = pearson_ln(af_d, af_base_d)

    print(f"Toro Vs-only vs 1D base  n={N_REAL}  σ_ln Vs=CoV={t.SIGMA_LN_VS:.2f}  H={t.H_SOIL} m")
    summarize("ξ_Q", r_q)
    summarize("0.25 ξ_Q", r_d)
    csv_path = OUT / "toro_vs_1d_pearson.csv"
    np.savetxt(
        csv_path,
        np.column_stack([np.arange(N_REAL), vs1, r_q, r_d]),
        delimiter=",",
        header="seed,vs1,r_xiQ,r_0p25xiQ",
        comments="",
    )
    print(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), constrained_layout=True, sharey=True)
    hist_panel(axes[0], r_q, "#0072B2", r"Taborda–Bielak $\xi_Q$")
    hist_panel(axes[1], r_d, "#D55E00", r"$0.25\,\xi_Q$")
    fig.suptitle(
        r"Pearson $r$: Toro $V_s$-only TF vs 1D base  (fixed $H$, no NHPP)",
        fontsize=11,
    )
    out = OUT / "toro_vs_1d_pearson_hist.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
