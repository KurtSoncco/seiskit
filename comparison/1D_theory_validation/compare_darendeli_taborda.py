"""Darendeli (2001) Dmin vs Taborda–Bielak ξ_Q, and the effect on Dmult TFs.

Dmult (Tao & Rathje 2019; contrast fit from Dawadi et al. 2026) was calibrated
on Darendeli Dmin at f = 3 Hz. The Response_Variability arm applies it to
Taborda–Bielak ξ_Q instead. This script quantifies the gap:

1. Dmin and ξ_Q vs Vs, and R = ξ_Q / Dmin for the 64 Sobol columns.
2. Depth profiles for the Hallal 5-layer column.
3. 1D Thomson–Haskell TFs for four bases: ξ_Q, Dmult·ξ_Q, Dmin, Dmult·Dmin,
   plus the multiplier k on ξ_Q that reproduces the Dmult·Dmin f0 peak.

Darendeli Dmin has no direct Vs dependence; it is driven by σ'm, taken once per
layer at its mid-depth (constant within a layer, like Vs). Soil
assumptions follow Dawadi et al. (2026): ρ = 2000 kg/m³, plus K0 and a water
table (dry by default). Rock Dmin uses the same Darendeli expression at rock depth
(PI = 0), as an extrapolation.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from seiskit.damping import compute_damping_from_Q, compute_darendeli_dmin, compute_quality_factor
from seiskit.plot_config import apply_style
from seiskit.profile_randomization import dmult_from_vs_contrast
from seiskit.theory.layered_1d_tf import Layer, RockHalfspace, layered_transfer_function

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "darendeli_taborda"
SOBOL_CSV = ROOT.parent / "Response_Variability" / "rv_sobol_base_cases.csv"

RHO = 2000.0  # kg/m³, Dawadi et al. (2026)
G = 9.81
GAMMA_W = 9.81  # kN/m³
GWT = np.inf  # m, water-table depth (dry: no water table)
K0 = 0.5  # at-rest coefficient (assumed)
OCR = 1.0
F_REF = 3.0  # Hz, Dmin reference frequency (Dawadi et al. 2026 used 3 Hz); --f-ref overrides
PI_LIST = (0.0, 15.0, 30.0)
PI_TF = 0.0  # PI used for the TF comparison
DZ = 2.0  # m, sublayer thickness for depth-varying Dmin

# Illustrative Vs–σ'm link (Hardin-type Vs = A (σ'm/pa)^n) for the Vs axis
VS_A = 200.0  # m/s at 1 atm (assumed)
VS_N = 0.25

FREQ = np.logspace(np.log10(0.1), np.log10(30.0), 3000)
K_GRID = np.logspace(np.log10(0.02), np.log10(20.0), 240)

COLORS = {"tb": "#0072B2", "tb_dm": "#56B4E9", "dar": "#D55E00", "dar_dm": "#E69F00"}


def xi_tb(vs: float) -> float:
    return float(compute_damping_from_Q(compute_quality_factor(float(vs))))


def sigma_m_kpa(z: np.ndarray | float) -> np.ndarray:
    """Mean effective stress (kPa) at depth z for uniform ρ and a water table."""
    z = np.asarray(z, dtype=float)
    sv_eff = RHO * G * z / 1000.0 - GAMMA_W * np.clip(z - GWT, 0.0, None)
    return sv_eff * (1.0 + 2.0 * K0) / 3.0


def dmin_at(z, PI: float = PI_TF, freq: float | None = None) -> np.ndarray:
    freq = F_REF if freq is None else freq
    return compute_darendeli_dmin(sigma_m_kpa(z), PI=PI, OCR=OCR, freq=freq)


def sublayer_mids(H: float) -> tuple[np.ndarray, np.ndarray]:
    """Sublayer thicknesses and the depth where each takes Dmin (its layer's mid-depth)."""
    n = max(1, int(np.ceil(H / DZ)))
    h = np.full(n, H / n)
    return h, np.full(n, 0.5 * H)


def column_layers(
    h: np.ndarray, vs: np.ndarray, z_mid: np.ndarray, vs_rock: float, z_rock: float, base: str, mult: float
) -> tuple[list[Layer], RockHalfspace]:
    if base == "tb":
        xi_soil = np.array([xi_tb(v) for v in vs])
        xi_rock = xi_tb(vs_rock)
    else:
        xi_soil = dmin_at(z_mid)
        xi_rock = float(dmin_at(z_rock, PI=0.0))
    layers = [Layer(float(hh), float(v), RHO, float(mult * x)) for hh, v, x in zip(h, vs, xi_soil)]
    return layers, RockHalfspace(float(vs_rock), RHO, float(mult * xi_rock))


def af_within(freq: np.ndarray, layers: list[Layer]) -> np.ndarray:
    """Vectorized |AF_within| (same propagator as ``layered_transfer_function``)."""
    omega = 2.0 * np.pi * np.asarray(freq, dtype=float)
    u = np.ones_like(omega, dtype=complex)
    tau = np.zeros_like(omega, dtype=complex)
    for L in layers:
        vs_c = L.Vs * np.sqrt(1.0 + 2.0j * L.xi)
        kh = omega / vs_c * L.H
        c, s = np.cos(kh), np.sin(kh)
        gk = 1j * omega * L.rho * vs_c
        u, tau = u * c + tau * (s / gk), -u * gk * s + tau * c
    return np.abs(1.0 / u)


def f0_peak(freq: np.ndarray, af: np.ndarray, f0: float) -> tuple[float, float]:
    band = (freq > 0.6 * f0) & (freq < 1.5 * f0)
    i = int(np.argmax(af[band]))
    return float(freq[band][i]), float(af[band][i])


def matching_k(h, vs, z_mid, vs_rock, z_rock, f0, a_target) -> float:
    """Multiplier k on ξ_Q whose f0 peak amplitude equals ``a_target``."""
    fb = FREQ[(FREQ > 0.6 * f0) & (FREQ < 1.5 * f0)]
    amps = []
    for k in K_GRID:
        ly, rk = column_layers(h, vs, z_mid, vs_rock, z_rock, "tb", k)
        amps.append(float(np.max(af_within(fb, ly))))
    amps = np.array(amps)  # decreasing in k
    return float(np.exp(np.interp(np.log(a_target), np.log(amps[::-1]), np.log(K_GRID[::-1]))))


def load_extracted():
    spec = importlib.util.spec_from_file_location("extracted_profile", ROOT / "plot_extracted_profile.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["extracted_profile"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_sobol() -> list[dict]:
    with SOBOL_CSV.open() as fh:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(fh)]


# ---------------------------------------------------------------------------
# Hallal 5-layer column
# ---------------------------------------------------------------------------


def hallal_column(mod) -> dict:
    tab = mod.TABLE
    soil = tab[:-1]
    h_all, vs_all, z_all = [], [], []
    for _, z_bot, h, vs, _ in soil:
        n = max(1, int(np.ceil(h / DZ)))
        z_top = z_bot - h
        h_all.append(np.full(n, h / n))
        vs_all.append(np.full(n, vs))
        z_all.append(np.full(n, z_top + 0.5 * h))  # Dmin at layer mid-depth
    h = np.concatenate(h_all)
    vs = np.concatenate(vs_all)
    z_mid = np.concatenate(z_all)
    vs_rock = float(tab[-1, 3])
    z_rock = float(tab[-2, 1] + 0.5 * tab[-1, 2])
    vs1 = float(np.sum(soil[:, 2]) / np.sum(soil[:, 2] / soil[:, 3]))
    H = float(np.sum(soil[:, 2]))
    return dict(h=h, vs=vs, z_mid=z_mid, vs_rock=vs_rock, z_rock=z_rock, vs1=vs1, H=H)


def run_hallal(mod) -> dict:
    col = hallal_column(mod)
    dm = dmult_from_vs_contrast(col["vs1"], col["vs_rock"])
    f0 = col["vs1"] / (4.0 * col["H"])
    raw = np.loadtxt(mod.HALLAL_RAW, delimiter=",", skiprows=1)
    f_h, a_h = mod.clean_digitized_tf(raw[:, 0], raw[:, 1])

    curves = {}
    for key, base, mult in (("tb", "tb", 1.0), ("tb_dm", "tb", dm), ("dar", "dar", 1.0), ("dar_dm", "dar", dm)):
        ly, rk = column_layers(col["h"], col["vs"], col["z_mid"], col["vs_rock"], col["z_rock"], base, mult)
        curves[key] = af_within(FREQ, ly)

    band = (f_h > 0.5) & (f_h < 10.0)
    misfit = {}
    for key, af in curves.items():
        af_i = np.exp(np.interp(np.log(f_h[band]), np.log(FREQ), np.log(af)))
        misfit[key] = float(np.sqrt(np.mean((np.log(af_i) - np.log(a_h[band])) ** 2)))

    print(f"Hallal column: Vs1_tt={col['vs1']:.1f}  Vs2={col['vs_rock']:.0f}  "
          f"contrast={col['vs_rock'] / col['vs1']:.2f}  Dmult={dm:.2f}")
    xi_q_soil = np.array([xi_tb(v) for v in col["vs"]])
    dmin_soil = dmin_at(col["z_mid"])
    w = col["h"] / col["vs"]
    print(f"  soil tt-avg: xi_TB={100 * np.sum(w * xi_q_soil) / w.sum():.2f}%  "
          f"Dmin({F_REF:g}Hz,PI=0)={100 * np.sum(w * dmin_soil) / w.sum():.2f}%  "
          f"rock: xi_TB={100 * xi_tb(col['vs_rock']):.2f}%  Dmin={100 * float(dmin_at(col['z_rock'], PI=0.0)):.2f}%")
    for key, m in misfit.items():
        print(f"  ln-TF RMS misfit vs Hallal digitized [{key}]: {m:.3f}")
    return dict(col=col, dm=dm, f0=f0, curves=curves, f_h=f_h, a_h=a_h, raw=raw, misfit=misfit)


# ---------------------------------------------------------------------------
# Sobol columns
# ---------------------------------------------------------------------------


def run_sobol(cases: list[dict]) -> list[dict]:
    rows = []
    for c in cases:
        vs1, H, vs2 = c["Vs1"], c["H"], c["Vs2"]
        h, z_mid = sublayer_mids(H)
        vs = np.full_like(h, vs1)
        z_rock = H + 0.5 * c["bedrock_thickness"]
        dm = dmult_from_vs_contrast(vs1, vs2)
        f0 = vs1 / (4.0 * H)
        row = dict(sobol_id=int(c["sobol_id"]), vs1=vs1, H=H, vs2=vs2, contrast=vs2 / vs1, dmult=dm, f0=f0)
        row["xi_tb_soil"] = xi_tb(vs1)
        row["xi_tb_rock"] = xi_tb(vs2)
        for pi in PI_LIST:
            row[f"dmin_soil_pi{pi:.0f}"] = float(np.mean(dmin_at(z_mid, PI=pi)))
        row["dmin_rock"] = float(dmin_at(z_rock, PI=0.0))
        row["R_soil"] = row["xi_tb_soil"] / row[f"dmin_soil_pi{PI_TF:.0f}"]
        for key, base, mult in (("tb", "tb", 1.0), ("tb_dm", "tb", dm), ("dar", "dar", 1.0), ("dar_dm", "dar", dm)):
            ly, rk = column_layers(h, vs, z_mid, vs2, z_rock, base, mult)
            fp, ap = f0_peak(FREQ, af_within(FREQ, ly), f0)
            row[f"A_{key}"] = ap
            row[f"f_{key}"] = fp
        row["k_equiv"] = matching_k(h, vs, z_mid, vs2, z_rock, f0, row["A_dar_dm"])
        rows.append(row)
    return rows


def summarize(rows: list[dict]) -> None:
    def q(name):
        v = np.array([r[name] for r in rows])
        return f"median={np.median(v):.3g}  [min {v.min():.3g}, max {v.max():.3g}]"

    print("\nSobol columns (n=%d), PI=%g, f=%g Hz:" % (len(rows), PI_TF, F_REF))
    print("  R = xi_TB / Dmin (soil):        ", q("R_soil"))
    print("  Dmult:                          ", q("dmult"))
    print("  A_peak ratio Dmult*xiTB / Dmult*Dmin:",
          f"median={np.median([r['A_tb_dm'] / r['A_dar_dm'] for r in rows]):.3f}")
    print("  k on xi_TB matching Dmult*Dmin: ", q("k_equiv"))
    print("  k_equiv / Dmult:                ",
          f"median={np.median([r['k_equiv'] / r['dmult'] for r in rows]):.3f}")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_damping(rows: list[dict], hal: dict) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    ax = axes[0]
    vs_grid = np.logspace(np.log10(100.0), np.log10(3000.0), 300)
    ax.semilogx(vs_grid, 100 * np.array([xi_tb(v) for v in vs_grid]), color=COLORS["tb"], lw=2.2,
                label=r"Taborda–Bielak $\xi_Q(V_s)$")
    sig_implied = 101.325 * (vs_grid / VS_A) ** (1.0 / VS_N)
    for pi, ls in zip(PI_LIST, ("-", "--", ":")):
        ax.semilogx(vs_grid, 100 * compute_darendeli_dmin(sig_implied, PI=pi, OCR=OCR, freq=F_REF),
                    color=COLORS["dar"], ls=ls, lw=1.6, label=f"Darendeli, PI={pi:.0f} (Vs-implied σ'm)")
    ax.set_xlabel(r"$V_s$ (m/s)")
    ax.set_ylabel("Damping ratio (%)")
    ax.set_title(f"(a) Damping vs Vs  [Vs = {VS_A:.0f}(σ'm/pa)$^{{{VS_N}}}$, f={F_REF:g} Hz]", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5)

    ax = axes[1]
    vs1 = np.array([r["vs1"] for r in rows])
    H = np.array([r["H"] for r in rows])
    for pi, mk in zip(PI_LIST, ("o", "s", "^")):
        R = np.array([r["xi_tb_soil"] / r[f"dmin_soil_pi{pi:.0f}"] for r in rows])
        sc = ax.scatter(vs1, R, c=H, cmap="viridis", marker=mk, s=26, edgecolor="k", lw=0.3, label=f"PI={pi:.0f}")
    ax.axhline(1.0, color="0.4", lw=1)
    ax.set_xlabel(r"Soil $V_{s1}$ (m/s)")
    ax.set_ylabel(r"$R = \xi_Q / D_\mathrm{min}(H/2)$ (soil)")
    ax.set_title("(b) Sobol columns: ξ_Q / Dmin at H/2", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.colorbar(sc, ax=ax, label="H (m)")

    ax = axes[2]
    col = hal["col"]
    z_edges = np.concatenate([[0.0], np.cumsum(col["h"])])
    xi_q = np.array([xi_tb(v) for v in col["vs"]])
    ax.step(100 * np.r_[xi_q, xi_q[-1]], z_edges, where="post", color=COLORS["tb"], lw=2, label=r"$\xi_Q$")
    for pi, ls in zip(PI_LIST, ("-", "--", ":")):
        d = dmin_at(col["z_mid"], PI=pi)
        ax.step(100 * np.r_[d, d[-1]], z_edges, where="post", color=COLORS["dar"], ls=ls, lw=1.5,
                label=f"Dmin PI={pi:.0f}, {F_REF:g} Hz")
    d = dmin_at(col["z_mid"], PI=0.0, freq=1.0)
    ax.step(100 * np.r_[d, d[-1]], z_edges, where="post", color="0.5", lw=1.2, label="Dmin PI=0, 1 Hz")
    ax.invert_yaxis()
    ax.set_xlabel("Damping ratio (%)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(f"(c) Hallal column (GWT={GWT:g} m, K0={K0:g})", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7.5)

    out = OUT / "dmin_vs_xiQ.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_hallal_tf(hal: dict) -> Path:
    fig, ax = plt.subplots(figsize=(8.4, 5.4), constrained_layout=True)
    ax.plot(hal["raw"][:, 0], hal["raw"][:, 1], ".", color="0.8", ms=2.5, alpha=0.5, label="Hallal raw")
    ax.loglog(hal["f_h"], hal["a_h"], color="0.1", lw=2.2, label="Hallal cleaned")
    dm = hal["dm"]
    labels = {
        "tb": r"$\xi_Q$",
        "tb_dm": rf"{dm:.1f}$\,\xi_Q$ (current)",
        "dar": r"$D_\mathrm{min}$",
        "dar_dm": rf"{dm:.1f}$\,D_\mathrm{{min}}$ (Tao & Rathje / Dawadi)",
    }
    for key, af in hal["curves"].items():
        ls = "--" if key.endswith("_dm") else "-"
        ax.loglog(FREQ, af, color=COLORS[key], ls=ls, lw=1.7,
                  label=f"{labels[key]}  (ln-RMS {hal['misfit'][key]:.2f})")
    ax.set_xlim(0.4, 12)
    ax.set_ylim(0.8, 200)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"$|AF_\mathrm{within}|$")
    ax.set_title(f"Hallal column: damping base × Dmult (PI={PI_TF:g}, {F_REF:g} Hz)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    out = OUT / "hallal_tf_damping_bases.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_sobol(rows: list[dict]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    contrast = np.array([r["contrast"] for r in rows])

    ax = axes[0]
    a_tb = np.array([r["A_tb_dm"] for r in rows])
    a_dar = np.array([r["A_dar_dm"] for r in rows])
    ax.scatter(a_dar, a_tb, c=contrast, cmap="plasma", s=28, edgecolor="k", lw=0.3)
    lim = [min(a_tb.min(), a_dar.min()) * 0.9, max(a_tb.max(), a_dar.max()) * 1.1]
    ax.plot(lim, lim, color="0.4", lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$A_{f_0}$, Dmult$\cdot D_\mathrm{min}$")
    ax.set_ylabel(r"$A_{f_0}$, Dmult$\cdot\xi_Q$ (current)")
    ax.set_title("(a) f0 peak amplitude", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    for key, mk in (("tb", "o"), ("tb_dm", "s"), ("dar", "^"), ("dar_dm", "D")):
        ax.scatter(contrast, [r[f"A_{key}"] for r in rows], color=COLORS[key], marker=mk, s=22,
                   edgecolor="k", lw=0.3, label=key)
    ax.set_yscale("log")
    ax.set_xlabel(r"$V_{s2}/V_{s1}$")
    ax.set_ylabel(r"$A_{f_0}$")
    ax.set_title("(b) f0 peak vs contrast", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.scatter(contrast, [r["dmult"] for r in rows], color="0.3", s=22, label="Dmult (on Dmin)")
    ax.scatter(contrast, [r["k_equiv"] for r in rows], color=COLORS["tb"], s=22, edgecolor="k", lw=0.3,
               label=r"$k$ on $\xi_Q$ matching Dmult$\cdot D_\mathrm{min}$")
    ax.set_xlabel(r"$V_{s2}/V_{s1}$")
    ax.set_ylabel("Multiplier")
    ax.set_yscale("log")
    ax.set_title("(c) Equivalent multiplier on ξ_Q", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    out = OUT / "sobol_dmult_bases.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> None:
    import argparse

    global F_REF, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--f-ref", type=float, default=F_REF, help="Darendeli excitation frequency (Hz)")
    F_REF = parser.parse_args().f_ref
    OUT = OUT / f"f{F_REF:g}Hz"
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Assumptions: rho={RHO:g}  GWT={GWT:g} m  K0={K0:g}  OCR={OCR:g}  f_ref={F_REF:g} Hz  PI_TF={PI_TF:g}")

    mod = load_extracted()
    ly, rk, _ = mod.build_column()
    ref = layered_transfer_function(FREQ[::50], ly, rk)[1]
    assert np.allclose(af_within(FREQ[::50], ly), ref, rtol=1e-10), "vectorized TF mismatch"

    hal = run_hallal(mod)
    rows = run_sobol(load_sobol())
    summarize(rows)

    csv_out = OUT / "sobol_dmult_bases.csv"
    with csv_out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    for p in (fig_damping(rows, hal), fig_hallal_tf(hal), fig_sobol(rows), csv_out):
        print(p)


if __name__ == "__main__":
    main()
