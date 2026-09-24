"""1D theoretical TF for the extracted multilayer Vs profile."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from seiskit.damping import compute_damping_from_Q, compute_quality_factor
from seiskit.plot_config import apply_style
from seiskit.theory.layered_1d_tf import Layer, RockHalfspace, layered_transfer_function

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "rayleigh_equivalent"
HALLAL_RAW = ROOT / "hallal_digitized.csv"

# Layer, z_bot (m), h (m), Vs (m/s), rho (kg/m^3). Last row is the rock halfspace.
TABLE = np.array(
    [
        [1, 10.59, 10.59, 139.19, 1800.0],
        [2, 28.87, 18.28, 174.73, 1500.0],
        [3, 39.99, 11.12, 305.03, 1600.0],
        [4, 71.89, 31.90, 254.69, 1800.0],
        [5, 88.30, 16.41, 334.65, 1900.0],
        [6, 104.00, 15.70, 2298.12, 2200.0],
    ]
)


def build_column():
    layers = []
    rows = []
    z_top = 0.0
    for n, z_bot, h, vs, rho in TABLE[:-1]:
        xi = float(compute_damping_from_Q(compute_quality_factor(float(vs))))
        layers.append(Layer(float(h), float(vs), float(rho), xi))
        rows.append((int(n), z_top, float(z_bot), float(h), float(vs), float(rho), xi))
        z_top = float(z_bot)
    n, z_bot, h, vs, rho = TABLE[-1]
    xi = float(compute_damping_from_Q(compute_quality_factor(float(vs))))
    rock = RockHalfspace(float(vs), float(rho), xi)
    rows.append((int(n), z_top, float(z_bot), float(h), float(vs), float(rho), xi))
    return layers, rock, rows


def travel_time_f0(layers) -> tuple[float, float]:
    tt = sum(L.H / L.Vs for L in layers)
    return tt, 1.0 / (4.0 * tt)


def clean_digitized_tf(
    f: np.ndarray,
    af: np.ndarray,
    n_bin: int = 420,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort, take the upper envelope in log-f bins, fill 1-bin notches."""
    f = np.asarray(f, dtype=float)
    af = np.asarray(af, dtype=float)
    ok = np.isfinite(f) & np.isfinite(af) & (f > 0.0) & (af > 0.0)
    f, af = f[ok], af[ok]
    order = np.argsort(f)
    f, af = f[order], af[order]

    edges = np.logspace(np.log10(f[0]), np.log10(f[-1]), n_bin + 1)
    fc = np.sqrt(edges[:-1] * edges[1:])
    idx = np.clip(np.digitize(f, edges) - 1, 0, n_bin - 1)
    env = np.full(n_bin, np.nan)
    for i in range(n_bin):
        sel = af[idx == i]
        if sel.size:
            env[i] = float(np.max(sel))

    filled = env.copy()
    for i in range(1, n_bin - 1):
        if np.isfinite(env[i - 1]) and np.isfinite(env[i + 1]):
            neighbor = max(env[i - 1], env[i + 1])
            if (not np.isfinite(env[i])) or (env[i] < 0.45 * neighbor and neighbor > 8.0):
                filled[i] = neighbor

    good = np.isfinite(filled)
    logf = np.log(fc)
    cleaned = np.exp(np.interp(logf, logf[good], np.log(filled[good])))
    return fc, cleaned


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    layers, rock, rows = build_column()
    tt, f0 = travel_time_f0(layers)
    z_soil = rows[-2][2]
    z_rock = rows[-1][2]

    freq = np.logspace(np.log10(0.1), np.log10(25.0), 2500)
    _, af_w, af_o = layered_transfer_function(freq, layers, rock)
    band = freq < 3.0 * f0
    i1 = int(np.argmax(af_w[band]))
    f_peak = float(freq[band][i1])
    a_peak = float(af_w[band][i1])
    i1o = int(np.argmax(af_o[band]))
    f_peak_o = float(freq[band][i1o])
    a_peak_o = float(af_o[band][i1o])

    raw = np.loadtxt(HALLAL_RAW, delimiter=",", skiprows=1)
    f_h, a_h = clean_digitized_tf(raw[:, 0], raw[:, 1])
    cleaned_path = OUT / "hallal_digitized_cleaned.csv"
    np.savetxt(
        cleaned_path,
        np.column_stack([f_h, a_h]),
        delimiter=",",
        header="freq_hz,af",
        comments="",
    )
    ih = int(np.argmax(a_h))
    print(
        f"Hallal cleaned: n_raw={len(raw)}  n_clean={len(f_h)}  "
        f"peak={a_h[ih]:.1f} at {f_h[ih]:.3f} Hz"
    )
    print(
        f"Theory: |AF_within|={a_peak:.2f} at {f_peak:.3f} Hz  "
        f"|AF_outcrop|={a_peak_o:.2f} at {f_peak_o:.3f} Hz"
    )

    print("layer  z_top    z_bot     H      Vs     rho     xi_Q")
    for n, zt, zb, h, vs, rho, xi in rows:
        tag = "rock" if n == rows[-1][0] else f"{n:4d}"
        print(f"{tag:>5s} {zt:8.2f} {zb:8.2f} {h:7.2f} {vs:8.2f} {rho:6.0f} {100*xi:6.2f}%")
    print(
        f"tt={tt:.4f} s  f0=1/(4tt)={f0:.3f} Hz  "
        f"first peak f={f_peak:.3f} Hz  |AF_within|={a_peak:.2f}"
    )

    z_stair, vs_stair = [], []
    for n, zt, zb, h, vs, rho, xi in rows:
        z_stair.extend([zt, zb])
        vs_stair.extend([vs, vs])

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.2), constrained_layout=True)

    ax = axes[0]
    ax.plot(vs_stair, z_stair, color="#0072B2", lw=2.0)
    ax.plot([r[4] for r in rows], [r[2] for r in rows], "o", color="#0072B2", ms=5, zorder=3)
    ax.axhline(z_soil, color="0.5", ls=":", lw=0.9)
    for n, zt, zb, h, vs, rho, xi in rows:
        ax.annotate(
            fr"$\rho$={rho:.0f}",
            xy=(min(vs, 700.0), 0.5 * (zt + zb)),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=7,
            va="center",
            color="0.35",
        )
    ax.set_ylim(z_rock, 0.0)
    ax.set_xlim(0.0, 2500.0)
    ax.set_xlabel(r"$V_s$ (m/s)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Layered column")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(
        raw[:, 0],
        raw[:, 1],
        ".",
        color="0.75",
        ms=3.0,
        alpha=0.55,
        zorder=2,
        label="Hallal digitized (raw)",
    )
    ax.loglog(f_h, a_h, color="0.15", lw=2.15, zorder=3, label="Hallal digitized (cleaned)")
    ax.loglog(freq, af_w, color="#0072B2", lw=1.8, zorder=4, label=r"This work $AF_\mathrm{within}$")
    ax.loglog(freq, af_o, color="#D55E00", lw=1.35, ls="--", zorder=4, label=r"This work $AF_\mathrm{outcrop}$")
    ax.axvline(f0, color="0.4", ls=":", lw=0.8)
    ax.plot(f_peak, a_peak, "o", color="#0072B2", ms=5, zorder=5)
    ax.plot(f_h[ih], a_h[ih], "o", color="0.15", ms=5, zorder=5)
    ax.annotate(
        fr"Hallal $A_1$={a_h[ih]:.0f}" + "\n" + fr"here $A_1$={a_peak:.1f}",
        xy=(f_h[ih], a_h[ih]),
        xytext=(1.55, a_h[ih] * 0.55),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="0.3"),
    )
    ax.set_xlim(0.4, 12)
    ax.set_ylim(0.8, 150)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"$|AF|$")
    ax.set_title(r"Thomson–Haskell $\xi_Q$ vs Hallal")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=7.5, frameon=True)

    fig.suptitle(
        r"Multilayer 1D TF  (Taborda–Bielak $Q(V_s)$, tabulated $\rho$)",
        fontsize=11,
    )
    out = OUT / "af_extracted_profile.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
