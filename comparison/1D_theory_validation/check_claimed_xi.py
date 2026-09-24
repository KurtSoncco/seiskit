"""Verify the claimed per-layer hysteretic ξ (Taborda–Bielak shape × 0.25)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from seiskit.plot_config import apply_style
from seiskit.theory.layered_1d_tf import Layer, RockHalfspace, layered_transfer_function

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "rayleigh_equivalent"

# Claimed fitted ξ, same order as soil layers 1–5.
XI_CLAIMED = np.array([0.0116, 0.0107, 0.0074, 0.0086, 0.0067])
CLAIMED_RATIOS = (1.00, 0.94, 0.84, 0.97, 0.85, 0.76)


def load_extracted():
    path = ROOT / "plot_extracted_profile.py"
    spec = importlib.util.spec_from_file_location("extracted_profile", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["extracted_profile"] = mod
    spec.loader.exec_module(mod)
    return mod


def make_column(rows, xi_soil, xi_rock):
    layers = []
    for xi, row in zip(xi_soil, rows[:-1]):
        _, _, _, h, vs, rho, _ = row
        layers.append(Layer(h, vs, rho, float(xi)))
    _, _, _, _, vs_r, rho_r, _ = rows[-1]
    rock = RockHalfspace(vs_r, rho_r, float(xi_rock))
    return layers, rock


def extrema(freq, af, fmin, fmax, *, valleys=False):
    m = (freq >= fmin) & (freq <= fmax)
    y = -af[m] if valleys else af[m]
    idx, _ = find_peaks(y, prominence=0.15, distance=10)
    return [(float(freq[m][i]), float(af[m][i])) for i in idx]


def nearest(pts, f_target):
    return min(pts, key=lambda p: abs(p[0] - f_target))


def band_max(freq, af, f0, rel=0.08):
    m = (freq >= (1.0 - rel) * f0) & (freq <= (1.0 + rel) * f0)
    if not np.any(m):
        i = int(np.argmin(np.abs(freq - f0)))
        return float(freq[i]), float(af[i])
    i = int(np.argmax(af[m]))
    return float(freq[m][i]), float(af[m][i])


def band_min(freq, af, f0, rel=0.12):
    m = (freq >= (1.0 - rel) * f0) & (freq <= (1.0 + rel) * f0)
    i = int(np.argmin(af[m]))
    return float(freq[m][i]), float(af[m][i])


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    mod = load_extracted()
    rows = mod.build_column()[2]
    xi_q = np.array([r[6] for r in rows[:-1]])
    xi_q_r = rows[-1][6]
    xi_025 = 0.25 * xi_q

    print("layer   Vs     ξ_claimed   ξ_Q    0.25 ξ_Q   claimed/0.25Q")
    for i, row in enumerate(rows[:-1]):
        print(
            f"{i+1:5d} {row[4]:7.1f} {100*XI_CLAIMED[i]:8.3f}% "
            f"{100*xi_q[i]:6.2f}% {100*xi_025[i]:8.3f}% "
            f"{XI_CLAIMED[i]/xi_025[i]:8.3f}"
        )
    print(
        f" rock {rows[-1][4]:7.1f}      —     "
        f"{100*xi_q_r:6.2f}% {100*0.25*xi_q_r:8.3f}%"
    )

    raw = np.loadtxt(mod.HALLAL_RAW, delimiter=",", skiprows=1)
    f_h, a_h = mod.clean_digitized_tf(raw[:, 0], raw[:, 1])
    pk_h = extrema(f_h, a_h, 0.55, 10.0, valleys=False)
    vl_h = extrema(f_h, a_h, 0.9, 10.0, valleys=True)
    print("\nHallal peaks:", [f"{f:.2f}/{a:.1f}" for f, a in pk_h])
    print("Hallal valleys:", [f"{f:.2f}/{a:.2f}" for f, a in vl_h])

    freq = np.logspace(np.log10(0.45), np.log10(11.0), 3500)
    layers, rock = make_column(rows, XI_CLAIMED, 0.25 * xi_q_r)
    _, af, _ = layered_transfer_function(freq, layers, rock)
    layers025, rock025 = make_column(rows, xi_025, 0.25 * xi_q_r)
    _, af025, _ = layered_transfer_function(freq, layers025, rock025)

    print("\nPEAKS  (ratio = model / Hallal)")
    print(f"{'f_H':>7} {'Hallal':>8} {'claimed':>8} {'ratio':>7} {'0.25Q':>8} {'ratio':>7}  claimed-note")
    peak_ratios = []
    for k, (fh, ah) in enumerate(pk_h):
        fm, am = band_max(freq, af, fh)
        f2, a2 = band_max(freq, af025, fh)
        r = am / ah
        peak_ratios.append(r)
        note = ""
        if k < len(CLAIMED_RATIOS):
            note = f"  stated {CLAIMED_RATIOS[k]:.2f}"
        print(f"{fh:7.2f} {ah:8.1f} {am:8.1f} {r:7.2f} {a2:8.1f} {a2/ah:7.2f}{note}")

    print("\nVALLEYS between consecutive Hallal peaks (min in each interval)")
    print(f"{'interval':>15} {'Hallal':>8} {'claimed':>8} {'ratio':>7} {'err%':>7}")
    valley_errs = []
    for (f1, _), (f2, _) in zip(pk_h, pk_h[1:]):
        mh = (f_h > f1) & (f_h < f2)
        mm = (freq > f1) & (freq < f2)
        ah = float(np.min(a_h[mh]))
        fh = float(f_h[mh][np.argmin(a_h[mh])])
        am = float(np.min(af[mm]))
        r = am / ah
        err = 100 * abs(r - 1.0)
        valley_errs.append(err)
        print(f"{f1:5.2f}–{f2:5.2f} {ah:8.2f} {am:8.2f} {r:7.2f} {err:6.1f}%  (f≈{fh:.2f})")
    n_ok = sum(e <= 15.0 for e in valley_errs)
    print(f"valleys within 15%: {n_ok}/{len(valley_errs)}")

    a1_h = pk_h[0][1]
    a1_m = band_max(freq, af, pk_h[0][0])[1]
    print(f"\nFundamental: Hallal {a1_h:.1f}  claimed {a1_m:.1f}  rel {(a1_m/a1_h-1)*100:+.2f}%")
    print("Stated peak ratios:", CLAIMED_RATIOS)
    print("Computed peak ratios:", tuple(round(r, 2) for r in peak_ratios[:6]))

    fig, ax = plt.subplots(figsize=(8.4, 5.3), constrained_layout=True)
    ax.plot(raw[:, 0], raw[:, 1], ".", color="0.78", ms=2.4, alpha=0.4, zorder=1)
    ax.loglog(f_h, a_h, color="0.12", lw=2.3, zorder=3, label="Hallal cleaned")
    ax.loglog(freq, af, color="#D55E00", lw=1.9, zorder=4, label=r"claimed per-layer $\xi$")
    ax.loglog(freq, af025, color="#0072B2", lw=1.2, ls="--", zorder=3, label=r"$0.25\,\xi_Q$")
    ax.set_xlim(0.45, 11)
    ax.set_ylim(0.8, 160)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"$|AF_\mathrm{within}|$")
    ax.set_title(r"Check: Taborda–Bielak-shaped $\xi$ scaled to the fundamental")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=8, frameon=True)
    out = OUT / "af_claimed_xi_check.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
