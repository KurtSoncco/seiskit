"""Check whether Hallal Fig. 6 amplitude offset is just too much ξ_Q."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from seiskit.plot_config import apply_style
from seiskit.theory.layered_1d_tf import Layer, RockHalfspace, layered_transfer_function

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "rayleigh_equivalent"


def load_extracted():
    path = ROOT / "plot_extracted_profile.py"
    spec = importlib.util.spec_from_file_location("extracted_profile", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["extracted_profile"] = mod
    spec.loader.exec_module(mod)
    return mod


def column_with_xi(mod, scale: float | None, xi_const: float | None):
    rows_q = mod.build_column()[2]
    layers = []
    for n, zt, zb, h, vs, rho, xi_q in rows_q[:-1]:
        xi = float(xi_const) if xi_const is not None else float(scale) * xi_q
        layers.append(Layer(h, vs, rho, xi))
    n, zt, zb, h, vs, rho, xi_q = rows_q[-1]
    xi_r = float(xi_const) if xi_const is not None else float(scale) * xi_q
    rock = RockHalfspace(vs, rho, xi_r)
    return layers, rock, rows_q


def local_peaks(freq, af, fmin=0.5, fmax=8.0, order=12):
    from scipy.signal import find_peaks

    m = (freq >= fmin) & (freq <= fmax)
    f, a = freq[m], af[m]
    idx, _ = find_peaks(a, distance=order)
    return [(float(f[i]), float(a[i])) for i in idx]


def nearest_peak(peaks, f_target):
    f, a = min(peaks, key=lambda p: abs(p[0] - f_target))
    return f, a


def interp_af(freq, af, f_h):
    return np.exp(np.interp(np.log(f_h), np.log(freq), np.log(np.clip(af, 1e-12, None))))


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    mod = load_extracted()
    layers_q, rock_q, rows_q = mod.build_column()
    tt, f0 = mod.travel_time_f0(layers_q)

    xi_q = np.array([r[6] for r in rows_q[:-1]])
    h = np.array([r[3] for r in rows_q[:-1]])
    vs = np.array([r[4] for r in rows_q[:-1]])
    tt_i = h / vs
    xi_tt = float(np.sum(tt_i * xi_q) / np.sum(tt_i))
    xi_h = float(np.sum(h * xi_q) / np.sum(h))

    freq = np.logspace(np.log10(0.4), np.log10(12.0), 4000)
    cases = {
        r"$\xi_Q$": (1.0, None),
        r"$0.25\,\xi_Q$": (0.25, None),
        r"$\xi=1.1\%$": (None, 0.011),
        r"$\xi=0.75\%$": (None, 0.0075),
    }
    curves = {}
    for name, (scale, xi_c) in cases.items():
        ly, rk, _ = column_with_xi(mod, scale, xi_c)
        _, aw, ao = layered_transfer_function(freq, ly, rk)
        curves[name] = (aw, ao)

    raw = np.loadtxt(mod.HALLAL_RAW, delimiter=",", skiprows=1)
    f_h, a_h = mod.clean_digitized_tf(raw[:, 0], raw[:, 1])
    peaks_h = local_peaks(f_h, a_h)
    targets = [0.78, 1.86, 3.34, 5.73]

    print(f"travel-time f0={f0:.3f} Hz")
    print(
        f"xi_Q top={100*xi_q[0]:.2f}%  thickness-wtd={100*xi_h:.2f}%  "
        f"travel-time-wtd={100*xi_tt:.2f}%"
    )
    print(f"2/(pi xi_tt)={2.0/(np.pi*xi_tt):.1f}   2/(pi*0.25*xi_tt)={2.0/(np.pi*0.25*xi_tt):.1f}")
    print()
    print(f"{'f_H':>7} {'Hallal':>7} {'xiQ':>7} {'ratio':>6} {'0.25Q':>7} {'1.1%':>7} {'0.75%':>7}")
    for ft in targets:
        fh, ah = nearest_peak(peaks_h, ft)
        row = [f"{fh:7.2f}", f"{ah:7.1f}"]
        for name in (r"$\xi_Q$", r"$0.25\,\xi_Q$", r"$\xi=1.1\%$", r"$\xi=0.75\%$"):
            pk = local_peaks(freq, curves[name][0])
            _, at = nearest_peak(pk, fh)
            if name == r"$\xi_Q$":
                row.append(f"{at:7.1f}")
                row.append(f"{ah/at:6.2f}")
            else:
                row.append(f"{at:7.1f}")
        print(" ".join(row))

    # scale that matches Hallal A1 on AF_within
    a1_h = nearest_peak(peaks_h, 0.78)[1]
    scales = np.linspace(0.12, 0.45, 34)
    a1s = []
    for s in scales:
        ly, rk, _ = column_with_xi(mod, s, None)
        _, aw, _ = layered_transfer_function(np.array([0.78]), ly, rk)
        # use a small grid around the peak
        fg = np.linspace(0.72, 0.86, 200)
        _, aw, _ = layered_transfer_function(fg, ly, rk)
        a1s.append(float(np.max(aw)))
    a1s = np.array(a1s)
    s_best = float(np.interp(a1_h, a1s[::-1], scales[::-1]))  # a1 decreases with s
    print(f"\nscale of xi_Q matching Hallal A1={a1_h:.1f}:  s={s_best:.3f}  "
          f"top xi={100*s_best*xi_q[0]:.2f}%  tt-wtd xi={100*s_best*xi_tt:.2f}%")

    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    ax.plot(raw[:, 0], raw[:, 1], ".", color="0.78", ms=2.8, alpha=0.5, zorder=1, label="Hallal raw")
    ax.loglog(f_h, a_h, color="0.12", lw=2.3, zorder=3, label="Hallal cleaned")
    ax.loglog(freq, curves[r"$\xi_Q$"][0], color="#0072B2", lw=1.8, zorder=4, label=r"$AF_\mathrm{within},\ \xi_Q$")
    ax.loglog(
        freq,
        curves[r"$0.25\,\xi_Q$"][0],
        color="#D55E00",
        lw=1.9,
        ls="--",
        zorder=5,
        label=r"$AF_\mathrm{within},\ 0.25\,\xi_Q$",
    )
    ax.loglog(
        freq,
        curves[r"$\xi=1.1\%$"][0],
        color="#009E73",
        lw=1.35,
        ls=":",
        zorder=4,
        label=r"$AF_\mathrm{within},\ \xi=1.1\%$ const.",
    )
    ax.set_xlim(0.45, 11)
    ax.set_ylim(0.8, 160)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"$|AF_\mathrm{within}|$")
    ax.set_title("Is the Hallal offset just damping?")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower left", fontsize=8, frameon=True)

    out = OUT / "af_hallal_dmin_check.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
