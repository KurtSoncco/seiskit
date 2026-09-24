"""Hallal-paper Dmin profile vs Darendeli (1 / 3 Hz) and Taborda–Bielak ξ_Q.

Left: Dmin vs depth for the Hallal 5-layer column. Middle / right: 1D
|AF_within| against the digitized Hallal TF, without and with Dmult.
Reuses the column, TF and Darendeli helpers of ``compare_darendeli_taborda``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from seiskit.plot_config import apply_style
from seiskit.theory.layered_1d_tf import Layer

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "darendeli_taborda"

# Digitized Dmin (%) vs depth (m) from the Hallal paper (Hallal column).
HALLAL_DMIN = np.array(
    [
        [1.4989733059548258, 2.517482517482456],
        [1.057494866529772, 10.42957042957039],
        [0.8726899383983562, 28.651348651348606],
        [0.7186858316221776, 39.92007992007991],
        [0.6160164271047206, 71.8081918081918],
        [0.55441478439425, 87.63236763236765],
        [0.48254620123203296, 119.88011988011971],
    ]
)

COLORS = {
    "hallal": "#CC79A7",
    "dar1": "#009E73",
    "dar3": "#D55E00",
    "tb": "#0072B2",
}


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / fname)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def hallal_dmin_at(z: np.ndarray) -> np.ndarray:
    """Log–log interpolation of the digitized profile (power-law extrapolation at the ends)."""
    x, zd = HALLAL_DMIN[:, 0] / 100.0, HALLAL_DMIN[:, 1]
    b, la = np.polyfit(np.log(zd), np.log(x), 1)
    z = np.asarray(z, dtype=float)
    inner = np.exp(np.interp(np.log(z), np.log(zd), np.log(x)))
    outer = np.exp(la) * z**b
    return np.where((z < zd[0]) | (z > zd[-1]), outer, inner)


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    cmp = _load("compare_darendeli_taborda", "compare_darendeli_taborda.py")
    ext = _load("extracted_profile", "plot_extracted_profile.py")
    hal = cmp.run_hallal(ext)
    col, dm = hal["col"], hal["dm"]
    freq = cmp.FREQ

    xi = {
        "hallal": hallal_dmin_at(col["z_mid"]),
        "dar1": cmp.dmin_at(col["z_mid"], freq=1.0),
        "dar3": cmp.dmin_at(col["z_mid"], freq=3.0),
        "tb": np.array([cmp.xi_tb(v) for v in col["vs"]]),
    }
    labels = {
        "hallal": r"Hallal paper $D_\mathrm{min}$",
        "dar1": r"Darendeli, 1 Hz",
        "dar3": r"Darendeli, 3 Hz",
        "tb": r"Taborda–Bielak $\xi_Q$",
    }

    band = (hal["f_h"] > 0.5) & (hal["f_h"] < 10.0)

    def tf_and_misfit(x: np.ndarray, mult: float) -> tuple[np.ndarray, float]:
        layers = [Layer(float(h), float(v), cmp.RHO, float(mult * xx)) for h, v, xx in zip(col["h"], col["vs"], x)]
        af = cmp.af_within(freq, layers)
        af_i = np.exp(np.interp(np.log(hal["f_h"][band]), np.log(freq), np.log(af)))
        return af, float(np.sqrt(np.mean((np.log(af_i) - np.log(hal["a_h"][band])) ** 2)))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), constrained_layout=True, gridspec_kw={"width_ratios": [0.7, 1, 1]})

    ax = axes[0]
    z_fine = np.linspace(1.0, 125.0, 300)
    ax.plot(HALLAL_DMIN[:, 0], HALLAL_DMIN[:, 1], "o", color=COLORS["hallal"], ms=7, zorder=5, label=labels["hallal"])
    ax.plot(100 * hallal_dmin_at(z_fine), z_fine, color=COLORS["hallal"], lw=1, ls=":", label="interp. (log–log)")
    for key, f in (("dar1", 1.0), ("dar3", 3.0)):
        ax.plot(100 * cmp.dmin_at(z_fine, freq=f), z_fine, color=COLORS[key], lw=1.8, label=labels[key])
    z_edges = np.concatenate([[0.0], np.cumsum(col["h"])])
    ax.step(100 * np.r_[xi["tb"], xi["tb"][-1]], z_edges, where="post", color=COLORS["tb"], lw=1.8, label=labels["tb"])
    ax.axhline(col["H"], color="0.5", lw=0.8, ls="--")
    ax.text(0.05, col["H"] - 1.5, "soil / rock", fontsize=8, color="0.4")
    ax.invert_yaxis()
    ax.set_xlim(0, 5)
    ax.set_xlabel("Damping ratio (%)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("(a) Hallal column damping profiles", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    for ax, mult, title in (
        (axes[1], 1.0, "(b) Base damping (no Dmult)"),
        (axes[2], dm, f"(c) Dmult × base  (Dmult = {dm:.1f})"),
    ):
        ax.plot(hal["raw"][:, 0], hal["raw"][:, 1], ".", color="0.82", ms=2.3, alpha=0.5)
        ax.loglog(hal["f_h"], hal["a_h"], color="0.35", lw=2.6, alpha=0.6, label="Hallal TF (digitized)")
        for key in ("hallal", "dar1", "dar3", "tb"):
            af, mis = tf_and_misfit(xi[key], mult)
            ls, lw, zo = ((0, (4, 2)), 2.2, 6) if key == "hallal" else ("-", 1.4, 4)
            ax.loglog(freq, af, color=COLORS[key], lw=lw, ls=ls, zorder=zo, label=f"{labels[key]}  (ln-RMS {mis:.2f})")
            print(f"[{'x' + format(mult, '.1f')}] {key:6s} ln-RMS misfit = {mis:.3f}")
        ax.set_xlim(0.4, 12)
        ax.set_ylim(0.8, 200)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel(r"$|AF_\mathrm{within}|$")
        ax.set_title(title, fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")

    out = OUT / "hallal_dmin_comparison.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
