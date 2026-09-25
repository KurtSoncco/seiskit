"""Hallal-paper Dmin profile vs Darendeli (1 / 3 Hz) and Taborda–Bielak ξ_Q.

The Hallal Dmin is a step profile: each digitized point is the bottom of a
layer. The column uses Hallal's layering (layer 1 split at 2.5 m; boundaries
snapped to the tabulated ones). Darendeli and ξ_Q are constant per layer, at
the layer mid-depth (thin lines in panel (a): continuous Darendeli curves).

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

# Digitized Hallal-paper step profile: Dmin (%) and the depth (m) of each layer's bottom.
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


SNAP_TOL = 1.0  # m, snap digitized boundaries to the tabulated layer boundaries


def hallal_layers(table: np.ndarray) -> list[tuple[float, float, float, float]]:
    """(z_top, z_bot, Vs, Dmin) per Hallal layer; the last row is the rock layer."""
    z_tab = table[:, 1]
    z_bot = []
    for z in HALLAL_DMIN[:, 1]:
        k = int(np.argmin(np.abs(z_tab - z)))
        z_bot.append(float(z_tab[k]) if abs(z_tab[k] - z) < SNAP_TOL else float(z))
    z_top = [0.0] + z_bot[:-1]
    rows = []
    for zt, zb, d in zip(z_top, z_bot, HALLAL_DMIN[:, 0] / 100.0):
        k = min(int(np.searchsorted(z_tab, 0.5 * (zt + zb))), len(table) - 1)
        rows.append((zt, zb, float(table[k, 3]), float(d)))
    return rows


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    cmp = _load("compare_darendeli_taborda", "compare_darendeli_taborda.py")
    ext = _load("extracted_profile", "plot_extracted_profile.py")
    rows = hallal_layers(ext.TABLE)
    z_top = np.array([r[0] for r in rows])
    z_bot = np.array([r[1] for r in rows])
    vs = np.array([r[2] for r in rows])
    h, z_mid = z_bot - z_top, 0.5 * (z_top + z_bot)
    soil = slice(0, len(rows) - 1)  # last Hallal layer is rock (AF_within ignores it)
    vs1 = float(np.sum(h[soil]) / np.sum(h[soil] / vs[soil]))
    dm = cmp.dmult_from_vs_contrast(vs1, float(ext.TABLE[-1, 3]))
    freq = cmp.FREQ
    raw = np.loadtxt(ext.HALLAL_RAW, delimiter=",", skiprows=1)
    f_h, a_h = ext.clean_digitized_tf(raw[:, 0], raw[:, 1])

    xi = {
        "hallal": np.array([r[3] for r in rows]),
        "dar1": cmp.dmin_at(z_mid, freq=1.0),
        "dar3": cmp.dmin_at(z_mid, freq=3.0),
        "tb": np.array([cmp.xi_tb(v) for v in vs]),
    }
    labels = {
        "hallal": r"Hallal paper $D_\mathrm{min}$",
        "dar1": r"Darendeli, 1 Hz",
        "dar3": r"Darendeli, 3 Hz",
        "tb": r"Taborda–Bielak $\xi_Q$",
    }
    print(f"Vs1_tt={vs1:.1f}  Dmult={dm:.2f}")
    print(f"{'layer (m)':>14} {'Vs':>6} {'Hallal':>7} {'Dar1Hz':>7} {'Dar3Hz':>7} {'xi_TB':>6}")
    for k in range(len(rows)):
        print(f"{z_top[k]:6.1f}-{z_bot[k]:6.1f} {vs[k]:6.0f} " + " ".join(
            f"{100 * xi[key][k]:7.3f}" for key in ("hallal", "dar1", "dar3", "tb")))

    band = (f_h > 0.5) & (f_h < 10.0)

    def tf_and_misfit(x: np.ndarray, mult: float) -> tuple[np.ndarray, float]:
        layers = [Layer(float(hh), float(v), cmp.RHO, float(mult * xx)) for hh, v, xx in zip(h[soil], vs[soil], x[soil])]
        af = cmp.af_within(freq, layers)
        af_i = np.exp(np.interp(np.log(f_h[band]), np.log(freq), np.log(af)))
        return af, float(np.sqrt(np.mean((np.log(af_i) - np.log(a_h[band])) ** 2)))

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), constrained_layout=True, gridspec_kw={"width_ratios": [0.7, 1, 1]})

    ax = axes[0]
    z_fine = np.linspace(0.5, 125.0, 300)
    z_edges = np.r_[z_top, z_bot[-1]]
    for key, f in (("dar1", 1.0), ("dar3", 3.0)):
        ax.plot(100 * cmp.dmin_at(z_fine, freq=f), z_fine, color=COLORS[key], lw=0.9, alpha=0.5)
    for key in ("dar1", "dar3", "tb"):
        ax.plot(*cmp.depth_steps(100 * xi[key], z_edges), color=COLORS[key], lw=1.8,
                label=labels[key] + (" (layer mid-depth)" if key != "tb" else ""))
    ax.plot(*cmp.depth_steps(100 * xi["hallal"], z_edges), color=COLORS["hallal"],
            lw=2.2, ls=(0, (4, 2)), zorder=6, label=labels["hallal"] + " (step)")
    ax.plot(HALLAL_DMIN[:, 0], HALLAL_DMIN[:, 1], "o", color=COLORS["hallal"], ms=6, zorder=7,
            label="digitized (layer bottoms)")
    ax.axhline(z_top[-1], color="0.5", lw=0.8, ls="--")
    ax.text(3.2, z_top[-1] - 1.5, "soil / rock", fontsize=8, color="0.4")
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
        ax.plot(raw[:, 0], raw[:, 1], ".", color="0.82", ms=2.3, alpha=0.5)
        ax.loglog(f_h, a_h, color="0.35", lw=2.6, alpha=0.6, label="Hallal TF (digitized)")
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
