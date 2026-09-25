"""Taborda–Bielak ξ_Q(Vs) and first-mode |AF_within| vs soil Vs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from seiskit.damping import compute_damping_from_Q, compute_quality_factor
from seiskit.plot_config import apply_style
from seiskit.theory.layered_1d_tf import Layer, RockHalfspace, layered_transfer_function

OUT = Path(__file__).resolve().parent / "results" / "rayleigh_equivalent"
H = 15.0
VS_ROCK = 1500.0
RHO = 2000.0
VS_MARK = 230.0


def first_mode_peak(vs: float, xi: float) -> float:
    f0 = vs / (4.0 * H)
    freq = np.linspace(0.7 * f0, 1.3 * f0, 400)
    _, af, _ = layered_transfer_function(
        freq,
        [Layer(H, vs, RHO, xi)],
        RockHalfspace(VS_ROCK, RHO, 0.0),
    )
    return float(np.max(af))


def main() -> None:
    apply_style()
    vs = np.linspace(100.0, 400.0, 121)
    q = np.array([compute_quality_factor(v) for v in vs])
    xi = compute_damping_from_Q(q)
    a_th = np.array([first_mode_peak(v, z) for v, z in zip(vs, xi)])
    a_approx = 2.0 / (np.pi * xi)

    q_m = compute_quality_factor(VS_MARK)
    xi_m = compute_damping_from_Q(q_m)
    a_m = first_mode_peak(VS_MARK, xi_m)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True)

    ax = axes[0]
    ax.plot(vs, 100 * xi, color="#0072B2", lw=2.0, label=r"$\xi_Q=1/(2Q(V_s))$")
    ax.axhline(2.5, color="#D55E00", ls=":", lw=1.4, label=r"fixed $\xi$=2.5%")
    ax.axvline(VS_MARK, color="0.4", ls="--", lw=0.9)
    ax.plot(VS_MARK, 100 * xi_m, "o", color="#0072B2", ms=7)
    ax.annotate(
        fr"$V_s$=230 m/s" + "\n" + fr"$\xi_Q$={100*xi_m:.2f}%",
        xy=(VS_MARK, 100 * xi_m),
        xytext=(250, 4.4),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="0.3"),
    )
    ax.set_xlabel(r"Soil $V_s$ (m/s)")
    ax.set_ylabel(r"Damping ratio $\xi_Q$ (%)")
    ax.set_title("Taborda–Bielak $Q(V_s)$")
    ax.set_xlim(100, 400)
    ax.set_ylim(2.0, 5.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    ax.plot(vs, a_th, color="#0072B2", lw=2.0, label=r"Thomson–Haskell, $\xi=\xi_Q(V_s)$")
    ax.plot(vs, a_approx, color="0.35", ls="--", lw=1.4, label=r"$2/(\pi\xi_Q)$ (rigid-rock)")
    ax.axhline(2.0 / (np.pi * 0.025), color="#D55E00", ls=":", lw=1.4, label=r"$2/(\pi\cdot 2.5\%)\approx 25.5$")
    ax.axvline(VS_MARK, color="0.4", ls="--", lw=0.9)
    ax.plot(VS_MARK, a_m, "o", color="#0072B2", ms=7)
    ax.annotate(
        fr"$A_1$={a_m:.1f}",
        xy=(VS_MARK, a_m),
        xytext=(255, 22),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="0.3"),
    )
    ax.set_xlabel(r"Soil $V_s$ (m/s)")
    ax.set_ylabel(r"First-mode $|AF_\mathrm{within}|$")
    ax.set_title(fr"Theoretical peak  ($H$={H:.0f} m, $V_{{s2}}$={VS_ROCK:.0f} m/s)")
    ax.set_xlim(100, 400)
    ax.set_ylim(10, 30)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

    fig.suptitle(r"Velocity-dependent damping from $Q(V_s)$ and the resulting first-mode TF peak", fontsize=11)
    out = OUT / "xiQ_and_A1_vs_vs.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Vs=230  Q={q_m:.2f}  xi={xi_m:.5f}  A1={a_m:.2f}  2/(pi xi)={2/(np.pi*xi_m):.2f}")
    print(f"Vs=100  xi={100*xi[0]:.2f}%  A1={a_th[0]:.2f}")
    print(f"Vs=400  xi={100*xi[-1]:.2f}%  A1={a_th[-1]:.2f}")
    print(out)


if __name__ == "__main__":
    main()
