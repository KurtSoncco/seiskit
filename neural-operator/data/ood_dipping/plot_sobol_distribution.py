"""Plot planned Sobol physics distribution for Tier B ood_dipping.

Axes (7D, n=32): Vs1, H, rH, aHV, dip_angle_deg, CoV, Vs2.
Seed axis (30 levels) shown separately.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm, qmc

THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR.parent
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from sobol import (  # noqa: E402
    DEFAULT_SAMPLER_SEED,
    bounds_CoV,
    bounds_rH,
    scale_Vs1,
    scale_Vs2,
    scale_aHV,
    sigma_Vs1,
    sigma_Vs2,
    sigma_aHV,
)

N_PHYSICS = 32
N_SEEDS = 30
SAMPLER_SEED = DEFAULT_SAMPLER_SEED
RF_SEED_MIN = 1
RF_SEED_MAX = 9_999_999

bounds_H = (25.0, 60.0)  # min 25 m so center stays soil under |dip|≤3°
bounds_dip = (-3.0, 3.0)

AXIS_LABELS = (
    "Vs1 [m/s]",
    "H [m]",
    "rH [m]",
    "aHV [-]",
    "dip [deg]",
    "CoV [-]",
    "Vs2 [m/s]",
)
CSV_COLUMNS = ("sobol_id", "Vs1", "H", "rH", "aHV", "dip_angle_deg", "CoV", "Vs2")
N_DIM = len(AXIS_LABELS)


def unit_to_physical(unit: np.ndarray) -> np.ndarray:
    raw = np.asarray(unit, dtype=float)
    phys = np.zeros_like(raw)
    phys[:, 0] = lognorm.ppf(raw[:, 0], s=sigma_Vs1, scale=scale_Vs1)
    phys[:, 1] = bounds_H[0] + raw[:, 1] * (bounds_H[1] - bounds_H[0])
    phys[:, 2] = bounds_rH[0] + raw[:, 2] * (bounds_rH[1] - bounds_rH[0])
    phys[:, 3] = lognorm.ppf(raw[:, 3], s=sigma_aHV, scale=scale_aHV)
    phys[:, 4] = bounds_dip[0] + raw[:, 4] * (bounds_dip[1] - bounds_dip[0])
    phys[:, 5] = bounds_CoV[0] + raw[:, 5] * (bounds_CoV[1] - bounds_CoV[0])
    phys[:, 6] = lognorm.ppf(raw[:, 6], s=sigma_Vs2, scale=scale_Vs2)
    return phys


def _valid_mask(phys: np.ndarray) -> np.ndarray:
    return (
        (phys[:, 0] >= 100.0)
        & (phys[:, 0] <= 360.0)
        & (phys[:, 1] >= bounds_H[0])
        & (phys[:, 1] <= bounds_H[1])
        & (phys[:, 2] >= bounds_rH[0])
        & (phys[:, 2] <= bounds_rH[1])
        & (phys[:, 3] >= 10.0)
        & (phys[:, 3] <= 50.0)
        & (phys[:, 4] >= bounds_dip[0])
        & (phys[:, 4] <= bounds_dip[1])
        & (phys[:, 5] >= bounds_CoV[0])
        & (phys[:, 5] <= bounds_CoV[1])
        & (phys[:, 6] >= 760.0)
        & (phys[:, 6] <= 1500.0)
    )


def generate_physics_samples(n: int = N_PHYSICS, seed: int = SAMPLER_SEED) -> np.ndarray:
    sampler = qmc.Sobol(d=N_DIM, scramble=True, seed=seed)
    valid: list[np.ndarray] = []
    while len(valid) < n:
        remaining = n - len(valid)
        batch = max(2 ** math.ceil(math.log2(remaining + 32)), 64)
        phys = unit_to_physical(sampler.random(n=batch))
        valid.extend(phys[_valid_mask(phys)])
    return np.asarray(valid[:n], dtype=float)


def generate_seed_levels(n: int = N_SEEDS, seed: int = SAMPLER_SEED + 1) -> np.ndarray:
    n_pow2 = 1 << math.ceil(math.log2(max(n, 1)))
    unit = qmc.Sobol(d=1, scramble=True, seed=seed).random(n=n_pow2)[:n, 0]
    return (RF_SEED_MIN + unit * (RF_SEED_MAX - RF_SEED_MIN)).astype(int)


def plot_corner(phys: np.ndarray, seeds: np.ndarray, out_path: Path) -> None:
    n_dim = phys.shape[1]
    drop_max = 500.0 * math.tan(math.radians(3.0))
    fig, axes = plt.subplots(n_dim, n_dim, figsize=(13, 13), constrained_layout=True)
    fig.suptitle(
        f"ood_dipping — {n_dim}D Sobol physics (n={len(phys)})\n"
        f"|dip|≤3° (~{drop_max:.1f} m full-span drop) | seed axis: {len(seeds)} levels",
        fontsize=11,
    )

    for i in range(n_dim):
        for j in range(n_dim):
            ax = axes[i, j]
            if i == j:
                ax.hist(phys[:, i], bins=8, color="#3d5a80", edgecolor="white", alpha=0.9)
                ax.set_ylabel("count" if j == 0 else "")
            elif i > j:
                ax.scatter(phys[:, j], phys[:, i], s=22, c="#ee6c4d", alpha=0.85, edgecolors="none")
            else:
                ax.axis("off")
                continue
            if i == n_dim - 1:
                ax.set_xlabel(AXIS_LABELS[j], fontsize=7)
            else:
                ax.tick_params(labelbottom=False)
            if j == 0 and i != j:
                ax.set_ylabel(AXIS_LABELS[i], fontsize=7)
            elif j != 0:
                ax.tick_params(labelleft=False)
            ax.tick_params(labelsize=6)

    seed_ax = fig.add_axes([0.78, 0.82, 0.18, 0.10])
    seed_ax.hist(seeds, bins=10, color="#293241", edgecolor="white", alpha=0.9)
    seed_ax.set_title("Sobol seed axis", fontsize=8)
    seed_ax.set_xlabel("rf seed", fontsize=7)
    seed_ax.tick_params(labelsize=6)
    seed_ax.set_yticks([])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_csv(phys: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for i, row in enumerate(phys):
            writer.writerow(
                {
                    "sobol_id": i,
                    "Vs1": float(row[0]),
                    "H": float(row[1]),
                    "rH": float(row[2]),
                    "aHV": float(row[3]),
                    "dip_angle_deg": float(row[4]),
                    "CoV": float(row[5]),
                    "Vs2": float(row[6]),
                }
            )


def main() -> None:
    phys = generate_physics_samples()
    seeds = generate_seed_levels()
    plots = THIS_DIR / "plots"
    write_csv(phys, plots / "sobol_physics.csv")
    np.save(plots / "sobol_seeds.npy", seeds)
    out = plots / "sobol_distribution.png"
    plot_corner(phys, seeds, out)
    print(f"Wrote {out}")
    print(f"Wrote {plots / 'sobol_physics.csv'}")
    print(
        f"dip [{phys[:, 4].min():+.2f}, {phys[:, 4].max():+.2f}] deg | "
        f"CoV [{phys[:, 5].min():.2f}, {phys[:, 5].max():.2f}] | "
        f"Vs2 [{phys[:, 6].min():.0f}, {phys[:, 6].max():.0f}]"
    )


if __name__ == "__main__":
    main()
