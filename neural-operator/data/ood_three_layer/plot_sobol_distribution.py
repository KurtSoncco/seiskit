"""Plot planned Sobol physics distribution for Tier B ood_three_layer.

Axes (8D, n=32): Vs_mid, H1, H2, rH, aHV, Vs_contrast, CoV, Vs_bedrock.
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
    lognormal_parameter,
    scale_aHV,
    scale_Vs2,
    sigma_aHV,
    sigma_Vs2,
)

N_PHYSICS = 32
N_SEEDS = 30
SAMPLER_SEED = DEFAULT_SAMPLER_SEED
RF_SEED_MIN = 1
RF_SEED_MAX = 9_999_999

scale_Vs_mid, sigma_Vs_mid = lognormal_parameter(450.0, 560.0)
bounds_H1 = (5.0, 12.0)
bounds_H2 = (5.0, 12.0)
bounds_Vs_contrast = (0.8, 1.6)
MAX_H_SUM = 24.0

AXIS_LABELS = (
    "Vs_mid [m/s]",
    "H1 [m]",
    "H2 [m]",
    "rH [m]",
    "aHV [-]",
    "Vs_contrast [-]",
    "CoV [-]",
    "Vs_bedrock [m/s]",
)
CSV_COLUMNS = (
    "sobol_id",
    "Vs_mid",
    "H1",
    "H2",
    "rH",
    "aHV",
    "Vs_contrast",
    "CoV",
    "Vs_bedrock",
)
N_DIM = len(AXIS_LABELS)


def unit_to_physical(unit: np.ndarray) -> np.ndarray:
    raw = np.asarray(unit, dtype=float)
    phys = np.zeros_like(raw)
    phys[:, 0] = lognorm.ppf(raw[:, 0], s=sigma_Vs_mid, scale=scale_Vs_mid)
    phys[:, 1] = bounds_H1[0] + raw[:, 1] * (bounds_H1[1] - bounds_H1[0])
    phys[:, 2] = bounds_H2[0] + raw[:, 2] * (bounds_H2[1] - bounds_H2[0])
    phys[:, 3] = bounds_rH[0] + raw[:, 3] * (bounds_rH[1] - bounds_rH[0])
    phys[:, 4] = lognorm.ppf(raw[:, 4], s=sigma_aHV, scale=scale_aHV)
    phys[:, 5] = bounds_Vs_contrast[0] + raw[:, 5] * (bounds_Vs_contrast[1] - bounds_Vs_contrast[0])
    phys[:, 6] = bounds_CoV[0] + raw[:, 6] * (bounds_CoV[1] - bounds_CoV[0])
    phys[:, 7] = lognorm.ppf(raw[:, 7], s=sigma_Vs2, scale=scale_Vs2)
    return phys


def _valid_mask(phys: np.ndarray) -> np.ndarray:
    return (
        (phys[:, 0] >= 450.0)
        & (phys[:, 0] <= 560.0)
        & (phys[:, 1] >= bounds_H1[0])
        & (phys[:, 1] <= bounds_H1[1])
        & (phys[:, 2] >= bounds_H2[0])
        & (phys[:, 2] <= bounds_H2[1])
        & (phys[:, 1] + phys[:, 2] <= MAX_H_SUM)
        & (phys[:, 3] >= bounds_rH[0])
        & (phys[:, 3] <= bounds_rH[1])
        & (phys[:, 4] >= 10.0)
        & (phys[:, 4] <= 50.0)
        & (phys[:, 5] >= bounds_Vs_contrast[0])
        & (phys[:, 5] <= bounds_Vs_contrast[1])
        & (phys[:, 6] >= bounds_CoV[0])
        & (phys[:, 6] <= bounds_CoV[1])
        & (phys[:, 7] >= 760.0)
        & (phys[:, 7] <= 1500.0)
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
    fig, axes = plt.subplots(n_dim, n_dim, figsize=(14, 14), constrained_layout=True)
    fig.suptitle(
        f"ood_three_layer — {n_dim}D Sobol physics (n={len(phys)})\n"
        f"shared rH/aHV/CoV across layers | seed axis: {len(seeds)} Sobol levels",
        fontsize=12,
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
                    "Vs_mid": float(row[0]),
                    "H1": float(row[1]),
                    "H2": float(row[2]),
                    "rH": float(row[3]),
                    "aHV": float(row[4]),
                    "Vs_contrast": float(row[5]),
                    "CoV": float(row[6]),
                    "Vs_bedrock": float(row[7]),
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
    vs1 = phys[:, 0] / np.exp(phys[:, 5])
    print(
        f"H1+H2 max={(phys[:, 1] + phys[:, 2]).max():.1f} m | "
        f"Vs1 derived [{vs1.min():.0f}, {vs1.max():.0f}] | "
        f"Vs_bedrock [{phys[:, 7].min():.0f}, {phys[:, 7].max():.0f}]"
    )


if __name__ == "__main__":
    main()
