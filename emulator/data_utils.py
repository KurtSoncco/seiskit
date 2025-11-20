"""Data utilities for normalization and data processing."""

import json
from pathlib import Path
from typing import Dict

import numpy as np


def load_recorder_outputs(
    output_dir: Path, sim_id: str, recorder_quantity: str = "accel", Ly: float = 150.0
) -> np.ndarray:
    """Load and combine recorder outputs into 4-channel format.

    Loads acceleration data from base and surface nodes (DOF 1 and 2)
    and combines into shape (T, 4) = [base_ax, base_ay, surf_ax, surf_ay].

    Args:
        output_dir: Directory containing recorder output files
        sim_id: Simulation identifier
        recorder_quantity: Type of recorder output ("accel", "disp", "vel")
        Ly: Domain height (used to determine surface Y position, default: 150.0)

    Returns:
        Array of shape (T, 4) with [base_ax, base_ay, surf_ax, surf_ay]
        Returns None if files are not found
    """
    # Try new naming convention first (Y-position based)
    base_dof1_file = output_dir / f"center_node_y0.00_dof1_{recorder_quantity}.txt"
    base_dof2_file = output_dir / f"center_node_y0.00_dof2_{recorder_quantity}.txt"
    top_dof1_file = output_dir / f"center_node_y{Ly:.2f}_dof1_{recorder_quantity}.txt"
    top_dof2_file = output_dir / f"center_node_y{Ly:.2f}_dof2_{recorder_quantity}.txt"

    # If new naming convention files don't exist, try old naming convention
    if not base_dof1_file.exists():
        base_dof1_file = output_dir / f"soil_base_dof1_{recorder_quantity}.txt"
        base_dof2_file = output_dir / f"soil_base_dof2_{recorder_quantity}.txt"
        top_dof1_file = output_dir / f"soil_top_dof1_{recorder_quantity}.txt"
        top_dof2_file = output_dir / f"soil_top_dof2_{recorder_quantity}.txt"

    # Check if files exist
    missing_files = []
    for name, path in [
        ("base_dof1", base_dof1_file),
        ("base_dof2", base_dof2_file),
        ("top_dof1", top_dof1_file),
        ("top_dof2", top_dof2_file),
    ]:
        if not path.exists():
            missing_files.append(f"{name}: {path}")

    if missing_files:
        raise FileNotFoundError(
            f"Recorder output files not found in {output_dir}:\n"
            + "\n".join(f"  - {f}" for f in missing_files)
            + f"\n\nAvailable files in directory:"
            + "\n".join(f"  - {f.name}" for f in sorted(output_dir.glob("*.txt")))
        )

    # Load data (first column is time, second is value)
    base_ax = np.loadtxt(base_dof1_file)[:, 1]
    base_ay = np.loadtxt(base_dof2_file)[:, 1]
    surf_ax = np.loadtxt(top_dof1_file)[:, 1]
    surf_ay = np.loadtxt(top_dof2_file)[:, 1]

    # Stack into (T, 4) format
    output = np.stack([base_ax, base_ay, surf_ax, surf_ay], axis=1)

    return output


def compute_normalization_stats(
    data_dir: Path,
    materials_dir: str = "materials",
    base_motion_dir: str = "base_motion",
    output_accel_dir: str = "low_fidelity/output_accel",
    train_indices: list[int] | None = None,
) -> Dict[str, Dict[str, float]]:
    """Compute normalization statistics from training data.

    Args:
        data_dir: Base data directory
        materials_dir: Subdirectory for material grids
        base_motion_dir: Subdirectory for base motion time-series
        output_accel_dir: Subdirectory for output acceleration time-series
        train_indices: List of simulation indices to use for training (if None, uses all)

    Returns:
        Dictionary with normalization stats:
        {
            "materials": {"vs_mean": ..., "vs_std": ..., "density_mean": ..., "density_std": ...},
            "base_motion": {"ax_mean": ..., "ax_std": ..., "ay_mean": ..., "ay_std": ...},
            "output_accel": {"base_ax_mean": ..., "base_ax_std": ..., ...}
        }
    """
    materials_path = data_dir / materials_dir
    base_motion_path = data_dir / base_motion_dir
    output_accel_path = data_dir / output_accel_dir

    # Find all available simulation files
    material_files = sorted(materials_path.glob("sim_*.npy"))
    if train_indices is None:
        # Use all available files
        indices = list(range(len(material_files)))
    else:
        indices = train_indices

    # Collect all data
    vs_values = []
    density_values = []
    base_ax_values = []
    base_ay_values = []
    surf_ax_values = []
    surf_ay_values = []

    for idx in indices:
        # Find corresponding material file
        mat_file = materials_path / f"sim_{idx:04d}.npy"
        if not mat_file.exists():
            continue

        # Load material grid (H, W, 2)
        material = np.load(mat_file)
        vs_values.append(material[:, :, 0].flatten())
        density_values.append(material[:, :, 1].flatten())

        # Load base motion (T, 2)
        base_motion_file = base_motion_path / f"sim_{idx:04d}.npy"
        if base_motion_file.exists():
            base_motion = np.load(base_motion_file)
            base_ax_values.append(base_motion[:, 0])
            base_ay_values.append(base_motion[:, 1])

        # Load output accel (T, 4)
        output_file = output_accel_path / f"sim_{idx:04d}.npy"
        if output_file.exists():
            output = np.load(output_file)
            base_ax_values.append(output[:, 0])
            base_ay_values.append(output[:, 1])
            surf_ax_values.append(output[:, 2])
            surf_ay_values.append(output[:, 3])

    # Compute statistics
    all_vs_values = np.concatenate(vs_values) if vs_values else np.array([])
    stats = {
        "vs_field": {
            "mean": float(all_vs_values.mean()) if len(all_vs_values) > 0 else 0.0,
            "std": float(all_vs_values.std()) if len(all_vs_values) > 0 else 1.0,
        },
        "materials": {
            "vs_mean": float(all_vs_values.mean()) if len(all_vs_values) > 0 else 0.0,
            "vs_std": float(all_vs_values.std()) if len(all_vs_values) > 0 else 1.0,
            "density_mean": float(np.concatenate(density_values).mean())
            if density_values
            else 0.0,
            "density_std": float(np.concatenate(density_values).std())
            if density_values
            else 1.0,
        },
        "base_motion": {
            "ax_mean": float(np.concatenate(base_ax_values).mean())
            if base_ax_values
            else 0.0,
            "ax_std": float(np.concatenate(base_ax_values).std())
            if base_ax_values
            else 1.0,
            "ay_mean": float(np.concatenate(base_ay_values).mean())
            if base_ay_values
            else 0.0,
            "ay_std": float(np.concatenate(base_ay_values).std())
            if base_ay_values
            else 1.0,
        },
        "output_accel": {},
    }

    # Fix the output_accel computation - collect all output arrays first
    all_outputs = []
    for idx in indices:
        output_file = output_accel_path / f"sim_{idx:04d}.npy"
        if output_file.exists():
            all_outputs.append(np.load(output_file))

    if all_outputs:
        all_outputs_array = np.concatenate(all_outputs, axis=0)  # (N*T, 4)
        stats["output_accel"] = {
            "base_ax_mean": float(all_outputs_array[:, 0].mean()),
            "base_ax_std": float(all_outputs_array[:, 0].std()),
            "base_ay_mean": float(all_outputs_array[:, 1].mean()),
            "base_ay_std": float(all_outputs_array[:, 1].std()),
            "surf_ax_mean": float(all_outputs_array[:, 2].mean()),
            "surf_ax_std": float(all_outputs_array[:, 2].std()),
            "surf_ay_mean": float(all_outputs_array[:, 3].mean()),
            "surf_ay_std": float(all_outputs_array[:, 3].std()),
        }
    else:
        # Default values if no outputs found
        stats["output_accel"] = {
            "base_ax_mean": 0.0,
            "base_ax_std": 1.0,
            "base_ay_mean": 0.0,
            "base_ay_std": 1.0,
            "surf_ax_mean": 0.0,
            "surf_ax_std": 1.0,
            "surf_ay_mean": 0.0,
            "surf_ay_std": 1.0,
        }

    return stats


def normalize_material(material: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    """Normalize material grid.

    Args:
        material: Material grid of shape (H, W, 2) = [Vs, density]
        stats: Normalization stats for materials

    Returns:
        Normalized material grid
    """
    normalized = material.copy()
    normalized[:, :, 0] = (normalized[:, :, 0] - stats["vs_mean"]) / stats["vs_std"]
    normalized[:, :, 1] = (normalized[:, :, 1] - stats["density_mean"]) / stats[
        "density_std"
    ]
    return normalized


def denormalize_material(material: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    """Denormalize material grid.

    Args:
        material: Normalized material grid of shape (H, W, 2)
        stats: Normalization stats for materials

    Returns:
        Denormalized material grid
    """
    denormalized = material.copy()
    denormalized[:, :, 0] = denormalized[:, :, 0] * stats["vs_std"] + stats["vs_mean"]
    denormalized[:, :, 1] = (
        denormalized[:, :, 1] * stats["density_std"] + stats["density_mean"]
    )
    return denormalized


def normalize_base_motion(
    base_motion: np.ndarray, stats: Dict[str, float]
) -> np.ndarray:
    """Normalize base motion time-series.

    Args:
        base_motion: Base motion of shape (T, 2) = [ax, ay]
        stats: Normalization stats for base_motion

    Returns:
        Normalized base motion
    """
    normalized = base_motion.copy()
    normalized[:, 0] = (normalized[:, 0] - stats["ax_mean"]) / stats["ax_std"]
    normalized[:, 1] = (normalized[:, 1] - stats["ay_mean"]) / stats["ay_std"]
    return normalized


def denormalize_base_motion(
    base_motion: np.ndarray, stats: Dict[str, float]
) -> np.ndarray:
    """Denormalize base motion time-series.

    Args:
        base_motion: Normalized base motion of shape (T, 2)
        stats: Normalization stats for base_motion

    Returns:
        Denormalized base motion
    """
    denormalized = base_motion.copy()
    denormalized[:, 0] = denormalized[:, 0] * stats["ax_std"] + stats["ax_mean"]
    denormalized[:, 1] = denormalized[:, 1] * stats["ay_std"] + stats["ay_mean"]
    return denormalized


def normalize_output_accel(
    output_accel: np.ndarray, stats: Dict[str, float]
) -> np.ndarray:
    """Normalize output acceleration time-series.

    Args:
        output_accel: Output accel of shape (T, 4) = [base_ax, base_ay, surf_ax, surf_ay]
        stats: Normalization stats for output_accel

    Returns:
        Normalized output accel
    """
    normalized = output_accel.copy()
    normalized[:, 0] = (normalized[:, 0] - stats["base_ax_mean"]) / stats["base_ax_std"]
    normalized[:, 1] = (normalized[:, 1] - stats["base_ay_mean"]) / stats["base_ay_std"]
    normalized[:, 2] = (normalized[:, 2] - stats["surf_ax_mean"]) / stats["surf_ax_std"]
    normalized[:, 3] = (normalized[:, 3] - stats["surf_ay_mean"]) / stats["surf_ay_std"]
    return normalized


def denormalize_output_accel(
    output_accel: np.ndarray, stats: Dict[str, float]
) -> np.ndarray:
    """Denormalize output acceleration time-series.

    Args:
        output_accel: Normalized output accel of shape (T, 4)
        stats: Normalization stats for output_accel

    Returns:
        Denormalized output accel
    """
    denormalized = output_accel.copy()
    denormalized[:, 0] = (
        denormalized[:, 0] * stats["base_ax_std"] + stats["base_ax_mean"]
    )
    denormalized[:, 1] = (
        denormalized[:, 1] * stats["base_ay_std"] + stats["base_ay_mean"]
    )
    denormalized[:, 2] = (
        denormalized[:, 2] * stats["surf_ax_std"] + stats["surf_ax_mean"]
    )
    denormalized[:, 3] = (
        denormalized[:, 3] * stats["surf_ay_std"] + stats["surf_ay_mean"]
    )
    return denormalized


def normalize_vs_field(vs_field: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    """Normalize Vs field.

    Args:
        vs_field: Vs field of shape (H, W)
        stats: Normalization stats for vs_field

    Returns:
        Normalized Vs field
    """
    normalized = (vs_field - stats["mean"]) / stats["std"]
    return normalized


def denormalize_vs_field(vs_field: np.ndarray, stats: Dict[str, float]) -> np.ndarray:
    """Denormalize Vs field.

    Args:
        vs_field: Normalized Vs field of shape (H, W)
        stats: Normalization stats for vs_field

    Returns:
        Denormalized Vs field
    """
    denormalized = vs_field * stats["std"] + stats["mean"]
    return denormalized


def save_normalizer(stats: Dict, path: Path) -> None:
    """Save normalization statistics to file.

    Args:
        stats: Normalization statistics dictionary
        path: Path to save file (JSON or NPY)
    """
    if path.suffix == ".json":
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)
    else:
        # Save as numpy file (dict of arrays)
        # np.save() can save arbitrary Python objects (including dicts) via pickle
        np.save(path, stats)  # type: ignore[arg-type]


def load_normalizer(path: Path) -> Dict:
    """Load normalization statistics from file.

    Args:
        path: Path to normalization file

    Returns:
        Normalization statistics dictionary
    """
    if path.suffix == ".json":
        with open(path, "r") as f:
            return json.load(f)
    else:
        return np.load(path, allow_pickle=True).item()
