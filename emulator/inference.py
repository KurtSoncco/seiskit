"""Inference utilities for transfer function emulator."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Add parent directory to path to allow importing emulator
sys.path.insert(0, str(Path(__file__).parent.parent))

from emulator.data_utils import (
    denormalize_output_accel,
    load_normalizer,
    normalize_base_motion,
    normalize_material,
)
from emulator.model import TransferFunctionEmulator


def run_inference(
    model: torch.nn.Module,
    material_grid: np.ndarray,
    base_motion: np.ndarray,
    stats: dict,
    device: torch.device,
) -> np.ndarray:
    """Run inference on new data.

    Args:
        model: Trained model
        material_grid: Material grid of shape (H, W, 2) = [Vs, density]
        base_motion: Base motion time-series of shape (T, 2) = [ax, ay]
        stats: Normalization statistics
        device: Device to run on

    Returns:
        Predicted output accelerations of shape (T, 4) = [base_ax, base_ay, surf_ax, surf_ay]
    """
    model.eval()

    # Normalize inputs
    material_norm = normalize_material(material_grid, stats["materials"])
    base_motion_norm = normalize_base_motion(base_motion, stats["base_motion"])

    # Convert to tensors and add batch dimension
    material_tensor = (
        torch.FloatTensor(material_norm).permute(2, 0, 1).unsqueeze(0).to(device)
    )  # (1, 2, H, W)
    base_motion_tensor = (
        torch.FloatTensor(base_motion_norm).unsqueeze(0).to(device)
    )  # (1, T, 2)

    # Run inference
    with torch.no_grad():
        prediction = model(material_tensor, base_motion_tensor)
        prediction = prediction.cpu().squeeze(0).numpy()  # (T, 4)

    # Denormalize output
    prediction = denormalize_output_accel(prediction, stats["output_accel"])

    return prediction


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """Load trained model from checkpoint.

    Args:
        model_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Loaded model
    """
    checkpoint = torch.load(model_path, map_location=device)
    model = TransferFunctionEmulator()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with transfer function emulator"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--normalizer_path", type=str, required=True, help="Path to normalizer file"
    )
    parser.add_argument(
        "--material_path", type=str, required=True, help="Path to material grid (.npy)"
    )
    parser.add_argument(
        "--base_motion_path", type=str, required=True, help="Path to base motion (.npy)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="prediction.npy",
        help="Path to save prediction",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, device)

    # Load normalizer
    print(f"Loading normalizer from {args.normalizer_path}...")
    stats = load_normalizer(Path(args.normalizer_path))

    # Load inputs
    print(f"Loading material grid from {args.material_path}...")
    material_grid = np.load(args.material_path)  # (H, W, 2)

    print(f"Loading base motion from {args.base_motion_path}...")
    base_motion = np.load(args.base_motion_path)  # (T, 2)

    # Run inference
    print("Running inference...")
    prediction = run_inference(model, material_grid, base_motion, stats, device)

    # Save prediction
    print(f"Saving prediction to {args.output_path}...")
    np.save(args.output_path, prediction)

    print(f"Prediction shape: {prediction.shape}")
    print(f"Prediction range: [{prediction.min():.4f}, {prediction.max():.4f}]")
    print("Done!")


if __name__ == "__main__":
    main()
