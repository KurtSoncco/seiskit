"""Evaluation script for PGA emulator.

Evaluates the emulator against HF oracle and compares with LF baseline.
Computes Absolute Relative Error (ARE) and generates parity plot.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from tqdm import tqdm

import wandb

# Add parent directory to path to allow importing emulator
sys.path.insert(0, str(Path(__file__).parent.parent))

from emulator.dataset import PGADataset
from emulator.model import PGAEmulator


def compute_are(predicted: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Compute Absolute Relative Error (ARE).

    Args:
        predicted: Predicted values
        true: True values

    Returns:
        Array of ARE values
    """
    return np.abs(predicted - true) / (np.abs(true) + 1e-10)


def compute_superiority_ratio(
    pga_pred: np.ndarray, pga_lf: np.ndarray, pga_hf: np.ndarray
) -> np.ndarray:
    """Compute Superiority Ratio (ξ) for each sample.

    ξ = |Emulator - HF| / |LF - HF|

    Args:
        pga_pred: Emulator predictions
        pga_lf: Low-fidelity predictions
        pga_hf: High-fidelity ground truth

    Returns:
        Array of superiority ratios (ξ)
    """
    numerator = np.abs(pga_pred - pga_hf)
    denominator = np.abs(pga_lf - pga_hf) + 1e-10  # Avoid division by zero
    return numerator / denominator


def evaluate_model(
    model: torch.nn.Module,
    test_dataset: PGADataset,
    device: torch.device,
) -> dict:
    """Evaluate model on test dataset using proper superiority metrics.

    Compares Emulator vs HF_Truth vs LF_Baseline using Superiority Ratio (ξ).

    Args:
        model: Trained model
        test_dataset: Test dataset
        device: Device to run on

    Returns:
        Dictionary with evaluation metrics and data for plotting
    """
    model.eval()

    all_pga_hf = []
    all_pga_lf = []
    all_pga_pred = []
    all_sim_ids = []

    with torch.no_grad():
        pbar = tqdm(range(len(test_dataset)), desc="Evaluating")
        for i in pbar:
            sample = test_dataset[i]

            # Skip if HF oracle not available (check for NaN)
            if "pga_hf" not in sample:
                continue
            pga_hf_tensor = sample["pga_hf"]
            # Check if it's a tensor and contains NaN
            if isinstance(pga_hf_tensor, torch.Tensor) and torch.isnan(pga_hf_tensor).any():
                continue

            # Extract tensors with type checking
            vs_field_tensor = sample["vs_field"]
            pga_lf_tensor = sample["pga_lf"]
            sim_id = sample["sim_id"]

            # Ensure they are tensors
            if not isinstance(vs_field_tensor, torch.Tensor):
                raise TypeError(
                    f"Expected vs_field to be torch.Tensor, got {type(vs_field_tensor)}"
                )
            if not isinstance(pga_lf_tensor, torch.Tensor):
                raise TypeError(
                    f"Expected pga_lf to be torch.Tensor, got {type(pga_lf_tensor)}"
                )
            if not isinstance(pga_hf_tensor, torch.Tensor):
                raise TypeError(
                    f"Expected pga_hf to be torch.Tensor, got {type(pga_hf_tensor)}"
                )

            vs_field = vs_field_tensor.unsqueeze(0).to(device)  # (1, 1, 150, 150)
            pga_lf = pga_lf_tensor.item()  # Scalar
            pga_hf = pga_hf_tensor.item()  # Scalar (oracle)

            # Run model
            pga_pred = model(vs_field).cpu().item()  # Scalar

            all_pga_hf.append(pga_hf)
            all_pga_lf.append(pga_lf)
            all_pga_pred.append(pga_pred)
            all_sim_ids.append(sim_id)
            
            # Update progress bar
            pbar.set_postfix({"samples": len(all_pga_hf)})

    # Convert to numpy arrays
    all_pga_hf = np.array(all_pga_hf)
    all_pga_lf = np.array(all_pga_lf)
    all_pga_pred = np.array(all_pga_pred)
    all_sim_ids = np.array(all_sim_ids)

    # Compute Superiority Ratio (ξ) for each sample
    xi = compute_superiority_ratio(all_pga_pred, all_pga_lf, all_pga_hf)

    # Compute aggregate metrics
    # 1. Geometric Mean of ξ (use log to avoid overflow)
    log_xi = np.log(xi + 1e-10)  # Add small epsilon to avoid log(0)
    geometric_mean_xi = np.exp(np.mean(log_xi))

    # 2. Win Rate (percentage where ξ < 1)
    win_rate = np.mean(xi < 1.0) * 100.0

    # 3. Relative Error Reduction
    mre_lf = np.mean(np.abs(all_pga_lf - all_pga_hf) / (np.abs(all_pga_hf) + 1e-10))
    mre_emulator = np.mean(
        np.abs(all_pga_pred - all_pga_hf) / (np.abs(all_pga_hf) + 1e-10)
    )
    relative_error_reduction = (mre_lf - mre_emulator) / (mre_lf + 1e-10) * 100.0

    # 4. Arithmetic mean for comparison
    mean_xi = np.mean(xi)
    median_xi = np.median(xi)

    # Legacy metrics (for backward compatibility)
    are_lf = compute_are(all_pga_lf, all_pga_hf)
    are_nn = compute_are(all_pga_pred, all_pga_hf)
    mean_are_lf = np.mean(are_lf)
    mean_are_nn = np.mean(are_nn)
    std_are_lf = np.std(are_lf)
    std_are_nn = np.std(are_nn)

    summary = {
        # Core Superiority Metrics
        "geometric_mean_xi": float(geometric_mean_xi),
        "arithmetic_mean_xi": float(mean_xi),
        "median_xi": float(median_xi),
        "win_rate": float(win_rate),
        "relative_error_reduction": float(relative_error_reduction),
        # Legacy metrics
        "mean_are_lf": float(mean_are_lf),
        "mean_are_nn": float(mean_are_nn),
        "std_are_lf": float(std_are_lf),
        "std_are_nn": float(std_are_nn),
        "mre_lf": float(mre_lf),
        "mre_emulator": float(mre_emulator),
        "n_samples": len(all_pga_hf),
        # Success criteria
        "is_superior": bool(geometric_mean_xi < 1.0 and win_rate > 50.0),
    }

    return {
        "summary": summary,
        "pga_hf": all_pga_hf,
        "pga_lf": all_pga_lf,
        "pga_pred": all_pga_pred,
        "xi": xi,
        "sim_ids": all_sim_ids,
        "are_lf": are_lf,
        "are_nn": are_nn,
    }


def plot_superiority_histogram(
    xi: np.ndarray, output_path: Path | None = None
) -> Figure:
    """Create superiority ratio histogram (the "Money Plot").

    X-axis: Superiority Ratio ξ (log scale)
    Vertical line at ξ = 1

    Args:
        xi: Array of superiority ratios
        output_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create histogram with log scale
    ax.hist(xi, bins=50, alpha=0.7, edgecolor="black", color="steelblue")
    ax.axvline(
        x=1.0, color="red", linewidth=3, linestyle="--", label="ξ = 1 (Imitation)"
    )

    ax.set_xlabel("Superiority Ratio (ξ)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(
        "Superiority Ratio Distribution: Neural Emulator vs Low-Fidelity Solver",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=11)

    # Add text annotation with win rate
    win_rate = np.mean(xi < 1.0) * 100.0
    ax.text(
        0.05,
        0.95,
        f"Win Rate: {win_rate:.1f}% (ξ < 1)",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved superiority histogram to {output_path}")

    return fig


def plot_parity(
    pga_hf: np.ndarray,
    pga_lf: np.ndarray,
    pga_pred: np.ndarray,
    output_path: Path | None = None,
) -> Figure:
    """Create parity plot comparing LF baseline and Emulator predictions.

    Args:
        pga_hf: High-fidelity PGA values (ground truth)
        pga_lf: Low-fidelity PGA values (baseline)
        pga_pred: Emulator predictions
        output_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot 1:1 line
    pga_min = min(pga_hf.min(), pga_lf.min(), pga_pred.min())
    pga_max = max(pga_hf.max(), pga_lf.max(), pga_pred.max())
    ax.plot([pga_min, pga_max], [pga_min, pga_max], "k-", linewidth=2, label="1:1 Line")

    # Plot LF baseline (blue dots)
    ax.scatter(
        pga_hf,
        pga_lf,
        alpha=0.6,
        s=50,
        c="blue",
        label="LF Solver",
        edgecolors="black",
        linewidths=0.5,
    )

    # Plot Emulator predictions (red dots)
    ax.scatter(
        pga_hf,
        pga_pred,
        alpha=0.6,
        s=50,
        c="red",
        label="Emulator",
        edgecolors="black",
        linewidths=0.5,
        marker="s",
    )

    ax.set_xlabel("High-Fidelity PGA (m/s²)", fontsize=12)
    ax.set_ylabel("Predicted/Calculated PGA (m/s²)", fontsize=12)
    ax.set_title(
        "PGA Parity Plot: LF Solver vs Emulator", fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved parity plot to {output_path}")

    return fig


def main(wandb_run=None):
    """Evaluate PGA emulator.

    Args:
        wandb_run: Optional wandb run to continue logging to. If None, creates a new run.
    """
    parser = argparse.ArgumentParser(description="Evaluate PGA emulator")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/best_model.pt",
        help="Model checkpoint path",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory containing checkpoints (used if model_path is relative)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    parser.add_argument(
        "--wandb_project", type=str, default="pga_emulator", help="W&B project name"
    )
    parser.add_argument(
        "--n_test", type=int, default=100, help="Number of test samples"
    )
    parser.add_argument(
        "--test_start_idx",
        type=int,
        default=120,
        help="Starting index for test samples",
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default="parity_plot.png",
        help="Output path for parity plot",
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        default="plots",
        help="Directory to save plots",
    )
    parser.add_argument(
        "--wandb_run_id",
        type=str,
        default=None,
        help="W&B run ID to resume (for continuing an existing run)",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    device = torch.device(args.device)

    # Create plots directory
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Initialize W&B (use existing run if provided, or resume by ID)
    if wandb_run is not None:
        run = wandb_run
        # Update config with evaluation parameters
        run.config.update(args.__dict__)
        run.config["job_type"] = "evaluate"
    elif args.wandb_run_id is not None:
        # Resume existing run by ID
        run = wandb.init(
            project=args.wandb_project,
            id=args.wandb_run_id,
            resume="allow",
            job_type="evaluate",
        )
        run.config.update(args.__dict__)
        run.config["job_type"] = "evaluate"
    else:
        run = wandb.init(
            project=args.wandb_project,
            job_type="evaluate",
            config=args.__dict__,
        )

    # Load normalization stats path (dataset will load it)
    normalizer_path = data_dir / "normalizer.npy"

    # Discover available test data (must have HF oracle)
    print("Discovering available test data with HF oracle...")
    materials_hf_dir = data_dir / "materials_hf"
    materials_dir = data_dir / "materials"
    lf_pga_dir = data_dir / "low_fidelity" / "pga"
    hf_pga_dir = data_dir / "high_fidelity" / "pga"

    # Check which materials directory exists
    if materials_hf_dir.exists():
        mat_dir = materials_hf_dir
    elif materials_dir.exists():
        mat_dir = materials_dir
    else:
        raise ValueError(f"Neither {materials_hf_dir} nor {materials_dir} exists")

    # Find all available test indices (must have both LF and HF PGA)
    available_test_indices = set()
    for mat_file in mat_dir.glob("sim_*.npy"):
        try:
            idx = int(mat_file.stem.split("_")[1])
            # Check if both LF and HF PGA files exist
            pga_lf_file = lf_pga_dir / f"sim_{idx:04d}.npy"
            pga_hf_file = hf_pga_dir / f"sim_{idx:04d}.npy"
            if pga_lf_file.exists() and pga_hf_file.exists():
                available_test_indices.add(idx)
        except (ValueError, IndexError):
            continue

    available_test_indices = sorted(list(available_test_indices))
    print(
        f"Found {len(available_test_indices)} test samples with HF oracle at indices: {available_test_indices[:10]}{'...' if len(available_test_indices) > 10 else ''}"
    )

    if len(available_test_indices) == 0:
        raise ValueError(
            f"No test data with HF oracle found in {data_dir}. "
            f"Test data must have both LF and HF PGA files."
        )

    # Filter by test_start_idx and limit to requested number
    filtered_indices = [idx for idx in available_test_indices if idx >= args.test_start_idx]
    
    # Auto-adjust test_start_idx if it's too high
    if len(filtered_indices) == 0:
        print(
            f"Warning: No test samples found >= {args.test_start_idx}. "
            f"Available indices range from {available_test_indices[0]} to {available_test_indices[-1]}. "
            f"Using all available test samples instead."
        )
        filtered_indices = available_test_indices
    
    test_indices = filtered_indices[: args.n_test]
    
    if len(test_indices) < args.n_test:
        print(
            f"Warning: Only {len(test_indices)} test samples available (starting from index {filtered_indices[0] if filtered_indices else 'N/A'}), "
            f"but requested {args.n_test}. Available indices: {len(available_test_indices)} total, "
            f"{len(filtered_indices)} >= {args.test_start_idx if len(filtered_indices) > 0 else 'auto-adjusted'}"
        )
    else:
        print(f"Using {len(test_indices)} test samples (indices {test_indices[0]} to {test_indices[-1]})")

    # Create test dataset
    test_dataset = PGADataset(
        data_dir=data_dir,
        indices=test_indices,
        normalizer_path=normalizer_path,
        split="test",
    )

    # Load model
    # If model_path is relative and doesn't exist, try in checkpoint_dir
    model_path = Path(args.model_path)
    if not model_path.is_absolute() and not model_path.exists():
        checkpoint_dir = Path(args.checkpoint_dir)
        model_path = checkpoint_dir / model_path.name

    checkpoint = torch.load(model_path, map_location=device)
    model = PGAEmulator(
        input_size=(150, 150),
        num_classes=1,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate
    print("Evaluating model...")
    results = evaluate_model(model, test_dataset, device)

    summary = results["summary"]
    pga_hf = results["pga_hf"]
    pga_lf = results["pga_lf"]
    pga_pred = results["pga_pred"]
    xi = results["xi"]
    are_lf = results["are_lf"]
    are_nn = results["are_nn"]

    # Log summary metrics
    wandb.log(summary)

    # Create and log superiority histogram (the "Money Plot")
    superiority_hist_path = plots_dir / "superiority_histogram.png"
    fig_hist = plot_superiority_histogram(xi, output_path=superiority_hist_path)
    wandb.log({"superiority_histogram": wandb.Image(fig_hist)})
    plt.close(fig_hist)
    print(f"Saved superiority histogram to {superiority_hist_path}")

    # Create and log parity plot
    parity_plot_path = plots_dir / "parity_plot.png"
    fig_parity = plot_parity(pga_hf, pga_lf, pga_pred, output_path=parity_plot_path)
    wandb.log({"parity_plot": wandb.Image(fig_parity)})
    plt.close(fig_parity)
    print(f"Saved parity plot to {parity_plot_path}")

    # Log histograms
    wandb.log(
        {
            "xi_hist": wandb.Histogram(xi),
            "are_lf_hist": wandb.Histogram(are_lf),
            "are_nn_hist": wandb.Histogram(are_nn),
            "pga_hf_hist": wandb.Histogram(pga_hf),
        }
    )

    # Save metrics to JSON file
    metrics_json_path = plots_dir / "evaluation_metrics.json"
    metrics_data = {
        "summary": summary,
        "superiority_statistics": {
            "geometric_mean_xi": float(summary["geometric_mean_xi"]),
            "arithmetic_mean_xi": float(summary["arithmetic_mean_xi"]),
            "median_xi": float(summary["median_xi"]),
            "std_xi": float(np.std(xi)),
            "min_xi": float(np.min(xi)),
            "max_xi": float(np.max(xi)),
            "percentile_25_xi": float(np.percentile(xi, 25)),
            "percentile_75_xi": float(np.percentile(xi, 75)),
            "win_rate_percent": float(summary["win_rate"]),
            "n_samples_xi_lt_1": int(np.sum(xi < 1.0)),
            "n_samples_xi_lt_0_5": int(np.sum(xi < 0.5)),
            "n_samples_xi_lt_0_8": int(np.sum(xi < 0.8)),
        },
        "error_metrics": {
            "mean_are_lf": float(summary["mean_are_lf"]),
            "mean_are_emulator": float(summary["mean_are_nn"]),
            "std_are_lf": float(summary["std_are_lf"]),
            "std_are_emulator": float(summary["std_are_nn"]),
            "mre_lf": float(summary["mre_lf"]),
            "mre_emulator": float(summary["mre_emulator"]),
            "relative_error_reduction_percent": float(summary["relative_error_reduction"]),
        },
        "pga_statistics": {
            "pga_hf_mean": float(np.mean(pga_hf)),
            "pga_hf_std": float(np.std(pga_hf)),
            "pga_hf_min": float(np.min(pga_hf)),
            "pga_hf_max": float(np.max(pga_hf)),
            "pga_lf_mean": float(np.mean(pga_lf)),
            "pga_lf_std": float(np.std(pga_lf)),
            "pga_pred_mean": float(np.mean(pga_pred)),
            "pga_pred_std": float(np.std(pga_pred)),
        },
        "test_configuration": {
            "n_samples": int(summary["n_samples"]),
            "test_start_idx": args.test_start_idx,
            "test_indices": test_indices.tolist() if isinstance(test_indices, np.ndarray) else list(test_indices),
            "model_path": str(model_path),
        },
    }
    
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"Saved evaluation metrics to {metrics_json_path}")
    
    # Also log JSON file to wandb
    wandb.save(str(metrics_json_path))

    # Print summary following exact protocol
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY - SUPERIORITY METRICS")
    print("=" * 70)
    print(f"Number of test samples: {summary['n_samples']}")
    print(f"\n{'─' * 70}")
    print("CORE SUPERIORITY METRICS (Following Protocol):")
    print(f"{'─' * 70}")
    print(f"  1. Geometric Mean of ξ: {summary['geometric_mean_xi']:.4f}")
    print("     → Goal: < 1.0 (e.g., 0.85 means 15% error reduction)")
    print("     → ξ < 1: SUCCESS (Superiority)")
    print("     → ξ = 1: Imitation (learned LF solver perfectly)")
    print("     → ξ > 1: Failure (added noise on top of LF)")
    print(f"\n  2. Win Rate: {summary['win_rate']:.1f}% (samples where ξ < 1)")
    print("     → Goal: > 50% (ideally > 80%)")
    print(
        f"\n  3. Relative Error Reduction: {summary['relative_error_reduction']:.2f}%"
    )
    print("     → (MRE_LF - MRE_Emulator) / MRE_LF × 100%")
    print("     → Positive % means emulator reduces error vs LF")
    print("\n  Additional Statistics:")
    print(f"     - Arithmetic Mean of ξ: {summary['arithmetic_mean_xi']:.4f}")
    print(f"     - Median of ξ: {summary['median_xi']:.4f}")
    print(f"\n{'─' * 70}")
    print("LEGACY METRICS (for reference):")
    print(f"{'─' * 70}")
    print(f"  Mean ARE (LF): {summary['mean_are_lf']:.6f}")
    print(f"  Mean ARE (Emulator): {summary['mean_are_nn']:.6f}")
    print(f"  MRE (LF): {summary['mre_lf']:.6f}")
    print(f"  MRE (Emulator): {summary['mre_emulator']:.6f}")
    print(f"\n{'─' * 70}")
    if summary["is_superior"]:
        print("✓ SUCCESS: Neural Emulator demonstrates SUPERIORITY!")
        print("  → Geometric Mean ξ < 1.0 AND Win Rate > 50%")
        print("  → The emulator has successfully corrected the physics!")
    else:
        print("✗ Emulator does not demonstrate superiority.")
        if summary["geometric_mean_xi"] >= 1.0:
            print(f"  → Geometric Mean ξ = {summary['geometric_mean_xi']:.4f} >= 1.0")
        if summary["win_rate"] < 50.0:
            print(f"  → Win Rate = {summary['win_rate']:.1f}% < 50%")
    print("=" * 70)

    # Don't finish wandb here - let the caller (main.py) finish it
    # Only finish if running standalone
    if wandb_run is None:
        wandb.finish()


if __name__ == "__main__":
    main()
    wandb.finish()  # Finish only when run standalone
