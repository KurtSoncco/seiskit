"""Training script for PGA emulator."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb

# Add parent directory to path to allow importing emulator
sys.path.insert(0, str(Path(__file__).parent.parent))

from emulator.data_utils import compute_normalization_stats, save_normalizer
from emulator.dataset import PGADataset
from emulator.model import PGAEmulator


def compute_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE loss for scalar PGA prediction.

    Args:
        prediction: Model prediction of shape (B, 1) = PGA
        target: Target output (LF PGA) of shape (B, 1)

    Returns:
        MSE loss
    """
    mse_loss = nn.MSELoss()(prediction, target)
    return mse_loss


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train for one epoch.

    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to run on

    Returns:
        Dictionary with loss metrics
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        vs_field = batch["vs_field"].to(device)  # (B, 1, 150, 150)
        target = batch["pga_lf"].to(device)  # (B, 1) - scalar PGA from LF solver

        # Forward pass
        optimizer.zero_grad()
        prediction = model(vs_field)  # (B, 1)

        # Compute loss (MSE on LF PGA labels)
        loss = compute_loss(prediction, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        n_batches += 1

    return {
        "train_loss": total_loss / n_batches,
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Validate model.

    Args:
        model: Model to validate
        dataloader: Validation data loader
        device: Device to run on

    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            vs_field = batch["vs_field"].to(device)  # (B, 1, 150, 150)
            target = batch["pga_lf"].to(device)  # (B, 1) - scalar PGA from LF solver

            # Forward pass
            prediction = model(vs_field)  # (B, 1)

            # Compute loss
            loss = compute_loss(prediction, target)

            # Accumulate metrics
            total_loss += loss.item()
            n_batches += 1

    return {
        "val_loss": total_loss / n_batches,
    }


def main(wandb_run=None):
    """Train PGA emulator.

    Args:
        wandb_run: Optional wandb run to continue logging to. If None, creates a new run.

    Returns:
        wandb run object (for continued logging in evaluation)
    """
    parser = argparse.ArgumentParser(description="Train PGA emulator")
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs)",
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
        "--n_train", type=int, default=1000, help="Number of training samples"
    )
    parser.add_argument(
        "--n_val", type=int, default=100, help="Number of validation samples"
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--plots_dir",
        type=str,
        default="plots",
        help="Directory to save plots",
    )

    args = parser.parse_args()

    # Resolve data_dir to absolute path to avoid issues with working directory changes
    data_dir = Path(args.data_dir).resolve()
    device = torch.device(args.device)

    # Create checkpoint and plots directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Initialize W&B (use existing run if provided)
    if wandb_run is None:
        run = wandb.init(
            project=args.wandb_project,
            job_type="train",
            config={
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "patience": args.patience,
                "device": str(device),
                "n_train": args.n_train,
                "n_val": args.n_val,
            },
        )
    else:
        run = wandb_run
        # Update config if needed
        run.config.update(
            {
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "patience": args.patience,
                "device": str(device),
                "n_train": args.n_train,
                "n_val": args.n_val,
            }
        )

    # First, discover what indices actually exist
    print("Discovering available data indices...")
    materials_hf_dir = data_dir / "materials_hf"
    materials_dir = data_dir / "materials"
    lf_pga_dir = data_dir / "low_fidelity" / "pga"

    # Check which materials directory exists
    if materials_hf_dir.exists():
        mat_dir = materials_hf_dir
    elif materials_dir.exists():
        mat_dir = materials_dir
    else:
        raise ValueError(f"Neither {materials_hf_dir} nor {materials_dir} exists")

    # Find all available simulation indices
    available_indices = set()
    for mat_file in mat_dir.glob("sim_*.npy"):
        # Extract index from filename (e.g., "sim_0000.npy" -> 0)
        try:
            idx = int(mat_file.stem.split("_")[1])
            # Check if corresponding PGA file exists
            pga_file = lf_pga_dir / f"sim_{idx:04d}.npy"
            if pga_file.exists():
                available_indices.add(idx)
        except (ValueError, IndexError):
            continue

    available_indices = sorted(list(available_indices))
    print(
        f"Found {len(available_indices)} available samples at indices: {available_indices[:10]}{'...' if len(available_indices) > 10 else ''}"
    )

    if len(available_indices) == 0:
        raise ValueError(f"No valid data files found in {data_dir}")

    # Determine train/val split
    use_train_val_split = False
    val_indices = list(range(args.n_train, args.n_train + args.n_val))

    # Check if separate validation indices exist
    val_indices_available = [idx for idx in val_indices if idx in available_indices]

    if len(val_indices_available) >= args.n_val:
        # Use separate validation set
        val_indices = val_indices_available[: args.n_val]
        train_indices_for_stats = [
            idx for idx in available_indices if idx < args.n_train
        ][: args.n_train]
        print(
            f"Using separate validation set with {len(val_indices)} samples at indices {val_indices[:10]}{'...' if len(val_indices) > 10 else ''}"
        )
    else:
        # Split available data into train/val
        print(
            f"Warning: Only {len(val_indices_available)} validation samples found at requested indices. "
            f"Splitting available data into train/val split."
        )
        use_train_val_split = True

        # Use available indices, split into train/val
        if len(available_indices) < args.n_train + args.n_val:
            print(
                f"Warning: Only {len(available_indices)} samples available, but requested {args.n_train} train + {args.n_val} val = {args.n_train + args.n_val} samples"
            )
            print(
                f"Using all available samples: {len(available_indices) - args.n_val} train, {args.n_val} val"
            )
            train_indices_for_stats = (
                available_indices[: -args.n_val]
                if len(available_indices) > args.n_val
                else available_indices[:1]
            )
            val_indices = (
                available_indices[-args.n_val :]
                if len(available_indices) > args.n_val
                else available_indices[-1:]
            )
        else:
            train_indices_for_stats = available_indices[: args.n_train - args.n_val]
            val_indices = available_indices[args.n_train - args.n_val : args.n_train]

    # Compute normalization stats from training data (excluding validation if split)
    print("Computing normalization statistics...")
    stats = compute_normalization_stats(data_dir, train_indices=train_indices_for_stats)
    normalizer_path = data_dir / "normalizer.npy"
    save_normalizer(stats, normalizer_path)
    print(f"Saved normalizer to {normalizer_path}")

    # Create datasets with normalization
    if use_train_val_split:
        train_dataset = PGADataset(
            data_dir=data_dir,
            indices=train_indices_for_stats,
            normalizer_path=normalizer_path,
            split="train",
        )
        val_dataset = PGADataset(
            data_dir=data_dir,
            indices=val_indices,
            normalizer_path=normalizer_path,
            split="train",  # Use "train" split since these are from training data
        )
        print(
            f"Using {len(train_dataset)} training samples and {len(val_dataset)} validation samples"
        )
    else:
        # If validation dataset was created successfully, create training dataset normally
        train_dataset = PGADataset(
            data_dir=data_dir,
            indices=list(range(args.n_train)),
            normalizer_path=normalizer_path,
            split="train",
        )
        # Recreate val_dataset with normalizer
        val_dataset = PGADataset(
            data_dir=data_dir,
            indices=val_indices,
            normalizer_path=normalizer_path,
            split="val",
        )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    # Create model
    model = PGAEmulator(
        input_size=(150, 150),
        num_classes=1,
    ).to(device)
    wandb.watch(model)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop with early stopping
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    # Track metrics for plotting
    train_losses = []
    val_losses = []
    epochs_list = []

    for epoch in range(args.epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device)

        # Validate
        val_metrics = validate(model, val_loader, device)

        # Track metrics for plotting
        train_losses.append(train_metrics["train_loss"])
        val_losses.append(val_metrics["val_loss"])
        epochs_list.append(epoch)

        # Log metrics
        metrics = {**train_metrics, **val_metrics, "epoch": epoch}
        wandb.log(metrics)

        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"train_loss={train_metrics['train_loss']:.6f}, "
            f"val_loss={val_metrics['val_loss']:.6f}"
        )

        # Early stopping check
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": args.__dict__,
            }
            best_model_path = checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_model_path)
            wandb.save(str(best_model_path))
            print(
                f"Saved best model to {best_model_path} (val_loss={best_val_loss:.6f})"
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(
                    f"Early stopping triggered at epoch {epoch + 1}. "
                    f"Best model was at epoch {best_epoch + 1} with val_loss={best_val_loss:.6f}"
                )
                break

    # Save final model as artifact
    artifact = wandb.Artifact("model", type="model")
    best_model_path = checkpoint_dir / "best_model.pt"
    artifact.add_file(str(best_model_path))
    artifact.add_file(str(normalizer_path))
    wandb.log_artifact(artifact)

    # Plot and save learning curves
    learning_curve_path = plots_dir / "learning_curves.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs_list, train_losses, label="Train Loss", marker="o", linewidth=2)
    ax.plot(epochs_list, val_losses, label="Validation Loss", marker="s", linewidth=2)
    ax.axvline(
        x=best_epoch,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Best Epoch ({best_epoch + 1})",
    )
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss (MSE)", fontsize=12)
    ax.set_title("Training and Validation Loss Curves", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(learning_curve_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Log learning curve to wandb
    wandb.log({"learning_curves": wandb.Image(str(learning_curve_path))})
    print(f"Saved learning curves to {learning_curve_path}")

    # Return the wandb run instead of finishing it
    # The caller (main.py) will finish it after evaluation
    return run


if __name__ == "__main__":
    run = main()
    wandb.finish()  # Finish only when run standalone
