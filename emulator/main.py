"""Main pipeline script for PGA emulator.

Orchestrates the complete workflow:
1. Data generation (HF input, LF targets, HF oracle)
2. Training with early stopping
3. Evaluation with ARE metrics and parity plot
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Get the project root directory (parent of emulator/)
PROJECT_ROOT = Path(__file__).parent.parent
EMULATOR_DIR = Path(__file__).parent

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))

# Import training and evaluation functions
from emulator.evaluate import main as evaluate_main
from emulator.train import main as train_main


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and handle errors.

    Args:
        cmd: Command to run as list of strings
        description: Description of what the command does

    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'=' * 60}")
    print(f"{description}")
    print(f"{'=' * 60}")
    print(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with exit code {result.returncode}")
        return False
    else:
        print(f"\n✅ Success: {description} completed")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="PGA Emulator Pipeline: Generate data, train, and evaluate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with default settings
  python emulator/main.py --mode all

  # Generate data only
  python emulator/main.py --mode generate --n_train 1000 --n_val 100 --n_test 100

  # Train only (assumes data already generated)
  python emulator/main.py --mode train --n_train 1000 --n_val 100 --epochs 100

  # Evaluate only (assumes model already trained)
  python emulator/main.py --mode evaluate --n_test 100

  # Generate + Train
  python emulator/main.py --mode generate_train --n_train 1000 --n_val 100
  
  # Train + Evaluate
  python emulator/main.py --mode train_evaluate --n_train 1000 --n_val 100 --epochs 100
        """,
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=[
            "all",
            "generate",
            "train",
            "evaluate",
            "generate_train",
            "train_evaluate",
        ],
        help="Pipeline mode: 'all' (full pipeline), 'generate', 'train', 'evaluate', etc.",
    )

    # Data generation arguments
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument(
        "--n_train", type=int, default=1000, help="Number of training samples"
    )
    parser.add_argument(
        "--n_val", type=int, default=100, help="Number of validation samples"
    )
    parser.add_argument(
        "--n_test", type=int, default=100, help="Number of test samples"
    )
    parser.add_argument(
        "--train_start_idx",
        type=int,
        default=0,
        help="Starting index for training data",
    )
    parser.add_argument(
        "--val_start_idx",
        type=int,
        default=None,
        help="Starting index for validation data (default: n_train)",
    )
    parser.add_argument(
        "--test_start_idx",
        type=int,
        default=None,
        help="Starting index for test data (default: n_train + n_val)",
    )
    parser.add_argument(
        "--duration", type=float, default=25.0, help="Simulation duration (seconds)"
    )
    parser.add_argument(
        "--dt_lf", type=float, default=0.2, help="LF time step (seconds)"
    )
    parser.add_argument(
        "--dt_hf", type=float, default=0.01, help="HF time step (seconds)"
    )

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
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
    parser.add_argument(
        "--wandb_project", type=str, default="pga_emulator", help="W&B project name"
    )

    # Evaluation arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Model checkpoint path (default: checkpoints/best_model.pt)",
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default="parity_plot.png",
        help="Output path for parity plot",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if sys.platform != "darwin" else "cpu",
        help="Device (cuda/cpu)",
    )

    args = parser.parse_args()

    # Set default indices if not provided
    if args.val_start_idx is None:
        args.val_start_idx = args.n_train
    if args.test_start_idx is None:
        args.test_start_idx = args.n_train + args.n_val

    data_dir = Path(args.data_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Determine which steps to run
    run_generate = args.mode in ["all", "generate", "generate_train"]
    run_train = args.mode in ["all", "train", "generate_train", "train_evaluate"]
    run_evaluate = args.mode in ["all", "evaluate", "train_evaluate"]

    success = True

    # Step 1: Generate data
    if run_generate:
        print("\n" + "=" * 60)
        print("STEP 1: DATA GENERATION")
        print("=" * 60)

        # Import here to avoid issues if seiskit not available
        try:
            from emulator.generate_data import generate_dataset
        except ImportError as e:
            print(f"❌ Error: Could not import generate_data: {e}")
            print("Make sure you're running from the project root directory")
            return 1

        # Generate training data (LF only)
        print(f"\nGenerating training data ({args.n_train} samples)...")
        try:
            generate_dataset(
                data_dir=data_dir,
                n_simulations=args.n_train,
                split="train",
                start_idx=args.train_start_idx,
                run_hf=False,
                duration=args.duration,
                dt_lf=args.dt_lf,
                dt_hf=args.dt_hf,
            )
        except Exception as e:
            print(f"❌ Error generating training data: {e}")
            success = False

        # Generate validation data (LF only)
        print(f"\nGenerating validation data ({args.n_val} samples)...")
        try:
            generate_dataset(
                data_dir=data_dir,
                n_simulations=args.n_val,
                split="val",
                start_idx=args.val_start_idx,
                run_hf=False,
                duration=args.duration,
                dt_lf=args.dt_lf,
                dt_hf=args.dt_hf,
            )
        except Exception as e:
            print(f"❌ Error generating validation data: {e}")
            success = False

        # Generate test data (LF + HF)
        print(f"\nGenerating test data ({args.n_test} samples with HF oracle)...")
        try:
            generate_dataset(
                data_dir=data_dir,
                n_simulations=args.n_test,
                split="test",
                start_idx=args.test_start_idx,
                run_hf=True,  # Generate HF oracle for test set
                duration=args.duration,
                dt_lf=args.dt_lf,
                dt_hf=args.dt_hf,
            )
        except Exception as e:
            print(f"❌ Error generating test data: {e}")
            success = False

        if not success:
            print("\n❌ Data generation failed. Exiting.")
            return 1

    # Step 2: Training
    wandb_run = None
    if run_train:
        print("\n" + "=" * 60)
        print("STEP 2: TRAINING")
        print("=" * 60)

        # Set up training arguments
        import sys as sys_module

        original_argv = sys_module.argv
        try:
            sys_module.argv = [
                "train.py",
                "--data_dir",
                str(data_dir),
                "--batch_size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--checkpoint_dir",
                str(checkpoint_dir),
                "--plots_dir",
                str(plots_dir),
                "--wandb_project",
                args.wandb_project,
                "--n_train",
                str(args.n_train),
                "--n_val",
                str(args.n_val),
                "--device",
                args.device,
            ]

            # Change to project root for execution
            original_cwd = os.getcwd()
            try:
                os.chdir(PROJECT_ROOT)
                wandb_run = train_main(wandb_run=None)
                print("\n✅ Training completed successfully")
            finally:
                os.chdir(original_cwd)
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            return 1
        finally:
            sys_module.argv = original_argv

    # Step 3: Evaluation
    if run_evaluate:
        print("\n" + "=" * 60)
        print("STEP 3: EVALUATION")
        print("=" * 60)

        model_path = args.model_path
        if model_path is None:
            model_path = checkpoint_dir / "best_model.pt"

        # Set up evaluation arguments
        import sys as sys_module

        original_argv = sys_module.argv
        try:
            sys_module.argv = [
                "evaluate.py",
                "--data_dir",
                str(data_dir),
                "--model_path",
                str(model_path),
                "--checkpoint_dir",
                str(checkpoint_dir),
                "--plots_dir",
                str(plots_dir),
                "--wandb_project",
                args.wandb_project,
                "--n_test",
                str(args.n_test),
                "--test_start_idx",
                str(args.test_start_idx),
                "--output_plot",
                str(plots_dir / "parity_plot.png"),
                "--device",
                args.device,
            ]

            # Change to project root for execution
            original_cwd = os.getcwd()
            try:
                os.chdir(PROJECT_ROOT)
                evaluate_main(wandb_run=wandb_run)
                print("\n✅ Evaluation completed successfully")
            finally:
                os.chdir(original_cwd)
        except Exception as e:
            print(f"\n❌ Evaluation failed: {e}")
            return 1
        finally:
            sys_module.argv = original_argv

        # Finish wandb run after evaluation
        if wandb_run is not None:
            import wandb

            wandb.finish()

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    if success:
        print("✅ Pipeline completed successfully!")
        if run_generate:
            print(f"  • Generated data in: {data_dir}")
        if run_train:
            print(f"  • Trained model saved in: {checkpoint_dir}")
        if run_evaluate:
            print("  • Evaluation results and plots saved")
    else:
        print("❌ Pipeline completed with errors")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
