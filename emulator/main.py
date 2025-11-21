"""Main pipeline script for PGA emulator.

Orchestrates the training and evaluation workflow:
1. Training with early stopping
2. Evaluation with ARE metrics and parity plot

Note: Data generation should be done separately using generate_data.py or SLURM scripts.
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

# Import configuration
from emulator.config import (
    default_evaluation_config,
    default_training_config,
)

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
        description="PGA Emulator Pipeline: Train and evaluate (reads config from config.py)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train and evaluate (uses config.py settings)
  python emulator/main.py --mode all

  # Train only (uses config.py settings)
  python emulator/main.py --mode train

  # Evaluate only (uses config.py settings)
  python emulator/main.py --mode evaluate
  
  # Train + Evaluate (uses config.py settings)
  python emulator/main.py --mode train_evaluate

Note: All configuration is read from emulator/config.py.
      Modify config.py to change model architecture, training hyperparameters, etc.
        """,
    )

    # Mode selection only - all other config comes from config.py
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=[
            "all",
            "train",
            "evaluate",
            "train_evaluate",
        ],
        help="Pipeline mode: 'all' (train + evaluate), 'train', 'evaluate', 'train_evaluate'",
    )

    args = parser.parse_args()

    # Load configuration from config.py
    train_config = default_training_config
    eval_config = default_evaluation_config

    # Create directories from config
    checkpoint_dir = Path(train_config.checkpoint_dir)
    plots_dir = Path(train_config.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Determine which steps to run
    run_train = args.mode in ["all", "train", "train_evaluate"]
    run_evaluate = args.mode in ["all", "evaluate", "train_evaluate"]

    success = True

    # Step 1: Training
    wandb_run = None
    if run_train:
        print("\n" + "=" * 60)
        print("STEP 1: TRAINING")
        print("=" * 60)

        # Set up training arguments from config
        import sys as sys_module

        original_argv = sys_module.argv
        try:
            sys_module.argv = [
                "train.py",
                "--data_dir",
                str(train_config.data_dir),
                "--batch_size",
                str(train_config.batch_size),
                "--lr",
                str(train_config.learning_rate),
                "--epochs",
                str(train_config.epochs),
                "--patience",
                str(train_config.patience),
                "--checkpoint_dir",
                str(train_config.checkpoint_dir),
                "--plots_dir",
                str(train_config.plots_dir),
                "--wandb_project",
                train_config.wandb_project,
                "--n_train",
                str(train_config.n_train),
                "--n_val",
                str(train_config.n_val),
                "--device",
                train_config.device,
            ]
            if train_config.wandb_run_id:
                sys_module.argv.extend(["--wandb_run_id", train_config.wandb_run_id])

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

    # Step 2: Evaluation
    if run_evaluate:
        print("\n" + "=" * 60)
        print("STEP 2: EVALUATION")
        print("=" * 60)

        # Use model path from config
        model_path = eval_config.model_path
        if not Path(model_path).is_absolute():
            model_path = str(checkpoint_dir / model_path)

        # Set up evaluation arguments from config
        import sys as sys_module

        original_argv = sys_module.argv
        try:
            sys_module.argv = [
                "evaluate.py",
                "--data_dir",
                str(eval_config.data_dir),
                "--model_path",
                str(model_path),
                "--checkpoint_dir",
                str(eval_config.checkpoint_dir),
                "--plots_dir",
                str(eval_config.plots_dir),
                "--wandb_project",
                eval_config.wandb_project,
                "--n_test",
                str(eval_config.n_test),
                "--test_start_idx",
                str(eval_config.test_start_idx),
                "--output_plot",
                str(Path(eval_config.plots_dir) / eval_config.output_plot),
                "--device",
                eval_config.device,
            ]
            if eval_config.wandb_run_id:
                sys_module.argv.extend(["--wandb_run_id", eval_config.wandb_run_id])

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
