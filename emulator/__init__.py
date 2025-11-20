"""Neural emulator for OpenSees soil simulations.

This package implements a PGA emulator that predicts scalar Peak Ground Acceleration
given high-resolution Vs field input. The model uses ResNet-18 architecture and is
trained on low-fidelity labels but evaluated against high-fidelity oracle.

Key components:
- main.py: Main pipeline script (orchestrates data generation, training, evaluation)
- generate_data.py: Generate dataset with HF input, LF targets, and HF oracle
- model.py: ResNet-18 based PGA emulator
- dataset.py: Dataset loader for PGA prediction
- train.py: Training script with early stopping
- evaluate.py: Evaluation with ARE metrics and parity plot

Usage:
    # Run full pipeline
    python emulator/main.py --mode all --n_train 1000 --n_val 100 --n_test 100

    # Or run individual steps
    python emulator/main.py --mode generate  # Generate data only
    python emulator/main.py --mode train    # Train only
    python emulator/main.py --mode evaluate # Evaluate only
"""
