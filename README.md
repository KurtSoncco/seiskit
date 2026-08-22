# Seiskit
<p align="left">
   <a href="https://github.com/KurtSoncco/seiskit/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/KurtSoncco/seiskit"></a>
   <a href="https://github.com/KurtSoncco/seiskit"><img alt="GitHub stars" src="https://img.shields.io/github/stars/KurtSoncco/seiskit?style=social"></a>
   <a href="https://github.com/KurtSoncco/seiskit/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/KurtSoncco/seiskit"></a>
   <a href="https://www.python.org/downloads/"><img alt="Python versions" src="https://img.shields.io/badge/python-3.10%2B-blue"></a>
   <a href="https://github.com/KurtSoncco/seiskit/commits/main"><img alt="Last commit" src="https://img.shields.io/github/last-commit/KurtSoncco/seiskit"></a>
</p>

**Seiskit** is a Python package for conducting seismic analyses using OpenSees, with a focus on spatial variability in soil wave propagation. The package provides powerful parallel processing capabilities for running multiple seismic analyses concurrently, making it ideal for parameter studies and large-scale simulations.

## Features

- **1D and 2D site response** via OpenSees (`boundary_condition_type="1D"` or `"2D"`)
- **Closed-form 1D TFs** (`seiskit.theory`) for multilayer validation
- **Spatial variability** (GRF / profile randomization) on 2D domains
- **Parallel execution** with process-isolated OpenSees runs
- **Parameter studies** and transfer-function post-processing

## Installation

### Prerequisites

- Python 3.10 or higher
- Git

### Option 1: Using pyenv (Recommended)

1. **Install pyenv** (if not already installed):
   ```bash
   # On Ubuntu/Debian
   curl https://pyenv.run | bash
   
   # Add to your shell profile (~/.bashrc or ~/.zshrc)
   echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
   echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
   echo 'eval "$(pyenv init -)"' >> ~/.bashrc
   
   # Reload shell
   source ~/.bashrc
   ```

2. **Install Python 3.11**:
   ```bash
   pyenv install 3.11.7
   pyenv global 3.11.7
   ```

3. **Clone and setup the repository**:
   ```bash
   git clone https://github.com/KurtSoncco/seiskit.git
   cd seiskit
   pyenv local 3.11.7
   ```

4. **Install dependencies with uv**:
   ```bash
   # Install uv if not already installed
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Minimal CPU runtime (recommended for HPC)
   uv sync

   # Optional: add research/data analysis stack
   # uv sync --extra analysis

   # Optional: add ML/interpretability stack
   # uv sync --extra ml

   # Optional: add notebook/plot export tooling
   # uv sync --extra notebook
   ```

5. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate  # On Linux/Mac
   # or
   .venv\Scripts\activate     # On Windows
   ```

### Option 2: Using uv (Fastest)

1. **Install uv**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and setup**:
   ```bash
   git clone https://github.com/KurtSoncco/seiskit.git
   cd seiskit

   # Minimal CPU runtime
   uv sync

   # Example: full local research environment
   # uv sync --extra analysis --extra ml --extra notebook --extra dev
   ```

3. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate
   ```

### Option 3: Traditional pip/venv

1. **Create virtual environment**:
   ```bash
   python -m venv seiskit-env
   source seiskit-env/bin/activate  # On Linux/Mac
   ```

2. **Install dependencies**:
   ```bash
   # Minimal CPU runtime
   pip install -e .

   # Optional extras
   # pip install -e .[analysis]
   # pip install -e .[ml]
   # pip install -e .[notebook]
   # pip install -e .[dev]
   ```

## Quick Start

### Basic Usage

```python
from seiskit import run_opensees_analysis, build_model_data, AnalysisConfig
import numpy as np

# Load material data
vs_data = np.loadtxt("vs_data.txt")
rho_data = np.loadtxt("rho_data.txt")
nu_data = np.loadtxt("nu_data.txt")

# Create analysis configuration
# 1D: single column, simple shear — use Lx=hx and boundary_condition_type="1D"
# 2D: wide free-field domain — boundary_condition_type="2D" (default for GRF studies)
config = AnalysisConfig(
    Ly=140.0,
    Lx=260.0,
    hx=5.0,
    duration=15.0,
    motion_freq=0.75,
    boundary_condition_type="2D",
)

# Build model data
model_data = build_model_data(config, vs_data, rho_data, nu_data)

# Run analysis
result = run_opensees_analysis(
    config=config, model_data=model_data, run_id="example_run", output_dir="results"
)

print(f"Analysis completed: {result}")
```

### Parallel Processing

Seiskit provides two parallel processing approaches:

#### 1. ProcessPoolExecutor-based Parallel Processing

```python
from seiskit import run_analyses_parallel, load_material_properties

# Load material data
material_data = load_material_properties(
    {"vs": "vs_data.txt", "rho": "rho_data.txt", "nu": "nu_data.txt"}
)

# Define multiple analyses
configs = [
    {"task_id": "coarse", "hx": 10.0, "duration": 10.0},
    {"task_id": "medium", "hx": 5.0, "duration": 10.0},
    {"task_id": "fine", "hx": 2.5, "duration": 10.0},
]

# Run in parallel
results = run_analyses_parallel(configs, material_data, max_workers=3)

# Check results
for result in results:
    print(f"{result.task_id}: {result.status}")
```

#### 2. Joblib-based Parallel Processing

```python
from seiskit.joblib_parallel import run_analyses_joblib_parallel

# Run analyses with joblib
results = run_analyses_joblib_parallel(configs, material_data, n_jobs=4, backend="loky", verbose=1)
```

#### Parameter Studies

```python
from seiskit import run_parameter_study

# Base configuration
base_config = {"Ly": 140.0, "Lx": 260.0, "duration": 15.0}

# Parameter variations
variations = {
    "hx": [2.5, 5.0, 10.0],  # 3 mesh sizes
    "motion_freq": [0.5, 0.75, 1.0],  # 3 frequencies
}

# Run parameter study (9 combinations total)
results = run_parameter_study(base_config, variations, material_data, max_workers=4)
```

## How Parallel Processing Works

Seiskit implements true parallel processing with complete process isolation to avoid OpenSees global state conflicts. Here's how it works:

### Process Isolation

1. **Separate Processes**: Each analysis runs in its own Python process
2. **Independent OpenSees Instances**: Each process creates its own OpenSees instance
3. **No Shared State**: Complete isolation prevents conflicts between analyses
4. **Safe Data Preparation**: Model data is prepared in the main process before parallel execution

### Architecture

```
Main Process
├── Data Preparation (parallel-safe)
│   ├── Load material properties
│   ├── Build model data
│   └── Create analysis tasks
└── Parallel Execution
    ├── Process 1: Analysis Task A
    ├── Process 2: Analysis Task B
    ├── Process 3: Analysis Task C
    └── Process N: Analysis Task N
```

### Key Components

- **`AnalysisTask`**: Contains all data needed for a single analysis
- **`AnalysisResult`**: Represents the outcome of an analysis
- **`prepare_analysis_tasks()`**: Prepares tasks in the main process
- **`run_parallel_analyses()`**: Executes tasks in parallel processes
- **`_run_isolated_analysis()`**: Runs a single analysis in isolation

### Benefits

- ✅ **Scalability**: Utilize all available CPU cores
- ✅ **Reliability**: Process isolation prevents crashes from affecting other analyses
- ✅ **Progress Tracking**: Monitor execution progress in real-time
- ✅ **Error Handling**: Individual analysis failures don't stop the entire batch
- ✅ **Memory Efficiency**: Each process manages its own memory

### Performance Considerations

- **CPU-bound**: Parallel processing is most effective for CPU-intensive analyses
- **Memory Usage**: Each process uses its own memory space
- **I/O Bottlenecks**: Disk I/O can become a bottleneck with many parallel processes
- **Optimal Workers**: Typically use `min(CPU_count, number_of_tasks)` workers

## Project Structure

```
seiskit/
├── seiskit/                 # Main package
│   ├── __init__.py         # Package initialization
│   ├── analysis.py         # Core analysis functions
│   ├── builder.py          # Model building utilities
│   ├── config.py           # Configuration management
│   ├── parallel.py         # ProcessPoolExecutor parallel processing
│   ├── joblib_parallel.py  # Joblib-based parallel processing
│   ├── isolated_runner.py  # Isolated analysis execution
│   └── recorders.py        # Data recording utilities
├── examples/               # Example scripts and tutorials
├── tests/                  # Test suite
├── comparison/             # Comparison studies
├── results/                # Analysis results (generated)
├── pyproject.toml         # Project configuration
├── uv.lock                # Dependency lock file
└── README.md              # This file
```

## Examples

Check the `examples/` directory for comprehensive examples:

- **Basic Analysis**: Simple seismic analysis setup
- **Parallel Processing**: Running multiple analyses concurrently
- **Parameter Studies**: Systematic parameter variation studies
- **Spatial Variability**: Modeling spatially variable soil properties

## Dependencies

- **Core (default, CPU-friendly)**: numpy, scipy, openseespy, joblib, numba, matplotlib, seaborn, plotly
- **Analysis extra**: pandas, h5py, hdf5plugin, tqdm, statsmodels, kerchunk, zarr, xarray, fsspec, ujson
- **ML extra**: scikit-learn, xgboost, shap, interpret, pygam, bambi, arviz
- **Notebook extra**: ipykernel, kaleido
- **Development extra**: ruff, pytest, pytest-cov

### Install profiles

- **Minimal HPC runtime (recommended for cluster jobs)**
   - `uv sync`
   - or `pip install -e .`
- **Local research/data workflow**
   - `uv sync --extra analysis --extra notebook`
   - or `pip install -e .[analysis,notebook]`
- **Full local workflow with ML tools**
   - `uv sync --extra analysis --extra ml --extra notebook --extra dev`
   - or `pip install -e .[analysis,ml,notebook,dev]`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `pytest`
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

- **Author**: Kurt Walter Soncco Sinchi
- **Email**: kurtwal98@berkeley.edu
- **Institution**: University of California, Berkeley

## Acknowledgments

- OpenSees development team for the excellent finite element framework
- The Python scientific computing community
- Contributors and users of the seiskit package

---

For more detailed information about parallel processing, see `PARALLEL_README.md`.
