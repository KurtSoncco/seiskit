import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from seiskit.plot_results import (
    plot_stacked_acceleration,
)
from seiskit.ttf.TTF import TTF


def read_seismolab_data(file_path: str) -> dict:
    """Read SeismoLab data from a file and extract acceleration for each surface node.

    Args:
        file_path: Path to the SeismoLab data file

    Returns:
        Dictionary containing:
            - 'header': dict with num_nodes, num_elements, num_timesteps, etc.
            - 'acceleration': numpy array with shape (num_timesteps, num_nodes * 2)
                              Each row contains [node_0_x, node_0_y, node_1_x, node_1_y, ...]
            - 'acceleration_x': numpy array with x-component acceleration (num_timesteps, num_nodes)
            - 'acceleration_y': numpy array with y-component acceleration (num_timesteps, num_nodes)
    """
    with open(file_path, "r") as f:
        # Read header line
        header_line = f.readline().strip()
        header_values = [int(x) for x in header_line.split()]
        num_nodes, num_elements, num_timesteps, _ = header_values

        # Skip element definition lines (num_elements lines)
        for _ in range(num_elements):
            f.readline()

        # Read time series data
        time_series_data = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Parse all floating point values from the line
            values = [float(x) for x in line.split()]
            if len(values) >= num_nodes * 2:  # Each node has x and y acceleration
                # Take only the first num_nodes * 2 values
                time_series_data.append(values[: num_nodes * 2])
            elif len(values) > 0:
                # Handle lines that might have fewer values (pad with zeros if needed)
                padded_values = values + [0.0] * (num_nodes * 2 - len(values))
                time_series_data.append(padded_values[: num_nodes * 2])

    # Convert to numpy array
    acceleration_array = np.array(time_series_data)

    # Extract x and y components separately
    # x components are at even indices (0, 2, 4, ...)
    # y components are at odd indices (1, 3, 5, ...)
    acceleration_x = acceleration_array[:, 0::2]  # Every other column starting from 0
    acceleration_y = acceleration_array[:, 1::2]  # Every other column starting from 1

    return {
        "header": {
            "num_nodes": num_nodes,
            "num_elements": num_elements,
            "num_timesteps": num_timesteps,
            "num_time_series": len(time_series_data),
        },
        "acceleration": acceleration_array,
        "acceleration_x": acceleration_x,
        "acceleration_y": acceleration_y,
    }


def load_opensees_surface_accel(file_path: str) -> tuple:
    """Load OpenSees surface nodes acceleration data.

    Uses seiskit's load_datasets helper function format for consistency.

    Args:
        file_path: Path to the surface_nodes_dof1_accel.txt file

    Returns:
        Tuple of (time_array, acceleration_array) where:
            - time_array: 1D numpy array of time values
            - acceleration_array: 2D numpy array with shape (num_timesteps, num_nodes)
    """
    # Use np.loadtxt directly (same as load_datasets does internally)
    data = np.loadtxt(file_path)
    time_array = data[:, 0]
    acceleration_array = data[:, 1:]  # All columns except the first (time)
    return time_array, acceleration_array


def resample_to_common_time(
    time1: np.ndarray, accel1: np.ndarray, time2: np.ndarray, accel2: np.ndarray
) -> tuple:
    """Resample two acceleration time series to a common time grid.

    Args:
        time1: Time array for first dataset
        accel1: Acceleration array for first dataset (can be 1D or 2D)
        time2: Time array for second dataset
        accel2: Acceleration array for second dataset (can be 1D or 2D)

    Returns:
        Tuple of (common_time, resampled_accel1, resampled_accel2)
    """
    # Use the finer time grid (more points)
    if len(time1) >= len(time2):
        common_time = time1
        reference_time = time2
        reference_accel = accel2
        target_accel = accel1
    else:
        common_time = time2
        reference_time = time1
        reference_accel = accel1
        target_accel = accel2

    # Resample the dataset with fewer points
    if reference_accel.ndim == 1:
        # Single node
        interp_func = interp1d(
            reference_time, reference_accel, bounds_error=False, fill_value=0.0
        )
        resampled_ref = interp_func(common_time)
        resampled_target = target_accel
    else:
        # Multiple nodes
        resampled_ref = np.zeros((len(common_time), reference_accel.shape[1]))
        for i in range(reference_accel.shape[1]):
            interp_func = interp1d(
                reference_time,
                reference_accel[:, i],
                bounds_error=False,
                fill_value=0.0,
            )
            resampled_ref[:, i] = interp_func(common_time)
        resampled_target = target_accel

    # Return in original order
    if len(time1) >= len(time2):
        return common_time, resampled_target, resampled_ref
    else:
        return common_time, resampled_ref, resampled_target


def calculate_pga(acceleration: np.ndarray) -> np.ndarray:
    """Calculate Peak Ground Acceleration (PGA) for each node.

    Args:
        acceleration: 2D numpy array with shape (num_timesteps, num_nodes)

    Returns:
        1D numpy array with PGA for each node
    """
    return np.max(np.abs(acceleration), axis=0)


def plot_error_comparison(
    time: np.ndarray,
    seismolab_accel: np.ndarray,
    opensees_accel: np.ndarray,
    output_dir: str,
):
    """Plot error comparison between SeismoLab and OpenSees accelerations.

    Args:
        time: Common time array
        seismolab_accel: SeismoLab acceleration array (num_timesteps, num_nodes)
        opensees_accel: OpenSees acceleration array (num_timesteps, num_nodes)
        output_dir: Directory to save plots
    """
    # Calculate errors
    errors = seismolab_accel - opensees_accel
    abs_errors = np.abs(errors)

    # Calculate statistics per node
    mean_abs_error = np.mean(abs_errors, axis=0)
    max_abs_error = np.max(abs_errors, axis=0)
    rmse = np.sqrt(np.mean(errors**2, axis=0))

    # Plot error statistics across nodes
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    node_indices = np.arange(seismolab_accel.shape[1])

    axes[0].plot(
        node_indices, mean_abs_error, "b-o", markersize=4, label="Mean Absolute Error"
    )
    axes[0].set_xlabel("Node Index")
    axes[0].set_ylabel("Mean Absolute Error (m/s²)")
    axes[0].set_title("Mean Absolute Error per Node")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        node_indices, max_abs_error, "r-o", markersize=4, label="Max Absolute Error"
    )
    axes[1].set_xlabel("Node Index")
    axes[1].set_ylabel("Max Absolute Error (m/s²)")
    axes[1].set_title("Maximum Absolute Error per Node")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(node_indices, rmse, "g-o", markersize=4, label="RMSE")
    axes[2].set_xlabel("Node Index")
    axes[2].set_ylabel("RMSE (m/s²)")
    axes[2].set_title("Root Mean Square Error per Node")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    output_file = os.path.join(output_dir, "error_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Error comparison plot saved to {output_file}")

    return {
        "mean_abs_error": mean_abs_error,
        "max_abs_error": max_abs_error,
        "rmse": rmse,
    }


def plot_pga_comparison(
    seismolab_accel: np.ndarray,
    opensees_accel: np.ndarray,
    output_dir: str,
):
    """Plot PGA comparison between SeismoLab and OpenSees.

    Args:
        seismolab_accel: SeismoLab acceleration array (num_timesteps, num_nodes)
        opensees_accel: OpenSees acceleration array (num_timesteps, num_nodes)
        output_dir: Directory to save plots
    """
    # Calculate PGA for each node
    seismolab_pga = calculate_pga(seismolab_accel)
    opensees_pga = calculate_pga(opensees_accel)

    # Calculate PGA error
    pga_error = seismolab_pga - opensees_pga
    pga_error_percent = (pga_error / opensees_pga) * 100

    # Plot PGA comparison
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    node_indices = np.arange(len(seismolab_pga))

    axes[0].plot(node_indices, seismolab_pga, "b-o", markersize=4, label="SeismoLab")
    axes[0].plot(node_indices, opensees_pga, "r-o", markersize=4, label="OpenSees")
    axes[0].set_xlabel("Node Index")
    axes[0].set_ylabel("PGA (m/s²)")
    axes[0].set_title("Peak Ground Acceleration Comparison")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(node_indices, pga_error, "g-o", markersize=4, label="PGA Error")
    axes[1].axhline(y=0, color="k", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Node Index")
    axes[1].set_ylabel("PGA Error (m/s²)")
    axes[1].set_title("PGA Error (SeismoLab - OpenSees)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(
        node_indices, pga_error_percent, "m-o", markersize=4, label="PGA Error %"
    )
    axes[2].axhline(y=0, color="k", linestyle="--", linewidth=1)
    axes[2].set_xlabel("Node Index")
    axes[2].set_ylabel("PGA Error (%)")
    axes[2].set_title("PGA Error Percentage")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    output_file = os.path.join(output_dir, "pga_comparison.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"PGA comparison plot saved to {output_file}")

    return {
        "seismolab_pga": seismolab_pga,
        "opensees_pga": opensees_pga,
        "pga_error": pga_error,
        "pga_error_percent": pga_error_percent,
    }


def plot_transfer_functions_opensees(
    base_file: str,
    surface_file: str,
    output_dir: str,
    dt: float = 1e-3,
    dz: float = 5.0,
):
    """Plot transfer function for OpenSees using center nodes at base (y=0) and surface (y=150).

    Args:
        base_file: Path to base acceleration file (center_node_y0.00_dof1_accel.txt)
        surface_file: Path to surface acceleration file (center_node_y150.00_dof1_accel.txt)
        output_dir: Directory to save plots
        dt: Time step (default: 1e-3)
        dz: Depth increment (default: 5.0)
    """
    # Load base acceleration (y=0) - center node
    base_data = np.loadtxt(base_file)
    base_time = base_data[:, 0]
    base_accel = base_data[:, 1]  # Single column for center node at y=0

    # Load surface acceleration (y=150) - center node
    surface_data = np.loadtxt(surface_file)
    surface_time = surface_data[:, 0]
    surface_accel = surface_data[:, 1]  # Single column for center node at y=150

    # Ensure time arrays match
    if not np.allclose(base_time, surface_time):
        print("Warning: Base and surface time arrays don't match. Using surface time.")
        common_time = surface_time
        # Resample base acceleration to match surface time
        interp_func = interp1d(
            base_time, base_accel, bounds_error=False, fill_value=0.0
        )
        base_accel = interp_func(common_time)
    else:
        common_time = base_time

    # Calculate transfer function for center nodes
    try:
        freq, tf = TTF(
            surface_acc=surface_accel,
            base_acc=base_accel,
            dt=dt,
            dz=dz,
        )

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.loglog(freq, tf, linewidth=2, label="Center Node (y=150 / y=0)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Transfer Function")
        ax.set_title("OpenSees Transfer Function (Center Surface / Center Base)")
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlim(0.1, 2.5)
        ax.legend(loc="best", fontsize=10)

        plt.tight_layout()
        output_file = os.path.join(output_dir, "transfer_functions_opensees.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Transfer function plot saved to {output_file}")
    except Exception as e:
        print(f"Error: Could not compute transfer function: {e}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    file_path = "results/Case_4b/Surface.0.out"
    data = read_seismolab_data(file_path)

    print("Header information:")
    for key, value in data["header"].items():
        print(f"  {key}: {value}")

    print(f"\nAcceleration data shape: {data['acceleration'].shape}")
    print(f"X-component acceleration shape: {data['acceleration_x'].shape}")
    print(f"Y-component acceleration shape: {data['acceleration_y'].shape}")
    print(f"\nNumber of surface nodes: {data['header']['num_nodes']}")
    print("\nFirst 5 timesteps, first 5 nodes - X-component acceleration:")
    print(data["acceleration_x"][:5, :5])
    print("\nFirst 5 timesteps, first 5 nodes - Y-component acceleration:")
    print(data["acceleration_y"][:5, :5])

    # Create time array for plotting
    # Based on run_case.py: duration=15.0s, dt=1e-3, but SeismoLab output is subsampled
    num_time_steps = data["acceleration_x"].shape[0]
    duration = 15.0  # seconds (from run_case.py)
    time_array = np.linspace(0, duration, num_time_steps)

    # Format data for plot_stacked_acceleration
    # It expects: datasets = {"model_name": {"surface": (time, acceleration)}}
    # where time is 1D array and acceleration is 2D array (num_timesteps, num_nodes)
    datasets = {
        "SeismoLab": {
            "surface": (
                time_array,
                data["acceleration_x"],
            ),  # (time, acceleration) tuple
        }
    }

    # Plot the surface acceleration
    plot_stacked_acceleration(
        datasets,
        data_config={
            "SeismoLab": {
                "surface": file_path,
            },
        },
        scale_factor=5.0,
        vertical_spacing=2.5,
        alpha=0.3,
    )

    # Load OpenSees surface acceleration data
    opensees_file = "results/Case_4b/surface_nodes_dof1_accel.txt"
    opensees_time, opensees_accel = load_opensees_surface_accel(opensees_file)

    print("\nOpenSees data loaded:")
    print(f"  Time array shape: {opensees_time.shape}")
    print(f"  Acceleration array shape: {opensees_accel.shape}")

    # Resample to common time grid
    common_time, seismolab_resampled, opensees_resampled = resample_to_common_time(
        time_array, data["acceleration_x"], opensees_time, opensees_accel
    )

    print("\nAfter resampling:")
    print(f"  Common time shape: {common_time.shape}")
    print(f"  SeismoLab acceleration shape: {seismolab_resampled.shape}")
    print(f"  OpenSees acceleration shape: {opensees_resampled.shape}")

    # Ensure same number of nodes
    min_nodes = min(seismolab_resampled.shape[1], opensees_resampled.shape[1])
    seismolab_resampled = seismolab_resampled[:, :min_nodes]
    opensees_resampled = opensees_resampled[:, :min_nodes]

    print(f"  Using {min_nodes} nodes for comparison")

    # Create output directory
    output_dir = "results/Case_4b"
    os.makedirs(output_dir, exist_ok=True)

    # Compare errors between surface nodes
    print("\n" + "=" * 60)
    print("Comparing errors between SeismoLab and OpenSees")
    print("=" * 60)
    error_stats = plot_error_comparison(
        common_time, seismolab_resampled, opensees_resampled, output_dir
    )

    # Compare PGA
    print("\n" + "=" * 60)
    print("Comparing PGA between SeismoLab and OpenSees")
    print("=" * 60)
    pga_stats = plot_pga_comparison(seismolab_resampled, opensees_resampled, output_dir)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    print(
        f"Mean Absolute Error - Mean: {np.mean(error_stats['mean_abs_error']):.6e} m/s²"
    )
    print(
        f"Mean Absolute Error - Max: {np.max(error_stats['mean_abs_error']):.6e} m/s²"
    )
    print(
        f"Max Absolute Error - Mean: {np.mean(error_stats['max_abs_error']):.6e} m/s²"
    )
    print(f"Max Absolute Error - Max: {np.max(error_stats['max_abs_error']):.6e} m/s²")
    print(f"RMSE - Mean: {np.mean(error_stats['rmse']):.6e} m/s²")
    print(f"RMSE - Max: {np.max(error_stats['rmse']):.6e} m/s²")
    print(f"\nPGA Error - Mean: {np.mean(np.abs(pga_stats['pga_error'])):.6e} m/s²")
    print(f"PGA Error - Max: {np.max(np.abs(pga_stats['pga_error'])):.6e} m/s²")
    print(f"PGA Error % - Mean: {np.mean(np.abs(pga_stats['pga_error_percent'])):.2f}%")
    print(f"PGA Error % - Max: {np.max(np.abs(pga_stats['pga_error_percent'])):.2f}%")

    # Plot transfer functions (OpenSees only - requires base and surface center nodes)
    print("\n" + "=" * 60)
    print("Plotting transfer functions (OpenSees only - center nodes)")
    print("=" * 60)
    base_file = "results/Case_4b/center_node_y0.00_dof1_accel.txt"
    surface_file = "results/Case_4b/center_node_y150.00_dof1_accel.txt"
    plot_transfer_functions_opensees(
        base_file=base_file,
        surface_file=surface_file,
        output_dir=output_dir,
        dt=1e-3,
        dz=5.0,
    )
