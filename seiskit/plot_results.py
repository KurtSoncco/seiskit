"""
Lightweight plotting helpers that wrap Plotly and Matplotlib for generating
seismic analysis comparison plots.

All figures follow the centralized publication-quality style defined in
:mod:`seiskit.plot_config`.
"""

import itertools
import sys
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Mapping, Optional, Tuple, Union

import numpy as np
from scipy.interpolate import interp1d

if TYPE_CHECKING:
    from seiskit.config import AnalysisConfig

from seiskit.damping import (
    compute_average_damping_harmonic,
    compute_damping_from_Q,
    compute_quality_factor,
)
from seiskit.plot_config.colormaps import get_crameri_cmap
from seiskit.plot_config.helpers import add_subfigure_label, format_title
from seiskit.plot_config.labels import format_label, to_title_case
from seiskit.plot_config.style import (
    COLORBLIND_COLORS,
    apply_style,
    get_plotly_layout,
)
from seiskit.ttf.TTF import TTF

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    go = None

    def _plotly_subplots_stub(*args, **kwargs):
        raise ImportError("plotly is not available; install plotly to enable plotting")

    make_subplots = _plotly_subplots_stub
    PLOTLY_AVAILABLE = False

# --- Data Structures ---
TimeSeries = Tuple[np.ndarray, np.ndarray]
TimeSeriesMulti = Tuple[np.ndarray, np.ndarray]
ModelResult = Dict[str, Union[TimeSeries, TimeSeriesMulti]]
DataSet = Dict[str, ModelResult]
PathType = Union[str, Path, PathLike]

# Deterministic colour mapping for known models
MODEL_COLORS: dict[str, str] = {
    "My_New_Run": COLORBLIND_COLORS[0],
    "PLAXIS": COLORBLIND_COLORS[1],
    "OpenSeesPy_Prev": COLORBLIND_COLORS[2],
}
FALLBACK_COLORS = itertools.cycle(COLORBLIND_COLORS)


def load_datasets(config: Mapping[str, Mapping[str, PathType]]) -> DataSet:
    """Loads multiple datasets from file paths defined in a config dictionary.

    Args:
        config: A mapping where keys are model names (e.g., "PLAXIS")
                and values are mappings from location ("base", "top", "surface") to a
                filesystem path.

    Returns:
        A dictionary containing the loaded and structured data.

    Raises:
        FileNotFoundError: If a file specified in the config does not exist.
    """
    datasets: DataSet = {}
    for model_name, paths in config.items():
        datasets[model_name] = {}
        for location, file_path in paths.items():
            if location in ("base", "top", "surface"):
                try:
                    data = np.loadtxt(file_path)
                    if data.size == 0:
                        raise ValueError(f"File {file_path} is empty")
                    if data.ndim == 1:
                        raise ValueError(
                            f"File {file_path} contains 1D data (expected 2D with time and values)"
                        )
                    if location == "surface" and data.shape[1] > 2:
                        datasets[model_name][location] = (data[:, 0], data[:, 1:])
                    else:
                        datasets[model_name][location] = (data[:, 0], data[:, 1])
                except ValueError as e:
                    raise ValueError(
                        f"Error loading {file_path}: {e}. File may be empty or malformed."
                    ) from e
    return datasets


def _resample_and_calculate_difference(reference: TimeSeries, comparison: TimeSeries) -> TimeSeries:
    """Resamples a comparison series and computes the difference with a reference."""
    ref_time, ref_accel = reference
    comp_time, comp_accel = comparison
    interp_func = interp1d(comp_time, comp_accel, bounds_error=False, fill_value=0.0)
    resampled_comp_accel = interp_func(ref_time)
    difference = ref_accel - resampled_comp_accel
    return (ref_time, difference)


def _setup_figure(title: str, subplot_titles: Tuple[str, str], y_axis_title: str):
    """Creates a standard 2-row Plotly figure with shared axes and publication style."""
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.1,
    )
    fig.update_yaxes(title_text=to_title_case(y_axis_title))
    fig.update_xaxes(title_text=to_title_case("Time (S)"), row=2, col=1)
    fig.update_layout(
        height=700,
        width=1000,
        title_text=f"<b>{title}</b>",
        **get_plotly_layout(),
    )
    return fig


def plot_acceleration_comparison(
    datasets: DataSet, reference_name: str, output_path: Path, show_fig: bool = False
) -> None:
    """Creates and saves a Plotly chart comparing acceleration time histories."""
    if not PLOTLY_AVAILABLE:
        print("Plotly is not installed. Skipping plot generation.")
        return

    fig = _setup_figure(
        title="Comparison Of Acceleration Time Histories",
        subplot_titles=(
            to_title_case("Top (Surface) Acceleration"),
            to_title_case("Base Acceleration"),
        ),
        y_axis_title="Acceleration ($m/s^2$)",
    )

    line_styles = ["dash", "dot", "dashdot"]
    fallback_colors = COLORBLIND_COLORS

    all_model_names = list(datasets.keys())
    model_styles: dict[str, dict] = {}
    for i, model_name in enumerate(all_model_names):
        color = MODEL_COLORS.get(model_name, fallback_colors[i % len(fallback_colors)])
        line_style = line_styles[i % len(line_styles)] if model_name != reference_name else "solid"
        model_styles[model_name] = {"color": color, "line_style": line_style}

    location_to_row = {"top": 1, "base": 2}
    for location in ["top", "base"]:
        row = location_to_row[location]
        for model_name, model_data in datasets.items():
            if model_name == reference_name or location not in model_data:
                continue
            time, accel = model_data[location]
            style = model_styles[model_name]
            fig.add_trace(
                go.Scatter(  # type: ignore[call-arg]
                    x=time,
                    y=accel,
                    mode="lines",
                    name=format_label(f"{model_name} ({location})"),
                    line=dict(dash=style["line_style"], color=style["color"]),
                    showlegend=True,
                ),
                row=row,
                col=1,
            )

        if location in datasets.get(reference_name, {}):
            time, accel = datasets[reference_name][location]
            style = model_styles[reference_name]
            fig.add_trace(
                go.Scatter(  # type: ignore[call-arg]
                    x=time,
                    y=accel,
                    mode="lines",
                    name=format_label(f"{reference_name} ({location})"),
                    line=dict(color=style["color"], width=2.5),
                    showlegend=True,
                ),
                row=row,
                col=1,
            )

    fig.write_html(str(output_path))
    print(f"Comparison plot saved to {output_path}")
    if show_fig:
        fig.show()


def plot_acceleration_differences(
    datasets: DataSet, reference_name: str, output_path: Path, show_fig: bool = False
) -> None:
    """Computes and plots the difference between a reference and other datasets."""
    if not PLOTLY_AVAILABLE:
        print("Plotly is not installed. Skipping plot generation.")
        return

    if reference_name not in datasets:
        raise ValueError(f"Reference '{reference_name}' not found in datasets.")

    fig = _setup_figure(
        title=f"Acceleration Difference Relative To {reference_name}",
        subplot_titles=(
            to_title_case("Difference At Base"),
            to_title_case("Difference At Top (Surface)"),
        ),
        y_axis_title="Acceleration Difference ($m/s^2$)",
    )

    reference_data = datasets[reference_name]
    for i, location in enumerate(["base", "top"], 1):
        if location not in reference_data:
            continue
        for model_name, model_data in datasets.items():
            if model_name == reference_name or location not in model_data:
                continue
            diff_time, diff_accel = _resample_and_calculate_difference(
                reference_data[location], model_data[location]
            )
            color = MODEL_COLORS.get(model_name, next(FALLBACK_COLORS))
            fig.add_trace(
                go.Scatter(  # type: ignore[call-arg]
                    x=diff_time,
                    y=diff_accel,
                    mode="lines",
                    name=format_label(f"{reference_name} - {model_name} ({location})"),
                    line=dict(color=color),
                ),
                row=i,
                col=1,
            )

    fig.write_html(str(output_path))
    print(f"Difference plot saved to {output_path}")
    if show_fig:
        fig.show()


def plot_stacked_acceleration(
    datasets: DataSet,
    data_config: Mapping[str, Mapping[str, PathType]],
    scale_factor: float = 5.0,
    vertical_spacing: float = 2.5,
    alpha: float = 0.3,
) -> None:
    """Plot stacked acceleration time histories for multiple surface nodes."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is not installed. Skipping plot generation.")
        return

    apply_style()

    for model_name, model_data in datasets.items():
        if "surface" not in model_data:
            continue

        time, accel = model_data["surface"]

        if model_name not in data_config or "surface" not in data_config[model_name]:
            continue

        surface_file_path = Path(data_config[model_name]["surface"])
        output_dir = surface_file_path.parent
        output_file = output_dir / f"{model_name}_surface_nodes_acceleration_stacked.png"

        fig, ax = plt.subplots(figsize=(12, 8))

        if accel.ndim == 2 and accel.shape[1] > 1:
            num_nodes = accel.shape[1]
            vertical_offset = 0.0
            for i in range(num_nodes):
                scaled_accel = accel[:, i] * scale_factor + vertical_offset
                ax.plot(scaled_accel, time, color="gray", alpha=alpha, linewidth=0.8)
                vertical_offset += vertical_spacing
        else:
            if accel.ndim == 2:
                accel = accel[:, 0]
            scaled_accel = accel * scale_factor
            ax.plot(scaled_accel, time, color="gray", alpha=alpha, linewidth=0.8)

        format_title(f"{model_name} — Surface Nodes Acceleration X (Stacked And Scaled)", ax=ax)
        ax.set_xlabel(to_title_case("Acceleration + Offset (m/s²)"))
        ax.set_ylabel(to_title_case("Time (S)"))
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 15)

        add_subfigure_label(ax, 0)

        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Stacked acceleration plot saved to {output_file}")


def plot_transfer_functions(
    datasets: DataSet,
    reference_name: str,
    output_path: Path,
    show_fig: bool = False,
    dz: float = 5.0,
) -> None:
    """Plot transfer functions between base and top accelerations for each model."""
    if not PLOTLY_AVAILABLE:
        print("Plotly is not installed. Skipping plot generation.")
        return

    fig = make_subplots(
        rows=1,
        cols=1,
        subplot_titles=(to_title_case("Transfer Functions (Top/Base)"),),
    )

    for model_name, model_data in datasets.items():
        if "base" not in model_data or "top" not in model_data:
            continue
        base_time, base_accel = model_data["base"]
        _, top_accel = model_data["top"]
        freq_model, tf_model = TTF(top_accel, base_accel, dt=base_time[1] - base_time[0], dz=dz)

        color = MODEL_COLORS.get(model_name, next(FALLBACK_COLORS))
        fig.add_trace(
            go.Scatter(  # type: ignore[call-arg]
                x=freq_model,
                y=tf_model,
                mode="lines",
                name=format_label(f"{model_name} TF"),
                line=dict(color=color),
            ),
            row=1,
            col=1,
        )

    fig.update_xaxes(title_text=to_title_case("Frequency (Hz)"), type="log", row=1, col=1)
    fig.update_yaxes(
        title_text=to_title_case("Transfer Function Magnitude"), type="log", row=1, col=1
    )
    fig.update_layout(
        height=600,
        width=800,
        title_text="<b>Transfer Functions From Base To Top</b>",
        **get_plotly_layout(),
    )

    fig.write_html(str(output_path))
    print(f"Transfer function plot saved to {output_path}")
    if show_fig:
        fig.show()


def get_damping_zeta_grid(
    Vs_extended: np.ndarray,
    damping_method: str,
    Lx: float,
    Lz: float,
    dx: float,
    dz: float,
    bedrock_mask: np.ndarray | None = None,
    damping_zeta: float | None = None,
    damping_freqs: tuple[float, float] | None = None,
    damping_f_target: float | None = None,
    config: "AnalysisConfig | None" = None,
) -> np.ndarray:
    """Compute damping ratio (zeta) grid for each element."""
    if config is not None:
        damping_zeta = damping_zeta if damping_zeta is not None else config.damping_zeta
        damping_freqs = damping_freqs if damping_freqs is not None else config.damping_freqs
        damping_f_target = (
            damping_f_target if damping_f_target is not None else config.damping_f_target
        )
    else:
        damping_zeta = damping_zeta if damping_zeta is not None else 0.02
        damping_f_target = damping_f_target if damping_f_target is not None else 0.75

    nz, nx = Vs_extended.shape
    zeta_grid = np.zeros_like(Vs_extended)

    if damping_method == "global_avg":
        if bedrock_mask is None:
            raise ValueError(
                "bedrock_mask is required for 'global_avg' damping method. "
                "Provide bedrock_mask from create_vs_realization() for accurate bedrock/soil identification."
            )
        if bedrock_mask.shape != (nz, nx):
            raise ValueError(
                f"bedrock_mask shape {bedrock_mask.shape} != Vs_extended shape {(nz, nx)}"
            )
        soil_mask = ~bedrock_mask
        soil_Vs = Vs_extended[soil_mask]
        if len(soil_Vs) > 0:
            Q_values_soil = [compute_quality_factor(vs) for vs in soil_Vs]
            avg_damping_soil = compute_average_damping_harmonic(Q_values_soil)
            zeta_grid[soil_mask] = avg_damping_soil
        if np.any(bedrock_mask):
            bedrock_Vs = 1500.0
            Q_bedrock = compute_quality_factor(bedrock_Vs)
            xi_bedrock = compute_damping_from_Q(Q_bedrock)
            zeta_grid[bedrock_mask] = xi_bedrock

    elif damping_method in ("elemental_varying", "elemental_mass_only"):
        for i in range(nz):
            for j in range(nx):
                vs = Vs_extended[i, j]
                Q = compute_quality_factor(vs)
                xi = compute_damping_from_Q(Q)
                zeta_grid[i, j] = xi
    elif damping_method == "uniform":
        zeta_grid.fill(damping_zeta)
    elif damping_method == "uniform_soil_only":
        if bedrock_mask is None:
            raise ValueError("bedrock_mask is required for 'uniform_soil_only' damping method.")
        if bedrock_mask.shape != (nz, nx):
            raise ValueError(
                f"bedrock_mask shape {bedrock_mask.shape} != Vs_extended shape {(nz, nx)}"
            )
        soil_mask = ~bedrock_mask
        zeta_grid[soil_mask] = damping_zeta
        if np.any(bedrock_mask):
            zeta_grid[bedrock_mask] = 0.0075
    else:
        raise ValueError(f"Unknown damping method: {damping_method}")

    return zeta_grid


def plot_damping_realization(
    Vs_extended: np.ndarray,
    damping_method: str,
    Lx: float,
    Lz: float,
    dx: float,
    dz: float,
    save_path: PathType,
    title: str | None = None,
    bedrock_mask: np.ndarray | None = None,
    damping_zeta: float | None = None,
    damping_freqs: tuple[float, float] | None = None,
    damping_f_target: float | None = None,
    config: "AnalysisConfig | None" = None,
) -> None:
    """Plot damping zeta values for each element based on the damping method."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is not installed. Skipping plot generation.")
        return

    apply_style()

    zeta_grid = get_damping_zeta_grid(
        Vs_extended,
        damping_method,
        Lx,
        Lz,
        dx,
        dz,
        bedrock_mask=bedrock_mask,
        damping_zeta=damping_zeta,
        damping_freqs=damping_freqs,
        damping_f_target=damping_f_target,
        config=config,
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    extent = (0, Lx, Lz, 0)
    vmin = zeta_grid.min()
    vmax = zeta_grid.max()

    cmap = get_crameri_cmap("batlow")

    im = ax.imshow(
        zeta_grid,
        extent=extent,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"Damping Ratio $\zeta$")

    ax.set_xlabel(to_title_case("Distance (M)"))
    ax.set_ylabel(to_title_case("Depth (M)"))
    format_title(
        title or f"Damping Realization ({damping_method.replace('_', ' ').title()})",
        ax=ax,
    )
    ax.grid(False)
    add_subfigure_label(ax, 0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_realization(
    Vs_1D_profile: np.ndarray,
    Vs_realization: np.ndarray,
    Lx: float,
    Lz: float,
    dx: float,
    dz: float,
    save_path: Optional[PathType] = None,
    title: Optional[str] = None,
    bedrock_mask: Optional[np.ndarray] = None,
) -> None:
    """Plot the Vs realization with a colour scale focused on the soil layer."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib is not installed. Skipping plot generation.")
        return

    apply_style()

    nz, nx = Vs_realization.shape

    if bedrock_mask is not None:
        if bedrock_mask.shape != (nz, nx):
            raise ValueError(
                f"bedrock_mask shape {bedrock_mask.shape} != Vs_realization shape {(nz, nx)}"
            )
        if bedrock_mask.dtype != bool:
            bedrock_mask = bedrock_mask.astype(bool)

    Vs_unique = np.unique(Vs_1D_profile)
    _, Vs2 = Vs_unique[0], Vs_unique[1]

    if bedrock_mask is not None:
        soil_mask = ~bedrock_mask
        soil_vs_values = Vs_realization[soil_mask]
    else:
        soil_vs_values = Vs_realization[Vs_realization < Vs2]

    vmin = soil_vs_values.min()
    vmax = soil_vs_values.max()

    cmap = get_crameri_cmap("batlow", reverse=True)
    cmap = cmap.copy()
    cmap.set_over("gray")

    fig, ax = plt.subplots(figsize=(12, 5))

    computed_Lx = nx * dx
    computed_Lz = nz * dz
    assert np.isclose(computed_Lx, Lx) and np.isclose(computed_Lz, Lz), (
        f"Computed Lx/Lz do not match provided Lx/Lz: {computed_Lx} m != {Lx} m, {computed_Lz} m != {Lz} m"
    )
    extent = (0, computed_Lx, computed_Lz, 0)

    im = ax.imshow(
        Vs_realization,
        extent=extent,
        aspect="auto",
        cmap=cmap,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    cbar = fig.colorbar(im, ax=ax, extend="max")
    cbar.set_label(r"Soil $V_s$ (m/s)")
    cbar.ax.text(
        0.5,
        1.15,
        f"$V_{{s2}}$ = {Vs2:.0f} m/s",
        transform=cbar.ax.transAxes,
        ha="center",
        va="bottom",
    )

    ax.set_xlabel(to_title_case("Distance (M)"))
    ax.set_ylabel(to_title_case("Depth (M)"))
    format_title(
        title
        if title is not None
        else r"Optimized 2D $V_s$ Realization (Soil-Focused Color Scale)",
        ax=ax,
    )
    ax.grid(False)
    add_subfigure_label(ax, 0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    plt.close()


# --- Main Execution ---
if __name__ == "__main__":
    RESULTS_DIR = Path("./results/Case_1a")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    DATA_CONFIG = {
        "My_New_Run": {
            "base": RESULTS_DIR / "soil_base.txt",
            "top": RESULTS_DIR / "soil_top.txt",
        },
        "PLAXIS": {
            "base": RESULTS_DIR / "BaseRight.txt",
            "top": RESULTS_DIR / "SurfaceRight.txt",
        },
        "OpenSeesPy_Prev": {
            "base": RESULTS_DIR / "acc_base_left_hor.out",
            "top": RESULTS_DIR / "acc_surf_left_hor.out",
        },
    }

    try:
        all_data = load_datasets(DATA_CONFIG)

        model_to_process = "OpenSeesPy_Prev"
        if model_to_process in all_data:
            for loc in ("base", "top"):
                if loc in all_data[model_to_process]:
                    time, accel = all_data[model_to_process][loc]
                    all_data[model_to_process][loc] = (time - time[0], accel)

    except FileNotFoundError as e:
        print(f"Error: Data file not found. {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    REFERENCE_MODEL = "My_New_Run"

    plot_acceleration_comparison(
        datasets=all_data,
        reference_name=REFERENCE_MODEL,
        output_path=Path("acceleration_time_histories_comparison.html"),
    )

    plot_acceleration_differences(
        datasets=all_data,
        reference_name=REFERENCE_MODEL,
        output_path=Path("acceleration_time_histories_differences.html"),
    )

    plot_transfer_functions(
        datasets=all_data,
        reference_name=REFERENCE_MODEL,
        output_path=Path("transfer_functions.html"),
    )

    plot_stacked_acceleration(
        datasets=all_data,
        data_config=DATA_CONFIG,
        vertical_spacing=5.0,
    )
