"""Nature figure: spatial ACF for each χ metric with aperture limitation cues.

Panel a — reference design cell: empirical ρ̂(h) + Exp / Gauss / CosWM fits.
Panel b — all 243 cells: thin gray ACFs + median / IQR.

One figure is produced per metric in ``spatial_acf.METRICS``.

Reads existing CSVs from ``figure_dir("chi_spatial", "spatial_acf")`` (no refit).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import (  # noqa: E402
    ANNOTATION_FONTSIZE,
    DATA_LINEWIDTH,
    LABEL_FONTSIZE,
    TOL_BRIGHT,
    TOL_MUTED,
    add_panel_label,
    apply_full_paper_style,
    figsize,
    figure_dir,
    metric_color,
    metric_label,
    save_figure,
)

# Import model ACF evaluators from the calculation module (same folder).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from spatial_acf import (  # noqa: E402
    H_FIT_MAX_M,
    H_MAX_M,
    METRICS,
    rho_coswm,
    rho_exp,
    rho_gauss,
)

apply_full_paper_style(auto_format=True, frame="open", grid=False)

REF = {"Height": 50.0, "Vs1": 230.0, "CoV": 0.2, "rH": 30.0, "aHV": 10.0}

COLOR_EMP = "0.15"
COLOR_EXP = TOL_BRIGHT["blue"]
COLOR_GAUSS = TOL_BRIGHT["red"]
COLOR_COSWM = TOL_MUTED["teal"]
COLOR_CELL = "0.75"
COLOR_BAND = TOL_MUTED["teal"]


YLIM = (-2.0, 1.5)


def _data_dir() -> Path:
    return figure_dir("chi_spatial", "spatial_acf")


def _ref_mask(df: pd.DataFrame, metric: str) -> pd.Series:
    m = pd.Series(True, index=df.index)
    for k, v in REF.items():
        m &= np.isclose(df[k].astype(float), float(v))
    return m & (df["metric"] == metric)


def _shade_aperture(ax: plt.Axes) -> None:
    """Mark half-aperture fit window and beyond-fit region."""
    ax.axvspan(H_FIT_MAX_M, H_MAX_M, color="0.85", alpha=0.55, zorder=0, lw=0)
    ax.axvline(H_FIT_MAX_M, color="0.35", ls="--", lw=0.7, zorder=2)
    ax.set_xlim(0.0, H_MAX_M)
    ax.set_ylim(*YLIM)
    ax.axhline(0.0, color="0.6", lw=0.4, zorder=1)


def _annotate_aperture(ax: plt.Axes, *, y_text: float | None = None) -> None:
    if y_text is None:
        y_text = YLIM[1] - 0.08
    ax.text(
        H_FIT_MAX_M / 2,
        y_text,
        "Fit window\n$h\\leq 100$ m",
        ha="center",
        va="top",
        fontsize=ANNOTATION_FONTSIZE,
        color="0.25",
    )
    ax.text(
        (H_FIT_MAX_M + H_MAX_M) / 2,
        y_text,
        "Beyond half-aperture\n(not used in fit)",
        ha="center",
        va="top",
        fontsize=ANNOTATION_FONTSIZE,
        color="0.35",
    )


def _model_curves(row: pd.Series, h: np.ndarray) -> dict[str, np.ndarray]:
    """Evaluate stored Exp / Gauss / CosWM ACF params on lag grid *h*."""
    out: dict[str, np.ndarray] = {}
    if bool(row.get("fit_ok_exp", False)):
        out["exp"] = rho_exp(h, float(row["c0_exp"]), float(row["a_m_exp"]))
    if bool(row.get("fit_ok_gauss", False)):
        out["gauss"] = rho_gauss(h, float(row["c0_gauss"]), float(row["a_m_gauss"]))
    if bool(row.get("fit_ok_coswm", False)):
        out["coswm"] = rho_coswm(
            h,
            float(row["c0_coswm"]),
            float(row["nu_coswm"]),
            float(row["scale_s_m_coswm"]),
            float(row["period_b_m_coswm"]),
        )
    return out


def panel_a(ax: plt.Axes, emp: pd.DataFrame, fits: pd.DataFrame, metric: str) -> None:
    """Reference-cell empirical ACF + three model overlays."""
    e = emp.loc[_ref_mask(emp, metric)].sort_values("h_m")
    f = fits.loc[_ref_mask(fits, metric)]
    if e.empty or f.empty:
        raise RuntimeError("Reference design cell not found in ACF CSVs.")
    row = f.iloc[0]

    _shade_aperture(ax)

    h_emp = e["h_m"].to_numpy(dtype=float)
    rho_emp = e["rho"].to_numpy(dtype=float)
    ax.plot(
        h_emp,
        rho_emp,
        "o-",
        color=COLOR_EMP,
        ms=2.0,
        lw=0.7,
        alpha=0.9,
        zorder=4,
        label="Empirical",
    )

    # Smooth model curves over full aperture (dashed beyond fit window)
    h_fit = np.linspace(0.0, H_FIT_MAX_M, 200)
    h_out = np.linspace(H_FIT_MAX_M, H_MAX_M, 80)
    models = _model_curves(row, np.concatenate([h_fit, h_out]))
    styles = {
        "exp": (COLOR_EXP, "Exp", 0.9),
        "gauss": (COLOR_GAUSS, "Gaussian", 0.9),
        "coswm": (COLOR_COSWM, "CosWM", 1.15),
    }
    for key, (color, label, lw_scale) in styles.items():
        if key not in models:
            continue
        rh = models[key]
        n_fit = len(h_fit)
        ax.plot(
            h_fit,
            rh[:n_fit],
            color=color,
            lw=DATA_LINEWIDTH * lw_scale,
            zorder=5,
            label=label,
        )
        ax.plot(
            h_out,
            rh[n_fit:],
            color=color,
            lw=DATA_LINEWIDTH * lw_scale,
            ls=":",
            alpha=0.65,
            zorder=5,
        )

    _annotate_aperture(ax)
    ax.set_xlabel(r"Lag distance $h$ (m)")
    ax.set_ylabel(r"Autocorrelation $\hat{\rho}(h)$")
    ax.set_title(
        f"Reference cell: {metric_label(metric)}",
        loc="left",
        fontsize=LABEL_FONTSIZE,
    )
    ax.text(
        0.02,
        0.02,
        (
            r"$H=50$ m, $V_{s1}=230$ m/s,"
            "\n"
            r"CoV$=0.2$, $r_h=30$ m, $a_{hv}=10$"
        ),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=ANNOTATION_FONTSIZE,
        color="0.3",
    )
    # Keep clear of design-factor text (bottom left) and aperture note (bottom right)
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.78),
        fontsize=ANNOTATION_FONTSIZE,
        frameon=False,
        ncol=1,
        handlelength=1.6,
    )
    ax.text(
        0.98,
        0.02,
        "Aperture $L=200$ m\nFits are descriptive only",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=ANNOTATION_FONTSIZE,
        color="0.25",
    )


def panel_b(ax: plt.Axes, emp: pd.DataFrame, metric: str) -> None:
    """All-cell empirical ACF envelope + median / IQR."""
    m = emp[emp["metric"] == metric].copy()
    color_med = metric_color(metric)
    _shade_aperture(ax)

    # Thin individual cell curves
    for _, g in m.groupby("cell"):
        g = g.sort_values("h_m")
        ax.plot(
            g["h_m"],
            g["rho"],
            color=COLOR_CELL,
            lw=0.35,
            alpha=0.35,
            zorder=2,
        )

    # Median + IQR at each lag
    by_h = m.groupby("h_m")["rho"]
    h = by_h.median().index.to_numpy(dtype=float)
    med = by_h.median().to_numpy(dtype=float)
    q25 = by_h.quantile(0.25).to_numpy(dtype=float)
    q75 = by_h.quantile(0.75).to_numpy(dtype=float)

    ax.fill_between(h, q25, q75, color=color_med, alpha=0.35, zorder=3, lw=0)
    ax.plot(h, med, color=color_med, lw=DATA_LINEWIDTH * 1.2, zorder=4, label="Median")

    _annotate_aperture(ax)
    ax.set_xlabel(r"Lag distance $h$ (m)")
    ax.set_title(
        f"All design cells ($n=243$): {metric_label(metric)}",
        loc="left",
        fontsize=LABEL_FONTSIZE,
    )

    handles = [
        Line2D([0], [0], color=COLOR_CELL, lw=0.6, alpha=0.7, label="Each cell"),
        Patch(facecolor=COLOR_BAND, alpha=0.45, edgecolor="none", label="IQR"),
        Line2D([0], [0], color=color_med, lw=1.2, label="Median"),
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        fontsize=ANNOTATION_FONTSIZE,
        frameon=False,
        handlelength=1.6,
    )


def plot_metric(emp: pd.DataFrame, fits: pd.DataFrame, metric: str, data_dir: Path) -> Path:
    """Build and save the two-panel ACF figure for a single metric."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize(aspect=0.48),
        sharey=True,
        constrained_layout=True,
    )
    panel_a(axes[0], emp, fits, metric)
    panel_b(axes[1], emp, metric)
    axes[0].set_ylim(*YLIM)
    axes[1].tick_params(labelleft=False)
    add_panel_label(axes[0], 0)
    add_panel_label(axes[1], 1)

    stem = f"spatial_acf_{metric}"
    out = save_figure(fig, stem, out_dir=data_dir)
    plt.close(fig)
    return out


def main() -> None:
    data_dir = _data_dir()
    emp_path = data_dir / "acf_empirical.csv"
    fit_path = data_dir / "acf_fit_params.csv"
    if not emp_path.is_file() or not fit_path.is_file():
        raise FileNotFoundError(f"Missing ACF CSVs in {data_dir}. Run spatial_acf.py first.")

    emp = pd.read_csv(emp_path)
    fits = pd.read_csv(fit_path)

    for metric in METRICS:
        out = plot_metric(emp, fits, metric, data_dir)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
