"""Center-node |TF| geomean ±1σ bands vs CoV on an rH × aHV grid.

Fixed case: Vs1 = 230 m/s, H = 50 m, channel 50 (center recorder).
3×3 panels (rows = rH, cols = aHV). Within each panel, CoV ∈ {0.1, 0.2, 0.3}
is shown as geomean ± 1σ on log|TF| (color + linestyle), plus the homogeneous
1D baseline. No individual seed curves.

Usage
-----
    python diagnostics/tf_cov_grid.py
"""

from __future__ import annotations

import string
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image

from seiskit.plot_config import apply_style, panel_letter, result_path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import FIG_WIDTH  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAVE_DPI = 600
TIGHT_PAD_IN = 0.02

# Upstream TF arrays live under the Statistical Analysis root (not /complete).
BOX = Path("/mnt/box/GIG Lab - UC Berkeley/Projects/Statistical Analysis")

H = 50.0
VS1 = 230.0
CENTER_CH = 50
N_SEEDS = 100

VS1_LIST = [100.0, 230.0, 360.0]
COV_LIST = [0.1, 0.2, 0.3]
RH_LIST = [10.0, 30.0, 50.0]
AHV_LIST = [1.0, 10.0, 50.0]

FREQ_LIM = (0.4, 1e1)
TF_LIM = (0.5, 5e2)

# CoV: color + linestyle (Paul Tol–adjacent greens / distinct hues)
COV_STYLE: dict[float, dict] = {
    0.1: {"color": "#D82424", "ls": "-", "label": "CoV = 0.1"},
    0.2: {"color": "#39EE39", "ls": "--", "label": "CoV = 0.2"},
    0.3: {"color": "#1A1AE5", "ls": "-.", "label": "CoV = 0.3"},
}
COLOR_1D = "black"
LW_1D = 1.5
FS = 10


# ---------------------------------------------------------------------------
# Data helpers (self-contained; mirrors full_paper qualitative indexing)
# ---------------------------------------------------------------------------


def tf_dir(h: float) -> Path:
    return BOX / f"h={h:.0f}" / "transfer_function_results"


def base_1d_path(h: float, vs1: float) -> Path:
    return BOX / f"h={h:.0f}" / "base_cases" / f"base_case_tf_Vs1{vs1:.0f}.npz"


def cell_start(vs1: float, cov: float, rh: float, ahv: float) -> int:
    """Index of the first seed for a factorial cell (Vs1→CoV→rH→aHV→seed)."""
    i = VS1_LIST.index(vs1)
    j = COV_LIST.index(cov)
    k = RH_LIST.index(rh)
    m = AHV_LIST.index(ahv)
    return (((i * len(COV_LIST) + j) * len(RH_LIST) + k) * len(AHV_LIST) + m) * N_SEEDS


def geomean_band(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Geomean and ±1σ envelope on log|TF| across curves (axis 0)."""
    log_tf = np.log(np.clip(stack, 1e-12, None))
    geo = np.exp(np.mean(log_tf, axis=0))
    sigma_ln = np.std(log_tf, axis=0, ddof=1)
    lo = geo * np.exp(-sigma_ln)
    hi = geo * np.exp(sigma_ln)
    return geo, lo, hi


def load_freq(h: float) -> np.ndarray:
    d = tf_dir(h)
    freq_npz = d / "freq.npz"
    if freq_npz.exists():
        return np.asarray(np.load(freq_npz)["freq"], dtype=np.float64)
    geo_npz = d / "tf_geomean.npz"
    if geo_npz.exists():
        return np.asarray(np.load(geo_npz)["freq"], dtype=np.float64)
    raise FileNotFoundError(f"No frequency axis found under {d}")


def load_data(h: float, vs1: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load freq, memmapped 2D TF, and 1D baseline (freq, |TF|)."""
    d = tf_dir(h)
    freq = load_freq(h)
    tf_path = d / "tf_per_sample.npy"
    if not tf_path.exists():
        raise FileNotFoundError(f"Missing TF array: {tf_path}")
    tf_all = np.load(tf_path, mmap_mode="r")
    base_path = base_1d_path(h, vs1)
    if not base_path.exists():
        raise FileNotFoundError(f"Missing 1D baseline: {base_path}")
    base = np.load(base_path)
    freq_1d = np.asarray(base["freq"], dtype=np.float64)
    tf_1d = np.asarray(base["tf_magnitude"], dtype=np.float64)
    return freq, tf_all, freq_1d, tf_1d


def center_stack(tf_all: np.ndarray, i0: int) -> np.ndarray:
    """``(N_SEEDS, n_freq)`` stack at the center node for one cell."""
    return np.asarray(tf_all[i0 : i0 + N_SEEDS, CENTER_CH, :], dtype=np.float64)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def save_tight_exact_width(fig: plt.Figure, out: str, *, fig_width: float = FIG_WIDTH) -> None:
    """Save tight-cropped PNG at exactly fig_width inches and 600 dpi."""
    target_px = int(round(fig_width * SAVE_DPI))
    fig.savefig(out, dpi=SAVE_DPI, bbox_inches="tight", pad_inches=TIGHT_PAD_IN)
    img = Image.open(out)
    if img.width != target_px:
        target_h = max(1, int(round(img.height * (target_px / img.width))))
        img = img.resize((target_px, target_h), Image.Resampling.LANCZOS)
    img.save(out, dpi=(SAVE_DPI, SAVE_DPI))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def plot_panel(
    ax: plt.Axes,
    freq: np.ndarray,
    tf_all: np.ndarray,
    freq_1d: np.ndarray,
    tf_1d: np.ndarray,
    *,
    rh: float,
    ahv: float,
) -> None:
    """Draw CoV geomean±1σ bands + 1D baseline (no sample curves)."""
    for cov in COV_LIST:
        style = COV_STYLE[cov]
        i0 = cell_start(VS1, cov, rh, ahv)
        stack = center_stack(tf_all, i0)
        geo, lo, hi = geomean_band(stack)
        ax.fill_between(
            freq,
            lo,
            hi,
            facecolor=style["color"],
            alpha=0.22,
            edgecolor="none",
            zorder=3,
        )
        ax.plot(
            freq,
            geo,
            color=style["color"],
            ls=style["ls"],
            lw=1.4,
            zorder=4,
        )

    ax.plot(
        freq_1d,
        tf_1d,
        color=COLOR_1D,
        ls="-",
        lw=LW_1D,
        zorder=1,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*FREQ_LIM)
    ax.set_ylim(*TF_LIM)
    ax.grid(True, which="major", alpha=0.18, lw=0.6)
    ax.grid(True, which="minor", alpha=0.08, lw=0.4)


def legend_handles() -> list:
    """Shared figure legend: CoV styles + 1D baseline."""
    handles: list = [
        Line2D(
            [0],
            [0],
            color=COV_STYLE[c]["color"],
            ls=COV_STYLE[c]["ls"],
            lw=1.4,
            label=COV_STYLE[c]["label"],
        )
        for c in COV_LIST
    ]
    handles.append(Line2D([0], [0], color=COLOR_1D, ls="-", lw=LW_1D, label="1D baseline"))
    return handles


def make_figure(
    freq: np.ndarray,
    tf_all: np.ndarray,
    freq_1d: np.ndarray,
    tf_1d: np.ndarray,
) -> plt.Figure:
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(FIG_WIDTH, FIG_WIDTH * 0.90),
        sharex=True,
        sharey=True,
    )
    for row, rh in enumerate(RH_LIST):
        for col, ahv in enumerate(AHV_LIST):
            ax = axes[row, col]
            plot_panel(ax, freq, tf_all, freq_1d, tf_1d, rh=rh, ahv=ahv)
            panel_letter(ax, string.ascii_lowercase[row * 3 + col], fontsize=FS)
            # Case labels in upper-left (panel letter stays upper-right).
            ax.text(
                0.03,
                0.97,
                rf"$r_h={rh:.0f}$ m" + "\n" + rf"$a_{{hv}}={ahv:.0f}$",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=FS,
                linespacing=1.15,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.00, "pad": 1.2},
                zorder=10,
            )
            if row == len(RH_LIST) - 1:
                ax.set_xlabel(r"$f$ (Hz)", fontsize=FS, labelpad=1)

            if col == 0:
                ax.set_ylabel(r"$\left| TF \right|$", fontsize=FS, labelpad=1)
            ax.tick_params(axis="both", labelsize=FS)

    handles = legend_handles()
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        fontsize=FS,
        frameon=False,
        columnspacing=1.0,
        handlelength=2.0,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.07, top=0.90, hspace=0.15, wspace=0.15)
    return fig


def main() -> None:
    apply_style(auto_format=True, font_size=10, frame="open")

    print(f"Loading TF data for H={H:.0f}, Vs1={VS1:.0f} …")
    freq, tf_all, freq_1d, tf_1d = load_data(H, VS1)
    print(f"  tf_per_sample shape={getattr(tf_all, 'shape', '?')}, n_freq={len(freq)}")

    fig = make_figure(freq, tf_all, freq_1d, tf_1d)
    out = result_path("plots", "tf_cov_grid.png")
    save_tight_exact_width(fig, out)
    plt.close(fig)

    img = Image.open(out)
    dpi = img.info.get("dpi", (SAVE_DPI, SAVE_DPI))
    d = float(dpi[0])
    print(f"saved {out} ({img.width}×{img.height} px, {img.width / d:.4f} in wide @ {d:.0f} dpi)")


if __name__ == "__main__":
    main()
