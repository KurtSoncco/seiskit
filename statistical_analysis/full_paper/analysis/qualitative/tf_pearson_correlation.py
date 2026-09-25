"""2D vs. 1D |TF| agreement: Pearson correlation and relative L1 error.

For every (Height, Vs1, CoV, rH, aHV, seed, node) realization in the
qualitative TF dataset (see ``_common.py``), computes two complementary
metrics between the 2D (heterogeneous) transfer function magnitude and the
matching homogeneous 1D baseline, across the shared frequency axis
(0.1-10 Hz, 1000 points; the 2D and 1D grids coincide exactly, so no
interpolation is needed):

``r`` (Pearson correlation, log space)
    Correlation between :math:`\\ln|TF|_{2D}(f)` and
    :math:`\\ln|TF|_{1D}(f)`. Scale/offset-invariant — measures whether the
    *shape* (peaks/troughs vs. frequency) tracks the baseline. Log space is
    used because |TF| spans multiple decades and the whole codebase already
    treats ``log(|TF|)`` as the natural scale for these curves (see
    ``_geomean_band`` in ``_common.py``).

``rel_l1`` (relative L1 error, linear space)
    :math:`\\sum_f \\left| |TF|_{2D}(f) - |TF|_{1D}(f) \\right| /
    \\sum_f |TF|_{1D}(f)`. Computed in linear |TF| space (not log) since it
    is a magnitude-error metric — it is sensitive to offset/scale, unlike
    ``r``, so the two metrics give genuinely different perspectives on
    2D-vs-1D agreement.

Outputs (under ``figure_dir("qualitative", "tf_pearson_correlation")``)
-------------------------------------------------------------------------
``tf_metrics.csv``
    One row per realization: Height, Vs1, CoV, rH, aHV, seed, node, r,
    rel_l1. Cached — rerun with ``--force`` to recompute.
``tf_pearson_overall_box.pdf`` / ``tf_rel_l1_overall_box.pdf``
    Single box plot of the metric pooled across the whole dataset.
``tf_pearson_by_factor_box.pdf`` / ``tf_rel_l1_by_factor_box.pdf``
    1x5 row of box plots, one panel per subsurface parameter
    (Vs1, Height, CoV, rH, aHV), showing how the metric's distribution
    changes across that parameter's three levels (marginalized over
    everything else).

Usage
-----
python tf_pearson_correlation.py [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    AHV_LIST,
    COV_LIST,
    H_LIST,
    N_NODES,
    N_SEEDS,
    RH_LIST,
    VS1_LIST,
    base_1d_path,
    cell_start,
    tf_dir,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import (  # noqa: E402
    FACTORS,
    LABEL_FONTSIZE,
    TICK_LABELSIZE,
    TOL_BRIGHT,
    figsize,
    figure_dir,
    save_figure,
)

EPS = 1e-12

# Low / mid / high level styling, shared across factor panels (matches the
# convention in analysis/code/chi_variables/central_profiles.py).
_LEVEL_COLORS = [TOL_BRIGHT["blue"], TOL_BRIGHT["green"], TOL_BRIGHT["red"]]

_FACTOR_LEVELS: dict[str, list[float]] = {
    "Vs1": VS1_LIST,
    "Height": H_LIST,
    "CoV": COV_LIST,
    "rH": RH_LIST,
    "aHV": AHV_LIST,
}

_FACTOR_TITLES: dict[str, str] = {
    "Vs1": r"$V_{s1}$",
    "Height": r"$H$",
    "CoV": r"$\mathrm{CoV}$",
    "rH": r"$r_h$",
    "aHV": r"$a_{hv}$",
}


def _level_label(factor: str, val: float) -> str:
    if factor == "Height":
        return f"{val:.0f} m"
    if factor == "Vs1":
        return f"{val:.0f} m/s"
    if factor == "CoV":
        return f"{val:g}"
    if factor == "rH":
        return f"{val:.0f} m"
    return f"{val:.0f}"  # aHV


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def _pearson_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson r of each row of *x* against fixed vector *y* (shared columns).

    Vectorized: since ``y`` is centered once, the cross term reduces to a
    single matrix-vector product, avoiding an explicit centering of *x*.
    """
    n = x.shape[1]
    y_c = y - y.mean()
    denom_y = np.sqrt(np.sum(y_c * y_c))
    num = x @ y_c
    sx = x.sum(axis=1)
    sx2 = np.sum(x * x, axis=1)
    denom_x = np.sqrt(np.clip(sx2 - sx * sx / n, 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / (denom_x * denom_y)


def _rel_l1_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Relative L1 error of each row of *x* against fixed vector *y* (linear space).

    ``sum_f |x_f - y_f| / sum_f |y_f|`` — the L1-norm analogue of a relative
    error, computed per row via broadcasting (no matrix trick needed since
    this is a plain elementwise difference, not a cross term).
    """
    denom = np.sum(np.abs(y))
    num = np.sum(np.abs(x - y[None, :]), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / denom


def _cell_frame(
    h: float,
    vs1: float,
    cov: float,
    rh: float,
    ahv: float,
    tf_all: np.ndarray,
    tf_1d: np.ndarray,
    y_log: np.ndarray,
) -> pd.DataFrame:
    """Both metrics for every (seed, node) in one design cell (100 x 101 rows)."""
    i0 = cell_start(vs1, cov, rh, ahv)
    block = np.asarray(tf_all[i0 : i0 + N_SEEDS, :, :], dtype=np.float64)
    n_freq = block.shape[-1]
    x_lin = block.reshape(-1, n_freq)
    x_log = np.log(np.clip(x_lin, EPS, None))
    r = _pearson_rows(x_log, y_log)
    rel_l1 = _rel_l1_rows(x_lin, tf_1d)
    return pd.DataFrame(
        {
            "Height": h,
            "Vs1": vs1,
            "CoV": cov,
            "rH": rh,
            "aHV": ahv,
            "seed": np.repeat(np.arange(N_SEEDS), N_NODES),
            "node": np.tile(np.arange(N_NODES), N_SEEDS),
            "r": r,
            "rel_l1": rel_l1,
        }
    )


def compute_all(cache_path: Path, *, force: bool = False) -> pd.DataFrame:
    """r and rel_l1 for every realization in the qualitative TF dataset (cached)."""
    if cache_path.exists() and not force:
        print(f"Loading cached metrics from {cache_path}")
        return pd.read_csv(cache_path)

    frames: list[pd.DataFrame] = []
    for h in tqdm(H_LIST, desc="Height"):
        tf_all = np.load(tf_dir(h) / "tf_per_sample.npy", mmap_mode="r")
        for vs1 in VS1_LIST:
            base = np.load(base_1d_path(h, vs1))
            tf_1d = np.asarray(base["tf_magnitude"], dtype=np.float64)
            y_log = np.log(np.clip(tf_1d, EPS, None))
            for cov in COV_LIST:
                for rh in RH_LIST:
                    for ahv in AHV_LIST:
                        frames.append(_cell_frame(h, vs1, cov, rh, ahv, tf_all, tf_1d, y_log))

    df = pd.concat(frames, ignore_index=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"Wrote {cache_path} ({len(df):,} rows)")
    return df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
_BOX_STYLE = dict(patch_artist=True, showfliers=False, widths=0.6)
_MEDIAN_PROPS = dict(color="black", lw=1.2)
_WHISKER_PROPS = dict(color="0.3", lw=0.8)

_METRIC_META: dict[str, dict[str, str]] = {
    "r": dict(
        ylabel=r"Pearson $r\left(\ln|TF|_{2D},\ \ln|TF|_{1D}\right)$",
        stem="tf_pearson",
        color=TOL_BRIGHT["blue"],
    ),
    "rel_l1": dict(
        ylabel=(
            r"Relative $L_1$ error "
            r"$\dfrac{\sum_f\left||TF|_{2D}-|TF|_{1D}\right|}{\sum_f|TF|_{1D}}$"
        ),
        stem="tf_rel_l1",
        color=TOL_BRIGHT["purple"],
    ),
}


def plot_overall_box(df: pd.DataFrame, out_dir: Path, metric: str) -> Path:
    """Single box plot of *metric* pooled across the whole dataset."""
    meta = _METRIC_META[metric]
    values = df[metric].to_numpy()

    fig, ax = plt.subplots(figsize=figsize(aspect=0.42))
    bp = ax.boxplot(
        [values],
        positions=[1],
        medianprops=_MEDIAN_PROPS,
        whiskerprops=_WHISKER_PROPS,
        capprops=_WHISKER_PROPS,
        **_BOX_STYLE,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(meta["color"])
        patch.set_alpha(0.55)
        patch.set_edgecolor(meta["color"])

    ax.set_xticks([1])
    ax.set_xticklabels(["All realizations"])
    ax.set_ylabel(meta["ylabel"], fontsize=LABEL_FONTSIZE)
    ax.grid(True, axis="y", alpha=0.18, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=TICK_LABELSIZE)

    median = np.median(values)
    q1, q3 = np.percentile(values, [25, 75])
    ax.text(
        0.98,
        0.03,
        rf"$n={len(df):,}$" + "\n" + rf"median $= {median:.3f}$" + "\n" + rf"IQR $= [{q1:.3f}, {q3:.3f}]$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=TICK_LABELSIZE,
        linespacing=1.3,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.0},
    )

    paths = save_figure(fig, f"{meta['stem']}_overall_box", out_dir=out_dir)
    plt.close(fig)
    return paths[0]


def plot_by_factor_box(df: pd.DataFrame, out_dir: Path, metric: str) -> Path:
    """1x5 row: box plots of *metric* across the three levels of each factor."""
    meta = _METRIC_META[metric]
    width, height = figsize(aspect=0.30)
    fig, axes = plt.subplots(
        1,
        len(FACTORS),
        figsize=(width, height),
        sharey=True,
        gridspec_kw=dict(wspace=0.08, left=0.07, right=0.995, bottom=0.16, top=0.88),
    )

    for ax, factor in zip(axes, FACTORS):
        levels = _FACTOR_LEVELS[factor]
        data = [df.loc[df[factor] == lv, metric].to_numpy() for lv in levels]
        positions = np.arange(1, len(levels) + 1)
        bp = ax.boxplot(
            data,
            positions=positions,
            medianprops=_MEDIAN_PROPS,
            whiskerprops=_WHISKER_PROPS,
            capprops=_WHISKER_PROPS,
            **_BOX_STYLE,
        )
        for patch, color in zip(bp["boxes"], _LEVEL_COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
            patch.set_edgecolor(color)

        ax.set_xticks(positions)
        ax.set_xticklabels([_level_label(factor, lv) for lv in levels], fontsize=TICK_LABELSIZE)
        # ax.text (not ax.set_title) to bypass the auto-format label patch, which
        # would re-substitute "CoV" inside the already-LaTeX title and break it.
        ax.text(
            0.5,
            1.02,
            _FACTOR_TITLES[factor],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=LABEL_FONTSIZE,
            fontweight="bold",
        )
        ax.grid(True, axis="y", alpha=0.18, lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=TICK_LABELSIZE)

    axes[0].set_ylabel(meta["ylabel"], fontsize=LABEL_FONTSIZE)

    paths = save_figure(fig, f"{meta['stem']}_by_factor_box", out_dir=out_dir)
    plt.close(fig)
    return paths[0]


def main() -> None:
    p = argparse.ArgumentParser(description="Pearson r and relative L1 error between 2D and 1D |TF| curves.")
    p.add_argument("--force", action="store_true", help="Recompute even if the cache CSV exists.")
    args = p.parse_args()

    out_dir = figure_dir("qualitative", "tf_pearson_correlation")
    cache_path = out_dir / "tf_metrics.csv"

    df = compute_all(cache_path, force=args.force)
    for metric in ("r", "rel_l1"):
        print(f"{metric}: n={len(df):,}  median={df[metric].median():.4f}  mean={df[metric].mean():.4f}")

    written: list[Path] = []
    for metric in ("r", "rel_l1"):
        written.append(plot_overall_box(df, out_dir, metric))
        written.append(plot_by_factor_box(df, out_dir, metric))

    print(f"Done: {', '.join(p.name for p in written)} -> {out_dir}")


if __name__ == "__main__":
    main()
