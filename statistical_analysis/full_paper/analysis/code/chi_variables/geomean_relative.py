r"""Between- vs within-seed geomean boxplots for all χ metrics.

Three Nature-width PDFs per (Height, Vs1), never mixed, each a 3×3 factor
grid (same layout as TF qualitative / central profiles: vary \(r_h\), CoV,
\(a_{hv}\); center column shared):

- ``seed`` (between seeds): one seed, all nodes —
  \(G_{\mathrm{seed},j}/G_{\mathrm{global}}\); a cross marks the first seed
  (``first_seed``)
- ``within`` (within seeds): every node of every seed relative to its own seed
  geomean — \(\chi_{ij}/G_{\mathrm{seed},j}\); the seed-specific spatial scatter
- ``node`` (persistent node effect): one node, all seeds —
  \(G_{\mathrm{node},i}/G_{\mathrm{global}}\); a cross marks the center node
  (``CENTER_NODE``) and a grey band the \(P_{10}\)–\(P_{90}\) spread expected
  from seed-averaging noise alone (no node effect)
- Native boxplots (IQR box, median line, \(P_{10}\)–\(P_{90}\) whiskers,
  fliers beyond) drawn straight from each cloud
- One symmetric log \(y\)-range (equal ln units) shared by every panel of
  every figure, set from the widest cloud, so box heights compare directly
  across kinds and (H, Vs1)

Writes under ``figure_dir("chi_variables", "geomean_relative")``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared import CENTER_NODE, first_seed  # noqa: E402
from config import (  # noqa: E402
    BOX_ROOT,
    DATA_LINEWIDTH,
    LABEL_FONTSIZE,
    METRICS,
    TICK_LABELSIZE,
    add_panel_label,
    apply_full_paper_style,
    figsize,
    figure_dir,
    metric_color,
    metric_label,
    save_figure,
)

apply_full_paper_style(auto_format=True, frame="open", grid=False)
# Bold black hatch lines for clear contrast against the light strip fills.
plt.rcParams["hatch.linewidth"] = 0.9
plt.rcParams["hatch.color"] = "black"

DATA_PATH = BOX_ROOT / "peak_analysis" / "join_master.h5"
H_LIST = [15.0, 50.0, 100.0]
VS1_LIST = [100.0, 230.0, 360.0]

# (rH, CoV, aHV) for panels (a)–(i), row-major — same as TF qualitative
PANELS: list[tuple[float, float, float]] = [
    (10.0, 0.2, 10.0),
    (30.0, 0.2, 10.0),
    (50.0, 0.2, 10.0),
    (30.0, 0.1, 10.0),
    (30.0, 0.2, 10.0),
    (30.0, 0.3, 10.0),
    (30.0, 0.2, 1.0),
    (30.0, 0.2, 10.0),
    (30.0, 0.2, 50.0),
]

BOX_WIDTH = 0.5
METRIC_GAP = 1.0
# Native Matplotlib hatch strings, shared by legend glyphs and panel boxes.
SEED_HATCH = "..."
WITHIN_HATCH = "xxx"
NODE_HATCH = "///"
SEED_FACE_ALPHA = 0.22
WITHIN_FACE_ALPHA = 0.18
NODE_FACE_ALPHA = 0.16
# Null band: P10–P90 of G_node/G_global if nodes had no persistent effect,
# i.e. N(0, σ_W²/N_s) in ln units with σ_W the pooled within-seed sd.
Z_P90 = 1.2815515655446004
NULL_BAND_COLOR = "0.55"
NULL_BAND_ALPHA = 0.35
NULL_BAND_HALF_WIDTH = 0.36
# Head-room on the shared symmetric ln range.
Y_PAD = 1.05
HATCH_EDGE = "0.2"
# Cross marking where the reference sample (center node / first seed) sits
# inside the full geomean box distribution.
REF_MARKER = "X"
# Offset of the first-seed bar from the pooled within-seed box centre.
REF_BAR_OFFSET = BOX_WIDTH / 2 + 0.1
REF_MARKER_SIZE = 5.5
REF_MARKER_COLOR = "black"
REF_MARKER_EDGE = "white"
# Whiskers reach the 10th/90th percentile of each box's own cloud (clipped to
# the data range) instead of the default 1.5x-IQR rule, so the whisker span
# keeps the same P10-P90 "spread" definition the figure used previously while
# the box itself now also shows the IQR and true outliers beyond it.
WHIS = (10, 90)
GRID_ALPHA = 0.18
TEXT_BBOX = {"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.6}
LEGEND_FRAME = {
    "frameon": True,
    "fancybox": False,
    "framealpha": 0.75,
    "facecolor": "white",
    "edgecolor": "none",
    "borderpad": 0.25,
}

DOC_WIDTH, FIG_HEIGHT = figsize(aspect=0.88)

# One figure per kind — the clouds are never drawn together. "ref" names the
# reference sample marked by the cross: a single value for ``seed``/``node``;
# for ``within`` the first seed's own node cloud, drawn as a P10–P90 bar
# beside the pooled box with the cross at its median.
KINDS: dict[str, dict] = {
    "seed": {
        "title": r"Between seeds: one seed, all nodes",
        "hatch": SEED_HATCH,
        "face_alpha": SEED_FACE_ALPHA,
        "legend": r"$G_{\mathrm{seed}}/G_{\mathrm{global}}$",
        "ref": "first_seed",
        "ref_legend": "first seed ({ref})",
        "ylabel": r"$G_{\mathrm{seed}}/G_{\mathrm{global}}$",
        "unity": r"$G_{\mathrm{seed}}=G_{\mathrm{global}}$",
    },
    "within": {
        "title": r"Within seeds: all nodes of each seed",
        "hatch": WITHIN_HATCH,
        "face_alpha": WITHIN_FACE_ALPHA,
        "legend": r"$\chi/G_{\mathrm{seed}}$",
        "ref": "first_seed",
        "ref_legend": r"first seed ({ref}): median, $P_{{10}}$–$P_{{90}}$ over nodes",
        "ylabel": r"$\chi/G_{\mathrm{seed}}$",
        "unity": r"$\chi=G_{\mathrm{seed}}$",
    },
    "node": {
        "title": r"Persistent node effect: one node, all seeds",
        "hatch": NODE_HATCH,
        "face_alpha": NODE_FACE_ALPHA,
        "legend": r"$G_{\mathrm{node}}/G_{\mathrm{global}}$",
        "ref": "center_node",
        "ref_legend": "center node ({ref})",
        "ylabel": r"$G_{\mathrm{node}}/G_{\mathrm{global}}$",
        "unity": r"$G_{\mathrm{node}}=G_{\mathrm{global}}$",
    },
}


def _h5_dataset_values(group: h5py.Group, key: str) -> np.ndarray:
    obj = group[key]
    if not isinstance(obj, h5py.Dataset):
        raise TypeError(
            f"Expected {key!r} in {group.name!r} to be an HDF5 dataset, got {type(obj).__name__}"
        )
    return np.asarray(obj[()])


def load_ratios(path: Path = DATA_PATH) -> pd.DataFrame:
    cols = ["Vs1", "Height", "CoV", "rH", "aHV", "channel", "seed", *METRICS]
    with h5py.File(path, "r") as f:
        g = f["master"]
        df = pd.DataFrame({c: _h5_dataset_values(g, c) for c in cols})
    return df.rename(columns={"channel": "node"})


def _panel_param_text(rh: float, cov: float, ahv: float) -> str:
    return (
        rf"$r_h = {rh:.0f}$ m" + "\n"
        rf"$\mathrm{{CoV}} = {cov:g}$" + "\n"
        rf"$a_{{hv}} = {ahv:.0f}$"
    )


def cell_matrix(
    df_hv: pd.DataFrame,
    rh: float,
    cov: float,
    ahv: float,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Node × seed χ matrix for one cell, plus its node and seed labels."""
    mask = (df_hv["rH"] == rh) & (df_hv["CoV"] == cov) & (df_hv["aHV"] == ahv)
    sub = df_hv.loc[mask]
    piv = sub.pivot(index="node", columns="seed", values=metric)
    arr = piv.to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        arr = np.where(np.isfinite(arr) & (arr > 0), arr, np.nan)
    return arr, piv.index.to_numpy(), piv.columns.to_numpy()


def cell_clouds(
    arr: np.ndarray, nodes: np.ndarray, seeds: np.ndarray, ref_seed: int
) -> dict[str, tuple[np.ndarray, np.ndarray, float]]:
    r"""Per-kind (ratio cloud, reference ratios, null half-width in ln) for a cell.

    ``seed``: \(G_{\mathrm{seed},j}/G_{\mathrm{global}}\); ``within``:
    \(\chi_{ij}/G_{\mathrm{seed},j}\) pooled over nodes and seeds; ``node``:
    \(G_{\mathrm{node},i}/G_{\mathrm{global}}\). Reference ratios are
    ``[value]`` for ``seed``/``node`` and ``[P10, median, P90]`` of the first
    seed's node cloud for ``within``. The null half-width (``node`` only) is
    the P90 of \(N(0,\sigma_W^2/N_s)\).
    """
    nan = float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        ln = np.log(arr)
    if not np.isfinite(ln).any():
        empty = np.asarray([], dtype=float)
        return {k: (empty, np.full(1, nan), nan) for k in KINDS}
    with np.errstate(invalid="ignore"):
        ln_global = np.nanmean(ln)
        ln_seed = np.nanmean(ln, axis=0) - ln_global
        ln_node = np.nanmean(ln, axis=1) - ln_global
    ln_within = ln - (ln_seed + ln_global)[None, :]
    within_flat = ln_within[np.isfinite(ln_within)]
    n_seeds = np.isfinite(ln).sum(axis=1)
    n_s = float(np.mean(n_seeds[n_seeds > 0]))
    null_hw = Z_P90 * float(np.std(within_flat)) / np.sqrt(n_s)

    def _at(vals: np.ndarray, labels: np.ndarray, ref: int) -> np.ndarray:
        hit = np.flatnonzero(labels == ref)
        ok = hit.size and np.isfinite(vals[hit[0]])
        return np.array([np.exp(vals[hit[0]]) if ok else nan])

    def _seed_spread(ref: int) -> np.ndarray:
        hit = np.flatnonzero(seeds == ref)
        col = ln_within[:, hit[0]] if hit.size else np.asarray([])
        col = col[np.isfinite(col)]
        return np.exp(np.percentile(col, [10, 50, 90])) if col.size else np.full(3, nan)

    def _cloud(vals: np.ndarray) -> np.ndarray:
        return np.exp(vals[np.isfinite(vals)])

    return {
        "seed": (_cloud(ln_seed), _at(ln_seed, seeds, ref_seed), nan),
        "within": (np.exp(within_flat), _seed_spread(ref_seed), nan),
        "node": (_cloud(ln_node), _at(ln_node, nodes, CENTER_NODE), null_hw),
    }


def _face_rgba(color: str, face_alpha: float) -> tuple[float, float, float, float]:
    r, g, b = to_rgb(color)
    return (r, g, b, face_alpha)


def _draw_boxplot(
    ax: plt.Axes,
    data: list[np.ndarray],
    positions: list[float],
    colors: list[str],
    *,
    hatch: str,
    face_alpha: float,
) -> None:
    """Draw one seed/node boxplot group (all metrics in a panel) at once."""
    keep = [(d, p, c) for d, p, c in zip(data, positions, colors) if d.size > 0]
    if not keep:
        return
    kept_data, kept_pos, kept_colors = zip(*keep)
    bp = ax.boxplot(
        kept_data,
        positions=kept_pos,
        widths=BOX_WIDTH,
        whis=WHIS,
        showfliers=True,
        patch_artist=True,
        manage_ticks=False,
        boxprops={"edgecolor": HATCH_EDGE, "linewidth": 0.65},
        medianprops={"color": "0.1", "linewidth": DATA_LINEWIDTH, "solid_capstyle": "butt"},
        whiskerprops={"color": HATCH_EDGE, "linewidth": 0.65},
        capprops={"color": HATCH_EDGE, "linewidth": 0.65},
        flierprops={
            "marker": ".",
            "markersize": 2.2,
            "markerfacecolor": HATCH_EDGE,
            "markeredgecolor": "none",
            "alpha": 0.6,
        },
        zorder=3,
    )
    for box, color in zip(bp["boxes"], kept_colors):
        box.set_facecolor(_face_rgba(color, face_alpha))
        box.set_hatch(hatch)


def _legend_handles(kind: str, ref: int | None) -> list:
    spec = KINDS[kind]
    handles: list = [
        Patch(
            facecolor=_face_rgba("0.65", spec["face_alpha"]),
            edgecolor=HATCH_EDGE,
            hatch=spec["hatch"],
            linewidth=0.8,
            label=spec["legend"],
        )
    ]
    if ref is not None:
        handles.append(
            Line2D(
                [0],
                [0],
                color=REF_MARKER_COLOR,
                lw=1.1,
                ls="-" if kind == "within" else "none",
                marker=REF_MARKER,
                markersize=REF_MARKER_SIZE,
                markerfacecolor=REF_MARKER_COLOR,
                markeredgecolor=REF_MARKER_EDGE,
                markeredgewidth=0.4,
                label=spec["ref_legend"].format(ref=ref),
            )
        )
    if kind == "node":
        handles.append(
            Patch(
                facecolor=NULL_BAND_COLOR,
                alpha=NULL_BAND_ALPHA,
                edgecolor="none",
                label=r"no node effect ($P_{10}$–$P_{90}$)",
            )
        )
    handles.append(Line2D([0], [0], color="0.25", ls="--", lw=DATA_LINEWIDTH, label=spec["unity"]))
    return handles


def _make_3x3_figure(
    *, h: float, vs1: float, kind: str, ref: int | None
) -> tuple[plt.Figure, np.ndarray]:
    fig = plt.figure(figsize=(DOC_WIDTH, FIG_HEIGHT))
    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[0.06, 1.0],
        hspace=0.02,
        left=0.08,
        right=0.995,
        bottom=0.06,
        top=0.99,
    )
    header = fig.add_subplot(gs[0, 0])
    header.axis("off")
    gs_panels = gs[1, 0].subgridspec(3, 3, wspace=0.08, hspace=0.10)
    axes = np.empty((3, 3), dtype=object)
    for r in range(3):
        for c in range(3):
            sharex = axes[0, 0] if (r, c) != (0, 0) else None
            sharey = axes[0, 0] if (r, c) != (0, 0) else None
            axes[r, c] = fig.add_subplot(gs_panels[r, c], sharex=sharex, sharey=sharey)

    header.text(
        0.5,
        0.95,
        rf"{KINDS[kind]['title']} ($H = {h:.0f}$ m, $V_{{s1}} = {vs1:.0f}$ m/s)",
        transform=header.transAxes,
        ha="center",
        va="top",
        fontsize=TICK_LABELSIZE,
        bbox=TEXT_BBOX,
    )
    header.legend(
        handles=_legend_handles(kind, ref),
        loc="lower center",
        ncol=4,
        fontsize=TICK_LABELSIZE,
        handlelength=1.8,
        handleheight=1.4,
        columnspacing=1.0,
        borderaxespad=0.0,
        labelspacing=0.1,
        bbox_to_anchor=(0.5, -0.15),
        **LEGEND_FRAME,
    )
    return fig, axes


def _annotate_panel(ax: plt.Axes, i: int, rh: float, cov: float, ahv: float) -> None:
    add_panel_label(ax, i, alpha=0.75)
    ax.text(
        0.02,
        0.97,
        _panel_param_text(rh, cov, ahv),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=TICK_LABELSIZE,
        linespacing=1.15,
        zorder=6,
        bbox=TEXT_BBOX,
    )
    ax.tick_params(labelsize=TICK_LABELSIZE)
    ax.grid(True, which="major", axis="y", alpha=GRID_ALPHA, lw=0.6)
    ax.set_axisbelow(True)


PanelRows = list[list[tuple[np.ndarray, np.ndarray, float, str]]]


def gather_clouds(df: pd.DataFrame, *, h: float, vs1: float) -> dict[str, PanelRows]:
    """Per-kind, per-panel (cloud, reference, null half-width, color) rows.

    Clouds stay raw — the boxplot computes its own quartiles/whiskers/fliers.
    """
    ref_seed = first_seed(df)
    df_hv = df[(df["Height"] == h) & (df["Vs1"] == vs1)]
    out: dict[str, PanelRows] = {k: [] for k in KINDS}
    for rh, cov, ahv in PANELS:
        rows: dict[str, list] = {k: [] for k in KINDS}
        for metric in METRICS:
            arr, nodes, seeds = cell_matrix(df_hv, rh, cov, ahv, metric)
            for kind, (cloud, ref_vals, null_hw) in cell_clouds(
                arr, nodes, seeds, ref_seed
            ).items():
                rows[kind].append((cloud, ref_vals, null_hw, metric_color(metric)))
        for kind in KINDS:
            out[kind].append(rows[kind])
    return out


def max_abs_ln(clouds: dict[str, PanelRows]) -> float:
    """Largest |ln ratio| over every cloud (sets the shared symmetric y-range)."""
    m = 0.0
    for panels in clouds.values():
        for rows in panels:
            for cloud, _, _, _ in rows:
                if cloud.size:
                    m = max(m, float(np.max(np.abs(np.log(cloud)))))
    return m


def plot_geomean_relative(
    panel_data: PanelRows,
    *,
    h: float,
    vs1: float,
    kind: str,
    ln_lim: float,
    ref: int | None,
    out_dir: Path,
) -> Path:
    spec = KINDS[kind]
    fig, axes = _make_3x3_figure(h=h, vs1=vs1, kind=kind, ref=ref)

    tick_pos = [float(i) * METRIC_GAP for i in range(len(METRICS))]
    tick_lab = [metric_label(m) for m in METRICS]

    # Symmetric log range in equal ln units, shared by every figure.
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylim(np.exp(-ln_lim), np.exp(ln_lim))
    plain = FuncFormatter(lambda v, _: f"{v:g}")
    axes[0, 0].yaxis.set_major_formatter(plain)
    axes[0, 0].yaxis.set_minor_formatter(
        FuncFormatter(
            lambda v, _: f"{v:g}" if round(v / 10 ** np.floor(np.log10(v))) in (2, 5) else ""
        )
    )

    for i, ((rh, cov, ahv), rows) in enumerate(zip(PANELS, panel_data)):
        ax = axes.flat[i]
        clouds = [cloud for cloud, _, _, _ in rows]
        colors = [color for _, _, _, color in rows]
        if kind == "node":
            for x, (_, _, hw, _) in zip(tick_pos, rows):
                if np.isfinite(hw):
                    ax.fill_between(
                        [x - NULL_BAND_HALF_WIDTH, x + NULL_BAND_HALF_WIDTH],
                        np.exp(-hw),
                        np.exp(hw),
                        color=NULL_BAND_COLOR,
                        alpha=NULL_BAND_ALPHA,
                        lw=0,
                        zorder=2.5,
                    )
        _draw_boxplot(
            ax, clouds, tick_pos, colors, hatch=spec["hatch"], face_alpha=spec["face_alpha"]
        )
        if kind == "within":
            for x, (_, (p10, _, p90), _, _) in zip(tick_pos, rows):
                ax.plot(
                    [x + REF_BAR_OFFSET] * 2,
                    [p10, p90],
                    color=REF_MARKER_COLOR,
                    lw=1.1,
                    solid_capstyle="butt",
                    zorder=4,
                )
        if ref is not None:
            x_ref = REF_BAR_OFFSET if kind == "within" else 0.0
            ax.plot(
                [x + x_ref for x in tick_pos],
                [ref_vals[len(ref_vals) // 2] for _, ref_vals, _, _ in rows],
                ls="none",
                marker=REF_MARKER,
                markersize=REF_MARKER_SIZE,
                markerfacecolor=REF_MARKER_COLOR,
                markeredgecolor=REF_MARKER_EDGE,
                markeredgewidth=0.4,
                zorder=5,
            )

        ax.axhline(1.0, color="0.25", ls="--", lw=DATA_LINEWIDTH, zorder=2)
        ax.set_xlim(tick_pos[0] - 0.55, tick_pos[-1] + 0.55)
        ax.set_xticks(tick_pos)
        _annotate_panel(ax, i, rh, cov, ahv)

        row, col = divmod(i, 3)
        if row == 2:
            ax.set_xticklabels(tick_lab, fontsize=TICK_LABELSIZE - 0.5)
        else:
            ax.tick_params(labelbottom=False)
        if col == 0:
            ax.set_ylabel(spec["ylabel"], fontsize=LABEL_FONTSIZE)
            ax.tick_params(labelleft=True)
        else:
            ax.tick_params(which="both", labelleft=False)

    stem = f"geomean_relative_{kind}_h{h:.0f}_vs1_{vs1:.0f}"
    paths = save_figure(fig, stem, out_dir=out_dir)
    plt.close(fig)
    return paths[0]


def build_summary_md(written: list[Path], ln_lim: float) -> str:
    lines = [
        "# Between- vs within-seed geomean boxplots",
        "",
        "3×3 factor-grid boxplots for all χ metrics, one figure per kind (never "
        "mixed). Panel layout matches TF qualitative / central profiles "
        r"(\(r_h\), CoV, \(a_{hv}\) sweeps; center column shared).",
        "",
        r"- `seed` — between seeds: \(G_{\mathrm{seed},j}/G_{\mathrm{global}}\) "
        r"with \(G_{\mathrm{seed},j}=\exp(N_x^{-1}\sum_i\ln\chi_{ij})\) "
        f"(hatch `{SEED_HATCH}`); cross = first seed",
        r"- `within` — within seeds: \(\chi_{ij}/G_{\mathrm{seed},j}\) pooled over "
        f"all nodes and seeds (hatch `{WITHIN_HATCH}`); the seed-specific spatial scatter; "
        "black bar = first seed's own \\(P_{10}\\)–\\(P_{90}\\) over nodes, cross at its median",
        r"- `node` — persistent node effect: \(G_{\mathrm{node},i}/G_{\mathrm{global}}\) "
        r"with \(G_{\mathrm{node},i}=\exp(N_s^{-1}\sum_j\ln\chi_{ij})\) "
        f"(hatch `{NODE_HATCH}`); cross = center node ({CENTER_NODE}); grey band = "
        r"\(\exp(\pm z_{0.9}\,\sigma_W/\sqrt{N_s})\), the \(P_{10}\)–\(P_{90}\) "
        "expected from seed-averaging noise alone",
        r"- \(G_{\mathrm{global}}=\exp(\overline{\ln\chi})\) over the cell",
        r"- One symmetric log \(y\)-range shared by every panel and figure: "
        rf"\(\exp(\pm{ln_lim:.2f})\) (widest cloud × {Y_PAD:g}), so box heights "
        "compare in equal ln units",
        r"- Boxplots drawn straight from each cloud (native Matplotlib "
        r"``ax.boxplot``): box spans the IQR, whiskers reach the "
        r"\(P_{10}\)–\(P_{90}\) range (clipped to the data), dots beyond are "
        r"fliers",
        "- Horizontal tick: median; dashed: unity",
        "- Variance shares (between-seed \\(f_\\mu\\), within-seed \\(f_W\\)) vs "
        "factors: see `variability_plots.py`",
        "",
        "## Outputs",
        "",
        "| File | Content |",
        "| --- | --- |",
    ]
    for p in written:
        kind = next(k for k in KINDS if f"_{k}_" in p.name)
        lines.append(f"| `{p.name}` | 3×3 {KINDS[kind]['title'].lower()} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    out_dir = figure_dir("chi_variables", "geomean_relative")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("geomean_relative_*.pdf"):
        old.unlink()
        print(f"  removed {old.name}")

    print(f"Loading {DATA_PATH} …")
    df = load_ratios()
    print(f"  rows={len(df):,}")
    print(f"  → {out_dir}")

    # Pass 1: every cloud first, so one y-range can be shared by all figures.
    clouds: dict[tuple[float, float], dict[str, PanelRows]] = {}
    for h in H_LIST:
        for vs1 in VS1_LIST:
            print(f"  gathering H={h:.0f}, Vs1={vs1:.0f} …")
            clouds[(h, vs1)] = gather_clouds(df, h=h, vs1=vs1)
    ln_lim = Y_PAD * max(max_abs_ln(c) for c in clouds.values())
    print(f"  shared y-range: exp(±{ln_lim:.2f})")

    refs = {"first_seed": first_seed(df), "center_node": CENTER_NODE, None: None}
    written: list[Path] = []
    for (h, vs1), by_kind in clouds.items():
        for kind, panel_data in by_kind.items():
            p = plot_geomean_relative(
                panel_data,
                h=h,
                vs1=vs1,
                kind=kind,
                ln_lim=ln_lim,
                ref=refs[KINDS[kind]["ref"]],
                out_dir=out_dir,
            )
            written.append(p)
            print(f"    {p.name}")

    md_path = out_dir / "summary.md"
    md_path.write_text(build_summary_md(written, ln_lim), encoding="utf-8")
    print(f"Wrote {md_path}")
    print(f"Done → {out_dir}")


if __name__ == "__main__":
    main()
