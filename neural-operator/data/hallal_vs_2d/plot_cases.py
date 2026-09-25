"""Case-panel figures + method boxplot for Hallal / Pretell vs NO two-layer 2D.

Per selected (Sobol sample, RF seed) run: **4×3** figure — one method per row
(Toro, Passeri, Dmult, Pretell), columns:
  (1) Vs (swarm / base / strip columns) + 2D center column
  (2) |TF| geomean ± σ_ln (or single Dmult) vs 2D center
  (3) relative |error| vs frequency vs 2D center

Methods are **not** overlaid — each row is independent.

Plus a standalone Pearson-r boxplot across all 7680 runs.

Outputs → Box ``hallal_vs_2d/figures/``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import hdf5plugin  # noqa: F401  — register Blosc2/ZFP for Box H5s
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

THIS_DIR = Path(__file__).resolve().parent
NO_DATA = THIS_DIR.parent
REPO = NO_DATA.parents[1]
sys.path.insert(0, str(NO_DATA))
sys.path.insert(0, str(REPO))

from sobol import (  # noqa: E402
    DEFAULT_RF_SEEDS_PER_SAMPLE,
    DEFAULT_SOBOL_SAMPLE_COUNT,
    build_manifest,
)
from seiskit.plot_config import apply_style  # noqa: E402
from seiskit.profile_randomization import (  # noqa: E402
    get_method,
    hallal_profile_config,
)

# Pretell column indexing (same as add_pretell / RV manifest)
from add_pretell import pretell_column_indices  # noqa: E402

BOX_ROOT = Path(
    os.getenv(
        "NO_BOX_ROOT",
        "/mnt/box/GIG Lab - UC Berkeley/Projects/Neural Operator/data",
    )
)
OUT_DIR = BOX_ROOT / "hallal_vs_2d"
FIG_DIR = OUT_DIR / "figures"
H5_DIR = BOX_ROOT / "h5"

CENTER_X = 750  # domain column at strip center (Lx=1500, dx=1)
N_VS_SWARM = 40
DZ = 1.0
BEDROCK_THICKNESS = 10.0
EPS = 1e-12

# Paul Tol Bright — one color per method row
C_TORO = "#4477AA"
C_PASSERI = "#EE6677"
C_DMULT = "#228833"
C_PRETELL = "#CCBB44"
C_2D = "#000000"
C_BASE = "#888888"

CASE_INDICES = [90, 301, 600, 3842, 3902, 4773, 4974, 6422, 7008]

ROW_SPECS = (
    {
        "name": "Toro",
        "key": "toro",
        "color": C_TORO,
        "vs_mode": "swarm",
        "rng0": 10_000,
    },
    {
        "name": "Passeri",
        "key": "passeri",
        "color": C_PASSERI,
        "vs_mode": "swarm",
        "rng0": 20_000,
    },
    {
        "name": "Dmult",
        "key": "dmult",
        "color": C_DMULT,
        "vs_mode": "base",
        "rng0": 0,
    },
    {
        "name": "Pretell",
        "key": "pretell",
        "color": C_PRETELL,
        "vs_mode": "strip",
        "rng0": 0,
    },
)


def _load_ensembles():
    with h5py.File(OUT_DIR / "ensembles.h5", "r") as f:
        sample_ids = np.asarray(f["sample_id"][:], dtype=int)
        freq = np.asarray(f["freq"][:], dtype=np.float64)
        data = {
            "sample_id": sample_ids,
            "freq": freq,
            "Vs1": np.asarray(f["Vs1"][:]),
            "H": np.asarray(f["H"][:]),
            "CoV": np.asarray(f["CoV"][:]),
            "Vs2": np.asarray(f["Vs2"][:]),
            "dmult": np.asarray(f["dmult"][:]),
            "toro_geomean": np.asarray(f["toro_geomean"][:], dtype=np.float64),
            "toro_sigma_ln": np.asarray(f["toro_sigma_ln"][:], dtype=np.float64),
            "passeri_geomean": np.asarray(f["passeri_geomean"][:], dtype=np.float64),
            "passeri_sigma_ln": np.asarray(f["passeri_sigma_ln"][:], dtype=np.float64),
            "dmult_tf": np.asarray(f["dmult_tf"][:], dtype=np.float64),
        }
    sid_to_row = {int(s): i for i, s in enumerate(sample_ids)}
    return data, sid_to_row


def _load_pretell():
    path = OUT_DIR / "pretell_ensembles.h5"
    if not path.is_file():
        return None
    with h5py.File(path, "r") as f:
        out = {
            "freq": np.asarray(f["freq"][:], dtype=np.float64),
            "geomean": np.asarray(f["geomean"][:], dtype=np.float64),
            "sigma_ln": np.asarray(f["sigma_ln"][:], dtype=np.float64),
            "valid": np.asarray(f["valid"][:], dtype=bool),
            "columns": np.asarray(f["columns"][:], dtype=int)
            if "columns" in f
            else pretell_column_indices(40),
        }
        if "p16" in f and "p84" in f:
            out["p16"] = np.asarray(f["p16"][:], dtype=np.float64)
            out["p84"] = np.asarray(f["p84"][:], dtype=np.float64)
        return out


def _load_2d_center():
    with h5py.File(OUT_DIR / "tf_2d_center.h5", "r") as f:
        freq = np.asarray(f["freq"][:], dtype=np.float64)
        tf = np.asarray(f["tf_center"][:], dtype=np.float64)
    return freq, tf


def _load_2d_vs_field(index: int) -> tuple[np.ndarray, float]:
    path = H5_DIR / f"run_{index}.h5"
    with h5py.File(path, "r") as f:
        vs2d = np.asarray(f["Vs_realization_2D"][:], dtype=np.float64)
        dz = float(f["grid"].attrs.get("dz", 1.0))
    return vs2d, dz


def _swarm_vs(entry, method: str, n: int, rng0: int) -> tuple[np.ndarray, np.ndarray]:
    cfg = hallal_profile_config(
        vs1=entry.Vs1,
        H=entry.H_discretized,
        cov=entry.CoV,
        vs2=entry.Vs2,
        dz=DZ,
        bedrock_thickness=BEDROCK_THICKNESS,
    )
    meth = get_method(method)
    profiles = []
    for i in range(n):
        rng = np.random.default_rng(rng0 + entry.sample_id * 1000 + i)
        profiles.append(meth.generate_profile(cfg, rng).vs_depth)
    stack = np.column_stack(profiles)
    z = (np.arange(stack.shape[0]) + 0.5) * DZ
    return stack, z


def _strip_vs(vs_field: np.ndarray, dz: float, n: int = N_VS_SWARM) -> tuple[np.ndarray, np.ndarray]:
    cols = pretell_column_indices(n)
    stack = vs_field[:, cols]
    z = (np.arange(stack.shape[0]) + 0.5) * dz
    return stack, z


def _base_stairs(entry) -> tuple[np.ndarray, np.ndarray]:
    z = np.array(
        [
            0.0,
            entry.H_discretized,
            entry.H_discretized,
            entry.H_discretized + BEDROCK_THICKNESS,
        ]
    )
    vs = np.array([entry.Vs1, entry.Vs1, entry.Vs2, entry.Vs2])
    return vs, z


def _rel_abs_err(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.abs(pred - ref) / np.clip(np.abs(ref), EPS, None)


def _rel_l1(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.sum(np.abs(pred - ref)) / np.sum(np.abs(ref)))


def _band(geo: np.ndarray, sig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return geo * np.exp(-sig), geo * np.exp(sig)


def _method_curves(
    ens: dict,
    pretell: dict | None,
    i_ens: int,
    index: int,
    key: str,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Return (tf_curve, lo_band or None, hi_band or None).

    For Toro/Passeri the band is geomean × exp(±σ_ln). For Pretell it prefers
    empirical p16/p84 when present, else falls back to the σ_ln band.
    """
    if key == "toro":
        geo = ens["toro_geomean"][i_ens]
        sig = ens["toro_sigma_ln"][i_ens]
        lo, hi = _band(geo, sig)
        return geo, lo, hi
    if key == "passeri":
        geo = ens["passeri_geomean"][i_ens]
        sig = ens["passeri_sigma_ln"][i_ens]
        lo, hi = _band(geo, sig)
        return geo, lo, hi
    if key == "dmult":
        return ens["dmult_tf"][i_ens], None, None
    if key == "pretell":
        if pretell is None or not pretell["valid"][index]:
            return None, None, None
        geo = pretell["geomean"][index]
        if "p16" in pretell and "p84" in pretell:
            return geo, pretell["p16"][index], pretell["p84"][index]
        lo, hi = _band(geo, pretell["sigma_ln"][index])
        return geo, lo, hi
    raise KeyError(key)

def plot_case(
    entry,
    index: int,
    ens: dict,
    pretell: dict | None,
    i_ens: int,
    freq: np.ndarray,
    tf_2d: np.ndarray,
    out_path: Path,
) -> None:
    apply_style(auto_format=False, frame="open", grid=False)
    n_rows = len(ROW_SPECS)
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(11.0, 3.0 * n_rows + 0.5),
        sharex="col",
        constrained_layout=True,
    )

    try:
        vs_field, dz = _load_2d_vs_field(index)
        vs_2d = vs_field[:, CENTER_X]
        z_2d = (np.arange(len(vs_2d)) + 0.5) * dz
    except Exception as exc:  # pragma: no cover
        print(f"[warn] Vs_2D load failed for index={index}: {exc}")
        vs_field, vs_2d, z_2d, dz = None, None, None, DZ

    vs_base, z_base = _base_stairs(entry)
    z_max = float(z_base[-1])
    dmult_val = float(ens["dmult"][i_ens])

    for r, spec in enumerate(ROW_SPECS):
        color = spec["color"]
        name = spec["name"]
        key = spec["key"]
        tf_m, lo, hi = _method_curves(ens, pretell, i_ens, index, key)

        # ----- col 0: Vs -----
        ax = axes[r, 0]
        mode = spec["vs_mode"]
        if mode == "swarm":
            swarm, z_s = _swarm_vs(entry, key, N_VS_SWARM, rng0=spec["rng0"])
            ax.plot(swarm, z_s[:, None], color=color, lw=0.4, alpha=0.30, zorder=2)
            ax.plot(vs_base, z_base, color=C_BASE, lw=1.6, ls="--", zorder=3)
        elif mode == "strip" and vs_field is not None:
            swarm, z_s = _strip_vs(vs_field, dz, n=N_VS_SWARM)
            ax.plot(swarm, z_s[:, None], color=color, lw=0.4, alpha=0.30, zorder=2)
        else:
            ax.plot(vs_base, z_base, color=color, lw=2.0, zorder=3)
        if vs_2d is not None:
            ax.plot(vs_2d, z_2d, color=C_2D, lw=1.5, zorder=4)
        ax.set_ylim(z_max, 0.0)
        ax.set_ylabel(f"{name}\nDepth (m)")
        ax.grid(True, alpha=0.25)
        if r == 0:
            ax.set_title(r"$V_s$ profile")
        if r == n_rows - 1:
            ax.set_xlabel(r"$V_s$ (m/s)")
        if key == "dmult":
            ax.text(
                0.97,
                0.03,
                fr"$D_{{\mathrm{{mult}}}}$={dmult_val:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.85"),
            )

        # ----- col 1: TF -----
        ax = axes[r, 1]
        if tf_m is None:
            ax.text(0.5, 0.5, "Pretell pending", ha="center", va="center", transform=ax.transAxes)
        elif lo is not None and hi is not None:
            ax.fill_between(freq, lo, hi, color=color, alpha=0.22, linewidth=0, zorder=2)
            ax.loglog(freq, tf_m, color=color, lw=1.7, zorder=3, label="geomean")
        else:
            ax.loglog(freq, tf_m, color=color, lw=1.7, zorder=3, label="Dmult")
        ax.loglog(freq, tf_2d, color=C_2D, lw=1.7, zorder=4, label="2D center")
        ax.set_xlim(0.1, 10)
        ax.set_ylabel(r"$\left| TF \right|$")
        ax.grid(True, which="both", alpha=0.25)
        if r == 0:
            ax.set_title(r"$|TF|$ vs 2D center")
        if r == n_rows - 1:
            ax.set_xlabel("Frequency (Hz)")

        # ----- col 2: relative error -----
        ax = axes[r, 2]
        if tf_m is None:
            ax.text(0.5, 0.5, "Pretell pending", ha="center", va="center", transform=ax.transAxes)
        else:
            err = _rel_abs_err(tf_m, tf_2d)
            ax.loglog(freq, err, color=color, lw=1.5)
            ax.text(
                0.03,
                0.97,
                fr"rel $L_1$={_rel_l1(tf_m, tf_2d):.2f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor="0.85",
                    alpha=0.9,
                ),
            )
        ax.set_xlim(0.1, 10)
        ax.set_ylabel(
            r"$\left|TF_{\mathrm{m}}-TF_{2\mathrm{D}}\right|/\left|TF_{2\mathrm{D}}\right|$"
        )
        ax.grid(True, which="both", alpha=0.25)
        if r == 0:
            ax.set_title("Relative error vs frequency")
        if r == n_rows - 1:
            ax.set_xlabel("Frequency (Hz)")

    handles = [
        Line2D([0], [0], color=C_2D, lw=1.7, label="2D center"),
        Line2D([0], [0], color=C_BASE, lw=1.5, ls="--", label="base Vs"),
        Patch(facecolor="0.5", alpha=0.25, label=r"p16–p84 / $\pm 1\sigma_{\ln}$"),
        Line2D([0], [0], color="0.35", lw=1.5, label="method geomean / Dmult"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        (
            fr"sample {entry.sample_id}, seed {entry.rf_seed}  |  "
            fr"$V_{{s1}}$={entry.Vs1:.0f}, $H$={entry.H_discretized:.0f} m, "
            fr"CoV={entry.CoV:.2f}, $V_{{s2}}$={entry.Vs2:.0f}"
        ),
        fontsize=10,
        y=1.05,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_boxplot(df: pd.DataFrame, out_path: Path) -> None:
    apply_style(auto_format=False, frame="open", grid=False)
    cols = ["r_toro", "r_passeri", "r_dmult"]
    labels = ["Toro", "Passeri", "Dmult"]
    colors = [C_TORO, C_PASSERI, C_DMULT]
    if "r_pretell" in df.columns and np.isfinite(df["r_pretell"]).any():
        cols.append("r_pretell")
        labels.append("Pretell")
        colors.append(C_PRETELL)

    fig, ax = plt.subplots(figsize=(5.8, 4.0), constrained_layout=True)
    data = [df[c].to_numpy() for c in cols]
    data = [a[np.isfinite(a)] for a in data]
    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        showfliers=False,
        widths=0.55,
        medianprops=dict(color="0.15", lw=1.4),
        whiskerprops=dict(color="0.3"),
        capprops=dict(color="0.3"),
        boxprops=dict(color="0.3"),
    )
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    for i, arr in enumerate(data, start=1):
        med = float(np.nanmedian(arr))
        ax.annotate(
            fr"$p_{{50}}$={med:.2f}",
            xy=(i, med),
            xytext=(i + 0.28, med),
            fontsize=8,
            va="center",
            color="0.25",
        )
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(
        r"Pearson $r(\ln|TF|_{\mathrm{method}},\,\ln|TF|_{2\mathrm{D},\mathrm{center}})$"
    )
    ax.set_title(f"Agreement with 2D center TF  (n={len(df)})")
    ax.grid(True, axis="y", alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    ens, sid_to_row = _load_ensembles()
    pretell = _load_pretell()
    freq, tf_all = _load_2d_center()
    assert np.allclose(freq, ens["freq"])

    manifest = build_manifest(
        sample_count=DEFAULT_SOBOL_SAMPLE_COUNT,
        seeds_per_sample=DEFAULT_RF_SEEDS_PER_SAMPLE,
    )
    by_index = {e.index: e for e in manifest}
    pearson = pd.read_csv(OUT_DIR / "pearson_center.csv")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    panel_dir = FIG_DIR / "panels_4x3"
    panel_dir.mkdir(parents=True, exist_ok=True)
    for idx in CASE_INDICES:
        entry = by_index[idx]
        i_ens = sid_to_row[entry.sample_id]
        out = panel_dir / (
            f"case4x3_s{entry.sample_id:03d}_rep{entry.replicate_id:02d}"
            f"_seed{entry.rf_seed}_idx{idx}.png"
        )
        plot_case(entry, idx, ens, pretell, i_ens, freq, tf_all[idx], out)

    plot_boxplot(pearson, FIG_DIR / "pearson_boxplot_methods.png")
    print(f"done → {FIG_DIR}")


if __name__ == "__main__":
    main()
