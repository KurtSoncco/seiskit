"""
GIFNO-FDO-XT surrogate inference for grf_2d (Response_Variability).

Publication checkpoint: ``checkpoints/xt_lat128_d128``
  (LATENT_CHANNELS=128, DEEPONET_LATENT_DIM=128).

Training/inference tensor shape is (4, NZ_MAX, NX) with NX=500 — the variability
strip only. Full OpenSees H5 files store 1500 m (BC + var + BC); the data loader
slices columns [BC_WIDTH, BC_WIDTH+LX_VAR) before the model (see
neural-operator/data and GIFNO data_loader.py).

Environment:
  RV_USE_SURROGATE_2D=1       Enable surrogate for grf_2d (default on).
  GIFNO_MODEL_DIR             Checkpoint dir with best_model.pt
  GIFNO_LATENT_CHANNELS=128
  GIFNO_DEEPONET_LATENT_DIM=128
  GIFNO_NUM_FNO_LAYERS=5
  GIFNO_SURROGATE_ROOT        Path to GIFNO-FDO-XT experiment (optional)
"""

from __future__ import annotations

import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

_DEFAULT_SURROGATE_ROOT = Path.home() / "surrogate-seismic-waves" / "experiments" / "GIFNO-FDO-XT"
_DEFAULT_MODEL_DIR = Path.home() / "surrogate-seismic-waves" / "checkpoints" / "xt_lat128_d128"


def use_surrogate_2d() -> bool:
    return os.getenv("RV_USE_SURROGATE_2D", "1") == "1"


def _surrogate_experiment_dir() -> Path:
    raw = os.getenv("GIFNO_SURROGATE_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    return _DEFAULT_SURROGATE_ROOT.resolve()


def _ensure_surrogate_imports() -> Any:
    exp_dir = _surrogate_experiment_dir()
    if not exp_dir.is_dir():
        raise FileNotFoundError(
            f"Surrogate experiment not found: {exp_dir}. "
            "Set GIFNO_SURROGATE_ROOT to experiments/GIFNO-FDO-XT."
        )

    # xt_lat128_d128 is the 4-channel single-path baseline. Never inherit a
    # seed-conditional recipe from the shell (6-ch / dual-path would mismatch).
    os.environ.pop("GIFNO_SEED_CONDITIONAL_RECIPE", None)
    os.environ.setdefault("GIFNO_MODEL_DIR", str(_DEFAULT_MODEL_DIR))
    os.environ.setdefault("GIFNO_LATENT_CHANNELS", "128")
    os.environ.setdefault("GIFNO_DEEPONET_LATENT_DIM", "128")
    os.environ.setdefault("GIFNO_NUM_FNO_LAYERS", "5")
    os.environ["GIFNO_SCALE_SPLIT_VS"] = "false"
    os.environ["GIFNO_DUAL_PATH_ENCODER"] = "false"
    os.environ["GIFNO_BRANCH_MODE"] = "surface"
    os.environ["GIFNO_OUTPUT_ACTIVATION"] = "none"
    os.environ["GIFNO_IN_CHANNELS"] = "4"

    # Drop stale config/model modules so env overrides take effect.
    for name in ("config", "model", "gifno_xt_config"):
        sys.modules.pop(name, None)

    cfg_path = exp_dir / "config.py"
    spec = importlib.util.spec_from_file_location("config", cfg_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load surrogate config from {cfg_path}")
    surrogate_config = importlib.util.module_from_spec(spec)
    sys.modules["config"] = surrogate_config
    spec.loader.exec_module(surrogate_config)
    surrogate_config.setup_import_paths()

    if int(surrogate_config.IN_CHANNELS) != 4:
        raise RuntimeError(
            f"Expected IN_CHANNELS=4 for xt_lat128_d128, got {surrogate_config.IN_CHANNELS}. "
            "Unset GIFNO_SEED_CONDITIONAL_RECIPE / SCALE_SPLIT_VS."
        )
    if bool(getattr(surrogate_config, "DUAL_PATH_ENCODER", False)):
        raise RuntimeError("DUAL_PATH_ENCODER must be false for xt_lat128_d128")
    return surrogate_config


def _extract_variability_strip(
    field: np.ndarray,
    *,
    dx: float,
    bc_width: float,
    lx_var: float,
) -> np.ndarray:
    """Return nz×NX variability strip (500 cols @ 1 m) from full or strip-native field."""
    cfg = _ensure_surrogate_imports()
    nx = field.shape[1]
    n_var = int(round(lx_var / dx))
    bc_cols = int(round(bc_width / dx))

    if nx == cfg.NX:
        return field.astype(np.float32)
    if nx == cfg.NX_FULL:
        i0 = cfg.X_SLICE_START
        i1 = cfg.X_SLICE_END
        return field[:, i0:i1].astype(np.float32)
    if nx == n_var:
        return field.astype(np.float32)
    if bc_cols + n_var <= nx:
        return field[:, bc_cols : bc_cols + n_var].astype(np.float32)
    raise ValueError(f"Cannot extract {cfg.NX}-column variability strip from field width nx={nx}")


def _resample_variability_strip(
    var_strip: np.ndarray,
    *,
    dx: float,
    dz: float,
    target_dx: float,
    target_dz: float,
    target_nx: int,
) -> np.ndarray:
    """Resample a variability strip to (nz_sur, target_nx) at target spacing."""
    from scipy.interpolate import RegularGridInterpolator

    nz_rv, nx_rv = var_strip.shape
    nz_sur = max(1, int(round((nz_rv * dz) / target_dz)))
    x_src = np.arange(nx_rv, dtype=np.float64) * dx
    z_src = np.arange(nz_rv, dtype=np.float64) * dz
    x_dst = np.arange(target_nx, dtype=np.float64) * target_dx
    z_dst = np.arange(nz_sur, dtype=np.float64) * target_dz

    interp = RegularGridInterpolator(
        (z_src, x_src),
        var_strip.astype(np.float64),
        method="linear",
        bounds_error=False,
        fill_value=None,
    )
    zz, xx = np.meshgrid(z_dst, x_dst, indexing="ij")
    pts = np.stack([zz.ravel(), xx.ravel()], axis=-1)
    return interp(pts).reshape(nz_sur, target_nx).astype(np.float32)


def prepare_variability_strip(
    vs_field: np.ndarray,
    zeta_field: np.ndarray,
    *,
    dx: float,
    dz: float,
    bc_width: float,
    lx_var: float,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """
    Vs/zeta -> (nz, 500) variability strips for surrogate input.

    Matches GIFNO ``data_loader._load_sample``: slice from 1500 m H5 or use
    strip if already 500 m wide; resample legacy RV grids when needed.
    """
    cfg = _ensure_surrogate_imports()
    vs_strip = _extract_variability_strip(vs_field, dx=dx, bc_width=bc_width, lx_var=lx_var)
    zeta_strip = _extract_variability_strip(zeta_field, dx=dx, bc_width=bc_width, lx_var=lx_var)

    if vs_strip.shape[1] != cfg.NX or abs(dx - cfg.DX) > 1e-6 or abs(dz - cfg.DZ) > 1e-6:
        vs_strip = _resample_variability_strip(
            vs_strip, dx=dx, dz=dz, target_dx=cfg.DX, target_dz=cfg.DZ, target_nx=cfg.NX
        )
        zeta_strip = _resample_variability_strip(
            zeta_strip, dx=dx, dz=dz, target_dx=cfg.DX, target_dz=cfg.DZ, target_nx=cfg.NX
        )

    nz = vs_strip.shape[0]
    lz = nz * cfg.DZ
    return vs_strip, zeta_strip, lz, nz


def resample_to_surrogate_grid(
    vs_field: np.ndarray,
    zeta_field: np.ndarray,
    *,
    dx: float,
    dz: float,
    bc_width: float,
    lx_var: float,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Deprecated alias: returns variability strips (nz×500), not full 1500 m domain."""
    return prepare_variability_strip(
        vs_field, zeta_field, dx=dx, dz=dz, bc_width=bc_width, lx_var=lx_var
    )


def build_model_input(
    vs_strip: np.ndarray,
    zeta_strip: np.ndarray,
    *,
    lz: float,
    nz: int,
) -> np.ndarray:
    """(4, NZ_MAX, NX) float32 tensor — variability strip only (NX=500)."""
    from data_loader import (  # noqa: WPS433
        _build_sample_coord_grids,
        _normalize_vs_by_surface,
        _normalize_zeta_by_max,
        _pad_depth,
    )

    cfg = _ensure_surrogate_imports()
    if vs_strip.shape[1] != cfg.NX:
        raise ValueError(f"Expected variability strip width NX={cfg.NX}, got {vs_strip.shape[1]}")

    vs = _pad_depth(vs_strip, cfg.NZ_MAX)
    zeta = _pad_depth(zeta_strip, cfg.NZ_MAX)
    x_coord, z_coord = _build_sample_coord_grids(
        nz,
        cfg.NX,
        cfg.NZ_MAX,
        lz,
        float(cfg.LX_VARIABILITY),
        dz=cfg.DZ,
        dx=cfg.DX,
    )
    if vs[0].max() <= 0:
        raise ValueError("Surface row has zero Vs on variability strip")
    vs = _normalize_vs_by_surface(vs, cfg.VS_NORM_EPS)
    zeta = _normalize_zeta_by_max(zeta, nz, cfg.ZETA_NORM_EPS)
    return np.stack([vs, zeta, x_coord, z_coord], axis=0).astype(np.float32)


def surrogate_freq() -> np.ndarray:
    cfg = _ensure_surrogate_imports()
    freq_path = cfg.TF_FREQ_PATH
    if freq_path.is_file():
        return np.load(freq_path).astype(np.float32)
    gifno_cfg_path = _surrogate_experiment_dir().parent / "GIFNO" / "config.py"
    spec = importlib.util.spec_from_file_location("gifno_base_config", gifno_cfg_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load GIFNO config from {gifno_cfg_path}")
    gifno_base = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gifno_base)
    return np.logspace(
        np.log10(gifno_base.FREQ_START_HZ),
        np.log10(gifno_base.FREQ_END_HZ),
        cfg.N_FREQ,
    ).astype(np.float32)


@lru_cache(maxsize=1)
def _load_model_cached() -> tuple[Any, Any]:
    import torch
    from model import create_model  # noqa: WPS433

    cfg = _ensure_surrogate_imports()
    model_dir = Path(os.environ.get("GIFNO_MODEL_DIR", str(_DEFAULT_MODEL_DIR)))
    checkpoint = model_dir / "best_model.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Surrogate checkpoint not found: {checkpoint}")
    # Guard against accidentally pointing at xt_seed_conditional / other recipes.
    if "xt_seed_conditional" in str(model_dir):
        raise RuntimeError(
            f"Refusing seed-conditional checkpoint for RV study: {model_dir}. "
            f"Use {_DEFAULT_MODEL_DIR}"
        )
    device = torch.device(os.getenv("GIFNO_DEVICE", str(cfg.DEVICE)))
    model = create_model(in_channels=4, dual_path=False, output_activation="none").to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint mismatch for {checkpoint}: "
            f"missing={len(missing)} unexpected={len(unexpected)}. "
            "Confirm GIFNO_MODEL_DIR=.../xt_lat128_d128 and LATENT=128."
        )
    model.eval()
    print(f"[surrogate_2d] loaded {checkpoint}  (IN_CHANNELS=4, surface, single-path)")
    return model, device


def predict_transfer_functions(
    vs_field: np.ndarray,
    zeta_field: np.ndarray,
    *,
    dx: float,
    dz: float,
    bc_width: float,
    lx_var: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """
    Run surrogate on RV fields.

    Returns
    -------
    freq : (n_freq,)
    af_spatial : (n_lateral, n_freq)
    af_spatial_stats : median, p16, p84, sigma_ln along recorder axis
      central AF is af_spatial[central_idx].
    """
    import torch

    from seiskit.intensity_measures import sigma_ln

    cfg = _ensure_surrogate_imports()
    vs_strip, zeta_strip, lz, nz = prepare_variability_strip(
        vs_field,
        zeta_field,
        dx=dx,
        dz=dz,
        bc_width=bc_width,
        lx_var=lx_var,
    )
    x = build_model_input(vs_strip, zeta_strip, lz=lz, nz=nz)
    model, device = _load_model_cached()

    with torch.no_grad():
        grid = model(torch.from_numpy(x[None]).to(device)).cpu().numpy()[0]
    rec = cfg.recorder_x_indices()
    af_spatial = grid[rec, :].astype(np.float32)
    freq = surrogate_freq()
    if af_spatial.shape[1] != len(freq):
        raise RuntimeError(f"Surrogate n_freq={af_spatial.shape[1]} != freq axis {len(freq)}")

    stats = {
        "median": np.median(af_spatial, axis=0).astype(np.float32),
        "p16": np.percentile(af_spatial, 16, axis=0).astype(np.float32),
        "p84": np.percentile(af_spatial, 84, axis=0).astype(np.float32),
        "sigma_ln": np.array(
            [sigma_ln(af_spatial[:, j]) for j in range(af_spatial.shape[1])],
            dtype=np.float32,
        ),
    }
    return freq, af_spatial, stats


def central_af_index() -> int:
    cfg = _ensure_surrogate_imports()
    rec = cfg.recorder_x_indices()
    return int(np.argmin(np.abs(rec - cfg.NX // 2)))
