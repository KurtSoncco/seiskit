"""Output-path routing for analysis scripts.

``result_path("plots", "figure.png")`` auto-infers which topic subdirectory
the calling script belongs to and returns a path under
``<root>/results/<topic>/plots/figure.png``, creating directories as needed.
"""

from __future__ import annotations

import inspect
from pathlib import Path

TOPIC_BY_SCRIPT = {
    # diagnostics — center recorder only
    "baseline_residual_diagnostics": "diagnostics",
    "heteroscedasticity_diagnostics": "diagnostics",
    "normality_assessment": "diagnostics",
    "eda_channel50": "diagnostics",
    "quantile_eda": "diagnostics",
    # performance
    "gbm_performance": "performance",
    "model_r2_ceiling": "performance",
    "qbm_performance": "performance",
    # extrapolation
    "interp_extrap_predictions": "extrapolation",
    "physics_extrapolation": "extrapolation",
    "seed_variance_and_extrapolation": "extrapolation",
    # quantile / QBM
    "quantile_channel_model": "quantile",
    "quantile_coefficients_abs_TF": "quantile",
    "quantile_coefficients_f_ratio": "quantile",
    "quantile_seed_error": "quantile",
    # SHAP (mean importance + directionality / quantile tails)
    "quantile_shap_interactions": "shap",
    "quantile_shap_tails": "shap",
    "quantile_shape_cell": "shap",
    "shap_interactions": "shap",
    "shap_seed_suite": "shap",
    "shap_summary": "shap",
    "shap_summary_abs_TF": "shap",
    "shap_summary_f_ratio": "shap",
    # seed
    "seed_adequacy": "seed",
    "seed_independence": "seed",
}

_RESULTS_ROOT: Path | None = None


def set_results_root(path: str | Path) -> None:
    """Override the root directory where ``results/`` is created."""
    global _RESULTS_ROOT
    _RESULTS_ROOT = Path(path)


def _default_root() -> Path:
    """Walk the call stack to find the script's project root."""
    this_file = Path(__file__).resolve()
    for frame_info in inspect.stack()[1:]:
        script_file = frame_info.frame.f_globals.get("__file__", "")
        if not script_file:
            continue
        script_path = Path(str(script_file)).resolve()
        if script_path != this_file:
            return script_path.parent.parent
    return Path.cwd()


def _infer_script_stem() -> str:
    this_file = Path(__file__).resolve()
    for frame_info in inspect.stack()[1:]:
        script_file = frame_info.frame.f_globals.get("__file__", "")
        if not script_file:
            continue
        script_path = Path(str(script_file)).resolve()
        if script_path != this_file:
            return script_path.stem
    return "misc"


def result_path(kind: str, filename: str, *, topic: str | None = None) -> str:
    """Return an output path under ``results/<topic>/<kind>/<filename>``.

    Parameters
    ----------
    kind:
        ``"plots"`` or ``"data"``.
    filename:
        Output filename (e.g. ``"eda_channel50.png"``).
    topic:
        Explicit topic override.  ``None`` auto-infers from the calling
        script's filename via :data:`TOPIC_BY_SCRIPT`.
    """
    if kind not in {"plots", "data"}:
        raise ValueError("kind must be either 'plots' or 'data'")
    if topic is None:
        topic = TOPIC_BY_SCRIPT.get(_infer_script_stem(), "misc")
    root = _RESULTS_ROOT if _RESULTS_ROOT is not None else _default_root()
    out_dir = root / "results" / topic / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / filename)
