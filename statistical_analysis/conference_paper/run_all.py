"""Run center-recorder conference-paper analyses in dependency order.

Phases
------
1. Diagnostics (no trained models required)
2. Model training (``quantile_channel_model.py``) — writes ``models/``
3. Performance / R² ceiling (consumes models)
4. Quantile coefficients, CV, seed-error (mostly independent of GBM pkls)
5. SHAP consumers (require mean / quantile models)
6. Seed adequacy / independence
7. Extrapolation studies

Usage (from the seiskit workspace root)::

    uv run python statistical_analysis/conference_paper/run_all.py
    uv run python statistical_analysis/conference_paper/run_all.py --parallel 4
    uv run python statistical_analysis/conference_paper/run_all.py --skip-train
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent  # seiskit/

# Explicit center-recorder script order. Training precedes model consumers.
PHASES: list[tuple[str, list[str]]] = [
    (
        "diagnostics",
        [
            "diagnostics/eda_channel50.py",
            "diagnostics/quantile_eda.py",
            "diagnostics/normality_assessment.py",
            "diagnostics/heteroscedasticity_diagnostics.py",
            "diagnostics/baseline_residual_diagnostics.py",
        ],
    ),
    (
        "train",
        [
            "quantile/quantile_channel_model.py",
        ],
    ),
    (
        "performance",
        [
            "performance/gbm_performance.py",
            "performance/qbm_performance.py",
            "performance/model_r2_ceiling.py",
        ],
    ),
    (
        "quantile",
        [
            "quantile/quantile_coefficients_abs_TF.py",
            "quantile/quantile_coefficients_f_ratio.py",
            "quantile/quantile_seed_error.py",
        ],
    ),
    (
        "shap",
        [
            "shap/shap_seed_suite.py",
        ],
    ),
    (
        "seed",
        [
            "seed/seed_independence.py",
            "seed/seed_adequacy.py",
        ],
    ),
    (
        "extrapolation",
        [
            "extrapolation/physics_extrapolation.py",
            "extrapolation/interp_extrap_predictions.py",
            "extrapolation/seed_variance_and_extrapolation.py",
        ],
    ),
]

ENV = {
    **os.environ,
    "PYTHONPATH": str(ROOT) + os.pathsep + str(WORKSPACE),
    "PYTHONUNBUFFERED": "1",
}


def run_script(script: Path) -> tuple[Path, int, str, float]:
    """Run a single script; return (path, exit_code, last_line, elapsed)."""
    t0 = time.monotonic()
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(WORKSPACE),
        env=ENV,
        text=True,
        capture_output=True,
    )
    elapsed = time.monotonic() - t0
    last = ""
    if proc.returncode != 0:
        lines = (proc.stderr or proc.stdout or "").strip().splitlines()
        last = lines[-1] if lines else ""
    elif proc.stdout.strip():
        last = proc.stdout.strip().splitlines()[-1]
    return script, proc.returncode, last, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parallel",
        "-j",
        type=int,
        default=1,
        help="Max parallel workers within each phase (default: sequential)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip quantile_channel_model training (use existing models/)",
    )
    parser.add_argument(
        "--phase",
        action="append",
        choices=[p for p, _ in PHASES],
        help="Run only the named phase(s); may be repeated",
    )
    args = parser.parse_args()

    phases = PHASES
    if args.phase:
        wanted = set(args.phase)
        phases = [(name, scripts) for name, scripts in PHASES if name in wanted]
    if args.skip_train:
        phases = [(name, scripts) for name, scripts in phases if name != "train"]

    scripts: list[Path] = []
    for _, rels in phases:
        for rel in rels:
            scripts.append(ROOT / rel)

    missing = [s for s in scripts if not s.exists()]
    if missing:
        print("Missing scripts:")
        for s in missing:
            print(f"  - {s.relative_to(ROOT)}")
        return 1

    print(f"Found {len(scripts)} center-recorder scripts in {len(phases)} phases")
    print(f"Workers per phase: {args.parallel}")
    print()

    failures: list[Path] = []
    t_total = time.monotonic()

    for phase_name, rels in phases:
        print(f"=== phase: {phase_name} ===")
        phase_scripts = [ROOT / rel for rel in rels]
        # Training must stay sequential; other phases may parallelize.
        parallel = 1 if phase_name == "train" else args.parallel

        if parallel <= 1:
            for script in phase_scripts:
                rel = script.relative_to(ROOT)
                print(f">>> {rel} ...", end="", flush=True)
                _, rc, msg, elapsed = run_script(script)
                if rc != 0:
                    failures.append(script)
                    print(f"  FAILED ({elapsed:.0f}s) {msg}")
                else:
                    print(f"  OK ({elapsed:.0f}s) {msg}")
        else:
            with ProcessPoolExecutor(max_workers=parallel) as pool:
                futures = {pool.submit(run_script, s): s for s in phase_scripts}
                for fut in as_completed(futures):
                    script, rc, msg, elapsed = fut.result()
                    rel = script.relative_to(ROOT)
                    status = "OK" if rc == 0 else "FAILED"
                    print(f"  {status} ({elapsed:.0f}s) {rel}  {msg}", flush=True)
                    if rc != 0:
                        failures.append(script)
        print()

    total_elapsed = time.monotonic() - t_total
    print(f"{'=' * 60}")
    print(f"Total time: {total_elapsed:.0f}s")
    print(f"Succeeded: {len(scripts) - len(failures)}")
    print(f"Failed:    {len(failures)}")
    for f in failures:
        print(f"  - {f.relative_to(ROOT)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
