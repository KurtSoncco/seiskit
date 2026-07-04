"""Run all conference paper analysis scripts and regenerate results.

Independent scripts are executed in parallel for faster total runtime.

Usage (from the seiskit workspace root)::

    uv run python statistical_analysis/conference_paper/run_all.py
    uv run python statistical_analysis/conference_paper/run_all.py --parallel 4
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

# Ordered by dependency: diagnostics first (no models needed), then performance,
# then quantile/shap/seed/spatial/extrapolation (which may load models).
TOPICS = [
    "diagnostics",
    "performance",
    "quantile",
    "shap",
    "seed",
    "spatial",
    "extrapolation",
]

ENV = {
    **os.environ,
    "PYTHONPATH": str(ROOT) + os.pathsep + str(WORKSPACE),
    "PYTHONUNBUFFERED": "1",
}


def run_script(script: Path) -> tuple[Path, int, str, float]:
    """Run a single script and return (path, exit_code, last_stdout_line, elapsed)."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parallel", "-j", type=int, default=1, help="Max parallel workers (default: sequential)"
    )
    args = parser.parse_args()

    scripts: list[Path] = []
    for topic in TOPICS:
        scripts.extend(sorted((ROOT / topic).glob("*.py")))

    # Skip __init__.py files
    scripts = [s for s in scripts if s.name != "__init__.py"]

    print(f"Found {len(scripts)} scripts across {len(TOPICS)} topics")
    print(f"Workers: {args.parallel}")
    print()

    failures: list[Path] = []
    t_total = time.monotonic()

    if args.parallel <= 1:
        for script in scripts:
            rel = script.relative_to(ROOT)
            print(f">>> {rel} ...", end="", flush=True)
            _, rc, msg, elapsed = run_script(script)
            if rc != 0:
                failures.append(script)
                print(f"  FAILED ({elapsed:.0f}s) {msg}")
            else:
                print(f"  OK ({elapsed:.0f}s) {msg}")
    else:
        with ProcessPoolExecutor(max_workers=args.parallel) as pool:
            futures = {pool.submit(run_script, s): s for s in scripts}
            for fut in as_completed(futures):
                script, rc, msg, elapsed = fut.result()
                rel = script.relative_to(ROOT)
                status = "OK" if rc == 0 else "FAILED"
                print(f"  {status} ({elapsed:.0f}s) {rel}  {msg}", flush=True)
                if rc != 0:
                    failures.append(script)

    total_elapsed = time.monotonic() - t_total
    print(f"\n{'=' * 60}")
    print(f"Total time: {total_elapsed:.0f}s")
    print(f"Succeeded: {len(scripts) - len(failures)}")
    print(f"Failed:    {len(failures)}")
    for f in failures:
        print(f"  - {f.relative_to(ROOT)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
