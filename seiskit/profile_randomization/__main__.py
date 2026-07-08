"""Run individual profile-randomization components for quick checks.

Examples (from repo root):
  python -m seiskit.profile_randomization.common
  python -m seiskit.profile_randomization.nhpp
  python -m seiskit.profile_randomization.toro
  python -m seiskit.profile_randomization.passeri

You can also run a file directly:
  python seiskit/profile_randomization/common.py
"""

from __future__ import annotations

import sys

MODULES = ("common", "nhpp", "toro", "passeri")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(__doc__)
        print("Available modules:", ", ".join(MODULES))
        return

    name = sys.argv[1].removesuffix(".py")
    if name not in MODULES:
        raise SystemExit(f"Unknown module '{name}'. Choose from: {', '.join(MODULES)}")

    import runpy

    runpy.run_module(f"seiskit.profile_randomization.{name}", run_name="__main__")


if __name__ == "__main__":
    main()
