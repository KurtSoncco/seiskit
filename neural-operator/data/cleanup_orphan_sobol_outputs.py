#!/usr/bin/env python3
"""Remove orphan Sobol outputs for indices that have no run_{idx}.h5.

Jobs killed at walltime often finish archiving raw runs but die before or after
H5 write, leaving archives (or partial raw dirs) that block reruns when
idempotent skip also checked those paths. Completion is defined by H5 only.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-base",
        type=Path,
        required=True,
        help="RUN_BASE (contains h5/, raw_runs/)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete orphans (default is dry-run listing only)",
    )
    args = parser.parse_args()

    run_base = args.run_base.resolve()
    h5_dir = run_base / "h5"
    raw_base = run_base / "raw_runs"
    archives = raw_base / "archives"

    if not h5_dir.is_dir():
        raise SystemExit(f"H5 dir not found: {h5_dir}")

    complete = {
        int(p.stem.split("_")[1])
        for p in h5_dir.glob("run_*.h5")
        if p.stem.split("_")[1].isdigit()
    }

    to_remove: list[Path] = []

    if archives.is_dir():
        for arc in list(archives.glob("run_*.tar.zst")) + list(archives.glob("run_*.tgz")):
            name = arc.name
            if name.endswith(".tar.zst"):
                idx_s = name.removeprefix("run_").removesuffix(".tar.zst")
            elif name.endswith(".tgz"):
                idx_s = name.removeprefix("run_").removesuffix(".tgz")
            else:
                continue
            if not idx_s.isdigit():
                continue
            idx = int(idx_s)
            if idx not in complete:
                to_remove.append(arc)

    if raw_base.is_dir():
        for run_dir in raw_base.glob("run_*"):
            if not run_dir.is_dir():
                continue
            idx_s = run_dir.name[len("run_") :]
            if not idx_s.isdigit():
                continue
            idx = int(idx_s)
            if idx not in complete:
                to_remove.append(run_dir)

    print(f"complete H5 indices: {len(complete)}")
    print(f"orphan paths to remove: {len(to_remove)}")
    for path in sorted(to_remove)[:30]:
        print(f"  {path}")
    if len(to_remove) > 30:
        print(f"  ... +{len(to_remove) - 30} more")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to delete.")
        return 0

    for path in to_remove:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    print(f"\nDeleted {len(to_remove)} orphan path(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
