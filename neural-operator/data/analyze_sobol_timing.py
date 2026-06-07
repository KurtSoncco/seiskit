#!/usr/bin/env python3
"""Summarize sobol_timing.db and compare against H5 outputs."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from sobol import load_manifest_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing-db", type=Path, required=True)
    parser.add_argument("--h5-dir", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=THIS_DIR / "sobol_manifest.csv")
    parser.add_argument("--total-tasks", type=int, default=7680)
    args = parser.parse_args()

    conn = sqlite3.connect(args.timing_db)
    rows = conn.execute(
        """
        SELECT idx, total_time_sec, field_sec, model_sec, analysis_sec, status,
               H_discretized, Lz_discretized, duration
        FROM sobol_timing_per_index
        ORDER BY idx
        """
    ).fetchall()
    conn.close()

    finished = [r for r in rows if (r[5] or "").startswith("Finished")]
    failed = [r for r in rows if (r[5] or "").startswith("Failed")]
    finished_idx = {r[0] for r in finished}

    print(f"timing rows: {len(rows)}  finished: {len(finished)}  failed: {len(failed)}")
    if finished:
        totals = np.array([r[1] for r in finished])
        H = np.array([r[6] for r in finished])
        print(
            f"total_time_sec: min={totals.min():.0f} p50={np.median(totals):.0f} "
            f"p90={np.percentile(totals, 90):.0f} max={totals.max():.0f} mean={totals.mean():.0f}"
        )
        print(f"corr(total, H)={np.corrcoef(totals, H)[0, 1]:.3f}")
        print("\nMedian hours by H bin:")
        for lo, hi in [(15, 30), (30, 50), (50, 70), (70, 90), (90, 101)]:
            mask = (H >= lo) & (H < hi)
            if mask.any():
                print(
                    f"  H=[{lo:3d},{hi:3d}): n={mask.sum():4d} "
                    f"median={np.median(totals[mask]) / 3600:.2f}h "
                    f"max={totals[mask].max() / 3600:.2f}h"
                )

    missing = sorted(set(range(args.total_tasks)) - finished_idx)
    print(f"\nmissing vs {args.total_tasks} finished indices: {len(missing)}")
    if missing:
        print(f"  first: {missing[0]}  last: {missing[-1]}  max finished: {max(finished_idx)}")

    if args.h5_dir and args.h5_dir.is_dir():
        h5_idx = {
            int(p.stem.split("_")[1])
            for p in args.h5_dir.glob("run_*.h5")
            if p.stem.split("_")[1].isdigit()
        }
        print(f"\nH5 files: {len(h5_idx)} in {args.h5_dir}")
        print(f"  finished but no H5: {len(finished_idx - h5_idx)}")
        print(f"  H5 but not finished in timing: {len(h5_idx - finished_idx)}")
        pending = sorted(set(range(args.total_tasks)) - h5_idx)
        print(f"  pending (no H5): {len(pending)}")

    if args.manifest_path.exists() and missing:
        manifest = {e.index: e for e in load_manifest_csv(args.manifest_path)}
        Hs = np.array([manifest[i].H_discretized for i in missing if i in manifest])
        print("\nRemaining tasks by H (from manifest):")
        for lo, hi in [(15, 30), (30, 50), (50, 70), (70, 90), (90, 101)]:
            mask = (Hs >= lo) & (Hs < hi)
            print(f"  H=[{lo:3d},{hi:3d}): {mask.sum()}")

        for cores in (384, 768):
            est = 0.0
            med_by_bin: dict[tuple[int, int], float] = {}
            if finished:
                for lo, hi in [(15, 30), (30, 50), (50, 70), (70, 90), (90, 101)]:
                    mask = (H >= lo) & (H < hi)
                    if mask.any():
                        med_by_bin[(lo, hi)] = float(np.median(totals[mask]))
            for idx in missing:
                if idx not in manifest:
                    continue
                h = manifest[idx].H_discretized
                for lo, hi in med_by_bin:
                    if lo <= h < hi:
                        est += med_by_bin[(lo, hi)]
                        break
            print(f"\nEst. wall @ {cores} cores (bin medians): {est / cores / 3600:.1f} h")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
