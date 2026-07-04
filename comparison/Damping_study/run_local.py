"""Local runner for damping study experiment.

This script allows running the damping study experiment locally (without SLURM).
It can run a single case, a range of cases, or all cases sequentially or in parallel.

Usage:
    # Run a single case
    python run_local.py --index 0

    # Run a range of cases
    python run_local.py --start 0 --end 9

    # Run all 40 cases sequentially
    python run_local.py --all

    # Run all cases in parallel (4 workers)
    python run_local.py --all --parallel --workers 4

    # Run specific indices
    python run_local.py --indices 0 5 10 15
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

# Import the run function from run_experiment
from run_experiment import _fmt_hms, run_damping_case


def run_single_case(index: int, verbose: bool = True) -> tuple[int, str, float]:
    """
    Run a single case and return results.

    Args:
        index: Parameter combination index
        verbose: Whether to print progress

    Returns:
        Tuple of (index, status, execution_time)
    """
    start_time = time.time()
    try:
        if verbose:
            print(f"[{index:3d}] Starting...")
        status = run_damping_case(index)
        execution_time = time.time() - start_time
        if verbose:
            print(f"[{index:3d}] Completed: {status} ({_fmt_hms(execution_time)})")
        return (index, status, execution_time)
    except Exception as e:
        execution_time = time.time() - start_time
        error_msg = f"Error: {str(e)}"
        if verbose:
            print(f"[{index:3d}] Failed: {error_msg} ({_fmt_hms(execution_time)})")
        return (index, error_msg, execution_time)


def run_sequential(
    indices: List[int], verbose: bool = True, show_progress: bool = True
) -> List[tuple[int, str, float]]:
    """
    Run cases sequentially.

    Args:
        indices: List of indices to run
        verbose: Whether to print detailed progress
        show_progress: Whether to show overall progress

    Returns:
        List of (index, status, execution_time) tuples
    """
    results = []
    total = len(indices)
    start_time = time.time()

    for i, index in enumerate(indices, 1):
        if show_progress:
            print(f"\n[{i}/{total}] Running index {index}...")
        result = run_single_case(index, verbose=verbose)
        results.append(result)
        if show_progress:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = avg_time * (total - i)
            print(
                f"Progress: {i}/{total} completed | "
                f"Elapsed: {_fmt_hms(elapsed)} | "
                f"Estimated remaining: {_fmt_hms(remaining)}"
            )

    return results


def run_parallel(
    indices: List[int],
    max_workers: int = 4,
    verbose: bool = False,
    show_progress: bool = True,
) -> List[tuple[int, str, float]]:
    """
    Run cases in parallel using ProcessPoolExecutor.

    Args:
        indices: List of indices to run
        max_workers: Maximum number of parallel workers
        verbose: Whether to print detailed progress from each worker
        show_progress: Whether to show overall progress

    Returns:
        List of (index, status, execution_time) tuples
    """
    results = []
    total = len(indices)
    completed = 0
    start_time = time.time()

    if show_progress:
        print(f"Running {total} cases in parallel with {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(run_single_case, idx, verbose=False): idx for idx in indices
        }

        # Process completed tasks
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1

                if show_progress:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed
                    remaining = avg_time * (total - completed)
                    status_str = (
                        "✓"
                        if "success" in result[1].lower() or "done" in result[1].lower()
                        else "✗"
                    )
                    print(
                        f"[{completed:3d}/{total}] {status_str} Index {index:3d}: {result[1]:30s} | "
                        f"Elapsed: {_fmt_hms(elapsed)} | "
                        f"Remaining: {_fmt_hms(remaining)}"
                    )
            except Exception as e:
                results.append((index, f"Exception: {str(e)}", 0.0))
                completed += 1
                if show_progress:
                    print(f"[{completed:3d}/{total}] ✗ Index {index:3d}: Exception: {str(e)}")

    # Sort results by index to maintain order
    results.sort(key=lambda x: x[0])
    return results


def print_summary(results: List[tuple[int, str, float]]) -> None:
    """Print summary of results."""
    total = len(results)
    successful = sum(
        1 for _, status, _ in results if "success" in status.lower() or "done" in status.lower()
    )
    failed = total - successful
    total_time = sum(time for _, _, time in results)
    avg_time = total_time / total if total > 0 else 0

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total cases: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {_fmt_hms(total_time)}")
    print(f"Average time per case: {_fmt_hms(avg_time)}")

    if failed > 0:
        print("\nFailed cases:")
        for index, status, exec_time in results:
            if "success" not in status.lower() and "done" not in status.lower():
                print(f"  Index {index:3d}: {status} ({_fmt_hms(exec_time)})")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run damping study experiment locally",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a single case
  python run_local.py --index 0

  # Run a range of cases
  python run_local.py --start 0 --end 9

  # Run all 40 cases sequentially
  python run_local.py --all

  # Run all cases in parallel (4 workers)
  python run_local.py --all --parallel --workers 4

  # Run specific indices
  python run_local.py --indices 0 5 10 15

  # Run with custom number of workers
  python run_local.py --all --parallel --workers 8
        """,
    )

    # Case selection options (mutually exclusive)
    case_group = parser.add_mutually_exclusive_group(required=True)
    case_group.add_argument(
        "--index",
        type=int,
        help="Run a single case with the specified index (0-39)",
    )
    case_group.add_argument(
        "--start",
        type=int,
        help="Start index for range (requires --end)",
    )
    case_group.add_argument(
        "--all",
        action="store_true",
        help="Run all 40 cases",
    )
    case_group.add_argument(
        "--indices",
        type=int,
        nargs="+",
        help="Run specific indices (e.g., --indices 0 5 10 15)",
    )

    parser.add_argument(
        "--end",
        type=int,
        help="End index for range (inclusive, requires --start)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run cases in parallel (default: sequential)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4, only used with --parallel)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=40,
        help="Total number of cases (default: 40)",
    )

    args = parser.parse_args()

    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Determine indices to run
    if args.index is not None:
        if args.index < 0 or args.index >= args.total:
            print(f"Error: Index must be between 0 and {args.total - 1}", file=sys.stderr)
            sys.exit(1)
        indices = [args.index]
    elif args.start is not None:
        if args.end is None:
            print("Error: --end is required when using --start", file=sys.stderr)
            sys.exit(1)
        if args.start < 0 or args.end >= args.total or args.start > args.end:
            print(
                f"Error: Invalid range. Start and end must be between 0 and {args.total - 1}, and start <= end",
                file=sys.stderr,
            )
            sys.exit(1)
        indices = list(range(args.start, args.end + 1))
    elif args.all:
        indices = list(range(args.total))
    elif args.indices is not None:
        # Validate indices
        for idx in args.indices:
            if idx < 0 or idx >= args.total:
                print(
                    f"Error: Index {idx} is out of range (0-{args.total - 1})",
                    file=sys.stderr,
                )
                sys.exit(1)
        indices = args.indices
    else:
        print(
            "Error: Must specify --index, --start/--end, --all, or --indices",
            file=sys.stderr,
        )
        sys.exit(1)

    # Set verbosity
    verbose = not args.quiet
    show_progress = True

    # Run cases
    print(f"Running {len(indices)} case(s)...")
    start_time = time.time()

    if args.parallel and len(indices) > 1:
        results = run_parallel(indices, args.workers, verbose, show_progress)
    else:
        if args.parallel and len(indices) == 1:
            print("Note: Only one case specified, running sequentially")
        results = run_sequential(indices, verbose, show_progress)

    total_time = time.time() - start_time

    # Print summary
    print_summary(results)
    print(f"\nTotal execution time: {_fmt_hms(total_time)}")

    # Exit with error code if any cases failed
    failed_count = sum(
        1
        for _, status, _ in results
        if "success" not in status.lower() and "done" not in status.lower()
    )
    sys.exit(1 if failed_count > 0 else 0)


if __name__ == "__main__":
    main()
