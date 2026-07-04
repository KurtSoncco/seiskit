"""Merge individual task timing files into a single CSV file.

This script combines all timing_data_task_*.csv files from the results directory
into a single timing_data.csv file. This avoids race conditions when multiple
tasks write timing data simultaneously.

After successfully merging, the individual timing files are deleted to keep
the results directory clean.

Usage:
    python merge_timing_data.py [--results-dir RESULTS_DIR] [--output OUTPUT_CSV]
"""

import argparse
import csv
import re
from pathlib import Path
from typing import List, Optional, Set


def parse_timing_filename(filename: str) -> Optional[int]:
    """
    Parse timing filename to extract index.

    Expected format: timing_data_task_{index}.csv
    Example: timing_data_task_0.csv -> 0

    Returns:
        Index or None if parsing fails
    """
    pattern = r"timing_data_task_(\d+)\.csv"
    match = re.match(pattern, filename)
    if match:
        return int(match.group(1))
    return None


def merge_timing_files(
    results_dir: Path,
    output_file: Path,
    strict: bool = False,
    keep_on_missing: bool = False,
    expected_total: int = 30,
) -> None:
    """
    Merge all timing_data_task_*.csv files into a single CSV file.

    Args:
        results_dir: Directory containing timing_data_task_*.csv files
        output_file: Path to output CSV file
        strict: If True, fail if any expected files are missing
        keep_on_missing: If True, don't delete individual files if any are missing
        expected_total: Expected total number of tasks (default: 30)
    """
    # Find all task-specific timing files
    task_files = sorted(results_dir.glob("timing_data_task_*.csv"))

    if not task_files:
        print(f"Warning: No timing_data_task_*.csv files found in {results_dir}")
        return

    print(f"Found {len(task_files)} timing files to merge")
    print(f"Output file: {output_file}")

    # Parse filenames to identify indices
    found_indices: Set[int] = set()
    file_map: dict[int, Path] = {}

    for task_file in task_files:
        index = parse_timing_filename(task_file.name)
        if index is None:
            print(f"  Warning: Could not parse filename: {task_file.name}")
            continue

        found_indices.add(index)
        file_map[index] = task_file

    # Report found files
    indices_list = sorted(found_indices)
    print(f"\nFound {len(indices_list)} files (indices: {min(indices_list)}-{max(indices_list)})")

    # Check for missing files
    expected_indices = set(range(expected_total))
    missing_indices = sorted(expected_indices - found_indices)

    if missing_indices:
        print("\n⚠️  WARNING: Missing timing files detected:")
        print(
            f"  Missing {len(missing_indices)} files (indices: {missing_indices[:10]}{'...' if len(missing_indices) > 10 else ''})"
        )
        print("  This indicates some array tasks may have failed or not completed yet.")

        if strict:
            print("\nERROR: Strict mode enabled. Exiting without merging.")
            return

        if keep_on_missing:
            print("\n  Keeping individual files (--keep-on-missing mode)")
    else:
        print("\n✓ All expected timing files are present")

    # Collect all rows (skip headers from individual files)
    all_rows: List[List[str]] = []
    header_written = False
    successful_reads = 0
    failed_reads = 0

    # Process files in sorted order (by index)
    for index in sorted(file_map.keys()):
        task_file = file_map[index]
        print(f"  Reading: {task_file.name}")
        try:
            with open(task_file, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)  # Read header

                if not header_written and header is not None:
                    # Use header from first file (only if header is present)
                    all_rows.append(header)
                    header_written = True

                # Read data rows
                rows_in_file = 0
                for row in reader:
                    if len(row) > 0:  # Skip empty rows
                        all_rows.append(row)
                        rows_in_file += 1

                if rows_in_file == 0:
                    print(f"    Warning: No data rows found in {task_file.name}")
                else:
                    successful_reads += 1

        except Exception as e:
            print(f"    ERROR: Failed to read {task_file.name}: {e}")
            failed_reads += 1
            continue

    if not header_written:
        print("Error: No valid headers found in timing files")
        return

    # Write merged data to output file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(all_rows)

    # Count data rows (excluding header)
    data_row_count = len(all_rows) - 1
    print("\nMerge summary:")
    print(f"  Successfully read: {successful_reads} files")
    if failed_reads > 0:
        print(f"  Failed to read: {failed_reads} files")
    if missing_indices:
        print(f"  Expected total: {expected_total} files")
        print(f"  Missing: {len(missing_indices)} files")
    print(f"  Data records: {data_row_count}")
    print(f"  Output file: {output_file}")

    # Delete individual timing files after successful merge
    # Skip deletion if files are missing and keep_on_missing is True
    should_delete = not (missing_indices and keep_on_missing)

    if should_delete:
        print("\nCleaning up individual timing files...")
        deleted_count = 0
        for task_file in task_files:
            try:
                task_file.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"  Warning: Failed to delete {task_file.name}: {e}")

        print(f"Deleted {deleted_count} individual timing files")
    else:
        print("\nSkipping cleanup (individual files kept due to missing files)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Merge individual task timing files into a single CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic merge (default behavior)
  python merge_timing_data.py

  # Merge with strict validation (fails if any files are missing)
  python merge_timing_data.py --strict

  # Merge but keep individual files if any are missing
  python merge_timing_data.py --keep-on-missing

  # Custom paths
  python merge_timing_data.py --results-dir ./my_results --output ./merged_timing.csv

  # Custom expected total
  python merge_timing_data.py --expected-total 30
        """,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing timing_data_task_*.csv files (default: results)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/timing_data.csv"),
        help="Output CSV file path (default: results/timing_data.csv)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any expected timing files are missing (default: False)",
    )
    parser.add_argument(
        "--keep-on-missing",
        action="store_true",
        help="Keep individual files if any are missing (default: False, deletes after merge)",
    )
    parser.add_argument(
        "--expected-total",
        type=int,
        default=30,
        help="Expected total number of tasks (default: 30)",
    )
    args = parser.parse_args()

    merge_timing_files(
        args.results_dir,
        args.output,
        args.strict,
        args.keep_on_missing,
        args.expected_total,
    )


if __name__ == "__main__":
    main()
