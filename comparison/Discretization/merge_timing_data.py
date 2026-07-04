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
from typing import Dict, List, Optional, Set, Tuple


def parse_timing_filename(filename: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Parse timing filename to extract case type and index.

    Expected format: timing_data_task_{case_type}_{index}.csv
    Example: timing_data_task_2x2_4node_0.csv -> ('2x2_4node', 0)

    Returns:
        Tuple of (case_type, index) or (None, None) if parsing fails
    """
    pattern = r"timing_data_task_([^_]+(?:_[^_]+)*)_(\d+)\.csv"
    match = re.match(pattern, filename)
    if match:
        case_type = match.group(1)
        index = int(match.group(2))
        return (case_type, index)
    return (None, None)


def find_missing_indices(
    found_files: Dict[str, Set[int]], expected_total: int = 30
) -> Dict[str, List[int]]:
    """
    Find missing indices for each case type.

    Args:
        found_files: Dictionary mapping case_type to set of found indices
        expected_total: Expected total number of tasks per case type (default: 30)

    Returns:
        Dictionary mapping case_type to list of missing indices
    """
    missing = {}
    for case_type, indices in found_files.items():
        expected = set(range(expected_total))
        missing_indices = sorted(expected - indices)
        if missing_indices:
            missing[case_type] = missing_indices
    return missing


def merge_timing_files(
    results_dir: Path,
    output_file: Path,
    strict: bool = False,
    keep_on_missing: bool = False,
) -> None:
    """
    Merge all timing_data_task_*.csv files into a single CSV file.

    Args:
        results_dir: Directory containing timing_data_task_*.csv files
        output_file: Path to output CSV file
        strict: If True, fail if any expected files are missing
        keep_on_missing: If True, don't delete individual files if any are missing
    """
    # Find all task-specific timing files
    task_files = sorted(results_dir.glob("timing_data_task_*.csv"))

    if not task_files:
        print(f"Warning: No timing_data_task_*.csv files found in {results_dir}")
        return

    print(f"Found {len(task_files)} timing files to merge")
    print(f"Output file: {output_file}")

    # Parse filenames to identify case types and indices
    found_files: Dict[str, Set[int]] = {}
    file_map: Dict[Tuple[str, int], Path] = {}

    for task_file in task_files:
        case_type, index = parse_timing_filename(task_file.name)
        if case_type is None or index is None:
            print(f"  Warning: Could not parse filename: {task_file.name}")
            continue

        if case_type not in found_files:
            found_files[case_type] = set()
        found_files[case_type].add(index)
        file_map[(case_type, index)] = task_file

    # Report found files by case type
    print("\nFiles found by case type:")
    for case_type in sorted(found_files.keys()):
        indices = sorted(found_files[case_type])
        print(f"  {case_type}: {len(indices)} files (indices: {min(indices)}-{max(indices)})")

    # Check for missing files
    missing = find_missing_indices(found_files, expected_total=30)
    if missing:
        print("\n⚠️  WARNING: Missing timing files detected:")
        for case_type, missing_indices in missing.items():
            print(
                f"  {case_type}: Missing {len(missing_indices)} files (indices: {missing_indices})"
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

    # Process files in sorted order (by case type, then index)
    for (case_type, index), task_file in sorted(file_map.items()):
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
    if missing:
        total_expected = sum(
            len(indices) + len(missing.get(ct, [])) for ct, indices in found_files.items()
        )
        print(f"  Expected total: {total_expected} files (across all case types)")
        print(f"  Missing: {sum(len(m) for m in missing.values())} files")
    print(f"  Data records: {data_row_count}")
    print(f"  Output file: {output_file}")

    # Delete individual timing files after successful merge
    # Skip deletion if files are missing and keep_on_missing is True
    should_delete = not (missing and keep_on_missing)

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
    args = parser.parse_args()

    merge_timing_files(args.results_dir, args.output, args.strict, args.keep_on_missing)


if __name__ == "__main__":
    main()
