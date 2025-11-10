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
from pathlib import Path
from typing import List


def merge_timing_files(results_dir: Path, output_file: Path) -> None:
    """
    Merge all timing_data_task_*.csv files into a single CSV file.

    Args:
        results_dir: Directory containing timing_data_task_*.csv files
        output_file: Path to output CSV file
    """
    # Find all task-specific timing files
    task_files = sorted(results_dir.glob("timing_data_task_*.csv"))

    if not task_files:
        print(f"Warning: No timing_data_task_*.csv files found in {results_dir}")
        return

    print(f"Found {len(task_files)} timing files to merge")
    print(f"Output file: {output_file}")

    # Collect all rows (skip headers from individual files)
    all_rows: List[List[str]] = []
    header_written = False

    for task_file in task_files:
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
                for row in reader:
                    if len(row) > 0:  # Skip empty rows
                        all_rows.append(row)

        except Exception as e:
            print(f"  Warning: Error reading {task_file}: {e}")
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
    print(f"\nMerged {data_row_count} timing records into {output_file}")

    # Delete individual timing files after successful merge
    print("\nCleaning up individual timing files...")
    deleted_count = 0
    for task_file in task_files:
        try:
            task_file.unlink()
            deleted_count += 1
            print(f"  Deleted: {task_file.name}")
        except Exception as e:
            print(f"  Warning: Failed to delete {task_file.name}: {e}")

    print(f"Deleted {deleted_count} individual timing files")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Merge individual task timing files into a single CSV"
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
    args = parser.parse_args()

    merge_timing_files(args.results_dir, args.output)


if __name__ == "__main__":
    main()
