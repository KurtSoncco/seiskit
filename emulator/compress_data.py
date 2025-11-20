"""Compress entire data directory into a zip file for better data management.

This script compresses the complete data folder including:
- Materials (LF and HF)
- Material parameters
- Base motions
- HF simulation outputs
- Temp outputs (optional)

Can be run standalone or as a SLURM job after data generation completes.
"""

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional


def compress_data_folder(
    data_dir: Path,
    output_zip: Optional[Path] = None,
    remove_original: bool = False,
    include_temp: bool = False,
    exclude_patterns: Optional[list] = None,
):
    """
    Compress entire data directory into a zip file.

    Args:
        data_dir: Base data directory to compress
        output_zip: Output zip file path (default: data_dir/../data.zip)
        remove_original: If True, remove original files after compression (use with caution!)
        include_temp: If True, include temp_outputs directory
        exclude_patterns: List of patterns to exclude (e.g., ["*.tmp", "__pycache__"])
    """
    data_dir = Path(data_dir).resolve()
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return False

    if output_zip is None:
        # Default: create zip in parent directory
        output_zip = data_dir.parent / "data.zip"
    else:
        output_zip = Path(output_zip)

    if exclude_patterns is None:
        exclude_patterns = ["__pycache__", "*.pyc", "*.tmp"]

    print("=" * 70)
    print("Data Directory Compression")
    print("=" * 70)
    print(f"Data Directory: {data_dir}")
    print(f"Output ZIP: {output_zip}")
    print(f"Remove Original: {remove_original}")
    print(f"Include Temp: {include_temp}")
    print("=" * 70)

    # Collect all files to compress
    files_to_compress = []
    dirs_to_compress = []
    
    # Directories to include
    include_dirs = [
        "materials",
        "materials_hf",
        "material_params",
        "base_motion",
        "high_fidelity",
    ]
    
    if include_temp:
        include_dirs.append("temp_outputs")

    for dir_name in include_dirs:
        dir_path = data_dir / dir_name
        if dir_path.exists() and dir_path.is_dir():
            dirs_to_compress.append(dir_path)
            # Count files
            file_count = sum(1 for _ in dir_path.rglob("*") if _.is_file())
            print(f"  Found {dir_name}/: {file_count} files")

    if not dirs_to_compress:
        print("Warning: No data directories found to compress.")
        return False

    # Create zip file
    print(f"\nCreating zip archive: {output_zip}")
    start_time = time.time()

    total_files = 0
    total_size = 0

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        for dir_path in dirs_to_compress:
            print(f"\nAdding {dir_path.name}/...")
            files_added = 0
            
            # Walk through directory
            for root, dirs, files in os.walk(dir_path):
                # Filter out excluded patterns
                dirs[:] = [d for d in dirs if not any(
                    pattern in d for pattern in exclude_patterns
                )]
                
                for file in files:
                    # Skip excluded files
                    if any(file.endswith(pattern.replace("*", "")) for pattern in exclude_patterns):
                        continue
                    
                    file_path = Path(root) / file
                    # Calculate relative path from data_dir
                    arcname = file_path.relative_to(data_dir)
                    
                    try:
                        zipf.write(file_path, arcname)
                        files_added += 1
                        total_files += 1
                        total_size += file_path.stat().st_size
                        
                        if files_added <= 10 or files_added % 50 == 0:
                            print(f"  Added: {arcname}")
                    except Exception as e:
                        print(f"  Warning: Failed to add {arcname}: {e}")
            
            print(f"  Added {files_added} files from {dir_path.name}/")

    elapsed = time.time() - start_time
    zip_size = output_zip.stat().st_size / (1024**3)  # Size in GB
    original_size = total_size / (1024**3)  # Size in GB

    print("\n" + "=" * 70)
    print("✓ Compression complete!")
    print("=" * 70)
    print(f"  Archive: {output_zip}")
    print(f"  Original size: {original_size:.2f} GB")
    print(f"  Compressed size: {zip_size:.2f} GB")
    print(f"  Compression ratio: {(1 - zip_size/original_size)*100:.1f}%" if original_size > 0 else "N/A")
    print(f"  Files compressed: {total_files}")
    print(f"  Time: {elapsed:.1f} seconds")
    print("=" * 70)

    # Optionally remove original files
    if remove_original:
        print("\nRemoving original files...")
        removed_count = 0
        removed_size = 0
        
        for dir_path in dirs_to_compress:
            for root, dirs, files in os.walk(dir_path):
                for file in files:
                    file_path = Path(root) / file
                    try:
                        size = file_path.stat().st_size
                        file_path.unlink()
                        removed_count += 1
                        removed_size += size
                    except Exception as e:
                        print(f"  Warning: Failed to remove {file_path}: {e}")
        
        print(f"  Removed {removed_count} files ({removed_size / (1024**3):.2f} GB)")

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compress entire data directory into a zip archive"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Base data directory to compress (default: data)",
    )
    parser.add_argument(
        "--output_zip",
        type=str,
        default=None,
        help="Output zip file path (default: ../data.zip)",
    )
    parser.add_argument(
        "--remove_original",
        action="store_true",
        help="Remove original files after compression (use with caution!)",
    )
    parser.add_argument(
        "--include_temp",
        action="store_true",
        help="Include temp_outputs directory in compression",
    )

    args = parser.parse_args()

    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    output_zip_path: Optional[Path] = None
    if args.output_zip:
        output_zip_path = Path(args.output_zip)

    success = compress_data_folder(
        data_dir=Path(args.data_dir),
        output_zip=output_zip_path,
        remove_original=args.remove_original,
        include_temp=args.include_temp,
    )

    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()

