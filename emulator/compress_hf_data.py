"""Compress HF data into a zip file for better data management.

This script compresses all HF simulation outputs into a zip archive.
Can be run standalone or as a SLURM job after HF generation completes.
"""

import argparse
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional


def compress_hf_data(
    data_dir: Path,
    output_zip: Optional[Path] = None,
    remove_original: bool = False,
    include_materials: bool = False,
):
    """
    Compress HF data into a zip file.

    Args:
        data_dir: Base data directory
        output_zip: Output zip file path (default: data_dir/high_fidelity.zip)
        remove_original: If True, remove original files after compression
        include_materials: If True, also include HF material grids
    """
    data_dir = Path(data_dir)
    hf_output_dir = data_dir / "high_fidelity" / "output_accel"
    hf_pga_dir = data_dir / "high_fidelity" / "pga"
    materials_hf_dir = data_dir / "materials_hf"

    if output_zip is None:
        output_zip = data_dir / "high_fidelity.zip"
    else:
        output_zip = Path(output_zip)

    print("=" * 70)
    print("HF Data Compression")
    print("=" * 70)
    print(f"Data Directory: {data_dir}")
    print(f"Output ZIP: {output_zip}")
    print(f"Remove Original: {remove_original}")
    print(f"Include Materials: {include_materials}")
    print("=" * 70)

    # Check if directories exist
    if not hf_output_dir.exists():
        print(f"Warning: HF output directory not found: {hf_output_dir}")
        print("No HF data to compress.")
        return False

    if not hf_pga_dir.exists():
        print(f"Warning: HF PGA directory not found: {hf_pga_dir}")
        print("No HF data to compress.")
        return False

    # Count files
    accel_files = list(hf_output_dir.glob("sim_*.npy"))
    pga_files = list(hf_pga_dir.glob("sim_*.npy"))

    print(f"\nFound {len(accel_files)} acceleration files")
    print(f"Found {len(pga_files)} PGA files")

    if len(accel_files) == 0 and len(pga_files) == 0:
        print("No HF data files found. Nothing to compress.")
        return False

    # Create zip file
    print(f"\nCreating zip archive: {output_zip}")
    start_time = time.time()

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add acceleration files
        print("Adding acceleration files...")
        for accel_file in sorted(accel_files):
            arcname = f"high_fidelity/output_accel/{accel_file.name}"
            zipf.write(accel_file, arcname)
            if len(accel_files) <= 20 or accel_files.index(accel_file) % 10 == 0:
                print(f"  Added: {accel_file.name}")

        # Add PGA files
        print("Adding PGA files...")
        for pga_file in sorted(pga_files):
            arcname = f"high_fidelity/pga/{pga_file.name}"
            zipf.write(pga_file, arcname)
            if len(pga_files) <= 20 or pga_files.index(pga_file) % 10 == 0:
                print(f"  Added: {pga_file.name}")

        # Optionally add material grids
        if include_materials and materials_hf_dir.exists():
            print("Adding HF material grids...")
            material_files = list(materials_hf_dir.glob("sim_*.npy"))
            for mat_file in sorted(material_files):
                arcname = f"materials_hf/{mat_file.name}"
                zipf.write(mat_file, arcname)
                if (
                    len(material_files) <= 20
                    or material_files.index(mat_file) % 10 == 0
                ):
                    print(f"  Added: {mat_file.name}")
            print(f"  Added {len(material_files)} material grid files")

    elapsed = time.time() - start_time
    zip_size = output_zip.stat().st_size / (1024**3)  # Size in GB

    print("\n✓ Compression complete!")
    print(f"  Archive: {output_zip}")
    print(f"  Size: {zip_size:.2f} GB")
    print(f"  Time: {elapsed:.1f} seconds")
    print(f"  Files: {len(accel_files)} acceleration + {len(pga_files)} PGA")

    # Optionally remove original files
    if remove_original:
        print("\nRemoving original files...")
        removed_count = 0
        for accel_file in accel_files:
            accel_file.unlink()
            removed_count += 1
        for pga_file in pga_files:
            pga_file.unlink()
            removed_count += 1
        print(f"  Removed {removed_count} original files")

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compress HF data into a zip archive")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Base data directory (default: data)",
    )
    parser.add_argument(
        "--output_zip",
        type=str,
        default=None,
        help="Output zip file path (default: data_dir/high_fidelity.zip)",
    )
    parser.add_argument(
        "--remove_original",
        action="store_true",
        help="Remove original files after compression (use with caution!)",
    )
    parser.add_argument(
        "--include_materials",
        action="store_true",
        help="Also include HF material grids in the archive",
    )

    args = parser.parse_args()

    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    output_zip_path: Optional[Path] = None
    if args.output_zip:
        output_zip_path = Path(args.output_zip)

    success = compress_hf_data(
        data_dir=Path(args.data_dir),
        output_zip=output_zip_path,
        remove_original=args.remove_original,
        include_materials=args.include_materials,
    )

    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
