"""Dataset class for PGA emulator.

Loads pairs of (vs_field_hf, pga_lf) for training.
- Input: 150x150 Vs field (HF resolution)
- Target: Scalar PGA from LF solver
- Oracle: Scalar PGA from HF solver (for evaluation only)
"""

import sys
from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# Add parent directory to path to allow importing emulator
sys.path.insert(0, str(Path(__file__).parent.parent))

from emulator.data_utils import (
    denormalize_vs_field,
    load_normalizer,
    normalize_vs_field,
)


class PGADataset(Dataset):
    """Dataset for PGA emulator.

    Loads pairs of (vs_field_hf, pga_lf) for training.
    - Input: 150x150 Vs field image (HF resolution)
    - Target: Scalar PGA from LF solver
    - Oracle: Scalar PGA from HF solver (for evaluation only, not used in training)
    """

    def __init__(
        self,
        data_dir: Path,
        indices: list[int],
        normalizer_path: Optional[Path] = None,
        split: str = "train",
    ):
        """Initialize dataset.

        Args:
            data_dir: Base data directory
            indices: List of simulation indices to include
            normalizer_path: Path to normalization statistics file
            split: Dataset split ("train", "val", "test")
        """
        # Resolve to absolute path to avoid issues with working directory changes
        self.data_dir = Path(data_dir).resolve()
        self.indices = indices
        self.split = split

        # Load normalization stats
        if normalizer_path is not None:
            self.stats = load_normalizer(Path(normalizer_path))
        else:
            # No normalization
            self.stats = None

        # Verify files exist
        # Use HF materials for input (150x150)
        self.materials_dir = self.data_dir / "materials_hf"
        if not self.materials_dir.exists():
            # Fallback to regular materials if HF version doesn't exist
            self.materials_dir = self.data_dir / "materials"

        self.lf_pga_dir = self.data_dir / "low_fidelity" / "pga"
        self.hf_pga_dir = self.data_dir / "high_fidelity" / "pga"

        # Filter indices to only those with all required files
        self.valid_indices = []
        missing_files = []
        for idx in self.indices:
            mat_file = self.materials_dir / f"sim_{idx:04d}.npy"
            pga_lf_file = self.lf_pga_dir / f"sim_{idx:04d}.npy"

            if mat_file.exists() and pga_lf_file.exists():
                self.valid_indices.append(idx)
            else:
                missing = []
                if not mat_file.exists():
                    missing.append(f"material: {mat_file}")
                if not pga_lf_file.exists():
                    missing.append(f"pga_lf: {pga_lf_file}")
                missing_files.append((idx, missing))

        if len(self.valid_indices) == 0:
            error_msg = f"No valid data files found for split {split}\n"
            error_msg += f"  Data directory: {self.data_dir}\n"
            error_msg += f"  Materials directory: {self.materials_dir}\n"
            error_msg += f"  LF PGA directory: {self.lf_pga_dir}\n"
            error_msg += f"  Requested indices: {self.indices[:10]}{'...' if len(self.indices) > 10 else ''}\n"
            if missing_files:
                error_msg += "  First few missing files:\n"
                for idx, missing in missing_files[:5]:
                    error_msg += f"    sim_{idx:04d}: {', '.join(missing)}\n"
            raise ValueError(error_msg)

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | float]:
        """Get a single data sample.

        Args:
            idx: Index into valid_indices

        Returns:
            Dictionary with:
            - "vs_field": Vs field image (1, H, W) where H=W=150
            - "pga_lf": Scalar PGA from LF solver
            - "pga_hf": Scalar PGA from HF solver (oracle, only for test/val)
            - "sim_id": Simulation ID
        """
        sim_id = self.valid_indices[idx]

        # Load material grid (H, W, 2) and extract only Vs field (first channel)
        material = np.load(self.materials_dir / f"sim_{sim_id:04d}.npy")
        vs_field = material[:, :, 0]  # (H, W) - extract Vs channel only

        # Ensure it's 150x150 (resize if necessary)
        if vs_field.shape != (150, 150):
            from scipy.ndimage import zoom

            zoom_factors = (150 / vs_field.shape[0], 150 / vs_field.shape[1])
            vs_field = zoom(vs_field, zoom_factors, order=1)

        # Add channel dimension: (1, H, W)
        vs_field = vs_field[np.newaxis, :, :]

        # Load LF PGA (scalar)
        pga_lf = float(np.load(self.lf_pga_dir / f"sim_{sim_id:04d}.npy"))

        # Load HF PGA (oracle) if available (for test/val)
        pga_hf = None
        if self.split in ["test", "val"]:
            hf_pga_file = self.hf_pga_dir / f"sim_{sim_id:04d}.npy"
            if hf_pga_file.exists():
                pga_hf = float(np.load(hf_pga_file))

        # Apply normalization if stats available
        if self.stats is not None:
            # Normalize Vs field
            vs_field_2d = np.asarray(vs_field[0])  # (H, W) - ensure it's ndarray
            vs_field = normalize_vs_field(vs_field_2d, self.stats["vs_field"])  # (H, W)
            vs_field = vs_field[np.newaxis, :, :]  # (1, H, W)

        # Convert to tensors
        result = {
            "vs_field": torch.FloatTensor(vs_field),  # (1, 150, 150)
            "pga_lf": torch.FloatTensor([pga_lf]),  # (1,) - scalar as tensor
            "sim_id": sim_id,
        }

        if pga_hf is not None:
            result["pga_hf"] = torch.FloatTensor([pga_hf])  # (1,) - oracle

        return result

    def denormalize_sample(  # type: ignore[return]
        self, sample: Mapping[str, torch.Tensor | int | float]
    ) -> dict[str, torch.Tensor | int | float]:
        """Denormalize a sample (for visualization/debugging).

        Args:
            sample: Sample dictionary with normalized tensors and sim_id

        Returns:
            Dictionary with denormalized numpy arrays and sim_id
        """
        if self.stats is None:
            denorm_result: Mapping[str, torch.Tensor | int | float] = {}
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    denorm_result[k] = v.float()
                else:
                    denorm_result[k] = v
            return dict(denorm_result)

        # Extract tensors
        vs_field_tensor = sample["vs_field"]
        sim_id = sample["sim_id"]

        if not isinstance(vs_field_tensor, torch.Tensor):
            raise TypeError("vs_field must be a torch.Tensor")

        vs_field_np: np.ndarray = vs_field_tensor.numpy()[0]  # (H, W)
        vs_field_denorm: np.ndarray = denormalize_vs_field(
            vs_field_np, self.stats["vs_field"]
        )

        result = {
            "vs_field": torch.from_numpy(vs_field_denorm).float(),  # (H, W)
            "sim_id": sim_id,
        }

        # Copy other fields
        for k, v in sample.items():
            if k not in ["vs_field", "sim_id"]:
                result[k] = v

        return result
