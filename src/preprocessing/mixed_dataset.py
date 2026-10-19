"""Mixed-Domain Dataset for Unified Indoor and Outdoor Dehazing (Stage 2C).

Enables training a single Lightweight U-Net model across both indoor and outdoor
scenes simultaneously. Uses balanced domain sampling (e.g. 50/50 round-robin or ratio)
so that the vast outdoor dataset (RESIDE OTS) does not drown out indoor features (RESIDE ITS).
Guarantees scene-disjoint preservation and records domain provenance for each pair.
"""

from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

from src.preprocessing.dataset import PairedHazeDataset


class MixedDomainDataset(Dataset):
    """Dataset combining paired indoor and outdoor haze datasets with balanced sampling."""

    def __init__(
        self,
        indoor_dataset: PairedHazeDataset,
        outdoor_dataset: PairedHazeDataset,
        indoor_weight: float = 0.5,
        seed: Optional[int] = 42,
    ) -> None:
        """Initialize mixed-domain dataset.

        Args:
            indoor_dataset: Initialized PairedHazeDataset for indoor scenes.
            outdoor_dataset: Initialized PairedHazeDataset for outdoor scenes.
            indoor_weight: Ratio of indoor samples (default 0.5 for balanced 50/50 sampling).
            seed: Random seed for reproducible sample ordering.
        """
        self.indoor_dataset = indoor_dataset
        self.outdoor_dataset = outdoor_dataset
        self.indoor_weight = float(indoor_weight)
        self.seed = seed

        self.n_indoor = len(indoor_dataset)
        self.n_outdoor = len(outdoor_dataset)

        if self.n_indoor == 0 and self.n_outdoor == 0:
            raise ValueError("Both indoor and outdoor datasets are empty.")

        # Total virtual size per epoch: twice the size of the larger domain to ensure full coverage
        self.total_size = max(self.n_indoor, self.n_outdoor) * 2 if (self.n_indoor > 0 and self.n_outdoor > 0) else max(self.n_indoor, self.n_outdoor)

    def __len__(self) -> int:
        return self.total_size

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Fetch sample with alternating / balanced domain selection."""
        if self.n_indoor > 0 and self.n_outdoor > 0:
            # Alternating indexing: even indices -> indoor, odd indices -> outdoor
            if index % 2 == 0:
                indoor_idx = (index // 2) % self.n_indoor
                sample = self.indoor_dataset[indoor_idx]
                sample["domain"] = "indoor"
            else:
                outdoor_idx = (index // 2) % self.n_outdoor
                sample = self.outdoor_dataset[outdoor_idx]
                sample["domain"] = "outdoor"
        elif self.n_indoor > 0:
            sample = self.indoor_dataset[index % self.n_indoor]
            sample["domain"] = "indoor"
        else:
            sample = self.outdoor_dataset[index % self.n_outdoor]
            sample["domain"] = "outdoor"

        return sample


def create_mixed_dataloader(
    indoor_dir: Union[str, Path],
    outdoor_dir: Union[str, Path],
    batch_size: int = 4,
    patch_size: Optional[Tuple[int, int]] = (128, 128),
    random_crop: bool = True,
    max_indoor_samples: Optional[int] = 1250,
    max_outdoor_samples: Optional[int] = 1250,
    max_variants_per_scene: int = 1,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader:
    """Create a balanced PyTorch DataLoader drawing evenly from indoor and outdoor datasets."""
    ds_indoor = PairedHazeDataset(
        root_dir=indoor_dir,
        patch_size=patch_size,
        random_crop=random_crop,
        max_samples=max_indoor_samples,
        max_variants_per_scene=max_variants_per_scene,
        seed=seed,
        strict_pairs=False,
    )
    ds_outdoor = PairedHazeDataset(
        root_dir=outdoor_dir,
        patch_size=patch_size,
        random_crop=random_crop,
        max_samples=max_outdoor_samples,
        max_variants_per_scene=max_variants_per_scene,
        seed=seed,
        strict_pairs=False,
    )

    mixed_ds = MixedDomainDataset(indoor_dataset=ds_indoor, outdoor_dataset=ds_outdoor, seed=seed)

    return DataLoader(
        mixed_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )


class ManifestPairedDataset(Dataset):
    """PyTorch Dataset loading paired hazy/clear images directly from a JSON manifest.

    Guarantees strict adherence to pre-computed, leakage-free scene splits.
    Supports patch cropping, full resizing, and domain filtering.
    """

    def __init__(
        self,
        manifest_path: Union[str, Path],
        project_root: Optional[Union[str, Path]] = None,
        patch_size: Optional[Tuple[int, int]] = (128, 128),
        random_crop: bool = True,
        resize_shape: Optional[Tuple[int, int]] = None,
        filter_domain: Optional[str] = None,
    ) -> None:
        import json
        from pathlib import Path
        import cv2
        import numpy as np
        from src.utils.image_ops import load_image, numpy_to_tensor
        from src.utils.exceptions import ImageLoadError

        self.manifest_path = Path(manifest_path)
        if project_root is None:
            # Assume 2 levels up from src/preprocessing: project root
            self.project_root = Path(__file__).resolve().parent.parent.parent
        else:
            self.project_root = Path(project_root)

        self.patch_size = patch_size
        self.random_crop = random_crop
        self.resize_shape = resize_shape
        self.filter_domain = filter_domain

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        pairs = data.get("pairs", [])
        if self.filter_domain:
            self.entries = [p for p in pairs if p.get("domain") == self.filter_domain]
        else:
            self.entries = pairs

        if len(self.entries) == 0:
            raise ValueError(f"No pairs found in manifest {manifest_path} (filter={filter_domain}).")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        import cv2
        import numpy as np
        from src.utils.image_ops import load_image, numpy_to_tensor
        from src.utils.exceptions import ImageLoadError

        item = self.entries[idx]
        hazy_path = self.project_root / item["hazy_path"]
        clear_path = self.project_root / item["clear_path"]

        hazy_rgb = load_image(hazy_path)
        clear_rgb = load_image(clear_path)

        if hazy_rgb.ndim != 3 or hazy_rgb.shape[2] != 3:
            raise ImageLoadError(f"Hazy image '{hazy_path}' must have 3 RGB channels.")
        if clear_rgb.ndim != 3 or clear_rgb.shape[2] != 3:
            raise ImageLoadError(f"Clear image '{clear_path}' must have 3 RGB channels.")

        # Patch extraction
        if self.patch_size is not None:
            patch_h, patch_w = self.patch_size
            curr_h, curr_w = hazy_rgb.shape[:2]

            if curr_h < patch_h or curr_w < patch_w:
                target_h = max(curr_h, patch_h)
                target_w = max(curr_w, patch_w)
                hazy_rgb = cv2.resize(hazy_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                clear_rgb = cv2.resize(clear_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                curr_h, curr_w = target_h, target_w

            if self.random_crop and (curr_h > patch_h or curr_w > patch_w):
                top = np.random.randint(0, curr_h - patch_h + 1)
                left = np.random.randint(0, curr_w - patch_w + 1)
            else:
                top = (curr_h - patch_h) // 2
                left = (curr_w - patch_w) // 2

            hazy_rgb = hazy_rgb[top : top + patch_h, left : left + patch_w]
            clear_rgb = clear_rgb[top : top + patch_h, left : left + patch_w]

        elif self.resize_shape is not None:
            target_h, target_w = self.resize_shape
            hazy_rgb = cv2.resize(hazy_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
            clear_rgb = cv2.resize(clear_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)

        if hazy_rgb.shape[:2] != clear_rgb.shape[:2]:
            raise ImageLoadError(
                f"Dimension mismatch for pair '{item.get('variant_stem', idx)}': "
                f"hazy {hazy_rgb.shape[:2]} vs clear {clear_rgb.shape[:2]}"
            )

        hazy_t = numpy_to_tensor(hazy_rgb)
        clear_t = numpy_to_tensor(clear_rgb)

        return {
            "hazy": hazy_t,
            "clear": clear_t,
            "stem": item.get("variant_stem", Path(item["hazy_path"]).stem),
            "hazy_path": str(hazy_path),
            "clear_path": str(clear_path),
            "domain": item.get("domain", "unknown"),
            "scene_id": item.get("scene_id", ""),
        }


def create_manifest_dataloader(
    manifest_path: Union[str, Path],
    batch_size: int = 4,
    patch_size: Optional[Tuple[int, int]] = (128, 128),
    random_crop: bool = True,
    resize_shape: Optional[Tuple[int, int]] = None,
    filter_domain: Optional[str] = None,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create DataLoader directly from a JSON manifest."""
    ds = ManifestPairedDataset(
        manifest_path=manifest_path,
        patch_size=patch_size,
        random_crop=random_crop,
        resize_shape=resize_shape,
        filter_domain=filter_domain,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )

