"""Paired Hazy and Clear Image Dataset and DataLoader for PyTorch.

Discovers, validates, and loads paired hazy and ground-truth clear images
for training and evaluating visibility enhancement systems.
"""

import os
from pathlib import Path
import random
from typing import Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from app.config import SUPPORTED_IMAGE_EXTENSIONS
from src.utils.exceptions import ImageLoadError
from src.utils.image_ops import load_image, numpy_to_tensor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PairedHazeDataset(Dataset):
    """PyTorch Dataset for paired hazy and ground-truth clear images.

    Discovers matching image pairs based on file stem (filename without extension)
    from separate hazy and clear directories or subdirectories.
    """

    CLEAR_DIR_CANDIDATES = {"clear", "its_clear", "its clear", "clean", "gt", "target", "ground_truth", "original"}
    HAZY_DIR_CANDIDATES = {"hazy", "haze", "its_haze", "its haze", "synthetic", "input", "images", "hazy_images"}
    TRANS_DIR_CANDIDATES = {"trans", "its_trans", "its trans", "transmission", "depth"}

    def __init__(
        self,
        root_dir: Optional[Union[str, Path]] = None,
        hazy_dir: Optional[Union[str, Path]] = None,
        clear_dir: Optional[Union[str, Path]] = None,
        trans_dir: Optional[Union[str, Path]] = None,
        split: Optional[str] = None,
        resize_shape: Optional[Tuple[int, int]] = None,
        patch_size: Optional[Tuple[int, int]] = None,
        random_crop: bool = False,
        max_samples: Optional[int] = None,
        max_variants_per_scene: Optional[int] = None,
        seed: Optional[int] = None,
        strict_pairs: bool = True,
        device: str = "cpu",
    ) -> None:
        """Initialize the paired dataset.

        Args:
            root_dir: Base directory containing hazy and clear subdirectories, or dataset root with splits.
            hazy_dir: Explicit path to directory containing hazy images.
            clear_dir: Explicit path to directory containing clear images.
            trans_dir: Optional explicit path to directory containing auxiliary transmission maps.
            split: Optional split selector (e.g. 'train', 'val', 'test') when root_dir contains splits.
            resize_shape: Optional target (Height, Width) to resize loaded images.
            patch_size: Optional target (Height, Width) for patch extraction.
            random_crop: If True, extracts random spatial patch (training); else center patch.
            max_samples: Optional cap on maximum pairs loaded (useful for CPU limits).
            max_variants_per_scene: Optional cap on maximum haze variants per unique clear scene.
            seed: Optional random seed for reproducible scene shuffling and diversity sampling.
            strict_pairs: If True, raises ValueError if unmatched images are found.
            device: Target compute device for returned tensors ('cpu' by default).

        Raises:
            ValueError: If neither root_dir nor both hazy_dir and clear_dir are specified,
                or if strict_pairs is True and unpaired images exist.
        """
        self.trans_dir: Optional[Path] = Path(trans_dir) if trans_dir is not None else None
        self.hazy_dir, self.clear_dir, detected_trans = self._resolve_directories(
            root_dir=root_dir,
            hazy_dir=hazy_dir,
            clear_dir=clear_dir,
            split=split,
        )
        if self.trans_dir is None and detected_trans is not None:
            self.trans_dir = detected_trans

        self.resize_shape = resize_shape
        self.patch_size = patch_size
        self.random_crop = random_crop
        self.max_samples = max_samples
        self.max_variants_per_scene = max_variants_per_scene
        self.seed = seed
        self.device = device
        self.strict_pairs = strict_pairs

        # Discover and pair images
        self.pairs: List[Tuple[Path, Path, str]] = []
        self.trans_pairs: Dict[str, Path] = {}
        self.unmatched_hazy: List[Path] = []
        self.unmatched_clear: List[Path] = []
        self._discover_pairs()

        # Apply scene diversity sampling if requested
        if self.seed is not None or self.max_variants_per_scene is not None:
            scenes_dict: Dict[str, List[Tuple[Path, Path, str]]] = {}
            for h_p, c_p, h_s in self.pairs:
                scene_id = c_p.stem
                scenes_dict.setdefault(scene_id, []).append((h_p, c_p, h_s))

            scene_keys = sorted(scenes_dict.keys())
            if self.seed is not None:
                rng = random.Random(self.seed)
                rng.shuffle(scene_keys)
                for s_id in scene_keys:
                    rng.shuffle(scenes_dict[s_id])

            if self.max_variants_per_scene is not None and self.max_variants_per_scene > 0:
                for s_id in scene_keys:
                    scenes_dict[s_id] = scenes_dict[s_id][: self.max_variants_per_scene]

            # Interleave round-robin across scenes to maximize scene diversity
            interleaved: List[Tuple[Path, Path, str]] = []
            max_vars = max((len(v) for v in scenes_dict.values()), default=0)
            for v_idx in range(max_vars):
                for s_id in scene_keys:
                    if v_idx < len(scenes_dict[s_id]):
                        interleaved.append(scenes_dict[s_id][v_idx])

            self.pairs = interleaved

        if self.max_samples is not None and len(self.pairs) > self.max_samples:
            self.pairs = self.pairs[: self.max_samples]

    def _resolve_directories(
        self,
        root_dir: Optional[Union[str, Path]],
        hazy_dir: Optional[Union[str, Path]],
        clear_dir: Optional[Union[str, Path]],
        split: Optional[str] = None,
    ) -> Tuple[Path, Path, Optional[Path]]:
        """Resolve hazy, clear, and optional transmission directories."""
        if hazy_dir is not None and clear_dir is not None:
            h_path = Path(hazy_dir)
            c_path = Path(clear_dir)
            if not h_path.is_dir():
                raise ValueError(f"Hazy images directory does not exist: {h_path}")
            if not c_path.is_dir():
                raise ValueError(f"Clear images directory does not exist: {c_path}")
            return h_path, c_path, None

        if root_dir is None:
            raise ValueError(
                "Either 'root_dir' (containing hazy and clear subdirectories) "
                "or both 'hazy_dir' and 'clear_dir' must be provided."
            )

        root = Path(root_dir)
        if not root.is_dir():
            raise ValueError(f"Specified root_dir does not exist: {root}")

        # If split is requested, navigate to split subfolder if it exists
        if split is not None:
            split_candidate = root / split
            if split_candidate.is_dir():
                root = split_candidate
            else:
                raise ValueError(f"Requested split '{split}' not found under '{root}'")

        # 1. Direct standard names: root/hazy and root/clear
        if (root / "hazy").is_dir() and (root / "clear").is_dir():
            trans_candidate = root / "trans" if (root / "trans").is_dir() else None
            return root / "hazy", root / "clear", trans_candidate

        # 2. Check if root contains 'train' / 'val' subfolders without split specified
        if split is None and (root / "train").is_dir() and (root / "val").is_dir():
            logger.info(f"PairedHazeDataset: Root '{root}' contains official splits; defaulting to 'train'.")
            return self._resolve_directories(root / "train", None, None, None)

        # 3. Dynamic scan of child folders in root (handles RESIDE ITS_clear, ITS_haze, etc.)
        found_clear: Optional[Path] = None
        found_hazy: Optional[Path] = None
        found_trans: Optional[Path] = None

        for entry in os.scandir(root):
            if entry.is_dir():
                name_norm = entry.name.lower().replace("-", "_").strip()
                # Exclude transmission directory from being chosen as clear or hazy
                if name_norm in self.TRANS_DIR_CANDIDATES or any(name_norm.endswith(tn) for tn in ["_trans", " trans"]):
                    found_trans = Path(entry.path)
                    continue
                if name_norm in self.CLEAR_DIR_CANDIDATES or any(name_norm.endswith(cn) for cn in ["_clear", " clear"]):
                    if found_clear is None:
                        found_clear = Path(entry.path)
                elif name_norm in self.HAZY_DIR_CANDIDATES or any(name_norm.endswith(hn) for hn in ["_haze", " haze", "_hazy", " hazy"]):
                    if found_hazy is None:
                        found_hazy = Path(entry.path)

        if found_clear is not None and found_hazy is not None:
            return found_hazy, found_clear, found_trans

        raise ValueError(
            f"Could not automatically detect clear and hazy subdirectories in '{root}'. "
            f"Found clear: {found_clear}, Found hazy: {found_hazy}. "
            f"Expected directories matching known clear/haze patterns."
        )

    @staticmethod
    def _collect_image_entries(dir_path: Path) -> List[Tuple[Path, str]]:
        """Collect image files from directory, discovering nested subdirectories if present."""
        entries: List[Tuple[Path, str]] = []
        has_subdirs = False
        for entry in os.scandir(dir_path):
            if entry.is_file():
                ext = Path(entry.name).suffix.lower()
                if ext in SUPPORTED_IMAGE_EXTENSIONS:
                    entries.append((Path(entry.path), Path(entry.name).stem))
            elif entry.is_dir():
                has_subdirs = True

        # If subdirectories exist (e.g. part1, part2 in OTS), perform recursive discovery
        if has_subdirs and (len(entries) == 0 or len(entries) < 10):
            entries = []
            for root, _, files in os.walk(dir_path):
                for f in files:
                    ext = Path(f).suffix.lower()
                    if ext in SUPPORTED_IMAGE_EXTENSIONS:
                        p = Path(root) / f
                        entries.append((p, p.stem))
        return entries

    def _discover_pairs(self) -> None:
        """Scan directories and establish matching pairs (exact stem or longest-prefix match)."""
        clear_entries = self._collect_image_entries(self.clear_dir)
        clear_map: Dict[str, Path] = {stem: p for p, stem in clear_entries}

        trans_map: Dict[str, Path] = {}
        if self.trans_dir is not None and self.trans_dir.is_dir():
            trans_entries = self._collect_image_entries(self.trans_dir)
            trans_map = {stem: p for p, stem in trans_entries}

        hazy_entries = self._collect_image_entries(self.hazy_dir)

        # Sort for deterministic ordering across OS/platforms
        hazy_entries.sort(key=lambda x: x[1])

        matched_clear_stems: Set[str] = set()

        for h_path, h_stem in hazy_entries:
            c_path = None
            # 1. Exact stem match
            if h_stem in clear_map:
                c_path = clear_map[h_stem]
                matched_clear_stems.add(h_stem)
            # 2. Longest matching prefix across underscores (handles RESIDE ITS and multi-part OTS)
            elif "_" in h_stem:
                parts = h_stem.split("_")
                for k in range(len(parts) - 1, 0, -1):
                    cand = "_".join(parts[:k])
                    if cand in clear_map:
                        c_path = clear_map[cand]
                        matched_clear_stems.add(cand)
                        break
            # 3. Longest matching prefix across hyphens
            if c_path is None and "-" in h_stem:
                parts = h_stem.split("-")
                for k in range(len(parts) - 1, 0, -1):
                    cand = "-".join(parts[:k])
                    if cand in clear_map:
                        c_path = clear_map[cand]
                        matched_clear_stems.add(cand)
                        break

            if c_path is not None:
                self.pairs.append((h_path, c_path, h_stem))
                # Optionally link transmission map if available
                if trans_map:
                    if h_stem in trans_map:
                        self.trans_pairs[h_stem] = trans_map[h_stem]
                    elif "_" in h_stem:
                        parts = h_stem.split("_")
                        for k in range(len(parts) - 1, 0, -1):
                            cand = "_".join(parts[:k])
                            if cand in trans_map:
                                self.trans_pairs[h_stem] = trans_map[cand]
                                break
            else:
                self.unmatched_hazy.append(h_path)

        for c_stem, path in clear_map.items():
            if c_stem not in matched_clear_stems:
                self.unmatched_clear.append(path)

        logger.info(
            f"PairedHazeDataset: discovered {len(self.pairs)} valid pairs "
            f"({len(self.unmatched_hazy)} unmatched hazy, {len(self.unmatched_clear)} unmatched clear)"
        )

        if self.strict_pairs and (self.unmatched_hazy or self.unmatched_clear):
            msg_parts = []
            if self.unmatched_hazy:
                msg_parts.append(
                    f"{len(self.unmatched_hazy)} hazy images have no matching clear pair "
                    f"(e.g., {[p.name for p in self.unmatched_hazy[:3]]})"
                )
            if self.unmatched_clear:
                msg_parts.append(
                    f"{len(self.unmatched_clear)} clear images have no matching hazy pair "
                    f"(e.g., {[p.name for p in self.unmatched_clear[:3]]})"
                )
            raise ValueError(
                f"Strict pairing failed in dataset '{self.hazy_dir.parent}': " + "; ".join(msg_parts)
            )

        if len(self.pairs) == 0:
            logger.warning(f"No matching image pairs found between {self.hazy_dir} and {self.clear_dir}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """Load and return a single paired sample.

        Args:
            idx: Integer index.

        Returns:
            dict containing:
                - 'hazy': PyTorch tensor (C, H, W) [0.0, 1.0]
                - 'clear': PyTorch tensor (C, H, W) [0.0, 1.0]
                - 'stem': Image name without extension
                - 'hazy_path': String path to hazy image
                - 'clear_path': String path to clear image

        Raises:
            ImageLoadError: If loading either image fails or shapes do not match.
        """
        hazy_path, clear_path, stem = self.pairs[idx]

        hazy_rgb = load_image(hazy_path)
        clear_rgb = load_image(clear_path)

        # Validate channel counts
        if hazy_rgb.ndim != 3 or hazy_rgb.shape[2] != 3:
            raise ImageLoadError(
                f"Hazy image '{hazy_path.name}' must have 3 RGB channels, got shape {hazy_rgb.shape}"
            )
        if clear_rgb.ndim != 3 or clear_rgb.shape[2] != 3:
            raise ImageLoadError(
                f"Clear image '{clear_path.name}' must have 3 RGB channels, got shape {clear_rgb.shape}"
            )

        # Optional patch extraction (aligned random crop or center crop)
        if self.patch_size is not None:
            patch_h, patch_w = self.patch_size
            curr_h, curr_w = hazy_rgb.shape[:2]

            # If image smaller than patch, resize up to patch size
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

        # Optional full-image resizing (only if patch_size is None)
        elif self.resize_shape is not None:
            target_h, target_w = self.resize_shape
            hazy_rgb = cv2.resize(hazy_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
            clear_rgb = cv2.resize(clear_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Validate matching spatial dimensions
        if hazy_rgb.shape[:2] != clear_rgb.shape[:2]:
            raise ImageLoadError(
                f"Dimension mismatch for pair '{stem}': hazy is {hazy_rgb.shape[:2]}, "
                f"clear is {clear_rgb.shape[:2]}"
            )

        # Convert to normalized float32 tensors (C, H, W) on target device
        hazy_tensor = numpy_to_tensor(hazy_rgb, device=self.device)
        clear_tensor = numpy_to_tensor(clear_rgb, device=self.device)

        sample = {
            "hazy": hazy_tensor,
            "clear": clear_tensor,
            "stem": stem,
            "hazy_path": str(hazy_path),
            "clear_path": str(clear_path),
        }
        if stem in self.trans_pairs and self.trans_pairs[stem] is not None:
            sample["trans_path"] = str(self.trans_pairs[stem])

        return sample


def create_paired_dataloader(
    dataset_or_dir: Union[PairedHazeDataset, str, Path],
    split: Optional[str] = None,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    resize_shape: Optional[Tuple[int, int]] = None,
    patch_size: Optional[Tuple[int, int]] = None,
    random_crop: bool = False,
    max_samples: Optional[int] = None,
    max_variants_per_scene: Optional[int] = None,
    seed: Optional[int] = None,
    strict_pairs: bool = True,
    device: str = "cpu",
) -> DataLoader:
    """Convenience helper to create a PyTorch DataLoader for paired hazy/clear images.

    Args:
        dataset_or_dir: An instantiated PairedHazeDataset or a root directory path.
        split: Optional split selector (e.g. 'train', 'val') when using dataset root.
        batch_size: Batch size (default 1 for varying resolutions).
        shuffle: Whether to shuffle samples each epoch.
        num_workers: Number of background worker processes (0 for main process on Windows/CPU).
        resize_shape: Optional (Height, Width) resize tuple if creating dataset from path.
        patch_size: Optional (Height, Width) for patch-based training crop.
        random_crop: Whether to randomly sample crop windows during training.
        max_samples: Optional cap on maximum samples loaded.
        max_variants_per_scene: Optional cap on maximum haze variants per unique clear scene.
        seed: Optional random seed for reproducible scene shuffling and diversity sampling.
        strict_pairs: Whether to enforce strict pair matching if creating from path.
        device: Target compute device.

    Returns:
        DataLoader: PyTorch DataLoader ready for iteration.
    """
    if isinstance(dataset_or_dir, PairedHazeDataset):
        dataset = dataset_or_dir
    else:
        dataset = PairedHazeDataset(
            root_dir=dataset_or_dir,
            split=split,
            resize_shape=resize_shape,
            patch_size=patch_size,
            random_crop=random_crop,
            max_samples=max_samples,
            max_variants_per_scene=max_variants_per_scene,
            seed=seed,
            strict_pairs=strict_pairs,
            device=device,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
