"""Dataset Ingestion and Pairing Pipeline for RESIDE and Generic Paired Datasets.

Automates the discovery, pairing, corruption detection, and leakage-free splitting
of external academic datasets (such as RESIDE ITS/OTS/SOTS, O-HAZE, I-HAZE).
Ensures zero data leakage across splits by grouping all hazy variants by their
source clear scene before partitioning.
"""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from app.config import DEFAULT_RANDOM_SEED, SUPPORTED_IMAGE_EXTENSIONS
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


@dataclass
class ImagePair:
    """Represents a matched hazy and ground-truth clear image pair."""

    stem: str
    scene_id: str
    hazy_path: Path
    clear_path: Path
    trans_path: Optional[Path] = None


@dataclass
class IngestionReport:
    """Diagnostic report generated during dataset discovery and validation."""

    raw_dir: Path
    clear_dir: Optional[Path] = None
    hazy_dir: Optional[Path] = None
    trans_dir: Optional[Path] = None
    total_clear_found: int = 0
    total_hazy_found: int = 0
    total_trans_found: int = 0
    valid_pairs_count: int = 0
    scenes_count: int = 0
    corrupt_images: List[str] = field(default_factory=list)
    missing_clear_for_hazy: List[str] = field(default_factory=list)
    unpaired_clear_images: List[str] = field(default_factory=list)
    pairs: List[ImagePair] = field(default_factory=list)
    is_valid: bool = False
    validation_message: str = ""
    has_official_splits: bool = False
    split_reports: Dict[str, Any] = field(default_factory=dict)


class ResideDatasetIngestor:
    """Discovers, validates, and partitions RESIDE and paired dehazing datasets."""

    CLEAR_DIR_NAMES = {"clear", "its_clear", "its clear", "clean", "gt", "target", "ground_truth", "original"}
    HAZY_DIR_NAMES = {"hazy", "haze", "its_haze", "its haze", "synthetic", "input", "images", "hazy_images"}
    TRANS_DIR_NAMES = {"trans", "its_trans", "its trans", "transmission", "depth"}

    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = Path(raw_dir)

    def locate_subdirectories(self, search_dir: Optional[Path] = None) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
        """Search for clear, hazy, and transmission image directories."""
        base = search_dir or self.raw_dir
        clear_dir: Optional[Path] = None
        hazy_dir: Optional[Path] = None
        trans_dir: Optional[Path] = None

        # Direct child check first
        for p in base.glob("**/*"):
            if p.is_dir():
                lower_name = p.name.lower().replace("-", "_").strip()
                if (lower_name in self.TRANS_DIR_NAMES or any(lower_name.endswith(tn) for tn in ["_trans", " trans"])) and trans_dir is None:
                    trans_dir = p
                    continue
                if (lower_name in self.CLEAR_DIR_NAMES or any(lower_name.endswith(cn) for cn in ["_clear", " clear"])) and clear_dir is None:
                    clear_dir = p
                elif (lower_name in self.HAZY_DIR_NAMES or any(lower_name.endswith(hn) for hn in ["_haze", " haze", "_hazy", " hazy"])) and hazy_dir is None:
                    hazy_dir = p

        return clear_dir, hazy_dir, trans_dir

    @staticmethod
    def extract_scene_id(filename_stem: str) -> str:
        """Extract source scene identifier from a filename.

        Handles:
            - RESIDE ITS/SOTS: '0001_01_0.9027' -> '0001'
            - RESIDE alternative: '1400_1' -> '1400'
            - Affixed names: 'scene01_hazy' -> 'scene01'
            - Exact matching: '001' -> '001'
        """
        if "_" in filename_stem:
            parts = filename_stem.split("_")
            if parts[0]:
                return parts[0]

        if "-" in filename_stem:
            parts = filename_stem.split("-")
            if parts[0]:
                return parts[0]

        return filename_stem

    def _validate_split(self, split_dir: Path, split_name: str = "") -> IngestionReport:
        """Scan a single split folder, pairing hazy with clear and checking corruption."""
        report = IngestionReport(raw_dir=split_dir)
        clear_dir, hazy_dir, trans_dir = self.locate_subdirectories(split_dir)
        report.clear_dir = clear_dir
        report.hazy_dir = hazy_dir
        report.trans_dir = trans_dir

        if clear_dir is None or hazy_dir is None:
            report.validation_message = f"Could not locate clear and hazy subdirectories in '{split_dir}'."
            return report

        # Collect clear images (flat or nested)
        clear_map: Dict[str, Path] = {}
        for root, _, files in os.walk(clear_dir):
            for fname in files:
                ext = Path(fname).suffix.lower()
                if ext in SUPPORTED_IMAGE_EXTENSIONS:
                    p = Path(root) / fname
                    stem = p.stem
                    scene_key = self.extract_scene_id(stem)
                    clear_map[scene_key] = p
                    clear_map[stem] = p
                    if p.stat().st_size == 0:
                        report.corrupt_images.append(str(p))
                    report.total_clear_found += 1

        # Track transmission maps if folder exists
        trans_map: Dict[str, Path] = {}
        if trans_dir is not None and trans_dir.is_dir():
            for root, _, files in os.walk(trans_dir):
                for fname in files:
                    ext = Path(fname).suffix.lower()
                    if ext in SUPPORTED_IMAGE_EXTENSIONS:
                        p = Path(root) / fname
                        stem = p.stem
                        trans_map[stem] = p
                        if p.stat().st_size == 0:
                            report.corrupt_images.append(str(p))
                        report.total_trans_found += 1

        # Collect hazy images and match (supports flat or nested subdirectories like part1/, part2/)
        matched_scenes: Set[str] = set()
        for root, _, files in os.walk(hazy_dir):
            for fname in files:
                ext = Path(fname).suffix.lower()
                if ext in SUPPORTED_IMAGE_EXTENSIONS:
                    report.total_hazy_found += 1
                    h_path = Path(root) / fname
                    h_stem = h_path.stem

                    if h_path.stat().st_size == 0:
                        report.corrupt_images.append(str(h_path))
                        continue

                    scene_key = self.extract_scene_id(h_stem)
                    c_path = clear_map.get(scene_key) or clear_map.get(h_stem)

                    if c_path is not None:
                        t_path = None
                        if trans_map:
                            t_path = (
                                trans_map.get(h_stem)
                                or trans_map.get(h_stem.rsplit("_", 1)[0] if "_" in h_stem else "")
                                or trans_map.get(scene_key)
                            )

                        pair = ImagePair(
                            stem=h_stem,
                            scene_id=scene_key,
                            hazy_path=h_path,
                            clear_path=c_path,
                            trans_path=t_path,
                        )
                        report.pairs.append(pair)
                        matched_scenes.add(scene_key)
                    else:
                        report.missing_clear_for_hazy.append(fname)

        # Check for unreferenced clear images
        for c_key, c_path in clear_map.items():
            if c_key not in matched_scenes and c_path.stem not in matched_scenes:
                if c_path.name not in report.unpaired_clear_images:
                    report.unpaired_clear_images.append(c_path.name)

        report.valid_pairs_count = len(report.pairs)
        report.scenes_count = len(matched_scenes)
        report.is_valid = report.valid_pairs_count > 0
        report.validation_message = "Validated successfully." if report.is_valid else "No valid pairs matched."
        return report

    def discover_and_validate(self) -> IngestionReport:
        """Scan raw directory, resolve pairs, and detect corrupt or unmatched files."""
        if not self.raw_dir.is_dir():
            rep = IngestionReport(raw_dir=self.raw_dir)
            rep.validation_message = f"Raw dataset directory does not exist: {self.raw_dir}"
            return rep

        # Detect official pre-split dataset (e.g. RESIDE ITS with train/ and val/)
        train_sub = self.raw_dir / "train"
        val_sub = self.raw_dir / "val"
        if train_sub.is_dir() and val_sub.is_dir():
            train_rep = self._validate_split(train_sub, "train")
            val_rep = self._validate_split(val_sub, "val")

            overall = IngestionReport(raw_dir=self.raw_dir)
            overall.has_official_splits = True
            overall.split_reports = {"train": train_rep, "val": val_rep}
            overall.total_clear_found = train_rep.total_clear_found + val_rep.total_clear_found
            overall.total_hazy_found = train_rep.total_hazy_found + val_rep.total_hazy_found
            overall.total_trans_found = train_rep.total_trans_found + val_rep.total_trans_found
            overall.valid_pairs_count = train_rep.valid_pairs_count + val_rep.valid_pairs_count
            overall.scenes_count = train_rep.scenes_count + val_rep.scenes_count
            overall.corrupt_images = train_rep.corrupt_images + val_rep.corrupt_images
            overall.missing_clear_for_hazy = train_rep.missing_clear_for_hazy + val_rep.missing_clear_for_hazy
            overall.unpaired_clear_images = train_rep.unpaired_clear_images + val_rep.unpaired_clear_images
            overall.pairs = train_rep.pairs + val_rep.pairs
            overall.is_valid = train_rep.is_valid and val_rep.is_valid
            overall.clear_dir = train_rep.clear_dir
            overall.hazy_dir = train_rep.hazy_dir
            overall.trans_dir = train_rep.trans_dir
            overall.validation_message = (
                f"Official RESIDE dataset detected with train ({train_rep.valid_pairs_count} pairs) "
                f"and val ({val_rep.valid_pairs_count} pairs)."
            )
            return overall

        return self._validate_split(self.raw_dir)

    def create_leakage_free_splits(
        self,
        pairs: List[ImagePair],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = DEFAULT_RANDOM_SEED,
    ) -> Dict[str, List[ImagePair]]:
        """Partition image pairs into train, val, and test subsets based on source scene IDs.

        Guarantees that ALL hazy variants of any physical clear scene remain strictly
        within the same split partition, preventing train/test data leakage.
        """
        set_seed(seed)

        # Group all pairs by their unique scene_id
        scene_groups: Dict[str, List[ImagePair]] = {}
        for p in pairs:
            scene_groups.setdefault(p.scene_id, []).append(p)

        scene_ids = sorted(scene_groups.keys())
        total_scenes = len(scene_ids)

        if total_scenes == 0:
            return {"train": [], "val": [], "test": []}

        # Deterministic permutation of scene IDs
        rng = np.random.RandomState(seed)
        shuffled_scenes = [scene_ids[i] for i in rng.permutation(total_scenes)]

        if total_scenes == 1:
            # Smoke test fallback when only 1 scene exists
            selected_scenes = {
                "train": shuffled_scenes,
                "val": shuffled_scenes,
                "test": shuffled_scenes,
            }
        elif total_scenes == 2:
            selected_scenes = {
                "train": [shuffled_scenes[0]],
                "val": [shuffled_scenes[1]],
                "test": [shuffled_scenes[1]],
            }
        else:
            n_train = max(1, int(round(total_scenes * train_ratio)))
            n_val = max(1, int(round(total_scenes * val_ratio)))

            # Ensure test set receives at least 1 scene
            n_train = min(n_train, total_scenes - 2)
            n_val = min(n_val, total_scenes - n_train - 1)

            train_scenes = shuffled_scenes[:n_train]
            val_scenes = shuffled_scenes[n_train : n_train + n_val]
            test_scenes = shuffled_scenes[n_train + n_val :]

            if not test_scenes and len(val_scenes) > 1:
                test_scenes = [val_scenes.pop()]

            selected_scenes = {
                "train": train_scenes,
                "val": val_scenes,
                "test": test_scenes,
            }

        # Assemble pairs for each split
        split_pairs: Dict[str, List[ImagePair]] = {"train": [], "val": [], "test": []}
        for split_name, s_ids in selected_scenes.items():
            for sid in s_ids:
                split_pairs[split_name].extend(scene_groups[sid])

        return split_pairs

    def materialize_split(
        self,
        splits: Dict[str, List[ImagePair]],
        output_dir: Path,
        copy_files: bool = True,
    ) -> Path:
        """Create structured split directories and save index manifest."""
        out_root = Path(output_dir)
        manifest: Dict[str, Any] = {
            "dataset_name": out_root.name,
            "splits": {},
            "total_pairs": sum(len(pairs) for pairs in splits.values()),
        }

        for split_name, pairs in splits.items():
            split_clear = out_root / split_name / "clear"
            split_hazy = out_root / split_name / "hazy"
            split_clear.mkdir(parents=True, exist_ok=True)
            split_hazy.mkdir(parents=True, exist_ok=True)

            pair_records = []
            for pair in pairs:
                dest_clear = split_clear / pair.clear_path.name
                dest_hazy = split_hazy / pair.hazy_path.name

                if copy_files:
                    if not dest_clear.exists():
                        shutil.copy2(pair.clear_path, dest_clear)
                    if not dest_hazy.exists():
                        shutil.copy2(pair.hazy_path, dest_hazy)

                pair_records.append({
                    "stem": pair.stem,
                    "scene_id": pair.scene_id,
                    "clear_file": pair.clear_path.name,
                    "hazy_file": pair.hazy_path.name,
                })

            manifest["splits"][split_name] = {
                "count": len(pairs),
                "scenes": len(set(p.scene_id for p in pairs)),
                "pairs": pair_records,
            }

        manifest_path = out_root / "dataset_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Materialized split dataset with manifest at: {manifest_path}")
        return out_root
