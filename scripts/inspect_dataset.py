"""Dataset Inspection and Health Check Utility.

Scans dataset directories, validates paired structure, detects missing pairs,
checks for unreadable or corrupt image files, and computes resolution statistics.
"""

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple

# Ensure project root is available in module path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from app.config import DATA_DIR, SUPPORTED_IMAGE_EXTENSIONS
from src.utils.logger import get_logger

logger = get_logger("inspect_dataset")


def inspect_directory(dataset_dir: Path) -> Dict[str, any]:
    """Inspect dataset directory containing 'hazy' and 'clear' subdirectories or flat images.

    Args:
        dataset_dir: Path to directory to inspect.

    Returns:
        dict: Inspection summary metrics.
    """
    summary = {
        "dataset_path": str(dataset_dir),
        "is_paired_structure": False,
        "total_images": 0,
        "valid_pairs": 0,
        "missing_clear": [],
        "missing_hazy": [],
        "corrupt_images": [],
        "resolutions": [],
        "channels": set(),
        "min_dim": (0, 0),
        "max_dim": (0, 0),
        "avg_dim": (0, 0),
    }

    if not dataset_dir.is_dir():
        print(f"[ERROR] Directory does not exist: {dataset_dir}")
        return summary

    hazy_dir = dataset_dir / "hazy"
    clear_dir = dataset_dir / "clear"

    if hazy_dir.is_dir() and clear_dir.is_dir():
        summary["is_paired_structure"] = True

        hazy_files = {
            p.stem: p for p in hazy_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        }
        clear_files = {
            p.stem: p for p in clear_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        }

        all_stems = sorted(set(hazy_files.keys()) | set(clear_files.keys()))
        summary["total_images"] = len(hazy_files) + len(clear_files)

        widths = []
        heights = []

        for stem in all_stems:
            in_hazy = stem in hazy_files
            in_clear = stem in clear_files

            if in_hazy and not in_clear:
                summary["missing_clear"].append(hazy_files[stem].name)
            elif in_clear and not in_hazy:
                summary["missing_hazy"].append(clear_files[stem].name)
            else:
                # Pair exists: verify readability and dimensions
                h_path = hazy_files[stem]
                c_path = clear_files[stem]

                h_img = cv2.imread(str(h_path))
                c_img = cv2.imread(str(c_path))

                if h_img is None:
                    summary["corrupt_images"].append(str(h_path))
                if c_img is None:
                    summary["corrupt_images"].append(str(c_path))

                if h_img is not None and c_img is not None:
                    summary["valid_pairs"] += 1
                    h_h, h_w = h_img.shape[:2]
                    c_h, c_w = c_img.shape[:2]
                    widths.extend([h_w, c_w])
                    heights.extend([h_h, c_h])
                    summary["channels"].add(h_img.shape[2] if h_img.ndim == 3 else 1)
                    summary["channels"].add(c_img.shape[2] if c_img.ndim == 3 else 1)
                    summary["resolutions"].append((h_w, h_h))

        if widths and heights:
            summary["min_dim"] = (min(widths), min(heights))
            summary["max_dim"] = (max(widths), max(heights))
            summary["avg_dim"] = (int(np.mean(widths)), int(np.mean(heights)))

    else:
        # Flat directory inspection
        files = [
            p for p in dataset_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        summary["total_images"] = len(files)
        widths = []
        heights = []

        for p in files:
            img = cv2.imread(str(p))
            if img is None:
                summary["corrupt_images"].append(str(p))
            else:
                h, w = img.shape[:2]
                widths.append(w)
                heights.append(h)
                summary["channels"].add(img.shape[2] if img.ndim == 3 else 1)
                summary["resolutions"].append((w, h))

        if widths and heights:
            summary["min_dim"] = (min(widths), min(heights))
            summary["max_dim"] = (max(widths), max(heights))
            summary["avg_dim"] = (int(np.mean(widths)), int(np.mean(heights)))

    return summary


def print_inspection_report(summary: Dict[str, any]) -> None:
    """Print readable CLI diagnostics from inspection summary."""
    print("=" * 60)
    print("DATASET INSPECTION REPORT")
    print("=" * 60)
    print(f"Dataset Path:         {summary['dataset_path']}")
    print(f"Paired Subfolders:    {'YES (clear/ & hazy/)' if summary['is_paired_structure'] else 'NO (flat directory)'}")
    print(f"Total Images Found:   {summary['total_images']}")

    if summary["is_paired_structure"]:
        print(f"Valid Matching Pairs: {summary['valid_pairs']}")
        print(f"Missing Clear Pairs:  {len(summary['missing_clear'])}")
        if summary["missing_clear"]:
            print(f"    Examples: {summary['missing_clear'][:5]}")
        print(f"Missing Hazy Pairs:   {len(summary['missing_hazy'])}")
        if summary["missing_hazy"]:
            print(f"    Examples: {summary['missing_hazy'][:5]}")

    print(f"Corrupt/Unreadable:   {len(summary['corrupt_images'])}")
    if summary["corrupt_images"]:
        print(f"    Corrupt files: {summary['corrupt_images'][:5]}")

    if summary["resolutions"]:
        channels_str = ", ".join(str(c) for c in sorted(summary["channels"]))
        print()
        print(f"Channel Counts:       {channels_str} (3 = RGB)")
        print(f"Min Dimensions (W,H): {summary['min_dim'][0]} x {summary['min_dim'][1]}")
        print(f"Max Dimensions (W,H): {summary['max_dim'][0]} x {summary['max_dim'][1]}")
        print(f"Avg Dimensions (W,H): {summary['avg_dim'][0]} x {summary['avg_dim'][1]}")

    print("=" * 60)
    is_usable = (
        len(summary["corrupt_images"]) == 0
        and summary["total_images"] > 0
        and (not summary["is_paired_structure"] or summary["valid_pairs"] > 0)
    )
    status_str = "USABLE" if is_usable else "NEEDS ATTENTION"
    print(f"Dataset Status: {status_str}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect dataset health and pairing status.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(DATA_DIR / "samples"),
        help="Path to dataset directory (e.g. data/processed/<dataset> or data/samples).",
    )
    args = parser.parse_args()

    summary = inspect_directory(Path(args.dataset_dir))
    print_inspection_report(summary)


if __name__ == "__main__":
    main()
