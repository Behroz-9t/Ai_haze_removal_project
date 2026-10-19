"""Reproducible Train / Validation / Test Dataset Splitter.

Splits a paired dataset (hazy/ and clear/ subdirectories) into deterministic
train, validation, and test subsets with a 70% / 15% / 15% ratio (or custom).
"""

import argparse
from pathlib import Path
import shutil
import sys
from typing import Dict, List, Tuple

# Ensure project root is available in module path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import DEFAULT_RANDOM_SEED, PROCESSED_DATA_DIR, SUPPORTED_IMAGE_EXTENSIONS
from src.preprocessing.dataset import PairedHazeDataset
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger("prepare_dataset")


def split_dataset(
    source_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = DEFAULT_RANDOM_SEED,
    copy_files: bool = True,
) -> Dict[str, int]:
    """Deterministically partition paired dataset into train, val, and test subsets.

    Args:
        source_dir: Directory containing 'clear' and 'hazy' subdirectories.
        output_dir: Target base directory for 'train', 'val', 'test' folders.
        train_ratio: Fraction for training (default 0.70).
        val_ratio: Fraction for validation (default 0.15).
        test_ratio: Fraction for testing (default 0.15).
        seed: Deterministic random seed.
        copy_files: If True, copies image files; otherwise creates directory structure.

    Returns:
        dict: Mapping subset names to counts of paired images.
    """
    set_seed(seed)

    if not source_dir.is_dir():
        raise ValueError(f"Source directory does not exist: {source_dir}")

    # Discover pairs using PairedHazeDataset
    dataset = PairedHazeDataset(root_dir=source_dir, strict_pairs=False)
    pairs = dataset.pairs

    total_pairs = len(pairs)
    if total_pairs == 0:
        raise ValueError(f"No valid image pairs found in '{source_dir}'")

    # Deterministic shuffle
    import numpy as np
    rng = np.random.RandomState(seed)
    shuffled_indices = rng.permutation(total_pairs)
    shuffled_pairs = [pairs[i] for i in shuffled_indices]

    # Calculate partition indices
    # For small datasets (< 3 pairs), ensure at least 1 training sample
    if total_pairs == 1:
        # Special case for smoke testing with single pair: use for train, val, and test
        split_dict = {
            "train": shuffled_pairs,
            "val": shuffled_pairs,
            "test": shuffled_pairs,
        }
    elif total_pairs == 2:
        split_dict = {
            "train": [shuffled_pairs[0]],
            "val": [shuffled_pairs[1]],
            "test": [shuffled_pairs[1]],
        }
    else:
        n_train = max(1, int(round(total_pairs * train_ratio)))
        n_val = max(1, int(round(total_pairs * val_ratio)))
        # Remaining goes to test
        n_train = min(n_train, total_pairs - 2)
        n_val = min(n_val, total_pairs - n_train - 1)

        train_pairs = shuffled_pairs[:n_train]
        val_pairs = shuffled_pairs[n_train : n_train + n_val]
        test_pairs = shuffled_pairs[n_train + n_val :]

        # Guarantee non-empty test set
        if not test_pairs and len(val_pairs) > 1:
            test_pairs = [val_pairs.pop()]

        split_dict = {
            "train": train_pairs,
            "val": val_pairs,
            "test": test_pairs,
        }

    counts = {}
    print("=" * 60)
    print("DATASET PREPARATION & REPRODUCIBLE SPLITTING")
    print("=" * 60)
    print(f"Source Directory:      {source_dir}")
    print(f"Target Splits Root:    {output_dir}")
    print(f"Total Discovered Pairs:{total_pairs}")
    print(f"Random Seed:           {seed}")
    print("-" * 60)

    for subset, sub_pairs in split_dict.items():
        subset_clear_dir = output_dir / subset / "clear"
        subset_hazy_dir = output_dir / subset / "hazy"
        subset_clear_dir.mkdir(parents=True, exist_ok=True)
        subset_hazy_dir.mkdir(parents=True, exist_ok=True)

        for hazy_src, clear_src, stem in sub_pairs:
            if copy_files:
                shutil.copy2(clear_src, subset_clear_dir / clear_src.name)
                shutil.copy2(hazy_src, subset_hazy_dir / hazy_src.name)

        counts[subset] = len(sub_pairs)
        pct = (len(sub_pairs) / total_pairs) * 100.0 if total_pairs > 0 else 0
        print(f"  {subset.upper():<6}: {len(sub_pairs):>4} pairs ({pct:>5.1f}%) -> {output_dir / subset}")

    print("-" * 60)
    print("Dataset splitting completed successfully.")
    print("=" * 60)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministically split paired dataset.")
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Path to directory containing clear/ and hazy/ subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Target splits directory (defaults to data/processed/splits/<dataset_name>).",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Training split fraction (default: 0.70).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split fraction (default: 0.15).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split fraction (default: 0.15).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Deterministic seed (default: {DEFAULT_RANDOM_SEED}).",
    )

    args = parser.parse_args()
    source_p = Path(args.source_dir)

    if args.output_dir is not None:
        target_p = Path(args.output_dir)
    else:
        target_p = PROCESSED_DATA_DIR / "splits" / source_p.name

    split_dataset(
        source_dir=source_p,
        output_dir=target_p,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
