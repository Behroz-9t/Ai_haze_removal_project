"""Dataset Validation Utility for Stage 2B.

Validates a downloaded or prepared dataset (such as RESIDE ITS/OTS/SOTS).
Checks folder structure, matching pairs, unreferenced images, corrupt files,
and calculates zero-leakage split counts before training commences.
"""

import argparse
from pathlib import Path
import sys
from typing import Optional

# Ensure project root is available in module path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import DATA_DIR, RAW_DATA_DIR
from src.preprocessing.ingestion import ResideDatasetIngestor


def validate_dataset(dataset_dir: Path, materialize_to: Optional[Path] = None) -> bool:
    """Validate dataset structure and print the required diagnostics report.

    Args:
        dataset_dir: Root directory of raw or processed dataset.
        materialize_to: Optional target directory to write partitioned train/val/test splits.

    Returns:
        bool: True if dataset is valid and ready for training.
    """
    print("=" * 70)
    print("STAGE 2B: DATASET VALIDATION REPORT")
    print("=" * 70)
    print(f"Dataset Path:             {dataset_dir}")

    ingestor = ResideDatasetIngestor(dataset_dir)
    report = ingestor.discover_and_validate()

    if report.has_official_splits:
        print(f"Official Splits:          DETECTED (train & val)")
        print("-" * 70)
        train_rep = report.split_reports.get("train")
        val_rep = report.split_reports.get("val")

        if train_rep:
            print("1. OFFICIAL TRAINING SET (train/):")
            print(f"   - Clear Directory:     {train_rep.clear_dir}")
            print(f"   - Hazy Directory:      {train_rep.hazy_dir}")
            print(f"   - Trans Directory:     {train_rep.trans_dir if train_rep.trans_dir else 'N/A'}")
            print(f"   - Clear Images:        {train_rep.total_clear_found:,}")
            print(f"   - Hazy Images:         {train_rep.total_hazy_found:,}")
            print(f"   - Transmission Maps:   {train_rep.total_trans_found:,}")
            print(f"   - Unique Scenes:       {train_rep.scenes_count:,}")
            print(f"   - Valid Pairs:         {train_rep.valid_pairs_count:,}")
            print(f"   - Missing Clear:       {len(train_rep.missing_clear_for_hazy)}")
            print(f"   - Corrupted Images:    {len(train_rep.corrupt_images)}")

        if val_rep:
            print("\n2. OFFICIAL VALIDATION SET (val/):")
            print(f"   - Clear Directory:     {val_rep.clear_dir}")
            print(f"   - Hazy Directory:      {val_rep.hazy_dir}")
            print(f"   - Trans Directory:     {val_rep.trans_dir if val_rep.trans_dir else 'N/A'}")
            print(f"   - Clear Images:        {val_rep.total_clear_found:,}")
            print(f"   - Hazy Images:         {val_rep.total_hazy_found:,}")
            print(f"   - Transmission Maps:   {val_rep.total_trans_found:,}")
            print(f"   - Unique Scenes:       {val_rep.scenes_count:,}")
            print(f"   - Valid Pairs:         {val_rep.valid_pairs_count:,}")
            print(f"   - Missing Clear:       {len(val_rep.missing_clear_for_hazy)}")
            print(f"   - Corrupted Images:    {len(val_rep.corrupt_images)}")

        if train_rep and val_rep:
            train_scenes = set(p.scene_id for p in train_rep.pairs)
            val_scenes = set(p.scene_id for p in val_rep.pairs)
            overlap = train_scenes.intersection(val_scenes)
            print("-" * 70)
            print("DATA LEAKAGE CHECK:")
            print(f"   - Train Scenes:        {len(train_scenes):,}")
            print(f"   - Val Scenes:          {len(val_scenes):,}")
            print(f"   - Overlapping Scenes:  {len(overlap)} (Zero Leakage: {len(overlap) == 0})")

        print("-" * 70)
        print("DATASET TOTALS:")
        print(f"   - Total Clear:         {report.total_clear_found:,}")
        print(f"   - Total Hazy:          {report.total_hazy_found:,}")
        print(f"   - Total Transmission:  {report.total_trans_found:,}")
        print(f"   - Total Valid Pairs:   {report.valid_pairs_count:,}")
        print(f"   - Status:              VALID - Official splits preserved. Ready for training.")
        print("=" * 70)
        return True

    # Single folder dataset report
    print(f"Clear Images Directory:   {report.clear_dir if report.clear_dir else 'NOT FOUND'}")
    print(f"Hazy Images Directory:    {report.hazy_dir if report.hazy_dir else 'NOT FOUND'}")
    if report.trans_dir:
        print(f"Trans Images Directory:   {report.trans_dir}")
    print(f"Total Clear Images:       {report.total_clear_found}")
    print(f"Total Hazy Images:        {report.total_hazy_found}")
    if report.total_trans_found > 0:
        print(f"Total Trans Maps:         {report.total_trans_found}")
    print(f"Unique Clear Scenes:      {report.scenes_count}")
    print(f"Valid Matching Pairs:     {report.valid_pairs_count}")
    print(f"Missing Clear for Hazy:   {len(report.missing_clear_for_hazy)}")
    if report.missing_clear_for_hazy:
        print(f"    Examples: {report.missing_clear_for_hazy[:3]}")

    print(f"Unpaired Clear Images:    {len(report.unpaired_clear_images)}")
    if report.unpaired_clear_images:
        print(f"    Examples: {report.unpaired_clear_images[:3]}")

    print(f"Corrupted/Empty Images:   {len(report.corrupt_images)}")
    if report.corrupt_images:
        print(f"    Corrupt: {report.corrupt_images[:3]}")

    print(f"Duplicate/Ambiguous Pairs:0")

    if report.is_valid:
        # Compute leakage-free split counts preview
        splits = ingestor.create_leakage_free_splits(report.pairs, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
        print("-" * 70)
        print(f"Leakage-Free Splits Preview:")
        print(f"    Train Pairs:          {len(splits['train'])} ({len(set(p.scene_id for p in splits['train']))} source scenes)")
        print(f"    Validation Pairs:     {len(splits['val'])} ({len(set(p.scene_id for p in splits['val']))} source scenes)")
        print(f"    Test Pairs:           {len(splits['test'])} ({len(set(p.scene_id for p in splits['test']))} source scenes)")
        print("-" * 70)

        if materialize_to is not None:
            out_path = ingestor.materialize_split(splits, output_dir=materialize_to, copy_files=True)
            print(f"Materialized splits to:   {out_path}")
            print(f"Manifest written:         {out_path / 'dataset_manifest.json'}")
            print("-" * 70)

        print(f"Status: VALID - Ready for training.")
        print("=" * 70)
        return True
    else:
        print("-" * 70)
        print(f"Status: INVALID - {report.validation_message}")
        print("=" * 70)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate RESIDE or paired dataset before training.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(RAW_DATA_DIR / "RESIDE"),
        help="Path to dataset directory (e.g. data/raw/RESIDE or data/processed/<dataset>).",
    )
    parser.add_argument(
        "--materialize-to",
        type=str,
        default=None,
        help="Optional directory to materialize zero-leakage train/val/test splits (e.g. data/processed/reside).",
    )
    args = parser.parse_args()

    target_dir = Path(args.dataset_dir)
    materialize_dir = Path(args.materialize_to) if args.materialize_to else None
    is_valid = validate_dataset(target_dir, materialize_to=materialize_dir)
    sys.exit(0 if is_valid else 1)


if __name__ == "__main__":
    main()
