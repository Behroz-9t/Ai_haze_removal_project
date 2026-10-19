# Dataset Directory & Setup Guide

This directory stores datasets and sample media used for training, evaluation, and baseline testing of the AI Atmospheric Visibility Enhancement System.

> [!IMPORTANT]
> **Data Compliance Notice**: Actual dataset images (ITS, OTS, SOTS, etc.) are excluded from Git via `.gitignore`. Users and contributors must independently download the datasets from their official academic sources and comply with their respective terms of use, research licenses, and citation requirements.

---

## 1. Directory Structure

```text
data/
├── raw/                      # Downloaded and extracted raw academic datasets (ignored by Git)
│   └── RESIDE/
│       ├── ITS/              # Indoor Training Set (hazy/ and clear/)
│       ├── OTS/              # Outdoor Training Set (haze/ and clear/)
│       └── SOTS/             # Synthetic Object Threat Set (indoor/ and outdoor/)
│
├── processed/                # Standardized, paired splits and manifests (ignored by Git)
│   ├── reside/               # Materialized train/, val/, test/ splits
│   └── manifests/            # JSON manifests with pair mappings and scene metadata
│
└── samples/                  # Lightweight synthetic test fixtures (< 5 MB, tracked by Git)
    ├── sample_clear.png      # Clean synthetic sample image for test verification
    ├── sample_haze.png       # Hazy synthetic sample image for test verification
    └── synthetic_paired/     # Paired clear/hazy smoke test fixtures
```

---

## 2. Benchmark Datasets Placement Guide

### 2.1 RESIDE ITS (Indoor Training Set)
- **Role**: Indoor training and indoor validation.
- **Official Source**: [RESIDE Dataset Benchmark](https://sites.google.com/view/reside-dehaze-datasets)
- **Expected Location**:
  ```text
  data/raw/RESIDE/ITS/
  ├── clear/                  # Clean ground truth images (e.g. 1400.png, 1401.png)
  └── hazy/                   # Hazy variants (e.g. 1400_1_0.08.png, 1400_2_0.12.png)
  ```
- **Pairing Rule**: Each hazy filename starts with the scene ID prefix followed by variant suffixes (e.g., `1400_1_0.08.png` matches clear scene `1400.png`).

### 2.2 RESIDE OTS (Outdoor Training Set)
- **Role**: Outdoor training and outdoor validation.
- **Official Source**: [RESIDE Dataset Benchmark](https://sites.google.com/view/reside-dehaze-datasets)
- **Handling Nested Folders**: In official OTS downloads, hazy images are often distributed across subfolders `part 1`, `part 2`, `part 3`, and `part 4`:
  ```text
  data/raw/RESIDE/OTS/
  ├── clear/                  # Clean ground truth images (e.g. 0001.jpg, 0002.jpg)
  └── haze/
      ├── part 1/             # Subfolder containing e.g. 0001_0.8_0.04.jpg
      ├── part 2/             # Subfolder containing e.g. 0500_0.8_0.04.jpg
      ├── part 3/             # Subfolder containing e.g. 1000_0.8_0.04.jpg
      └── part 4/             # Subfolder containing e.g. 1500_0.8_0.04.jpg
  ```
  The project ingestion tool (`src/preprocessing/ingestion.py` and `scripts/create_mixed_manifests.py`) automatically discovers images recursively across all `part *` folders, parsing the standard scene ID prefix (e.g. `0001_0.8_0.04.jpg` -> clear reference `0001.jpg`).

### 2.3 RESIDE SOTS (Synthetic Object Threat Set)
- **Role**: Strictly held-out benchmark evaluation. **Never use SOTS images for model training or validation.**
- **Expected Location**:
  ```text
  data/raw/RESIDE/SOTS/
  ├── indoor/
  │   ├── clear/ (or gt/)     # 50 clean ground truth images (1400.png to 1449.png)
  │   └── hazy/               # 500 hazy variants (10 variants per scene)
  └── outdoor/
      ├── clear/ (or gt/)     # 500 clean outdoor images
      └── hazy/               # 500 hazy outdoor images
  ```

---

## 3. Dataset Validation & Leakage Prevention

To ensure no scene overlap exists between training and validation/test splits, run:

```powershell
python scripts/validate_dataset.py --dataset-dir data/raw/RESIDE/ITS --materialize-to data/processed/reside
```

This utility:
1. Recursively discovers all hazy and clear files.
2. Checks for unreadable or corrupted image files.
3. Groups all 10 hazy variants of a scene strictly into the same split (zero data leakage).
4. Produces `data/processed/reside/dataset_manifest.json`.

---

## 4. Running a Training Pilot

To train the Lightweight U-Net on CPU with patch-based cropping:

```powershell
# Small pilot on 500 training pairs (fast verification)
python scripts/train.py --train-dir data/processed/reside/train --val-dir data/processed/reside/val --epochs 5 --batch-size 2 --patch-size 128 --max-train-samples 500

# Mixed indoor/outdoor pilot
python scripts/train_mixed.py --epochs 5 --batch-size 2 --patch-size 128
```

Checkpoints will be saved to `models/checkpoints/`.

---

## 5. Running Held-Out Evaluation

```powershell
# SOTS-Indoor 50-Scene Evaluation
python scripts/evaluate_sots_indoor.py --checkpoint models/checkpoints/medium_best_model.pth

# Comprehensive Pretrained vs U-Net vs DCP Multi-Domain Benchmark
python scripts/benchmark_pretrained_vs_unet.py
```
