# Models Directory & Checkpoint Guide

This directory stores deep learning model architectures, pretrained weights, and training checkpoints for atmospheric visibility enhancement and haze classification.

> [!IMPORTANT]
> **Model Checkpoints Notice**: Binary `.pth`, `.pt`, `.onnx`, and `.ckpt` files are excluded from Git via `.gitignore`. The repository preserves the directory layout via `.gitkeep` and provides reproducible download and training utilities.

---

## 1. Directory Structure

```text
models/
├── pretrained/               # External pretrained model weights
│   ├── .gitkeep              # Directory anchor
│   ├── README.md             # Detailed documentation of external candidates
│   └── dehazeformer_mct.pth  # Downloaded pretrained weights (ignored by Git)
│
└── checkpoints/              # Checkpoints generated from local training runs
    ├── .gitkeep              # Directory anchor
    ├── README.md             # Checkpoint documentation
    ├── medium_best_model.pth # Local training checkpoint (ignored by Git)
    └── ...
```

---

## 2. Obtaining the Recommended Pretrained Model

Stage 2C.3 established that **DehazeFormer-MCT** delivers state-of-the-art dehazing performance across both indoor (30.39 dB) and outdoor (34.44 dB) scenes, with practical CPU inference (~1.04 s).

To download and verify the official MIT-licensed checkpoint (5.65 MB), run:

```powershell
python scripts/download_pretrained.py
```

This utility:
1. Downloads `dehazeformer.pth` from the official [HuggingFace Space](https://huggingface.co/spaces/IDKiro/DehazeFormer_Demo/resolve/main/saved_models/dehazeformer.pth).
2. Verifies the expected SHA-256 checksum (`225ce2d00d1cd8f0a7aabf5dc09f020ee34636cc942aa8b1a87496506224e304`).
3. Saves it directly to `models/pretrained/dehazeformer_mct.pth`.

---

## 3. Training Local Lightweight U-Net Checkpoints

To train your own Lightweight U-Net model from scratch on CPU:

```powershell
# Ingest and partition RESIDE ITS into clean zero-leakage splits
python scripts/validate_dataset.py --dataset-dir data/raw/RESIDE/ITS --materialize-to data/processed/reside

# Train the model
python scripts/train.py --train-dir data/processed/reside/train --val-dir data/processed/reside/val --epochs 10 --batch-size 2 --patch-size 128
```

Checkpoints (`best_model.pth`, `latest_model.pth`) will be saved automatically to `models/checkpoints/` with validation metrics embedded in the checkpoint dictionary.
