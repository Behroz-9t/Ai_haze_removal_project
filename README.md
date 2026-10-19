# AI Atmospheric Visibility Enhancement System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-ff4b4b.svg)](https://streamlit.io/)
[![CPU-Optimized](https://img.shields.io/badge/Inference-CPU--Optimized-green.svg)](https://github.com)
[![Tests](https://img.shields.io/badge/Tests-52%20Passing-brightgreen.svg)](tests/)

An academic and practical computer vision system designed to detect haze density, restore lost visibility and contrast in hazy/foggy images, and process continuous video streams. 

Optimized specifically for **fast CPU inference** on consumer laptops (e.g. Intel Core Ultra / Core i5/i7) without requiring expensive NVIDIA GPUs or complex CUDA toolchains.

---

## 🌟 Key Features

1. **State-of-the-Art Pretrained DehazeFormer-MCT Integration**:
   - Integrates the Multi-Scale Context Transformer (MCT) variant of DehazeFormer (~1.23M parameters).
   - Global bilateral grid curve slicing on $256 \times 256$ downsampled context achieves resolution-invariant CPU forward passes in **~1.04 seconds**.
   - Outperforms prior baselines with **30.39 dB PSNR on SOTS-Indoor** and **34.44 dB PSNR on OTS-Outdoor**.
2. **Custom Lightweight Residual U-Net**:
   - Encoder-decoder network (~1.08M parameters) with depthwise-separable residual blocks and reflection padding.
   - High-speed fallback (~330 ms on CPU, ~3.0 FPS).
3. **Classical Dark Channel Prior (DCP) Baseline**:
   - Pure-array physical baseline with Guided Filter refinement.
   - Operates completely without learned neural weights for hardware-independent verification.
4. **Atmospheric Haze Density Diagnostics**:
   - Real-time dark channel statistics and contrast metrics classifying images as *Clear / Near-Clear*, *Light Haze*, or *Moderate / Heavy Haze*.
   - Prevents over-processing or color distortion on zero-haze images.
5. **Memory-Safe Offline Video Dehazing**:
   - Frame-by-frame processing pipeline with bounded memory consumption.
   - Automatic audio preservation and muxing via system FFmpeg.
6. **Interactive Streamlit Web Interface (`app/ui.py`)**:
   - Side-by-side comparative views (Input, Dehazed AI, Classical DCP Baseline, and U-Net Comparison).
   - Automatic fallback to Classical DCP baseline when neural weights are absent.
7. **Comprehensive Test Suite**:
   - 52 automated unit tests across all project components (100% passing).

---

## 📊 Benchmark Evaluation Summary

Head-to-head evaluation across 114 test images on an Intel Core Ultra 5 125U CPU (no GPU):

| Benchmark Suite | Image Count | Hazy Input | Classical DCP | Lightweight U-Net | DehazeFormer-MCT | CPU Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SOTS-Indoor** | 50 scenes | 15.43 dB / 0.778 | 19.54 dB / 0.881 | 23.27 dB / 0.881 | **30.39 dB / 0.976** | ~1.04 s |
| **OTS-Outdoor** | 20 scenes | 25.10 dB / 0.933 | 18.10 dB / 0.875 | 26.49 dB / 0.893 | **34.44 dB / 0.982** | ~1.03 s |
| **Synthetic Outdoor** | 4 scenes | 12.61 dB / 0.756 | 22.10 dB / 0.884 | 14.73 dB / 0.796 | **21.71 dB / 0.896** | ~1.03 s |
| **Clear Image Robustness**| 40 images | — | 18.08 dB / 0.896 | 24.88 dB / 0.875 | **31.62 dB / 0.977** | ~1.04 s |

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```powershell
git clone https://github.com/<YOUR_USERNAME>/Ai_haze_removal_project.git
cd Ai_haze_removal_project
```

### 2. Environment Setup

#### Windows (PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Linux / macOS (Bash)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Download the Pretrained Model Checkpoint (Recommended)
Download and verify the official MIT-licensed DehazeFormer-MCT checkpoint (5.65 MB):
```powershell
python scripts/download_pretrained.py
```
*Note: If this step is skipped, the application and tests will still function using the Classical DCP baseline and test fixtures.*

### 4. Launch the Interactive Web Application
```powershell
python -m streamlit run app/ui.py
```
Open `http://localhost:8501` in your browser.

### 5. Run Automated Tests
```powershell
python -m unittest discover tests
```
All 52 unit tests should pass with 0 failures and 0 errors.

---

## 📁 Repository Structure

```text
Ai_haze_removal_project/
├── app/
│   ├── config.py                   # Central path definitions and hardware configuration
│   ├── main.py                     # CLI diagnostic runner
│   └── ui.py                       # Multi-model Streamlit application
│
├── src/
│   ├── dehazing/                   # Dehazing models and classical algorithms
│   │   ├── external/               # External pretrained model architectures & adapters
│   │   │   ├── dehazeformer.py     # Pure PyTorch DehazeFormer-MCT implementation
│   │   │   └── dehazeformer_adapter.py # Clean inference wrapper with stage profiling
│   │   ├── dcp.py                  # Classical Dark Channel Prior (DCP) baseline
│   │   ├── guided_filter.py        # Fast guided image filter with pure NumPy fallback
│   │   ├── losses.py               # L1 reconstruction loss
│   │   └── model.py                # Lightweight Residual U-Net architecture
│   ├── detection/
│   │   └── density.py              # Atmospheric haze density estimation & diagnostics
│   ├── metrics/                    # PSNR and SSIM quantitative metrics
│   ├── preprocessing/              # Zero-leakage dataset loaders and synthetic degradation
│   ├── utils/                      # Robust image I/O, seed, logging, and multi-panel displays
│   └── video/
│       └── processor.py            # Bounded-memory offline video dehazing engine
│
├── data/
│   ├── raw/                        # Downloaded raw academic datasets (ignored by Git)
│   ├── processed/                  # Materialized splits and manifests (ignored by Git)
│   ├── samples/                    # Test fixtures for test execution (< 5 MB)
│   └── README.md                   # Comprehensive dataset setup guide (ITS, OTS, SOTS)
│
├── models/
│   ├── checkpoints/                # Local training checkpoints (ignored by Git)
│   ├── pretrained/                 # Pretrained weights (ignored by Git)
│   └── README.md                   # Model checkpoint guide and download instructions
│
├── outputs/
│   ├── evaluation/                 # Lightweight evaluation reports & documentation
│   ├── images/                     # Output images directory anchor
│   └── videos/                     # Output videos directory anchor
│
├── scripts/
│   ├── download_pretrained.py      # Automated download and SHA-256 verification
│   ├── benchmark_pretrained_vs_unet.py # Multi-domain fair benchmark runner
│   ├── train.py                    # Lightweight U-Net training pipeline
│   ├── train_mixed.py              # Mixed indoor/outdoor pilot training runner
│   ├── evaluate.py                 # Evaluation script on held-out test splits
│   └── validate_dataset.py         # Dataset health and zero-leakage split validator
│
├── tests/                          # 52 unit tests across stages 0 to 2C.3
├── docs/                           # Methodological documentation and experimental logs
├── requirements.txt                # Lean dependency manifest
├── CONTRIBUTING.md                 # Contributor guidelines and workflow
└── README.md
```

---

## 🛠️ CLI Usage Examples

### Single Image Dehazing
```powershell
python scripts/infer.py --input data/samples/sample_haze.png --output outputs/inference/result.png
```

### Dataset Validation & Materialization
```powershell
python scripts/validate_dataset.py --dataset-dir data/raw/RESIDE/ITS --materialize-to data/processed/reside
```

### Training a Model
```powershell
python scripts/train.py --train-dir data/processed/reside/train --val-dir data/processed/reside/val --epochs 10 --batch-size 2 --patch-size 128
```

### Multi-Domain Head-to-Head Benchmark
```powershell
python scripts/benchmark_pretrained_vs_unet.py
```

---

## 📚 References & Acknowledgments

- **DehazeFormer**: Yuda Song, Zhuqing He, Hui Qian, Xin Du. *"Vision Transformers for Single Image Dehazing"*, IEEE Transactions on Image Processing (TIP), 2023. [GitHub](https://github.com/IDKiro/DehazeFormer).
- **RESIDE Dataset**: Boyi Li et al. *"Benchmarking Single-Image Dehazing and Beyond"*, IEEE Transactions on Image Processing (TIP), 2018. [Website](https://sites.google.com/view/reside-dehaze-datasets).
- **Dark Channel Prior**: Kaiming He, Jian Sun, Xiaoou Tang. *"Single Image Haze Removal Using Dark Channel Prior"*, IEEE TPAMI, 2011.
- **Guided Filter**: Kaiming He, Jian Sun, Xiaoou Tang. *"Guided Image Filtering"*, IEEE TPAMI, 2013.

---

## 🤝 Contributing

Contributions, bug reports, and suggestions are welcome! Please review [CONTRIBUTING.md](CONTRIBUTING.md) for details on our workflow, coding standards, and pre-commit checklist.
