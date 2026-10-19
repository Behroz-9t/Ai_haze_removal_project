# MASTER PROJECT SUMMARY & AI ONBOARDING PROMPT

```markdown
# AI Atmospheric Visibility Enhancement & Haze Detection System
## Master Project Status & Handover Briefing (Stages 0 through 2B.4)

### 1. Project Mission & High-Level Statement of Work (SoW)
You are assisting in the engineering and research of an academic computer vision project titled:
"AI-Based Atmospheric Visibility Enhancement and Haze Detection System"

The core objective is to ingest RGB images and video streams (MP4, AVI, MOV, MKV, webcam) degraded by atmospheric scattering (fog, haze, mist, smog) and perform:
1. Atmospheric visibility enhancement / single-image & video dehazing using both classical and deep learning approaches.
2. Haze detection, density classification, and optical transmission estimation.
3. Temporal consistency across consecutive video frames to suppress flickering.
4. Comprehensive, academically rigorous benchmark evaluation (PSNR, SSIM, Latency, Win Rates) adhering strictly to standard dataset rules.

---

### 2. Execution Environment & Hardware Constraints
- **Operating System**: Windows 11
- **CPU**: Intel Core Ultra 5 125U (14 cores / 18 threads, 32 GB RAM)
- **GPU**: Intel Graphics (Integrated) — **NO NVIDIA CUDA GPU available**
- **PyTorch**: 2.14.1+cpu (all neural models and tensors execute on CPU)
- **Python**: 3.14.5 | **OpenCV**: 5.0.0 | **NumPy**: 2.5.3 | **Pillow**: 12.3.0
- **Crucial Rule**: Because inference and training run on CPU, all models must remain lightweight, memory-efficient, and computationally practical. PyTorch CPU thread count is typically clamped (`torch.set_num_threads(4)`).

---

### 3. Model Catalog & Current Performance

| Model / Checkpoint | Architecture / Type | Parameter Count | Training Data & Regime | Current Role | Key Benchmark Performance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Classical DCP Baseline** | Dark Channel Prior + Guided Filter (Box Filter $O(1)$) | **0 params** (algorithmic) | None (unsupervised physics-based prior) | Baseline reference across all benchmarks | - SOTS-Indoor: **19.66 dB PSNR / 0.8646 SSIM / 46.6 ms**<br>- Strengths: High edge sharpness, fast CPU execution<br>- Weaknesses: Severe dark-clipping, halos, color casts on white objects |
| **Stage 2A Toy Model** (`best_model.pth`) | Lightweight U-Net (4 encoder/decoder levels) | **1,084,731** (~1.08M) | 20 synthetic smoke-test pairs, 10 epochs | Deprecated smoke-test validation | Poor (PSNR ~8.47 dB on held-out tests); served purely to verify forward/backward pass functionality |
| **Stage 2B.1 Pilot Model** (`pilot_best_model.pth`) | Lightweight U-Net (identical 1.08M architecture) | **1,084,731** (~1.08M) | 100 RESIDE ITS pairs (100 distinct scenes), 15 epochs, batch size 4 | Proof of learning pilot | - RESIDE Val (20 pairs): **20.30 dB PSNR / 0.8142 SSIM** (vs Hazy 13.91 dB, +6.39 dB gain)<br>- Proved the architecture successfully learns real transmission inversion |
| **Stage 2B.2 / 2B.4 Production Model** (`medium_best_model.pth`) | Lightweight U-Net (identical 1.08M architecture) | **1,084,731** (~1.08M) | 2,500 RESIDE ITS pairs (2,500 distinct scenes), 20 epochs, batch size 8, Cosine Annealing, seed 42 | **Active Primary Benchmark Model** | - **RESIDE ITS Val (250 pairs, 256x256)**: **21.68 dB PSNR / 0.8242 SSIM / 46.2 ms** (+9.09 dB over hazy, +3.93 dB over DCP)<br>- **Official SOTS-Indoor (500 pairs, 620x460)**: **19.67 dB PSNR / 0.8201 SSIM / 273.6 ms** (100% win rate vs hazy, dead-even with DCP)<br>- **Strengths**: Zero color distortion, natural lighting, no dark halos<br>- **Weaknesses**: Slightly smoothed textures (MSE loss artifact) |

---

### 4. Chronological Progress Summary (Stage 0 to Stage 2B.4)

#### Stage 0 — Project Foundation & Engineering Scaffold
- Built clean modular repository layout: `src/`, `tests/`, `models/`, `data/`, `app/`, `outputs/`, `scripts/`.
- Implemented core utilities: image conversions (RGB, BGR, float32, uint8, torch.Tensor), deterministic seeding (`src/utils/seed.py`), structured logging, and configuration management.
- Initialized test harness.

#### Stage 1 — Classical Dehazing Baseline & Evaluation Framework
- Implemented **Dark Channel Prior (DCP)** with **Guided Filter refinement** (`src/dehazing/dcp.py`, `src/dehazing/guided_filter.py`).
  - Employs a pure-NumPy + `cv2.boxFilter` fallback that runs in $O(1)$ time per pixel without requiring `opencv-contrib`.
  - Tuned defaults: patch size 15, atmospheric transmission $\omega = 0.95$, guided filter radius 40, $\epsilon = 10^{-3}$.
- Implemented standard evaluation metrics: **PSNR** (`src/metrics/psnr.py`) and **SSIM** (`src/metrics/ssim.py`).
- Built `PairedHazeDataset` framework (`src/preprocessing/dataset.py`) supporting synthetic haze generation with Koschmieder's physical scattering model ($I = J \cdot t + A(1 - t)$).
- Verified DCP: improved synthetic hazy test from 12.56 dB to 22.72 dB PSNR.

#### Stage 2A — Deep Learning Scaffold & Lightweight U-Net
- Designed the project's standard neural network: **Lightweight U-Net** (`src/dehazing/model.py`, ~1.08M params).
  - Encoder: 4 stages with DoubleConv blocks (32 $\rightarrow$ 64 $\rightarrow$ 128 $\rightarrow$ 256 channels), BatchNorm, ReLU, MaxPool2d.
  - Bottleneck: 256 channels.
  - Decoder: 4 stages with ConvTranspose2d upsampling, skip connections, DoubleConv blocks.
  - Output head: $1 \times 1$ Conv with Sigmoid activation ($[0, 1]$ RGB range).
- Built training framework (`src/dehazing/trainer.py`): MSE loss, Adam optimizer, checkpoint saving.
- Executed smoke test training (20 toy pairs, 10 epochs), producing `best_model.pth`.
- Built Streamlit web dashboard (`app/main.py`) capable of running both classical DCP and deep learning inference interactively.

#### Stage 2B — Official RESIDE ITS Dataset Integration
- Integrated the official **RESIDE ITS (Indoor Training Set)**:
  - `data/raw/RESIDE/ITS/train/`: 10,000 clear scenes (`ITS clear`), 100,000 hazy images (`ITS haze`), 100,000 transmission maps (`ITS trans`).
  - `data/raw/RESIDE/ITS/val/`: 1,000 clear scenes (`clear`), 10,000 hazy images (`haze`), 10,000 transmission maps (`trans`).
- Refactored `PairedHazeDataset` to dynamically parse the official RESIDE naming format (`<scene_id>_<transmission>_<airlight>.png` $\rightarrow$ `<scene_id>.png`) without renaming or modifying raw dataset files.

#### Stage 2B.1 — Controlled Real-Data Training Pilot
- Trained Lightweight U-Net on 100 RESIDE ITS pairs (100 distinct scenes, $256 \times 256$) for 15 epochs.
- Produced `pilot_best_model.pth`.
- Confirmed valid convergence: validation loss dropped to 0.0097, achieving **20.30 dB PSNR** on validation pairs (outperforming raw hazy 13.91 dB and DCP 17.53 dB).

#### Stage 2B.2 — Medium-Scale RESIDE Training Experiment
- Scaled training to **2,500 distinct scenes** (2,500 training pairs) and **250 validation pairs** using seed 42 (zero scene leakage between train and val).
- Trained for 20 epochs at $256 \times 256$ with batch size 8 and Cosine Annealing learning rate schedule.
- Checkpoint produced: `models/checkpoints/medium_best_model.pth`.
- Results on 250 validation pairs ($256 \times 256$):
  - Hazy: 12.59 dB PSNR / 0.7410 SSIM
  - Classical DCP: 17.75 dB PSNR / 0.8251 SSIM / 14.8 ms
  - AI Medium Model: **21.68 dB PSNR / 0.8242 SSIM / 46.2 ms** (+9.09 dB over hazy, +3.93 dB over DCP, 90.4% win rate).

#### Stage 2B.3 — Independent Held-Out Generalization Evaluation
- Evaluated `medium_best_model.pth` on independent $1024 \times 1024$ outdoor scenes (`road`, `valley`, `village`) with zero fine-tuning.
- Results: AI improved hazy from 11.23 dB to 17.06 dB (+5.83 dB gain).
- Identified domain shift between indoor training data and outdoor depth/sky profiles.

#### Stage 2B.4 — Official RESIDE SOTS-Indoor Benchmark Evaluation
- Rigorously evaluated `medium_best_model.pth` on the official **RESIDE SOTS-Indoor** benchmark (`data/raw/RESIDE/SOTS/indoor/`).
- Verified all filenames and pairing rules before execution: exactly 50 ground-truth scenes (`1400.png` to `1449.png`) and 500 hazy variants (`1400_1.png` to `1449_10.png`), resolution $620 \times 460$.
- Rule enforced: **Zero files renamed or modified**. 100% pairing success via `stem.split('_')[0] + '.png'`.
- Results across all 500 images:
  - Hazy: 11.97 dB PSNR / 0.6962 SSIM
  - Classical DCP: 19.66 dB PSNR / 0.8646 SSIM / 46.6 ms latency
  - AI Medium Model: **19.67 dB PSNR / 0.8201 SSIM / 273.6 ms latency**
  - AI achieved a **500/500 (100.0%) win rate against raw hazy** on PSNR (+7.69 dB mean gain).
  - AI achieved a dead-even tie with classical DCP (19.67 dB vs 19.66 dB, AI won 253/500 pairs on PSNR; clean scene victories split exactly 25 / 50).
  - Generalization audit: 39 strictly unseen scenes achieved 19.66 dB (identical to DCP 19.67 dB), proving zero memorization bias.

---

### 5. Benchmark Performance Comparison Matrix

| Benchmark / Test Set | Image Count & Resolution | Metric | Raw Hazy Input | Classical DCP Baseline | AI Medium Model (`medium_best_model.pth`) | AI Delta vs Hazy | AI Delta vs DCP |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Stage 2B.2 Val Set** | 250 pairs ($256 \times 256$) | Mean PSNR<br>Mean SSIM<br>Latency | 12.59 dB<br>0.7410<br>— | 17.75 dB<br>0.8251<br>14.8 ms | **21.68 dB**<br>0.8242<br>46.2 ms | **+9.09 dB**<br>+0.0832<br>— | **+3.93 dB**<br>-0.0009<br>+31.4 ms |
| **Stage 2B.3 Independent** | 3 pairs ($1024 \times 1024$) | Mean PSNR<br>Mean SSIM | 11.23 dB<br>0.6272 | **18.06 dB**<br>**0.7937** | 17.06 dB<br>0.7441 | **+5.83 dB**<br>+0.1169 | -1.00 dB<br>-0.0496 |
| **Stage 2B.4 SOTS-Indoor** | 500 pairs ($620 \times 460$) | Mean PSNR<br>Median PSNR<br>Mean SSIM<br>Median SSIM<br>Median Latency<br>PSNR Win Rate | 11.97 dB<br>11.84 dB<br>0.6962<br>0.7030<br>—<br>— | 19.66 dB<br>19.88 dB<br>**0.8646**<br>**0.8712**<br>**46.6 ms**<br>476/500 (95.2%) | **19.67 dB**<br>**19.89 dB**<br>0.8201<br>0.8260<br>273.6 ms<br>**500/500 (100.0%)** | **+7.69 dB**<br>+8.05 dB<br>**+0.1239**<br>+0.1230<br>—<br>— | **+0.01 dB**<br>+0.01 dB<br>-0.0445<br>-0.0452<br>+227.0 ms<br>253/500 (50.6%) |

---

### 6. Directory Layout & Key Components
```text
Ai_haze_removal_project/
├── app/
│   ├── config.py                 # Central application settings & defaults
│   └── main.py                   # Streamlit web UI (DCP + AI live inference)
├── src/
│   ├── preprocessing/            # PairedHazeDataset, RESIDE ingestion, split manifest generators
│   ├── detection/                # Haze detection & density assessment modules
│   ├── dehazing/
│   │   ├── model.py              # Lightweight U-Net (~1.08M params)
│   │   ├── trainer.py            # PyTorch CPU training loop & validation
│   │   ├── dcp.py                # Dark Channel Prior classical dehazing
│   │   └── guided_filter.py      # O(1) box-filtered guided filter
│   ├── temporal/                 # Temporal consistency modules for video
│   ├── video/                    # Video frame extractor & pipeline
│   ├── metrics/                  # PSNR & SSIM evaluation implementations
│   └── utils/                    # Seed reproducibility, logging, image conversions
├── data/
│   └── raw/RESIDE/
│       ├── ITS/                  # Official RESIDE Indoor Training Set (train/ & val/)
│       └── SOTS/indoor/          # Official SOTS-Indoor benchmark (gt/ 50 images, hazy/ 500 images)
├── models/checkpoints/
│   ├── best_model.pth            # Stage 2A smoke-test checkpoint
│   ├── pilot_best_model.pth      # Stage 2B.1 100-pair pilot checkpoint
│   └── medium_best_model.pth     # Stage 2B.2 2,500-pair primary model checkpoint
├── outputs/evaluation/sots_indoor/
│   ├── sots_indoor_per_image_metrics.csv
│   ├── sots_indoor_summary.json
│   └── comparisons/              # 12 representative 4-panel comparison strips
└── tests/                        # Full automated test suite (36 tests, 100% pass)
```

---

### 7. Strict Engineering & Research Rules
1. **Zero Renaming Policy**: Never rename, move, delete, or modify original dataset files to force matching. All dataset loaders must parse existing naming patterns natively.
2. **Architecture Preservation**: The Lightweight U-Net (~1.08M params) is the baseline architecture. Do not alter its core layer structure or swap it out unless explicitly directed.
3. **Weight Integrity**: Never retrain or overwrite existing checkpoints (`medium_best_model.pth`, `pilot_best_model.pth`, `best_model.pth`) during evaluation tasks.
4. **Hardware Discipline**: Always maintain CPU compatibility. Profile latency honestly on CPU without assuming CUDA.
5. **Quality Assurance**: Maintain 100% test pass rate (currently 36/36 tests passing via `python -m unittest discover tests`).
6. **Reporting Standard**: At the completion of every task or experiment, generate a comprehensive, structured technical report artifact with quantitative tables, metrics, visual comparisons, and recommendations.

---

### 8. Immediate Next Objectives on the Agenda
1. **Perceptual & Structural Loss Exploration**: Supplement MSE loss with a composite loss ($\mathcal{L}_{L1} + \lambda_{1}\mathcal{L}_{\text{SSIM}} + \lambda_{2}\mathcal{L}_{\text{perceptual}}$) to eliminate high-frequency smoothing and surpass DCP in SSIM.
2. **Outdoor Domain Adaptation**: Evaluate and adapt models on outdoor datasets (RESIDE SOTS-Outdoor / OTS) to overcome depth and atmospheric lighting domain shift.
3. **Real-Time CPU Acceleration**: Explore ONNX Runtime export, FP16/INT8 dynamic quantization, or multi-threaded CPU tiling to reduce the 273.6 ms latency towards interactive frame rates (>15 FPS).
4. **Video Dehazing & Temporal Consistency**: Implement temporal smoothing buffers and video frame processing pipeline (MP4, AVI, MOV, MKV, webcam) preserving inter-frame consistency.
5. **Haze Detection & Optical Depth Estimation**: Activate `src/detection/` modules to classify haze density (light, moderate, dense) and dynamically adjust dehazing strength.
```
