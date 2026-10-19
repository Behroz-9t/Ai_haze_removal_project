# Stage 1: Dataset Engineering & Classical Dehazing Baseline Guide

This document explains the workflow, mathematics, and operational usage of Stage 1 components in beginner-friendly language.

---

## 1. Overview of Stage 1 Workflow

In Stage 1, we establish two fundamental pillars before introducing deep learning:
1. **Paired Dataset Infrastructure**: Organizing, loading, and synthetically degrading clean images so we have ground-truth references for quantitative validation.
2. **Classical Physical Baseline (DCP)**: Implementing the Dark Channel Prior algorithm by He et al. to benchmark physical dehazing performance on CPU.

```text
Clear Image J(x)
       ↓
Synthetic Degradation: I(x) = J(x)*t(x) + A*(1 - t(x))
       ↓
Hazy Image I(x)  +  Clear Image J(x)
       ↓
Paired Dataset Loader (PyTorch Dataset)
       ↓
DCP Dehazing Baseline:
   1. Dark Channel -> 2. Airlight A -> 3. Coarse Transmission t~(x)
   4. Guided Refinement t(x) -> 5. Recover Radiance J(x)
       ↓
Dehazed Output Image
       ↓
Quantitative Evaluation: PSNR & SSIM  +  Side-by-Side Visualization Strip
```

---

## 2. Dataset Structure and Pairing

To benchmark image restoration models, we need paired datasets consisting of:
* A **hazy** input image $I(x)$
* A matching ground-truth **clear** reference image $J(x)$

### Directory Layout
Datasets inside `data/processed/<dataset_name>/` or `data/samples/` follow this structure:

```text
data/processed/<dataset_name>/
├── clear/
│   ├── scene_01.png
│   └── scene_02.png
└── hazy/
    ├── scene_01.png
    └── scene_02.png
```

The loader [`PairedHazeDataset`](file:///c:/Users/BEHROZ%20MUSHARRAF/Ai_haze_removal_project/src/preprocessing/dataset.py) discovers pairs matching by **file stem** (`scene_01`). It verifies that every hazy image has an exact clear reference, converts colors to RGB, resizes if requested, and produces normalized PyTorch tensors in `[0.0, 1.0]`.

---

## 3. Atmospheric Scattering Model & Synthetic Haze

Real-world atmospheric degradation is modeled using Koschmieder's physical scattering equation:

$$I(x) = J(x) \cdot t(x) + A \cdot (1 - t(x))$$

Where:
* $J(x)$ is the true, unscattered scene radiance (the clear image).
* $t(x) = e^{-\beta \cdot d(x)}$ is the **transmission map** representing the fraction of light reaching the camera without scattering. $\beta$ is the scattering coefficient (haze density), and $d(x)$ is scene depth.
* $A$ is the **atmospheric light** (the ambient airlight scattered by suspended particles, typically bright white/gray).
* $I(x)$ is the observed degraded image captured by the camera sensor.

### How to Generate Synthetic Pairs:
```powershell
python scripts/generate_synthetic_haze.py --source data/samples/sample_clear.png --output-dir data/samples/synthetic_paired --beta 1.2 --light 0.9
```

---

## 4. Dark Channel Prior (DCP) Baseline

The Dark Channel Prior is based on an empirical observation made by Kaiming He et al.: in most non-sky outdoor clear patches, at least one color channel has very low intensity pixels close to zero.

### The 5 Stages of DCP:
1. **Dark Channel Calculation**:
   $$J^{\text{dark}}(x) = \min_{y \in \Omega(x)} \left( \min_{c \in \{R,G,B\}} I^c(y) \right)$$
   Computed via channel-wise minimum followed by morphological erosion with a square patch ($\Omega = 15 \times 15$).
2. **Atmospheric Light ($A$) Estimation**:
   Select the top 0.1% brightest pixels in $J^{\text{dark}}(x)$. Among those candidate locations in the input image $I$, choose the pixel with highest intensity.
3. **Coarse Transmission Map ($\tilde{t}$)**:
   $$\tilde{t}(x) = 1 - \omega \cdot \min_{y \in \Omega(x)} \left( \min_c \frac{I^c(y)}{A^c} \right)$$
   $\omega = 0.95$ leaves a small amount of haze so distant objects look naturally far away.
4. **Transmission Refinement (Guided Filter)**:
   Patch-based transmission creates blocky halo artifacts around depth discontinuities. A **Guided Filter** uses the grayscale input image as a structural guide to smooth flat areas while preserving sharp object edges.
5. **Scene Radiance Recovery**:
   $$J(x) = \frac{I(x) - A}{\max(t(x), t_0)} + A$$
   $t_0 = 0.1$ prevents dividing by near-zero transmission in deep haze regions.

---

## 5. Quality Metrics: PSNR and SSIM

When a ground-truth clear image is available, we evaluate restoration quality using two standard metrics:

1. **PSNR (Peak Signal-to-Noise Ratio)**:
   $$\text{PSNR} = 10 \log_{10}\left(\frac{\text{MAX}^2}{\text{MSE}}\right)$$
   Measures absolute pixel-level fidelity in decibels (dB). Higher is better. Identical images yield $\infty$ dB.
2. **SSIM (Structural Similarity Index Measure)**:
   Measures structural preservation, contrast, and luminance alignment perceived by the human visual system (scale: $-1.0$ to $1.0$, where $1.0$ is identical).

---

## 6. Running the Baseline

### Inspect Dataset Health:
```powershell
python scripts/inspect_dataset.py --dataset-dir data/samples/synthetic_paired
```

### Run DCP Experiment on CPU:
```powershell
python scripts/run_baseline.py --dataset-dir data/samples/synthetic_paired --output-dir outputs/images/baseline --metrics-csv outputs/metrics/baseline_results.csv
```

### Generated Outputs:
* **Images**: `outputs/images/baseline/<stem>_clear.png`, `<stem>_hazy.png`, `<stem>_dcp.png`
* **Visual Strips**: `outputs/images/baseline/<stem>_comparison.png` (`Clear | Hazy | DCP Dehazed`)
* **Quantitative CSV**: `outputs/metrics/baseline_results.csv`

---

## 7. Current Limitations & CPU Performance

* **CPU Execution**: On this machine (Intel Core Ultra 5 125U), a $1024 \times 1024$ image processes in $\approx 0.20\text{ seconds}$ on CPU without GPU acceleration.
* **Classical Bottlenecks**: DCP assumes atmospheric light is globally uniform and struggles on dense sky regions or white surfaces (e.g. snow, white walls), causing slight color cast or haloing. This establishes our benchmark motivation for developing AI/neural dehazing architectures in upcoming stages.
