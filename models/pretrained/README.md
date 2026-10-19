# External Pretrained Dehazing Models

This directory stores external, verified pretrained model checkpoints evaluated in Stage 2C.3.
Existing project checkpoints remain preserved in `models/checkpoints/` and are never overwritten.

---

## 1. DehazeFormer-MCT (`dehazeformer_mct.pth`)

- **Model Architecture**: DehazeFormer MCT (Multi-Scale Context Transformer)
- **Paper**: *Vision Transformers for Single Image Dehazing* (IEEE Transactions on Image Processing, 2023)
- **Authors**: Yuda Song, Zhuqing He, Hui Qian, Xin Du
- **Official Repository**: [https://github.com/IDKiro/DehazeFormer](https://github.com/IDKiro/DehazeFormer)
- **Official Demo Space**: [https://huggingface.co/spaces/IDKiro/DehazeFormer_Demo](https://huggingface.co/spaces/IDKiro/DehazeFormer_Demo)
- **Direct Checkpoint Source**: `https://huggingface.co/spaces/IDKiro/DehazeFormer_Demo/resolve/main/saved_models/dehazeformer.pth`
- **File Name**: `dehazeformer_mct.pth`
- **File Size**: 5,927,945 bytes (5.65 MB)
- **SHA-256 Checksum**: `225ce2d00d1cd8f0a7aabf5dc09f020ee34636cc942aa8b1a87496506224e304`
- **License**: MIT License (Copyright 2022 IDKiro)
- **Training Data / Domain**: Mixed Indoor + Outdoor dataset (RESIDE-based)
- **Key Characteristics**:
  - Employs a multi-scale context bilateral grid transformation.
  - Core transformer network operates on $256 \times 256$ downsampled context, generating 3D bilateral grid curves.
  - Applies bilateral slice mapping (`F.grid_sample`) to full-resolution images.
  - Parameter count: ~1.43 million parameters.
  - Inference input range: `[-1.0, 1.0]`.
  - Inference output range: `[-1.0, 1.0]`, mapped back to `[0.0, 1.0]` via `0.5 * output + 0.5`.
  - CPU-safe: Pure PyTorch convolutions, attention, and grid sampling. No custom CUDA C++ extensions.

---

## 2. FFA-Net Evaluation Record (Not Integrated)

- **Official Repository**: [https://github.com/zhilin007/FFA-Net](https://github.com/zhilin007/FFA-Net)
- **Status**: Researched and assessed in Stage 2C.3; not integrated into default pipeline.
- **Reasons**:
  - Checkpoints are strictly split into domain-specific models (`its_train_ffa.pk` indoor-only and `ots_train_ffa.pk` outdoor-only) hosted on Google Drive / Baidu Pan, conflicting with our single unified model objective.
  - High parameter count (~4.5M parameters across 57 attention blocks) results in significant CPU latency (>1.5–2.5 seconds per frame) on CPU-only laptops.

---

## 3. AirNet Evaluation Record (Not Integrated)

- **Official Repository**: [https://github.com/unacanal/AirNet](https://github.com/unacanal/AirNet)
- **Status**: Researched and assessed in Stage 2C.3; rejected due to hard compatibility blocker.
- **Reasons**:
  - Hard dependency on `mmcv-full == 1.3.11` and PyTorch 1.7.0.
  - `mmcv-full` requires C++/CUDA compilation toolchains that fail to install on modern Windows Python 3.14 / CPU environments.
  - Large all-in-one restoration model (~50M parameters) unsuitable for lightweight CPU video processing.
