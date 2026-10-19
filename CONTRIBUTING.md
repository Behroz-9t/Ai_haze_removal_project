# Contributing to AI Atmospheric Visibility Enhancement

Thank you for your interest in contributing to the **AI Atmospheric Visibility Enhancement System**! This guide outlines how to set up your isolated development environment, execute tests, run the Streamlit UI, develop new features, and submit pull requests cleanly.

---

## 1. Development Environment Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.10, 3.11, 3.12, 3.13, 3.14)
- Git
- (Optional) System FFmpeg in PATH if testing audio preservation in video dehazing

### Clone & Isolated Environment Setup (Windows PowerShell)

```powershell
# 1. Clone your fork
git clone https://github.com/<YOUR_USERNAME>/Ai_haze_removal_project.git
cd Ai_haze_removal_project

# 2. Create an isolated virtual environment (never commit this folder)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Clone & Setup (Linux / macOS)

```bash
git clone https://github.com/<YOUR_USERNAME>/Ai_haze_removal_project.git
cd Ai_haze_removal_project

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 2. Obtain Pretrained Weights (Optional but Recommended)

To run the leading DehazeFormer-MCT model locally, download the official verified checkpoint (5.65 MB):

```powershell
python scripts/download_pretrained.py
```

*Note: The application and test suite will run even without this checkpoint; Classical DCP and unit tests do not require external weights.*

---

## 3. Running Automated Tests

Before submitting any code changes, ensure all tests pass cleanly:

```powershell
python -m unittest discover tests
```

To run a specific test stage:
```powershell
python -m unittest tests/test_stage2c3.py
```

All 52 tests should pass with 0 failures and 0 errors.

---

## 4. Launching the Interactive Frontend

Run the Streamlit application to visually inspect changes:

```powershell
python -m streamlit run app/ui.py
```

Open your browser at `http://localhost:8501`.

---

## 5. Branching & Pull Request Workflow

We follow the standard GitHub fork-and-branch workflow:

1. **Fork the repository** on GitHub to your personal account.
2. **Create a descriptive feature branch**:
   ```bash
   git checkout -b feature/temporal-consistency-filter
   ```
3. **Make your changes**:
   - Write clean, documented Python code.
   - Follow PEP 8 style guidelines.
   - Preserve existing function signatures where backwards compatibility is needed.
   - Add unit tests under `tests/` for new algorithms or adapters.
4. **Pre-Commit Verification**:
   - Run tests: `python -m unittest discover tests`
   - Check `git status` to verify **no datasets, model binaries (`.pth`), generated images/videos, or `.env` files are staged**.
5. **Commit with descriptive messages**:
   ```bash
   git commit -m "Add exponential moving average temporal smoothing for video frames"
   ```
6. **Push to your fork**:
   ```bash
   git push origin feature/temporal-consistency-filter
   ```
7. **Open a Pull Request (PR)** against the `main` branch:
   - Describe what the PR does and why it is needed.
   - Attach quantitative evaluation metrics or before/after visual examples if applicable.
   - Confirm that all unit tests pass.

---

## 6. Contribution Rules & Safety Checklist

- [ ] **Do NOT commit model weights**: Checkpoint files (`.pth`, `.pt`, `.onnx`, etc.) must remain local.
- [ ] **Do NOT commit datasets**: Raw and processed dataset files (RESIDE, O-HAZE, SOTS, etc.) must never be committed.
- [ ] **Do NOT commit virtual environments**: The `.venv/` directory is strictly ignored.
- [ ] **Do NOT commit generated outputs**: Media files (`.mp4`, `.png`, `.jpg`) generated in `outputs/` are ignored.
- [ ] **Do NOT commit secrets**: Never hardcode API keys, tokens, or credentials.
- [ ] **Preserve existing baselines**: Ensure Classical DCP and Lightweight U-Net remain functional when adding new architectures.

---

## 7. Reporting Bugs & Proposing Enhancements

If you encounter a bug or have a suggestion:
1. Check existing GitHub Issues to see if it has already been reported.
2. Open a new Issue detailing:
   - Your operating system and hardware specifications (e.g. CPU-only vs CUDA).
   - Python and PyTorch versions (`python -c "import torch; print(torch.__version__)"`).
   - Exact commands executed and full traceback error logs.
   - Expected behavior vs actual observed behavior.
