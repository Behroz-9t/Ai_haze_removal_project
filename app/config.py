"""Central Configuration Module.

Provides path definitions relative to the project root, supported file extensions,
device configurations, and runtime hyperparameters. Ensures portability across
operating systems without hardcoding absolute paths.
"""

from pathlib import Path
from typing import Set, Tuple
import os


# -----------------------------------------------------------------------------
# Base Directories (Relative to Project Root)
# -----------------------------------------------------------------------------
APP_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = APP_DIR.parent

# Source code & modules
SRC_DIR: Path = PROJECT_ROOT / "src"

# Data directories
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
SAMPLES_DATA_DIR: Path = DATA_DIR / "samples"

# Model artifacts
MODELS_DIR: Path = PROJECT_ROOT / "models"
PRETRAINED_MODELS_DIR: Path = MODELS_DIR / "pretrained"
CHECKPOINTS_DIR: Path = MODELS_DIR / "checkpoints"

# Outputs
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
OUTPUT_IMAGES_DIR: Path = OUTPUTS_DIR / "images"
OUTPUT_VIDEOS_DIR: Path = OUTPUTS_DIR / "videos"
OUTPUT_METRICS_DIR: Path = OUTPUTS_DIR / "metrics"

# Ancillary project folders
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
TESTS_DIR: Path = PROJECT_ROOT / "tests"
SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"
DOCS_DIR: Path = PROJECT_ROOT / "docs"

# -----------------------------------------------------------------------------
# Supported Formats
# -----------------------------------------------------------------------------
SUPPORTED_IMAGE_EXTENSIONS: Set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tiff",
}

SUPPORTED_VIDEO_EXTENSIONS: Set[str] = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
}

# -----------------------------------------------------------------------------
# Default Runtime Settings
# -----------------------------------------------------------------------------
DEFAULT_RANDOM_SEED: int = 42

# Default spatial dimension (Height, Width) for standardized processing
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (512, 512)

# Logging configuration
DEFAULT_LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


def get_default_device() -> str:
    """Determine best available compute device ('cuda' if available else 'cpu').

    Returns:
        str: Device identifier string ('cuda' or 'cpu').
    """
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def ensure_project_directories() -> None:
    """Ensure that all core runtime directories exist.

    Creates directories safely if they do not exist already.
    """
    required_dirs = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SAMPLES_DATA_DIR,
        PRETRAINED_MODELS_DIR,
        CHECKPOINTS_DIR,
        OUTPUT_IMAGES_DIR,
        OUTPUT_VIDEOS_DIR,
        OUTPUT_METRICS_DIR,
        NOTEBOOKS_DIR,
        TESTS_DIR,
        SCRIPTS_DIR,
        DOCS_DIR,
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)
