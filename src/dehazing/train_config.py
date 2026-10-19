"""Training Configuration and Hyperparameters for Image Dehazing.

Provides CPU-safe conservative default hyperparameters for fast experimental verification
and smoke training runs, while remaining configurable for future GPU scaling.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import torch

from app.config import (
    CHECKPOINTS_DIR,
    DEFAULT_RANDOM_SEED,
    OUTPUTS_DIR,
    get_default_device,
)


@dataclass
class TrainingConfig:
    """Hyperparameters and runtime settings for dehazing model training."""

    # Data & Dimensions
    batch_size: int = 1
    patch_size: Tuple[int, int] = (128, 128)
    num_workers: int = 0  # 0 for Windows / CPU stability
    max_train_samples: Optional[int] = None
    max_val_samples: Optional[int] = None
    max_variants_per_scene: Optional[int] = None

    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 2
    validation_frequency: int = 1

    # Model Architecture
    base_channels: int = 24
    residual: bool = True

    # Reproducibility & Hardware
    seed: int = DEFAULT_RANDOM_SEED
    device: str = field(default_factory=get_default_device)

    # Checkpoint & Resume
    resume_path: Optional[Path] = None
    checkpoint_prefix: str = "model"

    # File Paths
    checkpoint_dir: Path = field(default_factory=lambda: CHECKPOINTS_DIR)
    history_dir: Path = field(default_factory=lambda: OUTPUTS_DIR / "training")

    def __post_init__(self) -> None:
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.history_dir = Path(self.history_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

    @property
    def best_checkpoint_path(self) -> Path:
        """Path for the best model checkpoint."""
        filename = "best_model.pth" if self.checkpoint_prefix in ("model", "") else f"{self.checkpoint_prefix}_best_model.pth"
        return self.checkpoint_dir / filename

    @property
    def latest_checkpoint_path(self) -> Path:
        """Path for the latest model checkpoint."""
        filename = "latest_model.pth" if self.checkpoint_prefix in ("model", "") else f"{self.checkpoint_prefix}_latest_model.pth"
        return self.checkpoint_dir / filename

    @property
    def history_csv_path(self) -> Path:
        """Path for the training history CSV file."""
        filename = "training_history.csv" if self.checkpoint_prefix in ("model", "") else f"{self.checkpoint_prefix}_training_history.csv"
        return self.history_dir / filename
