"""Reproducibility utility.

Provides a unified function to seed Python's random module, NumPy, and PyTorch
(including CUDA backends if available) for deterministic execution.
"""

import os
import random
from typing import Optional

import numpy as np
import torch

from app.config import DEFAULT_RANDOM_SEED
from src.utils.logger import get_logger

logger = get_logger(__name__)


def set_seed(seed: Optional[int] = None, deterministic: bool = True) -> int:
    """Set random seeds across Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Integer seed value. Defaults to DEFAULT_RANDOM_SEED (42) from config.
        deterministic: If True, sets CuDNN to deterministic mode when CUDA is present.

    Returns:
        int: The seed that was applied.
    """
    applied_seed = DEFAULT_RANDOM_SEED if seed is None else seed

    # Python built-in random and hash seed
    random.seed(applied_seed)
    os.environ["PYTHONHASHSEED"] = str(applied_seed)

    # NumPy random seed
    np.random.seed(applied_seed)

    # PyTorch CPU & GPU seeds
    torch.manual_seed(applied_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(applied_seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    logger.info(f"Deterministic seed set to {applied_seed} (deterministic={deterministic})")
    return applied_seed
