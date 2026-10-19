"""Dehazing algorithms, models, and baselines package.

Contains classical physical dehazing methods (Dark Channel Prior),
edge-preserving transmission map refinement routines, and deep learning
U-Net architectures.
"""

from src.dehazing.dcp import (
    compute_dark_channel,
    dcp_dehaze,
    estimate_atmospheric_light,
    estimate_transmission,
    recover_radiance,
)
from src.dehazing.guided_filter import get_guided_filter_backend, guided_filter
from src.dehazing.model import LightweightUNet, load_checkpoint, save_checkpoint

__all__ = [
    "compute_dark_channel",
    "dcp_dehaze",
    "estimate_atmospheric_light",
    "estimate_transmission",
    "recover_radiance",
    "guided_filter",
    "get_guided_filter_backend",
    "LightweightUNet",
    "save_checkpoint",
    "load_checkpoint",
]
