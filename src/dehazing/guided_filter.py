"""Guided Filter Implementation for Edge-Preserving Transmission Map Refinement.

Checks for native OpenCV ximgproc availability; if unavailable, uses an
optimized, zero-dependency NumPy + cv2.boxFilter fallback that matches the
mathematical formulation of He et al. (TPAMI 2013) with O(N) complexity.
"""

from typing import Tuple
import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Check if opencv-contrib ximgproc is installed
_HAS_XIMGPROC = hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter")

if _HAS_XIMGPROC:
    logger.info("Guided Filter backend: OpenCV ximgproc.guidedFilter (Native C++)")
else:
    logger.info(
        "Guided Filter backend: Modular NumPy + cv2.boxFilter fallback "
        "(opencv-contrib not installed; using pure-array O(1) box filtering)"
    )


def get_guided_filter_backend() -> str:
    """Return the name of the active guided filter implementation backend."""
    return "opencv_ximgproc" if _HAS_XIMGPROC else "modular_numpy_boxfilter"


def _box_filter(src: np.ndarray, radius: int) -> np.ndarray:
    """Fast box/mean filter using cv2.boxFilter with reflected boundary condition."""
    kernel_size = 2 * radius + 1
    return cv2.boxFilter(
        src,
        ddepth=-1,
        ksize=(kernel_size, kernel_size),
        borderType=cv2.BORDER_REFLECT,
    )


def guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = 40,
    eps: float = 1e-3,
) -> np.ndarray:
    """Apply edge-preserving guided filter to refine a source map.

    Args:
        guide: Guidance image (single channel 2D float32 [0.0..1.0] or uint8 [0..255]).
        src: Source map to be filtered (single channel 2D float32 [0.0..1.0]).
        radius: Local window filter radius.
        eps: Regularization parameter preventing division by near-zero variance.

    Returns:
        np.ndarray: Filtered float32 map of same shape as src, values in [0.0..1.0].
    """
    if guide.ndim != 2:
        raise ValueError(f"Guide must be a single-channel 2D image, got shape {guide.shape}")
    if src.ndim != 2:
        raise ValueError(f"Source map must be a single-channel 2D image, got shape {src.shape}")
    if guide.shape != src.shape:
        raise ValueError(f"Shape mismatch: guide is {guide.shape}, src is {src.shape}")

    # Prepare normalized float32 representations
    i_guide = guide.astype(np.float32)
    if guide.dtype == np.uint8:
        i_guide = i_guide / 255.0
    else:
        i_guide = np.clip(i_guide, 0.0, 1.0)

    p_src = np.clip(src.astype(np.float32), 0.0, 1.0)

    # Use native OpenCV if available
    if _HAS_XIMGPROC:
        refined = cv2.ximgproc.guidedFilter(
            guide=i_guide,
            src=p_src,
            radius=radius,
            eps=eps,
        )
        return np.clip(refined, 0.0, 1.0).astype(np.float32)

    # Modular Fallback Implementation (He et al. TPAMI 2013)
    mean_i = _box_filter(i_guide, radius)
    mean_p = _box_filter(p_src, radius)
    mean_ip = _box_filter(i_guide * p_src, radius)
    cov_ip = mean_ip - mean_i * mean_p

    mean_ii = _box_filter(i_guide * i_guide, radius)
    var_i = mean_ii - mean_i * mean_i

    # Linear coefficients a and b
    # var_i is guaranteed non-negative; eps prevents division by zero
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i

    mean_a = _box_filter(a, radius)
    mean_b = _box_filter(b, radius)

    q = mean_a * i_guide + mean_b

    # Safe handling: replace any anomalous NaN/Inf values with original src
    if np.isnan(q).any() or np.isinf(q).any():
        logger.warning("NaN or Inf detected in guided filter output; falling back to clamped src.")
        q = np.nan_to_num(q, nan=p_src, posinf=1.0, neginf=0.0)

    return np.clip(q, 0.0, 1.0).astype(np.float32)
