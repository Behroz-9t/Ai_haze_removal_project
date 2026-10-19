"""Structural Similarity Index Measure (SSIM) Implementation.

Computes mean SSIM according to Wang et al. (IEEE TIP 2004):
    SSIM(x, y) = [ (2*mu_x*mu_y + C1) * (2*sigma_xy + C2) ] /
                 [ (mu_x^2 + mu_y^2 + C1) * (sigma_x^2 + sigma_y^2 + C2) ]
"""

from typing import Optional, Union
import cv2
import numpy as np


def _ssim_single_channel(
    target: np.ndarray,
    reference: np.ndarray,
    max_val: float = 255.0,
    k1: float = 0.01,
    k2: float = 0.03,
    win_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Compute mean SSIM for a single 2D channel using Gaussian weighting."""
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    # Standard Gaussian kernel window
    x = target.astype(np.float64)
    y = reference.astype(np.float64)

    # Local means
    mu_x = cv2.GaussianBlur(x, (win_size, win_size), sigma)
    mu_y = cv2.GaussianBlur(y, (win_size, win_size), sigma)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    # Local variances and covariance
    sigma_x_sq = cv2.GaussianBlur(x * x, (win_size, win_size), sigma) - mu_x_sq
    sigma_y_sq = cv2.GaussianBlur(y * y, (win_size, win_size), sigma) - mu_y_sq
    sigma_xy = cv2.GaussianBlur(x * y, (win_size, win_size), sigma) - mu_xy

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)

    ssim_map = numerator / denominator
    return float(np.mean(ssim_map))


def calculate_ssim(
    target: np.ndarray,
    reference: np.ndarray,
    max_val: Union[float, int] = 255.0,
    win_size: int = 11,
    channel_axis: Optional[int] = -1,
) -> float:
    """Calculate Mean Structural Similarity Index Measure (MSSIM).

    Args:
        target: Evaluated image (2D or 3D NumPy array).
        reference: Ground-truth reference image (same shape and dtype as target).
        max_val: Dynamic range of pixel values (255.0 for uint8, 1.0 for float).
        win_size: Local Gaussian window size (default: 11).
        channel_axis: Axis index for color channels (default: -1 for HWC).

    Returns:
        float: Mean SSIM score in range [-1.0..1.0]. A score of 1.0 indicates identity.

    Raises:
        ValueError: If input shapes do not match.
    """
    if target.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch for SSIM calculation: target is {target.shape}, "
            f"reference is {reference.shape}"
        )

    # Check for exact array identity to return 1.0 directly
    if np.array_equal(target, reference):
        return 1.0

    if target.ndim == 2:
        return _ssim_single_channel(target, reference, max_val=float(max_val), win_size=win_size)

    if target.ndim == 3:
        num_channels = target.shape[channel_axis]
        channel_scores = []
        for c in range(num_channels):
            t_slice = target[:, :, c] if channel_axis == -1 or channel_axis == 2 else target[c, :, :]
            r_slice = reference[:, :, c] if channel_axis == -1 or channel_axis == 2 else reference[c, :, :]
            channel_scores.append(
                _ssim_single_channel(t_slice, r_slice, max_val=float(max_val), win_size=win_size)
            )
        return float(np.mean(channel_scores))

    raise ValueError(f"SSIM supports 2D or 3D images, got ndim={target.ndim}")
