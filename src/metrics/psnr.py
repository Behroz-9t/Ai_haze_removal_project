"""Peak Signal-to-Noise Ratio (PSNR) Metric Implementation.

Computes PSNR between target and reference images in dB.
    PSNR = 10 * log10(MAX_I^2 / MSE)
"""

from typing import Union
import numpy as np


def calculate_psnr(
    target: np.ndarray,
    reference: np.ndarray,
    max_val: Union[float, int] = 255.0,
) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).

    Args:
        target: Evaluated image (uint8 [0..255] or float32 [0.0..1.0]).
        reference: Ground-truth reference image (same shape and dtype as target).
        max_val: Maximum possible pixel intensity (255.0 for uint8, 1.0 for float).

    Returns:
        float: PSNR in dB. Returns float('inf') for identical images (MSE == 0).

    Raises:
        ValueError: If input dimensions or shapes do not match.
    """
    if target.shape != reference.shape:
        raise ValueError(
            f"Shape mismatch for PSNR calculation: target is {target.shape}, "
            f"reference is {reference.shape}"
        )

    t_f = target.astype(np.float64)
    r_f = reference.astype(np.float64)

    mse = np.mean((t_f - r_f) ** 2)

    # Identical images: zero mean squared error
    if mse == 0.0:
        return float("inf")

    # Guard against NaN/negative values
    if np.isnan(mse) or mse < 0.0:
        return 0.0

    psnr = 10.0 * np.log10((float(max_val) ** 2) / mse)
    return float(psnr)
