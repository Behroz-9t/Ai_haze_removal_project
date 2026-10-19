"""Dark Channel Prior (DCP) Classical Dehazing Baseline.

Implements the single image haze removal algorithm proposed by
He, Sun, and Tang (CVPR 2009 / IEEE TPAMI 2011).

The algorithm comprises 5 distinct stages:
    1. Dark Channel Computation
    2. Atmospheric Light (Airlight) Estimation
    3. Coarse Transmission Map Estimation
    4. Edge-Preserving Transmission Refinement (Guided Filter)
    5. Scene Radiance Recovery
"""

from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

from src.dehazing.guided_filter import guided_filter
from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_dark_channel(image: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """Compute the dark channel prior map: min across channels followed by local min filter.

    J^{dark}(x) = min_{y in Omega(x)} ( min_{c in {r,g,b}} I^c(y) )

    Args:
        image: RGB image as float32 in [0.0..1.0] or uint8 in [0..255] of shape (H, W, 3).
        patch_size: Square window dimension for local minimum filtering (must be odd).

    Returns:
        np.ndarray: Single-channel dark channel map of shape (H, W).
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Image must have shape (H, W, 3), got {image.shape}")

    # Channel-wise minimum
    min_channel = np.min(image, axis=2)

    # Local minimum filter implemented via rectangular morphological erosion
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)

    return dark_channel


def estimate_atmospheric_light(
    image: np.ndarray,
    dark_channel: np.ndarray,
    top_percent: float = 0.001,
) -> np.ndarray:
    """Estimate global atmospheric light (airlight) A from the brightest pixels in the dark channel.

    Among the top (top_percent * total_pixels) brightest locations in J^{dark},
    the pixel with the highest intensity in the original image is selected.

    Args:
        image: RGB image as float32 in [0.0..1.0] of shape (H, W, 3).
        dark_channel: Precomputed dark channel float32 map of shape (H, W).
        top_percent: Fraction of brightest dark-channel pixels to consider (default 0.1%).

    Returns:
        np.ndarray: Atmospheric light vector A of shape (3,) in [0.0..1.0].
    """
    height, width = dark_channel.shape
    total_pixels = height * width
    num_candidates = max(1, int(total_pixels * top_percent))

    # Flatten dark channel and find indices of top candidate pixels
    flat_dark = dark_channel.reshape(-1)
    flat_image = image.reshape(-1, 3)

    # argpartition puts top candidate indices at the end
    candidate_indices = np.argpartition(flat_dark, -num_candidates)[-num_candidates:]

    # Among candidates, find the pixel with the highest brightness (luminance / max intensity)
    candidate_pixels = flat_image[candidate_indices]
    brightness = np.sum(candidate_pixels, axis=1)
    brightest_idx = np.argmax(brightness)

    atmospheric_light = candidate_pixels[brightest_idx]

    # Guard against completely black or zero atmospheric light to avoid division by zero
    atmospheric_light = np.clip(atmospheric_light, 0.05, 1.0)
    return atmospheric_light.astype(np.float32)


def estimate_transmission(
    image: np.ndarray,
    atmospheric_light: np.ndarray,
    omega: float = 0.95,
    patch_size: int = 15,
) -> np.ndarray:
    """Estimate raw transmission map: t~(x) = 1 - omega * dark_channel(I(x) / A).

    Args:
        image: Normalized RGB float32 image in [0.0..1.0].
        atmospheric_light: Estimated airlight vector A of shape (3,).
        omega: Haze retention factor in [0..1] (0.95 retains natural depth cues).
        patch_size: Window size for local dark channel calculation.

    Returns:
        np.ndarray: Coarse transmission map of shape (H, W) in [0.0..1.0].
    """
    a_broadcast = atmospheric_light.reshape((1, 1, 3))
    # Normalize image channel-wise by airlight
    norm_image = image / a_broadcast

    dark_norm = compute_dark_channel(norm_image, patch_size=patch_size)

    # Compute raw transmission
    transmission = 1.0 - omega * dark_norm
    return np.clip(transmission, 0.0, 1.0).astype(np.float32)


def recover_radiance(
    image: np.ndarray,
    atmospheric_light: np.ndarray,
    transmission: np.ndarray,
    t0: float = 0.1,
) -> np.ndarray:
    """Recover scene radiance: J(x) = (I(x) - A) / max(t(x), t0) + A.

    Args:
        image: Normalized RGB float32 image in [0.0..1.0].
        atmospheric_light: Airlight vector A of shape (3,).
        transmission: Refined or raw transmission map of shape (H, W).
        t0: Lower transmission threshold to prevent noise magnification in deep haze.

    Returns:
        np.ndarray: Dehazed RGB image in [0.0..1.0] of shape (H, W, 3).
    """
    a_broadcast = atmospheric_light.reshape((1, 1, 3))
    t_clamped = np.clip(transmission, t0, 1.0)[:, :, np.newaxis]

    radiance = (image - a_broadcast) / t_clamped + a_broadcast

    # Check for numerical stability: handle any NaNs or Infs
    if np.isnan(radiance).any() or np.isinf(radiance).any():
        logger.warning("Numerical anomaly detected during radiance recovery; clipping safely.")
        radiance = np.nan_to_num(radiance, nan=image, posinf=1.0, neginf=0.0)

    return np.clip(radiance, 0.0, 1.0).astype(np.float32)


def dcp_dehaze(
    image: np.ndarray,
    patch_size: int = 15,
    omega: float = 0.95,
    t0: float = 0.1,
    use_guided_filter: bool = True,
    guided_radius: int = 40,
    guided_eps: float = 1e-3,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Execute complete Dark Channel Prior single image dehazing pipeline.

    Args:
        image: Input RGB image as uint8 [0..255] or float32 [0.0..1.0] with shape (H, W, 3).
        patch_size: Local neighborhood window for dark channel calculation.
        omega: Haze preservation factor (typically 0.95).
        t0: Lower bound for transmission to prevent division artifacts.
        use_guided_filter: If True, refines transmission map with guided filter.
        guided_radius: Guided filter local window radius.
        guided_eps: Guided filter regularization parameter.

    Returns:
        tuple (dehazed_image, metadata):
            - dehazed_image: Output RGB image matching input dtype.
            - metadata: Dictionary containing 'atmospheric_light', 'raw_transmission',
                        'refined_transmission', and 'dark_channel'.

    Raises:
        ValueError: If input image is empty or has invalid shape.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image cannot be None or empty.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Input image must have 3 RGB channels, got shape {image.shape}")

    is_uint8 = image.dtype == np.uint8
    if is_uint8:
        img_float = image.astype(np.float32) / 255.0
    else:
        img_float = np.clip(image.astype(np.float32), 0.0, 1.0)

    # 1. Dark Channel Computation
    dark_channel = compute_dark_channel(img_float, patch_size=patch_size)

    # 2. Atmospheric Light Estimation
    atmospheric_light = estimate_atmospheric_light(img_float, dark_channel)

    # 3. Coarse Transmission Map Estimation
    raw_transmission = estimate_transmission(
        img_float,
        atmospheric_light,
        omega=omega,
        patch_size=patch_size,
    )

    # 4. Transmission Map Refinement
    if use_guided_filter:
        # Grayscale guide for edge alignment
        guide_gray = cv2.cvtColor(
            (img_float * 255.0).round().astype(np.uint8),
            cv2.COLOR_RGB2GRAY,
        ).astype(np.float32) / 255.0

        refined_transmission = guided_filter(
            guide=guide_gray,
            src=raw_transmission,
            radius=guided_radius,
            eps=guided_eps,
        )
    else:
        refined_transmission = raw_transmission

    # 5. Scene Radiance Recovery
    radiance_float = recover_radiance(
        img_float,
        atmospheric_light,
        refined_transmission,
        t0=t0,
    )

    if is_uint8:
        dehazed_out = (radiance_float * 255.0).round().astype(np.uint8)
    else:
        dehazed_out = radiance_float

    metadata = {
        "atmospheric_light": atmospheric_light,
        "dark_channel": dark_channel,
        "raw_transmission": raw_transmission,
        "refined_transmission": refined_transmission,
        "guided_filter_used": use_guided_filter,
    }

    return dehazed_out, metadata
