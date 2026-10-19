"""Haze Density Assessment and Atmospheric Scattering Detection Module.

Analyzes image transmission properties, dark channel statistics, and contrast
distributions to assess whether an image contains atmospheric degradation (fog, haze,
mist, smog) or is already clear/near-clear. Prevents assuming that every uploaded image
requires aggressive dehazing.
"""

from typing import Any, Dict
import cv2
import numpy as np

from src.dehazing.dcp import compute_dark_channel


def estimate_haze_density(
    image_rgb: np.ndarray,
    patch_size: int = 15,
) -> Dict[str, Any]:
    """Analyze dark channel and contrast metrics to estimate atmospheric haze severity.

    In clear scenes without haze, the dark channel in non-sky regions has very low
    intensity, and the luminance contrast (standard deviation) is high. In hazy scenes,
    airlight scattering raises the dark channel while washing out overall contrast.

    We compute the composite Haze Obscuration Index:
        H_index = DarkChannel_mean / max(Contrast_std, 0.05)

    Args:
        image_rgb: Input RGB image as uint8 [0..255] or float32 [0.0..1.0].
        patch_size: Neighborhood window for local dark channel computation.

    Returns:
        Dict[str, Any]: Assessment metrics containing:
            - 'dark_channel_mean': Average dark channel intensity in [0..1].
            - 'dark_channel_p95': 95th percentile intensity of the dark channel.
            - 'contrast_std': Standard deviation of luminance (contrast measure).
            - 'haze_index': Composite ratio of dark channel to contrast.
            - 'is_likely_hazy': Boolean flag indicating if haze was detected.
            - 'severity_label': "Clear / Near-Clear", "Light Haze", "Moderate Haze", or "Dense / Heavy Haze".
            - 'recommended_action': Guidance string for user/pipeline.
    """
    if image_rgb is None or image_rgb.size == 0:
        raise ValueError("Input image cannot be None or empty.")

    if image_rgb.dtype == np.uint8:
        img_float = image_rgb.astype(np.float32) / 255.0
    else:
        img_float = np.clip(image_rgb.astype(np.float32), 0.0, 1.0)

    # 1. Dark Channel prior analysis
    dark_map = compute_dark_channel(img_float, patch_size=patch_size)
    dc_mean = float(np.mean(dark_map))
    dc_p95 = float(np.percentile(dark_map, 95))

    # 2. Contrast estimation via grayscale standard deviation
    gray = cv2.cvtColor((img_float * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    contrast_std = float(np.std(gray))

    # 3. Composite Haze Index
    haze_index = dc_mean / max(contrast_std, 0.05)

    # 4. Calibrated Decision Boundaries
    # Clear scenes: H_index < 1.05 and dc_mean < 0.28 (or very low dark channel < 0.08)
    if (haze_index < 1.10 and dc_mean < 0.28) or dc_mean < 0.08:
        severity = "Clear / Near-Clear"
        is_hazy = False
        action = "Image appears clear. Enhancement may alter natural contrast or saturation."
    elif haze_index < 1.85:
        severity = "Light Haze"
        is_hazy = True
        action = "Mild haze detected. Subtle enhancement recommended."
    elif haze_index < 2.80:
        severity = "Moderate Haze"
        is_hazy = True
        action = "Significant atmospheric scattering detected. Standard dehazing recommended."
    else:
        severity = "Dense / Heavy Haze"
        is_hazy = True
        action = "Severe atmospheric obscuration detected. Deep enhancement required."

    return {
        "dark_channel_mean": round(dc_mean, 4),
        "dark_channel_p95": round(dc_p95, 4),
        "contrast_std": round(contrast_std, 4),
        "haze_index": round(haze_index, 4),
        "is_likely_hazy": is_hazy,
        "severity_label": severity,
        "recommended_action": action,
    }
