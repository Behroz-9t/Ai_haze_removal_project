"""Visualization utilities for side-by-side comparative inspection.

Generates unified multi-panel comparison strips (Clear | Hazy | Dehazed)
with consistent dimensions and readable status labels.
"""

from typing import Sequence, Tuple
import cv2
import numpy as np


def create_multi_panel(
    images: Sequence[np.ndarray],
    labels: Sequence[str],
    banner_height: int = 36,
) -> np.ndarray:
    """Combine arbitrary number of images horizontally into a labeled multi-panel strip.

    Args:
        images: Sequence of RGB images (H, W, 3).
        labels: Sequence of label strings matching images length.
        banner_height: Pixel height for title banner above each panel.

    Returns:
        np.ndarray: Composite RGB uint8 image of shape (H + banner_height, N * W, 3).
    """
    if len(images) == 0:
        raise ValueError("Must provide at least one image.")
    if len(images) != len(labels):
        raise ValueError(f"Images count ({len(images)}) must match labels count ({len(labels)}).")

    target_h, target_w = images[0].shape[:2]
    norm_images = []

    for img in images:
        curr = img.copy()
        if curr.dtype != np.uint8:
            curr = np.clip(curr * 255.0, 0, 255).astype(np.uint8)
        if curr.shape[:2] != (target_h, target_w):
            curr = cv2.resize(curr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        norm_images.append(curr)

    if banner_height <= 0:
        return np.hstack(norm_images)

    paneled_images = []
    for idx, (img, label) in enumerate(zip(norm_images, labels)):
        banner = np.zeros((banner_height, target_w, 3), dtype=np.uint8)
        banner[:] = (35, 35, 40)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.50
        thickness = 1
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_x = max(8, (target_w - text_size[0]) // 2)
        text_y = (banner_height + text_size[1]) // 2

        cv2.putText(
            banner,
            label,
            (text_x, text_y),
            font,
            font_scale,
            (240, 240, 240),
            thickness,
            lineType=cv2.LINE_AA,
        )
        paneled_images.append(np.vstack([banner, img]))

    return np.hstack(paneled_images)


def create_comparison_panel(
    clear: np.ndarray,
    hazy: np.ndarray,
    dehazed: np.ndarray,
    labels: Sequence[str] = ("Ground Truth (Clear)", "Input (Hazy)", "Output (DCP Dehazed)"),
    banner_height: int = 36,
) -> np.ndarray:
    """Combine clear, hazy, and dehazed images horizontally into a labeled comparison strip.

    Args:
        clear: Ground truth RGB image of shape (H, W, 3) uint8.
        hazy: Hazy input RGB image of shape (H, W, 3) uint8.
        dehazed: Dehazed output RGB image of shape (H, W, 3) uint8.
        labels: 3-element sequence of header strings for each panel.
        banner_height: Pixel height for title banner above each panel (0 for no banner).

    Returns:
        np.ndarray: Composite RGB uint8 image of shape (H + banner_height, 3 * W, 3).
    """
    return create_multi_panel(
        images=[clear, hazy, dehazed],
        labels=labels,
        banner_height=banner_height,
    )
