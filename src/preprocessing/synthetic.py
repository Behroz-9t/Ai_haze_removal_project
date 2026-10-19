"""Controlled Synthetic Haze Generation using the Atmospheric Scattering Model.

Implements Koschmieder's physical optical degradation model:
    I(x) = J(x) * t(x) + A * (1 - t(x))
where:
    - J(x) is the ground-truth clear scene radiance
    - t(x) is the medium transmission map
    - A is the global atmospheric light (airlight)
    - I(x) is the observed hazy image
"""

from typing import Optional, Tuple, Union

import numpy as np

from src.utils.seed import set_seed


def create_transmission_map(
    shape: Tuple[int, int],
    beta: float = 1.0,
    depth_mode: str = "gradient",
    custom_depth: Optional[np.ndarray] = None,
    t_min: float = 0.05,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate a continuous transmission map t(x) = exp(-beta * d(x)).

    Args:
        shape: Spatial dimensions (Height, Width).
        beta: Atmospheric scattering coefficient (higher = denser haze).
        depth_mode: 'gradient' (horizon-distance gradient), 'uniform' (constant depth),
                    or 'radial' (distance from viewer/bottom-center).
        custom_depth: Optional user-provided normalized depth map d(x) in [0.0, 1.0].
        t_min: Lower bound clamp for transmission to avoid complete blackout/division issues.
        seed: Optional seed for reproducible subtle perlin/noise perturbation.

    Returns:
        np.ndarray: Transmission map of shape (Height, Width) with values in [t_min, 1.0].
    """
    height, width = shape

    if custom_depth is not None:
        depth = np.clip(custom_depth.astype(np.float32), 0.0, 1.0)
    elif depth_mode == "uniform":
        depth = np.ones((height, width), dtype=np.float32)
    elif depth_mode == "radial":
        # Viewer positioned at bottom center; distance increases radially outwards
        y_coords, x_coords = np.mgrid[0:height, 0:width].astype(np.float32)
        norm_y = (height - y_coords) / height
        norm_x = (x_coords - width / 2.0) / (width / 2.0)
        dist = np.sqrt(norm_y**2 + (0.5 * norm_x)**2)
        depth = np.clip(dist / np.max(dist), 0.0, 1.0)
    else:  # 'gradient' (default standard horizon model)
        # Depth is low near camera (bottom of image) and high near horizon/sky (top of image)
        y_coords = np.linspace(1.0, 0.0, height, dtype=np.float32)
        depth = np.repeat(y_coords[:, np.newaxis], width, axis=1)

    # Add subtle deterministic high-frequency perturbations if seed is given
    if seed is not None:
        rng = np.random.RandomState(seed)
        noise = rng.normal(0.0, 0.02, size=(height, width)).astype(np.float32)
        depth = np.clip(depth + noise, 0.0, 1.0)

    # Koschmieder transmission: t(x) = exp(-beta * d(x))
    transmission = np.exp(-beta * depth)
    transmission = np.clip(transmission, t_min, 1.0)

    return transmission.astype(np.float32)


def generate_synthetic_haze(
    clear_image: np.ndarray,
    beta: float = 1.0,
    atmospheric_light: Union[float, Tuple[float, float, float], np.ndarray] = 0.9,
    depth_mode: str = "gradient",
    custom_depth: Optional[np.ndarray] = None,
    t_min: float = 0.05,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a synthetically degraded hazy image from a clean RGB image.

    Args:
        clear_image: RGB image as uint8 [0..255] or float32 [0.0..1.0] of shape (H, W, 3).
        beta: Scattering coefficient (controls haze density).
        atmospheric_light: Atmospheric light value A. Either scalar in [0..1] or 3-element RGB.
        depth_mode: Model for depth estimation ('gradient', 'radial', 'uniform').
        custom_depth: Optional normalized depth map array in [0.0, 1.0].
        t_min: Minimum transmission clamp.
        seed: Deterministic random seed.

    Returns:
        tuple (hazy_image, transmission_map):
            - hazy_image: Degraded image matching input dtype and shape.
            - transmission_map: Float32 transmission map in range [t_min, 1.0] of shape (H, W).

    Raises:
        ValueError: If clear_image is invalid or has wrong dimensions.
    """
    if clear_image is None or clear_image.size == 0:
        raise ValueError("clear_image cannot be None or empty.")
    if clear_image.ndim != 3 or clear_image.shape[2] != 3:
        raise ValueError(f"clear_image must have shape (H, W, 3), got {clear_image.shape}")

    height, width, _ = clear_image.shape
    is_uint8 = clear_image.dtype == np.uint8

    # Normalize image to [0.0, 1.0] float32
    if is_uint8:
        j_norm = clear_image.astype(np.float32) / 255.0
    else:
        j_norm = np.clip(clear_image.astype(np.float32), 0.0, 1.0)

    # Parse and format atmospheric light A into shape (1, 1, 3)
    if isinstance(atmospheric_light, (int, float)):
        a_vec = np.full((1, 1, 3), float(atmospheric_light), dtype=np.float32)
    elif isinstance(atmospheric_light, (list, tuple, np.ndarray)):
        a_arr = np.array(atmospheric_light, dtype=np.float32).flatten()
        if len(a_arr) != 3:
            raise ValueError(f"atmospheric_light must have 3 RGB elements, got {len(a_arr)}")
        a_vec = a_arr.reshape((1, 1, 3))
        # If passed as uint8 range [0..255], convert to [0..1]
        if np.any(a_vec > 1.0):
            a_vec = a_vec / 255.0
    else:
        raise ValueError(f"Unsupported atmospheric_light type: {type(atmospheric_light)}")

    a_vec = np.clip(a_vec, 0.0, 1.0)

    # Compute continuous transmission map t(x)
    transmission = create_transmission_map(
        shape=(height, width),
        beta=beta,
        depth_mode=depth_mode,
        custom_depth=custom_depth,
        t_min=t_min,
        seed=seed,
    )
    # Broadcast transmission to (H, W, 1)
    t_broadcast = transmission[:, :, np.newaxis]

    # Atmospheric scattering equation: I(x) = J(x) * t(x) + A * (1 - t(x))
    hazy_float = j_norm * t_broadcast + a_vec * (1.0 - t_broadcast)
    hazy_float = np.clip(hazy_float, 0.0, 1.0)

    if is_uint8:
        hazy_out = (hazy_float * 255.0).round().astype(np.uint8)
    else:
        hazy_out = hazy_float

    return hazy_out, transmission
