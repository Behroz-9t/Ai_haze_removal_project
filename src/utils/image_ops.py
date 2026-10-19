"""Image I/O and Tensor Conversion Utilities.

Implements robust image loading, validation, conversion, and saving with
explicit error handling to prevent silent failures across pipeline stages.
"""

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch

from app.config import get_default_device
from src.utils.exceptions import ImageLoadError, ImageSaveError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_image(file_path: Union[str, Path]) -> np.ndarray:
    """Load an image from disk and convert to RGB NumPy array.

    Args:
        file_path: Path to the image file.

    Returns:
        np.ndarray: Loaded image as an RGB uint8 array of shape (H, W, C).

    Raises:
        ImageLoadError: If file does not exist, cannot be read, or is empty.
    """
    path = Path(file_path)
    if not path.is_file():
        raise ImageLoadError(f"Image file does not exist: {path}")

    # Read image using OpenCV (returns BGR)
    image_bgr = cv2.imread(str(path))
    if image_bgr is None or image_bgr.size == 0:
        raise ImageLoadError(
            f"Failed to decode image from '{path}'. File may be corrupted or in an unsupported format."
        )

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    logger.debug(f"Loaded image '{path.name}' with shape {image_rgb.shape} (RGB)")
    return image_rgb


def save_image(
    image: np.ndarray,
    destination_path: Union[str, Path],
    is_rgb: bool = True,
) -> Path:
    """Save a NumPy image array to disk.

    Args:
        image: Image array of shape (H, W, C) or (H, W).
        destination_path: Target path to write the image.
        is_rgb: If True, converts from RGB to BGR before saving with OpenCV.

    Returns:
        Path: Resolved path of the saved image file.

    Raises:
        ImageSaveError: If the destination directory cannot be created or saving fails.
    """
    if image is None or image.size == 0:
        raise ImageSaveError("Cannot save empty or None image array.")

    path = Path(destination_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        raise ImageSaveError(f"Could not create destination directory '{path.parent}': {err}") from err

    # Prepare array for OpenCV (requires BGR format)
    if is_rgb and image.ndim == 3 and image.shape[2] == 3:
        output_data = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        output_data = image

    # Ensure correct dtype
    if output_data.dtype != np.uint8:
        if np.issubdtype(output_data.dtype, np.floating):
            output_data = np.clip(output_data * 255.0, 0, 255).astype(np.uint8)
        else:
            output_data = np.clip(output_data, 0, 255).astype(np.uint8)

    success = cv2.imwrite(str(path), output_data)
    if not success:
        raise ImageSaveError(f"OpenCV failed to write image to '{path}'. Check format and permissions.")

    logger.debug(f"Successfully saved image to '{path}'")
    return path


def numpy_to_tensor(
    image: np.ndarray,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Convert an RGB NumPy image (H, W, C) [0..255] to a PyTorch tensor (C, H, W) [0.0..1.0].

    Args:
        image: RGB NumPy array with shape (H, W, C) or (H, W).
        device: Target compute device. Defaults to get_default_device().

    Returns:
        torch.Tensor: Normalized float32 tensor of shape (C, H, W) on target device.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image array cannot be None or empty.")

    target_device = torch.device(device if device is not None else get_default_device())

    # Handle grayscale by adding channel dimension
    if image.ndim == 2:
        image = np.expand_dims(image, axis=-1)

    # Convert HWC to CHW
    transposed = np.transpose(image, (2, 0, 1))

    # Convert to contiguous float32 tensor scaled to [0.0, 1.0]
    tensor = torch.from_numpy(transposed).float()
    if image.dtype == np.uint8:
        tensor = tensor / 255.0

    return tensor.to(target_device)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor (C, H, W) [0.0..1.0] back to an RGB NumPy image (H, W, C) [0..255].

    Args:
        tensor: Normalized PyTorch tensor with shape (C, H, W).

    Returns:
        np.ndarray: RGB uint8 NumPy array of shape (H, W, C).
    """
    if tensor is None:
        raise ValueError("Input tensor cannot be None.")

    # Detach from autograd and move to host memory
    arr = tensor.detach().cpu().numpy()

    # If batched (1, C, H, W), remove batch dimension
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr.squeeze(0)

    # Convert CHW to HWC
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))

    # Clamp to [0.0, 1.0] and scale to [0, 255] uint8
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).round().astype(np.uint8)

    return arr
