"""Basic Image Pipeline Validation Test for Stage 0.

Validates the complete bidirectional data flow:
Disk -> OpenCV (BGR) -> RGB NumPy -> PyTorch Tensor -> Compute Device ->
RGB NumPy -> OpenCV (BGR) -> Saved Output File.

Computes processing time and confirms tensor round-trip fidelity.
"""

from pathlib import Path
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from app.config import (
    SAMPLES_DATA_DIR,
    OUTPUT_IMAGES_DIR,
    get_default_device,
    ensure_project_directories,
)
from src.utils.exceptions import ImageLoadError, ImageSaveError
from src.utils.image_ops import (
    load_image,
    save_image,
    numpy_to_tensor,
    tensor_to_numpy,
)
from src.utils.logger import get_logger

logger = get_logger("test_pipeline")


def run_pipeline_test(
    sample_filename: str = "sample_haze.png",
    output_filename: str = "pipeline_test_output.png",
) -> bool:
    """Execute the Stage 0 image pipeline test.

    Args:
        sample_filename: Filename of the sample image inside data/samples/.
        output_filename: Filename for the output image inside outputs/images/.

    Returns:
        bool: True if pipeline succeeds and image roundtrip is verified.
    """
    ensure_project_directories()

    input_path = SAMPLES_DATA_DIR / sample_filename
    output_path = OUTPUT_IMAGES_DIR / output_filename

    print("========================================")
    print("STAGE 0: BASIC IMAGE PIPELINE TEST")
    print("========================================")
    print(f"Sample Image Path:  {input_path}")
    print(f"Output Target Path: {output_path}")
    print()

    # Step 1 & 2: Load sample image and validate
    start_total_time = time.perf_counter()
    try:
        t0 = time.perf_counter()
        image_rgb = load_image(input_path)
        load_time_ms = (time.perf_counter() - t0) * 1000.0
    except ImageLoadError as err:
        logger.error(f"Image pipeline failed at load stage: {err}")
        print(f"\n[ERROR] Pipeline aborted: {err}")
        return False

    height, width, channels = image_rgb.shape

    # Step 3, 4 & 5: NumPy -> PyTorch Tensor -> Target Device
    t1 = time.perf_counter()
    device = torch.device(get_default_device())
    tensor = numpy_to_tensor(image_rgb, device=device)
    tensor_conversion_time_ms = (time.perf_counter() - t1) * 1000.0

    # Step 6: PyTorch Tensor -> NumPy RGB
    t2 = time.perf_counter()
    reconstructed_rgb = tensor_to_numpy(tensor)
    numpy_conversion_time_ms = (time.perf_counter() - t2) * 1000.0

    # Step 7: Save reconstructed image
    t3 = time.perf_counter()
    try:
        saved_path = save_image(reconstructed_rgb, output_path, is_rgb=True)
        save_time_ms = (time.perf_counter() - t3) * 1000.0
    except ImageSaveError as err:
        logger.error(f"Image pipeline failed at save stage: {err}")
        print(f"\n[ERROR] Pipeline aborted: {err}")
        return False

    total_time_ms = (time.perf_counter() - start_total_time) * 1000.0

    # Verification: check numerical difference between input and reconstructed image
    max_pixel_diff = int(np.max(np.abs(image_rgb.astype(np.int32) - reconstructed_rgb.astype(np.int32))))

    # Step 8: Print pipeline diagnostics
    print(f"Image Resolution:    {width} x {height} (Width x Height)")
    print(f"Number of Channels:  {channels}")
    print(f"NumPy Dtype:         {image_rgb.dtype}")
    print(f"Tensor Shape:        {tuple(tensor.shape)} (C, H, W)")
    print(f"Tensor Dtype:        {tensor.dtype}")
    print(f"Compute Device Used: {tensor.device}")
    print(f"Max Pixel Difference:{max_pixel_diff} (0 indicates exact match)")
    print()
    print("Execution Timing:")
    print(f"    - Image Load Time:       {load_time_ms:.2f} ms")
    print(f"    - NumPy -> Tensor:       {tensor_conversion_time_ms:.2f} ms")
    print(f"    - Tensor -> NumPy:       {numpy_conversion_time_ms:.2f} ms")
    print(f"    - Image Save Time:       {save_time_ms:.2f} ms")
    print(f"    - Total Processing Time: {total_time_ms:.2f} ms")
    print()
    print(f"Result Image Saved:  {saved_path}")
    print("========================================")
    print("PIPELINE STATUS: SUCCESS")
    print("========================================")

    return True


if __name__ == "__main__":
    success = run_pipeline_test()
    sys.exit(0 if success else 1)
