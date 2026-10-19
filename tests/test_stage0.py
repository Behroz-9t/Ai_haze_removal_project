"""Basic unit tests for Stage 0 foundation components.

Tests environment imports, random seed determinism, and tensor roundtrip fidelity.
"""

from pathlib import Path
import unittest
import numpy as np
import torch

from app.config import PROJECT_ROOT, SAMPLES_DATA_DIR
from src.utils.seed import set_seed
from src.utils.image_ops import (
    load_image,
    save_image,
    numpy_to_tensor,
    tensor_to_numpy,
)
from src.utils.exceptions import ImageLoadError


class TestStage0Foundation(unittest.TestCase):
    """Test suite for Stage 0 foundation utilities."""

    def test_reproducible_seed(self):
        """Verify that set_seed ensures reproducible random state."""
        set_seed(1234)
        val1 = np.random.rand(5)
        torch_val1 = torch.rand(5)

        set_seed(1234)
        val2 = np.random.rand(5)
        torch_val2 = torch.rand(5)

        np.testing.assert_allclose(val1, val2)
        self.assertTrue(torch.allclose(torch_val1, torch_val2))

    def test_tensor_roundtrip_fidelity(self):
        """Verify that converting NumPy image to tensor and back preserves pixel values."""
        # Create synthetic RGB test array
        original = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        tensor = numpy_to_tensor(original, device="cpu")

        self.assertEqual(tensor.shape, (3, 64, 64))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(0.0 <= tensor.min() <= tensor.max() <= 1.0)

        reconstructed = tensor_to_numpy(tensor)
        self.assertEqual(reconstructed.shape, (64, 64, 3))
        self.assertEqual(reconstructed.dtype, np.uint8)

        # Allow at most 1 unit rounding difference due to float normalization
        max_diff = np.max(np.abs(original.astype(int) - reconstructed.astype(int)))
        self.assertLessEqual(max_diff, 1)

    def test_missing_image_raises_load_error(self):
        """Verify that loading a non-existent image explicitly raises ImageLoadError."""
        with self.assertRaises(ImageLoadError):
            load_image(PROJECT_ROOT / "data" / "samples" / "does_not_exist.png")


if __name__ == "__main__":
    unittest.main()
