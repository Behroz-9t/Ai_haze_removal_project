"""Unit Tests for Stage 2C.3: Pretrained Dehazing Model Integration and Fair Benchmarking.

Tests:
1. Pretrained checkpoint existence, file integrity, and SHA-256 verification.
2. DehazeFormerAdapter initialization and missing/corrupted checkpoint failure handling.
3. Adapter preprocessing (mapping uint8 [0, 255] to float32 [-1.0, 1.0]).
4. Adapter postprocessing (mapping float32 [-1.0, 1.0] to uint8 [0, 255]).
5. Full CPU inference pipeline on synthetic images, verifying shape, range, and latency keys.
6. Parameter count and architectural component validation.
"""

import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.dehazing.external.dehazeformer import MCT, DehazeFormer
from src.dehazing.external.dehazeformer_adapter import DehazeFormerAdapter


class TestStage2C3(unittest.TestCase):
    """Test suite for DehazeFormer adapter and pretrained model integration."""

    @classmethod
    def setUpClass(cls):
        cls.pretrained_dir = Path("models/pretrained")
        cls.chk_path = cls.pretrained_dir / "dehazeformer_mct.pth"
        cls.expected_sha256 = "225ce2d00d1cd8f0a7aabf5dc09f020ee34636cc942aa8b1a87496506224e304"

    def test_checkpoint_existence_and_checksum(self):
        """Verify that the official pretrained checkpoint exists and matches SHA-256."""
        self.assertTrue(self.chk_path.is_file(), f"Pretrained checkpoint missing at {self.chk_path}")
        size_bytes = self.chk_path.stat().st_size
        self.assertGreater(size_bytes, 5 * 1024 * 1024, f"Checkpoint size too small: {size_bytes} bytes")

        hasher = hashlib.sha256()
        with open(self.chk_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        calc_sha256 = hasher.hexdigest()
        self.assertEqual(calc_sha256, self.expected_sha256, "SHA-256 checksum mismatch on pretrained checkpoint")

    def test_adapter_missing_checkpoint_raises(self):
        """Verify that passing a nonexistent checkpoint path raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            DehazeFormerAdapter("models/pretrained/nonexistent_model.pth")

    def test_adapter_corrupted_checkpoint_raises(self):
        """Verify that a corrupted checkpoint file raises an exception."""
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            tmp.write(b"NOT_A_VALID_PYTORCH_CHECKPOINT_DATA_HEADER")
            tmp_path = Path(tmp.name)

        try:
            with self.assertRaises(Exception):
                DehazeFormerAdapter(tmp_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_adapter_preprocessing_and_normalization(self):
        """Verify uint8 [0, 255] is normalized to float32 [-1.0, 1.0] tensor."""
        adapter = DehazeFormerAdapter(self.chk_path, device="cpu")

        # Test extreme boundaries: black (0) and white (255)
        black_img = np.zeros((64, 64, 3), dtype=np.uint8)
        white_img = np.full((64, 64, 3), 255, dtype=np.uint8)

        t_black = adapter.preprocess(black_img)
        t_white = adapter.preprocess(white_img)

        self.assertEqual(t_black.shape, (1, 3, 64, 64))
        self.assertEqual(t_white.shape, (1, 3, 64, 64))
        self.assertEqual(t_black.dtype, torch.float32)
        self.assertEqual(t_white.dtype, torch.float32)

        self.assertAlmostEqual(float(t_black.min()), -1.0, places=3)
        self.assertAlmostEqual(float(t_black.max()), -1.0, places=3)
        self.assertAlmostEqual(float(t_white.min()), 1.0, places=3)
        self.assertAlmostEqual(float(t_white.max()), 1.0, places=3)

    def test_adapter_postprocessing_range(self):
        """Verify float32 [-1.0, 1.0] tensor is mapped back to uint8 [0, 255]."""
        adapter = DehazeFormerAdapter(self.chk_path, device="cpu")

        # Tensor with -1.0, 0.0, 1.0
        t_in = torch.tensor([[-1.0, 0.0], [1.0, 2.0]], dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)

        out_np = adapter.postprocess(t_in)
        self.assertEqual(out_np.dtype, np.uint8)
        self.assertEqual(out_np.shape, (2, 2, 3))
        self.assertEqual(int(out_np[0, 0, 0]), 0)     # -1.0 -> 0
        self.assertIn(int(out_np[0, 1, 0]), [127, 128])  # 0.0 -> 127.5 rounded
        self.assertEqual(int(out_np[1, 0, 0]), 255)   # 1.0 -> 255
        self.assertEqual(int(out_np[1, 1, 0]), 255)   # 2.0 -> clamped 255

    def test_cpu_inference_pipeline_and_timing(self):
        """Verify end-to-end CPU dehazing inference, shape preservation, and latency profiling."""
        adapter = DehazeFormerAdapter(self.chk_path, device="cpu")

        # Synthetic image of arbitrary dimensions
        test_h, test_w = 120, 160
        rng = np.random.RandomState(42)
        sample_img = rng.randint(0, 256, (test_h, test_w, 3), dtype=np.uint8)

        out_rgb, timing = adapter.dehaze(sample_img)

        self.assertEqual(out_rgb.shape, (test_h, test_w, 3))
        self.assertEqual(out_rgb.dtype, np.uint8)

        # Check required timing keys
        for key in ["pre_ms", "infer_ms", "post_ms", "total_ms"]:
            self.assertIn(key, timing)
            self.assertGreaterEqual(timing[key], 0.0)

        # Check total time equals sum of parts within rounding tolerance
        sum_parts = timing["pre_ms"] + timing["infer_ms"] + timing["post_ms"]
        self.assertAlmostEqual(timing["total_ms"], sum_parts, delta=0.2)

    def test_architecture_parameter_count(self):
        """Verify model architecture parameter count is approximately ~1.23M parameters."""
        adapter = DehazeFormerAdapter(self.chk_path, device="cpu")
        param_count = adapter.count_parameters()
        self.assertGreater(param_count, 1_000_000)
        self.assertLess(param_count, 2_000_000)
        self.assertEqual(param_count, 1_233_024)


if __name__ == "__main__":
    unittest.main()
