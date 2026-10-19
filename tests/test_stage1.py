"""Unit and Integration Tests for Stage 1 Components.

Covers:
1. Synthetic haze generation produces valid images.
2. Synthetic haze output stays within valid image ranges.
3. Dataset pairing correctly identifies matching images.
4. Missing pairs are detected.
5. DCP does not produce NaN/Inf.
6. DCP output has expected dimensions/channels.
7. PSNR works on identical images.
8. SSIM works on identical images.
9. Basic baseline pipeline completes on provided sample.
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from app.config import SAMPLES_DATA_DIR
from src.dehazing.dcp import dcp_dehaze
from src.metrics.psnr import calculate_psnr
from src.metrics.ssim import calculate_ssim
from src.preprocessing.dataset import PairedHazeDataset
from src.preprocessing.synthetic import (
    create_transmission_map,
    generate_synthetic_haze,
)
from src.utils.image_ops import load_image, save_image


class TestStage1DatasetAndBaseline(unittest.TestCase):
    """Test suite for Stage 1 Dataset Engineering and Classical DCP Baseline."""

    def setUp(self):
        # Create a small synthetic RGB test pattern (64x64)
        np.random.seed(42)
        self.clear_uint8 = np.random.randint(20, 220, (64, 64, 3), dtype=np.uint8)
        self.clear_float = (self.clear_uint8.astype(np.float32) / 255.0)

    # -------------------------------------------------------------------------
    # 1. Synthetic haze generation produces valid images
    # -------------------------------------------------------------------------
    def test_synthetic_haze_generation_valid_images(self):
        hazy, trans = generate_synthetic_haze(
            self.clear_uint8,
            beta=1.5,
            atmospheric_light=0.9,
            depth_mode="gradient",
            seed=42,
        )
        self.assertIsNotNone(hazy)
        self.assertEqual(hazy.shape, self.clear_uint8.shape)
        self.assertEqual(hazy.dtype, np.uint8)
        self.assertEqual(trans.shape, (64, 64))

    # -------------------------------------------------------------------------
    # 2. Synthetic haze output stays within valid image ranges
    # -------------------------------------------------------------------------
    def test_synthetic_haze_output_valid_ranges(self):
        # Test uint8 range
        hazy_u8, trans = generate_synthetic_haze(self.clear_uint8, beta=2.0)
        self.assertGreaterEqual(hazy_u8.min(), 0)
        self.assertLessEqual(hazy_u8.max(), 255)
        self.assertGreaterEqual(trans.min(), 0.05)
        self.assertLessEqual(trans.max(), 1.0)

        # Test float32 range
        hazy_f, _ = generate_synthetic_haze(self.clear_float, beta=1.0)
        self.assertGreaterEqual(hazy_f.min(), 0.0)
        self.assertLessEqual(hazy_f.max(), 1.0)

    # -------------------------------------------------------------------------
    # 3. Dataset pairing correctly identifies matching images
    # -------------------------------------------------------------------------
    def test_dataset_pairing_identifies_matching_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            clear_dir = root / "clear"
            hazy_dir = root / "hazy"
            clear_dir.mkdir()
            hazy_dir.mkdir()

            # Create 3 matching pairs
            for stem in ["img_01", "img_02", "img_03"]:
                save_image(self.clear_uint8, clear_dir / f"{stem}.png")
                save_image(self.clear_uint8, hazy_dir / f"{stem}.png")

            dataset = PairedHazeDataset(root_dir=root, strict_pairs=True)
            self.assertEqual(len(dataset), 3)
            self.assertEqual(len(dataset.unmatched_hazy), 0)
            self.assertEqual(len(dataset.unmatched_clear), 0)

            # Test loading sample via __getitem__
            sample = dataset[0]
            self.assertIn("hazy", sample)
            self.assertIn("clear", sample)
            self.assertIsInstance(sample["hazy"], torch.Tensor)
            self.assertEqual(sample["hazy"].shape, (3, 64, 64))

    # -------------------------------------------------------------------------
    # 4. Missing pairs are detected
    # -------------------------------------------------------------------------
    def test_dataset_pairing_detects_missing_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            clear_dir = root / "clear"
            hazy_dir = root / "hazy"
            clear_dir.mkdir()
            hazy_dir.mkdir()

            save_image(self.clear_uint8, clear_dir / "matched.png")
            save_image(self.clear_uint8, hazy_dir / "matched.png")
            save_image(self.clear_uint8, clear_dir / "unpaired_clear.png")
            save_image(self.clear_uint8, hazy_dir / "unpaired_hazy.png")

            # Non-strict mode should discover 1 valid pair and record unpaireeds
            dataset = PairedHazeDataset(root_dir=root, strict_pairs=False)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(len(dataset.unmatched_clear), 1)
            self.assertEqual(len(dataset.unmatched_hazy), 1)

            # Strict mode should raise ValueError
            with self.assertRaises(ValueError):
                PairedHazeDataset(root_dir=root, strict_pairs=True)

    # -------------------------------------------------------------------------
    # 5. DCP does not produce NaN/Inf
    # -------------------------------------------------------------------------
    def test_dcp_no_nan_inf(self):
        # Test on varied inputs including edge cases: uniform bright, uniform dark
        cases = [
            self.clear_uint8,
            np.zeros((64, 64, 3), dtype=np.uint8),  # Pure black
            np.full((64, 64, 3), 255, dtype=np.uint8),  # Pure white
        ]
        for img in cases:
            dehazed, meta = dcp_dehaze(img, patch_size=7, use_guided_filter=True)
            self.assertFalse(np.isnan(dehazed).any(), "DCP output contains NaN values")
            self.assertFalse(np.isinf(dehazed).any(), "DCP output contains Inf values")
            self.assertFalse(np.isnan(meta["refined_transmission"]).any())

    # -------------------------------------------------------------------------
    # 6. DCP output has expected dimensions and channels
    # -------------------------------------------------------------------------
    def test_dcp_output_expected_dimensions_and_channels(self):
        h, w = 48, 72
        custom_img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        dehazed, _ = dcp_dehaze(custom_img, patch_size=5)

        self.assertEqual(dehazed.shape, (h, w, 3))
        self.assertEqual(dehazed.dtype, np.uint8)

    # -------------------------------------------------------------------------
    # 7. PSNR works on identical images
    # -------------------------------------------------------------------------
    def test_psnr_identical_images(self):
        # Identical images must return float('inf') without division-by-zero errors
        psnr_ident = calculate_psnr(self.clear_uint8, self.clear_uint8, max_val=255.0)
        self.assertEqual(psnr_ident, float("inf"))

        # Slightly perturbed image should produce finite positive PSNR
        perturbed = np.clip(self.clear_uint8.astype(int) + 2, 0, 255).astype(np.uint8)
        psnr_val = calculate_psnr(perturbed, self.clear_uint8, max_val=255.0)
        self.assertTrue(30.0 < psnr_val < 50.0)

    # -------------------------------------------------------------------------
    # 8. SSIM works on identical images
    # -------------------------------------------------------------------------
    def test_ssim_identical_images(self):
        # Identical images must return 1.0
        ssim_ident = calculate_ssim(self.clear_uint8, self.clear_uint8, max_val=255.0)
        self.assertAlmostEqual(ssim_ident, 1.0, places=5)

        # Distorted image must have SSIM < 1.0
        distorted = np.clip(self.clear_uint8.astype(int) + 30, 0, 255).astype(np.uint8)
        ssim_val = calculate_ssim(distorted, self.clear_uint8, max_val=255.0)
        self.assertLess(ssim_val, 1.0)
        self.assertGreater(ssim_val, 0.0)

    # -------------------------------------------------------------------------
    # 9. Basic baseline pipeline completes on provided sample
    # -------------------------------------------------------------------------
    def test_basic_baseline_pipeline_completes_on_sample(self):
        sample_path = SAMPLES_DATA_DIR / "sample_haze.png"
        self.assertTrue(sample_path.is_file(), f"Sample image not found: {sample_path}")

        img = load_image(sample_path)
        # Run on a 256x256 crop for fast unit test execution
        crop = img[:256, :256, :]
        dehazed, meta = dcp_dehaze(crop, patch_size=15, use_guided_filter=True)

        self.assertEqual(dehazed.shape, crop.shape)
        self.assertIn("atmospheric_light", meta)
        self.assertIn("refined_transmission", meta)
        self.assertEqual(len(meta["atmospheric_light"]), 3)


if __name__ == "__main__":
    unittest.main()
