"""Unit Tests for Stage 2C.1: Unified Dehazing Preparation and Testing Suite.

Tests:
1. Checkpoint inventory integrity and LightweightUNet compatibility.
2. Atmospheric haze density and presence estimation.
3. Enhanced PairedHazeDataset pairing with nested directories and complex filenames.
4. Memory-safe offline video dehazing processor.
5. Balanced MixedDomainDataset domain alternation.
"""

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from app.config import CHECKPOINTS_DIR
from src.dehazing.model import LightweightUNet, load_checkpoint
from src.detection.density import estimate_haze_density
from src.preprocessing.dataset import PairedHazeDataset
from src.preprocessing.mixed_dataset import MixedDomainDataset
from src.video.processor import process_video_file


class TestStage2C1(unittest.TestCase):
    """Test suite for Stage 2C.1 capabilities."""

    def test_checkpoint_inventory_and_compatibility(self):
        """Verify that all existing checkpoints in models/checkpoints load into LightweightUNet."""
        chks = list(CHECKPOINTS_DIR.glob("*.pth"))
        self.assertGreater(len(chks), 0, "No checkpoint files found in models/checkpoints.")

        for chk_path in chks:
            model = LightweightUNet(in_channels=3, out_channels=3, base_channels=24, residual=True)
            res = load_checkpoint(chk_path, model=model, map_location="cpu")
            self.assertIn("epoch", res, f"Checkpoint {chk_path.name} missing epoch key.")
            # Test a forward pass
            dummy_in = torch.zeros(1, 3, 64, 64)
            with torch.no_grad():
                out = model(dummy_in)
            self.assertEqual(out.shape, (1, 3, 64, 64))

    def test_haze_density_diagnostics(self):
        """Verify that estimate_haze_density distinguishes clear vs hazy images."""
        # Pristine high-contrast dark image (clear)
        clear_img = np.zeros((100, 100, 3), dtype=np.uint8)
        clear_img[20:80, 20:80] = 255
        diag_clear = estimate_haze_density(clear_img)
        self.assertEqual(diag_clear["severity_label"], "Clear / Near-Clear")
        self.assertFalse(diag_clear["is_likely_hazy"])

        # Washed-out hazy image (elevated dark channel)
        hazy_img = np.full((100, 100, 3), 200, dtype=np.uint8)
        diag_hazy = estimate_haze_density(hazy_img)
        self.assertTrue(diag_hazy["is_likely_hazy"])
        self.assertIn("Haze", diag_hazy["severity_label"])

    def test_nested_discovery_and_complex_pairing(self):
        """Verify PairedHazeDataset discovers nested directories and multi-underscore stems."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            clear_dir = root / "clear"
            hazy_dir = root / "haze"
            part1_dir = hazy_dir / "part1"
            part2_dir = hazy_dir / "part2"

            clear_dir.mkdir()
            part1_dir.mkdir(parents=True)
            part2_dir.mkdir(parents=True)

            # Create clear images with multiple underscores
            dummy_img = np.full((32, 32, 3), 128, dtype=np.uint8)
            cv2.imwrite(str(clear_dir / "outdoor_park_01.png"), dummy_img)
            cv2.imwrite(str(clear_dir / "city_street_02.png"), dummy_img)

            # Create hazy variants inside nested subdirectories with multi-part suffixes
            cv2.imwrite(str(part1_dir / "outdoor_park_01_0.08_0.85.png"), dummy_img)
            cv2.imwrite(str(part2_dir / "city_street_02_1.png"), dummy_img)

            ds = PairedHazeDataset(root_dir=root, strict_pairs=True)
            self.assertEqual(len(ds), 2, "Failed to pair across nested subdirectories with complex stems.")
            matched_clear_stems = sorted([c.stem for _, c, _ in ds.pairs])
            self.assertEqual(matched_clear_stems, ["city_street_02", "outdoor_park_01"])

    def test_offline_video_processor(self):
        """Verify frame-by-frame offline video processing on synthetic video."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            in_video = tmp_path / "test_input.mp4"
            out_video = tmp_path / "test_output.mp4"

            # Create a 5-frame synthetic video
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(in_video), fourcc, 10.0, (64, 48))
            for i in range(5):
                frame = np.full((48, 64, 3), i * 40, dtype=np.uint8)
                writer.write(frame)
            writer.release()

            # Process using classical DCP
            res_dcp = process_video_file(
                input_path=in_video,
                output_path=out_video,
                method="dcp",
                preserve_audio=False,
            )
            self.assertEqual(res_dcp["status"], "success")
            self.assertEqual(res_dcp["frames_processed"], 5)
            self.assertTrue(out_video.is_file())
            self.assertGreater(out_video.stat().st_size, 0)

            # Process using lightweight model
            model = LightweightUNet(in_channels=3, out_channels=3, base_channels=24, residual=True)
            out_video_ai = tmp_path / "test_output_ai.mp4"
            res_ai = process_video_file(
                input_path=in_video,
                output_path=out_video_ai,
                model=model,
                method="ai",
                preserve_audio=False,
            )
            self.assertEqual(res_ai["status"], "success")
            self.assertEqual(res_ai["frames_processed"], 5)
            self.assertTrue(out_video_ai.is_file())

    def test_mixed_domain_dataset_alternation(self):
        """Verify MixedDomainDataset alternates domains appropriately."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            in_c = root / "indoor_clear"
            in_h = root / "indoor_haze"
            out_c = root / "outdoor_clear"
            out_h = root / "outdoor_haze"

            for d in [in_c, in_h, out_c, out_h]:
                d.mkdir()

            dummy = np.full((32, 32, 3), 100, dtype=np.uint8)
            cv2.imwrite(str(in_c / "room_01.png"), dummy)
            cv2.imwrite(str(in_h / "room_01_1.png"), dummy)

            cv2.imwrite(str(out_c / "forest_01.png"), dummy)
            cv2.imwrite(str(out_h / "forest_01_1.png"), dummy)

            ds_in = PairedHazeDataset(clear_dir=in_c, hazy_dir=in_h, strict_pairs=True)
            ds_out = PairedHazeDataset(clear_dir=out_c, hazy_dir=out_h, strict_pairs=True)

            mixed_ds = MixedDomainDataset(indoor_dataset=ds_in, outdoor_dataset=ds_out)
            self.assertEqual(len(mixed_ds), 2)
            sample0 = mixed_ds[0]
            sample1 = mixed_ds[1]
            self.assertEqual(sample0["domain"], "indoor")
            self.assertEqual(sample1["domain"], "outdoor")


if __name__ == "__main__":
    unittest.main()
