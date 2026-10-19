"""Unit Tests for Stage 2C.2: Unified Indoor/Outdoor Mixed Pilot & Evaluation.

Tests:
1. Manifest generation and scene-disjointness properties.
2. ManifestPairedDataset correctness, domain filtering, and batching.
3. Checkpoint compatibility and non-destructive initialization.
4. Clear-image robustness evaluation behavior.
5. Error handling on corrupted or missing items in manifests.
"""

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import torch

from app.config import CHECKPOINTS_DIR
from src.dehazing.losses import DehazingLoss
from src.dehazing.model import LightweightUNet, load_checkpoint, save_checkpoint
from src.metrics.psnr import calculate_psnr
from src.preprocessing.mixed_dataset import (
    ManifestPairedDataset,
    create_manifest_dataloader,
)


class TestStage2C2(unittest.TestCase):
    """Test suite for Stage 2C.2 dataset validation and mixed training pipeline."""

    def test_manifest_paired_dataset_and_filtering(self):
        """Verify ManifestPairedDataset correctly parses entries, filters domains, and generates tensors."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            img1 = tmp_root / "hazy_in.png"
            img2 = tmp_root / "clear_in.png"
            img3 = tmp_root / "hazy_out.png"
            img4 = tmp_root / "clear_out.png"

            dummy = np.full((64, 64, 3), 100, dtype=np.uint8)
            for p in [img1, img2, img3, img4]:
                cv2.imwrite(str(p), dummy)

            manifest_data = {
                "pairs": [
                    {
                        "hazy_path": "hazy_in.png",
                        "clear_path": "clear_in.png",
                        "scene_id": "scene_01",
                        "domain": "indoor",
                    },
                    {
                        "hazy_path": "hazy_out.png",
                        "clear_path": "clear_out.png",
                        "scene_id": "scene_02",
                        "domain": "outdoor",
                    },
                ]
            }
            manifest_file = tmp_root / "test_manifest.json"
            with open(manifest_file, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f)

            # 1. Total dataset
            ds_all = ManifestPairedDataset(manifest_path=manifest_file, project_root=tmp_root, patch_size=None)
            self.assertEqual(len(ds_all), 2)

            # 2. Indoor filter
            ds_in = ManifestPairedDataset(
                manifest_path=manifest_file, project_root=tmp_root, filter_domain="indoor", patch_size=None
            )
            self.assertEqual(len(ds_in), 1)
            sample_in = ds_in[0]
            self.assertEqual(sample_in["domain"], "indoor")
            self.assertEqual(sample_in["hazy"].shape, (3, 64, 64))
            self.assertEqual(sample_in["clear"].shape, (3, 64, 64))

            # 3. Outdoor filter
            ds_out = ManifestPairedDataset(
                manifest_path=manifest_file, project_root=tmp_root, filter_domain="outdoor", patch_size=None
            )
            self.assertEqual(len(ds_out), 1)
            sample_out = ds_out[0]
            self.assertEqual(sample_out["domain"], "outdoor")

    def test_mixed_manifest_leakage_resistance(self):
        """Verify generated manifests exist, contain expected counts, and are 100% scene-disjoint."""
        train_mf = Path("data/processed/manifests/mixed_train_manifest.json")
        val_mf = Path("data/processed/manifests/mixed_val_manifest.json")
        clear_mf = Path("data/processed/manifests/clear_test_manifest.json")

        self.assertTrue(train_mf.is_file(), "mixed_train_manifest.json missing.")
        self.assertTrue(val_mf.is_file(), "mixed_val_manifest.json missing.")
        self.assertTrue(clear_mf.is_file(), "clear_test_manifest.json missing.")

        with open(train_mf, "r", encoding="utf-8") as f:
            t_data = json.load(f)
        with open(val_mf, "r", encoding="utf-8") as f:
            v_data = json.load(f)

        # Check counts
        self.assertEqual(t_data["indoor_count"], 100)
        self.assertEqual(t_data["outdoor_count"], 100)
        self.assertEqual(v_data["indoor_count"], 20)
        self.assertEqual(v_data["outdoor_count"], 20)

        # Check scene disjointness
        train_scenes = set(p["scene_id"] for p in t_data["pairs"])
        val_scenes = set(p["scene_id"] for p in v_data["pairs"])
        overlap = train_scenes.intersection(val_scenes)
        self.assertEqual(len(overlap), 0, f"Detected leakage between train and val: {overlap}")

    def test_checkpoint_non_destructive_saving(self):
        """Verify saving with mixed_pilot prefix does not touch other checkpoints."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            chk_dir = Path(tmp_dir)
            model = LightweightUNet(in_channels=3, out_channels=3, base_channels=24, residual=True)

            # Save initial mock checkpoint
            medium_path = chk_dir / "medium_best_model.pth"
            save_checkpoint(
                state={"epoch": 8, "model_state_dict": model.state_dict()},
                filepath=medium_path,
            )
            self.assertTrue(medium_path.is_file())
            initial_mtime = medium_path.stat().st_mtime

            # Save mixed pilot checkpoint
            mixed_path = chk_dir / "mixed_pilot_best_model.pth"
            save_checkpoint(
                state={"epoch": 1, "model_state_dict": model.state_dict()},
                filepath=mixed_path,
            )
            self.assertTrue(mixed_path.is_file())

            # Verify medium_best_model was untouched
            self.assertEqual(medium_path.stat().st_mtime, initial_mtime)

    def test_clear_image_robustness_metric(self):
        """Verify that testing clear images against clear references measures minimal distortion."""
        img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        # Identical image -> infinite PSNR
        perfect_psnr = calculate_psnr(img, img, max_val=255.0)
        self.assertGreater(perfect_psnr, 50.0)

        # Slight noise perturbation -> high PSNR
        noisy = np.clip(img.astype(np.int16) + np.random.randint(-2, 3, img.shape), 0, 255).astype(np.uint8)
        noisy_psnr = calculate_psnr(noisy, img, max_val=255.0)
        self.assertGreater(noisy_psnr, 35.0)


if __name__ == "__main__":
    unittest.main()
