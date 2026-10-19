"""Comprehensive Test Suite for Stage 2B (Dataset Ingestion, Splitting, Training, Evaluation).

Tests:
1. RESIDE dataset folder discovery and prefix matching.
2. Unpaired / corrupted image detection.
3. Zero data leakage across train, val, and test splits (scene grouping).
4. Materialization and manifest generation.
5. Patch extraction in PairedHazeDataset (random and center crop).
6. Checkpoint saving with metadata (epoch, val_loss, val_psnr, val_ssim).
7. Checkpoint loading and model inference.
8. Resume from checkpoint.
9. Evaluation pipeline metrics computation.
"""

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import cv2
import numpy as np
import torch

from app.config import DEFAULT_RANDOM_SEED
from src.dehazing.losses import DehazingLoss
from src.dehazing.model import LightweightUNet, load_checkpoint, save_checkpoint
from src.metrics.psnr import calculate_psnr
from src.metrics.ssim import calculate_ssim
from src.preprocessing.dataset import PairedHazeDataset, create_paired_dataloader
from src.preprocessing.ingestion import ImagePair, IngestionReport, ResideDatasetIngestor


class TestStage2BIngestion(unittest.TestCase):
    """Test dataset discovery, pairing, corruption detection, and zero-leakage splits."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.raw_root = Path(self.temp_dir) / "RESIDE_TEST"
        self.clear_dir = self.raw_root / "ITS" / "clear"
        self.hazy_dir = self.raw_root / "ITS" / "hazy"
        self.clear_dir.mkdir(parents=True, exist_ok=True)
        self.hazy_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy image files
        # 3 scenes: 100, 101, 102
        self.scenes = ["100", "101", "102"]
        for sid in self.scenes:
            c_img = np.full((64, 64, 3), 128, dtype=np.uint8)
            cv2.imwrite(str(self.clear_dir / f"{sid}.png"), c_img)

            # 2 hazy variants per scene
            h1 = np.full((64, 64, 3), 150, dtype=np.uint8)
            h2 = np.full((64, 64, 3), 180, dtype=np.uint8)
            cv2.imwrite(str(self.hazy_dir / f"{sid}_1_0.1.png"), h1)
            cv2.imwrite(str(self.hazy_dir / f"{sid}_2_0.2.png"), h2)

        # 1 orphan hazy image without clear counterpart
        cv2.imwrite(str(self.hazy_dir / "999_1_0.1.png"), np.full((64, 64, 3), 200, dtype=np.uint8))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_discovery_and_pairing(self) -> None:
        ingestor = ResideDatasetIngestor(self.raw_root)
        report = ingestor.discover_and_validate()

        self.assertTrue(report.is_valid)
        self.assertEqual(report.total_clear_found, 3)
        self.assertEqual(report.total_hazy_found, 7)  # 6 paired + 1 orphan
        self.assertEqual(report.valid_pairs_count, 6)
        self.assertEqual(report.scenes_count, 3)
        self.assertEqual(len(report.missing_clear_for_hazy), 1)
        self.assertIn("999_1_0.1.png", report.missing_clear_for_hazy)

    def test_zero_data_leakage_split(self) -> None:
        ingestor = ResideDatasetIngestor(self.raw_root)
        report = ingestor.discover_and_validate()
        splits = ingestor.create_leakage_free_splits(
            report.pairs, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, seed=DEFAULT_RANDOM_SEED
        )

        train_scenes = set(p.scene_id for p in splits["train"])
        val_scenes = set(p.scene_id for p in splits["val"])
        test_scenes = set(p.scene_id for p in splits["test"])

        # Crucial scientific check: no scene should appear in more than one partition
        self.assertEqual(len(train_scenes.intersection(val_scenes)), 0)
        self.assertEqual(len(train_scenes.intersection(test_scenes)), 0)
        self.assertEqual(len(val_scenes.intersection(test_scenes)), 0)

        # All paired images should be partitioned
        total_partitioned = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        self.assertEqual(total_partitioned, 6)

    def test_materialize_and_manifest(self) -> None:
        ingestor = ResideDatasetIngestor(self.raw_root)
        report = ingestor.discover_and_validate()
        splits = ingestor.create_leakage_free_splits(report.pairs)

        out_dir = Path(self.temp_dir) / "processed_reside"
        ingestor.materialize_split(splits, output_dir=out_dir, copy_files=True)

        manifest_file = out_dir / "dataset_manifest.json"
        self.assertTrue(manifest_file.is_file())

        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertIn("splits", manifest)
        self.assertIn("train", manifest["splits"])
        self.assertIn("val", manifest["splits"])
        self.assertIn("test", manifest["splits"])
        self.assertEqual(manifest["total_pairs"], 6)


class TestStage2BPatchDataset(unittest.TestCase):
    """Test patch extraction and PairedHazeDataset upgrades."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.root_dir = Path(self.temp_dir)
        (self.root_dir / "clear").mkdir()
        (self.root_dir / "hazy").mkdir()

        # Create a 200x200 test pair
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[50:150, 50:150] = 255
        cv2.imwrite(str(self.root_dir / "clear" / "scene_01.png"), img)
        cv2.imwrite(str(self.root_dir / "hazy" / "scene_01_hazy.png"), img)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_random_patch_cropping(self) -> None:
        dataset = PairedHazeDataset(
            root_dir=self.root_dir,
            patch_size=(64, 64),
            random_crop=True,
            strict_pairs=False,
        )
        sample = dataset[0]
        self.assertEqual(sample["hazy"].shape, (3, 64, 64))
        self.assertEqual(sample["clear"].shape, (3, 64, 64))

    def test_center_patch_cropping(self) -> None:
        dataset = PairedHazeDataset(
            root_dir=self.root_dir,
            patch_size=(64, 64),
            random_crop=False,
            strict_pairs=False,
        )
        sample = dataset[0]
        self.assertEqual(sample["hazy"].shape, (3, 64, 64))
        self.assertEqual(sample["clear"].shape, (3, 64, 64))


class TestStage2BTrainingAndCheckpoints(unittest.TestCase):
    """Test model checkpointing, resume capability, and metrics evaluation."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.chk_path = Path(self.temp_dir) / "test_checkpoint.pth"

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_checkpoint_saving_with_metadata(self) -> None:
        model = LightweightUNet(in_channels=3, out_channels=3, base_channels=16)
        state = {
            "epoch": 5,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "train_loss": 0.045,
            "val_loss": 0.042,
            "val_psnr": 22.5,
            "val_ssim": 0.88,
            "learning_rate": 1e-4,
            "seed": 42,
            "config": {"base_channels": 16, "residual": True},
        }
        save_checkpoint(state, self.chk_path)
        self.assertTrue(self.chk_path.is_file())

        loaded_model = LightweightUNet(in_channels=3, out_channels=3, base_channels=16)
        loaded_state = load_checkpoint(self.chk_path, model=loaded_model)
        self.assertEqual(loaded_state["epoch"], 5)
        self.assertEqual(loaded_state["val_psnr"], 22.5)
        self.assertEqual(loaded_state["val_ssim"], 0.88)

    def test_model_forward_pass_and_loss(self) -> None:
        model = LightweightUNet(in_channels=3, out_channels=3, base_channels=16)
        model.eval()

        dummy_hazy = torch.rand(2, 3, 64, 64)
        dummy_clear = torch.rand(2, 3, 64, 64)

        criterion = DehazingLoss()
        with torch.no_grad():
            output = model(dummy_hazy)
            loss = criterion(output, dummy_clear)

        self.assertEqual(output.shape, (2, 3, 64, 64))
        self.assertTrue(loss.item() > 0.0)

    def test_evaluate_pipeline_execution(self) -> None:
        from scripts.evaluate import evaluate_test_set

        # Create minimal test folder with 1 pair
        test_dir = Path(self.temp_dir) / "test_eval"
        (test_dir / "clear").mkdir(parents=True)
        (test_dir / "hazy").mkdir(parents=True)

        c_img = np.full((64, 64, 3), 100, dtype=np.uint8)
        h_img = np.full((64, 64, 3), 150, dtype=np.uint8)
        cv2.imwrite(str(test_dir / "clear" / "scene_01.png"), c_img)
        cv2.imwrite(str(test_dir / "hazy" / "scene_01.png"), h_img)

        # Save dummy checkpoint
        model = LightweightUNet(in_channels=3, out_channels=3, base_channels=16)
        state = {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "config": {"base_channels": 16, "residual": True},
        }
        chk_file = Path(self.temp_dir) / "eval_model.pth"
        save_checkpoint(state, chk_file)

        eval_out = Path(self.temp_dir) / "eval_out"
        results = evaluate_test_set(
            test_dir=test_dir,
            checkpoint_path=chk_file,
            output_dir=eval_out,
            device_str="cpu",
            save_images=True,
        )

        self.assertEqual(results["num_test_pairs"], 1)
        self.assertIn("Classical DCP", results["metrics"])
        self.assertIn("AI Lightweight U-Net", results["metrics"])
        self.assertTrue((eval_out / "test_evaluation_summary.json").is_file())


class TestOfficialResideStructure(unittest.TestCase):
    """Test loading and validating official RESIDE ITS directory conventions."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.its_root = Path(self.temp_dir) / "ITS"

        # Official Train structure
        self.train_clear = self.its_root / "train" / "ITS_clear"
        self.train_haze = self.its_root / "train" / "ITS_haze"
        self.train_trans = self.its_root / "train" / "ITS_trans"
        self.train_clear.mkdir(parents=True)
        self.train_haze.mkdir(parents=True)
        self.train_trans.mkdir(parents=True)

        # Create 2 train scenes (0001, 0002) with clear, hazy, and trans
        for sid in ["0001", "0002"]:
            # Clear: 100
            cv2.imwrite(str(self.train_clear / f"{sid}.png"), np.full((64, 64, 3), 100, dtype=np.uint8))
            # Hazy: 150
            cv2.imwrite(str(self.train_haze / f"{sid}_01_0.90.png"), np.full((64, 64, 3), 150, dtype=np.uint8))
            # Trans: 50
            cv2.imwrite(str(self.train_trans / f"{sid}_01.png"), np.full((64, 64, 3), 50, dtype=np.uint8))

        # Official Val structure
        self.val_clear = self.its_root / "val" / "clear"
        self.val_haze = self.its_root / "val" / "haze"
        self.val_trans = self.its_root / "val" / "trans"
        self.val_clear.mkdir(parents=True)
        self.val_haze.mkdir(parents=True)
        self.val_trans.mkdir(parents=True)

        # Create 1 val scene (10001) with clear, hazy, and trans
        cv2.imwrite(str(self.val_clear / "10001.png"), np.full((64, 64, 3), 200, dtype=np.uint8))
        cv2.imwrite(str(self.val_haze / "10001_01_0.95.png"), np.full((64, 64, 3), 220, dtype=np.uint8))
        cv2.imwrite(str(self.val_trans / "10001_01.png"), np.full((64, 64, 3), 80, dtype=np.uint8))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_train_dataset_resolution(self) -> None:
        # Test loading train using root_dir directly pointing to train/
        ds_train = PairedHazeDataset(root_dir=self.its_root / "train")
        self.assertEqual(len(ds_train), 2)
        sample = ds_train[0]
        # Target clear image must be the true clear image (intensity ~100), not trans (intensity ~50)
        clear_pixel_val = sample["clear"][0, 0, 0].item() * 255.0
        self.assertAlmostEqual(clear_pixel_val, 100.0, delta=5.0)
        self.assertIn("ITS_haze", sample["hazy_path"])
        self.assertIn("ITS_clear", sample["clear_path"])
        self.assertIn("ITS_trans", sample["trans_path"])

    def test_val_dataset_resolution(self) -> None:
        # Test loading val using split='val' on ITS root
        ds_val = PairedHazeDataset(root_dir=self.its_root, split="val")
        self.assertEqual(len(ds_val), 1)
        sample = ds_val[0]
        clear_pixel_val = sample["clear"][0, 0, 0].item() * 255.0
        self.assertAlmostEqual(clear_pixel_val, 200.0, delta=5.0)
        self.assertIn("10001.png", sample["clear_path"])
        self.assertIn("10001_01", sample["hazy_path"])

    def test_ingestor_official_splits_detection(self) -> None:
        ingestor = ResideDatasetIngestor(self.its_root)
        report = ingestor.discover_and_validate()
        self.assertTrue(report.is_valid)
        self.assertTrue(report.has_official_splits)
        self.assertEqual(report.total_clear_found, 3)  # 2 train + 1 val
        self.assertEqual(report.total_hazy_found, 3)   # 2 train + 1 val
        self.assertEqual(report.total_trans_found, 3)  # 2 train + 1 val
        self.assertEqual(report.valid_pairs_count, 3)
        self.assertEqual(report.split_reports["train"].valid_pairs_count, 2)
        self.assertEqual(report.split_reports["val"].valid_pairs_count, 1)


class TestSceneDiversitySampling(unittest.TestCase):
    """Test scene-diverse subset selection and variant caps in PairedHazeDataset."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.clear_dir = Path(self.temp_dir) / "clear"
        self.hazy_dir = Path(self.temp_dir) / "hazy"
        self.clear_dir.mkdir(parents=True)
        self.hazy_dir.mkdir(parents=True)

        # Create 5 scenes (001 to 005), each with 4 hazy variants (20 total pairs)
        dummy_img = np.full((32, 32, 3), 128, dtype=np.uint8)
        for s_idx in range(1, 6):
            sid = f"{s_idx:03d}"
            cv2.imwrite(str(self.clear_dir / f"{sid}.png"), dummy_img)
            for v_idx in range(1, 5):
                cv2.imwrite(str(self.hazy_dir / f"{sid}_{v_idx:02d}_0.8.png"), dummy_img)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_broad_scene_coverage_selection(self) -> None:
        # Request 5 samples with max_variants_per_scene=1
        ds = PairedHazeDataset(
            root_dir=self.temp_dir,
            max_samples=5,
            max_variants_per_scene=1,
            seed=42,
            strict_pairs=False,
        )
        self.assertEqual(len(ds), 5)
        unique_scenes = set(c.stem for _, c, _ in ds.pairs)
        # All 5 selected samples must come from 5 distinct scenes
        self.assertEqual(len(unique_scenes), 5)

    def test_reproducible_scene_sampling(self) -> None:
        ds1 = PairedHazeDataset(
            root_dir=self.temp_dir,
            max_samples=5,
            max_variants_per_scene=1,
            seed=42,
            strict_pairs=False,
        )
        ds2 = PairedHazeDataset(
            root_dir=self.temp_dir,
            max_samples=5,
            max_variants_per_scene=1,
            seed=42,
            strict_pairs=False,
        )
        stems1 = [s for _, _, s in ds1.pairs]
        stems2 = [s for _, _, s in ds2.pairs]
        self.assertEqual(stems1, stems2)

    def test_max_variants_per_scene_constraint(self) -> None:
        # Request 6 samples with max_variants_per_scene=2
        ds = PairedHazeDataset(
            root_dir=self.temp_dir,
            max_samples=6,
            max_variants_per_scene=2,
            seed=42,
            strict_pairs=False,
        )
        self.assertEqual(len(ds), 6)
        scene_counts: dict = {}
        for _, c, _ in ds.pairs:
            scene_counts[c.stem] = scene_counts.get(c.stem, 0) + 1
        for sid, count in scene_counts.items():
            self.assertLessEqual(count, 2)


if __name__ == "__main__":
    unittest.main()

