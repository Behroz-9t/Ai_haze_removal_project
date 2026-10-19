"""Unit and Integration Tests for Stage 2A AI Dehazing Pipeline.

Covers:
1. Model can be instantiated.
2. Model accepts expected tensor shape.
3. Model returns 3-channel output.
4. Output has expected spatial dimensions (including non-multiple of 8).
5. Output contains no NaN/Inf.
6. Loss can be calculated.
7. One training step completes.
8. Checkpoint can be saved.
9. Checkpoint can be loaded.
10. Inference works from a checkpoint.
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.dehazing.losses import DehazingLoss
from src.dehazing.model import (
    LightweightUNet,
    load_checkpoint,
    save_checkpoint,
)


class TestStage2DehazingModel(unittest.TestCase):
    """Test suite for LightweightUNet, loss calculation, checkpointing, and inference."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        self.model = LightweightUNet(
            in_channels=3,
            out_channels=3,
            base_channels=16,  # Small base channels for ultra-fast test execution
            residual=True,
        ).to(self.device)

    # -------------------------------------------------------------------------
    # 1. Model can be instantiated
    # -------------------------------------------------------------------------
    def test_model_instantiation(self):
        self.assertIsNotNone(self.model)
        params = self.model.count_parameters()
        self.assertGreater(params, 0)
        self.assertTrue(hasattr(self.model, "enc1"))
        self.assertTrue(hasattr(self.model, "bottleneck"))

    # -------------------------------------------------------------------------
    # 2. Model accepts expected tensor shape
    # -------------------------------------------------------------------------
    def test_model_accepts_tensor_shape(self):
        x = torch.rand(2, 3, 64, 64)
        out = self.model(x)
        self.assertIsInstance(out, torch.Tensor)

    # -------------------------------------------------------------------------
    # 3. Model returns 3-channel output
    # -------------------------------------------------------------------------
    def test_model_returns_three_channels(self):
        x = torch.rand(1, 3, 64, 64)
        out = self.model(x)
        self.assertEqual(out.shape[1], 3)

    # -------------------------------------------------------------------------
    # 4. Output has expected spatial dimensions (including odd dimensions)
    # -------------------------------------------------------------------------
    def test_output_spatial_dimensions(self):
        # Standard power of 2
        x1 = torch.rand(1, 3, 128, 128)
        self.assertEqual(self.model(x1).shape, (1, 3, 128, 128))

        # Non-multiple of 8 / odd spatial dimensions (tests reflection padding/unpadding)
        x2 = torch.rand(1, 3, 67, 73)
        out2 = self.model(x2)
        self.assertEqual(out2.shape, (1, 3, 67, 73))

    # -------------------------------------------------------------------------
    # 5. Output contains no NaN/Inf and is bounded in [0, 1]
    # -------------------------------------------------------------------------
    def test_output_no_nan_inf_bounded(self):
        x = torch.rand(2, 3, 64, 64)
        out = self.model(x)

        self.assertFalse(torch.isnan(out).any(), "Model output contains NaN values")
        self.assertFalse(torch.isinf(out).any(), "Model output contains Inf values")
        self.assertTrue((out >= 0.0).all(), "Model output has values < 0.0")
        self.assertTrue((out <= 1.0).all(), "Model output has values > 1.0")

    # -------------------------------------------------------------------------
    # 6. Loss can be calculated
    # -------------------------------------------------------------------------
    def test_loss_calculation(self):
        criterion = DehazingLoss()
        pred = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)

        loss = criterion(pred, target)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.ndim, 0)  # Scalar
        self.assertGreaterEqual(loss.item(), 0.0)

    # -------------------------------------------------------------------------
    # 7. One training step completes
    # -------------------------------------------------------------------------
    def test_one_training_step(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = DehazingLoss()

        x = torch.rand(2, 3, 64, 64)
        target = torch.rand(2, 3, 64, 64)

        optimizer.zero_grad()
        out = self.model(x)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        self.assertTrue(loss.item() >= 0.0)

    # -------------------------------------------------------------------------
    # 8. Checkpoint can be saved
    # -------------------------------------------------------------------------
    def test_checkpoint_saving(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chk_path = Path(tmpdir) / "test_model.pth"
            state = {
                "epoch": 1,
                "model_state_dict": self.model.state_dict(),
                "val_loss": 0.045,
            }
            saved_path = save_checkpoint(state, chk_path)
            self.assertTrue(saved_path.is_file())
            self.assertGreater(saved_path.stat().st_size, 0)

    # -------------------------------------------------------------------------
    # 9. Checkpoint can be loaded
    # -------------------------------------------------------------------------
    def test_checkpoint_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chk_path = Path(tmpdir) / "test_model.pth"
            state = {
                "epoch": 3,
                "model_state_dict": self.model.state_dict(),
                "val_loss": 0.0123,
            }
            save_checkpoint(state, chk_path)

            fresh_model = LightweightUNet(
                in_channels=3,
                out_channels=3,
                base_channels=16,
            )
            loaded_state = load_checkpoint(chk_path, model=fresh_model)

            self.assertEqual(loaded_state["epoch"], 3)
            self.assertAlmostEqual(loaded_state["val_loss"], 0.0123)

    # -------------------------------------------------------------------------
    # 10. Inference works from a checkpoint
    # -------------------------------------------------------------------------
    def test_inference_from_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chk_path = Path(tmpdir) / "best_model.pth"
            save_checkpoint(
                {
                    "epoch": 1,
                    "model_state_dict": self.model.state_dict(),
                    "config": {"base_channels": 16, "residual": True},
                },
                chk_path,
            )

            # Re-instantiate and run inference
            eval_model = LightweightUNet(base_channels=16, residual=True)
            load_checkpoint(chk_path, eval_model)
            eval_model.eval()

            test_input = torch.rand(1, 3, 48, 48)
            with torch.no_grad():
                res = eval_model(test_input)

            self.assertEqual(res.shape, (1, 3, 48, 48))
            self.assertFalse(torch.isnan(res).any())


if __name__ == "__main__":
    unittest.main()
