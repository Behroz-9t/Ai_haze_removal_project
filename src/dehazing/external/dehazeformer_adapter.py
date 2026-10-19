"""Adapter interface for DehazeFormer-MCT inference.

Integrates the external pretrained DehazeFormer-MCT model into the project's
evaluation pipeline and Streamlit UI without modifying the existing
Lightweight U-Net or Classical DCP baseline implementations.
"""

from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from src.dehazing.external.dehazeformer import MCT
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DehazeFormerAdapter:
    """Inference wrapper for pretrained DehazeFormer-MCT."""

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: str = "cpu",
    ) -> None:
        """Initialize and load the DehazeFormer-MCT model.

        Args:
            checkpoint_path: Path to dehazeformer_mct.pth weight file.
            device: Target device string ('cpu' or 'cuda').
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"DehazeFormer checkpoint not found at: {self.checkpoint_path}")

        self.model = MCT().to(self.device)
        raw_chk = torch.load(str(self.checkpoint_path), map_location=self.device, weights_only=False)

        state_dict = raw_chk.get("state_dict", raw_chk)
        # Strip unexpected module prefixes if saved under DataParallel
        cleaned_state = {}
        for k, v in state_dict.items():
            clean_k = k.replace("module.", "")
            cleaned_state[clean_k] = v

        self.model.load_state_dict(cleaned_state, strict=True)
        self.model.eval()
        logger.info(f"Loaded DehazeFormer-MCT successfully from {self.checkpoint_path.name}")

    def count_parameters(self) -> int:
        """Count total trainable parameters in the model."""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def preprocess(self, image_rgb: np.ndarray) -> torch.Tensor:
        """Convert uint8 [0, 255] RGB NumPy array of shape (H, W, 3) to float32 [-1.0, 1.0] tensor of shape (1, 3, H, W)."""
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected RGB image of shape (H, W, 3), got {image_rgb.shape}")
        img_f32 = (image_rgb.astype(np.float32) / 255.0) * 2.0 - 1.0
        return torch.from_numpy(img_f32).permute((2, 0, 1)).unsqueeze(0).to(self.device)

    def postprocess(self, tensor_out: torch.Tensor) -> np.ndarray:
        """Convert float32 [-1.0, 1.0] tensor of shape (3, H, W) or (1, 3, H, W) to uint8 [0, 255] RGB NumPy array."""
        t = tensor_out
        if t.ndim == 4:
            t = t[0]
        clamped = t.clamp(-1.0, 1.0) * 0.5 + 0.5
        clamped_np = clamped.permute((1, 2, 0)).cpu().numpy()
        return np.clip(np.round(clamped_np * 255.0), 0, 255).astype(np.uint8)

    def dehaze(
        self,
        image_rgb: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Perform dehazing on an input RGB image with fine-grained latency profiling.

        Args:
            image_rgb: Input image as RGB NumPy uint8 array of shape (H, W, 3).

        Returns:
            Tuple of:
                - Dehazed RGB image as NumPy uint8 array of shape (H, W, 3).
                - Dictionary containing stage latencies in milliseconds:
                  {'pre_ms': float, 'infer_ms': float, 'post_ms': float, 'total_ms': float}.
        """
        # 1. Preprocessing stage
        t0_pre = time.perf_counter()
        t_in = self.preprocess(image_rgb)
        t_pre = (time.perf_counter() - t0_pre) * 1000.0

        # 2. Neural forward pass stage
        t0_infer = time.perf_counter()
        with torch.no_grad():
            t_out = self.model(t_in)
        t_infer = (time.perf_counter() - t0_infer) * 1000.0

        # 3. Postprocessing stage
        t0_post = time.perf_counter()
        dehazed_rgb = self.postprocess(t_out)
        t_post = (time.perf_counter() - t0_post) * 1000.0

        total_ms = t_pre + t_infer + t_post

        latency_breakdown = {
            "pre_ms": round(t_pre, 2),
            "infer_ms": round(t_infer, 2),
            "post_ms": round(t_post, 2),
            "total_ms": round(total_ms, 2),
        }

        return dehazed_rgb, latency_breakdown
