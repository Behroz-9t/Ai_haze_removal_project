"""Lightweight U-Net-Style Neural Architecture for Image Dehazing.

Implements an efficient encoder-decoder network with skip connections,
designed specifically for CPU-friendly experimentation and fast inference.
Features residual radiance learning and spatial reflection padding to handle
arbitrary image resolutions without tensor dimension mismatches.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DoubleConv(nn.Module):
    """Two consecutive 3x3 convolutions with LeakyReLU activations and GroupNorm."""

    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 8) -> None:
        super().__init__()
        # Determine valid group count for GroupNorm
        groups = num_groups if out_channels % num_groups == 0 else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DownBlock(nn.Module):
    """Downsampling via MaxPool2d followed by DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Upsampling followed by feature concatenation with skip connection and DoubleConv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2,
        )
        self.conv = DoubleConv((in_channels // 2) + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x_up = self.up(x)

        # Handle potential 1-pixel rounding discrepancies when spatial dims are odd
        diff_y = skip.size()[2] - x_up.size()[2]
        diff_x = skip.size()[3] - x_up.size()[3]

        if diff_y != 0 or diff_x != 0:
            x_up = F.pad(
                x_up,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )

        cat_features = torch.cat([skip, x_up], dim=1)
        return self.conv(cat_features)


class LightweightUNet(nn.Module):
    """Lightweight U-Net for Single Image Visibility Enhancement / Dehazing.

    Progression:
        Input: (B, 3, H, W)
        Encoder 1: base_channels (e.g. 24)
        Encoder 2: base_channels * 2 (e.g. 48)
        Encoder 3: base_channels * 4 (e.g. 96)
        Bottleneck: base_channels * 8 (e.g. 192)
        Decoder 3: base_channels * 4 (e.g. 96)
        Decoder 2: base_channels * 2 (e.g. 48)
        Decoder 1: base_channels (e.g. 24)
        Output: (B, 3, H, W) in [0.0, 1.0]

    Args:
        in_channels: Input image channel count (default 3 for RGB).
        out_channels: Output image channel count (default 3 for RGB).
        base_channels: Starting feature width (default 24 for CPU speed).
        residual: If True, learns residual adjustment Delta and adds to input: clamp(x + Delta, 0, 1).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 24,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.residual = residual

        b = base_channels

        # Encoder path
        self.enc1 = DoubleConv(in_channels, b)
        self.down1 = DownBlock(b, b * 2)
        self.down2 = DownBlock(b * 2, b * 4)

        # Bottleneck
        self.bottleneck = DownBlock(b * 4, b * 8)

        # Decoder path
        self.up3 = UpBlock(in_channels=b * 8, skip_channels=b * 4, out_channels=b * 4)
        self.up2 = UpBlock(in_channels=b * 4, skip_channels=b * 2, out_channels=b * 2)
        self.up1 = UpBlock(in_channels=b * 2, skip_channels=b, out_channels=b)

        # Final projection to 3 RGB channels
        self.final_conv = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with spatial padding to nearest multiple of 8 and residual clamping.

        Args:
            x: Input tensor of shape (B, C, H, W) normalized to [0.0, 1.0].

        Returns:
            torch.Tensor: Dehazed tensor of shape (B, C, H, W) in [0.0, 1.0].
        """
        _, _, h, w = x.shape

        # Calculate padding needed to ensure dimensions are divisible by 8 (3 pooling stages)
        pad_h = (8 - (h % 8)) % 8
        pad_w = (8 - (w % 8)) % 8

        if pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            x_padded = x

        # Encoder
        e1 = self.enc1(x_padded)
        e2 = self.down1(e1)
        e3 = self.down2(e2)

        # Bottleneck
        b = self.bottleneck(e3)

        # Decoder with skip connections
        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        raw_out = self.final_conv(d1)

        # Unpad back to original spatial dimensions
        if pad_h > 0 or pad_w > 0:
            raw_out = raw_out[:, :, :h, :w]

        # Scene radiance recovery: residual addition or direct bounded prediction
        if self.residual:
            # Learned transmission / atmospheric light residual
            out = torch.clamp(x + raw_out, 0.0, 1.0)
        else:
            out = torch.clamp(torch.sigmoid(raw_out), 0.0, 1.0)

        return out

    def count_parameters(self) -> int:
        """Return the total number of trainable model parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# -----------------------------------------------------------------------------
# Checkpoint Helpers
# -----------------------------------------------------------------------------
def save_checkpoint(
    state: Dict[str, Any],
    filepath: Union[str, Path],
) -> Path:
    """Save model and training state checkpoint to disk.

    Args:
        state: Dictionary with keys 'model_state_dict', 'epoch', 'config', etc.
        filepath: Target checkpoint path.

    Returns:
        Path: Resolved checkpoint file path.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))
    logger.info(f"Saved checkpoint to: {path}")
    return path


def load_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Load model and optional optimizer state from checkpoint.

    Args:
        filepath: Checkpoint path.
        model: Target model instance to load weights into.
        optimizer: Optional optimizer to restore state into.
        map_location: Device to map tensors onto (default 'cpu').

    Returns:
        dict: Complete loaded checkpoint dictionary.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    checkpoint = torch.load(str(path), map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    logger.info(
        f"Loaded checkpoint '{path.name}' (Epoch: {checkpoint.get('epoch', 'N/A')}, "
        f"Val Loss: {checkpoint.get('val_loss', 'N/A')})"
    )
    return checkpoint
