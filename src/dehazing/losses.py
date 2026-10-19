"""Loss Functions for Image Dehazing Model Training.

Focuses on L1 reconstruction loss between network prediction and clean reference radiance.
L1 loss preserves edge contrast and sharp boundaries without penalizing high-frequency
differences quadratically like L2/MSE, which prevents blurry or over-smoothed restorations.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DehazingLoss(nn.Module):
    """Primary reconstruction loss for visibility enhancement.

    Computes L1 pixel loss:
        L = mean(|predicted - target|)

    Args:
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.l1_loss = nn.L1Loss(reduction=reduction)

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute L1 loss.

        Args:
            predicted: Model output tensor of shape (B, C, H, W) in [0.0, 1.0].
            target: Ground truth clear image tensor of shape (B, C, H, W) in [0.0, 1.0].

        Returns:
            torch.Tensor: Scalar loss tensor.
        """
        if predicted.shape != target.shape:
            raise ValueError(
                f"Shape mismatch in loss calculation: predicted is {predicted.shape}, "
                f"target is {target.shape}"
            )
        return self.l1_loss(predicted, target)
