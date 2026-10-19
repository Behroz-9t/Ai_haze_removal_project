"""Mixed-Domain Unified Indoor/Outdoor Training Script (Stage 2C.1 Preparation).

Fine-tunes the LightweightUNet architecture on both indoor (RESIDE ITS) and
outdoor (RESIDE OTS) paired datasets simultaneously with balanced domain sampling.
Initializes from a copy of `medium_best_model.pth` and saves new checkpoints with
the `mixed_` prefix, guaranteeing that `medium_best_model.pth` is never overwritten.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from app.config import (
    CHECKPOINTS_DIR,
    DEFAULT_RANDOM_SEED,
    OUTPUTS_DIR,
    get_default_device,
)
from src.dehazing.losses import DehazingLoss
from src.dehazing.model import LightweightUNet, load_checkpoint, save_checkpoint
from src.metrics.psnr import calculate_psnr
from src.metrics.ssim import calculate_ssim
from src.preprocessing.mixed_dataset import (
    create_manifest_dataloader,
    create_mixed_dataloader,
)
from src.utils.image_ops import tensor_to_numpy
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger("train_mixed")


def train_mixed_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    """Execute one training epoch across balanced mixed-domain batches."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        hazy = batch["hazy"].to(device)
        clear = batch["clear"].to(device)

        optimizer.zero_grad()
        predicted = model(hazy)
        loss = criterion(predicted, clear)
        loss_val = loss.item()

        if not np.isfinite(loss_val):
            raise ValueError(f"Non-finite loss ({loss_val}) at batch {num_batches + 1}.")

        loss.backward()
        optimizer.step()

        total_loss += loss_val
        num_batches += 1

    return total_loss / max(1, num_batches)


def evaluate_domain(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluate model on a domain-specific validation set."""
    model.eval()
    total_loss = 0.0
    psnr_scores: List[float] = []
    ssim_scores: List[float] = []
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            hazy = batch["hazy"].to(device)
            clear = batch["clear"].to(device)

            predicted = model(hazy)
            loss = criterion(predicted, clear)
            total_loss += loss.item()
            num_batches += 1

            for b in range(predicted.shape[0]):
                pred_np = tensor_to_numpy(predicted[b])
                clear_np = tensor_to_numpy(clear[b])
                psnr_val = calculate_psnr(pred_np, clear_np, max_val=255.0)
                ssim_val = calculate_ssim(pred_np, clear_np, max_val=255.0)

                if np.isfinite(psnr_val):
                    psnr_scores.append(psnr_val)
                if np.isfinite(ssim_val):
                    ssim_scores.append(ssim_val)

    mean_loss = total_loss / max(1, num_batches)
    mean_psnr = float(np.mean(psnr_scores)) if psnr_scores else 0.0
    mean_ssim = float(np.mean(ssim_scores)) if ssim_scores else 0.0
    return mean_loss, mean_psnr, mean_ssim


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Indoor/Outdoor Mixed Training")
    parser.add_argument(
        "--train-manifest",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "manifests" / "mixed_train_manifest.json"),
        help="Path to mixed-domain training manifest JSON.",
    )
    parser.add_argument(
        "--val-manifest",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "manifests" / "mixed_val_manifest.json"),
        help="Path to mixed-domain validation manifest JSON.",
    )
    parser.add_argument(
        "--indoor-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "RESIDE" / "ITS" / "train"),
        help="Fallback indoor directory if manifests are not used.",
    )
    parser.add_argument(
        "--outdoor-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw" / "RESIDE" / "OTS"),
        help="Fallback outdoor directory if manifests are not used.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=str(CHECKPOINTS_DIR / "medium_best_model.pth"),
        help="Base checkpoint to initialize weights from (copied, not overwritten).",
    )
    parser.add_argument(
        "--checkpoint-prefix",
        type=str,
        default="mixed_pilot",
        help="Prefix for saved checkpoints (e.g. 'mixed_pilot' -> 'mixed_pilot_best_model.pth').",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of fine-tuning epochs (default: 5 for pilot).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Mini-batch size (default: 4 for CPU).",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=128,
        help="Patch resolution for training (default: 128).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate for fine-tuning (default: 5e-5, conservative for fine-tuning).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Deterministic random seed.",
    )

    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(get_default_device())

    train_mf = Path(args.train_manifest)
    val_mf = Path(args.val_manifest)

    print("=" * 70)
    print("STAGE 2C.2: UNIFIED INDOOR/OUTDOOR MIXED PILOT TRAINING")
    print("=" * 70)
    print(f"Training Manifest:      {train_mf}")
    print(f"Validation Manifest:    {val_mf}")
    print(f"Initialization Weights: {args.init_checkpoint}")
    print(f"Target Prefix:          {args.checkpoint_prefix}")
    print(f"Epochs:                 {args.epochs}")
    print(f"Batch Size:             {args.batch_size}")
    print(f"Patch Size:             {args.patch_size}x{args.patch_size}")
    print(f"Learning Rate:          {args.lr}")
    print(f"Device:                 {device}")
    print("-" * 70)

    # 1. DataLoaders
    if train_mf.is_file() and val_mf.is_file():
        print(f"Loading training pairs from manifest: {train_mf.name}")
        train_loader = create_manifest_dataloader(
            manifest_path=train_mf,
            batch_size=args.batch_size,
            patch_size=(args.patch_size, args.patch_size),
            random_crop=True,
            shuffle=True,
        )
        print(f"Loading validation pairs from manifest: {val_mf.name}")
        val_loader_indoor = create_manifest_dataloader(
            manifest_path=val_mf,
            batch_size=args.batch_size,
            patch_size=(args.patch_size, args.patch_size),
            random_crop=False,
            filter_domain="indoor",
            shuffle=False,
        )
        val_loader_outdoor = create_manifest_dataloader(
            manifest_path=val_mf,
            batch_size=args.batch_size,
            patch_size=(args.patch_size, args.patch_size),
            random_crop=False,
            filter_domain="outdoor",
            shuffle=False,
        )
        print(f"Train Pairs: {len(train_loader.dataset)} | Val Indoor: {len(val_loader_indoor.dataset)} | Val Outdoor: {len(val_loader_outdoor.dataset)}")
    else:
        print("Manifests not found, falling back to directory-based mixed loader...")
        train_loader = create_mixed_dataloader(
            indoor_dir=args.indoor_dir,
            outdoor_dir=args.outdoor_dir,
            batch_size=args.batch_size,
            patch_size=(args.patch_size, args.patch_size),
            random_crop=True,
            seed=args.seed,
        )
        val_loader_indoor = None
        val_loader_outdoor = None

    # 2. Model Initialization
    model = LightweightUNet(in_channels=3, out_channels=3, base_channels=24, residual=True).to(device)
    init_chk = Path(args.init_checkpoint)
    if init_chk.is_file():
        chk_data = load_checkpoint(init_chk, model=model, map_location=str(device))
        print(f"Successfully loaded initialization weights from: {init_chk.name}")
        print(f"Base checkpoint trained epoch: {chk_data.get('epoch', 'N/A')}")
    else:
        print(f"[!] Warning: Checkpoint {init_chk} not found, initializing from scratch.")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = DehazingLoss()

    best_composite_loss = float("inf")
    history: List[Dict[str, Any]] = []
    training_outputs_dir = OUTPUTS_DIR / "training"
    training_outputs_dir.mkdir(parents=True, exist_ok=True)

    print("-" * 70)
    print("Beginning mixed-domain fine-tuning...")
    try:
        for epoch in range(1, args.epochs + 1):
            t0 = time.perf_counter()
            train_loss = train_mixed_epoch(model, train_loader, optimizer, criterion, device)

            # Validation tracking
            val_in_loss, val_in_psnr, val_in_ssim = (0.0, 0.0, 0.0)
            val_out_loss, val_out_psnr, val_out_ssim = (0.0, 0.0, 0.0)

            if val_loader_indoor is not None and len(val_loader_indoor.dataset) > 0:
                val_in_loss, val_in_psnr, val_in_ssim = evaluate_domain(model, val_loader_indoor, criterion, device)
            if val_loader_outdoor is not None and len(val_loader_outdoor.dataset) > 0:
                val_out_loss, val_out_psnr, val_out_ssim = evaluate_domain(model, val_loader_outdoor, criterion, device)

            composite_val_loss = 0.5 * (val_in_loss + val_out_loss) if (val_loader_indoor and val_loader_outdoor) else train_loss
            dur = time.perf_counter() - t0

            epoch_record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "composite_val_loss": round(composite_val_loss, 5),
                "val_indoor_loss": round(val_in_loss, 5),
                "val_indoor_psnr": round(val_in_psnr, 2),
                "val_indoor_ssim": round(val_in_ssim, 4),
                "val_outdoor_loss": round(val_out_loss, 5),
                "val_outdoor_psnr": round(val_out_psnr, 2),
                "val_outdoor_ssim": round(val_out_ssim, 4),
                "duration_seconds": round(dur, 2),
            }
            history.append(epoch_record)

            print(
                f"Epoch {epoch:2d}/{args.epochs} [{dur:.1f}s] | "
                f"Train Loss: {train_loss:.5f} | "
                f"Val In: PSNR {val_in_psnr:.2f}dB (L={val_in_loss:.4f}) | "
                f"Val Out: PSNR {val_out_psnr:.2f}dB (L={val_out_loss:.4f}) | "
                f"Comp Val: {composite_val_loss:.5f}"
            )

            # Checkpoint selection: based on composite validation loss across both domains
            is_best = composite_val_loss < best_composite_loss
            if is_best:
                best_composite_loss = composite_val_loss

            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": composite_val_loss,
                "val_indoor_psnr": val_in_psnr,
                "val_indoor_ssim": val_in_ssim,
                "val_outdoor_psnr": val_out_psnr,
                "val_outdoor_ssim": val_out_ssim,
                "learning_rate": args.lr,
                "seed": args.seed,
                "config": {
                    "base_channels": 24,
                    "residual": True,
                    "patch_size": (args.patch_size, args.patch_size),
                    "batch_size": args.batch_size,
                    "init_checkpoint": Path(args.init_checkpoint).name,
                },
            }

            latest_path = CHECKPOINTS_DIR / f"{args.checkpoint_prefix}_latest_model.pth"
            best_path = CHECKPOINTS_DIR / f"{args.checkpoint_prefix}_best_model.pth"

            # Save latest checkpoint every epoch
            save_checkpoint(state, latest_path)

            # Save best checkpoint whenever composite validation loss improves
            if is_best:
                save_checkpoint(state, best_path)
                print(f"  [BEST] New best checkpoint saved: {best_path.name} (Comp Loss: {best_composite_loss:.5f})")

            # Persist updated history JSON
            history_path = training_outputs_dir / f"{args.checkpoint_prefix}_history.json"
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump({
                    "config": vars(args),
                    "best_composite_loss": best_composite_loss,
                    "history": history,
                }, f, indent=2)

    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user. Latest checkpoint and history preserved.")

    print("=" * 70)
    print(f"Mixed Pilot Training Finished.")
    print(f"  Best Checkpoint:   {CHECKPOINTS_DIR / f'{args.checkpoint_prefix}_best_model.pth'}")
    print(f"  Latest Checkpoint: {CHECKPOINTS_DIR / f'{args.checkpoint_prefix}_latest_model.pth'}")
    print(f"  History Log:       {training_outputs_dir / f'{args.checkpoint_prefix}_history.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()

