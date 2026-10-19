"""Model Training Script for AI-Based Image Dehazing (Stage 2B).

Trains the LightweightUNet architecture on paired hazy/clear images using L1 loss
and CPU-friendly patch-based random cropping.
Evaluates both validation loss, validation PSNR, and validation SSIM after each epoch,
saving best_model.pth and latest_model.pth checkpoints with full resume support.
"""

import argparse
import csv
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is available in module path
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
    PROCESSED_DATA_DIR,
    SAMPLES_DATA_DIR,
    ensure_project_directories,
    get_default_device,
)
from src.dehazing.losses import DehazingLoss
from src.dehazing.model import LightweightUNet, load_checkpoint, save_checkpoint
from src.dehazing.train_config import TrainingConfig
from src.metrics.psnr import calculate_psnr
from src.metrics.ssim import calculate_ssim
from src.preprocessing.dataset import create_paired_dataloader
from src.utils.image_ops import tensor_to_numpy
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger("train")


def get_system_ram_info() -> Dict[str, Any]:
    """Get system RAM statistics on Windows / platform-agnostic fallback."""
    info: Dict[str, Any] = {}
    try:
        import ctypes
        from ctypes import wintypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            info["total_ram_gb"] = round(stat.ullTotalPhys / (1024**3), 2)
            info["avail_ram_gb"] = round(stat.ullAvailPhys / (1024**3), 2)
            info["memory_load_pct"] = int(stat.dwMemoryLoad)
    except Exception:
        pass
    return info
def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    """Execute one training epoch using mini-batch patch sampling."""
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
            raise ValueError(
                f"Encountered non-finite training loss ({loss_val}) at batch {num_batches + 1}. "
                "Halting training to preserve model state."
            )

        loss.backward()
        optimizer.step()

        total_loss += loss_val
        num_batches += 1

    return total_loss / max(1, num_batches)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluate model on the validation dataset.

    Returns:
        tuple (mean_val_loss, mean_val_psnr, mean_val_ssim):
            - mean_val_loss: Average validation L1 loss.
            - mean_val_psnr: Average validation PSNR in dB.
            - mean_val_ssim: Average validation SSIM in [-1..1].
    """
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
            loss_val = loss.item()

            if not np.isfinite(loss_val):
                raise ValueError(
                    f"Encountered non-finite validation loss ({loss_val}) at batch {num_batches + 1}. "
                    "Halting evaluation."
                )

            total_loss += loss_val
            num_batches += 1

            for b in range(predicted.shape[0]):
                pred_np = tensor_to_numpy(predicted[b])
                clear_np = tensor_to_numpy(clear[b])

                psnr_val = calculate_psnr(pred_np, clear_np, max_val=255.0)
                ssim_val = calculate_ssim(pred_np, clear_np, max_val=255.0)

                if not np.isinf(psnr_val) and not np.isnan(psnr_val):
                    psnr_scores.append(psnr_val)
                if not np.isnan(ssim_val):
                    ssim_scores.append(ssim_val)

    mean_loss = total_loss / max(1, num_batches)
    mean_psnr = float(np.mean(psnr_scores)) if psnr_scores else 0.0
    mean_ssim = float(np.mean(ssim_scores)) if ssim_scores else 0.0
    return mean_loss, mean_psnr, mean_ssim


def train_dehazer(
    train_dir: Path,
    val_dir: Path,
    config: TrainingConfig,
) -> Dict[str, Any]:
    """Execute complete training pipeline with patch extraction, validation, and checkpoints."""
    ensure_project_directories()
    set_seed(config.seed)

    device = torch.device(config.device)
    print("=" * 65)
    print("STAGE 2B: LIGHTWEIGHT U-NET DEHAZING MODEL TRAINING")
    print("=" * 65)
    print(f"Compute Device:       {device}")
    print(f"Training Data Dir:    {train_dir}")
    print(f"Validation Data Dir:  {val_dir}")
    print(f"Epochs:               {config.epochs}")
    print(f"Batch Size:           {config.batch_size}")
    print(f"Patch Size (H x W):   {config.patch_size[0]} x {config.patch_size[1]}")
    print(f"Learning Rate:        {config.learning_rate}")
    print(f"Model Base Channels:  {config.base_channels}")
    print(f"Residual Learning:    {config.residual}")
    if config.max_train_samples:
        print(f"Max Train Samples:    {config.max_train_samples}")
    if config.max_val_samples:
        print(f"Max Val Samples:      {config.max_val_samples}")
    if config.resume_path:
        print(f"Resume Checkpoint:    {config.resume_path}")
    print("-" * 65)

    # 1. Patch-Based Datasets and DataLoaders
    train_loader = create_paired_dataloader(
        dataset_or_dir=train_dir,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        patch_size=config.patch_size,
        random_crop=True,
        max_samples=config.max_train_samples,
        max_variants_per_scene=config.max_variants_per_scene,
        seed=config.seed,
        strict_pairs=False,
    )
    val_loader = create_paired_dataloader(
        dataset_or_dir=val_dir,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        patch_size=config.patch_size,
        random_crop=False,
        max_samples=config.max_val_samples,
        max_variants_per_scene=config.max_variants_per_scene,
        seed=config.seed,
        strict_pairs=False,
    )

    train_pairs_count = len(train_loader.dataset)
    val_pairs_count = len(val_loader.dataset)

    if train_pairs_count == 0:
        raise ValueError(f"No valid image pairs found in training directory: {train_dir}")
    if val_pairs_count == 0:
        logger.warning(f"No validation pairs found in {val_dir}; reusing train loader for validation.")
        val_loader = train_loader
        val_pairs_count = train_pairs_count

    # Count unique source scenes in active datasets
    def _count_unique_scenes(ds: Any) -> int:
        if hasattr(ds, "pairs"):
            return len(set(c_path.stem for _, c_path, _ in ds.pairs))
        return 0

    train_scenes_count = _count_unique_scenes(train_loader.dataset)
    val_scenes_count = _count_unique_scenes(val_loader.dataset)

    print(f"Active Train Pairs:   {train_pairs_count} ({train_scenes_count} unique scenes)")
    print(f"Active Val Pairs:     {val_pairs_count} ({val_scenes_count} unique scenes)")

    ram_initial = get_system_ram_info()
    if ram_initial:
        print(f"System RAM (Avail):   {ram_initial.get('avail_ram_gb')} GB / {ram_initial.get('total_ram_gb')} GB ({ram_initial.get('memory_load_pct')}% load)")

    # 2. Model & Optimizer
    model = LightweightUNet(
        in_channels=3,
        out_channels=3,
        base_channels=config.base_channels,
        residual=config.residual,
    ).to(device)

    param_count = model.count_parameters()
    print(f"Model Parameter Count:{param_count:,} ({param_count / 1e6:.2f}M params)")
    print("-" * 65)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = DehazingLoss()

    start_epoch = 1
    best_val_psnr = -float("inf")
    best_val_loss = float("inf")
    best_epoch = 0

    # 3. Resume from checkpoint if specified
    if config.resume_path is not None and Path(config.resume_path).is_file():
        chk = load_checkpoint(config.resume_path, model=model, optimizer=optimizer, map_location=str(device))
        start_epoch = chk.get("epoch", 0) + 1
        best_val_psnr = chk.get("val_psnr", -float("inf"))
        best_val_loss = chk.get("val_loss", float("inf"))
        best_epoch = chk.get("epoch", 0)
        print(f"Resumed from epoch {start_epoch - 1} (Previous best Val PSNR: {best_val_psnr:.2f} dB)")

    # 4. Training Loop
    history: List[Dict[str, Any]] = []
    total_start_time = time.perf_counter()

    for epoch in range(start_epoch, start_epoch + config.epochs):
        t_start = time.perf_counter()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        if epoch % config.validation_frequency == 0:
            val_loss, val_psnr, val_ssim = evaluate(model, val_loader, criterion, device)
        else:
            val_loss, val_psnr, val_ssim = 0.0, 0.0, 0.0

        epoch_duration = time.perf_counter() - t_start

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_psnr": round(val_psnr, 2),
            "val_ssim": round(val_ssim, 4),
            "duration_sec": round(epoch_duration, 2),
        }
        history.append(record)

        print(
            f"Epoch [{epoch}/{start_epoch + config.epochs - 1}] | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_loss:.5f} | "
            f"Val PSNR: {val_psnr:.2f} dB | "
            f"Val SSIM: {val_ssim:.4f} | "
            f"Time: {epoch_duration:.2f}s"
        )

        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "learning_rate": config.learning_rate,
            "seed": config.seed,
            "config": {
                "base_channels": config.base_channels,
                "residual": config.residual,
                "patch_size": config.patch_size,
            },
        }

        # Save latest checkpoint
        save_checkpoint(checkpoint_state, config.latest_checkpoint_path)

        # Save best checkpoint (based on validation loss improvement)
        if val_loss < best_val_loss and val_loss > 0.0:
            best_val_loss = val_loss
            best_val_psnr = val_psnr
            best_val_ssim = val_ssim
            best_epoch = epoch
            save_checkpoint(checkpoint_state, config.best_checkpoint_path)
            print(f"  --> [Saved new best checkpoint at epoch {epoch} (Val Loss: {val_loss:.5f}, PSNR: {val_psnr:.2f} dB, SSIM: {val_ssim:.4f})]")

    total_time = time.perf_counter() - total_start_time

    # 5. Save History CSV (append if resuming, else write)
    history_csv = config.history_csv_path
    mode = "a" if (config.resume_path and history_csv.is_file()) else "w"
    with open(history_csv, mode=mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "val_loss", "val_psnr", "val_ssim", "duration_sec"]
        )
        if mode == "w":
            writer.writeheader()
        writer.writerows(history)

    ram_final = get_system_ram_info()

    print("-" * 65)
    print("TRAINING RUN COMPLETE:")
    print(f"    Total Duration:     {total_time:.2f} s")
    print(f"    Best Epoch:         {best_epoch}")
    print(f"    Best Val Loss:      {best_val_loss:.5f}")
    print(f"    Best Val PSNR:      {best_val_psnr:.2f} dB")
    print(f"    Best Val SSIM:      {best_val_ssim:.4f}")
    print(f"    Best Checkpoint:    {config.best_checkpoint_path}")
    print(f"    Latest Checkpoint:  {config.latest_checkpoint_path}")
    print(f"    History Logged to:  {history_csv}")
    if ram_final:
        print(f"    System RAM Ending:  {ram_final.get('avail_ram_gb')} GB available ({ram_final.get('memory_load_pct')}% load)")
    print("=" * 65)

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_psnr": best_val_psnr,
        "best_val_ssim": best_val_ssim,
        "history": history,
        "total_time": total_time,
        "train_pairs": train_pairs_count,
        "train_scenes": train_scenes_count,
        "val_pairs": val_pairs_count,
        "val_scenes": val_scenes_count,
        "best_checkpoint": str(config.best_checkpoint_path),
        "latest_checkpoint": str(config.latest_checkpoint_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Lightweight U-Net Image Dehazer (Stage 2B)")
    parser.add_argument(
        "--train-dir",
        type=str,
        default=str(PROCESSED_DATA_DIR / "splits" / "synthetic_benchmark" / "train"),
        help="Path to training directory (containing clear/ and hazy/ subdirs).",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default=str(PROCESSED_DATA_DIR / "splits" / "synthetic_benchmark" / "val"),
        help="Path to validation directory (containing clear/ and hazy/ subdirs).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs (default: 2).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Mini-batch size (default: 1).",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=128,
        help="Patch resolution (H and W) for patch-based random cropping (default: 128).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Adam learning rate (default: 1e-4).",
    )
    parser.add_argument(
        "--channels",
        type=int,
        default=24,
        help="Base channels for LightweightUNet (default: 24).",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional maximum training sample pairs to use.",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=None,
        help="Optional maximum validation sample pairs to use.",
    )
    parser.add_argument(
        "--val-freq",
        type=int,
        default=1,
        help="Validation evaluation frequency in epochs (default: 1).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional path to checkpoint file to resume training from.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=get_default_device(),
        help="Target device (cpu or cuda).",
    )
    parser.add_argument(
        "--checkpoint-prefix",
        type=str,
        default="model",
        help="Prefix for saved checkpoints and history file (e.g. 'pilot' -> 'pilot_best_model.pth').",
    )
    parser.add_argument(
        "--max-variants-per-scene",
        type=int,
        default=None,
        help="Optional maximum haze variants to sample per unique clear scene (enforces scene diversity).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    config = TrainingConfig(
        batch_size=args.batch_size,
        patch_size=(args.patch_size, args.patch_size),
        learning_rate=args.lr,
        epochs=args.epochs,
        base_channels=args.channels,
        validation_frequency=args.val_freq,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_variants_per_scene=args.max_variants_per_scene,
        resume_path=Path(args.resume) if args.resume else None,
        checkpoint_prefix=args.checkpoint_prefix,
        device=args.device,
        seed=args.seed,
    )

    train_dehazer(
        train_dir=Path(args.train_dir),
        val_dir=Path(args.val_dir),
        config=config,
    )


if __name__ == "__main__":
    main()
