"""Held-Out Test Set Evaluation Script for AI Dehazing vs DCP Baseline (Stage 2B).

Evaluates the trained Lightweight U-Net model and the classical DCP baseline
on the held-out test split. Computes:
- PSNR (dB)
- SSIM
- Average Inference Latency (ms)
- 4-panel visual comparison figures (Hazy | DCP | AI U-Net | Ground Truth)
- Machine-readable evaluation summary JSON
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

# Ensure project root is available in module path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch

from app.config import (
    CHECKPOINTS_DIR,
    OUTPUTS_DIR,
    PROCESSED_DATA_DIR,
    get_default_device,
)
from src.dehazing.dcp import dcp_dehaze
from src.dehazing.model import LightweightUNet, load_checkpoint
from src.metrics.psnr import calculate_psnr
from src.metrics.ssim import calculate_ssim
from src.preprocessing.dataset import PairedHazeDataset
from src.utils.image_ops import numpy_to_tensor, tensor_to_numpy
from src.utils.logger import get_logger

logger = get_logger("evaluate")


def create_comparison_mosaic(
    hazy_rgb: np.ndarray,
    dcp_rgb: np.ndarray,
    ai_rgb: np.ndarray,
    clear_rgb: np.ndarray,
    psnr_hazy: float,
    psnr_dcp: float,
    psnr_ai: float,
) -> np.ndarray:
    """Create an annotated 4-panel horizontal comparison image strip."""
    h, w, c = hazy_rgb.shape

    # Ensure all panels have identical dimensions
    panels = [hazy_rgb, dcp_rgb, ai_rgb, clear_rgb]
    resized_panels = []
    for p in panels:
        if p.shape[:2] != (h, w):
            p = cv2.resize(p, (w, h), interpolation=cv2.INTER_AREA)
        resized_panels.append(p)

    banner_height = 40
    annotated = []
    titles = [
        f"Hazy ({psnr_hazy:.1f} dB)",
        f"DCP Baseline ({psnr_dcp:.1f} dB)",
        f"AI U-Net ({psnr_ai:.1f} dB)",
        "Ground Truth Clear",
    ]

    for panel, title in zip(resized_panels, titles):
        # Add banner header
        banner = np.zeros((banner_height, w, 3), dtype=np.uint8)
        cv2.putText(
            banner,
            title,
            (10, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        stacked = np.vstack([banner, panel])
        annotated.append(stacked)

    mosaic = np.hstack(annotated)
    return mosaic


def evaluate_test_set(
    test_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    device_str: str = "cpu",
    save_images: bool = True,
) -> Dict[str, Any]:
    """Run rigorous evaluation on the held-out test split.

    Args:
        test_dir: Path to directory containing clear/ and hazy/ test subdirectories.
        checkpoint_path: Path to trained model .pth checkpoint.
        output_dir: Directory to save visual comparisons and evaluation summary JSON.
        device_str: Compute device ("cpu" or "cuda").
        save_images: Whether to save comparison mosaics to disk.

    Returns:
        dict: Evaluation statistics and metric tables.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "comparisons"
    if save_images:
        images_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_str)

    # 1. Load Test Dataset
    dataset = PairedHazeDataset(
        root_dir=test_dir,
        strict_pairs=False,
    )

    if len(dataset) == 0:
        raise ValueError(f"No valid test pairs found in: {test_dir}")

    # 2. Load AI Model
    print("=" * 70)
    print("STAGE 2B: HELD-OUT TEST SET EVALUATION")
    print("=" * 70)
    print(f"Test Set Directory:   {test_dir}")
    print(f"Test Pairs Found:     {len(dataset)}")
    print(f"Model Checkpoint:     {checkpoint_path}")
    print(f"Evaluation Device:    {device_str.upper()}")
    print("-" * 70)

    raw_chk = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    chk_config = raw_chk.get("config", {})
    base_channels = chk_config.get("base_channels", 24)
    residual = chk_config.get("residual", True)

    model = LightweightUNet(
        in_channels=3,
        out_channels=3,
        base_channels=base_channels,
        residual=residual,
    ).to(device)

    load_checkpoint(checkpoint_path, model=model, map_location=str(device))
    model.eval()

    param_count = model.count_parameters()
    print(f"Model Parameter Count:{param_count:,} ({param_count / 1e6:.2f}M params)")
    print("-" * 70)

    # Metrics storage
    hazy_psnr_list, hazy_ssim_list = [], []
    dcp_psnr_list, dcp_ssim_list, dcp_latency_list = [], [], []
    ai_psnr_list, ai_ssim_list, ai_latency_list = [], [], []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        hazy_np = tensor_to_numpy(sample["hazy"])
        clear_np = tensor_to_numpy(sample["clear"])
        stem = sample["stem"]

        # 1. Hazy baseline metrics
        p_hazy = calculate_psnr(hazy_np, clear_np, max_val=255.0)
        s_hazy = calculate_ssim(hazy_np, clear_np, max_val=255.0)
        hazy_psnr_list.append(p_hazy)
        hazy_ssim_list.append(s_hazy)

        # 2. Classical DCP baseline
        t0_dcp = time.perf_counter()
        dcp_np, _ = dcp_dehaze(hazy_np, use_guided_filter=True)
        dcp_lat = (time.perf_counter() - t0_dcp) * 1000.0
        p_dcp = calculate_psnr(dcp_np, clear_np, max_val=255.0)
        s_dcp = calculate_ssim(dcp_np, clear_np, max_val=255.0)
        dcp_psnr_list.append(p_dcp)
        dcp_ssim_list.append(s_dcp)
        dcp_latency_list.append(dcp_lat)

        # 3. AI Lightweight U-Net
        tensor_in = sample["hazy"].unsqueeze(0).to(device)
        t0_ai = time.perf_counter()
        with torch.no_grad():
            tensor_out = model(tensor_in)
        ai_lat = (time.perf_counter() - t0_ai) * 1000.0
        ai_np = tensor_to_numpy(tensor_out.squeeze(0))

        p_ai = calculate_psnr(ai_np, clear_np, max_val=255.0)
        s_ai = calculate_ssim(ai_np, clear_np, max_val=255.0)
        ai_psnr_list.append(p_ai)
        ai_ssim_list.append(s_ai)
        ai_latency_list.append(ai_lat)

        print(
            f"[{idx + 1}/{len(dataset)}] Pair: {stem} | "
            f"Hazy PSNR: {p_hazy:.2f} dB | "
            f"DCP PSNR: {p_dcp:.2f} dB | "
            f"AI PSNR: {p_ai:.2f} dB"
        )

        # 4. Save representative comparison image
        if save_images:
            mosaic = create_comparison_mosaic(
                hazy_np, dcp_np, ai_np, clear_np, p_hazy, p_dcp, p_ai
            )
            mosaic_bgr = cv2.cvtColor(mosaic, cv2.COLOR_RGB2BGR)
            mosaic_path = images_dir / f"compare_{stem}.png"
            cv2.imwrite(str(mosaic_path), mosaic_bgr)

    # Compute averages
    results = {
        "num_test_pairs": len(dataset),
        "device": device_str,
        "model_parameters": param_count,
        "checkpoint": str(checkpoint_path),
        "metrics": {
            "Hazy (Input)": {
                "psnr": float(np.mean(hazy_psnr_list)),
                "ssim": float(np.mean(hazy_ssim_list)),
                "latency_ms": 0.0,
            },
            "Classical DCP": {
                "psnr": float(np.mean(dcp_psnr_list)),
                "ssim": float(np.mean(dcp_ssim_list)),
                "latency_ms": float(np.mean(dcp_latency_list)),
            },
            "AI Lightweight U-Net": {
                "psnr": float(np.mean(ai_psnr_list)),
                "ssim": float(np.mean(ai_ssim_list)),
                "latency_ms": float(np.mean(ai_latency_list)),
            },
        },
    }

    # Print summary table
    print("=" * 70)
    print("EVALUATION SUMMARY RESULTS")
    print("=" * 70)
    print(f"{'Method':<24} | {'PSNR (dB)':<12} | {'SSIM':<10} | {'Latency (ms)':<14}")
    print("-" * 70)
    for method, data in results["metrics"].items():
        lat_str = f"{data['latency_ms']:.1f} ms" if data["latency_ms"] > 0 else "N/A"
        print(
            f"{method:<24} | {data['psnr']:<12.2f} | {data['ssim']:<10.4f} | {lat_str:<14}"
        )
    print("=" * 70)

    # Save summary JSON
    json_path = output_dir / "test_evaluation_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {json_path}")
    if save_images:
        print(f"Visual comparison strips saved to: {images_dir}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Dehazing Checkpoint on Held-Out Test Set")
    parser.add_argument(
        "--test-dir",
        type=str,
        default=str(PROCESSED_DATA_DIR / "reside" / "test"),
        help="Path to held-out test split containing clear/ and hazy/ subdirs.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINTS_DIR / "best_model.pth"),
        help="Path to trained model checkpoint file (.pth).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUTS_DIR / "evaluation"),
        help="Directory to save comparison images and summary report.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=get_default_device(),
        help="Compute device (cpu or cuda).",
    )
    parser.add_argument(
        "--no-save-images",
        action="store_true",
        help="Disable saving visual comparison images.",
    )

    args = parser.parse_args()

    test_path = Path(args.test_dir)
    chk_path = Path(args.checkpoint)

    # If reside/test doesn't exist, fallback to synthetic_benchmark/test
    if not test_path.exists():
        fallback_path = PROCESSED_DATA_DIR / "splits" / "synthetic_benchmark" / "test"
        if fallback_path.exists():
            print(f"Notice: '{test_path}' not found, falling back to '{fallback_path}'.")
            test_path = fallback_path

    if not chk_path.is_file():
        print(f"Error: Checkpoint file not found: {chk_path}")
        print("Please train a model first using: python scripts/train.py")
        sys.exit(1)

    evaluate_test_set(
        test_dir=test_path,
        checkpoint_path=chk_path,
        output_dir=Path(args.output_dir),
        device_str=args.device,
        save_images=not args.no_save_images,
    )


if __name__ == "__main__":
    main()
