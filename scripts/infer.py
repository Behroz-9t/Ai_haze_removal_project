"""Inference and Comparative Evaluation Script for AI Image Dehazing.

Loads a trained LightweightUNet checkpoint and runs dehazing on an input image.
Optionally compares against ground-truth reference and the Stage 1 DCP baseline,
generating a 4-panel comparative visualization (Clear | Hazy | DCP | AI).
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Optional

# Ensure project root is available in module path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from app.config import (
    CHECKPOINTS_DIR,
    OUTPUTS_DIR,
    ensure_project_directories,
    get_default_device,
)
from src.dehazing.dcp import dcp_dehaze
from src.dehazing.model import LightweightUNet, load_checkpoint
from src.metrics.psnr import calculate_psnr
from src.metrics.ssim import calculate_ssim
from src.utils.image_ops import load_image, numpy_to_tensor, save_image, tensor_to_numpy
from src.utils.logger import get_logger
from src.utils.visualization import create_multi_panel

logger = get_logger("infer")


def run_inference(
    input_path: Path,
    checkpoint_path: Path,
    output_path: Optional[Path] = None,
    reference_path: Optional[Path] = None,
    compare_dcp: bool = True,
    device_str: Optional[str] = None,
) -> Path:
    """Run model inference on single hazy image and optionally benchmark against reference and DCP.

    Args:
        input_path: Path to input hazy image.
        checkpoint_path: Path to trained model weights (.pth).
        output_path: Optional destination path for restored image.
        reference_path: Optional ground truth clean image path.
        compare_dcp: Whether to generate comparative DCP baseline output when reference is present.
        device_str: Device to run on ('cpu' or 'cuda').

    Returns:
        Path: Path to saved AI dehazed image.
    """
    ensure_project_directories()
    device = torch.device(device_str if device_str else get_default_device())

    if not input_path.is_file():
        raise FileNotFoundError(f"Input hazy image not found: {input_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    # Set default output path if not specified
    if output_path is None:
        infer_dir = OUTPUTS_DIR / "inference"
        infer_dir.mkdir(parents=True, exist_ok=True)
        output_path = infer_dir / f"{input_path.stem}_ai_dehazed.png"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load image and prepare tensor
    hazy_rgb = load_image(input_path)
    h_orig, w_orig, _ = hazy_rgb.shape

    # (C, H, W) float32 in [0, 1]
    tensor_input = numpy_to_tensor(hazy_rgb, device=device).unsqueeze(0)  # (1, 3, H, W)

    # 2. Instantiate model and load checkpoint
    # Peek at config stored in checkpoint if available
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

    # 3. Model forward pass
    t0 = time.perf_counter()
    with torch.no_grad():
        tensor_output = model(tensor_input)
    infer_time_ms = (time.perf_counter() - t0) * 1000.0

    # 4. Convert back to RGB image
    ai_dehazed_rgb = tensor_to_numpy(tensor_output.squeeze(0))
    save_image(ai_dehazed_rgb, output_path, is_rgb=True)

    print("=" * 65)
    print("STAGE 2A: AI DEHAZING INFERENCE REPORT")
    print("=" * 65)
    print(f"Input:          {input_path}")
    print(f"Output:         {output_path}")
    print(f"Dimensions:     {w_orig} x {h_orig} (Width x Height)")
    print(f"Model:          LightweightUNet (base_ch={base_channels}, {model.count_parameters():,} params)")
    print(f"Checkpoint:     {checkpoint_path.name}")
    print(f"Device:         {device}")
    print(f"Inference Time: {infer_time_ms:.2f} ms ({infer_time_ms / 1000.0:.3f} s)")

    # 5. Optional reference and DCP comparison
    if reference_path is not None and reference_path.is_file():
        clear_rgb = load_image(reference_path)
        psnr_hazy = calculate_psnr(hazy_rgb, clear_rgb, max_val=255.0)
        ssim_hazy = calculate_ssim(hazy_rgb, clear_rgb, max_val=255.0)

        psnr_ai = calculate_psnr(ai_dehazed_rgb, clear_rgb, max_val=255.0)
        ssim_ai = calculate_ssim(ai_dehazed_rgb, clear_rgb, max_val=255.0)

        print("-" * 65)
        print("QUANTITATIVE BENCHMARK (AGAINST GROUND TRUTH CLEAR):")
        print(f"  Hazy Input:  PSNR = {psnr_hazy:.2f} dB  |  SSIM = {ssim_hazy:.4f}")
        print(f"  AI Dehazed:  PSNR = {psnr_ai:.2f} dB  |  SSIM = {ssim_ai:.4f}  (Delta: {psnr_ai - psnr_hazy:+.2f} dB)")

        if compare_dcp:
            t_dcp = time.perf_counter()
            dcp_rgb, _ = dcp_dehaze(hazy_rgb, use_guided_filter=True)
            dcp_time_ms = (time.perf_counter() - t_dcp) * 1000.0

            psnr_dcp = calculate_psnr(dcp_rgb, clear_rgb, max_val=255.0)
            ssim_dcp = calculate_ssim(dcp_rgb, clear_rgb, max_val=255.0)

            print(f"  DCP Baseline:PSNR = {psnr_dcp:.2f} dB  |  SSIM = {ssim_dcp:.4f}  (Time: {dcp_time_ms:.1f} ms)")

            # Generate 4-panel comparison: Clear | Hazy | DCP | AI
            comp_dir = OUTPUTS_DIR / "inference" / "comparisons"
            comp_dir.mkdir(parents=True, exist_ok=True)
            comp_path = comp_dir / f"{input_path.stem}_4panel_comparison.png"

            strip = create_multi_panel(
                images=[clear_rgb, hazy_rgb, dcp_rgb, ai_dehazed_rgb],
                labels=[
                    "Clear Reference",
                    f"Hazy ({psnr_hazy:.1f}dB)",
                    f"DCP Baseline ({psnr_dcp:.1f}dB)",
                    f"AI Output ({psnr_ai:.1f}dB)",
                ],
            )
            save_image(strip, comp_path, is_rgb=True)
            print(f"  Comparison:  {comp_path}")

    print("=" * 65)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AI Dehazing Inference on an Image")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input hazy image file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINTS_DIR / "best_model.pth"),
        help="Path to trained checkpoint (.pth).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Target output path for dehazed image.",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default=None,
        help="Optional path to clear ground truth reference image.",
    )
    parser.add_argument(
        "--no-dcp",
        action="store_true",
        help="Do not execute DCP comparison when reference is present.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=get_default_device(),
        help="Device to run inference on (cpu or cuda).",
    )

    args = parser.parse_args()

    chk_path = Path(args.checkpoint)
    if not chk_path.is_file():
        # Fallback to latest_model.pth if best_model.pth not yet created
        latest_path = CHECKPOINTS_DIR / "latest_model.pth"
        if latest_path.is_file():
            chk_path = latest_path

    run_inference(
        input_path=Path(args.input),
        checkpoint_path=chk_path,
        output_path=Path(args.output) if args.output else None,
        reference_path=Path(args.reference) if args.reference else None,
        compare_dcp=not args.no_dcp,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
