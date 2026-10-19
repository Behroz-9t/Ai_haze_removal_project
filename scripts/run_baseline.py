"""Run Classical Dark Channel Prior (DCP) Dehazing Baseline on Paired Images.

Processes a small, CPU-friendly batch of paired hazy/clear images, computes
quantitative evaluation metrics (PSNR, SSIM), saves comparative visualizations,
and writes an experiment summary CSV.
"""

import argparse
import csv
from pathlib import Path
import sys
import time
from typing import List, Tuple

import numpy as np

# Ensure project root is available in module path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (
    OUTPUTS_DIR,
    OUTPUT_IMAGES_DIR,
    OUTPUT_METRICS_DIR,
    SAMPLES_DATA_DIR,
    ensure_project_directories,
)
from src.dehazing.dcp import dcp_dehaze
from src.dehazing.guided_filter import get_guided_filter_backend
from src.metrics.psnr import calculate_psnr
from src.metrics.ssim import calculate_ssim
from src.preprocessing.dataset import PairedHazeDataset
from src.utils.image_ops import load_image, save_image
from src.utils.logger import get_logger
from src.utils.visualization import create_comparison_panel

logger = get_logger("run_baseline")


def run_baseline_experiment(
    dataset_dir: Path,
    output_dir: Path,
    metrics_csv_path: Path,
    max_samples: int = 5,
    patch_size: int = 15,
    omega: float = 0.95,
    use_guided_filter: bool = True,
) -> List[dict]:
    """Execute the DCP baseline on paired images and log quantitative metrics.

    Args:
        dataset_dir: Path to directory containing 'clear' and 'hazy' subdirectories.
        output_dir: Directory where processed image panels will be saved.
        metrics_csv_path: Destination path for the CSV metrics log.
        max_samples: Maximum number of image pairs to evaluate.
        patch_size: Neighborhood patch size for the dark channel.
        omega: Haze retention factor.
        use_guided_filter: Whether to refine the transmission map with guided filter.

    Returns:
        list of dicts containing evaluation records.
    """
    ensure_project_directories()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)

    backend = get_guided_filter_backend()
    print("=" * 65)
    print("STAGE 1: CLASSICAL DEHAZING BASELINE (DARK CHANNEL PRIOR)")
    print("=" * 65)
    print(f"Dataset Directory:       {dataset_dir}")
    print(f"Output Directory:        {output_dir}")
    print(f"Metrics CSV Output:      {metrics_csv_path}")
    print(f"Guided Filter Backend:   {backend}")
    print(f"Max Samples:             {max_samples} (CPU-friendly)")
    print("-" * 65)

    dataset = PairedHazeDataset(
        root_dir=dataset_dir,
        strict_pairs=False,
    )

    if len(dataset) == 0:
        print(f"[WARNING] No paired images discovered under '{dataset_dir}'.")
        print("Please verify that 'clear/' and 'hazy/' subdirectories exist with matching stems.")
        return []

    num_eval = min(len(dataset), max_samples)
    results: List[dict] = []

    for i in range(num_eval):
        hazy_path, clear_path, stem = dataset.pairs[i]
        print(f"Processing sample [{i + 1}/{num_eval}]: '{stem}' ...")

        # Load RGB images directly as uint8
        clear_rgb = load_image(clear_path)
        hazy_rgb = load_image(hazy_path)

        # Execute DCP dehazing
        t0 = time.perf_counter()
        dehazed_rgb, meta = dcp_dehaze(
            hazy_rgb,
            patch_size=patch_size,
            omega=omega,
            use_guided_filter=use_guided_filter,
        )
        elapsed_sec = time.perf_counter() - t0

        # Calculate metrics against ground truth clear image
        psnr_hazy = calculate_psnr(hazy_rgb, clear_rgb, max_val=255.0)
        ssim_hazy = calculate_ssim(hazy_rgb, clear_rgb, max_val=255.0)

        psnr_dcp = calculate_psnr(dehazed_rgb, clear_rgb, max_val=255.0)
        ssim_dcp = calculate_ssim(dehazed_rgb, clear_rgb, max_val=255.0)

        # Save individual outputs
        save_image(clear_rgb, output_dir / f"{stem}_clear.png", is_rgb=True)
        save_image(hazy_rgb, output_dir / f"{stem}_hazy.png", is_rgb=True)
        save_image(dehazed_rgb, output_dir / f"{stem}_dcp.png", is_rgb=True)

        # Save 3-panel side-by-side comparison
        comparison = create_comparison_panel(
            clear=clear_rgb,
            hazy=hazy_rgb,
            dehazed=dehazed_rgb,
            labels=(
                f"Clear Reference",
                f"Hazy Input (PSNR:{psnr_hazy:.2f}dB)",
                f"DCP Output (PSNR:{psnr_dcp:.2f}dB)",
            ),
        )
        save_image(comparison, output_dir / f"{stem}_comparison.png", is_rgb=True)

        record = {
            "sample_stem": stem,
            "resolution": f"{clear_rgb.shape[1]}x{clear_rgb.shape[0]}",
            "runtime_seconds": round(elapsed_sec, 3),
            "psnr_hazy_input_db": round(psnr_hazy, 2),
            "psnr_dcp_output_db": round(psnr_dcp, 2),
            "psnr_delta_db": round(psnr_dcp - psnr_hazy, 2),
            "ssim_hazy_input": round(ssim_hazy, 4),
            "ssim_dcp_output": round(ssim_dcp, 4),
            "ssim_delta": round(ssim_dcp - ssim_hazy, 4),
            "airlight_r": round(float(meta["atmospheric_light"][0]), 3),
            "airlight_g": round(float(meta["atmospheric_light"][1]), 3),
            "airlight_b": round(float(meta["atmospheric_light"][2]), 3),
        }
        results.append(record)

        print(
            f"   Time: {elapsed_sec:.2f}s | "
            f"PSNR: {psnr_hazy:.2f} dB -> {psnr_dcp:.2f} dB (Delta: {record['psnr_delta_db']:+.2f} dB) | "
            f"SSIM: {ssim_hazy:.4f} -> {ssim_dcp:.4f} (Delta: {record['ssim_delta']:+.4f})"
        )

    # Write summary CSV
    if results:
        fieldnames = list(results[0].keys())
        with open(metrics_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        avg_psnr_delta = np.mean([r["psnr_delta_db"] for r in results])
        avg_ssim_delta = np.mean([r["ssim_delta"] for r in results])
        avg_time = np.mean([r["runtime_seconds"] for r in results])

        print("-" * 65)
        print("EXPERIMENT SUMMARY:")
        print(f"    Evaluated Pairs:     {len(results)}")
        print(f"    Average Runtime:     {avg_time:.2f} s / image")
        print(f"    Mean PSNR Delta:     {avg_psnr_delta:+.2f} dB")
        print(f"    Mean SSIM Delta:     {avg_ssim_delta:+.4f}")
        print(f"    Metrics CSV Saved:   {metrics_csv_path}")
        print(f"    Images Saved Under:  {output_dir}")
        print("=" * 65)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Classical DCP Dehazing Baseline on Paired Image Dataset"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=str(SAMPLES_DATA_DIR / "synthetic_paired"),
        help="Path to directory containing 'clear/' and 'hazy/' subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_IMAGES_DIR / "baseline"),
        help="Path to save output images and comparison strips.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=str,
        default=str(OUTPUT_METRICS_DIR / "baseline_results.csv"),
        help="Path to save baseline results CSV.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=3,
        help="Maximum pairs to process (default: 3).",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=15,
        help="Dark channel patch window size (default: 15).",
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=0.95,
        help="Haze retention factor (default: 0.95).",
    )
    parser.add_argument(
        "--no-guided-filter",
        action="store_true",
        help="Disable guided filtering transmission refinement.",
    )

    args = parser.parse_args()
    run_baseline_experiment(
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        metrics_csv_path=Path(args.metrics_csv),
        max_samples=args.max_samples,
        patch_size=args.patch_size,
        omega=args.omega,
        use_guided_filter=not args.no_guided_filter,
    )


if __name__ == "__main__":
    main()
