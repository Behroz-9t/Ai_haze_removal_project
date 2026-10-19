"""Generate Paired Synthetic Hazy Dataset from Clear Images.

Applies Koschmieder's atmospheric scattering model:
    I(x) = J(x) * t(x) + A * (1 - t(x))
Saves paired clear and hazy images into structured directories:
    <output_dir>/clear/
    <output_dir>/hazy/
"""

import argparse
from pathlib import Path
import sys

# Ensure project root is available in module path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (
    DATA_DIR,
    PROCESSED_DATA_DIR,
    SAMPLES_DATA_DIR,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from src.preprocessing.synthetic import generate_synthetic_haze
from src.utils.exceptions import ImageLoadError
from src.utils.image_ops import load_image, save_image
from src.utils.logger import get_logger

logger = get_logger("generate_synthetic_haze")


def generate_paired_dataset(
    source_dir_or_file: Path,
    output_dir: Path,
    beta: float = 1.0,
    atmospheric_light: float = 0.9,
    depth_mode: str = "gradient",
    max_count: int = 10,
    seed: int = 42,
) -> int:
    """Process clear images and output paired clear/hazy image sets.

    Args:
        source_dir_or_file: Source directory of clear images, or a single clear image file.
        output_dir: Destination base directory where 'clear' and 'hazy' subdirs will be created.
        beta: Haze density / scattering coefficient.
        atmospheric_light: Global airlight intensity [0.0..1.0].
        depth_mode: Depth distribution model ('gradient', 'radial', 'uniform').
        max_count: Maximum number of images to synthesize.
        seed: Deterministic random seed.

    Returns:
        int: Number of image pairs successfully generated.
    """
    clear_out_dir = output_dir / "clear"
    hazy_out_dir = output_dir / "hazy"
    clear_out_dir.mkdir(parents=True, exist_ok=True)
    hazy_out_dir.mkdir(parents=True, exist_ok=True)

    # Collect source images
    if source_dir_or_file.is_file():
        image_files = [source_dir_or_file]
    elif source_dir_or_file.is_dir():
        image_files = [
            p for p in source_dir_or_file.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        image_files.sort()
    else:
        raise ValueError(f"Source path does not exist: {source_dir_or_file}")

    if not image_files:
        logger.warning(f"No valid image files found at: {source_dir_or_file}")
        return 0

    processed_count = 0
    print(f"Generating synthetic haze pairs from: {source_dir_or_file}")
    print(f"Parameters: beta={beta}, A={atmospheric_light}, depth_mode='{depth_mode}', seed={seed}")
    print(f"Destination: {output_dir}")
    print("-" * 60)

    for idx, img_path in enumerate(image_files[:max_count]):
        try:
            clear_rgb = load_image(img_path)
            hazy_rgb, _ = generate_synthetic_haze(
                clear_image=clear_rgb,
                beta=beta,
                atmospheric_light=atmospheric_light,
                depth_mode=depth_mode,
                seed=seed + idx,
            )

            # Save clear and hazy pairs with identical filenames
            stem_name = f"{img_path.stem}.png"
            save_image(clear_rgb, clear_out_dir / stem_name, is_rgb=True)
            save_image(hazy_rgb, hazy_out_dir / stem_name, is_rgb=True)

            processed_count += 1
            print(f"  [{processed_count}/{min(len(image_files), max_count)}] Generated pair: {stem_name}")
        except (ImageLoadError, ValueError) as err:
            logger.error(f"Failed to generate pair for '{img_path.name}': {err}")

    print("-" * 60)
    print(f"Completed: {processed_count} paired samples created under '{output_dir}'.")
    return processed_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controlled Synthetic Haze Generator using Atmospheric Scattering Model"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(SAMPLES_DATA_DIR / "sample_clear.png"),
        help="Path to clear source image or directory of clear images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(SAMPLES_DATA_DIR / "synthetic_paired"),
        help="Directory to store clear/ and hazy/ output pairs.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.2,
        help="Atmospheric scattering coefficient beta (default: 1.2).",
    )
    parser.add_argument(
        "--light",
        type=float,
        default=0.88,
        help="Atmospheric light intensity A in [0.0..1.0] (default: 0.88).",
    )
    parser.add_argument(
        "--depth-mode",
        type=str,
        choices=["gradient", "radial", "uniform"],
        default="gradient",
        help="Depth approximation model (default: 'gradient').",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=5,
        help="Maximum images to process (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )

    args = parser.parse_args()
    generate_paired_dataset(
        source_dir_or_file=Path(args.source),
        output_dir=Path(args.output_dir),
        beta=args.beta,
        atmospheric_light=args.light,
        depth_mode=args.depth_mode,
        max_count=args.count,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
