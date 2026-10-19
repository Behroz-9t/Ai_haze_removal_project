"""Setup Sample RESIDE ITS Dataset for Pipeline Verification (Stage 2B).

Creates a realistic mock/sample RESIDE ITS directory structure in:
    data/raw/RESIDE/ITS/
        clear/  (1400.png, 1401.png, 1402.png, 1403.png, 1404.png)
        hazy/   (1400_1_0.08.png, 1400_2_0.12.png, ...)

This allows full local testing of ingestion, pairing, zero-leakage splitting,
training, and evaluation without downloading multi-gigabyte datasets over the network.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from app.config import RAW_DATA_DIR, SAMPLES_DATA_DIR
from src.preprocessing.synthetic import generate_synthetic_haze


def generate_scene(base_seed: int, width: int = 256, height: int = 256) -> np.ndarray:
    """Generate a procedurally structured RGB image with distinct geometric features."""
    rng = np.random.RandomState(base_seed)
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Gradient background
    c1 = rng.randint(40, 200, size=3)
    c2 = rng.randint(40, 200, size=3)
    for y in range(height):
        alpha = y / height
        img[y, :] = (1 - alpha) * c1 + alpha * c2

    # Draw geometric shapes simulating indoor room objects (rectangles, circles)
    for _ in range(5):
        color = [int(x) for x in rng.randint(20, 240, size=3)]
        pt1 = (int(rng.randint(0, width - 40)), int(rng.randint(0, height - 40)))
        pt2 = (int(pt1[0] + rng.randint(30, 80)), int(pt1[1] + rng.randint(30, 80)))
        cv2.rectangle(img, pt1, pt2, color, -1)

    for _ in range(4):
        color = [int(x) for x in rng.randint(20, 240, size=3)]
        center = (int(rng.randint(30, width - 30)), int(rng.randint(30, height - 30)))
        radius = int(rng.randint(15, 45))
        cv2.circle(img, center, radius, color, -1)

    return img


def setup_sample_reside(target_dir: Path) -> None:
    reside_its = target_dir / "ITS"
    clear_dir = reside_its / "clear"
    hazy_dir = reside_its / "hazy"
    clear_dir.mkdir(parents=True, exist_ok=True)
    hazy_dir.mkdir(parents=True, exist_ok=True)

    scenes = [1400, 1401, 1402, 1403, 1404]
    haze_params = [
        (1, 0.08, 0.85),
        (2, 0.14, 0.90),
    ]

    print(f"Setting up sample RESIDE ITS dataset in: {reside_its}")

    for idx, scene_id in enumerate(scenes):
        clear_img = generate_scene(base_seed=100 + idx, width=256, height=256)
        clear_path = clear_dir / f"{scene_id}.png"
        cv2.imwrite(str(clear_path), cv2.cvtColor(clear_img, cv2.COLOR_RGB2BGR))
        print(f"  Created clear scene: {clear_path.name}")

        for h_idx, beta, airlight in haze_params:
            hazy_img, _ = generate_synthetic_haze(
                clear_img,
                beta=beta * 10.0,
                atmospheric_light=airlight,
                depth_mode="gradient",
                seed=scene_id + h_idx,
            )
            hazy_name = f"{scene_id}_{h_idx}_{beta:.2f}.png"
            hazy_path = hazy_dir / hazy_name
            cv2.imwrite(str(hazy_path), cv2.cvtColor(hazy_img, cv2.COLOR_RGB2BGR))
            print(f"    -> Hazy variant: {hazy_name}")

    print(f"Successfully generated 5 clear scenes and 10 hazy images.")


if __name__ == "__main__":
    setup_sample_reside(RAW_DATA_DIR / "RESIDE")
