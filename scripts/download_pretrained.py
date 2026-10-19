"""Official Pretrained DehazeFormer-MCT Model Download Utility.

Downloads and verifies the official pretrained DehazeFormer-MCT checkpoint
from the verified HuggingFace Space repository.

Official Source:
    URL: https://huggingface.co/spaces/IDKiro/DehazeFormer_Demo/resolve/main/saved_models/dehazeformer.pth
    Paper: "Vision Transformers for Single Image Dehazing" (IEEE TIP 2023)
    Authors: Yuda Song, Zhuqing He, Hui Qian, Xin Du
    License: MIT License
    Expected Size: 5,927,945 bytes (~5.65 MB)
    Expected SHA-256: 225ce2d00d1cd8f0a7aabf5dc09f020ee34636cc942aa8b1a87496506224e304
"""

import hashlib
from pathlib import Path
import sys
import urllib.request

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TARGET_PATH = PROJECT_ROOT / "models" / "pretrained" / "dehazeformer_mct.pth"
SOURCE_URL = (
    "https://huggingface.co/spaces/IDKiro/DehazeFormer_Demo/resolve/main/saved_models/dehazeformer.pth"
)
EXPECTED_SHA256 = "225ce2d00d1cd8f0a7aabf5dc09f020ee34636cc942aa8b1a87496506224e304"
EXPECTED_SIZE = 5927945


def verify_sha256(file_path: Path) -> bool:
    """Calculate and verify SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest().lower() == EXPECTED_SHA256.lower()


def download_checkpoint(force: bool = False) -> None:
    """Download and verify pretrained DehazeFormer-MCT checkpoint."""
    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)

    if TARGET_PATH.is_file() and not force:
        print(f"Found existing checkpoint at: {TARGET_PATH}")
        print("Verifying SHA-256 checksum...")
        if verify_sha256(TARGET_PATH):
            print("Check: SHA-256 checksum verified successfully! File is intact.")
            return
        else:
            print("Warning: Existing file checksum mismatch. Re-downloading...")

    print("=" * 70)
    print("DOWNLOADING OFFICIAL PRETRAINED DEHAZEFORMER-MCT CHECKPOINT")
    print("=" * 70)
    print(f"Source URL:  {SOURCE_URL}")
    print(f"Target Path: {TARGET_PATH}")
    print(f"File Size:   {EXPECTED_SIZE / (1024*1024):.2f} MB")
    print(f"License:     MIT License (IDKiro)")
    print("-" * 70)

    def progress_callback(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded / total_size * 100.0)
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\rDownloading: {mb_down:.2f} MB / {mb_total:.2f} MB ({percent:.1f}%)")
            sys.stdout.flush()

    try:
        temp_dest = TARGET_PATH.with_suffix(".tmp")
        urllib.request.urlretrieve(SOURCE_URL, temp_dest, reporthook=progress_callback)
        print("\nDownload complete. Verifying file integrity...")

        hasher = hashlib.sha256()
        with open(temp_dest, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        calc_sha = hasher.hexdigest().lower()

        if calc_sha != EXPECTED_SHA256.lower():
            temp_dest.unlink(missing_ok=True)
            raise ValueError(
                f"Checksum mismatch!\nExpected: {EXPECTED_SHA256}\nCalculated: {calc_sha}"
            )

        if temp_dest.stat().st_size != EXPECTED_SIZE:
            temp_dest.unlink(missing_ok=True)
            raise ValueError(
                f"Size mismatch: expected {EXPECTED_SIZE} bytes, got {temp_dest.stat().st_size} bytes."
            )

        # Atomic rename
        if TARGET_PATH.exists():
            TARGET_PATH.unlink()
        temp_dest.rename(TARGET_PATH)

        print(f"SUCCESS: Saved and verified checkpoint at: {TARGET_PATH}")
        print("Ready for inference with Streamlit UI (app/ui.py) or scripts/infer.py.")

    except Exception as e:
        print(f"\nERROR: Failed to download pretrained checkpoint: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    force_download = "--force" in sys.argv
    download_checkpoint(force=force_download)
