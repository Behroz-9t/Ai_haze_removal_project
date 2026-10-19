"""Offline Video Dehazing Processor.

Performs memory-efficient, frame-by-frame atmospheric visibility enhancement
on stored video files (MP4, AVI, MOV, MKV). Preserves frame dimensions, frame rate,
and temporal ordering. Optionally preserves audio tracks using system FFmpeg when available.
"""

from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Callable, Dict, Optional, Union

import cv2
import numpy as np
import torch

from app.config import OUTPUT_VIDEOS_DIR
from src.dehazing.dcp import dcp_dehaze
from src.utils.image_ops import numpy_to_tensor, tensor_to_numpy
from src.utils.logger import get_logger

logger = get_logger("video_processor")


def get_ffmpeg_path() -> Optional[str]:
    """Check if ffmpeg executable is available in PATH or system environment."""
    return shutil.which("ffmpeg")


def process_video_file(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    model: Optional[torch.nn.Module] = None,
    method: str = "ai",
    device_str: str = "cpu",
    max_frames: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int, float], None]] = None,
    preserve_audio: bool = True,
) -> Dict[str, Any]:
    """Enhance video file frame-by-frame with bounded memory consumption.

    Args:
        input_path: Path to source video file.
        output_path: Destination path for enhanced video. If None, auto-generated in outputs/videos/.
        model: Loaded PyTorch dehazing model (LightweightUNet). Required if method='ai'.
        method: Enhancement method ('ai' or 'dcp').
        device_str: Compute device ('cpu').
        max_frames: Optional maximum frames to process (useful for previewing large videos).
        progress_callback: Optional callback func(frame_idx, total_frames, elapsed_sec).
        preserve_audio: If True, attempts to mux original audio using FFmpeg if available.

    Returns:
        Dict[str, Any]: Summary metrics (frame_count, processing_fps, duration, audio_preserved).
    """
    in_file = Path(input_path)
    if not in_file.is_file():
        raise FileNotFoundError(f"Input video file not found: {in_file}")

    OUTPUT_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        out_file = OUTPUT_VIDEOS_DIR / f"{in_file.stem}_enhanced_{method}.mp4"
    else:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(in_file))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file with OpenCV: {in_file}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames_in_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_to_process = min(total_frames_in_file, max_frames) if max_frames else total_frames_in_file

    if width == 0 or height == 0:
        cap.release()
        raise ValueError(f"Invalid video dimensions ({width}x{height}) in {in_file}")

    # Prepare temporary video output path (without audio)
    temp_video_out = out_file.with_name(f"temp_{out_file.name}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_video_out), fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to initialize VideoWriter at {temp_video_out}")

    device = torch.device(device_str)
    if method == "ai":
        if model is None:
            cap.release()
            writer.release()
            raise ValueError("PyTorch model must be provided when method='ai'")
        model.eval()

    frames_processed = 0
    t_start = time.perf_counter()

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            if method == "ai":
                if hasattr(model, "dehaze"):
                    enhanced_rgb, _ = model.dehaze(frame_rgb)
                else:
                    tensor_in = numpy_to_tensor(frame_rgb, device=device).unsqueeze(0)
                    with torch.no_grad():
                        tensor_out = model(tensor_in)
                    enhanced_rgb = tensor_to_numpy(tensor_out.squeeze(0))
            elif method == "dcp":
                enhanced_rgb, _ = dcp_dehaze(frame_rgb, use_guided_filter=True)
            else:
                enhanced_rgb = frame_rgb

            enhanced_bgr = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
            writer.write(enhanced_bgr)
            frames_processed += 1

            if progress_callback:
                progress_callback(frames_processed, total_to_process, time.perf_counter() - t_start)

            if max_frames and frames_processed >= max_frames:
                break

    finally:
        cap.release()
        writer.release()

    total_time = time.perf_counter() - t_start
    avg_fps = frames_processed / max(total_time, 1e-4)

    # Audio preservation step using FFmpeg if available
    audio_preserved = False
    audio_msg = "Audio preservation skipped or FFmpeg unavailable."
    ffmpeg_exe = get_ffmpeg_path()

    if preserve_audio and ffmpeg_exe is not None and frames_processed > 0:
        try:
            # Mux original audio from in_file with enhanced video from temp_video_out
            cmd = [
                ffmpeg_exe,
                "-y",
                "-i",
                str(temp_video_out),
                "-i",
                str(in_file),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-shortest",
                str(out_file),
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
            if res.returncode == 0 and out_file.is_file() and out_file.stat().st_size > 0:
                audio_preserved = True
                audio_msg = "Audio track successfully preserved via FFmpeg muxing."
                temp_video_out.unlink(missing_ok=True)
            else:
                # If muxing failed, fallback to video-only file
                shutil.move(str(temp_video_out), str(out_file))
                audio_msg = "Audio muxing completed without audio stream (video-only preserved)."
        except Exception as ex:
            if temp_video_out.is_file():
                shutil.move(str(temp_video_out), str(out_file))
            audio_msg = f"Audio preservation failed ({ex}); video-only output preserved."
    else:
        # Move temp video to final output path
        if temp_video_out.is_file():
            shutil.move(str(temp_video_out), str(out_file))
        if ffmpeg_exe is None and preserve_audio:
            audio_msg = "FFmpeg executable not found in PATH; saved as video-only MP4."

    return {
        "status": "success",
        "input_file": str(in_file),
        "output_file": str(out_file),
        "method": method,
        "width": width,
        "height": height,
        "original_fps": round(fps, 2),
        "frames_processed": frames_processed,
        "total_duration_sec": round(total_time, 2),
        "processing_fps": round(avg_fps, 2),
        "audio_preserved": audio_preserved,
        "audio_message": audio_msg,
    }
