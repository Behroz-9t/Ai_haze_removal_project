"""Video Processing and Offline Dehazing Module.

Responsible for:
- Video file reading/writing (MP4, AVI, MOV, MKV)
- Frame-by-frame dehazing pipelines
- Audio stream preservation via FFmpeg muxing
"""

from src.video.processor import get_ffmpeg_path, process_video_file

__all__ = ["get_ffmpeg_path", "process_video_file"]
