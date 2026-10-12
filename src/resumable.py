"""
Resumable chunked video processing and lossless assembly for FlashVSR.
Enables resilient long-video upscaling across Colab disconnects and timeouts.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import Optional, List, Union

from .media import MediaTools, MediaError


def get_resume_dir(destination: Union[str, Path]) -> Path:
    """Return the dedicated hidden checkpoint directory for a destination video."""
    dest = Path(destination).resolve()
    return dest.parent / f".{dest.name}_chunks"


def load_resume_manifest(chunks_dir: Path) -> Optional[dict]:
    """Read the resume manifest JSON file if it exists and is valid."""
    manifest_file = chunks_dir / "resume_manifest.json"
    if not manifest_file.is_file():
        return None
    try:
        with open(manifest_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "chunks" in data:
            return data
    except Exception:
        pass
    return None


def save_resume_manifest(chunks_dir: Path, manifest: dict) -> None:
    """Save the resume manifest JSON file atomically."""
    chunks_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = chunks_dir / "resume_manifest.json"
    temp_file = chunks_dir / "resume_manifest.json.tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    os.replace(temp_file, manifest_file)


def assemble_chunks(
    chunk_paths: List[str],
    source_video: Optional[str],
    destination: Union[str, Path],
    fps: float,
    keep_audio: bool = True
) -> bool:
    """
    Concatenate chunk MP4 files losslessly using FFmpeg concat demuxer (<1s).
    Optionally muxes audio from source_video if keep_audio is True.
    """
    if not chunk_paths:
        return False

    dest_path = Path(destination).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest_path.parent / f".{dest_path.name}.assembling.mp4"
    concat_list = dest_path.parent / f".{dest_path.name}.concat_list.txt"

    media = MediaTools()
    ffmpeg_bin = media.ffmpeg

    has_audio = False
    if keep_audio and source_video and os.path.exists(source_video):
        try:
            has_audio = media.has_audio(source_video)
        except Exception:
            has_audio = False

    try:
        with open(concat_list, "w", encoding="utf-8") as f:
            for cp in chunk_paths:
                abs_cp = str(Path(cp).resolve()).replace("'", "'\\''")
                f.write(f"file '{abs_cp}'\n")

        cmd = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_list),
        ]
        if has_audio and source_video:
            cmd += ["-i", str(Path(source_video).resolve())]

        cmd += ["-c:v", "copy"]

        if has_audio and source_video:
            cmd += [
                "-c:a", "aac", "-b:a", "192k",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest"
            ]
        else:
            cmd += ["-an"]

        if dest_path.suffix.lower() in (".mp4", ".mov"):
            cmd += ["-movflags", "+faststart"]

        cmd.append(str(temp_dest))

        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not temp_dest.is_file() or temp_dest.stat().st_size == 0:
            # Fallback to video-only concat if audio mux failed
            cmd_fallback = [
                ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "concat", "-safe", "0", "-i", str(concat_list),
                "-c:v", "copy", "-an",
            ]
            if dest_path.suffix.lower() in (".mp4", ".mov"):
                cmd_fallback += ["-movflags", "+faststart"]
            cmd_fallback.append(str(temp_dest))

            res_fb = subprocess.run(cmd_fallback, capture_output=True, text=True)
            if res_fb.returncode != 0 or not temp_dest.is_file() or temp_dest.stat().st_size == 0:
                raise MediaError(f"FFmpeg chunk assembly failed: {res_fb.stderr.strip() or res.stderr.strip()}")

        os.replace(temp_dest, dest_path)
        return True

    finally:
        if concat_list.is_file():
            concat_list.unlink(missing_ok=True)
        if temp_dest.is_file():
            temp_dest.unlink(missing_ok=True)


def cleanup_chunks(chunks_dir: Path) -> None:
    """Safely remove intermediate chunk files and manifest after successful assembly."""
    if not chunks_dir.is_dir():
        return
    for item in chunks_dir.iterdir():
        try:
            if item.is_file():
                item.unlink(missing_ok=True)
        except OSError:
            pass
    try:
        chunks_dir.rmdir()
    except OSError:
        pass
