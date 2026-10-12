from __future__ import annotations
"""
Media utilities for FlashVSR.
Provides FFmpeg/FFprobe wrappers, hardware NVENC probing, atomic file output,
and streaming video encoding with fallback support.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

try:
    import numpy as np
except ImportError:
    np = None
logger = logging.getLogger("flashvsr")


class MediaError(RuntimeError):
    """Exception raised for media probe, encoding, or tool errors."""
    pass


def quality_to_crf(quality: Union[int, float] = 10) -> int:
    """
    Converts quality (0..10) to CRF/CQ value.
    Formula: int(26 - quality * 0.6) (quality 0..10 -> crf 26..20).
    """
    q = max(0.0, min(10.0, float(quality)))
    return int(26 - q * 0.6)


def find_ffmpeg() -> str:
    """Finds the ffmpeg executable on PATH or via imageio_ffmpeg, raising MediaError if missing."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        return ffmpeg_bin
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:
        pass
    raise MediaError("FFmpeg executable not found in PATH or environment.")


def find_ffprobe() -> str:
    """Finds the ffprobe executable on PATH, raising MediaError if missing."""
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin:
        return ffprobe_bin
    raise MediaError("FFprobe executable not found in PATH or environment.")


@contextmanager
def atomic_output(destination: Union[str, Path]):
    """
    Context manager yielding a temporary file in destination's parent directory.
    On normal exit, atomically replaces destination with the temp file.
    On exception, unlinks temp file and preserves existing destination.
    """
    dest = Path(destination).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path_str = tempfile.mkstemp(
        dir=dest.parent,
        prefix=f".tmp_{dest.stem}_",
        suffix=dest.suffix,
    )
    os.close(fd)
    temp_path = Path(temp_path_str)
    try:
        yield temp_path
        if temp_path.exists():
            temp_path.replace(dest)
    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise


def _prepare_frames(frames: Any) -> np.ndarray:
    """
    Converts frame input (tensor, array, list) into a uint8 RGB ndarray
    of shape (N, H, W, 3).
    """
    global np
    if np is None:
        try:
            import numpy as np
        except ImportError:
            raise MediaError("NumPy is required to convert and stream video frames.")

    if hasattr(frames, "detach"):
        frames = frames.detach().cpu().numpy()
    elif isinstance(frames, (list, tuple)):
        frames = np.asarray(frames)
    elif not isinstance(frames, np.ndarray):
        frames = np.asarray(frames)
    # Normalize dimensions
    if frames.ndim == 3:
        # (H, W, C) or (C, H, W)
        if frames.shape[0] in (1, 3, 4) and frames.shape[-1] not in (1, 3, 4):
            frames = np.transpose(frames, (1, 2, 0))
        frames = frames[np.newaxis, ...]
    elif frames.ndim == 4:
        # (N, H, W, C) or (N, C, H, W)
        if frames.shape[1] in (1, 3, 4) and frames.shape[-1] not in (1, 3, 4):
            frames = np.transpose(frames, (0, 2, 3, 1))
    else:
        raise MediaError(f"Unsupported frame array dimensions: {frames.ndim} (expected 3 or 4)")

    # Normalize dtype to uint8 [0, 255]
    if np.issubdtype(frames.dtype, np.floating):
        if frames.min() < 0.0:
            frames = (frames + 1.0) * 127.5
        elif frames.max() <= 1.0:
            frames = frames * 255.0
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    elif frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)

    # If alpha channel present, slice RGB
    if frames.shape[-1] == 4:
        frames = frames[..., :3]

    return frames


class MediaTools:
    """Wrapper for media probing, capability detection, and streaming encoding."""

    def __init__(
        self,
        ffmpeg: Optional[str] = None,
        ffprobe: Optional[str] = None,
    ):
        if ffmpeg is not None:
            if not os.path.isfile(ffmpeg) and not shutil.which(ffmpeg):
                raise MediaError(f"FFmpeg binary not found at '{ffmpeg}'")
            self.ffmpeg = ffmpeg
        else:
            self.ffmpeg = find_ffmpeg()

        if ffprobe is not None:
            if not os.path.isfile(ffprobe) and not shutil.which(ffprobe):
                raise MediaError(f"FFprobe binary not found at '{ffprobe}'")
            self.ffprobe = ffprobe
        else:
            try:
                self.ffprobe = find_ffprobe()
            except MediaError:
                self.ffprobe = None

    find_ffmpeg = staticmethod(find_ffmpeg)
    find_ffprobe = staticmethod(find_ffprobe)

    def probe(self, path: Union[str, Path]) -> Dict[str, Any]:
        """Runs ffprobe on the target file, returning JSON format and stream info."""
        ffprobe_bin = self.ffprobe or find_ffprobe()
        path_str = str(Path(path).resolve())
        if not os.path.exists(path_str):
            raise MediaError(f"File not found: {path_str}")

        cmd = [
            ffprobe_bin,
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            path_str,
        ]
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                raise MediaError(f"ffprobe failed for '{path_str}': {proc.stderr}")
            return json.loads(proc.stdout)
        except json.JSONDecodeError as e:
            raise MediaError(f"Failed to parse ffprobe output for '{path_str}'") from e
        except Exception as e:
            if isinstance(e, MediaError):
                raise
            raise MediaError(f"ffprobe execution error: {e}") from e

    def has_audio(self, path: Union[str, Path]) -> bool:
        """Returns True if the target media file contains at least one audio stream."""
        try:
            info = self.probe(path)
            streams = info.get("streams", [])
            return any(s.get("codec_type") == "audio" for s in streams)
        except Exception:
            return False

    def verify_video(self, path: Union[str, Path]) -> bool:
        """Sanity checks video file: exists, non-empty, and has a valid video stream."""
        p = Path(path)
        if not p.is_file() or p.stat().st_size == 0:
            return False
        try:
            info = self.probe(p)
            streams = info.get("streams", [])
            return any(s.get("codec_type") == "video" for s in streams)
        except Exception:
            return False

    @staticmethod
    def hevc_nvenc_arguments(quality: int = 10, device_index: int = 0) -> List[str]:
        """Returns FFmpeg argument list for HEVC NVENC encoding."""
        cq = quality_to_crf(quality)
        return [
            "-c:v", "hevc_nvenc",
            "-gpu", str(device_index),
            "-preset", "p4",
            "-rc:v", "vbr",
            "-cq", str(cq),
            "-pix_fmt", "yuv420p",
        ]

    @staticmethod
    def h264_nvenc_arguments(quality: int = 10, device_index: int = 0) -> List[str]:
        """Returns FFmpeg argument list for H.264 NVENC encoding."""
        cq = quality_to_crf(quality)
        return [
            "-c:v", "h264_nvenc",
            "-gpu", str(device_index),
            "-preset", "p4",
            "-rc:v", "vbr",
            "-cq", str(cq),
            "-pix_fmt", "yuv420p",
        ]

    @staticmethod
    def cfr_arguments() -> List[str]:
        """Returns FFmpeg arguments to force constant frame rate."""
        return ["-fps_mode", "cfr"]

    @staticmethod
    @lru_cache(maxsize=16)
    def nvenc_works(device_index: int = 0, quality: int = 10) -> bool:
        """Caches probe of HEVC NVENC capability via a 40-frame testsrc encoding test."""
        try:
            ffmpeg_bin = find_ffmpeg()
            args = MediaTools.hevc_nvenc_arguments(quality=quality, device_index=device_index)
            cmd = [
                ffmpeg_bin,
                "-y",
                "-f", "lavfi",
                "-i", "testsrc=duration=2:size=128x128:rate=30",
                "-frames:v", "40",
                *args,
                "-f", "null",
                "-",
            ]
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
            )
            return proc.returncode == 0
        except Exception:
            return False

    @staticmethod
    @lru_cache(maxsize=16)
    def _h264_nvenc_works(device_index: int = 0) -> bool:
        """Caches probe of H.264 NVENC capability via a 40-frame testsrc encoding test."""
        try:
            ffmpeg_bin = find_ffmpeg()
            cmd = [
                ffmpeg_bin,
                "-y",
                "-f", "lavfi",
                "-i", "testsrc=duration=2:size=128x128:rate=30",
                "-frames:v", "40",
                "-c:v", "h264_nvenc",
                "-gpu", str(device_index),
                "-preset", "p4",
                "-f", "null",
                "-",
            ]
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
            )
            return proc.returncode == 0
        except Exception:
            return False

    def run(
        self,
        cmd: Union[List[str], str],
        input_bytes: Optional[bytes] = None,
        check: bool = True,
        **kwargs,
    ) -> subprocess.CompletedProcess:
        """Wraps subprocess execution with stderr capture and MediaError on failure."""
        if isinstance(cmd, str):
            import shlex
            cmd = shlex.split(cmd)
        try:
            proc = subprocess.run(
                cmd,
                input=input_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                **kwargs,
            )
            if check and proc.returncode != 0:
                err_msg = (
                    proc.stderr.decode("utf-8", errors="replace")
                    if isinstance(proc.stderr, bytes)
                    else str(proc.stderr)
                )
                cmd_str = " ".join(str(c) for c in cmd)
                raise MediaError(f"Command '{cmd_str}' failed (code {proc.returncode}):\n{err_msg}")
            return proc
        except subprocess.SubprocessError as e:
            raise MediaError(f"Subprocess execution error: {e}") from e

    def _pipe_frames(
        self,
        cmd: List[str],
        initial_chunks: List[np.ndarray],
        chunk_iterator: Iterable,
    ) -> None:
        """Streams frame chunks into FFmpeg stdin pipe and waits for process completion."""
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            for c in initial_chunks:
                proc.stdin.write(c.tobytes())
            for chunk in chunk_iterator:
                c_np = _prepare_frames(chunk)
                proc.stdin.write(c_np.tobytes())
            proc.stdin.close()
            stdout = proc.stdout.read() if proc.stdout else b""
            stderr = proc.stderr.read() if proc.stderr else b""
            proc.wait()
            if proc.returncode != 0:
                err_msg = stderr.decode("utf-8", errors="replace") if stderr else ""
                raise MediaError(f"FFmpeg encoding failed (exit code {proc.returncode}):\n{err_msg}")
        except Exception as e:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            if isinstance(e, MediaError):
                raise
            raise MediaError(f"Failed while streaming frames to FFmpeg: {e}") from e

    def save_video(
        self,
        frames_iter: Iterable,
        destination: Union[str, Path],
        fps: Union[int, float],
        quality: int = 10,
        audio_source: Optional[Union[str, Path]] = None,
        device_index: int = 0,
    ) -> Path:
        """
        Saves video frames to destination using streaming FFmpeg pipe and atomic rename.
        Selects HEVC/H.264 NVENC hardware encoder with automatic libx264 software fallback.
        Preserves audio track from audio_source if provided and present.
        """
        dest = Path(destination).resolve()
        iterator = iter(frames_iter)
        try:
            first_chunk = next(iterator)
        except StopIteration:
            raise MediaError("No frames provided to save_video.")

        first_np = _prepare_frames(first_chunk)
        if first_np.ndim != 4 or first_np.shape[-1] != 3:
            raise MediaError(f"Expected frame chunks with shape (N, H, W, 3), got shape {first_np.shape}")

        height = first_np.shape[1]
        width = first_np.shape[2]
        ext = dest.suffix.lower()

        # Check audio source
        audio_present = False
        if audio_source is not None and os.path.exists(str(audio_source)):
            audio_present = self.has_audio(audio_source)

        # Codec selection: hardware NVENC with software fallback
        use_nvenc = False
        if ext == ".avi":
            if self._h264_nvenc_works(device_index=device_index):
                video_codec_args = self.h264_nvenc_arguments(quality=quality, device_index=device_index)
                use_nvenc = True
            else:
                video_codec_args = [
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", str(quality_to_crf(quality)),
                    "-pix_fmt", "yuv420p",
                ]
        elif ext == ".webm":
            video_codec_args = [
                "-c:v", "libvpx-vp9",
                "-crf", str(quality_to_crf(quality)),
                "-b:v", "0",
                "-pix_fmt", "yuv420p",
            ]
        else:  # Default: .mp4, .mkv, .mov, etc.
            if self.nvenc_works(device_index=device_index, quality=quality):
                video_codec_args = self.hevc_nvenc_arguments(quality=quality, device_index=device_index)
                use_nvenc = True
            else:
                video_codec_args = [
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", str(quality_to_crf(quality)),
                    "-pix_fmt", "yuv420p",
                ]

        def _build_cmd(target_path: str, codec_args: List[str]) -> List[str]:
            cmd = [
                self.ffmpeg,
                "-y",
                "-f", "rawvideo",
                "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}",
                "-r", str(fps),
                "-i", "-",
            ]
            if audio_present:
                cmd += ["-i", str(audio_source)]
            cmd += codec_args
            if width % 2 != 0 or height % 2 != 0:
                cmd += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
            if audio_present:
                cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
            if ext in [".mp4", ".mov"]:
                cmd += ["-movflags", "+faststart"]
            cmd.append(str(target_path))
            return cmd

        is_reiterable = isinstance(frames_iter, (list, tuple))

        with atomic_output(dest) as temp_dest:
            cmd = _build_cmd(str(temp_dest), video_codec_args)
            try:
                self._pipe_frames(cmd, [first_np], iterator)
            except MediaError as e:
                # If hardware NVENC was attempted and failed, try software fallback if reiterable
                if use_nvenc and is_reiterable:
                    logger.warning("Hardware NVENC encoding failed (%s); retrying with libx264 software encoder.", e)
                    fallback_args = [
                        "-c:v", "libx264",
                        "-preset", "fast",
                        "-crf", str(quality_to_crf(quality)),
                        "-pix_fmt", "yuv420p",
                    ]
                    cmd_fallback = _build_cmd(str(temp_dest), fallback_args)
                    self._pipe_frames(cmd_fallback, [], iter(frames_iter))
                else:
                    raise

        return dest
