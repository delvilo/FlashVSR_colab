"""
Hardware-Accelerated Video Encoder for FlashVSR (Linux & Google Colab Native)
Provides:
- NVENC GPU-accelerated encoding (h264_nvenc) with libx264 fallback
- Single-pass Audio Muxing without leftover temporary files
- Flexible filename templating with date (%date:yyyyMMdd_HHmmss%), dimensions (%width%x%height%),
  subfolders, and automatic sequence numbering.
"""

import os
import sys
import re
import shutil
import subprocess
from datetime import datetime
from typing import Optional, Union, Generator, Iterable
import torch
import numpy as np


_NVENC_AVAILABLE: Optional[bool] = None


def find_ffmpeg() -> str:
    """Finds the ffmpeg binary on Linux/Colab."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    raise RuntimeError("FFmpeg was not found. Please install ffmpeg (`apt-get install -y ffmpeg` or `pip install imageio-ffmpeg`).")


def has_nvenc(ffmpeg_bin: Optional[str] = None) -> bool:
    """Checks whether the system and FFmpeg support h264_nvenc hardware encoding."""
    global _NVENC_AVAILABLE
    if _NVENC_AVAILABLE is not None:
        return _NVENC_AVAILABLE

    if not torch.cuda.is_available():
        _NVENC_AVAILABLE = False
        return False

    try:
        bin_path = ffmpeg_bin or find_ffmpeg()
        proc = subprocess.run([bin_path, "-encoders"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
        _NVENC_AVAILABLE = "h264_nvenc" in proc.stdout
    except Exception:
        _NVENC_AVAILABLE = False

    return _NVENC_AVAILABLE


def resolve_filename_prefix(
    prefix: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    now: Optional[datetime] = None,
    out_dir: str = "output",
    extension: str = ".mp4",
) -> str:
    """
    Expands format variables:
      %date:yyyyMMdd_HHmmss% -> 20260831_154230
      %year%, %month%, %day%, %hour%, %minute%, %second%
      %width%, %height%
    Creates parent directories if required and generates sequential numbering if not explicit.
    """
    if now is None:
        now = datetime.now()

    result = prefix or "FlashVSR"

    # 1. Custom date pattern: %date:yyyy-MM-dd_HH-mm-ss%
    def _replace_date(match):
        fmt = match.group(1)
        # Convert C#/Java date tokens to Python strftime
        token_map = [
            ("yyyy", now.strftime("%Y")),
            ("yy", now.strftime("%y")),
            ("MM", now.strftime("%m")),
            ("dd", now.strftime("%d")),
            ("HH", now.strftime("%H")),
            ("mm", now.strftime("%M")),
            ("ss", now.strftime("%S")),
            ("fff", f"{int(now.microsecond / 1000):03d}"),
        ]
        res = fmt
        for k, v in token_map:
            res = res.replace(k, v)
        return res

    result = re.sub(r"%date:([^%]+)%", _replace_date, result)

    # 2. Legacy time tokens
    result = result.replace("%year%", now.strftime("%Y"))
    result = result.replace("%month%", now.strftime("%m"))
    result = result.replace("%day%", now.strftime("%d"))
    result = result.replace("%hour%", now.strftime("%H"))
    result = result.replace("%minute%", now.strftime("%M"))
    result = result.replace("%second%", now.strftime("%S"))

    # 3. Dimension tokens
    if width is not None:
        result = result.replace("%width%", str(width))
    if height is not None:
        result = result.replace("%height%", str(height))

    # 4. Resolve path
    if not os.path.isabs(result):
        full_path = os.path.join(out_dir, result)
    else:
        full_path = result

    dir_name = os.path.dirname(full_path)
    base_name = os.path.basename(full_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    # If extension not included, add sequential numbering
    if not base_name.endswith(extension):
        # Scan dir for next sequence number
        pattern = re.compile(rf"^{re.escape(base_name)}_(\d+){re.escape(extension)}$")
        max_idx = 0
        if os.path.exists(dir_name or "."):
            for fname in os.listdir(dir_name or "."):
                m = pattern.match(fname)
                if m:
                    max_idx = max(max_idx, int(m.group(1)))
        final_filename = f"{base_name}_{max_idx + 1:05d}{extension}"
        final_path = os.path.join(dir_name, final_filename)
    else:
        final_path = full_path

    return os.path.abspath(final_path)


def encode_video_ffmpeg(
    frames: Union[torch.Tensor, Iterable[torch.Tensor], np.ndarray],
    output_path: str,
    fps: float = 30.0,
    codec: Optional[str] = None,
    crf: int = 18,
    audio_path: Optional[str] = None,
    preset: str = "p4",
) -> str:
    """
    Pipes rawvideo frames directly into FFmpeg stdin.
    Accelerated with NVENC if available, and muxes audio in a single pass.
    """
    ffmpeg_bin = find_ffmpeg()
    use_nvenc = has_nvenc(ffmpeg_bin) if (codec is None or codec == "auto") else (codec == "h264_nvenc")

    # Detect dimensions from first frame
    if isinstance(frames, torch.Tensor):
        # Tensor can be (N, H, W, C) or (N, C, H, W)
        if frames.shape[-1] in (1, 3, 4):
            N, H, W, C = frames.shape
            frame_iter = (f for f in frames)
            is_chw = False
        else:
            N, C, H, W = frames.shape
            frame_iter = (f for f in frames)
            is_chw = True
    elif isinstance(frames, (list, tuple)):
        f0 = frames[0]
        if isinstance(f0, torch.Tensor):
            if f0.shape[-1] in (1, 3, 4):
                H, W = f0.shape[0], f0.shape[1]
                is_chw = False
            else:
                H, W = f0.shape[1], f0.shape[2]
                is_chw = True
        else:
            H, W = f0.shape[0], f0.shape[1]
            is_chw = False
        frame_iter = iter(frames)
    elif isinstance(frames, Generator) or hasattr(frames, "__iter__"):
        # Peek first item
        f0 = next(frames)
        if isinstance(f0, torch.Tensor):
            if f0.shape[-1] in (1, 3, 4):
                H, W = f0.shape[0], f0.shape[1]
                is_chw = False
            else:
                H, W = f0.shape[1], f0.shape[2]
                is_chw = True
        else:
            H, W = f0.shape[0], f0.shape[1]
            is_chw = False

        def _rechain():
            yield f0
            yield from frames

        frame_iter = _rechain()
    else:
        raise ValueError(f"Unsupported frames type: {type(frames)}")

    # Ensure even dimensions (required by libx264 and yuv420p)
    W = W - (W % 2)
    H = H - (H % 2)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    cmd = [
        ffmpeg_bin,
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}",
        "-r", str(fps),
        "-i", "-",  # read video frames from stdin pipe
    ]

    # Add audio input if provided
    has_audio = audio_path and os.path.exists(audio_path)
    if has_audio:
        cmd += ["-i", audio_path]

    # Video codec settings
    if use_nvenc:
        cmd += [
            "-c:v", "h264_nvenc",
            "-preset", preset,
            "-rc:v", "vbr",
            "-cq", str(crf),
            "-pix_fmt", "yuv420p",
        ]
    else:
        cmd += [
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
        ]

    # Audio codec settings
    if has_audio:
        cmd += [
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",  # finish when video finishes
        ]

    cmd.append(output_path)

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        for f in frame_iter:
            if isinstance(f, torch.Tensor):
                f_cpu = f.detach().cpu()
                if is_chw:
                    f_cpu = f_cpu.permute(1, 2, 0)
                # Normalize float [0, 1] or [-1, 1] to uint8 [0, 255]
                if f_cpu.is_floating_point():
                    if f_cpu.min() < 0.0:
                        f_cpu = (f_cpu + 1.0) * 127.5
                    elif f_cpu.max() <= 1.0:
                        f_cpu = f_cpu * 255.0
                    f_bytes = f_cpu.clamp(0, 255).byte().numpy().tobytes()
                else:
                    f_bytes = f_cpu.numpy().tobytes()
            elif isinstance(f, np.ndarray):
                if f.dtype != np.uint8:
                    if f.min() < 0.0:
                        f = (f + 1.0) * 127.5
                    elif f.max() <= 1.0:
                        f = f * 255.0
                    f = np.clip(f, 0, 255).astype(np.uint8)
                f_bytes = f.tobytes()
            else:
                f_bytes = bytes(f)

            proc.stdin.write(f_bytes)

        proc.stdin.close()
        stderr_output = proc.stderr.read().decode("utf-8", errors="replace")
        proc.stderr.close()
        proc.wait()

        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg encoding failed (code {proc.returncode}):\n{stderr_output}")

    except Exception as e:
        proc.kill()
        raise e

    return output_path
