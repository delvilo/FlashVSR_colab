#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlashVSR Command-Line Interface
===============================

A mirror-grade CLI that maps 1:1 with the ComfyUI node inputs.
All parameters from FlashVSRNode, FlashVSRNodeAdv, and FlashVSRNodeInitPipe
are exposed as command-line arguments.

Usage:
    python cli_main.py --input video.mp4 --output upscaled.mp4 --scale 2

For full help:
    python cli_main.py --help
"""

import argparse
import os
import sys
import gc
import torch
import numpy as np
# =============================================================================
# CLI argument parsing - EXHAUSTIVE mapping from ComfyUI node INPUT_TYPES
# =============================================================================

def parse_args():
    """
    Parse command-line arguments.
    
    Every argument corresponds directly to a parameter in the ComfyUI node
    INPUT_TYPES (FlashVSRNode, FlashVSRNodeAdv, FlashVSRNodeInitPipe).
    """
    parser = argparse.ArgumentParser(
        description="FlashVSR CLI - Video Super Resolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic 2x upscale with defaults
    python cli_main.py --input video.mp4 --output upscaled.mp4 --scale 2

    # 4x upscale with tiling enabled for lower VRAM
    python cli_main.py --input video.mp4 --output upscaled.mp4 --scale 4 \\
        --tiled_vae --tiled_dit --tile_size 256 --tile_overlap 24

    # Long video with chunking to prevent OOM
    python cli_main.py --input long_video.mp4 --output upscaled.mp4 \\
        --frame_chunk_size 50 --mode tiny-long

    # Low VRAM mode (8GB GPUs)
    python cli_main.py --input video.mp4 --output upscaled.mp4 --scale 2 \\
        --vae_model LightVAE_W2.1 --tiled_vae --tiled_dit \\
        --frame_chunk_size 20 --resize_factor 0.5

For more information, visit: https://github.com/naxci1/ComfyUI-FlashVSR_Stable
"""
    )

    # ==========================================================================
    # Required arguments
    # ==========================================================================
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input video file path (e.g., video.mp4)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output video file path (e.g., upscaled.mp4)'
    )

    # ==========================================================================
    # FlashVSRNodeInitPipe parameters (Pipeline Initialization)
    # ==========================================================================
    parser.add_argument(
        '--model',
        type=str,
        choices=['FlashVSR', 'FlashVSR-v1.1'],
        default='FlashVSR-v1.1',
        help='FlashVSR model version. V1.1 is recommended for better stability. (default: FlashVSR-v1.1)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['tiny', 'tiny-long', 'full'],
        default='tiny',
        help='Operation mode. "tiny": faster, standard memory. "tiny-long": optimized for long videos (lower VRAM). "full": higher quality but max VRAM. (default: tiny)'
    )
    parser.add_argument(
        '--vae_model',
        type=str,
        choices=['Wan2.1', 'Wan2.2', 'LightVAE_W2.1', 'TAE_W2.2', 'LightTAE_HY1.5'],
        default='Wan2.1',
        help='VAE model: Wan2.1 (default), Wan2.2, LightVAE_W2.1 (50%% less VRAM), TAE_W2.2, LightTAE_HY1.5. Auto-downloads if missing. (default: Wan2.1)'
    )
    parser.add_argument(
        '--force_offload',
        action='store_true',
        default=True,
        help='Force offloading of models to CPU RAM after execution to free up VRAM. (default: True)'
    )
    parser.add_argument(
        '--no_force_offload',
        action='store_true',
        help='Disable force offloading (keeps models in VRAM).'
    )
    parser.add_argument(
        '--precision',
        type=str,
        choices=['fp16', 'bf16', 'auto'],
        default='auto',
        help="Inference precision. 'auto' selects bf16 if supported (RTX 30/40/50 series), otherwise fp16. (default: auto)"
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Computation device (e.g., "cuda:0", "cuda:1", "cpu", "auto"). (default: auto)'
    )
    parser.add_argument(
        '--attention_mode',
        type=str,
        choices=['auto', 'sdpa', 'sparse_sage_attention', 'block_sparse_attention', 'flash_attention_2'],
        default='auto',
        help='Attention mechanism backend. "auto" selects optimal for GPU (sdpa for T4/Colab, sparse_sage for RTX 30/40/50). (default: auto)'
    )

    # ==========================================================================
    # FlashVSRNodeAdv parameters (Processing)
    # ==========================================================================
    parser.add_argument(
        '--scale',
        type=int,
        choices=[2, 4],
        default=2,
        help='Upscaling factor. 2x or 4x. Higher scale requires more VRAM and compute. (default: 2)'
    )
    parser.add_argument(
        '--color_fix',
        action='store_true',
        default=True,
        help='Apply wavelet-based color correction to match output colors with input. (default: True)'
    )
    parser.add_argument(
        '--no_color_fix',
        action='store_true',
        help='Disable color correction.'
    )
    parser.add_argument(
        '--tiled_vae',
        action='store_true',
        default=False,
        help='Enable spatial tiling for the VAE decoder. Reduces VRAM usage significantly but is slower.'
    )
    parser.add_argument(
        '--tiled_dit',
        action='store_true',
        default=False,
        help='Enable spatial tiling for the Diffusion Transformer (DiT). Crucial for saving VRAM on large inputs.'
    )
    parser.add_argument(
        '--tile_size',
        type=int,
        default=256,
        help='Size of the tiles for DiT processing (32-1024). Smaller = less VRAM, more tiles, slower. (default: 256)'
    )
    parser.add_argument(
        '--tile_overlap',
        type=int,
        default=24,
        help='Overlap pixels between tiles to blend seams (8-512). Higher = smoother transitions. (default: 24)'
    )
    parser.add_argument(
        '--unload_dit',
        action='store_true',
        default=False,
        help='Unload the DiT model from VRAM before VAE decoding starts. Use if VAE decode runs out of memory.'
    )
    parser.add_argument(
        '--sparse_ratio',
        type=float,
        default=2.0,
        help='Control for sparse attention (1.5-2.0). 1.5 is faster, 2.0 is more stable/quality. (default: 2.0)'
    )
    parser.add_argument(
        '--kv_ratio',
        type=float,
        default=3.0,
        help='Key/Value cache ratio (1.0-3.0). 1.0 uses less VRAM; 3.0 provides highest quality retention. (default: 3.0)'
    )
    parser.add_argument(
        '--local_range',
        type=int,
        choices=[9, 11],
        default=11,
        help='Local attention range window. 9 = sharper details; 11 = more stable/consistent results. (default: 11)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for noise generation. Same seed + same settings = reproducible results. (default: 0)'
    )
    parser.add_argument(
        '--frame_chunk_size',
        type=int,
        default=0,
        help='Process video in chunks of N frames to prevent VRAM OOM. 0 = Process all frames at once. (default: 0)'
    )
    parser.add_argument(
        '--enable_debug',
        action='store_true',
        default=False,
        help='Enable verbose logging to console. Shows VRAM usage, step times, tile info, and detailed progress.'
    )
    parser.add_argument(
        '--keep_models_on_cpu',
        action='store_true',
        default=True,
        help='Move models to CPU RAM instead of keeping them in VRAM when not in use. (default: True)'
    )
    parser.add_argument(
        '--no_keep_models_on_cpu',
        action='store_true',
        help='Keep models in VRAM (faster but uses more VRAM).'
    )
    parser.add_argument(
        '--resize_factor',
        type=float,
        default=1.0,
        help='Resize input frames before processing (0.1-1.0). Set to 0.5 for large 1080p+ videos. (default: 1.0)'
    )
    # ==========================================================================
    # TE-Speed Optimization parameters (Features 1, 2, 3, 5, 6)
    # ==========================================================================
    parser.add_argument(
        '--quality_profile',
        type=str,
        choices=['detail', 'balanced', 'throughput'],
        default='balanced',
        help='TE-Speed dynamic attention budget profile (default: balanced)'
    )
    parser.add_argument(
        '--motion_intensity',
        type=float,
        default=1.0,
        help='Motion sensitivity multiplier for dynamic budget (default: 1.0)'
    )
    parser.add_argument(
        '--spatial_strategy',
        type=str,
        choices=['auto', 'full_frame', 'adaptive_tiles'],
        default='auto',
        help='Spatial partitioning strategy: auto (dynamic), full_frame, or adaptive_tiles (default: auto)'
    )
    parser.add_argument(
        '--memory_policy',
        type=str,
        choices=['auto', 'staged', 'resident'],
        default='auto',
        help='VRAM memory policy: auto (auto-detect based on VRAM/size), staged (swap DiT/VAE to avoid OOM), or resident (default: auto)'
    )
    parser.add_argument(
        '--audio',
        type=str,
        default=None,
        help='Audio file path to mux into final video. (default: extract from input video if present)'
    )
    parser.add_argument(
        '--no_validate_outputs',
        action='store_true',
        help='Disable NaN/Inf numerical sanity checks.'
    )

    # ==========================================================================
    # Video I/O parameters
    # ==========================================================================
    parser.add_argument(
        '--fps',
        type=float,
        default=None,
        help='Output video FPS. If not specified, uses input video FPS.'
    )
    # Output quality (replaces --crf/--codec)
    parser.add_argument(
        '--quality',
        type=int,
        default=8,
        metavar='Q',
        help='Output video quality 0–10 (10=best, maps to CRF/CQ). Default: 8'
    )
    parser.add_argument(
        '--start_frame',
        type=int,
        default=0,
        help='Start processing from this frame index (0-indexed). (default: 0)'
    )
    parser.add_argument(
        '--end_frame',
        type=int,
        default=-1,
        help='Stop processing at this frame index (-1 = process all). (default: -1)'
    )

    # ==========================================================================
    # Model paths (optional, for custom model locations)
    # ==========================================================================
    parser.add_argument(
        '--models_dir',
        type=str,
        default=None,
        help='Custom path to FlashVSR models directory. If not set, uses ComfyUI default or ./models'
    )
    # Audio
    parser.add_argument(
        '--keep_audio',
        action='store_true',
        default=False,
        help='Copy audio from input video to output. Requires FFmpeg.'
    )
    # Logging
    parser.add_argument(
        '--log_level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Log verbosity (default: INFO)'
    )
    parser.add_argument(
        '--log_file',
        type=str,
        default=None,
        help='Append JSONL structured logs to this file.'
    )

    return parser.parse_args()


# =============================================================================
# Video I/O utilities
# =============================================================================

# =============================================================================
# Video I/O utilities (Stream based)
# =============================================================================

try:
    from torchcodec.decoders import VideoDecoder
    TORCHCODEC_AVAILABLE = True
except (ImportError, OSError):
    TORCHCODEC_AVAILABLE = False


class VideoReader:
    """
    High-performance video reader powered by TorchCodec with pure FFmpeg pipe streaming fallback.
    Zero OpenCV dependency.
    """
    def __init__(self, video_path, start_frame=0, end_frame=-1, chunk_size=0):
        self.video_path = video_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.chunk_size = chunk_size
        self.decoder = None
        self.proc = None
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")

        # 1. Try TorchCodec first (native tensor decoding)
        if TORCHCODEC_AVAILABLE:
            try:
                self.decoder = VideoDecoder(video_path)
                meta = self.decoder.metadata
                self.width = int(meta.width)
                self.height = int(meta.height)
                self.fps = float(meta.average_fps_from_header or 30.0)
                self.total_frames = int(meta.num_frames_from_header or len(self.decoder))
            except Exception:
                self.decoder = None

        # 2. Pure FFmpeg pipe fallback (zero OpenCV)
        if self.decoder is None:
            from src.media import MediaTools
            media = MediaTools()
            info = media.probe(video_path)
            v_stream = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
            if not v_stream:
                raise RuntimeError(f"No video stream found in: {video_path}")
            self.width = int(v_stream.get("width", 0))
            self.height = int(v_stream.get("height", 0))
            self.total_frames = int(v_stream.get("nb_frames") or 0)
            if self.total_frames == 0:
                dur = float(v_stream.get("duration") or info.get("format", {}).get("duration", 0) or 0)
                from fractions import Fraction
                r_fps = float(Fraction(v_stream.get("avg_frame_rate", "30/1"))) if v_stream.get("avg_frame_rate") else 30.0
                self.total_frames = round(dur * r_fps) if dur > 0 else 0
            avg_fps = v_stream.get("avg_frame_rate", "30/1")
            try:
                from fractions import Fraction
                self.fps = float(Fraction(avg_fps)) if avg_fps else 30.0
            except Exception:
                self.fps = 30.0

        # Adjust end_frame
        if self.end_frame < 0 or self.end_frame > self.total_frames:
            self.end_frame = self.total_frames
            
        if self.start_frame >= self.total_frames and self.total_frames > 0:
            print(f"Warning: Start frame {self.start_frame} is beyond total frames {self.total_frames}.")
            self.end_frame = self.start_frame

        self.current_frame = self.start_frame
        self.frame_bytes = self.width * self.height * 3

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_frame >= self.end_frame:
            self.release()
            raise StopIteration

        frames_to_read = self.chunk_size if self.chunk_size > 0 else (self.end_frame - self.current_frame)
        frames_to_read = min(frames_to_read, self.end_frame - self.current_frame)
        
        if frames_to_read <= 0:
            self.release()
            raise StopIteration

        target_start = self.current_frame
        target_end = target_start + frames_to_read

        # Path A: TorchCodec direct zero-copy tensor slicing
        if self.decoder is not None:
            try:
                # TorchCodec outputs [N, C, H, W] uint8
                # FlashVSR pipeline expects [N, H, W, C] float32 in [0, 1]
                frame_batch = self.decoder.get_frames_in_range(target_start, target_end).data
                tensor_out = frame_batch.permute(0, 2, 3, 1).float() / 255.0
                self.current_frame = target_end
                return tensor_out.contiguous()
            except Exception:
                # Fallback to FFmpeg if range error occurs
                pass

        # Path B: Pure FFmpeg pipe streaming
        import subprocess
        import numpy as np
        from src.media import MediaTools
        
        if self.proc is None:
            ffmpeg_bin = MediaTools().ffmpeg
            cmd = [
                ffmpeg_bin, "-nostdin", "-hide_banner", "-loglevel", "error",
            ]
            if target_start > 0:
                cmd += ["-ss", str(target_start / max(self.fps, 1.0))]
            cmd += [
                "-i", self.video_path,
                "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"
            ]
            self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        bytes_needed = frames_to_read * self.frame_bytes
        raw_bytes = self.proc.stdout.read(bytes_needed)
        actual_frames = len(raw_bytes) // self.frame_bytes
        
        if actual_frames <= 0:
            self.release()
            raise StopIteration

        np_arr = np.frombuffer(raw_bytes[:actual_frames * self.frame_bytes], dtype=np.uint8).reshape((actual_frames, self.height, self.width, 3)).copy()
        self.current_frame += actual_frames
        return (torch.from_numpy(np_arr).float() / 255.0).contiguous()

    def release(self):
        self.decoder = None
        if self.proc:
            try:
                if self.proc.stdout:
                    self.proc.stdout.close()
                if self.proc.stderr:
                    self.proc.stderr.close()
                self.proc.wait()
            except Exception:
                pass
            self.proc = None

    def get_info(self):
        return self.fps, self.total_frames


class VideoWriter:
    """
    Incremental video writer powered by FFmpeg (HEVC NVENC / libx264).
    Zero OpenCV dependency.
    """
    def __init__(self, output_path, fps, width, height, quality=8, audio_path=None, **kwargs):
        self.output_path = output_path
        self.width = width - (width % 2)
        self.height = height - (height % 2)
        self.fps = fps
        self.audio_path = audio_path
        self.proc = None
        
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        import subprocess
        from src.media import MediaTools, MediaError
        media = MediaTools()
        ffmpeg_bin = media.ffmpeg
        crf = int(26 - quality * 0.6)
        suffix = os.path.splitext(output_path)[1].lower()
        use_hevc = media.nvenc_works(0, quality) if suffix in {'.mp4', '.mov', '.mkv'} else False
        use_h264nvenc = media._h264_nvenc_works(0) if suffix == '.avi' else False
        
        cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps),
            "-i", "-",
        ]
        has_audio = False
        if self.audio_path and os.path.exists(self.audio_path):
            try:
                has_audio = media.has_audio(self.audio_path)
            except Exception:
                has_audio = False
            if has_audio:
                cmd += ["-i", self.audio_path]
        
        if use_hevc:
            cmd += list(media.hevc_nvenc_arguments(quality, 0))
        elif use_h264nvenc:
            cmd += ["-c:v", "h264_nvenc", "-preset", "p4", "-rc:v", "vbr", "-cq", str(crf), "-pix_fmt", "yuv420p"]
        else:
            cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", str(crf), "-pix_fmt", "yuv420p"]
        
        if suffix in {'.mp4', '.mov'}:
            cmd += ["-movflags", "+faststart"]
            
        if has_audio:
            cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
            
        cmd.append(self.output_path)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frames_tensor):
        import torch
        import numpy as np
        
        if self.proc and self.proc.stdin:
            if isinstance(frames_tensor, torch.Tensor):
                f_cpu = frames_tensor.detach().cpu()
                if f_cpu.shape[1] in (1, 3, 4) and f_cpu.shape[-1] not in (1, 3, 4):
                    f_cpu = f_cpu.permute(0, 2, 3, 1)
                if f_cpu.is_floating_point():
                    if f_cpu.min() < 0.0:
                        f_cpu = (f_cpu + 1.0) * 127.5
                    elif f_cpu.max() <= 1.0:
                        f_cpu = f_cpu * 255.0
                    f_bytes = f_cpu.clamp(0, 255).byte().numpy().tobytes()
                else:
                    f_bytes = f_cpu.numpy().tobytes()
            else:
                f_np = np.asarray(frames_tensor)
                if f_np.dtype != np.uint8:
                    if f_np.min() < 0.0:
                        f_np = (f_np + 1.0) * 127.5
                    elif f_np.max() <= 1.0:
                        f_np = f_np * 255.0
                    f_np = np.clip(f_np, 0, 255).astype(np.uint8)
                f_bytes = f_np.tobytes()
            self.proc.stdin.write(f_bytes)

    def release(self):
        if self.proc:
            try:
                if self.proc.stdin:
                    self.proc.stdin.close()
                if self.proc.stderr:
                    self.proc.stderr.close()
                self.proc.wait()
            except Exception:
                pass
            self.proc = None
def format_time(seconds):
    """
    Format seconds into HH:MM:SS or MM:SS
    """
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


# =============================================================================
# Main CLI entry point
# =============================================================================

def main():
    args = parse_args()
    from src.flashvsr_log import configure_logging, get_logger
    configure_logging(args.log_level, args.log_file)
    logger = get_logger("flashvsr")

    # Safety check: ensure output file does not already exist
    if os.path.exists(args.output):
        print(f"Error: Output file '{args.output}' already exists. Aborting to prevent overwrite.", file=sys.stderr)
        sys.exit(1)
    
    # Handle boolean flag pairs
    force_offload = args.force_offload and not args.no_force_offload
    color_fix = args.color_fix and not args.no_color_fix
    keep_models_on_cpu = args.keep_models_on_cpu and not args.no_keep_models_on_cpu

    print("=" * 60)
    print("FlashVSR CLI - Video Super Resolution")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Model: {args.model}, Mode: {args.mode}")
    print(f"VAE: {args.vae_model}, Scale: {args.scale}x")
    print("=" * 60)

    # ==========================================================================
    # Setup environment and imports
    # ==========================================================================
    
    # Now import FlashVSR modules directly
    import torch
    from src.processor import (
        init_pipeline, flashvsr, log,
        VAE_MODEL_OPTIONS, VAE_MODEL_MAP
    )
    from src.models import wan_video_dit
    from src.te_speed import resolve_filename_prefix

    # Set device
    device = args.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda:0"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    print(f"Device: {device}")
    

    
    # ==========================================================================
    # Load input video (Lazily)
    # ==========================================================================
    print("\nInitializing Video Reader...")
    reader = VideoReader(
        args.input, 
        start_frame=args.start_frame, 
        end_frame=args.end_frame,
        chunk_size=args.frame_chunk_size
    )
    
    input_fps, file_total_frames = reader.get_info()
    
    # Calculate actual frames to process based on reader's resolved range
    total_frames_to_process = reader.end_frame - reader.start_frame
    
    if args.end_frame > 0 or args.start_frame > 0:
        print(f"Input: {args.input} ({input_fps:.2f} FPS)")
        print(f"Processing frames: {reader.start_frame} to {reader.end_frame} (Total: {total_frames_to_process})")
    else:
        print(f"Input: {args.input} ({input_fps:.2f} FPS, {total_frames_to_process} frames)")
        
    # Use output FPS if specified, otherwise use input FPS
    output_fps = args.fps if args.fps is not None else input_fps
    
    # ==========================================================================
    # Initialize pipeline
    # ==========================================================================
    print("\nInitializing FlashVSR pipeline...")
    
    # Set attention mode with sm < 8.0 (T4 / Colab) fallback
    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            cap = torch.cuda.get_device_capability(device)
            if cap < (8, 0) and args.attention_mode in ["sparse_sage_attention", "flash_attention_2", "block_sparse_attention"]:
                print(f"[FlashVSR] Device {device} (compute capability {cap[0]}.{cap[1]} < 8.0) does not support '{args.attention_mode}'. Automatically switching to 'sdpa'.")
                args.attention_mode = "sdpa"
        except Exception:
            pass
    wan_video_dit.ATTENTION_MODE = args.attention_mode
    
    # Determine dtype
    if args.precision == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            print("Auto-detected bf16 support.")
        else:
            dtype = torch.float16
            print("Defaulting to fp16.")
    elif args.precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    
    # Set CUDA device if using CUDA
    if device.startswith("cuda"):
        torch.cuda.set_device(device)
    
    # Initialize the pipeline
    pipe = init_pipeline(
        model=args.model,
        mode=args.mode,
        device=device,
        dtype=dtype,
        vae_model=args.vae_model
    )
    
    # ==========================================================================
    # Process video with FlashVSR (Chunk by Chunk)
    # ==========================================================================
    print("\nProcessing video with FlashVSR...")
    
    writer = None
    total_processed = 0
    start_time_glob = 0
    resolved_out = None
    temp_path = None
    processing_error = None
    try:
        import time
        start_time_glob = time.time()
        
        for chunk_idx, frames in enumerate(reader):
            # Calculate progress metrics
            elapsed = time.time() - start_time_glob
            
            # Speed (fps) - avoid division by zero
            if total_processed > 0 and elapsed > 0:
                speed_fps = total_processed / elapsed
                remaining_frames = total_frames_to_process - total_processed
                eta_seconds = remaining_frames / speed_fps
            else:
                speed_fps = 0.0
                eta_seconds = 0
            
            formatted_elapsed = format_time(elapsed)
            formatted_eta = format_time(eta_seconds)
            
            # Print status for the *current state* (before processing this chunk)
            # format: Progress:   8.34% | Processed: 6464/77514 | Elapsed: 1:34:31 | ETA: 0:12:10 | Speed: 1.25 fps
            progress_pct = (total_processed / total_frames_to_process) * 100 if total_frames_to_process > 0 else 0
            print(f"Progress: {progress_pct:6.2f}% | Processed: {total_processed}/{total_frames_to_process} | "
                  f"Elapsed: {formatted_elapsed} | ETA: {formatted_eta} | Speed: {speed_fps:.2f} fps")
            
            # Process the chunk
            # Note: We pass chunk_size=0 to flashvsr because we are feeding it an explicit chunk
            # that we want processed fully right now.
            output_frames = flashvsr(
                pipe=pipe,
                frames=frames,
                scale=args.scale,
                color_fix=color_fix,
                tiled_vae=args.tiled_vae,
                tiled_dit=args.tiled_dit,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
                unload_dit=args.unload_dit,
                sparse_ratio=args.sparse_ratio,
                kv_ratio=args.kv_ratio,
                local_range=args.local_range,
                seed=args.seed,
                force_offload=keep_models_on_cpu,
                enable_debug=args.enable_debug,
                chunk_size=0,
                resize_factor=args.resize_factor,
                mode=args.mode,
                quality_profile=args.quality_profile,
                motion_intensity=args.motion_intensity,
                spatial_strategy=args.spatial_strategy,
                memory_policy=args.memory_policy,
                validate_outputs=not args.no_validate_outputs,
            )
            # Initialize Writer on first chunk
            if writer is None:
                h, w = output_frames.shape[1], output_frames.shape[2]
                resolved_out = resolve_filename_prefix(args.output, width=w, height=h)
                suffix = os.path.splitext(resolved_out)[1] or ".mp4"
                temp_path = resolved_out + ".tmp" + suffix
                
                if args.audio and os.path.exists(args.audio):
                    audio_source = args.audio
                elif args.keep_audio:
                    audio_source = args.input
                else:
                    audio_source = None
                    
                print(f"Output dimensions: {w}x{h}")
                print(f"Saving output video to: {resolved_out}")
                writer = VideoWriter(
                    output_path=temp_path,
                    fps=output_fps,
                    width=w,
                    height=h,
                    quality=args.quality,
                    audio_path=audio_source
                )
            
            # Write frames
            writer.write(output_frames)
            total_processed += frames.shape[0]
            
            # Cleanup
            del frames, output_frames
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    except StopIteration:
        pass
    except Exception as e:
        processing_error = e
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if writer:
            writer.release()
        if processing_error is None and temp_path and os.path.exists(temp_path) and resolved_out:
            os.replace(temp_path, resolved_out)
        elif processing_error is not None and temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        logger.info(f"Summary: processed {total_processed}/{total_frames_to_process} frames")
    # ==========================================================================
    # Cleanup
    # ==========================================================================
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    end_time_glob = time.time()
    total_duration = end_time_glob - start_time_glob
    avg_fps = total_processed / total_duration if total_duration > 0 else 0
    
    print("\n" + "=" * 60)
    print("FlashVSR processing complete!")
    print(f"Total Frames Processed: {total_processed}/{total_frames_to_process}")
    print(f"Total Time: {format_time(total_duration)} ({avg_fps:.2f} FPS)")
    print("=" * 60)


if __name__ == "__main__":
    main()
