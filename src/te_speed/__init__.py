"""
TE-Speed Runtime for FlashVSR (Linux & Google Colab Native)
Provides:
- Motion-Aware Dynamic Attention Budgeting
- Staged Memory Policy for Ultra-Low VRAM
- Hardware-Accelerated Video Encoding & Audio Muxing (FFmpeg / NVENC)
- Adaptive Spatial Tiling & Smooth Blending
- Numerical Sanity Validation (NaN/Inf Detection)
"""

from .motion import motion_profile, AttentionBudgetController
from .memory import MemoryPolicy, ExecutionPlan
from .spatial import SpatialExecutor, compute_feather_mask
from .validator import validate_tensor, TensorValidationError
from .video_encoder import (
    find_ffmpeg,
    has_nvenc,
    resolve_filename_prefix,
    encode_video_ffmpeg,
)

__all__ = [
    "motion_profile",
    "AttentionBudgetController",
    "MemoryPolicy",
    "ExecutionPlan",
    "SpatialExecutor",
    "compute_feather_mask",
    "validate_tensor",
    "TensorValidationError",
    "find_ffmpeg",
    "has_nvenc",
    "resolve_filename_prefix",
    "encode_video_ffmpeg",
]
