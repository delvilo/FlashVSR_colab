"""
FlashVSR Standalone Library (Linux & Google Colab Native)
"""

from .src.processor import (
    init_pipeline,
    flashvsr,
    process_chunk,
    prepare_input_tensor,
    estimate_vram_usage,
    get_optimal_settings,
    VAE_MODEL_MAP,
    VAE_MODEL_OPTIONS,
)
from .src.te_speed import (
    motion_profile,
    AttentionBudgetController,
    MemoryPolicy,
    ExecutionPlan,
    SpatialExecutor,
    validate_tensor,
    encode_video_ffmpeg,
    resolve_filename_prefix,
)

__all__ = [
    "init_pipeline",
    "flashvsr",
    "process_chunk",
    "prepare_input_tensor",
    "estimate_vram_usage",
    "get_optimal_settings",
    "VAE_MODEL_MAP",
    "VAE_MODEL_OPTIONS",
    "motion_profile",
    "AttentionBudgetController",
    "MemoryPolicy",
    "ExecutionPlan",
    "SpatialExecutor",
    "validate_tensor",
    "encode_video_ffmpeg",
    "resolve_filename_prefix",
]
