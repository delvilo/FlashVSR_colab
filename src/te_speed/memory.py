"""
Memory Policy and Staged Execution Plan for FlashVSR
Manages whether models reside simultaneously in GPU VRAM (resident) or are
swapped between DiT denoising and VAE decoding (staged) to maximize performance
and eliminate OOM on 8GB-16GB GPUs like Tesla T4.
"""

from typing import Optional
import torch
import gc


class MemoryPolicy:
    AUTO = "auto"
    RESIDENT = "resident"
    STAGED = "staged"


class ExecutionPlan:
    """
    Determines memory staging and offloading strategy based on hardware and input size.
    """

    @staticmethod
    def get_available_vram_gb(device: Optional[str] = None) -> float:
        if not torch.cuda.is_available():
            return 0.0
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            return free_bytes / (1024 ** 3)
        except Exception:
            return 0.0

    @staticmethod
    def get_total_vram_gb(device: Optional[str] = None) -> float:
        if not torch.cuda.is_available():
            return 0.0
        try:
            idx = torch.cuda.current_device() if device is None else device
            props = torch.cuda.get_device_properties(idx)
            return props.total_memory / (1024 ** 3)
        except Exception:
            return 0.0

    @classmethod
    def should_stage(
        cls,
        height: int,
        width: int,
        frame_count: int,
        device: Optional[str] = None,
        memory_policy: str = "auto",
        mode: str = "tiny",
    ) -> bool:
        """
        Determines whether to swap DiT and VAE between stages instead of keeping both resident.
        """
        policy = memory_policy.lower()
        if policy == MemoryPolicy.STAGED:
            return True
        if policy == MemoryPolicy.RESIDENT:
            return False

        # AUTO policy
        if not torch.cuda.is_available():
            return True

        total_vram = cls.get_total_vram_gb(device)
        free_vram = cls.get_available_vram_gb(device)

        # Full mode has higher peak memory due to full VAE
        is_full = (mode == "full")

        # For <= 16GB VRAM GPUs (including Colab T4 ~15GB)
        if total_vram <= 16.5:
            # On T4 / 16GB, stage if resolution is 720p+ or frames >= 40 or free VRAM < 9GB
            pixels = height * width
            if is_full:
                return True
            if pixels >= 720 * 1280 or frame_count >= 40 or free_vram < 8.5:
                return True
            return False

        # For 24GB+ GPUs (RTX 3090, 4090, A100)
        if is_full and (height * width >= 1080 * 1920 or frame_count >= 120):
            return True

        return False

    @staticmethod
    def clean_memory():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
