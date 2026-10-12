"""
FlashVSR Core Processor (Linux & Google Colab Native)
High-performance Video Super Resolution pipeline with TE-Speed acceleration:
- Motion-Aware Dynamic Attention Budgeting
- Staged Memory Policy (eliminates OOM on Tesla T4 & 8-16GB GPUs)
- Adaptive Spatial Tiling with cosine feather blending
- Anomaly Output Validation (NaN/Inf detection)
"""

import os
import gc
import math
import time
import sys
import psutil
import yaml
import threading
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub import snapshot_download, hf_hub_download

from . import ModelManager, FlashVSRFullPipeline, FlashVSRTinyPipeline, FlashVSRTinyLongPipeline
from .models.TCDecoder import build_tcdecoder
from .models.utils import clean_vram, get_device_list, Buffer_LQ4x_Proj, Causal_LQ4x_Proj
from .models import wan_video_dit
from .models.wan_video_vae import (
    WanVideoVAE, Wan22VideoVAE, LightX2VVAE, create_video_vae
)
from .te_speed import (
    motion_profile,
    AttentionBudgetController,
    MemoryPolicy,
    ExecutionPlan,
    SpatialExecutor,
    compute_feather_mask,
    validate_tensor,
    TensorValidationError,
)
import hashlib

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

# Known SHA-256 digests. Only entries present here are verified.
# Leave blank string to skip verification for a file.
SHA256_MANIFEST: dict[str, str] = {}

def verify_model_file(path: str, expected_sha256: str = "") -> bool:
    """Returns True if file is present and hash matches (or no expected hash given)."""
    if not os.path.exists(path):
        return False
    if not expected_sha256:
        return True
    actual = sha256_file(path)
    if actual != expected_sha256:
        log(f"SHA-256 mismatch for {os.path.basename(path)}: expected {expected_sha256[:12]}… got {actual[:12]}…",
            message_type="warning", icon="⚠️")
        return False
    return True


# Constants
DEFAULT_MODELS_DIR = os.environ.get("FLASHVSR_MODELS_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))

VAE_DEFAULT_DIM = 96
VAE_Z_DIM = 16
VAE_LIGHT_DIM = 64

VAE_MODEL_OPTIONS = [
    "Wan2.1",
    "Wan2.2",
    "LightVAE_W2.1",
    "TAE_W2.2",
    "LightTAE_HY1.5"
]

VAE_MODEL_MAP = {
    "Wan2.1": {
        "class": WanVideoVAE,
        "file": "Wan2.1_VAE.pth",
        "internal_name": "wan2.1",
        "url": "https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1/resolve/main/Wan2.1_VAE.pth",
        "dim": VAE_DEFAULT_DIM,
        "z_dim": VAE_Z_DIM,
        "use_full_arch": False
    },
    "Wan2.2": {
        "class": Wan22VideoVAE,
        "file": "Wan2.2_VAE.pth",
        "internal_name": "wan2.2",
        "url": "https://huggingface.co/lightx2v/Autoencoders/resolve/main/Wan2.2_VAE.pth",
        "dim": VAE_DEFAULT_DIM,
        "z_dim": VAE_Z_DIM,
        "use_full_arch": False
    },
    "LightVAE_W2.1": {
        "class": LightX2VVAE,
        "file": "lightvaew2_1.pth",
        "internal_name": "lightvae_w2.1",
        "url": "https://huggingface.co/lightx2v/Autoencoders/resolve/main/lightvaew2_1.pth",
        "dim": VAE_LIGHT_DIM,
        "z_dim": VAE_Z_DIM,
        "use_full_arch": True
    },
    "TAE_W2.2": {
        "class": Wan22VideoVAE,
        "file": "taew2_2.safetensors",
        "internal_name": "tae_w2.2",
        "url": "https://huggingface.co/lightx2v/Autoencoders/resolve/main/taew2_2.safetensors",
        "dim": VAE_DEFAULT_DIM,
        "z_dim": VAE_Z_DIM,
        "use_full_arch": False
    },
    "LightTAE_HY1.5": {
        "class": LightX2VVAE,
        "file": "lighttaehy1_5.pth",
        "internal_name": "lighttae_hy1.5",
        "url": "https://huggingface.co/lightx2v/Autoencoders/resolve/main/lighttaehy1_5.pth",
        "dim": VAE_LIGHT_DIM,
        "z_dim": VAE_Z_DIM,
        "use_full_arch": True
    },
}

VRAM_OOM_THRESHOLD = 0.95
_cached_model_path = None
_config_loaded = False
_config_lock = threading.Lock()


def load_model_paths_config() -> Optional[str]:
    global _cached_model_path, _config_loaded
    with _config_lock:
        if _config_loaded:
            return _cached_model_path

    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(current_dir, "model_paths.yaml")
    if not os.path.exists(config_path):
        with _config_lock:
            _config_loaded = True
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if config and isinstance(config, dict):
            flashvsr_path = config.get("flashvsr_model_path", "").strip()
            if flashvsr_path:
                flashvsr_path = os.path.expanduser(flashvsr_path)
                if not os.path.isabs(flashvsr_path):
                    flashvsr_path = os.path.abspath(os.path.join(current_dir, flashvsr_path))
                with _config_lock:
                    _cached_model_path = flashvsr_path
                    _config_loaded = True
                return flashvsr_path
    except Exception as e:
        print(f"⚠️ Warning: Could not load model_paths.yaml: {e}")

    with _config_lock:
        _config_loaded = True
    return None


def get_flashvsr_model_base_dir() -> str:
    custom_path = load_model_paths_config()
    if custom_path:
        return custom_path
    return DEFAULT_MODELS_DIR


def log(message: str, message_type: str = "normal", icon: str = "", end: str = "\n", in_place: bool = False):
    if icon:
        message = f"{icon} {message}"
    if message_type == "error":
        message = "\033[1;41m" + message + "\033[m"
    elif message_type == "warning":
        message = "\033[1;31m" + message + "\033[m"
    elif message_type == "finish":
        message = "\033[1;32m" + message + "\033[m"
    elif message_type == "info":
        message = "\033[1;33m" + message + "\033[m"

    if in_place:
        sys.stdout.write("\r\033[K" + message)
        sys.stdout.flush()
    else:
        print(f"{message}", end=end, flush=True)


def get_vram_info() -> Tuple[float, float, float]:
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / (1024 ** 3)
        vram_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        vram_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return vram_used, vram_reserved, vram_total
    return 0.0, 0.0, 0.0


def model_download(model_name: str = "JunhaoZhuang/FlashVSR"):
    base_dir = get_flashvsr_model_base_dir()
    model_dir = os.path.join(base_dir, model_name.split("/")[-1])
    if not os.path.exists(model_dir):
        log(f"Downloading model '{model_name}' from huggingface...", message_type="info", icon="⬇️")
        snapshot_download(repo_id=model_name, local_dir=model_dir, local_dir_use_symlinks=False, resume_download=True)
        for root, _, files in os.walk(model_dir):
            for file in files:
                if file.endswith((".pth", ".safetensors")):
                    expected = SHA256_MANIFEST.get(file, "")
                    verify_model_file(os.path.join(root, file), expected)


def download_vae_if_missing(vae_file: str, model_path: str, vae_config: dict) -> str:
    target_path = os.path.join(model_path, vae_file)
    if os.path.exists(target_path):
        return target_path

    url = vae_config.get("url")
    if not url:
        raise RuntimeError(f"VAE file '{vae_file}' not found and no download URL available.")

    log(f"VAE file '{vae_file}' not found locally. Auto-downloading...", message_type="info", icon="⬇️")
    os.makedirs(model_path, exist_ok=True)

    match = False
    if "huggingface.co" in url and "/resolve/" in url:
        parts = url.replace("https://huggingface.co/", "").split("/resolve/")
        if len(parts) == 2:
            repo_id = parts[0]
            subparts = parts[1].split("/")
            branch = subparts[0]
            filename = "/".join(subparts[1:])
            try:
                downloaded_file = hf_hub_download(repo_id=repo_id, filename=filename, revision=branch, local_dir=model_path, local_dir_use_symlinks=False)
                if os.path.exists(downloaded_file):
                    target_path = downloaded_file
                    match = True
            except Exception as e:
                log(f"hf_hub_download failed: {e}. Trying direct download...", message_type="warning", icon="⚠️")

    if not match:
        import urllib.request
        urllib.request.urlretrieve(url, target_path)

    log(f"VAE downloaded successfully to: {target_path}", message_type="finish", icon="✅")
    expected = SHA256_MANIFEST.get(vae_file, "")
    verify_model_file(target_path, expected)
    return target_path


def estimate_vram_usage(width: int, height: int, num_frames: int, scale: int, tiled_vae: bool = False, tiled_dit: bool = False, chunk_size: int = 0, mode: str = "full") -> float:
    SAFETY_FACTOR = 4.0
    if mode == "full":
        base_model_gb = 5.0
    elif mode == "tiny-long":
        base_model_gb = 3.5
    else:
        base_model_gb = 4.0

    output_h, output_w = height * scale, width * scale
    latent_h, latent_w = output_h // 8, output_w // 8
    effective_frames = chunk_size if (chunk_size > 0 and chunk_size <= num_frames) else num_frames

    input_tensor_bytes = output_h * output_w * 3 * effective_frames * 4
    input_tensor_gb = (input_tensor_bytes * SAFETY_FACTOR) / (1024 ** 3)

    bytes_per_frame = latent_h * latent_w * 16 * 2
    total_latent_gb = (bytes_per_frame * effective_frames * SAFETY_FACTOR) / (1024 ** 3)

    seq_len = latent_h * latent_w * (effective_frames // 4)
    attention_gb = (seq_len * seq_len * 2 * SAFETY_FACTOR) / (1024 ** 3) * 0.001
    vae_decode_gb = (output_h * output_w * 3 * effective_frames * 2 * SAFETY_FACTOR) / (1024 ** 3)

    if tiled_dit:
        attention_gb *= 0.3
    if tiled_vae:
        vae_decode_gb *= 0.4

    return base_model_gb + input_tensor_gb + total_latent_gb + attention_gb + vae_decode_gb


def get_optimal_settings(width: int, height: int, num_frames: int, scale: int, available_vram_gb: float, mode: str = "full") -> Dict[str, Any]:
    target_vram = available_vram_gb * 0.85
    recommended = {
        "chunk_size": 0,
        "resize_factor": 1.0,
        "tiled_vae": False,
        "tiled_dit": False,
        "warning": None
    }
    estimated = estimate_vram_usage(width, height, num_frames, scale, tiled_vae=False, tiled_dit=False, chunk_size=0, mode=mode)
    if estimated <= target_vram:
        return recommended

    recommended["tiled_vae"] = True
    estimated = estimate_vram_usage(width, height, num_frames, scale, tiled_vae=True, tiled_dit=False, chunk_size=0, mode=mode)
    if estimated <= target_vram:
        return recommended

    recommended["tiled_dit"] = True
    estimated = estimate_vram_usage(width, height, num_frames, scale, tiled_vae=True, tiled_dit=True, chunk_size=0, mode=mode)
    if estimated <= target_vram:
        return recommended

    for chunk in [64, 48, 32, 24, 16]:
        if chunk < num_frames:
            recommended["chunk_size"] = chunk
            estimated = estimate_vram_usage(width, height, num_frames, scale, tiled_vae=True, tiled_dit=True, chunk_size=chunk, mode=mode)
            if estimated <= target_vram:
                return recommended

    for factor in [0.75, 0.5, 0.25]:
        recommended["resize_factor"] = factor
        estimated = estimate_vram_usage(int(width * factor), int(height * factor), num_frames, scale, tiled_vae=True, tiled_dit=True, chunk_size=16, mode=mode)
        if estimated <= target_vram:
            return recommended

    recommended["warning"] = "Warning: Resolution is very large. Consider lower scale factor."
    return recommended


# Math helpers for temporal/spatial alignment
def largest_8n1_leq(n: int) -> int:
    if n < 1:
        return 0
    k = (n - 1) // 8
    return 8 * k + 1


def next_8n5(x: int) -> int:
    return ((x - 5 + 7) // 8) * 8 + 5


def resolve_target_resolution(w0: int, h0: int, resolution: Optional[int] = None, scale: Optional[int] = None) -> Tuple[int, int, int]:
    """
    Resolves (target_w, target_h, effective_scale) from input dimensions.
    When resolution is specified, sets the shortest edge to resolution (e.g. 720, 1080).
    Automatically picks 2x or 4x model based on scale ratio.
    """
    if resolution is not None and resolution > 0:
        short_edge = min(w0, h0)
        scale_ratio = resolution / max(short_edge, 1)
        effective_scale = 2 if scale_ratio <= 2.8 else 4
        target_w = round(w0 * scale_ratio)
        target_h = round(h0 * scale_ratio)
        target_w = target_w - (target_w % 2)
        target_h = target_h - (target_h % 2)
        return target_w, target_h, effective_scale
    else:
        effective_scale = scale if scale in (2, 4) else 2
        return w0 * effective_scale, h0 * effective_scale, effective_scale


def compute_scaled_and_target_dims(w0: int, h0: int, scale: int = 4, multiple: int = 128, target_dims: Optional[Tuple[int, int]] = None) -> Tuple[int, int, int, int, int, int]:
    if target_dims is not None:
        sW, sH = target_dims
    else:
        sW = w0 * scale
        sH = h0 * scale
    tW = math.ceil(sW / multiple) * multiple
    tH = math.ceil(sH / multiple) * multiple
    pad_left = (tW - sW) // 2
    pad_top = (tH - sH) // 2
    return sW, sH, tW, tH, pad_left, pad_top

def tensor_upscale_then_center_crop(frame_tensor: torch.Tensor, scale: int, tW: int, tH: int, pad_left: int, pad_top: int, sW: Optional[int] = None, sH: Optional[int] = None) -> torch.Tensor:
    if frame_tensor.dim() == 3:
        frame_tensor = frame_tensor.unsqueeze(0)
    if frame_tensor.shape[1] not in (1, 3, 4) and frame_tensor.shape[-1] in (1, 3, 4):
        frame_tensor = frame_tensor.permute(0, 3, 1, 2)

    if sW is not None and sH is not None:
        upscaled = F.interpolate(frame_tensor, size=(sH, sW), mode="bicubic", align_corners=False)
    else:
        upscaled = F.interpolate(frame_tensor, scale_factor=float(scale), mode="bicubic", align_corners=False)
    _, _, uH, uW = upscaled.shape
    pad_right = max(0, tW - (pad_left + uW))
    pad_bottom = max(0, tH - (pad_top + uH))
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        padded = F.pad(upscaled, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
    else:
        padded = upscaled
    cropped = padded[:, :, :tH, :tW]
    return cropped.squeeze(0)


def prepare_input_tensor(image_tensor: torch.Tensor, device: Any, scale: int = 4, dtype: torch.dtype = torch.float16, target_dims: Optional[Tuple[int, int]] = None):
    N0, h0, w0, _ = image_tensor.shape
    multiple = 128
    sW, sH, tW, tH, pad_left, pad_top = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=multiple, target_dims=target_dims)
    F_frames = max(25, largest_8n1_leq(N0 + 4))

    frames = []
    for i in range(F_frames):
        frame_idx = min(i, N0 - 1)
        frame_slice = image_tensor[frame_idx]
        tensor_chw = tensor_upscale_then_center_crop(frame_slice, scale=scale, tW=tW, tH=tH, pad_left=pad_left, pad_top=pad_top, sW=sW, sH=sH).to("cpu", dtype=dtype) * 2.0 - 1.0
        frames.append(tensor_chw)

    vid_final = torch.stack(frames, dim=1).unsqueeze(0)  # (1, C, F, H, W)
    return vid_final, tH, tW, F_frames, sH, sW, pad_top, pad_left


def init_pipeline(model: str, mode: str, device: str, dtype: torch.dtype, vae_model: str = "Wan2.1"):
    """
    Initializes FlashVSR pipeline with selected model, mode, precision, and VAE architecture.
    """
    model_download(model_name="JunhaoZhuang/" + model)
    base_dir = get_flashvsr_model_base_dir()
    model_path = os.path.join(base_dir, model)
    if not os.path.exists(model_path):
        raise RuntimeError(f'Model directory does not exist: "{model_path}"')

    ckpt_path = os.path.join(model_path, "diffusion_pytorch_model_streaming_dmd.safetensors")
    if not os.path.exists(ckpt_path):
        raise RuntimeError(f'Checkpoint does not exist: "{ckpt_path}"')

    if vae_model not in VAE_MODEL_MAP:
        vae_model = "Wan2.1"
    vae_config = VAE_MODEL_MAP[vae_model]
    vae_path = download_vae_if_missing(vae_config["file"], model_path, vae_config)

    lq_path = os.path.join(model_path, "LQ_proj_in.ckpt")
    tcd_path = os.path.join(model_path, "TCDecoder.ckpt")
    prompt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "posi_prompt.pth")

    mm = ModelManager(torch_dtype=dtype, device="cpu")
    if mode == "full":
        mm.load_models([ckpt_path, vae_path])
        pipe = FlashVSRFullPipeline.from_model_manager(mm, device=device)
        vae_class = vae_config["class"]
        if vae_path.endswith(".safetensors"):
            import safetensors.torch
            sd = safetensors.torch.load_file(vae_path)
        else:
            sd = torch.load(vae_path, map_location="cpu", weights_only=False)

        if vae_class == LightX2VVAE:
            pipe.vae = LightX2VVAE(z_dim=vae_config["z_dim"], dim=vae_config["dim"], use_full_arch=vae_config["use_full_arch"])
        elif vae_class == Wan22VideoVAE:
            pipe.vae = Wan22VideoVAE(z_dim=vae_config["z_dim"], dim=vae_config["dim"])
        else:
            pipe.vae = WanVideoVAE(z_dim=vae_config["z_dim"], dim=vae_config["dim"])

        pipe.vae.load_state_dict(sd, strict=False)
        pipe.vae = pipe.vae.to(device=device, dtype=dtype)
        pipe.vae.model.encoder = None
        pipe.vae.model.conv1 = None

        multi_scale_channels = [512, 256, 128, 128]
        pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, device=device, dtype=dtype, new_latent_channels=16+768)
        pipe.TCDecoder.load_state_dict(torch.load(tcd_path, map_location="cpu", weights_only=False), strict=False)
        pipe.TCDecoder.to(device=device, dtype=dtype)
        pipe.TCDecoder.clean_mem()
    else:
        mm.load_models([ckpt_path])
        if mode == "tiny":
            pipe = FlashVSRTinyPipeline.from_model_manager(mm, device=device)
        else:
            pipe = FlashVSRTinyLongPipeline.from_model_manager(mm, device=device)

        multi_scale_channels = [512, 256, 128, 128]
        pipe.TCDecoder = build_tcdecoder(new_channels=multi_scale_channels, device=device, dtype=dtype, new_latent_channels=16+768)
        pipe.TCDecoder.load_state_dict(torch.load(tcd_path, map_location="cpu", weights_only=False), strict=False)
        pipe.TCDecoder.to(device=device, dtype=dtype)
        pipe.TCDecoder.clean_mem()

    if model == "FlashVSR":
        pipe.denoising_model().LQ_proj_in = Buffer_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
    else:
        pipe.denoising_model().LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to(device, dtype=dtype)
    pipe.denoising_model().LQ_proj_in.load_state_dict(torch.load(lq_path, map_location="cpu", weights_only=False), strict=True)
    pipe.denoising_model().LQ_proj_in.to(device)

    pipe.to(device, dtype=dtype)
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv(prompt_path=prompt_path)
    pipe.load_models_to_device(["dit", "vae"])
    pipe.offload_model()
    return pipe


def tensor2video(tensor: torch.Tensor) -> torch.Tensor:
    """Converts pipeline output (B, C, T, H, W) to (T, H, W, C) float in [0, 1]."""
    if tensor.dim() == 5:
        tensor = tensor[0].permute(1, 2, 3, 0)
    elif tensor.dim() == 4 and tensor.shape[0] in (1, 3, 4):
        tensor = tensor.permute(1, 2, 3, 0)
    tensor = (tensor + 1.0) / 2.0
    return tensor.clamp(0.0, 1.0).float()


def process_chunk(
    pipe: Any,
    frames: torch.Tensor,
    scale: int,
    color_fix: bool = True,
    tiled_vae: bool = True,
    tiled_dit: bool = True,
    tile_size: int = 256,
    tile_overlap: int = 24,
    unload_dit: bool = False,
    sparse_ratio: float = 2.0,
    kv_ratio: float = 3.0,
    local_range: int = 11,
    seed: int = 0,
    force_offload: bool = True,
    enable_debug: bool = False,
    budget_controller: Optional[AttentionBudgetController] = None,
    stage_memory: bool = False,
    validate_outputs: bool = True,
    target_dims: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """
    Processes a temporal chunk of frames.
    """
    clean_vram()
    _device = pipe.device
    dtype = pipe.torch_dtype

    target_count = max(21, next_8n5(frames.shape[0]))
    if target_count > frames.shape[0]:
        add = target_count - frames.shape[0]
        padding_frames = frames[-1:, :, :, :].repeat(add, 1, 1, 1)
        _frames = torch.cat([frames, padding_frames], dim=0)
    else:
        _frames = frames
    # Input tensor preparation
    # Input tensor preparation
    LQ, th, tw, F_frames, sH, sW, pad_top, pad_left = prepare_input_tensor(_frames, _device, scale=scale, dtype=dtype, target_dims=target_dims)
    if not isinstance(pipe, FlashVSRTinyLongPipeline):
        LQ = LQ.to(_device)

    # Run pipeline
    output_gpu = pipe(
        prompt="",
        negative_prompt="",
        cfg_scale=1.0,
        num_inference_steps=1,
        seed=seed,
        tiled=tiled_vae,
        LQ_video=LQ,
        num_frames=F_frames,
        height=th,
        width=tw,
        is_full_block=False,
        if_buffer=True,
        topk_ratio=sparse_ratio * 768 * 1280 / (th * tw),
        kv_ratio=kv_ratio,
        local_range=local_range,
        color_fix=color_fix,
        unload_dit=unload_dit,
        force_offload=force_offload,
        enable_debug_logging=enable_debug,
        budget_controller=budget_controller,
        stage_memory=stage_memory,
    )

    processed_cpu = tensor2video(output_gpu).to("cpu")

    # Crop out temporal/spatial padding
    max_crop_h = min(pad_top + sH, processed_cpu.shape[1])
    max_crop_w = min(pad_left + sW, processed_cpu.shape[2])
    final_output = processed_cpu[:frames.shape[0], pad_top:max_crop_h, pad_left:max_crop_w, :]

    if validate_outputs:
        validate_tensor(final_output, name="chunk_output", auto_clean=True)

    del LQ, output_gpu, processed_cpu
    clean_vram()
    return final_output


def flashvsr(
    pipe: Any,
    frames: torch.Tensor,
    scale: int = 2,
    color_fix: bool = True,
    tiled_vae: bool = True,
    tiled_dit: bool = True,
    tile_size: int = 256,
    tile_overlap: int = 24,
    unload_dit: bool = False,
    sparse_ratio: float = 2.0,
    kv_ratio: float = 3.0,
    local_range: int = 11,
    seed: int = 0,
    force_offload: bool = True,
    enable_debug: bool = False,
    chunk_size: int = 0,
    resize_factor: float = 1.0,
    mode: str = "tiny",
    # TE-Speed features
    quality_profile: str = "balanced",
    motion_intensity: float = 1.0,
    spatial_strategy: str = "auto",
    memory_policy: str = "auto",
    validate_outputs: bool = True,
    target_dims: Optional[Tuple[int, int]] = None,
    resolution: Optional[int] = None,
) -> torch.Tensor:
    """
    Main FlashVSR video restoration function with TE-Speed dynamic enhancements.
    """
    clean_vram()
    start_time = time.time()

    # Feature: Automatic short-edge resolution resolution
    if resolution is not None and resolution > 0:
        orig_H, orig_W = frames.shape[1], frames.shape[2]
        target_w, target_h, eff_scale = resolve_target_resolution(orig_W, orig_H, resolution=resolution, scale=scale)
        target_dims = (target_w, target_h)
        scale = eff_scale
        if enable_debug:
            log(f"Resolution target {resolution}px: scaled to {target_w}x{target_h} (model scale: {scale}x)", message_type="info", icon="📐")
    # Pre-resize if requested
    if 0.0 < resize_factor < 1.0:
        orig_H, orig_W = frames.shape[1], frames.shape[2]
        new_H, new_W = int(orig_H * resize_factor), int(orig_W * resize_factor)
        frames_permuted = frames.permute(0, 3, 1, 2)
        frames_resized = F.interpolate(frames_permuted, size=(new_H, new_W), mode="bicubic", align_corners=False)
        frames = frames_resized.permute(0, 2, 3, 1)
        del frames_permuted, frames_resized

    total_frames = frames.shape[0]

    # Feature 1: Motion-Aware Dynamic Attention Budgeting
    num_chunks = max(1, math.ceil(total_frames / 8))
    motion = motion_profile(frames, num_chunks)
    budget_ctrl = AttentionBudgetController(
        motion=motion,
        quality_profile=quality_profile,
        intensity=motion_intensity,
    )
    if enable_debug:
        log(budget_ctrl.summary(), message_type="info", icon="⚡")

    # Feature 2: Staged Memory Policy
    stage_memory = ExecutionPlan.should_stage(
        height=frames.shape[1],
        width=frames.shape[2],
        frame_count=total_frames,
        device=pipe.device,
        memory_policy=memory_policy,
        mode=mode,
    )
    if enable_debug and stage_memory:
        log("Execution plan: Staged memory swapping enabled to prevent VRAM saturation.", message_type="info", icon="💾")

    # Feature 5: Adaptive Spatial Tiling
    spatial_exec = SpatialExecutor(
        strategy=spatial_strategy,
        max_tile_edge=tile_size,
        blend_overlap=tile_overlap,
    )

    def _process_spatial_slice(slice_frames: torch.Tensor) -> torch.Tensor:
        # Executes either directly or chunked through process_chunk
        effective_chunk = chunk_size if (0 < chunk_size < slice_frames.shape[0]) else slice_frames.shape[0]
        if effective_chunk >= slice_frames.shape[0]:
            return process_chunk(
                pipe=pipe,
                frames=slice_frames,
                scale=scale,
                color_fix=color_fix,
                tiled_vae=tiled_vae,
                tiled_dit=False,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                unload_dit=unload_dit,
                sparse_ratio=sparse_ratio,
                kv_ratio=kv_ratio,
                local_range=local_range,
                seed=seed,
                force_offload=force_offload,
                enable_debug=enable_debug,
                budget_controller=budget_ctrl,
                stage_memory=stage_memory,
                validate_outputs=validate_outputs,
                target_dims=target_dims,
            )

        # Chunked temporal processing
        c_outputs = []
        n_c = math.ceil(slice_frames.shape[0] / effective_chunk)
        for c_i in range(n_c):
            c_start = c_i * effective_chunk
            c_end = min((c_i + 1) * effective_chunk, slice_frames.shape[0])
            c_frames = slice_frames[c_start:c_end]
            c_out = process_chunk(
                pipe=pipe,
                frames=c_frames,
                scale=scale,
                color_fix=color_fix,
                tiled_vae=tiled_vae,
                tiled_dit=False,
                tile_size=tile_size,
                tile_overlap=tile_overlap,
                unload_dit=unload_dit,
                sparse_ratio=sparse_ratio,
                kv_ratio=kv_ratio,
                local_range=local_range,
                seed=seed,
                force_offload=force_offload,
                enable_debug=enable_debug,
                budget_controller=budget_ctrl,
                stage_memory=stage_memory,
                validate_outputs=validate_outputs,
                target_dims=target_dims,
            )
            c_outputs.append(c_out)
            del c_out
            clean_vram()
        return torch.cat(c_outputs, dim=0)

    # Execute spatial strategy (Adaptive tiles or Full-frame)
    final_output = spatial_exec.execute(
        frames=frames,
        scale=scale,
        process_fn=_process_spatial_slice,
    )

    # Feature 6: Final tensor validation
    if validate_outputs:
        validate_tensor(final_output, name="final_video", auto_clean=True)

    elapsed = time.time() - start_time
    fps_val = total_frames / elapsed if elapsed > 0 else 0.0
    log(f"FlashVSR completed: {total_frames} frames in {elapsed:.2f}s ({fps_val:.2f} FPS)", message_type="finish", icon="🚀")
    return final_output
