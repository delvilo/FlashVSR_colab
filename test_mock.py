import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Mock dependencies that might be missing or require heavy setup
sys.modules['sageattention'] = MagicMock()
sys.modules['flash_attn'] = MagicMock()

import torch
if not torch.cuda.is_available():
    torch.cuda.is_available = MagicMock(return_value=False)

from src.processor import (
    flashvsr, init_pipeline,
    VAE_MODEL_OPTIONS, VAE_MODEL_MAP,
    estimate_vram_usage, get_optimal_settings
)
from src.pipelines.flashvsr_full import FlashVSRFullPipeline
from src.models.wan_video_vae import WanVideoVAE, Wan22VideoVAE, LightX2VVAE, create_video_vae
from src.te_speed import (
    motion_profile, AttentionBudgetController,
    MemoryPolicy, ExecutionPlan,
    SpatialExecutor, compute_feather_mask,
    validate_tensor, TensorValidationError,
    resolve_filename_prefix,
)


class TestFlashVSRCoreAndTESpeed(unittest.TestCase):
    def test_pipeline_instantiation(self):
        self.assertTrue(hasattr(FlashVSRFullPipeline, '__call__'))

    def test_vae_model_options_available(self):
        self.assertIn("Wan2.1", VAE_MODEL_OPTIONS)
        self.assertIn("Wan2.2", VAE_MODEL_OPTIONS)
        self.assertIn("LightVAE_W2.1", VAE_MODEL_OPTIONS)
        self.assertIn("TAE_W2.2", VAE_MODEL_OPTIONS)
        self.assertIn("LightTAE_HY1.5", VAE_MODEL_OPTIONS)
        self.assertEqual(len(VAE_MODEL_OPTIONS), 5)

    def test_vae_model_map_configured(self):
        for key in ["Wan2.1", "Wan2.2", "LightVAE_W2.1", "TAE_W2.2", "LightTAE_HY1.5"]:
            self.assertIn(key, VAE_MODEL_MAP)
            cfg = VAE_MODEL_MAP[key]
            self.assertIn("class", cfg)
            self.assertIn("file", cfg)
            self.assertIn("url", cfg)

    def test_estimate_vram_usage_modes(self):
        vram_full = estimate_vram_usage(1280, 720, 100, 2, mode='full')
        vram_tiny = estimate_vram_usage(1280, 720, 100, 2, mode='tiny')
        vram_tiny_long = estimate_vram_usage(1280, 720, 100, 2, mode='tiny-long')
        self.assertGreater(vram_full, 0)
        self.assertGreater(vram_tiny, 0)
        self.assertGreater(vram_tiny_long, 0)
        self.assertGreater(vram_full, vram_tiny_long)

    def test_estimate_vram_usage_tiling_reduces(self):
        vram_no_tile = estimate_vram_usage(1280, 720, 100, 2, tiled_vae=False, tiled_dit=False)
        vram_tiled = estimate_vram_usage(1280, 720, 100, 2, tiled_vae=True, tiled_dit=True)
        self.assertLess(vram_tiled, vram_no_tile)

    def test_estimate_vram_usage_chunking_reduces(self):
        vram_no_chunk = estimate_vram_usage(1280, 720, 100, 2, chunk_size=0)
        vram_chunked = estimate_vram_usage(1280, 720, 100, 2, chunk_size=25)
        self.assertLess(vram_chunked, vram_no_chunk)

    def test_get_optimal_settings_high_vram(self):
        settings = get_optimal_settings(640, 480, 50, 2, available_vram_gb=32.0, mode='full')
        self.assertFalse(settings['tiled_vae'])
        self.assertFalse(settings['tiled_dit'])
        self.assertEqual(settings['chunk_size'], 0)
        self.assertEqual(settings['resize_factor'], 1.0)

    def test_get_optimal_settings_low_vram(self):
        settings = get_optimal_settings(1920, 1080, 200, 4, available_vram_gb=4.0, mode='full')
        self.assertTrue(settings['tiled_vae'])
        self.assertTrue(settings['tiled_dit'])
        self.assertTrue(settings['chunk_size'] > 0 or settings['resize_factor'] < 1.0)

    def test_vae_factory_function(self):
        vae1 = create_video_vae('wan2.1')
        self.assertIsInstance(vae1, WanVideoVAE)
        vae2 = create_video_vae('wan2.2')
        self.assertIsInstance(vae2, Wan22VideoVAE)
        vae3 = create_video_vae('lightx2v', use_full_arch=True)
        self.assertIsInstance(vae3, LightX2VVAE)

    def test_t4_gpu_attention_mode_resolution(self):
        """Verify that Tesla T4 (compute capability 7.5) correctly resolves to SDPA."""
        from src.models import wan_video_dit
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.get_device_capability', return_value=(7, 5)):
            self.assertFalse(wan_video_dit.is_sm80_or_newer('cuda:0'))

            wan_video_dit.ATTENTION_MODE = 'auto'
            self.assertEqual(wan_video_dit.get_effective_attention_mode('cuda:0'), 'sdpa')

            wan_video_dit.ATTENTION_MODE = 'sparse_sage_attention'
            self.assertEqual(wan_video_dit.get_effective_attention_mode('cuda:0'), 'sdpa')

            wan_video_dit.ATTENTION_MODE = 'flash_attention_2'
            self.assertEqual(wan_video_dit.get_effective_attention_mode('cuda:0'), 'sdpa')

    def test_self_attention_sdpa_forward(self):
        """Verify SelfAttention executes SDPA correctly without sparse mask on CPU/T4."""
        from src.models.wan_video_dit import SelfAttention
        import src.models.wan_video_dit as wvd
        wvd.ATTENTION_MODE = 'sdpa'
        dim = 64
        num_heads = 4
        attn = SelfAttention(dim=dim, num_heads=num_heads)
        attn.eval()

        B, f, h, w = 1, 2, 8, 8
        L = f * h * w
        x = torch.randn(B, L, dim)
        head_dim = dim // num_heads
        freqs = torch.polar(torch.ones(1, L, 1, head_dim // 2), torch.zeros(1, L, 1, head_dim // 2))

        with torch.no_grad():
            out = attn(x, freqs, f=f, h=h, w=w, is_stream=False)
        self.assertEqual(out.shape, (B, L, dim))

    def test_dit_block_non_stream_unpacking(self):
        """Verify DiTBlock executes with is_stream=False without unpacking error."""
        from src.models.wan_video_dit import DiTBlock
        import src.models.wan_video_dit as wvd
        wvd.ATTENTION_MODE = 'sdpa'
        dim = 64
        num_heads = 4
        ffn_dim = 128
        block = DiTBlock(dim=dim, num_heads=num_heads, ffn_dim=ffn_dim)
        block.eval()

        B, f, h, w = 1, 2, 8, 8
        L = f * h * w
        x = torch.randn(B, L, dim)
        context = torch.randn(B, 16, dim)
        block.cross_attn.init_cache(context)
        t_mod = torch.randn(B, 6, dim)
        head_dim = dim // num_heads
        freqs = torch.polar(torch.ones(1, L, 1, head_dim // 2), torch.zeros(1, L, 1, head_dim // 2))

        with torch.no_grad():
            out = block(x, context, t_mod, freqs, f=f, h=h, w=w, is_stream=False)
        self.assertEqual(out.shape, (B, L, dim))

    # =========================================================================
    # Tests for Feature 1: Motion-Aware Dynamic Attention Budgeting
    # =========================================================================
    def test_motion_profile_and_budget_controller(self):
        video = torch.randn(9, 64, 64, 3)
        chunks = 3
        motion = motion_profile(video, chunks)
        self.assertEqual(len(motion), chunks)
        for m in motion:
            self.assertTrue(0.0 <= m <= 1.0)

        # Test quality profiles
        for profile in ['detail', 'balanced', 'throughput']:
            ctrl = AttentionBudgetController(motion=motion, quality_profile=profile, intensity=1.0)
            topk, kv, r = ctrl.values_for(0, base_topk=2.0, base_kv=3.0, base_range=11)
            self.assertGreater(topk, 0.0)
            self.assertGreater(kv, 0.0)
            self.assertIn(r, [9, 11])
            self.assertTrue(len(ctrl.summary()) > 0)

    # =========================================================================
    # Tests for Feature 2: Memory Staging Execution Plan
    # =========================================================================
    def test_memory_policy_and_execution_plan(self):
        # Explicit policy checks
        self.assertTrue(ExecutionPlan.should_stage(100, 100, 10, memory_policy='staged'))
        self.assertFalse(ExecutionPlan.should_stage(1000, 1000, 100, memory_policy='resident'))

        # Auto policy with simulated T4 GPU (<= 16GB)
        with patch.object(ExecutionPlan, 'get_total_vram_gb', return_value=15.0), \
             patch.object(ExecutionPlan, 'get_available_vram_gb', return_value=7.0), \
             patch('torch.cuda.is_available', return_value=True):
            should_stage = ExecutionPlan.should_stage(720, 1280, 50, memory_policy='auto')
            self.assertTrue(should_stage)

    # =========================================================================
    # Tests for Feature 3: Filename Prefix Resolution
    # =========================================================================
    def test_filename_prefix_resolution(self):
        res = resolve_filename_prefix("test_output_%width%x%height%", width=1920, height=1080, out_dir="/tmp")
        self.assertIn("1920x1080", res)
        self.assertTrue(res.endswith(".mp4"))

        # Test date format token replacement
        res_date = resolve_filename_prefix("video_%date:yyyyMMdd%", out_dir="/tmp")
        self.assertNotIn("%date:", res_date)

    # =========================================================================
    # Tests for Feature 5: Spatial Tiling & Blending
    # =========================================================================
    def test_spatial_executor_and_feather_mask(self):
        executor = SpatialExecutor(strategy="adaptive_tiles", max_tile_edge=256, blend_overlap=24)
        self.assertTrue(executor.should_tile(512, 512))
        self.assertFalse(executor.should_tile(128, 128))

        # Test smooth feather mask
        mask = compute_feather_mask(256, 256, overlap=24)
        self.assertEqual(mask.shape, (1, 256, 256, 1))
        # Center should be 1.0, corners should be smaller
        self.assertAlmostEqual(mask[0, 128, 128, 0].item(), 1.0, places=3)
        self.assertLess(mask[0, 0, 0, 0].item(), 0.5)

    # =========================================================================
    # Tests for Feature 6: Tensor Sanity Validation (NaN / Inf)
    # =========================================================================
    def test_tensor_validation(self):
        clean_tensor = torch.randn(4, 4)
        self.assertTrue(validate_tensor(clean_tensor))

        nan_tensor = torch.tensor([1.0, float('nan'), 3.0])
        with self.assertRaises(TensorValidationError):
            validate_tensor(nan_tensor, raise_error=True)

        # Auto-clean test
        repaired = nan_tensor.clone()
        validate_tensor(repaired, raise_error=False, auto_clean=True)
        self.assertFalse(torch.isnan(repaired).any().item())

    # =========================================================================
    # Tests for Feature 3: Hardware-Accelerated Video Encoder
    # =========================================================================
    def test_video_encoding_ffmpeg(self):
        from src.te_speed.video_encoder import encode_video_ffmpeg
        frames = torch.randint(0, 255, (5, 64, 64, 3), dtype=torch.uint8)
        out_path = "/tmp/test_unit_encoding.mp4"
        if os.path.exists(out_path):
            os.remove(out_path)
        encode_video_ffmpeg(frames, out_path, fps=10.0, codec="libx264")
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 0)
        if os.path.exists(out_path):
            os.remove(out_path)
    # =========================================================================
    # Tests for Ported FlashVSR-Pro Features (1, 2, 4, 6, 8, 10)
    # =========================================================================
    def test_sha256_verification(self):
        from src.processor import sha256_file, verify_model_file
        import tempfile, hashlib
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"flashvsr model weights content")
            temp_path = f.name
        try:
            expected = hashlib.sha256(b"flashvsr model weights content").hexdigest()
            self.assertEqual(sha256_file(temp_path), expected)
            self.assertTrue(verify_model_file(temp_path, expected))
            self.assertFalse(verify_model_file(temp_path, "wrong_hash"))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_atomic_output(self):
        from src.media import atomic_output
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            dest = os.path.join(td, "target.txt")
            with atomic_output(dest) as tmp:
                tmp.write_text("verified content")
            self.assertTrue(os.path.exists(dest))
            with open(dest) as f:
                self.assertEqual(f.read(), "verified content")

            # Failure case: original untouched
            dest_fail = os.path.join(td, "keep_orig.txt")
            with open(dest_fail, "w") as f:
                f.write("original")
            try:
                with atomic_output(dest_fail) as tmp:
                    tmp.write_text("corrupted")
                    raise RuntimeError("simulated crash")
            except RuntimeError:
                pass
            with open(dest_fail) as f:
                self.assertEqual(f.read(), "original")

    def test_quality_and_media_tools(self):
        from src.media import quality_to_crf, MediaTools
        self.assertEqual(quality_to_crf(10), 20)
        self.assertEqual(quality_to_crf(0), 26)
        self.assertEqual(quality_to_crf(8), 21)
        media = MediaTools()
        self.assertIsNotNone(media.ffmpeg)
        self.assertIsNotNone(media.ffprobe)

    def test_jsonl_logging(self):
        from src.flashvsr_log import configure_logging, get_logger
        import tempfile, json
        with tempfile.TemporaryDirectory() as td:
            log_file = os.path.join(td, "run.jsonl")
            configure_logging(level="INFO", log_file=log_file)
            logger = get_logger("unit_test")
            logger.info("Test message for JSONL")
            self.assertTrue(os.path.exists(log_file))
            with open(log_file) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 1)
            data = json.loads(lines[0])
            self.assertEqual(data["message"], "Test message for JSONL")
            self.assertEqual(data["level"], "INFO")
if __name__ == '__main__':
    unittest.main()
