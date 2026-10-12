"""
Adaptive Spatial Tile Executor for FlashVSR
Handles full-frame vs adaptive tiled spatial processing, with cosine/feather
blending for artifact-free seam transitions and optimal VRAM control.
"""

from typing import List, Tuple, Callable, Optional
import math
import torch
import torch.nn.functional as F


def compute_positions(length: int, limit: int, overlap: int) -> List[Tuple[int, int]]:
    """
    Computes 1D tile ranges [(start, end), ...] covering [0, length].
    """
    if length <= limit:
        return [(0, length)]

    stride = max(1, limit - overlap)
    positions = []
    start = 0
    while start < length:
        end = min(length, start + limit)
        positions.append((start, end))
        if end >= length:
            break
        start += stride

    # Ensure the last tile reaches the very edge
    if positions[-1][1] < length:
        positions.append((max(0, length - limit), length))

    return positions


def compute_grid(
    height: int,
    width: int,
    max_tile_edge: int = 256,
    overlap: int = 24,
) -> List[Tuple[int, int, int, int]]:
    """
    Computes 2D grid coordinates [(x1, y1, x2, y2), ...] for tiles.
    """
    y_ranges = compute_positions(height, max_tile_edge, overlap)
    x_ranges = compute_positions(width, max_tile_edge, overlap)

    grid = []
    for y1, y2 in y_ranges:
        for x1, x2 in x_ranges:
            grid.append((x1, y1, x2, y2))
    return grid


def compute_feather_mask(
    tile_h: int,
    tile_w: int,
    overlap: int,
    device: Optional[str] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Constructs a 2D smooth cosine feathering mask of shape (1, tile_h, tile_w, 1)
    to eliminate boundary seams during tile reconstruction.
    """
    if overlap <= 0:
        return torch.ones((1, tile_h, tile_w, 1), device=device, dtype=dtype)

    actual_overlap = min(overlap, min(tile_h, tile_w) // 2)
    if actual_overlap <= 0:
        return torch.ones((1, tile_h, tile_w, 1), device=device, dtype=dtype)

    ramp = (1.0 - torch.cos(torch.linspace(0, math.pi, actual_overlap, device=device, dtype=dtype))) / 2.0

    mask_h = torch.ones(tile_h, device=device, dtype=dtype)
    mask_h[:actual_overlap] = ramp
    mask_h[-actual_overlap:] = ramp.flip(0)

    mask_w = torch.ones(tile_w, device=device, dtype=dtype)
    mask_w[:actual_overlap] = ramp
    mask_w[-actual_overlap:] = ramp.flip(0)

    mask_2d = torch.outer(mask_h, mask_w).unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)
    return mask_2d


class SpatialExecutor:
    """
    Manages spatial partitioning strategy and tile execution.
    """

    STRATEGY_AUTO = "auto"
    STRATEGY_FULL_FRAME = "full_frame"
    STRATEGY_ADAPTIVE_TILES = "adaptive_tiles"

    def __init__(
        self,
        strategy: str = "auto",
        max_tile_edge: int = 256,
        blend_overlap: int = 24,
    ):
        self.strategy = strategy.lower()
        if self.strategy not in [self.STRATEGY_AUTO, self.STRATEGY_FULL_FRAME, self.STRATEGY_ADAPTIVE_TILES]:
            self.strategy = self.STRATEGY_AUTO

        self.max_tile_edge = max(128, int(max_tile_edge))
        self.blend_overlap = max(8, int(blend_overlap))

    def should_tile(self, height: int, width: int, available_vram_gb: float = 15.0) -> bool:
        if self.strategy == self.STRATEGY_FULL_FRAME:
            return False
        if self.strategy == self.STRATEGY_ADAPTIVE_TILES:
            return height > self.max_tile_edge or width > self.max_tile_edge

        # AUTO Strategy
        if height <= self.max_tile_edge and width <= self.max_tile_edge:
            return False

        # If on <= 16GB GPU (like Colab T4) and resolution is larger than 360p, use tiling
        if available_vram_gb <= 16.5 and (height * width > 360 * 640):
            return True

        return height > self.max_tile_edge or width > self.max_tile_edge

    def execute(
        self,
        frames: torch.Tensor,
        scale: int,
        process_fn: Callable[[torch.Tensor], torch.Tensor],
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> torch.Tensor:
        """
        Executes spatial processing over frames: (N, H, W, C).
        """
        N, H, W, C = frames.shape
        out_H, out_W = H * scale, W * scale

        if not self.should_tile(H, W):
            return process_fn(frames)

        grid = compute_grid(H, W, self.max_tile_edge, self.blend_overlap)
        total_tiles = len(grid)

        # Accumulation canvases on CPU to save GPU VRAM
        output_canvas = torch.zeros((N, out_H, out_W, C), dtype=torch.float32, device="cpu")
        weight_canvas = torch.zeros((N, out_H, out_W, 1), dtype=torch.float32, device="cpu")

        scaled_overlap = self.blend_overlap * scale

        for idx, (x1, y1, x2, y2) in enumerate(grid):
            if progress_cb:
                progress_cb(idx + 1, total_tiles)

            tile_in = frames[:, y1:y2, x1:x2, :]
            tile_out = process_fn(tile_in).to("cpu", dtype=torch.float32)

            tN, tH, tW, tC = tile_out.shape
            out_x1, out_y1 = x1 * scale, y1 * scale
            out_x2, out_y2 = out_x1 + tW, out_y1 + tH

            # Compute feather mask for blending
            mask = compute_feather_mask(tH, tW, scaled_overlap, device="cpu", dtype=torch.float32)

            output_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += tile_out * mask
            weight_canvas[:, out_y1:out_y2, out_x1:out_x2, :] += mask

            del tile_in, tile_out

        weight_canvas = torch.clamp(weight_canvas, min=1e-5)
        final_output = output_canvas / weight_canvas
        return final_output
