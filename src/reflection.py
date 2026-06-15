"""Reflection separation on a surface light field.

Turns a SurfaceLightField into a diffuse central-view image (uint8 RGB) suitable
for feature matching.  The heavy lifting lives in ``reflection_separation`` — this
module only packs the per-point colours into the pixel grid that solver expects,
handles sRGB/linear conversion, and caches the result to disk (separation is slow).

To swap in a different separator, replace the body of ``compute_diffuse``.
"""

from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image

from reflection_separation import separate_reflection
from src.surface_light_field import SurfaceLightField
from utils import linear_to_srgb, srgb_to_linear


def central_view(frame: dict, s_size: int, t_size: int) -> np.ndarray:
    """Raw central sub-aperture view → [H, W, 3] uint8.

    The LoFTR fallback used when reflection separation is disabled.
    """
    img = frame["LF"][s_size // 2, t_size // 2].cpu().numpy()
    return (img * 255).clip(0, 255).astype(np.uint8)


def compute_diffuse(
    slf: SurfaceLightField,
    alpha: float,
    depth: torch.Tensor,
    iterations: int = 200,
    verbose: bool = True,
) -> np.ndarray:
    """Separate the diffuse component of ``slf`` → [H, W, 3] uint8 RGB.

    ``alpha == 1`` (no reflection) short-circuits inside ``separate_reflection``
    and returns the central view directly, so this is also the cheap path.
    """
    H, W = slf.H, slf.W
    mask = slf.mask.cpu()

    colors = slf.colors.permute(1, 0, 2).cpu()  # [N, V, 3]
    explicit_lf = torch.zeros(H, W, colors.shape[1], 3)
    explicit_lf[mask] = colors
    explicit_lf = srgb_to_linear(explicit_lf).reshape(H, W, slf.s_size, slf.t_size, 3)

    normal_map = torch.zeros(H, W, 3)
    normal_map[mask] = slf.normals.cpu().float()

    depth_map = depth.cpu().clone().float()
    depth_map[~mask] = 0.0

    diffuse, _ = separate_reflection(
        explicit_surface_lf=explicit_lf,
        alpha=alpha,
        mask=mask.float(),
        normal_map=normal_map,
        depth_map=depth_map,
        iterations=iterations,
        verbose=verbose,
    )
    diffuse = linear_to_srgb(diffuse)
    return (diffuse.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def frame_diffuse(
    frame: dict,
    mask: torch.Tensor,
    depth: torch.Tensor,
    alpha: float,
    s_size: int,
    t_size: int,
    cache_path: str | None = None,
    iterations: int = 200,
    verbose: bool = True,
) -> np.ndarray:
    """Diffuse central view for one frame, loading from / saving to ``cache_path``.

    On a cache hit the surface light field is never built and separation is skipped.
    """
    if cache_path is not None and os.path.exists(cache_path):
        return np.array(Image.open(cache_path))

    slf = SurfaceLightField.from_frame(frame, mask, depth, s_size, t_size)
    diffuse = compute_diffuse(slf, alpha, depth, iterations=iterations, verbose=verbose)

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        Image.fromarray(diffuse).save(cache_path)
    return diffuse
