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
import torch.nn.functional as F
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


def _env_cache_path(cache_path: str) -> str:
    return os.path.splitext(cache_path)[0] + "_env.npy"


def compute_diffuse(
    slf: SurfaceLightField,
    alpha: float,
    iterations: int = 200,
    verbose: bool = True,
    previous_environment_map: torch.Tensor | None = None,
) -> tuple[np.ndarray, torch.Tensor]:
    """Separate ``slf`` into a diffuse central view + a reflected environment map.

    Returns ``(diffuse_uint8 [H, W, 3], environment_map [env_h, env_w, 3])``.
    ``alpha == 1`` (no reflection) short-circuits inside ``separate_reflection``.
    ``previous_environment_map`` warm-starts/anchors the env map for multi-frame
    accumulation as the object reorients in the (static) camera frame.
    """
    # Pack the per-point surface light field as [P, M, *] for the point-based solver.
    colors = slf.colors.permute(1, 0, 2)  # [P, M, 3]
    view_dirs = slf.view_dirs.permute(1, 0, 2)  # [P, M, 3]
    valid = slf.valid.permute(1, 0)  # [P, M]
    normals = slf.normals.float()  # [P, 3]

    normals_rep = normals[:, None, :].expand_as(view_dirs)
    reflected_dirs = F.normalize(
        view_dirs
        - 2.0 * (view_dirs * normals_rep).sum(dim=-1, keepdim=True) * normals_rep,
        dim=-1,
    )

    diffuse_point, environment_map, _ = separate_reflection(
        colors=srgb_to_linear(colors),
        alpha=alpha,
        reflected_dirs=reflected_dirs,
        valid=valid,
        view_dirs=view_dirs,
        normals=normals,
        previous_environment_map=previous_environment_map,
        iterations=iterations,
        verbose=verbose,
    )

    # Scatter the per-point diffuse back to the central-view image grid.
    diffuse_img = torch.zeros(slf.H, slf.W, 3, device=diffuse_point.device)
    diffuse_img[slf.mask] = diffuse_point
    diffuse_img = linear_to_srgb(diffuse_img)
    diffuse_u8 = (diffuse_img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return diffuse_u8, environment_map


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
    previous_environment_map: torch.Tensor | None = None,
) -> tuple[np.ndarray, torch.Tensor | None]:
    """Diffuse central view + env map for one frame, with disk caching.

    Returns ``(diffuse_uint8, environment_map)``.  On a cache hit the surface
    light field is never built and separation is skipped; the env map is loaded
    alongside the diffuse image (``None`` if it was never cached) so multi-frame
    accumulation survives resumed runs.
    """
    if cache_path is not None and os.path.exists(cache_path):
        diffuse = np.array(Image.open(cache_path))
        env_path = _env_cache_path(cache_path)
        env = None
        if os.path.exists(env_path):
            env = torch.from_numpy(np.load(env_path)).cuda()
        return diffuse, env

    slf = SurfaceLightField.from_frame(frame, mask, depth, s_size, t_size)
    diffuse, environment_map = compute_diffuse(
        slf,
        alpha,
        iterations=iterations,
        verbose=verbose,
        previous_environment_map=previous_environment_map,
    )

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        Image.fromarray(diffuse).save(cache_path)
        np.save(_env_cache_path(cache_path), environment_map.detach().cpu().numpy())
    return diffuse, environment_map
