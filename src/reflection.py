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

from reflection_separation import estimate_alpha, separate_reflection
from src.surface_light_field import SurfaceLightField
from utils import linear_to_srgb


def central_view(frame: dict, s_size: int, t_size: int) -> np.ndarray:
    """Raw central sub-aperture view → [H, W, 3] float32 linear [0, 1].

    The LoFTR fallback used when reflection separation is disabled.
    LF is already in linear space (converted in dataset.__getitem__).
    """
    return frame["LF"][s_size // 2, t_size // 2].cpu().numpy().astype(np.float32)


def _env_cache_path(cache_path: str) -> str:
    return os.path.splitext(cache_path)[0] + "_env.npy"


def _envconf_cache_path(cache_path: str) -> str:
    return os.path.splitext(cache_path)[0] + "_envconf.npy"


def _alpha_cache_path(cache_path: str) -> str:
    return os.path.splitext(cache_path)[0] + "_alpha.npy"


def compute_diffuse(
    slf: SurfaceLightField,
    alpha: float | None = None,
    iterations: int = 200,
    verbose: bool = True,
    previous_environment_map: torch.Tensor | None = None,
    previous_env_confidence: torch.Tensor | None = None,
    alpha_stat_history: list[float] | None = None,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, float, list[float]]:
    """Separate ``slf`` into a diffuse central view + a reflected environment map.

    ``alpha`` (diffuse fraction) is estimated on the fly from the surface light
    field when ``None``, accumulating evidence across frames via
    ``alpha_stat_history`` (alpha is a material constant, so the estimate sharpens
    as more frames are seen). Pass a float to override with a fixed value.

    Returns ``(diffuse_linear [H, W, 3] float32 [0,1], environment_map
    [env_h, env_w, 3], env_confidence [env_h, env_w], alpha, alpha_stat_history)``.
    LF colors are already in linear space (converted at dataset load time), so no
    sRGB conversion is applied here.  The env map stays linear throughout.
    ``env_confidence`` marks observed (high) vs interpolated (low) env pixels.
    """
    # Pack the per-point surface light field as [P, M, *] for the point-based solver.
    colors = slf.colors.permute(1, 0, 2)  # [P, M, 3]  — already linear
    view_dirs = slf.view_dirs.permute(1, 0, 2)  # [P, M, 3]
    valid = slf.valid.permute(1, 0)  # [P, M]
    normals = slf.normals.float()  # [P, 3]

    normals_rep = normals[:, None, :].expand_as(view_dirs)
    reflected_dirs = F.normalize(
        view_dirs
        - 2.0 * (view_dirs * normals_rep).sum(dim=-1, keepdim=True) * normals_rep,
        dim=-1,
    )

    # Estimate the diffuse fraction from frames seen so far when not supplied; an
    # env map is always fit (alpha is clamped away from a perfect mirror/diffuser).
    history = list(alpha_stat_history) if alpha_stat_history else []
    if alpha is None:
        alpha, history = estimate_alpha(
            colors, valid, view_dirs, normals, stat_history=history
        )

    diffuse_point, environment_map, _, env_confidence = separate_reflection(
        colors=colors,  # already linear — no srgb_to_linear needed
        alpha=alpha,
        reflected_dirs=reflected_dirs,
        valid=valid,
        view_dirs=view_dirs,
        normals=normals,
        mask=slf.mask,
        previous_environment_map=previous_environment_map,
        previous_env_confidence=previous_env_confidence,
        iterations=iterations,
        verbose=verbose,
    )

    # Keep the per-point diffuse on the SLF for differentiable relighting.
    slf.diffuse_colors = diffuse_point.detach().clamp(0.0, 1.0)

    # Scatter the per-point diffuse to the central-view grid; keep in linear.
    diffuse_img = torch.zeros(slf.H, slf.W, 3, device=diffuse_point.device)
    diffuse_img[slf.mask] = diffuse_point
    diffuse_linear = diffuse_img.clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
    return diffuse_linear, environment_map, env_confidence, float(alpha), history


def _diffuse_npy_path(cache_path: str) -> str:
    """Derive the .npy cache path for the linear diffuse image."""
    return os.path.splitext(cache_path)[0] + ".npy"


def frame_diffuse(
    frame: dict,
    mask: torch.Tensor,
    depth: torch.Tensor,
    alpha: float | None = None,
    s_size: int = 5,
    t_size: int = 5,
    cache_path: str | None = None,
    iterations: int = 200,
    verbose: bool = True,
    previous_environment_map: torch.Tensor | None = None,
    previous_env_confidence: torch.Tensor | None = None,
    alpha_stat_history: list[float] | None = None,
) -> tuple[
    np.ndarray, torch.Tensor | None, torch.Tensor | None, SurfaceLightField, float,
    list[float],
]:
    """Diffuse central view + env map + the SLF for one frame, with disk caching.

    ``alpha`` is estimated on the fly from the surface light field when ``None``,
    accumulating across frames through ``alpha_stat_history`` (threaded forward by
    the caller). Pass a float to pin it to a known value.

    Returns ``(diffuse_linear [H,W,3] float32 [0,1], environment_map,
    env_confidence, slf, alpha, alpha_stat_history)``.  ``env_confidence`` is the
    [env_h, env_w] observation map (``None`` on legacy caches without it, in which
    case the relight loss falls back to uniform weighting).  The
    :class:`SurfaceLightField` (gsplat-backed, with
    per-point diffuse) is always rebuilt so the photometric refinement can
    rasterize it; only the slow reflection separation is skipped on a cache hit.
    The cached diffuse image and the SLF share the same masked-point ordering, so
    the per-point diffuse is reconstructed exactly from the image via ``slf.mask``.

    Cache is stored as a float32 .npy file (linear, no gamma encoding).
    Old .png caches are silently ignored and regenerated as .npy.
    """
    slf = SurfaceLightField.from_frame(frame, mask, depth, s_size, t_size)
    history = list(alpha_stat_history) if alpha_stat_history else []

    if cache_path is not None:
        npy_path = _diffuse_npy_path(cache_path)
        if os.path.exists(npy_path):
            diffuse = np.load(npy_path)
            env_path = _env_cache_path(cache_path)
            env = (
                torch.from_numpy(np.load(env_path)).cuda()
                if os.path.exists(env_path)
                else None
            )
            envconf_path = _envconf_cache_path(cache_path)
            env_conf = (
                torch.from_numpy(np.load(envconf_path)).cuda()
                if os.path.exists(envconf_path)
                else None
            )
            alpha_path = _alpha_cache_path(cache_path)
            if alpha is not None:
                alpha_cached = float(alpha)
            elif os.path.exists(alpha_path):
                alpha_cached = float(np.load(alpha_path))
            else:
                # Older cache without a stored alpha: recover it from the SLF.
                colors = slf.colors.permute(1, 0, 2)
                alpha_cached, history = estimate_alpha(
                    colors,
                    slf.valid.permute(1, 0),
                    slf.view_dirs.permute(1, 0, 2),
                    slf.normals.float(),
                    stat_history=history,
                )
                alpha_cached = float(alpha_cached)
            diffuse_t = torch.from_numpy(diffuse).to(slf.points.device)
            slf.diffuse_colors = diffuse_t[slf.mask].clamp(0.0, 1.0)
            return diffuse, env, env_conf, slf, alpha_cached, history

    diffuse, environment_map, env_confidence, alpha, history = compute_diffuse(
        slf,
        alpha,
        iterations=iterations,
        verbose=verbose,
        previous_environment_map=previous_environment_map,
        previous_env_confidence=previous_env_confidence,
        alpha_stat_history=history,
    )

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(_diffuse_npy_path(cache_path), diffuse)
        np.save(_env_cache_path(cache_path), environment_map.detach().cpu().numpy())
        np.save(
            _envconf_cache_path(cache_path),
            env_confidence.detach().cpu().numpy(),
        )
        np.save(_alpha_cache_path(cache_path), np.asarray(alpha, dtype=np.float32))
    return diffuse, environment_map, env_confidence, slf, alpha, history
