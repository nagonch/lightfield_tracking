"""Reflection separation for one LF frame: SurfaceLightField → diffuse view + env map."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from reflection_separation import estimate_alpha, separate_reflection
from src.surface_light_field import SurfaceLightField


def frame_diffuse(
    frame: dict,
    mask: torch.Tensor,
    depth: torch.Tensor,
    s_size: int,
    t_size: int,
    alpha: float | None = None,
    iterations: int = 300,
    previous_environment_map: torch.Tensor | None = None,
    previous_env_confidence: torch.Tensor | None = None,
    alpha_stat_history: list[float] | None = None,
) -> tuple[
    np.ndarray, torch.Tensor, torch.Tensor, SurfaceLightField, float, list[float]
]:
    """Build the SLF for one frame and separate it into a diffuse central view
    and a reflected environment map.

    alpha is estimated from the SLF when None, accumulated via alpha_stat_history.
    Returns (diffuse_linear [H,W,3] float32, env_map, env_confidence, slf, alpha,
    history).
    """
    slf = SurfaceLightField.from_frame(frame, mask, depth, s_size, t_size)

    # Pack the per-point surface light field as [P, M, *] for the point-based solver.
    colors = slf.colors.permute(1, 0, 2)  # [P, M, 3]  linear
    view_dirs = slf.view_dirs.permute(1, 0, 2)  # [P, M, 3]
    valid = slf.valid.permute(1, 0)  # [P, M]
    normals = slf.normals.float()  # [P, 3]

    normals_rep = normals[:, None, :].expand_as(view_dirs)
    reflected_dirs = F.normalize(
        view_dirs
        - 2.0 * (view_dirs * normals_rep).sum(dim=-1, keepdim=True) * normals_rep,
        dim=-1,
    )

    # Estimate the diffuse fraction from frames seen so far when not supplied.
    history = list(alpha_stat_history) if alpha_stat_history else []
    if alpha is None:
        alpha, history = estimate_alpha(
            colors, valid, view_dirs, normals, stat_history=history
        )

    diffuse_point, environment_map, _, env_confidence = separate_reflection(
        colors=colors,
        alpha=alpha,
        reflected_dirs=reflected_dirs,
        valid=valid,
        view_dirs=view_dirs,
        normals=normals,
        mask=slf.mask,
        previous_environment_map=previous_environment_map,
        previous_env_confidence=previous_env_confidence,
        iterations=iterations,
    )

    # Keep the per-point diffuse on the SLF for differentiable relighting.
    slf.diffuse_colors = diffuse_point.detach().clamp(0.0, 1.0)

    # Scatter the per-point diffuse to the central-view grid; keep in linear.
    diffuse_img = torch.zeros(slf.H, slf.W, 3, device=diffuse_point.device)
    diffuse_img[slf.mask] = diffuse_point
    diffuse_linear = diffuse_img.clamp(0.0, 1.0).cpu().numpy().astype(np.float32)
    return diffuse_linear, environment_map, env_confidence, slf, float(alpha), history
