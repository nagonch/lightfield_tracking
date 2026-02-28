from surface_lf import SurfaceLF, SurfaceLFRig
import torch
from PIL import Image
import numpy as np
import os
import torch.nn.functional as F
from loss import loss
from utils import compose_pose, matrix_to_axis_angle
from tracking import rebase_poses, pose_errors
from src.utilities import Visualizer
import math
import matplotlib.pyplot as plt


def pose_delta_about_pivot(
    compose_pose_fn, rotation_param, translation_param, pivot_world: torch.Tensor
):
    """
    rotation_param: (3,) axis-angle (or whatever compose_pose_fn expects)
    translation_param: (3,) translation in world coords *in the pivoted parameterization*
    pivot_world: (3,) world-space pivot
    """
    # pose that rotates about origin + translates by translation_param
    delta_origin = compose_pose_fn(rotation_param, torch.zeros_like(translation_param))

    # Extract R from delta_origin (assuming 4x4)
    rotation_matrix = delta_origin[:3, :3]

    # Convert to equivalent LHS translation that corresponds to rotating about pivot_world
    # t_lhs = t + (I - R) c
    identity_3 = torch.eye(
        3, device=rotation_matrix.device, dtype=rotation_matrix.dtype
    )
    translation_lhs = translation_param + (identity_3 - rotation_matrix) @ pivot_world

    delta_lhs = compose_pose_fn(rotation_param, translation_lhs)
    return delta_lhs


def refine_pose(
    surface_lf_prev,
    pose_coarse: torch.Tensor,
    image: torch.Tensor,
    depth: torch.Tensor,
    mask: torch.Tensor,
    mask_prev: torch.Tensor,
    loss_fn,
    pivot_world: torch.Tensor,
    compose_pose_fn,
    num_iterations: int = 500,
    learning_rate_rot: float = 1e-3,
    learning_rate_trans: float = 1e-3,
    grad_clip_norm: float = 10.0,
    device: str | torch.device | None = None,
):
    if device is None:
        device = pose_coarse.device

    pose_coarse = pose_coarse.to(device)
    image = image.to(device)
    depth = depth.to(device)
    mask = mask.to(device)
    mask_prev = mask_prev.to(device)

    rotation_param = (
        torch.zeros(3, device=device) + torch.randn(3, device=device) * 1e-4
    ).requires_grad_()

    translation_param = (
        torch.zeros(3, device=device) + torch.randn(3, device=device) * 1e-4
    ).requires_grad_()

    optimizer = torch.optim.Adam(
        [
            {"params": [rotation_param], "lr": float(learning_rate_rot)},
            {"params": [translation_param], "lr": float(learning_rate_trans)},
        ]
    )
    pivot_world = pivot_world.to(device)
    best_loss_value = float("inf")
    best_pose = pose_coarse.detach().clone()

    for _ in range(int(num_iterations)):
        optimizer.zero_grad(set_to_none=True)

        pose_delta = pose_delta_about_pivot(
            compose_pose_fn, rotation_param, translation_param, pivot_world
        )
        pose_current = pose_delta @ pose_coarse  # LHS update, but pivot-conditioned

        surf_image, surf_depth = surface_lf_prev.rasterize(pose_current)

        loss_value, _ = loss_fn(
            surf_image,
            surf_depth,
            mask_prev,
            image,
            depth,
            mask,
            pose_coarse,
            pose_current,
            False,
            0,
        )

        loss_value.backward()

        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [rotation_param, translation_param], float(grad_clip_norm)
            )

        optimizer.step()

        loss_scalar = float(loss_value.detach().cpu().item())
        if loss_scalar < best_loss_value:
            best_loss_value = loss_scalar
            best_pose = pose_current.detach().clone()
    print(f"refine_pose final_loss = {best_loss_value:.6f}")
    return best_pose, best_loss_value


if __name__ == "__main__":
    data = torch.load("data.pt", weights_only=False)
    pose_rel_lhs = data["pose_rel_lhs"]
    surface_lf_prev = data["surface_lf_prev"]
    image = data["image"]
    depth = data["depth"]
    mask = data["mask"]
    mask_prev = data["mask_prev"]
    pivot_world = data["pivot_world"]
    pose_rel_gt = data["pose_rel_gt"]
    best_pose, best_loss = refine_pose(
        surface_lf_prev=surface_lf_prev,
        pose_coarse=torch.tensor(pose_rel_lhs, dtype=torch.float32).cuda(),
        image=image,
        depth=depth,
        mask=mask,
        mask_prev=mask_prev,
        loss_fn=loss,
        pivot_world=pivot_world,
        compose_pose_fn=compose_pose,
        num_iterations=500,
        learning_rate_rot=1e-3,
    )
    print(torch.norm(pose_rel_gt - best_pose).item())
    print(torch.norm(pose_rel_gt - torch.tensor(pose_rel_lhs).cuda()).item())

    print(torch.norm(pose_rel_gt[:3, :3] - best_pose[:3, :3]).item())
    print(
        torch.norm(
            pose_rel_gt[:3, :3] - torch.tensor(pose_rel_lhs[:3, :3]).cuda()
        ).item()
    )

    print(torch.norm(pose_rel_gt[:3, 3] - best_pose[:3, 3]).item())
    print(torch.norm(pose_rel_gt[:3, 3] - best_pose[:3, 3]).item())
