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


def refine_pose(
    surface_lf_prev,
    pose_coarse: torch.Tensor,
    image: torch.Tensor,
    depth: torch.Tensor,
    mask: torch.Tensor,
    mask_prev: torch.Tensor,
    loss_fn,
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

    best_loss_value = float("inf")
    best_pose = pose_coarse.detach().clone()

    for _ in range(int(num_iterations)):
        optimizer.zero_grad(set_to_none=True)

        pose_delta = compose_pose_fn(rotation_param, translation_param)
        pose_current = pose_coarse @ pose_delta

        surf_values = surface_lf_prev.transform(pose_current)
        surf_image, surf_depth = surface_lf_prev.rasterize(surf_values)

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
    # v = Visualizer()
    K = torch.load("pts/K.pt")
    poses = torch.load("pts/poses_4x4.pt")
    poses_object = torch.load("pts/poses_gt.pt")
    surface_lf_rig = SurfaceLFRig.build(
        K=K,
        poses_4x4=poses,
        image_size_hw=(720, 1280),
    )
    poses = [
        torch.load(f"pts/coarse_pose_{i:04d}.pt", weights_only=True) for i in range(20)
    ]
    poses_gt = [torch.load(f"pts/pose_gt{i:04d}.pt") for i in range(20)]
    poses_rel = [torch.eye(4).cuda()]
    poses_refined = [poses_gt[0].cuda()]
    for i in range(1, 20):
        pose_rel = poses[i] @ torch.linalg.inv(poses[i - 1])
        poses_rel.append(pose_rel)
    for i in range(20):
        surface_lf = SurfaceLF(
            rig=surface_lf_rig,
            pc=torch.load(f"pts/pc_{i:04d}.pt"),
            images=torch.load(f"pts/images_{i:04d}.pt"),
        )
        mask = torch.load(f"pts/mask_{i:04d}.pt")
        image, depth = surface_lf.rasterize()
        if i > 0:
            pose_refined, final_loss = refine_pose(
                surface_lf_prev=surface_lf_prev,
                pose_coarse=poses_rel[i],
                image=image,
                depth=depth,
                mask=mask,
                mask_prev=mask_prev,
                loss_fn=loss,
                compose_pose_fn=compose_pose,
            )
            poses_refined.append(pose_refined @ poses_refined[-1])
        surface_lf_prev = surface_lf
        mask_prev = mask
    poses_refined = torch.stack(poses_refined, dim=0)
    poses_gt = torch.stack(poses_gt, dim=0)
    poses_coarse = torch.stack(poses, dim=0)

    poses_coarse = rebase_poses(poses_gt.cpu().numpy(), poses_coarse.cpu().numpy())
    # poses_refined = rebase_poses(poses_gt.cpu().numpy(), poses_refined.cpu().numpy())
    # for i, (coarse_pose, pose, gt_pose) in enumerate(
    #     zip(
    #         poses_coarse,
    #         poses_refined.cpu().numpy(),
    #         poses_gt.cpu().numpy(),
    #     )
    # ):
    #     v.add_frame(f"refined_{i:04d}", pose)
    #     # v.add_frame(f"gt_{i:04d}", gt_pose)
    #     v.add_frame(f"coarse_{i:04d}", coarse_pose)
    print(pose_errors(poses_gt.cpu().numpy(), poses_coarse))
    print(pose_errors(poses_gt.cpu().numpy(), poses_refined.cpu().numpy()))
    # v.run()
