from surface_lf import SurfaceLF, SurfaceLFRig
import torch
from PIL import Image
import numpy as np
import os
import torch.nn.functional as F
from loss import loss
from utils import compose_pose, matrix_to_axis_angle


def refine_pose(
    surface_lf_prev,
    image,
    depth,
    pose_coarse,
    i,
    mask_prev,
    mask,
    num_iterations: int = 50,
    lr_translation: float = 1e-2,
    lr_rotation: float = 5e-3,
):
    device = image.device
    values = surface_lf_prev.transform(pose_coarse)
    surf_image, surf_depth = surface_lf_prev.rasterize(values)
    loss_before, breakdown = loss(
        surf_image,
        surf_depth,
        mask_prev,
        image,
        depth,
        mask,
        pose_coarse,
        pose_coarse,
        # visualize=True,
        # i=i,
    )

    # --- Initialize parameters from coarse pose ---
    R_init = pose_coarse[:3, :3]
    t_init = pose_coarse[:3, 3]

    # Convert initial rotation to axis-angle (approx small-angle assumption)
    rotation_init = matrix_to_axis_angle(R_init).detach()
    translation_param = t_init.clone().detach().requires_grad_(True)
    rotation_param = rotation_init.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam(
        [
            {"params": translation_param, "lr": lr_translation},
            {"params": rotation_param, "lr": lr_rotation},
        ]
    )

    best_loss = float("inf")
    best_pose = pose_coarse.clone()

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        pose_delta = compose_pose(rotation_param, translation_param)
        pose_current = pose_delta @ pose_coarse

        surf_values = surface_lf_prev.transform(pose_current)
        surf_image, surf_depth = surface_lf_prev.rasterize(surf_values)

        loss_value, _ = loss(
            surf_image,
            surf_depth,
            mask_prev,
            image,
            depth,
            mask,
            pose_coarse,  # anchor
            pose_current,  # refined
            visualize=False,
            i=0,
        )

        loss_value.backward()

        # Optional: gradient clipping for stability
        torch.nn.utils.clip_grad_norm_([rotation_param, translation_param], 10.0)

        optimizer.step()

        if loss_value.item() < best_loss:
            best_loss = loss_value.item()
            best_pose = pose_current.detach().clone()

    # --- Final visualization pass ---
    # surf_values = surface_lf_prev.transform(best_pose)
    # surf_image, surf_depth = surface_lf_prev.rasterize(surf_values)

    # final_loss, breakdown = loss(
    #     surf_image,
    #     surf_depth,
    #     mask_prev,
    #     image,
    #     depth,
    #     mask,
    #     pose_coarse,
    #     best_pose,
    #     visualize=True,
    #     i=i,
    # )

    # print(
    #     f"[Frame {i}] loss_before = {loss_before.item():6f}, final_loss = {final_loss.item():.6f}"
    # )

    return best_pose


if __name__ == "__main__":
    K = torch.load("K.pt")
    poses = torch.load("poses_4x4.pt")
    surface_lf_rig = SurfaceLFRig.build(
        K=K,
        poses_4x4=poses,
        image_size_hw=(720, 1280),
    )
    poses = [
        torch.load(f"coarse_pose_{i:04d}.pt", weights_only=True) for i in range(20)
    ]
    poses_gt = [torch.load(f"pose_gt{i:04d}.pt", weights_only=True) for i in range(20)]
    poses_rel = [torch.eye(4).cuda()]
    poses_refined = [torch.eye(4).cuda()]
    for i in range(1, 20):
        pose_rel = poses[i] @ torch.linalg.inv(poses[i - 1])
        poses_rel.append(pose_rel)
    for i in range(20):
        surface_lf = SurfaceLF(
            rig=surface_lf_rig,
            pc=torch.load(f"pc_{i:04d}.pt"),
            images=torch.load(f"images_{i:04d}.pt"),
        )
        mask = torch.load(f"mask_{i:04d}.pt")
        image, depth = surface_lf.rasterize()
        if i > 0:
            pose_refined = refine_pose(
                surface_lf_prev, image, depth, poses_rel[i], i, mask_prev, mask
            )
            poses_refined.append(pose_refined)

        surface_lf_prev = surface_lf
        mask_prev = mask
    poses_gt = torch.stack(poses_gt, dim=0).cpu().numpy()
    poses_coarse = torch.stack(poses, dim=0).cpu().numpy()
    poses_refined = torch.stack(poses_refined, dim=0).cpu().numpy()
    from tracking import rebase_poses, pose_errors

    result_poses_before = rebase_poses(poses_gt, poses_refined)
    result_poses_after = rebase_poses(poses_gt, poses_coarse)
    print(pose_errors(poses_gt, result_poses_before))
    print(pose_errors(poses_gt, result_poses_after))
