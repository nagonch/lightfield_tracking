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

    # --- Initialize parameters from coarse pose ---
    R_init = pose_coarse[:3, :3]
    t_init = pose_coarse[:3, 3]

    # Convert initial rotation to axis-angle (approx small-angle assumption)
    rotation_param = torch.zeros(
        3, device=device, requires_grad=True
    )  # axis-angle delta
    translation_param = torch.zeros(3, device=device, requires_grad=True)

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

        pose_delta = compose_pose(
            rotation_param, translation_param
        )  # local/body increment
        pose_current = pose_coarse @ pose_delta  # right update

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
    surf_values = surface_lf_prev.transform(best_pose)
    surf_image, surf_depth = surface_lf_prev.rasterize(surf_values)

    final_loss, breakdown = loss(
        surf_image,
        surf_depth,
        mask_prev,
        image,
        depth,
        mask,
        pose_coarse,
        best_pose,
        visualize=True,
        i=i,
    )

    print(f"[Frame {i}] final_loss = {final_loss.item():.6f}")

    return best_pose, final_loss.item()


import math
import torch


def refine_pose_nuclear_rotation_multistart(
    surface_lf_prev,
    image,
    depth,
    pose_coarse,  # anchor for loss()
    pose_init,  # center pose you trust (your current refined pose)
    i,
    mask_prev,
    mask,
    init_loss: float,
    num_samples: int = 96,  # 64–192 is typical
    max_angle_deg: float = 25.0,  # be brave here
    topk: int = 5,  # evaluate top-k seeds more carefully
    refine_topk: int = 3,  # run Adam from best 1–3
    adam_iters: int = 25,
    lr_rotation: float = 2e-2,  # higher than your current
    lr_translation: float = 0.0,  # rotation-only first; you can set 1e-3 later if needed
    clip_grad: float = 100.0,  # don’t choke rotation
):
    """
    Nuclear but not overengineered:
    - Sample many random rotation deltas within max_angle.
    - Rank by your existing loss (no gradients).
    - Optionally run a short Adam refinement from the best few.
    """
    device = image.device
    pose_center = pose_init.detach().clone()

    def eval_pose(pose_candidate: torch.Tensor, visualize: bool = False):
        surf_values = surface_lf_prev.transform(pose_candidate)
        surf_image, surf_depth = surface_lf_prev.rasterize(surf_values)
        loss_value, _ = loss(
            surf_image,
            surf_depth,
            mask_prev,
            image,
            depth,
            mask,
            pose_coarse,
            pose_candidate,
            visualize=visualize,
            i=0 if not visualize else i,
        )
        return loss_value

    # --- 1) Hypothesis blast (rotation-only) ---
    candidates = []
    with torch.no_grad():
        # Always include center pose
        center_loss = float(init_loss)
        candidates.append((center_loss, pose_center))

        max_angle_rad = math.radians(max_angle_deg)

        # Random axis directions + random angle in [-max, max]
        # (uniform-ish enough for local search)
        axis = torch.randn(num_samples, 3, device=device)
        axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-9)
        angles = (2.0 * torch.rand(num_samples, device=device) - 1.0) * max_angle_rad
        axis_angle = axis * angles[:, None]  # [N,3]

        zero_t = torch.zeros(3, device=device)
        for k in range(num_samples):
            pose_delta = compose_pose(axis_angle[k], zero_t)
            pose_candidate = pose_center @ pose_delta
            loss_val = float(eval_pose(pose_candidate, visualize=False).item())
            candidates.append((loss_val, pose_candidate))

    candidates.sort(key=lambda x: x[0])
    best_loss_value, best_pose = candidates[0]
    top_candidates = candidates[: max(topk, 1)]

    # --- 2) Short Adam refine from best few seeds ---
    final_best_loss = best_loss_value
    final_best_pose = best_pose.detach().clone()

    for seed_idx in range(min(refine_topk, len(top_candidates))):
        seed_loss, seed_pose = top_candidates[seed_idx]
        seed_pose = seed_pose.detach().clone()

        rotation_param = torch.zeros(3, device=device, requires_grad=True)
        translation_param = torch.zeros(3, device=device, requires_grad=True)

        params = [{"params": rotation_param, "lr": lr_rotation}]
        if lr_translation > 0.0:
            params.append({"params": translation_param, "lr": lr_translation})

        optimizer = torch.optim.Adam(params)

        best_local_loss = float(seed_loss)
        best_local_pose = seed_pose.detach().clone()

        for _ in range(adam_iters):
            optimizer.zero_grad()

            pose_delta = compose_pose(rotation_param, translation_param)
            pose_current = seed_pose @ pose_delta  # local right-update around seed

            loss_value = eval_pose(pose_current, visualize=False)
            loss_value.backward()

            torch.nn.utils.clip_grad_norm_(
                [rotation_param, translation_param], clip_grad
            )

            optimizer.step()

            loss_scalar = float(loss_value.item())
            if loss_scalar < best_local_loss:
                best_local_loss = loss_scalar
                best_local_pose = pose_current.detach().clone()

        if best_local_loss < final_best_loss:
            final_best_loss = best_local_loss
            final_best_pose = best_local_pose

    # --- Final visualization pass ---
    final_loss = eval_pose(final_best_pose, visualize=True)

    print(
        f"[Frame {i}] nuclear_rot_multistart final_loss = {float(final_loss.item()):.6f}"
    )
    return final_best_pose, float(final_loss.item())


if __name__ == "__main__":
    v = Visualizer()
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
            current_pose=poses[i],
        )
        mask = torch.load(f"pts/mask_{i:04d}.pt")
        image, depth = surface_lf.rasterize()
        if i > 0:
            pose_refined, final_loss = refine_pose(
                surface_lf_prev, image, depth, poses_rel[i], i, mask_prev, mask
            )
            pose_refined, final_loss = refine_pose_nuclear_rotation_multistart(
                surface_lf_prev,
                image,
                depth,
                pose_coarse=pose_refined,  # keep your anchor logic consistent with your codepath
                pose_init=pose_refined,
                i=i,
                mask_prev=mask_prev,
                mask=mask,
                init_loss=final_loss,
                num_samples=96,
                max_angle_deg=25.0,
                topk=5,
                refine_topk=3,
                adam_iters=25,
                lr_rotation=2e-2,
                lr_translation=0.0,
            )
            poses_refined.append(pose_refined @ poses_refined[-1])
        surface_lf_prev = surface_lf
        mask_prev = mask
    poses_refined = torch.stack(poses_refined, dim=0)
    poses_gt = torch.stack(poses_gt, dim=0)
    poses_coarse = torch.stack(poses, dim=0)

    poses_coarse = rebase_poses(poses_gt.cpu().numpy(), poses_coarse.cpu().numpy())
    # poses_refined = rebase_poses(poses_gt.cpu().numpy(), poses_refined.cpu().numpy())
    for i, (coarse_pose, pose, gt_pose) in enumerate(
        zip(
            poses_coarse,
            poses_refined.cpu().numpy(),
            poses_gt.cpu().numpy(),
        )
    ):
        v.add_frame(f"refined_{i:04d}", pose)
        # v.add_frame(f"gt_{i:04d}", gt_pose)
        v.add_frame(f"coarse_{i:04d}", coarse_pose)
    print(pose_errors(poses_gt.cpu().numpy(), poses_coarse))
    print(pose_errors(poses_gt.cpu().numpy(), poses_refined.cpu().numpy()))
    v.run()
