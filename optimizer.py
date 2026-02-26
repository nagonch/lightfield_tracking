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


import torch
import torch.nn.functional as F


import torch
import torch.nn.functional as F


def _allocate_iterations(total_iters: int, num_levels: int):
    # Prefer finer levels.
    if num_levels == 4:
        weights = torch.tensor([1, 1, 2, 3], dtype=torch.float32)
    else:
        weights = torch.ones(num_levels, dtype=torch.float32)

    weights = weights / weights.sum()
    iters = (weights * total_iters).round().to(torch.int64).tolist()

    diff = total_iters - sum(iters)
    if diff != 0:
        iters[-1] += diff
    return iters


def _downsample_hwc_image_hw_depth_long_mask(
    image_hwc, depth_hw, mask_hw_long, target_hw
):
    """
    image_hwc: float [H,W,3]
    depth_hw:  float [H,W]
    mask_hw_long: long [H,W]
    returns: (image_hwc_ds [h,w,3], depth_hw_ds [h,w], mask_hw_ds_long [h,w])
    """
    target_h, target_w = target_hw

    image_ds = None
    if image_hwc is not None:
        if image_hwc.dim() != 3 or image_hwc.shape[-1] != 3:
            raise ValueError(f"Expected image [H,W,3], got {tuple(image_hwc.shape)}")
        # HWC -> NCHW
        image_nchw = image_hwc.permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
        image_nchw_ds = F.interpolate(
            image_nchw, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        # NCHW -> HWC
        image_ds = image_nchw_ds.squeeze(0).permute(1, 2, 0)  # [h,w,3]

    depth_ds = None
    if depth_hw is not None:
        if depth_hw.dim() != 2:
            raise ValueError(f"Expected depth [H,W], got {tuple(depth_hw.shape)}")
        depth_n1hw = depth_hw.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        depth_n1hw_ds = F.interpolate(
            depth_n1hw, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        depth_ds = depth_n1hw_ds.squeeze(0).squeeze(0)  # [h,w]

    mask_ds = None
    if mask_hw_long is not None:
        if mask_hw_long.dim() != 2:
            raise ValueError(f"Expected mask [H,W], got {tuple(mask_hw_long.shape)}")
        if mask_hw_long.dtype != torch.long:
            raise ValueError(f"Mask must be long, got {mask_hw_long.dtype}")
        mask_float = (
            mask_hw_long.to(torch.float32).unsqueeze(0).unsqueeze(0)
        )  # [1,1,H,W]
        mask_float_ds = F.interpolate(
            mask_float, size=(target_h, target_w), mode="nearest"
        )
        mask_ds = mask_float_ds.squeeze(0).squeeze(0).to(torch.long)  # [h,w]

    return image_ds, depth_ds, mask_ds


def refine_pose(
    surface_lf_prev,
    image,  # [H,W,3]
    depth,  # [H,W]
    pose_coarse,
    i,
    mask_prev,  # [H,W] long
    mask,  # [H,W] long
    num_iterations: int = 200,
    lr_translation: float = 1e-2,
    lr_rotation: float = 5e-3,
    pyramid_scales=(1 / 8, 1 / 4, 1 / 2, 1.0),
):
    device = image.device

    # Validate / enforce dtypes
    if mask_prev.dtype != torch.long:
        mask_prev = mask_prev.to(torch.long)
    if mask.dtype != torch.long:
        mask = mask.to(torch.long)

    if image.dim() != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected image [H,W,3], got {tuple(image.shape)}")
    if depth.dim() != 2:
        raise ValueError(f"Expected depth [H,W], got {tuple(depth.shape)}")
    if mask_prev.dim() != 2 or mask.dim() != 2:
        raise ValueError(
            f"Expected masks [H,W], got {tuple(mask_prev.shape)} and {tuple(mask.shape)}"
        )

    base_h, base_w, _ = image.shape
    level_iters = _allocate_iterations(num_iterations, len(pyramid_scales))

    best_pose_global = pose_coarse.clone()
    best_loss_global = float("inf")

    for level_index, (scale, iters_this_level) in enumerate(
        zip(pyramid_scales, level_iters)
    ):
        if iters_this_level <= 0:
            continue

        target_h = max(1, int(round(base_h * float(scale))))
        target_w = max(1, int(round(base_w * float(scale))))
        target_hw = (target_h, target_w)

        # Downsample observations/masks for this level
        image_level, depth_level, mask_level = _downsample_hwc_image_hw_depth_long_mask(
            image, depth, mask, target_hw
        )
        _, _, mask_prev_level = _downsample_hwc_image_hw_depth_long_mask(
            image, depth, mask_prev, target_hw
        )

        # Re-init small deltas each level (keeps the local parameterization sane)
        rotation_param = torch.zeros(3, device=device, requires_grad=True)
        translation_param = torch.zeros(3, device=device, requires_grad=True)

        optimizer = torch.optim.Adam(
            [
                {"params": translation_param, "lr": lr_translation},
                {"params": rotation_param, "lr": lr_rotation},
            ]
        )

        best_loss_level = float("inf")
        best_pose_level = best_pose_global.clone()

        # Anchor per level: refine around the best from previous level
        pose_anchor = best_pose_global.detach().clone()

        for iteration in range(iters_this_level):
            optimizer.zero_grad(set_to_none=True)

            pose_delta = compose_pose(rotation_param, translation_param)
            pose_current = pose_anchor @ pose_delta  # right update

            # Render at full res, then downsample rendered outputs to this level
            surf_values_full = surface_lf_prev.transform(pose_current)
            surf_image_full, surf_depth_full = surface_lf_prev.rasterize(
                surf_values_full
            )

            # Expect renderer outputs to match your loss input conventions.
            # If surf_image_full is [H,W,3] and surf_depth_full is [H,W], this just works.
            surf_image_level, surf_depth_level, _ = (
                _downsample_hwc_image_hw_depth_long_mask(
                    surf_image_full, surf_depth_full, None, target_hw
                )
            )

            loss_value, _ = loss(
                surf_image_level,
                surf_depth_level,
                mask_prev_level,  # long
                image_level,  # [h,w,3]
                depth_level,  # [h,w]
                mask_level,  # long
                pose_anchor,  # anchor for this level
                pose_current,  # refined
                visualize=False,
                i=0,
            )

            loss_value.backward()
            torch.nn.utils.clip_grad_norm_([rotation_param, translation_param], 10.0)
            optimizer.step()

            loss_scalar = float(loss_value.detach().item())
            if loss_scalar < best_loss_level:
                best_loss_level = loss_scalar
                best_pose_level = pose_current.detach().clone()

        best_pose_global = best_pose_level
        if best_loss_level < best_loss_global:
            best_loss_global = best_loss_level

    # Final eval at full res
    surf_values = surface_lf_prev.transform(best_pose_global)
    surf_image, surf_depth = surface_lf_prev.rasterize(surf_values)

    final_loss, breakdown = loss(
        surf_image,
        surf_depth,
        mask_prev,  # [H,W] long
        image,  # [H,W,3]
        depth,  # [H,W]
        mask,  # [H,W] long
        pose_coarse,
        best_pose_global,
        visualize=False,
    )

    print(f"[Frame {i}] final_loss = {final_loss:.6f}")
    return best_pose_global, final_loss


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
