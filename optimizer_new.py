from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Literal

import torch


# ---------------------------
# Types
# ---------------------------

LossFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]
# loss_fn(pred_image, pred_depth, tgt_image, tgt_depth, mask) -> scalar loss


@dataclass
class OptimizeResult:
    pose_rel_optimized: torch.Tensor
    final_loss: float
    info: Dict[str, object]


# ---------------------------
# Pose utilities (SE(3) from axis-angle + translation)
# ---------------------------


def _skew(vec3: torch.Tensor) -> torch.Tensor:
    """vec3: [3] -> [3,3] skew-symmetric matrix"""
    x, y, z = vec3[0], vec3[1], vec3[2]
    return torch.stack(
        [
            torch.stack([torch.zeros_like(x), -z, y]),
            torch.stack([z, torch.zeros_like(x), -x]),
            torch.stack([-y, x, torch.zeros_like(x)]),
        ],
        dim=0,
    )


def _so3_exp(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Exponential map for SO(3).
    axis_angle: [3]
    returns R: [3,3]
    """
    theta = torch.linalg.norm(axis_angle) + 1e-12
    axis = axis_angle / theta
    K = _skew(axis)
    I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)

    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    R = I + sin_t * K + (1.0 - cos_t) * (K @ K)
    return R


def se3_from_axis_angle_translation(
    axis_angle: torch.Tensor, translation: torch.Tensor
) -> torch.Tensor:
    """
    axis_angle: [3], translation: [3] -> T: [4,4]
    """
    R = _so3_exp(axis_angle)
    T = torch.eye(4, device=axis_angle.device, dtype=axis_angle.dtype)
    T[:3, :3] = R
    T[:3, 3] = translation
    return T


# ---------------------------
# Rendering hook
# ---------------------------


def _render(
    surface_lf_prev, pose_rel: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Required: pred_image, pred_depth = surface_lf_prev.render(pose_rel)
    """
    if not hasattr(surface_lf_prev, "render"):
        raise AttributeError(
            "surface_lf_prev must implement .render(pose_rel) -> (image, depth)"
        )
    return surface_lf_prev.render(pose_rel)


# ---------------------------
# Main API
# ---------------------------


def optimize(
    surface_lf_prev,
    pose_rel_init: torch.Tensor,
    image: torch.Tensor,
    depth: torch.Tensor,
    mask: torch.Tensor,
    *,
    loss_fn: LossFn,
    method: Literal["adam", "nuclear"] = "adam",
    # adam params
    steps: int = 60,
    lr_rot: float = 3e-3,
    lr_trans: float = 3e-3,
    grad_clip: float = 10.0,
    # nuclear multistart params (rotation-only sampling)
    num_samples: int = 96,
    max_angle_deg: float = 25.0,
    topk: int = 8,
    refine_topk: int = 3,
    refine_steps: int = 25,
    refine_lr_rot: float = 2e-2,
    refine_lr_trans: float = 0.0,
) -> OptimizeResult:
    """
    Frame-to-frame relative pose optimization.

    Args:
      surface_lf_prev: previous-frame surface (must support .render(pose_rel))
      pose_rel_init: initial relative pose guess (4x4)
      image/depth/mask: target observations
      loss_fn: callable(pred_image, pred_depth, tgt_image, tgt_depth, mask)->scalar
      method:
        - "adam": gradient refinement from pose_rel_init
        - "nuclear": rotation multistart + optional local adam refine

    Returns:
      OptimizeResult with pose_rel_optimized (4x4) and final_loss.
    """
    if method == "adam":
        return _optimize_adam(
            surface_lf_prev,
            pose_rel_init,
            image,
            depth,
            mask,
            loss_fn=loss_fn,
            steps=steps,
            lr_rot=lr_rot,
            lr_trans=lr_trans,
            grad_clip=grad_clip,
        )

    if method == "nuclear":
        return _optimize_nuclear(
            surface_lf_prev,
            pose_rel_init,
            image,
            depth,
            mask,
            loss_fn=loss_fn,
            num_samples=num_samples,
            max_angle_deg=max_angle_deg,
            topk=topk,
            refine_topk=refine_topk,
            refine_steps=refine_steps,
            refine_lr_rot=refine_lr_rot,
            refine_lr_trans=refine_lr_trans,
            grad_clip=grad_clip,
        )

    raise ValueError(f"Unknown method: {method}")


# ---------------------------
# Adam (gradient) backend
# ---------------------------


def _optimize_adam(
    surface_lf_prev,
    pose_rel_init: torch.Tensor,
    image: torch.Tensor,
    depth: torch.Tensor,
    mask: torch.Tensor,
    *,
    loss_fn: LossFn,
    steps: int,
    lr_rot: float,
    lr_trans: float,
    grad_clip: float,
) -> OptimizeResult:
    device = pose_rel_init.device
    dtype = pose_rel_init.dtype

    pose_rel_anchor = pose_rel_init.detach()

    rot_param = torch.zeros(3, device=device, dtype=dtype, requires_grad=True)
    trans_param = torch.zeros(3, device=device, dtype=dtype, requires_grad=True)

    optimizer = torch.optim.Adam(
        [
            {"params": [rot_param], "lr": float(lr_rot)},
            {"params": [trans_param], "lr": float(lr_trans)},
        ]
    )

    best_pose = pose_rel_anchor.clone()
    best_loss = float("inf")

    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)

        pose_delta = se3_from_axis_angle_translation(rot_param, trans_param)
        pose_rel = pose_rel_anchor @ pose_delta  # right-update, relative-only

        pred_image, pred_depth = _render(surface_lf_prev, pose_rel)
        loss_val = loss_fn(pred_image, pred_depth, image, depth, mask)

        loss_val.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([rot_param, trans_param], float(grad_clip))

        optimizer.step()

        loss_scalar = float(loss_val.detach().cpu())
        if loss_scalar < best_loss:
            best_loss = loss_scalar
            best_pose = pose_rel.detach().clone()

    return OptimizeResult(
        pose_rel_optimized=best_pose,
        final_loss=best_loss,
        info={
            "method": "adam",
            "steps": int(steps),
            "lr_rot": lr_rot,
            "lr_trans": lr_trans,
        },
    )


# ---------------------------
# "Nuclear" backend (multistart rotations)
# ---------------------------


@torch.no_grad()
def _sample_axis_angle_uniform(
    num_samples: int, max_angle_rad: float, device, dtype
) -> torch.Tensor:
    axis = torch.randn(num_samples, 3, device=device, dtype=dtype)
    axis = axis / (torch.linalg.norm(axis, dim=-1, keepdim=True) + 1e-9)
    angles = (
        2.0 * torch.rand(num_samples, device=device, dtype=dtype) - 1.0
    ) * max_angle_rad
    return axis * angles[:, None]  # [N,3]


def _optimize_nuclear(
    surface_lf_prev,
    pose_rel_init: torch.Tensor,
    image: torch.Tensor,
    depth: torch.Tensor,
    mask: torch.Tensor,
    *,
    loss_fn: LossFn,
    num_samples: int,
    max_angle_deg: float,
    topk: int,
    refine_topk: int,
    refine_steps: int,
    refine_lr_rot: float,
    refine_lr_trans: float,
    grad_clip: float,
) -> OptimizeResult:
    device = pose_rel_init.device
    dtype = pose_rel_init.dtype

    pose_center = pose_rel_init.detach()

    def eval_pose(pose_rel: torch.Tensor) -> float:
        pred_image, pred_depth = _render(surface_lf_prev, pose_rel)
        loss_val = loss_fn(pred_image, pred_depth, image, depth, mask)
        return float(loss_val.detach().cpu())

    # 1) Coarse sampling around init (rotation-only)
    candidates: list[tuple[float, torch.Tensor]] = []
    candidates.append((eval_pose(pose_center), pose_center))

    max_angle_rad = math.radians(float(max_angle_deg))
    axis_angles = _sample_axis_angle_uniform(
        int(num_samples), max_angle_rad, device, dtype
    )

    for k in range(int(num_samples)):
        pose_delta = se3_from_axis_angle_translation(
            axis_angles[k], torch.zeros(3, device=device, dtype=dtype)
        )
        pose_candidate = pose_center @ pose_delta
        candidates.append((eval_pose(pose_candidate), pose_candidate))

    candidates.sort(key=lambda x: x[0])
    seeds = candidates[: max(1, int(topk))]

    best_loss, best_pose = seeds[0][0], seeds[0][1].detach().clone()

    # 2) Optional local refine from best seeds (Adam)
    for seed_idx in range(min(int(refine_topk), len(seeds))):
        _, seed_pose = seeds[seed_idx]
        local = _optimize_local_around_seed(
            surface_lf_prev=surface_lf_prev,
            seed_pose=seed_pose.detach(),
            image=image,
            depth=depth,
            mask=mask,
            loss_fn=loss_fn,
            steps=refine_steps,
            lr_rot=refine_lr_rot,
            lr_trans=refine_lr_trans,
            grad_clip=grad_clip,
        )
        if local.final_loss < best_loss:
            best_loss = local.final_loss
            best_pose = local.pose_rel_optimized.detach().clone()

    return OptimizeResult(
        pose_rel_optimized=best_pose,
        final_loss=best_loss,
        info={
            "method": "nuclear",
            "num_samples": int(num_samples),
            "max_angle_deg": float(max_angle_deg),
            "topk": int(topk),
            "refine_topk": int(refine_topk),
            "refine_steps": int(refine_steps),
        },
    )


def _optimize_local_around_seed(
    *,
    surface_lf_prev,
    seed_pose: torch.Tensor,
    image: torch.Tensor,
    depth: torch.Tensor,
    mask: torch.Tensor,
    loss_fn: LossFn,
    steps: int,
    lr_rot: float,
    lr_trans: float,
    grad_clip: float,
) -> OptimizeResult:
    device = seed_pose.device
    dtype = seed_pose.dtype

    rot_param = torch.zeros(3, device=device, dtype=dtype, requires_grad=True)
    trans_param = torch.zeros(3, device=device, dtype=dtype, requires_grad=True)

    param_groups = [{"params": [rot_param], "lr": float(lr_rot)}]
    if lr_trans and lr_trans > 0:
        param_groups.append({"params": [trans_param], "lr": float(lr_trans)})

    optimizer = torch.optim.Adam(param_groups)

    best_pose = seed_pose.detach().clone()
    best_loss = float("inf")

    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)

        pose_delta = se3_from_axis_angle_translation(rot_param, trans_param)
        pose_rel = seed_pose @ pose_delta

        pred_image, pred_depth = _render(surface_lf_prev, pose_rel)
        loss_val = loss_fn(pred_image, pred_depth, image, depth, mask)

        loss_val.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([rot_param, trans_param], float(grad_clip))

        optimizer.step()

        loss_scalar = float(loss_val.detach().cpu())
        if loss_scalar < best_loss:
            best_loss = loss_scalar
            best_pose = pose_rel.detach().clone()

    return OptimizeResult(
        pose_rel_optimized=best_pose,
        final_loss=best_loss,
        info={
            "method": "local_adam",
            "steps": int(steps),
            "lr_rot": lr_rot,
            "lr_trans": lr_trans,
        },
    )


def _normalize_mask_hw(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Returns boolean mask with shape [H, W].
    Accepts mask shapes: [H,W], [H,W,1], [1,H,W], [H,W,C] (any C>1 treated as any-channel).
    """
    if mask.ndim == 2:
        mask_hw = mask
    elif mask.ndim == 3:
        if mask.shape[-1] == 1:  # [H,W,1]
            mask_hw = mask[..., 0]
        elif mask.shape[0] == 1:  # [1,H,W]
            mask_hw = mask[0]
        else:  # [H,W,C]
            # treat as "valid if any channel is valid"
            mask_hw = (
                mask.any(dim=-1)
                if mask.dtype == torch.bool
                else (mask > 0.5).any(dim=-1)
            )
    else:
        raise ValueError(f"Unsupported mask shape: {tuple(mask.shape)}")

    if mask_hw.shape != (height, width):
        raise ValueError(
            f"Mask shape {tuple(mask_hw.shape)} doesn't match H,W={(height,width)}"
        )

    return mask_hw if mask_hw.dtype == torch.bool else (mask_hw > 0.5)


def masked_l1(
    pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    pred/tgt: [H,W] or [H,W,C]
    mask: flexible, normalized internally to [H,W]
    """
    if pred.shape != tgt.shape:
        raise ValueError(
            f"pred shape {tuple(pred.shape)} != tgt shape {tuple(tgt.shape)}"
        )

    h, w = pred.shape[0], pred.shape[1]
    mask_hw = _normalize_mask_hw(mask, h, w)

    diff = (pred - tgt).abs()
    if diff.ndim == 3:
        # broadcast [H,W] -> [H,W,1] for multiplication, but for boolean indexing expand to [H,W,C]
        mask_b = mask_hw.unsqueeze(-1).expand_as(diff)
        diff_masked = diff[mask_b]
    else:
        diff_masked = diff[mask_hw]
    return diff_masked.mean() if diff_masked.numel() > 0 else diff.sum() * 0.0


def loss_fn(pred_image, pred_depth, tgt_image, tgt_depth, mask):
    # photometric + depth, tune weights as needed
    photo = masked_l1(pred_image, tgt_image, mask)
    dep = masked_l1(pred_depth, tgt_depth, mask)
    return photo + 0.2 * dep


def summarize_tensor(name: str, tensor: torch.Tensor):
    print(
        f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device} "
        f"min={tensor.min().item():.4g} max={tensor.max().item():.4g}"
    )


if __name__ == "__main__":
    surf_debug = torch.load("surf_debug.pt", weights_only=False)

    pose_rel_init = torch.tensor(surf_debug["pose_rel"]).cuda()
    image_tgt = surf_debug["image"]
    depth_tgt = surf_debug["depth"]
    mask_tgt = surf_debug["mask"]
    surface_lf_prev = surf_debug["surface_lf"]

    print("Loaded surface_lf_prev:", surface_lf_prev)

    # Put everything on one device
    device = image_tgt.device
    pose_rel_init = pose_rel_init.to(device)
    depth_tgt = depth_tgt.to(device)
    mask_tgt = mask_tgt.to(device)

    summarize_tensor("pose_rel_init", pose_rel_init)
    summarize_tensor("image_tgt", image_tgt)
    summarize_tensor("depth_tgt", depth_tgt)
    summarize_tensor("mask_tgt", mask_tgt)

    # --- Baseline loss (init pose) ---
    with torch.no_grad():
        pred_image_init, pred_depth_init = surface_lf_prev.render(pose_rel_init)
        baseline_loss = float(
            loss_fn(
                pred_image_init, pred_depth_init, image_tgt, depth_tgt, mask_tgt
            ).cpu()
        )
    print(f"\nBaseline loss @ init pose_rel: {baseline_loss:.6f}")

    # --- ADAM refine ---
    result_adam = optimize(
        surface_lf_prev,
        pose_rel_init,
        image_tgt,
        depth_tgt,
        mask_tgt,
        loss_fn=loss_fn,
        method="adam",
        steps=80,
        lr_rot=3e-3,
        lr_trans=3e-3,
        grad_clip=10.0,
    )
    print("\n[ADAM] final_loss:", result_adam.final_loss)
    print("[ADAM] info:", result_adam.info)

    with torch.no_grad():
        pred_image_adam, pred_depth_adam = surface_lf_prev.render(
            result_adam.pose_rel_optimized
        )
        adam_check_loss = float(
            loss_fn(
                pred_image_adam, pred_depth_adam, image_tgt, depth_tgt, mask_tgt
            ).cpu()
        )
    print(f"[ADAM] recomputed loss: {adam_check_loss:.6f}")

    # --- Nuclear multistart refine ---
    result_nuclear = optimize(
        surface_lf_prev,
        pose_rel_init,
        image_tgt,
        depth_tgt,
        mask_tgt,
        loss_fn=loss_fn,
        method="nuclear",
        num_samples=128,
        max_angle_deg=25.0,
        topk=10,
        refine_topk=3,
        refine_steps=30,
        refine_lr_rot=2e-2,
        refine_lr_trans=0.0,  # keep 0 if you want rotation-only refine
        grad_clip=10.0,
    )
    print("\n[NUCLEAR] final_loss:", result_nuclear.final_loss)
    print("[NUCLEAR] info:", result_nuclear.info)

    with torch.no_grad():
        pred_image_nuc, pred_depth_nuc = surface_lf_prev.render(
            result_nuclear.pose_rel_optimized
        )
        nuclear_check_loss = float(
            loss_fn(
                pred_image_nuc, pred_depth_nuc, image_tgt, depth_tgt, mask_tgt
            ).cpu()
        )
    print(f"[NUCLEAR] recomputed loss: {nuclear_check_loss:.6f}")

    # --- Quick sanity verdict ---
    print("\nSanity:")
    print("  baseline:", baseline_loss)
    print("  adam    :", adam_check_loss)
    print("  nuclear :", nuclear_check_loss)
    if min(adam_check_loss, nuclear_check_loss) < baseline_loss:
        print("  OK: at least one optimizer improved the loss.")
    else:
        print(
            "  WARNING: no improvement — likely loss is wrong, render isn't differentiable, or pose convention mismatch."
        )
