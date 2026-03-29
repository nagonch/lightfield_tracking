import itertools

import torch
from PIL import Image
import numpy as np
import os
from utils import matrix_to_axis_angle


def _to_hw(tensor_hw_or_hw1: torch.Tensor) -> torch.Tensor:
    """Ensure depth/mask are [H,W]."""
    if tensor_hw_or_hw1.ndim == 3 and tensor_hw_or_hw1.shape[-1] == 1:
        return tensor_hw_or_hw1[..., 0]
    return tensor_hw_or_hw1


def _rotation_geodesic_distance(
    R_a: torch.Tensor, R_b: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    """
    Geodesic distance on SO(3) between rotation matrices.
    Returns angle in radians.
    R_a, R_b: [3,3]
    """
    R_rel = R_a.transpose(-1, -2) @ R_b
    trace = R_rel.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    cos_theta = (trace - 1.0) * 0.5
    cos_theta = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    return torch.acos(cos_theta)


def _pose_anchor_loss(pose_pred: torch.Tensor, pose_est: torch.Tensor) -> torch.Tensor:
    """
    Weak anchor: translation L2 + rotation geodesic^2.
    pose_*: [4,4]
    """
    R_pred, t_pred = pose_pred[:3, :3], pose_pred[:3, 3]
    R_est, t_est = pose_est[:3, :3], pose_est[:3, 3]

    rotation_angle = _rotation_geodesic_distance(R_pred, R_est)  # radians
    translation_error = torch.norm(t_est - t_pred, p=2)

    return rotation_angle**2 + translation_error**2


def _save_grayscale_image(
    tensor_hw: torch.Tensor, path: str, gamma: float = 0.5
) -> None:
    """
    Save [H,W] tensor as 8-bit grayscale. Normalizes to [0,1] using min/max, then applies gamma.
    """
    tensor_hw = tensor_hw.detach()
    tensor_hw = tensor_hw - tensor_hw.min()
    tensor_hw = tensor_hw / (tensor_hw.max() + 1e-8)
    tensor_hw = torch.clamp(tensor_hw, 0.0, 1.0)

    # Gamma for visualization: values <1 brighten small errors; >1 darken them.
    tensor_hw = torch.pow(tensor_hw, gamma)

    image_u8 = (tensor_hw.cpu().numpy() * 255.0).astype(np.uint8)
    Image.fromarray(image_u8, mode="L").save(path)


def _save_rgb_image(image_hw3: torch.Tensor, path: str) -> None:
    """
    Save [H,W,3] float image in [0,1] as PNG.
    """
    image_hw3 = torch.clamp(image_hw3.detach(), 0.0, 1.0)
    image_u8 = (image_hw3.cpu().numpy() * 255.0).astype(np.uint8)
    Image.fromarray(image_u8, mode="RGB").save(path)


def loss(
    image_from: torch.Tensor,
    depth_from: torch.Tensor,
    mask_from: torch.Tensor,
    image_to: torch.Tensor,
    depth_to: torch.Tensor,
    mask_to: torch.Tensor,
    pose_coarse: torch.Tensor,  # predicted pose anchor
    pose_refined: torch.Tensor,  # optimized pose
    visualize: bool = True,
    i: int = 0,
    out_dir: str = "losses",
    gamma: float = 0.5,
):
    """
    Tracking loss:
      - masked RGB L2
      - masked depth L2
      - mask loss (soft IoU)
      - weak pose anchor to pose_coarse

    No SSIM, no depth-edge alignment. No weights (just sum).
    """

    # --- shape cleanup ---
    depth_from_hw = _to_hw(depth_from)
    depth_to_hw = _to_hw(depth_to)
    mask_from_hw = _to_hw(mask_from).float()
    mask_to_hw = _to_hw(mask_to).float()

    # --- build valid region ---
    # Use intersection of masks. If object disappears / is heavily occluded, this can go tiny,
    # but that’s exactly what you described: tracking becomes ill-posed.
    valid_mask_hw = (mask_to_hw > 0.5).float() * (mask_from_hw > 0.5).float()

    # Depth valid gating (optional but usually prevents NaNs / garbage depth)
    # depth_valid_hw = (depth_from_hw > 0).float() * (depth_to_hw > 0).float()
    valid_mask_hw = valid_mask_hw  # * depth_valid_hw

    valid_den = valid_mask_hw.sum().clamp(min=1.0)

    # --- per-pixel losses ---
    rgb_residual_hw3 = image_from - image_to  # [H,W,3]
    rgb_loss_hw = (rgb_residual_hw3**2).mean(dim=-1)  # [H,W]

    depth_residual_hw = depth_from_hw - depth_to_hw  # [H,W]
    depth_loss_hw = depth_residual_hw**2  # [H,W]

    # --- mask loss (soft IoU) ---
    # Treat mask_from as soft silhouette (if it’s binary, still ok).
    intersection = (mask_from_hw * mask_to_hw).sum()
    union = (
        (mask_from_hw + mask_to_hw - mask_from_hw * mask_to_hw).sum().clamp(min=1e-6)
    )
    mask_loss = 1.0 - intersection / union

    # --- masked averages ---
    rgb_loss = (rgb_loss_hw * valid_mask_hw).sum() / valid_den
    depth_loss = (depth_loss_hw * valid_mask_hw).sum() / valid_den

    # --- weak pose anchor ---
    pose_anchor = _pose_anchor_loss(pose_coarse, pose_refined)

    total_loss = rgb_loss + depth_loss + mask_loss + pose_anchor

    # --- visualization ---
    if visualize:
        os.makedirs(out_dir, exist_ok=True)

        # Per-pixel total map: include only pixel terms (rgb+depth) for the image.
        # Mask & pose are scalars; don’t smear them over the image.
        per_pixel_total_hw = (rgb_loss_hw + depth_loss_hw) * valid_mask_hw

        _save_grayscale_image(
            per_pixel_total_hw,
            os.path.join(out_dir, f"loss_total_{i:04d}.png"),
            gamma=gamma,
        )
        _save_grayscale_image(
            rgb_loss_hw * valid_mask_hw,
            os.path.join(out_dir, f"loss_rgb_{i:04d}.png"),
            gamma=gamma,
        )
        _save_grayscale_image(
            depth_loss_hw * valid_mask_hw,
            os.path.join(out_dir, f"loss_depth_{i:04d}.png"),
            gamma=gamma,
        )

        _save_grayscale_image(
            valid_mask_hw, os.path.join(out_dir, f"valid_mask_{i:04d}.png"), gamma=1.0
        )

        _save_rgb_image(image_to, os.path.join(out_dir, f"target_image_{i:04d}.png"))
        _save_rgb_image(
            image_from, os.path.join(out_dir, f"rendered_image_{i:04d}.png")
        )

        # Also save masks for sanity (scaled to grayscale)
        _save_grayscale_image(
            mask_to_hw, os.path.join(out_dir, f"mask_to_{i:04d}.png"), gamma=1.0
        )
        _save_grayscale_image(
            mask_from_hw, os.path.join(out_dir, f"mask_from_{i:04d}.png"), gamma=1.0
        )

    # Return both total and the breakdown so you can print / log them.
    return total_loss, {
        "rgb_loss": rgb_loss,
        "depth_loss": depth_loss,
        "mask_loss": mask_loss,
        "pose_anchor": pose_anchor,
        "valid_pixels": valid_den,
    }


def simple_loss(rendered_rgb, gt_rgb, aggregate=False):
    result = (rendered_rgb - gt_rgb) ** 2
    if aggregate:
        result = torch.mean(result)
    return result


def get_neighborhood(
    gt_rel_pose,
    translation_step=0.01,
    rotation_step=0.05,
    translation_radius=2,
    rotation_radius=2,
):

    gt_trans = gt_rel_pose[:3, 3]
    gt_rot = matrix_to_axis_angle(gt_rel_pose[:3, :3])
    gt_params = torch.cat((gt_trans, gt_rot), dim=0)  # [6]

    translation_offsets = (
        torch.arange(-translation_radius, translation_radius + 1) * translation_step
    )
    rotation_offsets = (
        torch.arange(-rotation_radius, rotation_radius + 1) * rotation_step
    )

    grid_values = [
        translation_offsets,
        translation_offsets,
        translation_offsets,
        rotation_offsets,
        rotation_offsets,
        rotation_offsets,
    ]

    offset_combinations = torch.tensor(
        list(itertools.product(*grid_values)), dtype=gt_params.dtype
    ).cuda()

    neighborhood = gt_params[None, :] + offset_combinations

    return neighborhood


if __name__ == "__main__":
    pass
