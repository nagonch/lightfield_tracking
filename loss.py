import itertools
import torch
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from utils import compose_pose, matrix_to_axis_angle, rhs_to_lhs_rel
from matplotlib import pyplot as plt
from itertools import combinations


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


def simple_loss(
    rendered_rgb,
    target_rgb,
    rendered_depth,
    target_depth,
    depth_lambda=1.0,
    aggregate=True,
):
    rgb_loss = (rendered_rgb - target_rgb) ** 2
    depth_loss = (rendered_depth - target_depth) ** 2
    result = rgb_loss.mean(axis=-1) + depth_lambda * depth_loss
    if aggregate:
        result = torch.mean(result)
    return result


def get_neighborhood(
    rel_pose_rhs,
    translation_step=5e-4 * 10,
    rotation_step=1e-3 * 10,
    translation_radius=10,
    rotation_radius=10,
):
    trans = rel_pose_rhs[:3, 3]
    rot = matrix_to_axis_angle(rel_pose_rhs[:3, :3])
    orig_params = torch.cat((trans, rot), dim=0)  # [6]

    translation_offsets = (
        torch.arange(-translation_radius, translation_radius + 1) * translation_step
    )

    rotation_offsets = (
        torch.arange(-rotation_radius, rotation_radius + 1) * rotation_step
    )

    offsets_per_dim = [
        translation_offsets,
        translation_offsets,
        translation_offsets,
        rotation_offsets,
        rotation_offsets,
        rotation_offsets,
    ]

    dim_pairs = list(combinations(range(6), 2))

    neighborhoods = []

    for dim_a, dim_b in dim_pairs:

        offsets_a = offsets_per_dim[dim_a]
        offsets_b = offsets_per_dim[dim_b]

        grid_a, grid_b = torch.meshgrid(offsets_a, offsets_b, indexing="ij")

        num_points = grid_a.numel()

        params = orig_params.repeat(num_points, 1)

        params[:, dim_a] += grid_a.reshape(-1).cuda()
        params[:, dim_b] += grid_b.reshape(-1).cuda()

        neighborhoods.append(params.reshape(grid_a.shape[0], grid_a.shape[1], 6))

    neighborhood = torch.stack(neighborhoods).cuda()  # [15, N, M, 6]

    return neighborhood, orig_params


def probe_neighbourhood(
    surface_lv_prev,
    target_rgb,
    target_depth,
    neighborhood,
    orig_params,
    current_abs_pose,
    gt_pose_rhs,
    stride=1,
    plot_filename="plot.png",
    plot_1d_only=False,
):

    param_names = ["x", "y", "z", "θ", "φ", "γ"]
    dim_pairs = list(combinations(range(6), 2))
    gt_trans = gt_pose_rhs[:3, 3]
    gt_rot = matrix_to_axis_angle(gt_pose_rhs[:3, :3])
    gt_params = torch.cat((gt_trans, gt_rot), dim=0)  # [6]

    gamma_value = 0.5
    loss_values_all = []

    # -------------------------
    # 2D plots
    # -------------------------
    if not plot_1d_only:
        fig2d, axes2d = plt.subplots(3, 5, figsize=(18, 10))
        axes2d = axes2d.flatten()

    # storage for 1D slices
    one_d_losses = [[] for _ in range(6)]
    one_d_params = [[] for _ in range(6)]

    total_evals = 0
    for pair_index in range(len(dim_pairs)):
        grid = neighborhood[pair_index][::stride, ::stride]
        total_evals += grid.shape[0] * grid.shape[1]

    pbar = tqdm(total=total_evals, desc="Evaluating loss landscape")

    for pair_index, (dim_a, dim_b) in enumerate(dim_pairs):

        params_grid = neighborhood[pair_index][::stride, ::stride]
        losses = []

        for i in range(params_grid.shape[0]):
            row_losses = []
            for j in range(params_grid.shape[1]):
                params = params_grid[i, j]
                pose_rel = compose_pose(params[3:], params[:3])
                pose_rel_lhs = rhs_to_lhs_rel(pose_rel, current_abs_pose)

                image, depth = surface_lv_prev.rasterize(pose_rel_lhs)

                loss_val = simple_loss(
                    image, target_rgb, depth, target_depth, aggregate=False
                )

                row_losses.append(loss_val.mean())
                pbar.update(1)

            losses.append(torch.stack(row_losses))

        losses = torch.stack(losses)
        loss_values_all.append(losses)

        # -------------------------
        # extract 1D slices
        # -------------------------
        center_i = losses.shape[0] // 2
        center_j = losses.shape[1] // 2

        one_d_losses[dim_a].append(losses[:, center_j])
        one_d_params[dim_a].append(params_grid[:, center_j, dim_a])

        one_d_losses[dim_b].append(losses[center_i, :])
        one_d_params[dim_b].append(params_grid[center_i, :, dim_b])

        # -------------------------
        # 2D plot
        # -------------------------
        if not plot_1d_only:
            ax = axes2d[pair_index]
            loss_np = losses.cpu().numpy()

            x_vals = params_grid[0, :, dim_b].detach().cpu().numpy()
            y_vals = params_grid[:, 0, dim_a].detach().cpu().numpy()
            extent = [x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()]

            im = ax.imshow(loss_np, origin="lower", aspect="auto", extent=extent)

            gt_x = (
                orig_params[dim_b].item()
                if torch.is_tensor(orig_params)
                else orig_params[dim_b]
            )
            gt_y = (
                orig_params[dim_a].item()
                if torch.is_tensor(orig_params)
                else orig_params[dim_a]
            )

            ax.axvline(gt_x, linestyle="--", linewidth=1.5, color="black")
            ax.axhline(gt_y, linestyle="--", linewidth=1.5, color="black")

            gt_x_val = gt_params[dim_b].item()
            gt_y_val = gt_params[dim_a].item()
            ax.scatter(
                gt_x_val, gt_y_val, marker="x", color="red", s=80, label="gt_params"
            )

            ax.set_xlabel(param_names[dim_b])
            ax.set_ylabel(param_names[dim_a])
            ax.set_title(f"{param_names[dim_a]} vs {param_names[dim_b]}")
            ax.legend()
            fig2d.colorbar(im, ax=ax)

    pbar.close()

    if not plot_1d_only:
        plt.tight_layout()
        if plot_filename is not None:
            plt.savefig(plot_filename.replace(".png", "_2d.png"))
        plt.close()

    # -------------------------
    # 1D plots
    # -------------------------
    fig1d, axes1d = plt.subplots(2, 3, figsize=(12, 6))
    axes1d = axes1d.flatten()

    for dim in range(6):
        if len(one_d_losses[dim]) == 0:
            continue

        losses = torch.cat(one_d_losses[dim])
        params = torch.cat(one_d_params[dim])

        sorted_indices = torch.argsort(params)
        params = params[sorted_indices]
        losses = losses[sorted_indices]

        params_np = params.detach().cpu().numpy()
        loss_np = losses.detach().cpu().numpy()

        ax = axes1d[dim]
        ax.plot(params_np, loss_np, label="loss slice")

        gt_x = (
            orig_params[dim].item()
            if torch.is_tensor(orig_params)
            else orig_params[dim]
        )
        ax.axvline(
            gt_x, linestyle="--", linewidth=1.5, color="black", label="orig_params"
        )

        gt_param_val = gt_params[dim].item()
        ax.axvline(
            gt_param_val, linestyle=":", linewidth=1.5, color="red", label="gt_params"
        )

        ax.set_xlabel(param_names[dim])
        ax.set_ylabel("loss")
        ax.set_title(param_names[dim])
        ax.legend()

    plt.tight_layout()
    if plot_filename is not None:
        plt.savefig(plot_filename.replace(".png", "_1d.png"))
    plt.close()

    return torch.stack(loss_values_all)


if __name__ == "__main__":
    pass
