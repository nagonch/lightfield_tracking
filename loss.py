import itertools
import torch
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
from utils import compose_pose, matrix_to_axis_angle, rhs_to_lhs_rel
from matplotlib import pyplot as plt
from itertools import combinations
import torch
import torch.nn.functional as F

from src.slf_refinement_viewer import SurfaceLFRefinementViewer


def safe_normalize(x, eps=1e-6):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:

    inputs = d6.view(-1, 2, 3)
    a1 = inputs[:, 0]
    a2 = inputs[:, 1]

    b1 = safe_normalize(a1)

    proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = safe_normalize(a2 - proj)

    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack((b1, b2, b3), dim=-1).squeeze()


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Extracts the first two columns of a rotation matrix to form the 6D representation.
    Input: (3, 3) matrix. Output: (6,) vector.
    """
    return matrix[:3, :2].transpose(0, 1).reshape(-1)


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
    tensor_hw: torch.Tensor, path: str, gamma: float = 0.5, normalize_by=None
) -> None:
    """
    Save [H,W] tensor as 8-bit grayscale. Normalizes to [0,1] using min/max, then applies gamma.
    """
    tensor_hw = tensor_hw.detach()
    tensor_hw = tensor_hw - tensor_hw.min()
    if normalize_by is not None:
        tensor_hw = tensor_hw / (normalize_by + 1e-8)
    else:
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
    depth_lambda=0.0,
    aggregate=True,
):
    rgb_loss = (rendered_rgb - target_rgb) ** 2
    depth_loss = (rendered_depth - target_depth) ** 2
    result = rgb_loss.mean(axis=-1) + depth_lambda * depth_loss
    if aggregate:
        result = torch.mean(result)
    return result


def _save_convergence_projections(
    surface_lf_prev,
    target_rgb,
    target_depth,
    pivot_world,
    pose_history_rhs,
    loss_history,
    final_pose_rhs,
    pose_coarse_rhs,
    pose_gt_rhs=None,
    plot_filename="convergence.png",
    grid_resolution: int = 21,
):
    """
    Save 15 pairwise 2D projections of optimizer trajectory in 6D pose parameter space.
    Each subplot includes a color-coded loss landscape and the optimizer path.
    """
    if pose_history_rhs is None or len(pose_history_rhs) == 0:
        return

    param_names = ["x", "y", "z", "θ", "φ", "γ"]
    dim_pairs = list(combinations(range(6), 2))

    params_history = []
    for pose_rhs in pose_history_rhs:
        if not torch.is_tensor(pose_rhs):
            pose_rhs = torch.from_numpy(pose_rhs)
        pose_rhs = pose_rhs.float()

        trans = pose_rhs[:3, 3]
        rot = matrix_to_axis_angle(pose_rhs[:3, :3])
        params_history.append(torch.cat((trans, rot), dim=0))

    params_history = torch.stack(params_history)  # [T, 6]

    coarse_trans = pose_coarse_rhs[:3, 3]
    coarse_rot = matrix_to_axis_angle(pose_coarse_rhs[:3, :3])
    coarse_params = torch.cat((coarse_trans, coarse_rot), dim=0).float()

    final_trans = final_pose_rhs[:3, 3]
    final_rot = matrix_to_axis_angle(final_pose_rhs[:3, :3])
    final_params = torch.cat((final_trans, final_rot), dim=0).float()

    gt_params = None
    if pose_gt_rhs is not None:
        gt_trans = pose_gt_rhs[:3, 3]
        gt_rot = matrix_to_axis_angle(pose_gt_rhs[:3, :3])
        gt_params = torch.cat((gt_trans, gt_rot), dim=0).float()

    params_history_np = params_history.detach().cpu().numpy()
    coarse_params_np = coarse_params.detach().cpu().numpy()
    final_params_np = final_params.detach().cpu().numpy()
    gt_params_np = None if gt_params is None else gt_params.detach().cpu().numpy()

    fig2d, axes2d = plt.subplots(3, 5, figsize=(18, 10))
    axes2d = axes2d.flatten()

    for pair_index, (dim_a, dim_b) in enumerate(dim_pairs):
        ax = axes2d[pair_index]

        x_vals = params_history_np[:, dim_b]
        y_vals = params_history_np[:, dim_a]

        anchor_x = [
            x_vals[0],
            x_vals[-1],
            coarse_params_np[dim_b],
            final_params_np[dim_b],
        ]
        anchor_y = [
            y_vals[0],
            y_vals[-1],
            coarse_params_np[dim_a],
            final_params_np[dim_a],
        ]
        if gt_params_np is not None:
            anchor_x.append(gt_params_np[dim_b])
            anchor_y.append(gt_params_np[dim_a])

        x_min = min(anchor_x)
        x_max = max(anchor_x)
        y_min = min(anchor_y)
        y_max = max(anchor_y)

        x_range = x_max - x_min
        y_range = y_max - y_min

        x_margin = max(0.15 * x_range, 1e-4)
        y_margin = max(0.15 * y_range, 1e-4)

        x_grid = torch.linspace(
            x_min - x_margin,
            x_max + x_margin,
            grid_resolution,
            device=coarse_params.device,
            dtype=coarse_params.dtype,
        )
        y_grid = torch.linspace(
            y_min - y_margin,
            y_max + y_margin,
            grid_resolution,
            device=coarse_params.device,
            dtype=coarse_params.dtype,
        )

        loss_map = torch.zeros(
            (grid_resolution, grid_resolution),
            device=coarse_params.device,
            dtype=coarse_params.dtype,
        )

        with torch.no_grad():
            for yi, yv in enumerate(y_grid):
                for xi, xv in enumerate(x_grid):
                    params = coarse_params.clone()
                    params[dim_a] = yv
                    params[dim_b] = xv

                    pose_rel_rhs = compose_pose(params[3:], params[:3])
                    pose_rel_lhs = rhs_to_lhs_rel(pose_rel_rhs, pivot_world)
                    image_rendered, depth_rendered = surface_lf_prev.rasterize(
                        pose_rel_lhs
                    )
                    loss_val = simple_loss(
                        image_rendered,
                        target_rgb,
                        depth_rendered,
                        target_depth,
                        aggregate=False,
                    )
                    loss_map[yi, xi] = loss_val.mean()

        loss_np = loss_map.detach().cpu().numpy()
        extent = [
            x_grid[0].item(),
            x_grid[-1].item(),
            y_grid[0].item(),
            y_grid[-1].item(),
        ]
        im = ax.imshow(loss_np, origin="lower", aspect="auto", extent=extent)

        ax.plot(
            x_vals,
            y_vals,
            color="white",
            linewidth=1.2,
            alpha=0.45,
            antialiased=False,
            solid_joinstyle="miter",
            solid_capstyle="butt",
        )
        ax.scatter(
            x_vals,
            y_vals,
            color="white",
            s=8,
            alpha=0.5,
            label="steps",
            zorder=2,
        )
        ax.scatter(
            x_vals[0],
            y_vals[0],
            marker="o",
            color="black",
            s=40,
            label="start",
            zorder=3,
        )
        ax.scatter(
            final_params_np[dim_b],
            final_params_np[dim_a],
            marker="*",
            color="red",
            s=80,
            label="final",
            zorder=3,
        )

        ax.scatter(
            coarse_params_np[dim_b],
            coarse_params_np[dim_a],
            marker="+",
            color="cyan",
            s=90,
            linewidths=2.0,
            label="coarse",
            zorder=3,
        )

        if gt_params_np is not None:
            ax.scatter(
                gt_params_np[dim_b],
                gt_params_np[dim_a],
                marker="x",
                color="red",
                s=90,
                linewidths=2.0,
                label="gt",
                zorder=3,
            )

        ax.set_xlabel(param_names[dim_b])
        ax.set_ylabel(param_names[dim_a])
        ax.set_title(f"{param_names[dim_a]} vs {param_names[dim_b]}")
        fig2d.colorbar(im, ax=ax)

    plt.tight_layout()
    if plot_filename is not None:
        plt.savefig(plot_filename)
    plt.close(fig2d)

    if loss_history is not None and len(loss_history) > 0 and plot_filename is not None:
        loss_fig, loss_ax = plt.subplots(1, 1, figsize=(8, 4))
        steps = np.arange(len(loss_history))
        loss_ax.plot(steps, loss_history, color="tab:blue", linewidth=1.8)
        loss_ax.scatter(steps, loss_history, color="tab:blue", s=10, alpha=0.7)
        loss_ax.set_xlabel("step")
        loss_ax.set_ylabel("loss")
        loss_ax.set_title("Loss vs Step")
        loss_ax.grid(True, alpha=0.3)
        loss_fig.tight_layout()

        base, ext = os.path.splitext(plot_filename)
        loss_plot_filename = f"{base}_loss_vs_step{ext if ext else '.png'}"
        loss_fig.savefig(loss_plot_filename)
        plt.close(loss_fig)


def refine_pose(
    surface_lf_prev,
    surface_lf,
    pose_coarse_rhs,
    image,
    depth,
    pivot_world,
    pose_gt_rhs=None,
    num_iterations=500,
    learning_rate_rot: float = 1e-3,
    learning_rate_trans: float = 1e-3,
    convergence_plot_filename: str = None,
    loss_images_dir: str = None,
    loss_image_gamma: float = 0.5,
    rendered_images_dir: str = None,
    rendered_depth_images_dir: str = None,
    rendered_depth_gamma: float = 0.5,
    refinement_viewer: SurfaceLFRefinementViewer = None,
):
    device = pose_coarse_rhs.device

    # 1. Initialize parameters
    # Convert initial rotation matrix to 6D representation
    rot_6d = matrix_to_rotation_6d(pose_coarse_rhs[:3, :3])
    trans = pose_coarse_rhs[:3, 3].clone()

    rotation_param = (
        (rot_6d + torch.randn_like(rot_6d) * 1e-3).clone().detach().requires_grad_(True)
    )
    translation_param = (
        (trans + torch.randn_like(trans) * 1e-3).clone().detach().requires_grad_(True)
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": [rotation_param], "lr": float(learning_rate_rot)},
            {"params": [translation_param], "lr": float(learning_rate_trans)},
        ],
    )
    # Variables to track the best state
    best_loss = float("inf")
    best_pose_rhs = pose_coarse_rhs.clone().detach()

    pose_history_rhs = []
    loss_history = []

    if loss_images_dir is not None:
        os.makedirs(loss_images_dir, exist_ok=True)
    if rendered_images_dir is not None:
        os.makedirs(rendered_images_dir, exist_ok=True)
    if rendered_depth_images_dir is not None:
        os.makedirs(rendered_depth_images_dir, exist_ok=True)
    normalize_by = None
    depth_normalize_by = None
    for i in range(num_iterations):
        optimizer.zero_grad()

        # 2. Map 6D parameters back to a valid SO(3) rotation matrix
        rot_matrix = rotation_6d_to_matrix(rotation_param)

        # 3. Construct the 4x4 pose matrix
        pose_rel_rhs = torch.eye(4, device=device)
        pose_rel_rhs[:3, :3] = rot_matrix
        pose_rel_rhs[:3, 3] = translation_param

        # Append to history for visualization
        pose_history_rhs.append(pose_rel_rhs.detach().cpu().numpy())

        # 4. Transformation and Rendering
        pose_rel_lhs = rhs_to_lhs_rel(pose_rel_rhs, pivot_world)
        image_rendered, depth_rendered = surface_lf_prev.rasterize(pose_rel_lhs)
        if rendered_images_dir is not None:
            _save_rgb_image(
                image_rendered,
                os.path.join(rendered_images_dir, f"image_{i:04d}.png"),
            )

        if rendered_depth_images_dir is not None:
            if depth_normalize_by is None:
                depth_normalize_by = depth_rendered.max().item()
            _save_grayscale_image(
                depth_rendered,
                os.path.join(rendered_depth_images_dir, f"depth_{i:04d}.png"),
                gamma=rendered_depth_gamma,
                normalize_by=depth_normalize_by,
            )

        # 5. Loss calculation
        per_pixel_loss = simple_loss(
            image_rendered,
            image,
            depth_rendered,
            depth,
            aggregate=False,
        )
        loss = per_pixel_loss.mean()
        current_loss_val = loss.item()
        if normalize_by is None:
            normalize_by = per_pixel_loss.max().item()
        if loss_images_dir is not None:
            _save_grayscale_image(
                per_pixel_loss,
                os.path.join(loss_images_dir, f"loss_{i:04d}.png"),
                gamma=loss_image_gamma,
                normalize_by=normalize_by,
            )

        # Track the minimum loss state
        loss_history.append(current_loss_val)
        if current_loss_val < best_loss:
            best_loss = current_loss_val
            best_pose_rhs = pose_rel_rhs.detach().clone()

        if refinement_viewer is not None:
            transformed_values = surface_lf_prev.transform(pose_rel_lhs)
            refinement_viewer.update(
                transformed_values=transformed_values,
                loss_value=current_loss_val,
                iteration=i,
                rendered_image=image_rendered,
                target_image=image,
            )
        # 6. Optimization step
        loss.backward()
        optimizer.step()

    # Final conversion for the return values using the BEST observed pose
    final_pose_rhs_np = best_pose_rhs.cpu().numpy()
    # We must re-calculate the LHS version for the best RHS pose
    final_pose_lhs_best = rhs_to_lhs_rel(best_pose_rhs, pivot_world)
    final_pose_lhs_np = final_pose_lhs_best.detach().cpu().numpy()

    if convergence_plot_filename is not None:
        _save_convergence_projections(
            surface_lf_prev=surface_lf_prev,
            target_rgb=image,
            target_depth=depth,
            pivot_world=pivot_world,
            pose_history_rhs=pose_history_rhs,
            loss_history=loss_history,
            final_pose_rhs=best_pose_rhs,
            pose_coarse_rhs=pose_coarse_rhs,
            pose_gt_rhs=pose_gt_rhs,
            plot_filename=convergence_plot_filename,
        )
    return (
        final_pose_rhs_np,
        final_pose_lhs_np,
        pose_history_rhs,
    )


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
