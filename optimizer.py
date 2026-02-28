import torch
import numpy as np
import os
from loss import loss
from utils import compose_pose
import matplotlib.pyplot as plt
from PIL import Image


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


def _to_hw(tensor_hw_or_hw1: torch.Tensor) -> torch.Tensor:
    if tensor_hw_or_hw1.ndim == 3 and tensor_hw_or_hw1.shape[-1] == 1:
        return tensor_hw_or_hw1[..., 0]
    return tensor_hw_or_hw1


def _compute_pixel_loss_map(
    rendered_image: torch.Tensor,
    rendered_depth: torch.Tensor,
    mask_prev: torch.Tensor,
    target_image: torch.Tensor,
    target_depth: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    depth_from_hw = _to_hw(rendered_depth)
    depth_to_hw = _to_hw(target_depth)
    mask_from_hw = _to_hw(mask_prev).float()
    mask_to_hw = _to_hw(target_mask).float()
    valid_mask_hw = (mask_to_hw > 0.5).float() * (mask_from_hw > 0.5).float()

    rgb_residual_hw3 = rendered_image - target_image
    rgb_loss_hw = (rgb_residual_hw3**2).mean(dim=-1)
    depth_residual_hw = depth_from_hw - depth_to_hw
    depth_loss_hw = depth_residual_hw**2
    return (rgb_loss_hw + depth_loss_hw) * valid_mask_hw


def _save_grayscale_fixed(
    tensor_hw: torch.Tensor, path: str, fixed_max: float, gamma: float = 0.5
) -> None:
    value = torch.clamp(tensor_hw.detach(), min=0.0)
    denom = max(float(fixed_max), 1e-12)
    value = torch.clamp(value / denom, 0.0, 1.0)
    value = torch.pow(value, gamma)
    image_u8 = (value.cpu().numpy() * 255.0).astype(np.uint8)
    Image.fromarray(image_u8, mode="L").save(path)


def _plot_curve(values, title, y_label, path):
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(len(values)), values, linewidth=1.8)
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel(y_label)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _safe_normalize(values):
    if len(values) == 0:
        return values
    base = float(values[0])
    if abs(base) < 1e-12:
        base = 1e-12
    return [float(v) / base for v in values]


def _finite_difference_grad_map(
    surface_lf_prev,
    compose_pose_fn,
    pose_coarse: torch.Tensor,
    pivot_world: torch.Tensor,
    rotation_param: torch.Tensor,
    translation_param: torch.Tensor,
    image: torch.Tensor,
    depth: torch.Tensor,
    mask: torch.Tensor,
    mask_prev: torch.Tensor,
    parameter_type: str,
    epsilon: float,
) -> torch.Tensor:
    grad_sq = None
    for dim in range(3):
        rot_plus = rotation_param.detach().clone()
        rot_minus = rotation_param.detach().clone()
        trans_plus = translation_param.detach().clone()
        trans_minus = translation_param.detach().clone()
        if parameter_type == "rotation":
            rot_plus[dim] += epsilon
            rot_minus[dim] -= epsilon
        else:
            trans_plus[dim] += epsilon
            trans_minus[dim] -= epsilon

        pose_plus = (
            pose_delta_about_pivot(compose_pose_fn, rot_plus, trans_plus, pivot_world)
            @ pose_coarse
        )
        pose_minus = (
            pose_delta_about_pivot(compose_pose_fn, rot_minus, trans_minus, pivot_world)
            @ pose_coarse
        )

        image_plus, depth_plus = surface_lf_prev.rasterize(pose_plus)
        image_minus, depth_minus = surface_lf_prev.rasterize(pose_minus)

        loss_plus = _compute_pixel_loss_map(
            image_plus, depth_plus, mask_prev, image, depth, mask
        )
        loss_minus = _compute_pixel_loss_map(
            image_minus, depth_minus, mask_prev, image, depth, mask
        )
        grad_dim = (loss_plus - loss_minus) / (2.0 * epsilon)
        grad_sq = grad_dim**2 if grad_sq is None else grad_sq + grad_dim**2
    return torch.sqrt(torch.clamp(grad_sq, min=0.0))


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
    diagnostics_dir: str = "results/optimizer_diagnostics",
    save_step_images: bool = True,
    save_step_gradient_images: bool = True,
    finite_diff_eps: float = 1e-4,
    image_gamma: float = 0.5,
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

    optimizer = torch.optim.AdamW(
        [
            {"params": [rotation_param], "lr": float(learning_rate_rot)},
            {"params": [translation_param], "lr": float(learning_rate_trans)},
        ]
    )
    pivot_world = pivot_world.to(device)
    best_loss_value = float("inf")
    best_pose = pose_coarse.detach().clone()
    os.makedirs(diagnostics_dir, exist_ok=True)
    loss_images_dir = os.path.join(diagnostics_dir, "loss_images")
    grad_rot_dir = os.path.join(diagnostics_dir, "grad_rotation_images")
    grad_trans_dir = os.path.join(diagnostics_dir, "grad_translation_images")
    os.makedirs(loss_images_dir, exist_ok=True)
    if save_step_gradient_images:
        os.makedirs(grad_rot_dir, exist_ok=True)
        os.makedirs(grad_trans_dir, exist_ok=True)

    total_losses = []
    pose_delta_rot_norms = []
    pose_delta_trans_norms = []
    rotation_param_norms = []
    translation_param_norms = []
    rotation_grad_norms = []
    translation_grad_norms = []
    loss_img_norm_const = None
    grad_rot_norm_const = None
    grad_trans_norm_const = None

    for i in range(int(num_iterations)):
        optimizer.zero_grad(set_to_none=True)

        pose_delta = pose_delta_about_pivot(
            compose_pose_fn, rotation_param, translation_param, pivot_world
        )
        pose_current = pose_delta @ pose_coarse  # LHS update, but pivot-conditioned

        surf_image, surf_depth = surface_lf_prev.rasterize(pose_current)
        pixel_loss_map = _compute_pixel_loss_map(
            surf_image, surf_depth, mask_prev, image, depth, mask
        )

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
            i,
        )

        loss_value.backward()

        if grad_clip_norm is not None and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [rotation_param, translation_param], float(grad_clip_norm)
            )

        optimizer.step()

        loss_scalar = float(loss_value.detach().cpu().item())
        total_losses.append(loss_scalar)
        rotation_param_norms.append(
            float(torch.norm(rotation_param.detach(), p=2).item())
        )
        translation_param_norms.append(
            float(torch.norm(translation_param.detach(), p=2).item())
        )
        rotation_grad_norms.append(
            float(torch.norm(rotation_param.grad.detach(), p=2).item())
            if rotation_param.grad is not None
            else 0.0
        )
        translation_grad_norms.append(
            float(torch.norm(translation_param.grad.detach(), p=2).item())
            if translation_param.grad is not None
            else 0.0
        )

        delta_rotation_trace = torch.clamp(
            (torch.trace(pose_delta[:3, :3]) - 1.0) * 0.5,
            min=-1.0 + 1e-7,
            max=1.0 - 1e-7,
        )
        delta_rotation_angle = torch.acos(delta_rotation_trace)
        pose_delta_rot_norms.append(float(delta_rotation_angle.detach().cpu().item()))
        pose_delta_trans_norms.append(
            float(torch.norm(pose_delta[:3, 3].detach(), p=2).cpu().item())
        )

        if save_step_images:
            if loss_img_norm_const is None:
                loss_img_norm_const = float(torch.max(pixel_loss_map.detach()).item())
                if loss_img_norm_const <= 1e-12:
                    loss_img_norm_const = 1.0
            _save_grayscale_fixed(
                pixel_loss_map,
                os.path.join(loss_images_dir, f"loss_total_{i:04d}.png"),
                fixed_max=loss_img_norm_const,
                gamma=image_gamma,
            )

        if save_step_gradient_images:
            with torch.no_grad():
                grad_map_rot = _finite_difference_grad_map(
                    surface_lf_prev=surface_lf_prev,
                    compose_pose_fn=compose_pose_fn,
                    pose_coarse=pose_coarse,
                    pivot_world=pivot_world,
                    rotation_param=rotation_param,
                    translation_param=translation_param,
                    image=image,
                    depth=depth,
                    mask=mask,
                    mask_prev=mask_prev,
                    parameter_type="rotation",
                    epsilon=float(finite_diff_eps),
                )
                grad_map_trans = _finite_difference_grad_map(
                    surface_lf_prev=surface_lf_prev,
                    compose_pose_fn=compose_pose_fn,
                    pose_coarse=pose_coarse,
                    pivot_world=pivot_world,
                    rotation_param=rotation_param,
                    translation_param=translation_param,
                    image=image,
                    depth=depth,
                    mask=mask,
                    mask_prev=mask_prev,
                    parameter_type="translation",
                    epsilon=float(finite_diff_eps),
                )
            if grad_rot_norm_const is None:
                grad_rot_norm_const = float(torch.max(grad_map_rot.detach()).item())
                if grad_rot_norm_const <= 1e-12:
                    grad_rot_norm_const = 1.0
            if grad_trans_norm_const is None:
                grad_trans_norm_const = float(torch.max(grad_map_trans.detach()).item())
                if grad_trans_norm_const <= 1e-12:
                    grad_trans_norm_const = 1.0
            _save_grayscale_fixed(
                grad_map_rot,
                os.path.join(grad_rot_dir, f"grad_rot_{i:04d}.png"),
                fixed_max=grad_rot_norm_const,
                gamma=image_gamma,
            )
            _save_grayscale_fixed(
                grad_map_trans,
                os.path.join(grad_trans_dir, f"grad_trans_{i:04d}.png"),
                fixed_max=grad_trans_norm_const,
                gamma=image_gamma,
            )

        if loss_scalar < best_loss_value:
            best_loss_value = loss_scalar
            best_pose = pose_current.detach().clone()

    _plot_curve(
        total_losses,
        "Total Loss",
        "Loss",
        os.path.join(diagnostics_dir, "curve_total_loss.png"),
    )
    _plot_curve(
        _safe_normalize(total_losses),
        "Total Loss (Normalized to Iteration 0)",
        "Loss / Loss(0)",
        os.path.join(diagnostics_dir, "curve_total_loss_normalized.png"),
    )
    _plot_curve(
        pose_delta_rot_norms,
        "Pose Delta Rotation Norm",
        "Radians",
        os.path.join(diagnostics_dir, "curve_pose_delta_rotation_norm.png"),
    )
    _plot_curve(
        pose_delta_trans_norms,
        "Pose Delta Translation Norm",
        "L2 Norm",
        os.path.join(diagnostics_dir, "curve_pose_delta_translation_norm.png"),
    )
    _plot_curve(
        rotation_param_norms,
        "Rotation Parameter Norm",
        "L2 Norm",
        os.path.join(diagnostics_dir, "curve_rotation_param_norm.png"),
    )
    _plot_curve(
        translation_param_norms,
        "Translation Parameter Norm",
        "L2 Norm",
        os.path.join(diagnostics_dir, "curve_translation_param_norm.png"),
    )
    _plot_curve(
        rotation_grad_norms,
        "Rotation Gradient Norm",
        "L2 Norm",
        os.path.join(diagnostics_dir, "curve_rotation_grad_norm.png"),
    )
    _plot_curve(
        translation_grad_norms,
        "Translation Gradient Norm",
        "L2 Norm",
        os.path.join(diagnostics_dir, "curve_translation_grad_norm.png"),
    )

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
