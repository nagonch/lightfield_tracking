import torch
import numpy as np
from PIL import Image
from src.dataset import LFDataset
from src.utilities import backproject_depth_to_pointcloud
from surface_lf import SurfaceLF, SurfaceLFRig
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path


def srgb_to_linear(
    srgb: torch.Tensor,
    cutoff: float = 0.04045,
    linear_scale: float = 12.92,
    gamma_offset: float = 0.055,
    gamma_scale: float = 1.055,
    gamma_exponent: float = 2.4,
) -> torch.Tensor:
    """
    srgb: float tensor in [0,1]
    returns linear float tensor in [0,1]
    """
    below = srgb <= cutoff
    linear = torch.empty_like(srgb)
    linear[below] = srgb[below] / linear_scale
    linear[~below] = ((srgb[~below] + gamma_offset) / gamma_scale) ** gamma_exponent
    return linear


def linear_to_srgb(
    linear: torch.Tensor,
    cutoff: float = 0.0031308,
    linear_scale: float = 12.92,
    gamma_offset: float = 0.055,
    gamma_scale: float = 1.055,
    gamma_exponent: float = 2.4,
) -> torch.Tensor:
    """
    linear: float tensor in [0,1]
    returns srgb float tensor in [0,1]
    """
    below = linear <= cutoff
    srgb = torch.empty_like(linear)
    srgb[below] = linear[below] * linear_scale
    srgb[~below] = (
        gamma_scale * (linear[~below] ** (1.0 / gamma_exponent)) - gamma_offset
    )
    return srgb


def build_surface_lf_first_frame(dataset: LFDataset):
    frame = dataset[0]
    s_size, t_size = dataset.metadata["n_views"]

    mask = frame["masks"][s_size // 2, t_size // 2]
    depth = frame["depth"]
    camera_matrix = frame["camera_matrix"]

    pc, pc_scales = backproject_depth_to_pointcloud(
        pixel_indices=None,
        depths=depth,
        camera_matrix=camera_matrix,
        return_scales=True,
    )
    pc = pc[(mask > 0).reshape(-1)]
    pc_scales = pc_scales[(mask > 0).reshape(-1)]

    surface_lf_rig = SurfaceLFRig.build(
        K=frame["camera_matrix"],
        poses_4x4=frame["camera_poses_rel"].reshape(-1, 4, 4),
        image_size_hw=(
            frame["LF"].shape[2],
            frame["LF"].shape[3],
        ),
    )

    surface_lf = SurfaceLF(
        surface_lf_rig,
        pc,
        frame["LF"]
        .reshape(-1, frame["LF"].shape[2], frame["LF"].shape[3], 3)
        .permute(0, 3, 1, 2),
        pc_scales,
        previous_environment_map=None,
    )
    image, depth_rendered, target_mask = surface_lf.rasterize(torch.eye(4).cuda())
    return surface_lf, image, depth_rendered, target_mask


def compute_tv_weight(normal_map, depth_map, sigma_n=1.0, sigma_d=1.0):
    """
    Computes weights for TV based on geometric discontinuities.
    High gradient in normals or depth = lower weight (allows edges).
    """
    # normal_map: [U, V, 3], depth_map: [U, V]
    dn_u = torch.abs(normal_map[1:, :, :] - normal_map[:-1, :, :]).sum(dim=-1)
    dn_v = torch.abs(normal_map[:, 1:, :] - normal_map[:, :-1, :]).sum(dim=-1)
    dd_u = torch.abs(depth_map[1:, :] - depth_map[:-1, :])
    dd_v = torch.abs(depth_map[:, 1:] - depth_map[:, :-1])

    weight_u = torch.exp(-dn_u / sigma_n - dd_u / sigma_d)
    weight_v = torch.exp(-dn_v / sigma_n - dd_v / sigma_d)
    return weight_u, weight_v


class Decomposer(nn.Module):
    def __init__(
        self,
        u,
        v,
        n,
        alpha,
        mask,
        stability_eps=1e-8,
        diffuse_init_scale=0.5,
    ):
        super().__init__()
        self.u, self.v, self.n = u, v, n
        self.alpha = alpha  # [U, V, 1] or scalar
        self.mask = mask  # [U, V]
        self.eps = stability_eps

        # Diffuse is the only free variable. Reflective is solved from exact
        # reconstruction constraint per-pixel/per-view.
        self.diffuse_map = nn.Parameter(
            torch.rand((u, v, 3), device="cuda") * diffuse_init_scale
        )

    def forward(self, color_map_obs):
        # Keep diffuse in [0, 1].
        d = torch.sigmoid(self.diffuse_map)

        # Broadcast diffuse over n dimension: [U, V, 3] -> [U, V, n, 3]
        d_ext = d.unsqueeze(2).expand(-1, -1, self.n, -1)

        # Enforce exact reconstruction in linear space:
        # color_map_obs == alpha * diffuse + (1 - alpha) * reflective
        denom = torch.clamp(1.0 - self.alpha, min=self.eps)
        r = (color_map_obs - self.alpha * d_ext) / denom
        r = torch.nan_to_num(r, nan=0.0, posinf=1.0, neginf=0.0)
        recon = self.alpha * d_ext + (1.0 - self.alpha) * r
        recon = torch.nan_to_num(recon, nan=0.0, posinf=1.0, neginf=0.0)
        return recon, d, r


def optimize_decomposition(
    color_map_obs,
    alpha,
    mask,
    normal_map,
    depth_map,
    n_dims=(5, 5),
    iterations=500,
    return_loss_history=False,
    lr=1e-2,
    diffuse_init_logit_min=1e-4,
    diffuse_init_logit_max=1.0 - 1e-4,
    loss_weight_reflective_spatial_tv=0.2,
    loss_weight_reflective_angular_tv=0.05,
    loss_weight_reflective_range=0.2,
    loss_weight_reconstruction=1e-6,
    log_interval=100,
    model_stability_eps=1e-8,
    model_diffuse_init_scale=0.5,
):
    u, v, n, _ = color_map_obs.shape

    # Move inputs to CUDA
    color_map_obs = color_map_obs.cuda()
    alpha = torch.tensor(alpha).cuda()
    mask_2d = mask.cuda()
    mask = mask_2d.unsqueeze(-1).unsqueeze(-1)  # [U, V, 1, 1]

    # Deterministic boundary behavior for extreme alpha values.
    if torch.allclose(alpha, torch.ones_like(alpha)):
        middle_view_idx = n // 2
        diffuse = color_map_obs[:, :, middle_view_idx, :] * mask_2d.unsqueeze(-1)
        reflective = torch.zeros_like(color_map_obs)
        if return_loss_history:
            return diffuse.detach(), reflective.detach(), [0.0]
        return diffuse.detach(), reflective.detach()

    if torch.allclose(alpha, torch.zeros_like(alpha)):
        diffuse = torch.zeros(
            (u, v, 3), device=color_map_obs.device, dtype=color_map_obs.dtype
        )
        reflective = color_map_obs * mask
        if return_loss_history:
            return diffuse.detach(), reflective.detach(), [0.0]
        return diffuse.detach(), reflective.detach()

    model = Decomposer(
        u,
        v,
        n,
        alpha,
        mask,
        stability_eps=model_stability_eps,
        diffuse_init_scale=model_diffuse_init_scale,
    ).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Start near the per-pixel mean to reduce early degenerate solutions where
    # reflective can collapse to near-black.
    with torch.no_grad():
        d0 = color_map_obs.mean(dim=2).clamp(
            diffuse_init_logit_min, diffuse_init_logit_max
        )
        model.diffuse_map.copy_(torch.log(d0 / (1.0 - d0)))

    # Precompute TV weights from geometry
    w_u, w_v = compute_tv_weight(normal_map.cuda(), depth_map.cuda())
    loss_history = []

    for i in range(iterations):
        optimizer.zero_grad()

        recon, d, r = model(color_map_obs)

        # 1. Reflective Spatial Smoothness (geometry-aware low frequency prior)
        diff_ru = torch.abs(r[1:, :, :, :] - r[:-1, :, :, :]).mean(dim=(2, 3))
        diff_rv = torch.abs(r[:, 1:, :, :] - r[:, :-1, :, :]).mean(dim=(2, 3))
        loss_tv_r_sp = (diff_ru * w_u).mean() + (diff_rv * w_v).mean()

        # 2. Reflective Angular Smoothness
        # Reshape n back to [S, T] to ensure neighbors in angular space are smooth
        r_angular = r.view(u, v, n_dims[0], n_dims[1], 3)
        loss_tv_r_ang = (
            torch.abs(r_angular[:, :, 1:, :, :] - r_angular[:, :, :-1, :, :]).mean()
            + torch.abs(r_angular[:, :, :, 1:, :] - r_angular[:, :, :, :-1, :]).mean()
        )

        # Soft range constraint keeps reflective in [0, 1] while preserving
        # exact reconstruction (no hard clamp in the model path).
        loss_r_range = (F.relu(-r) + F.relu(r - 1.0)).mean()

        # Tiny numerical term for reporting the exactness constraint.
        loss_recon = F.mse_loss(recon * mask, color_map_obs * mask)

        # Total Loss: no diffuse TV, reflective is encouraged to be low-frequency.
        total_loss = (
            loss_weight_reflective_spatial_tv * loss_tv_r_sp
            + loss_weight_reflective_angular_tv * loss_tv_r_ang
            + loss_weight_reflective_range * loss_r_range
            + loss_weight_reconstruction * loss_recon
        )

        total_loss.backward()
        optimizer.step()
        loss_history.append(total_loss.item())

        if i % log_interval == 0:
            print(
                f"Iter {i} | Loss: {total_loss.item():.6f} | Recon(MSE): {loss_recon.item():.8f} "
                f"| R-range: {loss_r_range.item():.6f}"
            )

    with torch.no_grad():
        _, d_final, r_final = model(color_map_obs)

    if return_loss_history:
        return d_final.detach(), r_final.detach(), loss_history
    return d_final.detach(), r_final.detach()


def evaluate_results(diffuse_pred, reflective_pred, diffuse_gt, reflective_gt, mask):
    """
    Evaluates Pred vs GT for both components.
    Tensors expected in [U, V, 3] for diffuse and [U, V, N, 3] for reflective.
    """
    # Move to CPU and numpy for standard imaging metrics
    mask_np = mask.cpu().numpy()

    # --- 1. Diffuse Evaluation ---
    d_pred_np = diffuse_pred.cpu().numpy()
    d_gt_np = diffuse_gt.cpu().numpy().mean(axis=-2)

    # MSE (masked)
    mse_d = np.mean((d_pred_np[mask_np > 0] - d_gt_np[mask_np > 0]) ** 2)
    psnr_d = psnr_func(d_gt_np, d_pred_np, data_range=1.0)

    # SSIM requires a bit of care with the mask (often easier to crop or pad)
    ssim_d = ssim_func(d_gt_np, d_pred_np, channel_axis=-1, data_range=1.0)

    # --- 2. Reflective Evaluation ---
    r_pred_np = reflective_pred.cpu().numpy()
    r_gt_np = reflective_gt.cpu().numpy()

    # MSE (masked, averaged over n)
    # mask_np is [U, V], reflective is [U, V, N, 3]
    expanded_mask = mask_np[:, :, np.newaxis, np.newaxis]
    mse_r = np.mean(((r_pred_np - r_gt_np) * expanded_mask) ** 2)

    # For PSNR on the 4D volume, we treat it as a flattened set of pixels
    reflective_valid_mask = np.broadcast_to(expanded_mask > 0, r_gt_np.shape)
    psnr_r = psnr_func(
        r_gt_np[reflective_valid_mask],
        r_pred_np[reflective_valid_mask],
        data_range=1.0,
    )

    print(f"--- Diffuse Metrics ---")
    print(f"MSE: {mse_d:.6f} | PSNR: {psnr_d:.2f}dB | SSIM: {ssim_d:.4f}")
    print(f"--- Reflective Metrics ---")
    print(f"MSE: {mse_r:.6f} | PSNR: {psnr_r:.2f}dB")

    return {"mse_d": mse_d, "psnr_d": psnr_d, "ssim_d": ssim_d, "mse_r": mse_r}


if __name__ == "__main__":
    sequence_name = "bleach0"

    MIDDLE_REFLECTIVITY = 0.0
    ALPHA = 1 - MIDDLE_REFLECTIVITY

    path_diffuse = f"/home/ngoncharov/cvpr2026/ycbv-eoat-lf/dataset_simple_box_reflective_full_0.0/{sequence_name}"
    path_reflective = f"/home/ngoncharov/cvpr2026/ycbv-eoat-lf/dataset_simple_box_reflective_full_1.0/{sequence_name}"
    path_middle = f"/home/ngoncharov/cvpr2026/ycbv-eoat-lf/dataset_simple_box_reflective_full_{MIDDLE_REFLECTIVITY}/{sequence_name}"

    dataset_diffuse = LFDataset(path_diffuse)
    dataset_reflective = LFDataset(path_reflective)
    dataset_middle = LFDataset(path_middle)

    lf_diffuse_0 = dataset_diffuse[0]["LF"][
        dataset_diffuse.metadata["n_views"][0] // 2,
        dataset_diffuse.metadata["n_views"][1] // 2,
    ]
    lf_reflective_0 = dataset_reflective[0]["LF"][
        dataset_diffuse.metadata["n_views"][0] // 2,
        dataset_diffuse.metadata["n_views"][1] // 2,
    ]
    lf_middle_0 = dataset_middle[0]["LF"][
        dataset_diffuse.metadata["n_views"][0] // 2,
        dataset_diffuse.metadata["n_views"][1] // 2,
    ]
    mask = dataset_diffuse[0]["masks"][
        dataset_diffuse.metadata["n_views"][0] // 2,
        dataset_diffuse.metadata["n_views"][1] // 2,
    ]

    lf_diffuse_0[mask == 0] = 0
    lf_reflective_0[mask == 0] = 0
    lf_middle_0[mask == 0] = 0

    surface_lf_diffuse, image_diffuse, depth_diffuse, mask_diffuse = (
        build_surface_lf_first_frame(dataset_diffuse)
    )
    surface_lf_reflective, image_reflective, depth_reflective, mask_reflective = (
        build_surface_lf_first_frame(dataset_reflective)
    )
    surface_lf_middle, image_middle, depth_middle, mask_middle = (
        build_surface_lf_first_frame(dataset_middle)
    )
    surface_normals = surface_lf_middle.surface_normals

    colors_middle = surface_lf_middle.colors.permute(1, 0, 2)
    object_mask = torch.clone(mask)

    normal_map = torch.zeros(
        (*mask.shape, 3),
        device=surface_normals.device,
    )
    normal_map[mask > 0] = surface_normals.float()

    depth_map = torch.clone(depth_diffuse)
    depth_map[mask == 0] = 0
    color_map = torch.zeros(
        (*mask.shape, colors_middle.shape[1], colors_middle.shape[2]),
        device=colors_middle.device,
    )
    color_map[mask > 0] = colors_middle

    colors_diffuse = surface_lf_diffuse.colors.permute(1, 0, 2)
    color_map_diffuse = torch.zeros(
        (*mask.shape, colors_diffuse.shape[1], colors_diffuse.shape[2]),
        device=colors_diffuse.device,
    )
    color_map_diffuse[mask > 0] = colors_diffuse

    colors_reflective = surface_lf_reflective.colors.permute(1, 0, 2)
    color_map_reflective = torch.zeros(
        (*mask.shape, colors_reflective.shape[1], colors_reflective.shape[2]),
        device=colors_reflective.device,
    )
    color_map_reflective[mask > 0] = colors_reflective

    color_map_diffuse = srgb_to_linear(color_map_diffuse)
    color_map_reflective = srgb_to_linear(color_map_reflective)
    color_map = srgb_to_linear(color_map)

    diffuse, reflective = optimize_decomposition(
        color_map_obs=color_map,
        alpha=ALPHA,
        mask=object_mask,
        normal_map=normal_map,
        depth_map=depth_map,
    )
