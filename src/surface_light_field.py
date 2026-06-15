"""Production surface light field.

A SurfaceLightField captures, for one light-field frame, the appearance of each
object surface point across every sub-aperture view.  It is the minimal input
needed by reflection separation: per-point colours, surface normals and the
object mask.  Nothing else (no SH fit, env map, relighting, rasterisation).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from pytorch3d.ops import knn_points
from pytorch3d.renderer.cameras import PerspectiveCameras

from src.utilities import backproject_depth_to_pointcloud


def _build_cameras(
    K: torch.Tensor, poses_cam2world: torch.Tensor, H: int, W: int
) -> PerspectiveCameras:
    """PyTorch3D cameras for every sub-aperture view (project coordinate flips)."""
    device, dtype = poses_cam2world.device, poses_cam2world.dtype
    N = poses_cam2world.shape[0]

    R = poses_cam2world[:, :3, :3].transpose(1, 2)
    T = -(R @ poses_cam2world[:, :3, 3].unsqueeze(-1)).squeeze(-1)
    R, T = R.clone(), T.clone()
    R[:, 0, :] *= -1
    R[:, 1, :] *= -1
    T[:, 0] *= -1
    T[:, 1] *= -1

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return PerspectiveCameras(
        focal_length=torch.stack([fx.expand(N), fy.expand(N)], dim=-1),
        principal_point=torch.stack([cx.expand(N), cy.expand(N)], dim=-1),
        R=R,
        T=T,
        in_ndc=False,
        image_size=torch.tensor([[H, W]], device=device, dtype=dtype).expand(N, -1),
        device=device,
    )


def _estimate_normals(points: torch.Tensor, k_neighbors: int = 32) -> torch.Tensor:
    """PCA surface normals from local neighbourhoods, robust to degenerate patches."""
    device, dtype = points.device, points.dtype
    n = points.shape[0]
    if n < 3:
        out = torch.zeros((n, 3), device=device, dtype=dtype)
        out[:, 2] = 1.0
        return out

    finite = torch.isfinite(points).all(dim=-1)
    safe = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0).unsqueeze(0)
    k_eff = max(2, min(k_neighbors + 1, n))
    neighbors = knn_points(safe, safe, K=k_eff, return_nn=True).knn[0, :, 1:, :]
    neighbors = torch.nan_to_num(neighbors, nan=0.0, posinf=0.0, neginf=0.0)

    centered = neighbors - neighbors.mean(dim=1, keepdim=True)
    cov = centered.transpose(1, 2) @ centered / max(neighbors.shape[1], 1)
    cov = 0.5 * (cov + cov.transpose(-1, -2))
    cov = cov + 1e-6 * torch.eye(3, device=device, dtype=cov.dtype).unsqueeze(0)
    try:
        _, eigvecs = torch.linalg.eigh(cov)
    except RuntimeError:
        _, eigvecs = torch.linalg.eigh(cov.float().cpu())
        eigvecs = eigvecs.to(device=device, dtype=cov.dtype)

    normals = F.normalize(
        torch.nan_to_num(eigvecs[:, :, 0], nan=0.0, posinf=0.0, neginf=0.0),
        dim=-1,
        eps=1e-8,
    )
    if (~finite).any():
        normals = normals.clone()
        normals[~finite] = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
    return normals


@dataclass
class SurfaceLightField:
    """Per-surface-point appearance across the sub-aperture views of one frame.

    points    : [N, 3]    surface points in the central-camera frame
    colors    : [V, N, 3] colour of each point in each view (0 where not visible)
    normals   : [N, 3]    surface normals
    view_dirs : [V, N, 3] unit view direction (camera→point) per point and view
    valid     : [V, N]    bool visibility mask
    mask      : [H, W]    bool object mask of the central view
    """

    points: torch.Tensor
    colors: torch.Tensor
    normals: torch.Tensor
    view_dirs: torch.Tensor
    valid: torch.Tensor
    mask: torch.Tensor
    s_size: int
    t_size: int
    H: int
    W: int

    @classmethod
    def from_frame(
        cls,
        frame: dict,
        mask: torch.Tensor,
        depth: torch.Tensor,
        s_size: int,
        t_size: int,
    ) -> "SurfaceLightField":
        K = frame["camera_matrix"]
        H, W = frame["LF"].shape[2], frame["LF"].shape[3]
        mask_bool = mask > 0

        points, _ = backproject_depth_to_pointcloud(
            pixel_indices=None, depths=depth, camera_matrix=K, return_scales=True
        )
        points = points[mask_bool.reshape(-1)].float()

        poses = frame["camera_poses_rel"].reshape(-1, 4, 4)
        images = frame["LF"].reshape(-1, H, W, 3).permute(0, 3, 1, 2)
        cameras = _build_cameras(K, poses, H, W)

        N = poses.shape[0]
        points_rep = points.unsqueeze(0).expand(N, -1, -1)
        image_size = torch.tensor([[H, W]], device=points.device).expand(N, -1)
        screen = cameras.transform_points_screen(points_rep, image_size=image_size)
        x, y, z = screen[..., 0], screen[..., 1], screen[..., 2]
        valid = (x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1) & (z > 0)

        grid = torch.stack(
            [(x / (W - 1)) * 2 - 1, (y / (H - 1)) * 2 - 1], dim=-1
        ).unsqueeze(2)
        sampled = F.grid_sample(
            images, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        colors = sampled.squeeze(-1).permute(0, 2, 1).contiguous()
        colors = colors * valid.unsqueeze(-1).to(colors.dtype)

        cam_centers = cameras.get_camera_center()  # [N, 3]
        view_dirs = F.normalize(
            points_rep - cam_centers[:, None, :], dim=-1, eps=1e-8
        )

        return cls(
            points=points,
            colors=colors,
            normals=_estimate_normals(points),
            view_dirs=view_dirs,
            valid=valid,
            mask=mask_bool,
            s_size=s_size,
            t_size=t_size,
            H=H,
            W=W,
        )
