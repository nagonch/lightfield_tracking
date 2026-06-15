"""Production surface light field — Gaussian Splatting backed.

A SurfaceLightField captures, for one light-field frame, the appearance of each
object surface point across every sub-aperture view.  The representation stores
Gaussian Splatting parameters (means, SH harmonics, quats, scales, opacities)
and supports differentiable rasterization via gsplat, in addition to the
per-view colour arrays consumed by the reflection separator.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from gsplat import rasterization
from pytorch3d.ops import knn_points
from pytorch3d.renderer.cameras import PerspectiveCameras

from sh_helpers import fit_sh_coeffs_per_point
from src.utilities import backproject_depth_to_pointcloud


# ── SH rotation (Wigner-D via e3nn) ──────────────────────────────────────────

def transform_shs(shs_feat: torch.Tensor, rotation_matrix: torch.Tensor) -> torch.Tensor:
    """Rotate SH coefficients (up to degree-2) by rotation_matrix."""
    from e3nn import o3

    P = torch.tensor(
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        dtype=rotation_matrix.dtype,
        device=rotation_matrix.device,
    )
    perm_R = torch.linalg.inv(P) @ rotation_matrix @ P
    angles = o3._rotation.matrix_to_angles(perm_R.cpu())

    D1 = o3.wigner_D(1, angles[0], -angles[1], angles[2]).to(device=rotation_matrix.device)
    D2 = o3.wigner_D(2, angles[0], -angles[1], angles[2]).to(device=rotation_matrix.device)

    return torch.cat([
        shs_feat[:, :1],
        D1 @ shs_feat[:, 1:4],
        D2 @ shs_feat[:, 4:9],
    ], dim=1)


# ── gsplat rasterization ──────────────────────────────────────────────────────

def _gs_rasterize(
    means: torch.Tensor,       # [N, 3] in camera space
    quats: torch.Tensor,       # [N, 4]
    scales: torch.Tensor,      # [N, 3]
    opacities: torch.Tensor,   # [N]
    harmonics: torch.Tensor,   # [N, K, 3]  K = (degree+1)^2
    K_mat: torch.Tensor,       # [3, 3]
    H: int,
    W: int,
    sh_degree: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rasterize Gaussians from the identity viewpoint (camera-space means).

    Returns (image [H,W,3] float32, depth [H,W] float32, mask [H,W] bool).
    """
    device = means.device
    dtype = means.dtype

    # Identity cam2world → world2cam (viewmat) is also identity.
    identity = torch.eye(4, device=device, dtype=dtype)
    poses = identity.unsqueeze(0)                                   # [1, 4, 4]

    rendered, alphas, _ = rasterization(
        means=means.unsqueeze(0),                                   # [1, N, 3]
        quats=quats.unsqueeze(0),                                   # [1, N, 4]
        scales=scales.unsqueeze(0),                                 # [1, N, 3]
        opacities=opacities.unsqueeze(0),                           # [1, N]
        colors=harmonics.unsqueeze(0),                              # [1, N, K, 3]
        viewmats=torch.linalg.inv(poses).unsqueeze(0),              # [1, 1, 4, 4]
        Ks=torch.stack([K_mat]).unsqueeze(0),                       # [1, 1, 3, 3]
        width=W,
        height=H,
        sh_degree=sh_degree,
        packed=False,
        render_mode="RGB+D",
    )

    alpha = alphas[0, 0, ..., -1]          # [H, W]
    frame = rendered[0, 0]                  # [H, W, 4]
    image = frame[..., :3].clamp(0.0, 1.0)
    depth = frame[..., 3]
    return image, depth, alpha > 0.98


# ── normal estimation ─────────────────────────────────────────────────────────

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


# ── PyTorch3D camera helper ───────────────────────────────────────────────────

def _build_cameras(
    K: torch.Tensor, poses_cam2world: torch.Tensor, H: int, W: int
) -> PerspectiveCameras:
    """PyTorch3D cameras for every sub-aperture view."""
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


# ── main dataclass ─────────────────────────────────────────────────────────────

@dataclass
class SurfaceLightField:
    """Per-surface-point appearance across the sub-aperture views of one frame.

    Gaussian Splatting backed: each surface point is an isotropic Gaussian with
    SH colour coefficients; ``rasterize()`` renders via gsplat.

    Reflection-separation compatible fields (consumed by reflection.py):
      points    : [N, 3]    surface points in the central-camera frame
      colors    : [V, N, 3] colour of each point in each view (0 where not visible)
      normals   : [N, 3]    PCA surface normals
      view_dirs : [V, N, 3] unit view direction (camera→point) per view
      valid     : [V, N]    bool visibility mask
      mask      : [H, W]    bool object mask (central view)

    GS rendering fields:
      harmonics : [N, (deg+1)^2, 3] SH coefficients (degree 2 by default → 9 coeffs)
      quats     : [N, 4]            unit quaternions (identity → spherical Gaussians)
      scales    : [N, 3]            isotropic scales derived from depth footprint
      opacities : [N]               all 1.0
      K         : [3, 3]            camera intrinsics
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
    # GS fields
    harmonics: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor
    K: torch.Tensor

    @classmethod
    def from_frame(
        cls,
        frame: dict,
        mask: torch.Tensor,
        depth: torch.Tensor,
        s_size: int,
        t_size: int,
        sh_degree: int = 2,
        scale_factor: float = 0.5,
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

        cam_centers = cameras.get_camera_center()
        view_dirs = F.normalize(
            points_rep - cam_centers[:, None, :], dim=-1, eps=1e-8
        )

        normals = _estimate_normals(points)

        # Fit SH coefficients from multi-view colour observations.
        harmonics = fit_sh_coeffs_per_point(
            colors.float(),
            view_dirs.float(),
            valid.float(),
            max_degree=sh_degree,
            lambda_reg=1e-3,
        )

        # Isotropic Gaussian scales: one-pixel footprint at the point's depth.
        fx = K[0, 0].float()
        z_vals = points[:, 2].clamp(min=1e-4)
        scale_iso = (z_vals / fx * scale_factor).unsqueeze(-1)   # [N, 1]
        scales = scale_iso.expand(-1, 3).contiguous()

        n_pts = points.shape[0]
        device = points.device
        quats = torch.zeros(n_pts, 4, device=device, dtype=points.dtype)
        quats[:, 0] = 1.0                                          # [1,0,0,0] identity
        opacities = torch.ones(n_pts, device=device, dtype=points.dtype)

        return cls(
            points=points,
            colors=colors,
            normals=normals,
            view_dirs=view_dirs,
            valid=valid,
            mask=mask_bool,
            s_size=s_size,
            t_size=t_size,
            H=H,
            W=W,
            harmonics=harmonics,
            quats=quats,
            scales=scales,
            opacities=opacities,
            K=K.float(),
        )

    def rasterize(
        self,
        rel_pose: torch.Tensor | None = None,
        sh_degree: int = 2,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render via gsplat from the central camera.

        rel_pose : [4, 4] rigid transform applied to the Gaussians before
                   rasterization (for rendering after a pose change).  The SH
                   coefficients are rotated consistently via Wigner-D matrices.
                   Pass ``None`` to render the canonical (prev-frame) view.

        Returns  : (image [H,W,3] float32, depth [H,W] float32, mask [H,W] bool).
        """
        device = self.points.device
        dtype = self.points.dtype

        if rel_pose is not None:
            rel_pose = rel_pose.to(device=device, dtype=dtype)
            R = rel_pose[:3, :3]
            t = rel_pose[:3, 3]
            means = (R @ self.points.T).T + t
            harmonics = transform_shs(self.harmonics.float(), R)
        else:
            means = self.points
            harmonics = self.harmonics

        return _gs_rasterize(
            means=means.float(),
            quats=self.quats.float(),
            scales=self.scales.float(),
            opacities=self.opacities.float(),
            harmonics=harmonics.float(),
            K_mat=self.K.float(),
            H=self.H,
            W=self.W,
            sh_degree=sh_degree,
        )
