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

from sh_helpers import fit_sh_coeffs_per_point, get_sh_bases_torch
from src.utilities import backproject_depth_to_pointcloud


# ── differentiable equirect env-map sampling ─────────────────────────────────

def sample_env_equirect(env_map_hwc: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Bilinear equirect lookup (flip_u=True, flip_v=True), fully differentiable.

    env_map_hwc : [H, W, 3]  linear float [0, 1]
    dirs        : [N, 3]     unit directions (grad flows through here)
    returns     : [N, 3]
    """
    dirs = F.normalize(dirs.float(), dim=-1, eps=1e-8)
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    env_h, env_w = env_map_hwc.shape[:2]

    lon = torch.atan2(x, z)
    lat = torch.asin(torch.clamp(y, -1.0 + 1e-6, 1.0 - 1e-6))
    u_px = (lon / (2.0 * torch.pi) + 0.5) * (env_w - 1)
    u_px = (env_w - 1) - u_px
    v_px = (0.5 + lat / torch.pi) * (env_h - 1)

    u_n = u_px / (env_w - 1) * 2.0 - 1.0
    v_n = v_px / (env_h - 1) * 2.0 - 1.0
    grid = torch.stack([u_n, v_n], dim=-1).reshape(1, 1, -1, 2)

    img = env_map_hwc.permute(2, 0, 1).unsqueeze(0).float()
    out = F.grid_sample(
        img, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return out.squeeze(0).squeeze(1).T  # [N, 3]


def eval_sh_rgb(
    harmonics: torch.Tensor, dirs: torch.Tensor, sh_degree: int = 2
) -> torch.Tensor:
    """Evaluate per-point SH colour at ``dirs`` (matches gsplat / the fitting basis).

    harmonics : [N, K, 3]  K = (sh_degree+1)^2
    dirs      : [N, 3]     view directions in the SH (object) frame
    returns   : [N, 3]     linear RGB, clamped [0, 1]

    The fitting convention (sh_helpers) guarantees ``bases · H = rgb - 0.5``, so
    this reproduces gsplat's own SH evaluation while staying differentiable.
    """
    bases = get_sh_bases_torch(dirs.unsqueeze(0), max_degree=sh_degree)  # [1, N, K]
    K = (sh_degree + 1) ** 2
    rgb = torch.einsum("bnk,nkc->bnc", bases, harmonics[:, :K].float())[0] + 0.5
    return rgb.clamp(0.0, 1.0)


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


def _gs_rasterize_direct(
    means: torch.Tensor,       # [N, 3] camera space (grad ok)
    quats: torch.Tensor,       # [N, 4]
    scales: torch.Tensor,      # [N, 3]
    opacities: torch.Tensor,   # [N]
    colors: torch.Tensor,      # [N, 3] precomputed per-Gaussian RGB (grad ok)
    K_mat: torch.Tensor,       # [3, 3]
    H: int,
    W: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rasterize Gaussians with *precomputed* per-point colours (no SH eval).

    Differentiable w.r.t. both ``means`` and ``colors`` — the path used by the
    photometric refinement, where colours are evaluated/relit analytically so
    pose gradient flows through geometry *and* appearance.

    Returns (image [H,W,3], depth [H,W], mask [H,W] bool).
    """
    device, dtype = means.device, means.dtype
    identity = torch.eye(4, device=device, dtype=dtype)
    poses = identity.unsqueeze(0)

    rendered, alphas, _ = rasterization(
        means=means.unsqueeze(0),
        quats=quats.unsqueeze(0),
        scales=scales.unsqueeze(0),
        opacities=opacities.unsqueeze(0),
        colors=colors.unsqueeze(0),                       # [1, N, 3] → direct colours
        viewmats=torch.linalg.inv(poses).unsqueeze(0),
        Ks=torch.stack([K_mat]).unsqueeze(0),
        width=W,
        height=H,
        sh_degree=None,
        packed=False,
        render_mode="RGB+D",
    )

    alpha = alphas[0, 0, ..., -1]
    frame = rendered[0, 0]
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
    # Per-point separated diffuse colour [N, 3] linear (set by reflection
    # separation).  Used by ``render_relit``; ``None`` falls back to SH ambient.
    diffuse_colors: torch.Tensor | None = None

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

    def render_relit(
        self,
        rel_pose: torch.Tensor | None = None,
        env_map: torch.Tensor | None = None,
        alpha: float = 1.0,
        sh_degree: int = 2,
        scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Differentiable single central-view render with optional relighting.

        Computes a *precomputed* per-Gaussian RGB and splats it via gsplat, so
        the pose gradient flows through both the Gaussian means (geometry) and
        the analytically-evaluated colours (appearance) — no Wigner-D rotation
        and no per-step SH refit.

        rel_pose : [4, 4] rigid transform applied to the Gaussians (object →
                   current camera).  ``None`` renders the canonical view.
        env_map  : [H, W, 3] (or [3, H, W]) linear equirect env map.  When given
                   and ``alpha < 1`` the colour is
                   ``alpha * diffuse + (1 - alpha) * env[reflect(view, normal)]``;
                   otherwise the stored SH appearance is used.

        Returns  : (image [H,W,3], depth [H,W], mask [H,W] bool), differentiable.
        """
        device = self.points.device
        dtype = torch.float32

        pts = self.points.to(dtype)
        normals = self.normals.to(dtype)
        if rel_pose is not None:
            rel_pose = rel_pose.to(device=device, dtype=dtype)
            R = rel_pose[:3, :3]
            t = rel_pose[:3, 3]
            means = (R @ pts.T).T + t
            normals_cur = F.normalize((R @ normals.T).T, dim=-1, eps=1e-8)
        else:
            R = torch.eye(3, device=device, dtype=dtype)
            means = pts
            normals_cur = F.normalize(normals, dim=-1, eps=1e-8)

        # Camera is at the origin in camera space, so the view direction to each
        # Gaussian is just the (normalised) camera-space mean.
        view_dirs = F.normalize(means, dim=-1, eps=1e-8)

        relit = env_map is not None and alpha < 0.999
        if relit:
            env_hwc = env_map.to(device=device, dtype=dtype)
            if env_hwc.ndim == 3 and env_hwc.shape[0] == 3:
                env_hwc = env_hwc.permute(1, 2, 0)
            reflected = F.normalize(
                view_dirs
                - 2.0 * (view_dirs * normals_cur).sum(-1, keepdim=True) * normals_cur,
                dim=-1,
                eps=1e-8,
            )
            specular = sample_env_equirect(env_hwc, reflected)
            if self.diffuse_colors is not None:
                diffuse = self.diffuse_colors.to(device=device, dtype=dtype)
            else:  # SH ambient (degree-0) fallback
                diffuse = (self.harmonics[:, 0].float() + 0.5).clamp(0.0, 1.0)
            colors = (alpha * diffuse + (1.0 - alpha) * specular).clamp(0.0, 1.0)
        else:
            # View-dependent SH appearance: evaluate the stored (object-frame)
            # coefficients at the view direction expressed in the object frame.
            dirs_obj = (R.T @ view_dirs.T).T
            colors = eval_sh_rgb(self.harmonics, dirs_obj, sh_degree=sh_degree)

        H, W, K_mat = self.H, self.W, self.K.float()
        if scale != 1.0:
            H = max(1, round(self.H * scale))
            W = max(1, round(self.W * scale))
            sx, sy = W / self.W, H / self.H
            K_mat = K_mat.clone()
            K_mat[0, 0] *= sx
            K_mat[0, 2] *= sx
            K_mat[1, 1] *= sy
            K_mat[1, 2] *= sy

        return _gs_rasterize_direct(
            means=means,
            quats=self.quats.float(),
            scales=self.scales.float(),
            opacities=self.opacities.float(),
            colors=colors,
            K_mat=K_mat,
            H=H,
            W=W,
        )
