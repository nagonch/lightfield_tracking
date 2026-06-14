"""Canonical surface light field model.

Accumulates a growing point cloud (+ normals + diffuse colour + env map) in
*object* coordinates (= camera frame of frame 0).  Renders with relighting so
that the rendered image accounts for current illumination when used for pose
refinement.

Coordinate convention
---------------------
- Object frame  : canonical frame defined by the first-frame object pose P0.
  p_obj = inv(P0[:3,:3]) @ (p_cam0 - P0[:3,3])
- Camera frame  : frame of the (fixed) RGBD camera; changes as the object
  moves.  p_cam_t = R_t @ p_obj + t_t  where P_t = est_poses[t].
- The LF rig cameras are defined relative to the CENTRAL view (= camera
  frame).  Their relative arrangement is fixed for all frames.
"""

import torch
import torch.nn.functional as F
from sh_helpers import RGB2SH, fit_sh_coeffs_per_point
from surface_lf import SurfaceLF, SurfaceLFRig, batch_rasterize


# ── voxel downsampling ────────────────────────────────────────────────────────

def _voxel_downsample(
    points: torch.Tensor,
    normals: torch.Tensor,
    diffuse: torch.Tensor,
    scales: torch.Tensor,
    voxel_size: float,
    max_points: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Average attributes within each voxel cell; hard-cap at max_points."""
    # Assign each point to a voxel
    voxel_idx = (points / voxel_size).long()
    # Cantor-like hash → unique int per voxel
    ix, iy, iz = voxel_idx[:, 0], voxel_idx[:, 1], voxel_idx[:, 2]
    key = ix * 1_000_003 + iy * 1_009 + iz  # pseudo-unique; rare collisions ok
    unique_keys, inverse = torch.unique(key, return_inverse=True)

    n_voxels = unique_keys.shape[0]
    out_pts = torch.zeros(n_voxels, 3, device=points.device, dtype=points.dtype)
    out_nrm = torch.zeros(n_voxels, 3, device=points.device, dtype=points.dtype)
    out_dif = torch.zeros(n_voxels, 3, device=points.device, dtype=points.dtype)
    out_scl = torch.zeros(n_voxels, 3, device=points.device, dtype=points.dtype)
    cnt     = torch.zeros(n_voxels, 1, device=points.device, dtype=points.dtype)

    out_pts.index_add_(0, inverse, points)
    out_nrm.index_add_(0, inverse, normals)
    out_dif.index_add_(0, inverse, diffuse)
    out_scl.index_add_(0, inverse, scales)
    cnt.index_add_(0, inverse, torch.ones(points.shape[0], 1, device=points.device, dtype=points.dtype))

    out_pts = out_pts / cnt
    out_nrm = F.normalize(out_nrm / cnt, dim=-1, eps=1e-8)
    out_dif = (out_dif / cnt).clamp(0.0, 1.0)
    out_scl = out_scl / cnt

    if n_voxels > max_points:
        # Random subsample — keeps most recent additions by virtue of random
        perm = torch.randperm(n_voxels, device=points.device)[:max_points]
        out_pts = out_pts[perm]
        out_nrm = out_nrm[perm]
        out_dif = out_dif[perm]
        out_scl = out_scl[perm]

    return out_pts, out_nrm, out_dif, out_scl


# ── canonical model ───────────────────────────────────────────────────────────

class CanonicalModel:
    """Growing point cloud + env map in object frame, with relighting render."""

    MAX_POINTS = 60_000
    VOXEL_SIZE = 3e-3   # 3 mm — roughly 1 px at 0.5 m depth with typical intrinsics
    ENV_FUSION_ALPHA = 0.6  # weight for new env map observation

    def __init__(
        self,
        points_obj: torch.Tensor,
        normals_obj: torch.Tensor,
        diffuse_colors: torch.Tensor,
        pc_scales: torch.Tensor,
        environment_map: torch.Tensor | None,
        rig: SurfaceLFRig,
        alpha: float,
    ):
        self.points_obj = points_obj        # [N, 3]
        self.normals_obj = normals_obj      # [N, 3]
        self.diffuse_colors = diffuse_colors # [N, 3]
        self.pc_scales = pc_scales          # [N, 3]
        self.environment_map = environment_map  # [H_e, W_e, 3] or None
        self.rig = rig
        self.alpha = alpha

    # ── construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_surface_lf(
        cls,
        surface_lf: SurfaceLF,
        pose0: torch.Tensor,
        alpha: float,
    ) -> "CanonicalModel":
        """Initialise from first frame.

        pose0 : [4, 4] object pose in camera frame at t=0 (est_poses[0]).
        """
        pts_cam = surface_lf.values["means"].clone()
        nrm_cam = surface_lf.surface_normals
        if nrm_cam is None:
            nrm_cam = _fallback_normals(pts_cam)

        pts_obj, nrm_obj = _cam_to_obj(pts_cam, nrm_cam, pose0)

        dif = surface_lf.diffuse_color_per_point
        if dif is None:
            # Fallback: mean colour
            valid = surface_lf.valid.float()
            count = valid.sum(dim=0).clamp(min=1).unsqueeze(-1)
            dif = (surface_lf.colors * valid.unsqueeze(-1)).sum(dim=0) / count
        dif = dif.to(pts_obj.device, dtype=pts_obj.dtype).clamp(0.0, 1.0)

        scl = surface_lf.values["scales"].clone().to(pts_obj.device, dtype=pts_obj.dtype)

        env = None
        if surface_lf.environment_map is not None:
            env = surface_lf.environment_map.detach().clone()
            # Ensure HWC
            if env.ndim == 3 and env.shape[0] == 3:
                env = env.permute(1, 2, 0).contiguous()

        return cls(
            points_obj=pts_obj,
            normals_obj=nrm_obj,
            diffuse_colors=dif,
            pc_scales=scl,
            environment_map=env,
            rig=surface_lf.rig,
            alpha=alpha,
        )

    # ── fusion ────────────────────────────────────────────────────────────────

    def fuse_frame(self, surface_lf: SurfaceLF, pose_t: torch.Tensor) -> None:
        """Add observations from frame t into the canonical model.

        pose_t : [4, 4] refined object pose in camera frame at time t.
        """
        pts_cam = surface_lf.values["means"].clone()
        nrm_cam = surface_lf.surface_normals
        if nrm_cam is None:
            nrm_cam = _fallback_normals(pts_cam)

        new_pts_obj, new_nrm_obj = _cam_to_obj(pts_cam, nrm_cam, pose_t)

        dif = surface_lf.diffuse_color_per_point
        if dif is None:
            valid = surface_lf.valid.float()
            count = valid.sum(dim=0).clamp(min=1).unsqueeze(-1)
            dif = (surface_lf.colors * valid.unsqueeze(-1)).sum(dim=0) / count
        dif = dif.to(new_pts_obj.device, dtype=new_pts_obj.dtype).clamp(0.0, 1.0)
        new_scl = surface_lf.values["scales"].clone().to(new_pts_obj.device, dtype=new_pts_obj.dtype)

        # Concatenate
        self.points_obj   = torch.cat([self.points_obj,   new_pts_obj],  dim=0)
        self.normals_obj  = torch.cat([self.normals_obj,  new_nrm_obj],  dim=0)
        self.diffuse_colors = torch.cat([self.diffuse_colors, dif],       dim=0)
        self.pc_scales    = torch.cat([self.pc_scales,    new_scl],       dim=0)

        # Bound size via voxel merge
        if self.points_obj.shape[0] > self.MAX_POINTS:
            self.points_obj, self.normals_obj, self.diffuse_colors, self.pc_scales = (
                _voxel_downsample(
                    self.points_obj,
                    self.normals_obj,
                    self.diffuse_colors,
                    self.pc_scales,
                    self.VOXEL_SIZE,
                    self.MAX_POINTS,
                )
            )

        # Fuse env map
        if surface_lf.environment_map is not None:
            curr_env = surface_lf.environment_map.detach()
            if curr_env.ndim == 3 and curr_env.shape[0] == 3:
                curr_env = curr_env.permute(1, 2, 0).contiguous()
            curr_env = curr_env.to(self.points_obj.device, dtype=self.points_obj.dtype)

            if self.environment_map is None:
                self.environment_map = curr_env
            else:
                prev_env = self.environment_map.to(curr_env.device, curr_env.dtype)
                prev_valid = prev_env.sum(dim=-1, keepdim=True) > 1e-8
                curr_valid = curr_env.sum(dim=-1, keepdim=True) > 1e-8
                fused = prev_env.clone()
                only_curr = (~prev_valid) & curr_valid
                both = prev_valid & curr_valid
                blended = (1.0 - self.ENV_FUSION_ALPHA) * prev_env + self.ENV_FUSION_ALPHA * curr_env
                fused = torch.where(only_curr.expand_as(fused), curr_env, fused)
                fused = torch.where(both.expand_as(fused), blended, fused)
                self.environment_map = fused

    # ── rendering ─────────────────────────────────────────────────────────────

    def rasterize(
        self,
        pose_t: torch.Tensor,
        min_valid_views: int = 3,
        lambda_reg: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Render canonical model re-lit at object pose pose_t.

        pose_t : [4, 4] object pose in camera frame (est_poses[t]).
        Returns image [H,W,3], depth [H,W], mask [H,W] (bool).
        """
        device = self.points_obj.device
        dtype = self.points_obj.dtype

        pose_t = pose_t.to(device=device, dtype=dtype)
        R = pose_t[:3, :3]
        t = pose_t[:3, 3]

        # Transform to camera frame
        pts_cam = (R @ self.points_obj.T).T + t[None, :]
        nrm_cam = F.normalize((R @ self.normals_obj.T).T, dim=-1, eps=1e-8)
        scl = self.pc_scales.to(device=device, dtype=dtype)
        dif = self.diffuse_colors.to(device=device, dtype=dtype)

        N = pts_cam.shape[0]
        n_lf_views = self.rig.poses_4x4.shape[0]

        if self.environment_map is not None:
            # Relight: sample env map using reflected dirs from LF rig cameras
            env_hwc = self.environment_map.to(device=device, dtype=dtype)
            if env_hwc.ndim == 3 and env_hwc.shape[0] == 3:
                env_hwc = env_hwc.permute(1, 2, 0).contiguous()

            pts_rep = pts_cam.unsqueeze(0).expand(n_lf_views, -1, -1)
            cam_centers = self.rig.cameras.get_camera_center().to(device=device, dtype=dtype)
            view_vec = pts_rep - cam_centers[:, None, :]
            view_dirs = F.normalize(view_vec, dim=-1, eps=1e-8)

            nrm_rep = nrm_cam.unsqueeze(0).expand(n_lf_views, -1, -1)
            refl_dirs = (
                view_dirs
                - 2.0 * (view_dirs * nrm_rep).sum(dim=-1, keepdim=True) * nrm_rep
            )
            refl_dirs = F.normalize(refl_dirs, dim=-1, eps=1e-8)

            sampled_rgb, env_valid = _sample_env_map(refl_dirs, env_hwc)

            alpha_t = torch.tensor(self.alpha, device=device, dtype=dtype)
            dif_exp = dif.unsqueeze(0).expand(n_lf_views, -1, -1)
            relit = alpha_t * dif_exp + (1.0 - alpha_t) * sampled_rgb

            # Skip relighting for pts where env coverage is insufficient
            vis_count = torch.ones(n_lf_views, N, device=device, dtype=dtype)
            has_missing = (vis_count > 0) & (~env_valid)
            can_relight = (~has_missing).any(dim=0)

            fit_valid = env_valid.float()
            harmonics = fit_sh_coeffs_per_point(
                relit.float(), view_dirs.float(), fit_valid, max_degree=2, lambda_reg=lambda_reg
            )
            # Fall back to diffuse-only SH where env coverage is bad
            harmonics_dif = _diffuse_harmonics(dif, device, dtype)
            can_rl = can_relight[:, None, None]
            harmonics = torch.where(can_rl, harmonics, harmonics_dif)
        else:
            harmonics = _diffuse_harmonics(dif, device, dtype)

        quats = torch.zeros(N, 4, device=device, dtype=dtype)
        quats[:, 0] = 1.0
        opacities = torch.ones(N, device=device, dtype=dtype)

        H, W = self.rig.image_size_hw
        K = self.rig.K.to(device=device, dtype=dtype)
        pose_eye = torch.eye(4, device=device, dtype=dtype).unsqueeze(0)

        image, depth, mask = batch_rasterize(
            points=pts_cam.float(),
            quats=quats.float(),
            scales=scl.float(),
            opacities=opacities.float(),
            colors=harmonics.float(),
            poses=pose_eye.float(),
            camera_matrix=K.float(),
            height=H,
            width=W,
        )
        image = torch.clamp(image, 0.0, 1.0)
        return image, depth, mask

    def __len__(self) -> int:
        return self.points_obj.shape[0]


# ── helpers ───────────────────────────────────────────────────────────────────

def _cam_to_obj(
    pts_cam: torch.Tensor,
    nrm_cam: torch.Tensor,
    pose: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform camera-frame points/normals to object frame using pose."""
    device = pts_cam.device
    dtype = pts_cam.dtype
    pose = pose.to(device=device, dtype=dtype)
    R = pose[:3, :3]
    t = pose[:3, 3]
    # p_cam = R @ p_obj + t  →  p_obj = R^T @ (p_cam - t)
    pts_obj = (R.T @ (pts_cam - t[None, :]).T).T
    nrm_obj = F.normalize((R.T @ nrm_cam.to(device=device, dtype=dtype).T).T, dim=-1, eps=1e-8)
    return pts_obj, nrm_obj


def _fallback_normals(pts: torch.Tensor) -> torch.Tensor:
    """Trivial +Z normals when proper normals are unavailable."""
    n = torch.zeros_like(pts)
    n[:, 2] = 1.0
    return n


def _diffuse_harmonics(
    diffuse_rgb: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    N = diffuse_rgb.shape[0]
    n_coeffs = (2 + 1) ** 2
    h = torch.zeros(N, n_coeffs, 3, device=device, dtype=dtype)
    h[:, 0, :] = RGB2SH(diffuse_rgb.to(device=device, dtype=dtype))
    return h


def _sample_env_map(
    reflected_dirs: torch.Tensor,
    env_hwc: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear equirect sampling. reflected_dirs: [V, N, 3]. Returns [V, N, 3], [V, N] bool."""
    env_h, env_w = env_hwc.shape[:2]
    dirs_flat = reflected_dirs.reshape(-1, 3)
    dirs_flat = F.normalize(dirs_flat, dim=-1)
    x, y, z = dirs_flat[:, 0], dirs_flat[:, 1], dirs_flat[:, 2]
    lon = torch.atan2(x, z)
    lat = torch.asin(torch.clamp(y, -1.0, 1.0))
    u = (lon / (2.0 * torch.pi) + 0.5) * (env_w - 1)
    v = (0.5 - lat / torch.pi) * (env_h - 1)

    u0 = torch.floor(u).long()
    v0 = torch.floor(v).long()
    u1, v1 = u0 + 1, v0 + 1
    u0w = torch.remainder(u0, env_w)
    u1w = torch.remainder(u1, env_w)
    v0c = torch.clamp(v0, 0, env_h - 1)
    v1c = torch.clamp(v1, 0, env_h - 1)
    du = (u - u0.to(u.dtype)).unsqueeze(-1)
    dv = (v - v0.to(v.dtype)).unsqueeze(-1)
    w00 = (1 - du) * (1 - dv)
    w10 = du * (1 - dv)
    w01 = (1 - du) * dv
    w11 = du * dv
    sampled = (
        w00 * env_hwc[v0c, u0w]
        + w10 * env_hwc[v0c, u1w]
        + w01 * env_hwc[v1c, u0w]
        + w11 * env_hwc[v1c, u1w]
    )
    valid_map = (env_hwc.sum(dim=-1) > eps)
    valid = (
        0.25 * (
            valid_map[v0c, u0w].float()
            + valid_map[v0c, u1w].float()
            + valid_map[v1c, u0w].float()
            + valid_map[v1c, u1w].float()
        ) > 0.5
    )
    finite = torch.isfinite(sampled).all(dim=-1)
    valid = valid & finite
    sampled = torch.nan_to_num(sampled, nan=0.0)
    V_orig, N_orig = reflected_dirs.shape[:2]
    return sampled.view(V_orig, N_orig, 3), valid.view(V_orig, N_orig)
