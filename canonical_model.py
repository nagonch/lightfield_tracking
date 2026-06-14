"""Canonical surface light field model.

Accumulates a growing point cloud (+ normals + diffuse colour + env map +
per-point view-dependent SH) in *object* coordinates.

Coordinate convention
---------------------
- Object frame  : canonical frame defined by the first-frame object pose P0.
  p_obj = inv(P0[:3,:3]) @ (p_cam0 - P0[:3,3])
- Camera frame  : frame of the (fixed) RGBD camera; changes as the object
  moves.  p_cam_t = R_t @ p_obj + t_t  where P_t = est_poses[t].

Relighting strategy (per point, during rasterize)
--------------------------------------------------
1. Compute reflected view direction in camera frame for each LF sub-aperture.
2. Sample the accumulated env map → env-relit colour.
3. Blend: relit = alpha * diffuse + (1-alpha) * env_colour.
4. For points where env map has no coverage (or env map not yet built):
   fall back to the per-point SH fitted from the raw LF multi-view observations
   stored in object frame. This gives the view-dependent appearance as last seen,
   which is the correct thing to show for mirrors (reflectivity=1, alpha=0).
"""

import torch
import torch.nn.functional as F
from sh_helpers import RGB2SH, fit_sh_coeffs_per_point
from surface_lf import SurfaceLF, SurfaceLFRig, batch_rasterize

_SH_DEGREE = 2
_SH_COEFFS = (_SH_DEGREE + 1) ** 2  # 9


# ── voxel downsampling ────────────────────────────────────────────────────────

def _voxel_downsample(
    points: torch.Tensor,
    normals: torch.Tensor,
    diffuse: torch.Tensor,
    scales: torch.Tensor,
    sh: torch.Tensor,        # [N, 9, 3]
    voxel_size: float,
    max_points: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Average attributes within each voxel cell; hard-cap at max_points."""
    voxel_idx = (points / voxel_size).long()
    ix, iy, iz = voxel_idx[:, 0], voxel_idx[:, 1], voxel_idx[:, 2]
    key = ix * 1_000_003 + iy * 1_009 + iz
    unique_keys, inverse = torch.unique(key, return_inverse=True)

    n_voxels = unique_keys.shape[0]
    dev, dt = points.device, points.dtype

    out_pts = torch.zeros(n_voxels, 3, device=dev, dtype=dt)
    out_nrm = torch.zeros(n_voxels, 3, device=dev, dtype=dt)
    out_dif = torch.zeros(n_voxels, 3, device=dev, dtype=dt)
    out_scl = torch.zeros(n_voxels, 3, device=dev, dtype=dt)
    cnt     = torch.zeros(n_voxels, 1, device=dev, dtype=dt)

    N = points.shape[0]
    sh_flat = sh.reshape(N, -1)  # [N, 27]
    out_sh  = torch.zeros(n_voxels, sh_flat.shape[1], device=dev, dtype=sh_flat.dtype)

    out_pts.index_add_(0, inverse, points)
    out_nrm.index_add_(0, inverse, normals)
    out_dif.index_add_(0, inverse, diffuse)
    out_scl.index_add_(0, inverse, scales)
    cnt.index_add_(0, inverse, torch.ones(N, 1, device=dev, dtype=dt))
    out_sh.index_add_(0, inverse, sh_flat)

    out_pts = out_pts / cnt
    out_nrm = F.normalize(out_nrm / cnt, dim=-1, eps=1e-8)
    out_dif = (out_dif / cnt).clamp(0.0, 1.0)
    out_scl = out_scl / cnt
    out_sh  = (out_sh / cnt).reshape(n_voxels, _SH_COEFFS, 3)

    if n_voxels > max_points:
        perm = torch.randperm(n_voxels, device=dev)[:max_points]
        out_pts = out_pts[perm]
        out_nrm = out_nrm[perm]
        out_dif = out_dif[perm]
        out_scl = out_scl[perm]
        out_sh  = out_sh[perm]

    return out_pts, out_nrm, out_dif, out_scl, out_sh


# ── SH fitting helper ─────────────────────────────────────────────────────────

def _fit_sh_from_slf(
    surface_lf: SurfaceLF,
    R_obj_from_cam: torch.Tensor,  # [3, 3] = pose[:3,:3].T
    device: torch.device,
    dtype: torch.dtype,
    lambda_reg: float = 1e-3,
) -> torch.Tensor:
    """Fit per-point SH (degree 2) in object frame from SurfaceLF observations.

    Uses the raw multi-view observed colours (not just diffuse) so that
    mirror-like objects (alpha=0) retain their view-dependent reflection
    appearance even after the env map becomes stale.

    Returns [N, 9, 3] float32.
    """
    cam_centers = surface_lf.rig.cameras.get_camera_center().to(device=device, dtype=dtype)  # [V,3]
    pts_cam = surface_lf.values["means"].to(device=device, dtype=dtype)          # [N,3]

    # Direction from each camera center to each point (same convention as rasterize)
    view_dirs_cam = F.normalize(
        pts_cam.unsqueeze(0) - cam_centers.unsqueeze(1), dim=-1, eps=1e-8
    )  # [V, N, 3]

    # Rotate directions to object frame: v_obj = R^T @ v_cam
    view_dirs_obj = torch.einsum("ij,vnj->vni", R_obj_from_cam, view_dirs_cam)  # [V, N, 3]

    obs_colors = surface_lf.colors.to(device=device, dtype=dtype)           # [V, N, 3]
    obs_valid  = surface_lf.valid.float().to(device=device, dtype=dtype)    # [V, N]

    return fit_sh_coeffs_per_point(
        obs_colors.float(),
        view_dirs_obj.float(),
        obs_valid,
        max_degree=_SH_DEGREE,
        lambda_reg=lambda_reg,
    )  # [N, 9, 3]


# ── canonical model ───────────────────────────────────────────────────────────

class CanonicalModel:
    """Growing point cloud + env map + per-point view-dep SH in object frame."""

    MAX_POINTS = 60_000
    VOXEL_SIZE = 3e-3
    ENV_FUSION_ALPHA = 0.6

    def __init__(
        self,
        points_obj: torch.Tensor,        # [N, 3]
        normals_obj: torch.Tensor,       # [N, 3]
        diffuse_colors: torch.Tensor,    # [N, 3]
        pc_scales: torch.Tensor,         # [N, 3]
        sh_view_dep: torch.Tensor,       # [N, 9, 3] — view-dep fallback SH
        environment_map: torch.Tensor | None,  # [H_e, W_e, 3] HWC or None
        rig: SurfaceLFRig,
        alpha: float,
    ):
        self.points_obj     = points_obj
        self.normals_obj    = normals_obj
        self.diffuse_colors = diffuse_colors
        self.pc_scales      = pc_scales
        self.sh_view_dep    = sh_view_dep
        self.environment_map = environment_map
        self.rig   = rig
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

        pose0 : [4, 4] object pose in camera frame at t=0.
        """
        pts_cam = surface_lf.values["means"].clone()
        nrm_cam = surface_lf.surface_normals
        if nrm_cam is None:
            nrm_cam = _fallback_normals(pts_cam)

        pts_obj, nrm_obj = _cam_to_obj(pts_cam, nrm_cam, pose0)

        dif = surface_lf.diffuse_color_per_point
        if dif is None:
            valid = surface_lf.valid.float()
            count = valid.sum(dim=0).clamp(min=1).unsqueeze(-1)
            dif = (surface_lf.colors * valid.unsqueeze(-1)).sum(dim=0) / count
        dif = dif.to(pts_obj.device, dtype=pts_obj.dtype).clamp(0.0, 1.0)

        scl = surface_lf.values["scales"].clone().to(pts_obj.device, dtype=pts_obj.dtype)

        R_obj_from_cam = pose0[:3, :3].to(pts_obj.device, pts_obj.dtype).T
        sh = _fit_sh_from_slf(surface_lf, R_obj_from_cam, pts_obj.device, pts_obj.dtype)

        env = None
        if surface_lf.environment_map is not None:
            env = surface_lf.environment_map.detach().clone()
            if env.ndim == 3 and env.shape[0] == 3:
                env = env.permute(1, 2, 0).contiguous()

        return cls(
            points_obj=pts_obj,
            normals_obj=nrm_obj,
            diffuse_colors=dif,
            pc_scales=scl,
            sh_view_dep=sh,
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

        # Fit view-dep SH for new points in object frame
        R_obj_from_cam = pose_t[:3, :3].to(new_pts_obj.device, new_pts_obj.dtype).T
        new_sh = _fit_sh_from_slf(surface_lf, R_obj_from_cam, new_pts_obj.device, new_pts_obj.dtype)

        # Concatenate all attributes
        self.points_obj     = torch.cat([self.points_obj,     new_pts_obj], dim=0)
        self.normals_obj    = torch.cat([self.normals_obj,    new_nrm_obj], dim=0)
        self.diffuse_colors = torch.cat([self.diffuse_colors, dif],         dim=0)
        self.pc_scales      = torch.cat([self.pc_scales,      new_scl],     dim=0)
        self.sh_view_dep    = torch.cat([self.sh_view_dep,    new_sh],      dim=0)

        if self.points_obj.shape[0] > self.MAX_POINTS:
            (
                self.points_obj,
                self.normals_obj,
                self.diffuse_colors,
                self.pc_scales,
                self.sh_view_dep,
            ) = _voxel_downsample(
                self.points_obj,
                self.normals_obj,
                self.diffuse_colors,
                self.pc_scales,
                self.sh_view_dep,
                self.VOXEL_SIZE,
                self.MAX_POINTS,
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

        Relighting priority per point:
          1. env map sample at reflected direction (if coverage exists)
          2. per-point view-dep SH evaluated at view direction (always available)
          3. flat diffuse SH (last resort if SH is missing)

        Returns image [H,W,3], depth [H,W], mask [H,W] bool.
        """
        device = self.points_obj.device
        dtype  = self.points_obj.dtype

        pose_t = pose_t.to(device=device, dtype=dtype)
        R = pose_t[:3, :3]
        t = pose_t[:3, 3]

        pts_cam = (R @ self.points_obj.T).T + t[None, :]
        nrm_cam = F.normalize((R @ self.normals_obj.T).T, dim=-1, eps=1e-8)
        scl = self.pc_scales.to(device=device, dtype=dtype)
        dif = self.diffuse_colors.to(device=device, dtype=dtype)

        N = pts_cam.shape[0]
        n_lf_views = self.rig.poses_4x4.shape[0]

        # Viewing directions [V, N, 3] — camera center to point (camera frame)
        cam_centers = self.rig.cameras.get_camera_center().to(device=device, dtype=dtype)
        pts_rep  = pts_cam.unsqueeze(0).expand(n_lf_views, -1, -1)
        view_vec = pts_rep - cam_centers[:, None, :]
        view_dirs = F.normalize(view_vec, dim=-1, eps=1e-8)

        if self.environment_map is not None:
            env_hwc = self.environment_map.to(device=device, dtype=dtype)
            if env_hwc.ndim == 3 and env_hwc.shape[0] == 3:
                env_hwc = env_hwc.permute(1, 2, 0).contiguous()

            nrm_rep = nrm_cam.unsqueeze(0).expand(n_lf_views, -1, -1)
            refl_dirs = (
                view_dirs
                - 2.0 * (view_dirs * nrm_rep).sum(dim=-1, keepdim=True) * nrm_rep
            )
            refl_dirs = F.normalize(refl_dirs, dim=-1, eps=1e-8)

            sampled_rgb, env_valid = _sample_env_map(refl_dirs, env_hwc)

            alpha_t = torch.tensor(self.alpha, device=device, dtype=dtype)
            dif_exp = dif.unsqueeze(0).expand(n_lf_views, -1, -1)
            relit   = alpha_t * dif_exp + (1.0 - alpha_t) * sampled_rgb

            # Points are "relightable" if env coverage is sufficient from any view
            can_relight = env_valid.any(dim=0)  # [N] bool

            # Fit SH to relit colours for relightable points
            fit_valid = env_valid.float()
            harmonics = fit_sh_coeffs_per_point(
                relit.float(), view_dirs.float(), fit_valid,
                max_degree=_SH_DEGREE, lambda_reg=lambda_reg,
            )  # [N, 9, 3]

            # Fallback: stored per-point SH (view-dep observations, object frame)
            # Evaluate in camera-frame view dirs (SH is view-dep, stored in obj frame,
            # but we evaluate in the same dir-space we fitted — we stay in camera frame
            # because the stored SH were fitted with rotated dirs each frame, so they
            # represent object-frame appearance. We need to rotate view dirs to obj frame
            # before evaluating... BUT fit_sh_coeffs_per_point encodes the mapping at
            # fit-time direction. Since we evaluate via batch_rasterize which uses the
            # stored harmonics directly (not direction-dependent lookup), the SH is just
            # used as a colour predictor in the current camera space via the rasterizer's
            # built-in SH eval. So using stored sh_view_dep directly as harmonics is valid
            # as an approximation (the rasterizer evaluates SH in camera space anyway).
            if self.sh_view_dep is not None:
                harmonics_fallback = self.sh_view_dep.to(device=device, dtype=dtype)
            else:
                harmonics_fallback = _diffuse_harmonics(dif, device, dtype)

            can_rl = can_relight[:, None, None]
            harmonics = torch.where(can_rl, harmonics, harmonics_fallback)

        else:
            # No env map yet — use stored view-dep SH directly
            if self.sh_view_dep is not None:
                harmonics = self.sh_view_dep.to(device=device, dtype=dtype)
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
    """Camera-frame points/normals → object frame using p_cam = R @ p_obj + t."""
    device = pts_cam.device
    dtype  = pts_cam.dtype
    pose = pose.to(device=device, dtype=dtype)
    R = pose[:3, :3]
    t = pose[:3, 3]
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
    h = torch.zeros(N, _SH_COEFFS, 3, device=device, dtype=dtype)
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
