from time import time
from tqdm import tqdm
import torch
from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.ops import knn_points
import torch.nn.functional as F
from PIL import Image
from sh_helpers import fit_sh_coeffs_per_point
from gsplat import rasterization
from PIL import Image
import numpy as np
from dataclasses import dataclass
from e3nn import o3


def transform_shs(shs_feat, rotation_matrix):
    P = torch.tensor(
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        dtype=rotation_matrix.dtype,
        device=rotation_matrix.device,
    )
    permuted_rotation_matrix = torch.linalg.inv(P) @ rotation_matrix @ P
    rot_angles = o3._rotation.matrix_to_angles(permuted_rotation_matrix.cpu())

    D_1 = o3.wigner_D(1, rot_angles[0], -rot_angles[1], rot_angles[2]).to(
        device=rotation_matrix.device
    )
    D_2 = o3.wigner_D(2, rot_angles[0], -rot_angles[1], rot_angles[2]).to(
        device=rotation_matrix.device
    )
    D_3 = o3.wigner_D(3, rot_angles[0], -rot_angles[1], rot_angles[2]).to(
        device=rotation_matrix.device
    )

    shs_feat_1 = D_1 @ shs_feat[:, 1:4]
    shs_feat_2 = D_2 @ shs_feat[:, 4:9]
    # shs_feat_3 = D_3 @ shs_feat[:, 9:]
    shs_feat = torch.concatenate([shs_feat[:, :1], shs_feat_1, shs_feat_2], dim=1)
    return shs_feat


def batch_rasterize(
    points,
    quats,
    scales,
    opacities,
    colors,
    poses,
    camera_matrix,
    height,
    width,
    render_mode="RGB+D",
    backgrounds=None,
):
    total_sh_degrees = 2
    rendered, alphas, info = rasterization(
        means=points.unsqueeze(0),
        quats=quats.unsqueeze(0),
        scales=scales.unsqueeze(0),
        opacities=opacities.unsqueeze(0),
        colors=colors.unsqueeze(0),
        viewmats=torch.linalg.inv(poses).unsqueeze(0),
        Ks=torch.stack(
            [
                camera_matrix,
            ]
            * poses.shape[0]
        ).unsqueeze(0),
        width=width,
        height=height,
        sh_degree=total_sh_degrees,
        packed=False,
        render_mode=render_mode,
        backgrounds=backgrounds,
    )
    rendered = rendered[0]
    depth = rendered[:, :, :, -1]
    rendered = rendered[:, :, :, :3]
    return rendered[0], depth[0]


@dataclass(frozen=True)
class SurfaceLFRig:
    K: torch.Tensor
    poses_4x4: torch.Tensor
    image_size_hw: tuple[int, int]
    pose_is_cam2world: bool
    cameras: PerspectiveCameras

    @staticmethod
    def build(
        K: torch.Tensor,
        poses_4x4: torch.Tensor,
        image_size_hw: tuple[int, int],
        pose_is_cam2world: bool = True,
    ) -> "SurfaceLFRig":
        device = poses_4x4.device
        dtype = poses_4x4.dtype
        N = poses_4x4.shape[0]
        H, W = image_size_hw

        K_shared = K.to(device=device, dtype=dtype)
        fx, fy = K_shared[0, 0], K_shared[1, 1]
        cx, cy = K_shared[0, 2], K_shared[1, 2]

        R_pose = poses_4x4[:, :3, :3]
        t_pose = poses_4x4[:, :3, 3]

        if pose_is_cam2world:
            R = R_pose.transpose(1, 2)
            T = -(R @ t_pose.unsqueeze(-1)).squeeze(-1)
        else:
            R = R_pose
            T = t_pose

        # your coordinate flips
        R = R.clone()
        T = T.clone()
        R[:, 0, :] *= -1
        R[:, 1, :] *= -1
        T[:, 0] *= -1
        T[:, 1] *= -1

        focal_length = torch.stack([fx.expand(N), fy.expand(N)], dim=-1)
        principal_point = torch.stack([cx.expand(N), cy.expand(N)], dim=-1)
        image_size = torch.tensor([[H, W]], device=device, dtype=dtype).expand(N, -1)

        cameras = PerspectiveCameras(
            focal_length=focal_length,
            principal_point=principal_point,
            R=R,
            T=T,
            in_ndc=False,
            image_size=image_size,
            device=device,
        )

        return SurfaceLFRig(
            K=K_shared,
            poses_4x4=poses_4x4,
            image_size_hw=image_size_hw,
            pose_is_cam2world=pose_is_cam2world,
            cameras=cameras,
        )


class SurfaceLF:
    def __init__(
        self,
        rig: SurfaceLFRig,
        pc,
        images,
        pc_scales,
        previous_environment_map: torch.Tensor | None = None,
        env_fusion_alpha: float = 0.6,
        use_relight: bool = True,
        flip_env_u: bool = True,
        flip_env_v: bool = True,
    ):
        self.rig = rig
        self.use_relight = use_relight
        self.flip_env_u = flip_env_u
        self.flip_env_v = flip_env_v
        self.env_fusion_alpha = env_fusion_alpha
        self.calculate(pc, images, pc_scales, previous_environment_map)

    @staticmethod
    def _ensure_env_hwc(env_map: torch.Tensor) -> torch.Tensor:
        if env_map.ndim != 3:
            raise ValueError("Environment map must be 3D (H,W,3) or (3,H,W)")
        if env_map.shape[-1] == 3:
            return env_map
        if env_map.shape[0] == 3:
            return env_map.permute(1, 2, 0)
        raise ValueError("Environment map must have a 3-channel dimension")

    def _fuse_environment_maps(
        self,
        previous_environment_map: torch.Tensor | None,
        current_environment_map: torch.Tensor,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        if previous_environment_map is None:
            return current_environment_map

        prev = self._ensure_env_hwc(previous_environment_map).to(
            device=current_environment_map.device,
            dtype=current_environment_map.dtype,
        )
        curr = self._ensure_env_hwc(current_environment_map)

        # Keep older values in unseen bins, blend where both maps contain data.
        prev_w = prev.sum(dim=-1, keepdim=True)
        curr_w = curr.sum(dim=-1, keepdim=True)
        prev_valid = prev_w > eps
        curr_valid = curr_w > eps

        fused = prev.clone()
        only_curr = (~prev_valid) & curr_valid
        both_valid = prev_valid & curr_valid

        blended = (1.0 - self.env_fusion_alpha) * prev + self.env_fusion_alpha * curr
        fused = torch.where(only_curr.expand_as(fused), curr, fused)
        fused = torch.where(both_valid.expand_as(fused), blended, fused)
        return fused

    @staticmethod
    def _estimate_normals(points_world: torch.Tensor, k_neighbors: int = 64):
        # PCA normals from local neighborhoods with guards for NaN/Inf and ill-conditioned patches.
        device = points_world.device
        dtype = points_world.dtype
        n_points = points_world.shape[0]

        if n_points < 3:
            fallback = torch.zeros((n_points, 3), device=device, dtype=dtype)
            fallback[:, 2] = 1.0
            return fallback

        finite_points = torch.isfinite(points_world).all(dim=-1)
        points_safe = torch.nan_to_num(points_world, nan=0.0, posinf=0.0, neginf=0.0)

        points = points_safe.unsqueeze(0)
        k_eff = max(2, min(k_neighbors + 1, n_points))
        knn = knn_points(points, points, K=k_eff, return_nn=True)
        neighbors = knn.knn[0, :, 1:, :]
        neighbors = torch.nan_to_num(neighbors, nan=0.0, posinf=0.0, neginf=0.0)

        centered = neighbors - neighbors.mean(dim=1, keepdim=True)
        cov = centered.transpose(1, 2) @ centered / max(neighbors.shape[1], 1)
        cov = torch.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)

        # Symmetrize + diagonal jitter to keep covariance numerically PSD.
        cov = 0.5 * (cov + cov.transpose(-1, -2))
        eye = torch.eye(3, device=device, dtype=cov.dtype).unsqueeze(0)
        cov = cov + 1e-6 * eye

        try:
            _, eigvecs = torch.linalg.eigh(cov)
        except RuntimeError:
            # Fallback avoids sporadic cuSOLVER batched-eigh failures.
            cov_cpu = cov.float().cpu()
            _, eigvecs_cpu = torch.linalg.eigh(cov_cpu)
            eigvecs = eigvecs_cpu.to(device=device, dtype=cov.dtype)

        normals = eigvecs[:, :, 0]
        normals = torch.nan_to_num(normals, nan=0.0, posinf=0.0, neginf=0.0)
        normals = F.normalize(normals, dim=-1, eps=1e-8)

        # Default invalid source points to a stable up normal.
        if (~finite_points).any():
            normals = normals.clone()
            normals[~finite_points] = torch.tensor(
                [0.0, 0.0, 1.0], device=device, dtype=normals.dtype
            )
        return normals

    @staticmethod
    def _dirs_to_equirect_uv(
        dirs: torch.Tensor,
        env_h: int,
        env_w: int,
        flip_u: bool = False,
        flip_v: bool = False,
    ):
        dirs = F.normalize(dirs, dim=-1)
        x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]

        lon = torch.atan2(x, z)
        lat = torch.asin(torch.clamp(y, -1.0, 1.0))

        u = (lon / (2.0 * torch.pi) + 0.5) * (env_w - 1)
        if flip_u:
            u = (env_w - 1) - u
        if flip_v:
            v = (0.5 + lat / torch.pi) * (env_h - 1)
        else:
            v = (0.5 - lat / torch.pi) * (env_h - 1)
        u_idx = torch.clamp(u.round().long(), 0, env_w - 1)
        v_idx = torch.clamp(v.round().long(), 0, env_h - 1)
        return u_idx, v_idx

    @staticmethod
    def _dirs_to_equirect_uv_float(
        dirs: torch.Tensor,
        env_h: int,
        env_w: int,
        flip_u: bool = False,
        flip_v: bool = False,
    ):
        dirs = F.normalize(dirs, dim=-1)
        x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]

        lon = torch.atan2(x, z)
        lat = torch.asin(torch.clamp(y, -1.0, 1.0))

        u = (lon / (2.0 * torch.pi) + 0.5) * (env_w - 1)
        if flip_u:
            u = (env_w - 1) - u
        if flip_v:
            v = (0.5 + lat / torch.pi) * (env_h - 1)
        else:
            v = (0.5 - lat / torch.pi) * (env_h - 1)
        return u, v

    def _build_environment_map(
        self,
        reflected_dirs: torch.Tensor,
        colors: torch.Tensor,
        valid: torch.Tensor,
        view_dirs: torch.Tensor,
        normals: torch.Tensor,
        env_h: int = 256,
        env_w: int = 512,
        eps: float = 1e-8,
    ):
        n_views, n_points, _ = reflected_dirs.shape
        dirs_flat = reflected_dirs.reshape(n_views * n_points, 3)
        colors_flat = colors.reshape(n_views * n_points, 3)
        valid_flat = valid.reshape(n_views * n_points)

        # Confidence from grazing-angle attenuation.
        cos_term = torch.abs((view_dirs * normals.unsqueeze(0)).sum(dim=-1))
        weights_flat = cos_term.reshape(n_views * n_points)

        keep = valid_flat > 0
        if keep.sum() == 0:
            return torch.zeros(
                (3, env_h, env_w), device=colors.device, dtype=colors.dtype
            )

        dirs_flat = dirs_flat[keep]
        colors_flat = colors_flat[keep]
        weights_flat = weights_flat[keep].to(colors_flat.dtype)

        u_idx, v_idx = self._dirs_to_equirect_uv(
            dirs_flat,
            env_h=env_h,
            env_w=env_w,
            flip_u=self.flip_env_u,
            flip_v=self.flip_env_v,
        )
        lin_idx = v_idx * env_w + u_idx

        accum_rgb = torch.zeros(
            (env_h * env_w, 3), device=colors.device, dtype=colors.dtype
        )
        accum_w = torch.zeros(
            (env_h * env_w,), device=colors.device, dtype=colors.dtype
        )

        accum_rgb.index_add_(0, lin_idx, colors_flat * weights_flat.unsqueeze(-1))
        accum_w.index_add_(0, lin_idx, weights_flat)

        env = accum_rgb / (accum_w.unsqueeze(-1) + eps)
        env = env.reshape(env_h, env_w, 3).permute(2, 0, 1).contiguous()
        return env

    def _sample_environment_map(
        self,
        reflected_dirs: torch.Tensor,
        env_map_hwc: torch.Tensor,
        eps: float = 1e-8,
    ):
        """
        Nearest-neighbor sample from an equirect environment map.

        Returns sampled RGB and a validity mask where missing/invalid env bins are False.
        """
        env_h, env_w = env_map_hwc.shape[:2]
        dirs_flat = reflected_dirs.reshape(-1, 3)
        u, v = self._dirs_to_equirect_uv_float(
            dirs_flat,
            env_h=env_h,
            env_w=env_w,
            flip_u=self.flip_env_u,
            flip_v=self.flip_env_v,
        )

        # Bilinear equirect sampling with horizontal wrap to avoid specular banding.
        u0 = torch.floor(u).long()
        v0 = torch.floor(v).long()
        u1 = u0 + 1
        v1 = v0 + 1

        u0w = torch.remainder(u0, env_w)
        u1w = torch.remainder(u1, env_w)
        v0c = torch.clamp(v0, 0, env_h - 1)
        v1c = torch.clamp(v1, 0, env_h - 1)

        du = (u - u0.to(u.dtype)).unsqueeze(-1)
        dv = (v - v0.to(v.dtype)).unsqueeze(-1)
        w00 = (1.0 - du) * (1.0 - dv)
        w10 = du * (1.0 - dv)
        w01 = (1.0 - du) * dv
        w11 = du * dv

        c00 = env_map_hwc[v0c, u0w]
        c10 = env_map_hwc[v0c, u1w]
        c01 = env_map_hwc[v1c, u0w]
        c11 = env_map_hwc[v1c, u1w]
        sampled = w00 * c00 + w10 * c10 + w01 * c01 + w11 * c11
        sampled = sampled.view(*reflected_dirs.shape[:2], 3)

        valid_map = (env_map_hwc.sum(dim=-1) > eps).to(env_map_hwc.dtype)
        vm00 = valid_map[v0c, u0w].unsqueeze(-1)
        vm10 = valid_map[v0c, u1w].unsqueeze(-1)
        vm01 = valid_map[v1c, u0w].unsqueeze(-1)
        vm11 = valid_map[v1c, u1w].unsqueeze(-1)
        valid_interp = w00 * vm00 + w10 * vm10 + w01 * vm01 + w11 * vm11
        valid = valid_interp.squeeze(-1) > 0.5
        valid = valid.view(*reflected_dirs.shape[:2])

        finite = torch.isfinite(sampled).all(dim=-1)
        valid = valid & finite
        sampled = torch.nan_to_num(sampled, nan=0.0, posinf=0.0, neginf=0.0)
        return sampled, valid

    @property
    def K(self):
        return self.rig.K

    @property
    def cameras(self):
        return self.rig.cameras

    @property
    def H(self):
        return self.rig.image_size_hw[0]

    @property
    def W(self):
        return self.rig.image_size_hw[1]

    @property
    def device(self):
        return self.rig.K.device

    def calculate(
        self,
        points_world,
        images,
        pc_scales,
        scale_constant=0.5,
        previous_environment_map: torch.Tensor | None = None,
        eps=1e-8,
    ):
        device = self.device
        N, _, H, W = images.shape

        surface_normals = self._estimate_normals(points_world)

        points_rep = points_world.unsqueeze(0).expand(N, -1, -1)
        image_size = torch.tensor([[H, W]], device=device).expand(N, -1)
        points_screen = self.cameras.transform_points_screen(
            points_rep.float(), image_size=image_size
        )

        uv_px = points_screen[..., :2]
        z = points_screen[..., 2]

        x, y = uv_px[..., 0], uv_px[..., 1]
        in_bounds = (x >= 0) & (x <= (W - 1)) & (y >= 0) & (y <= (H - 1))
        in_front = z > 0
        valid = in_bounds & in_front

        grid_x = (x / (W - 1)) * 2 - 1
        grid_y = (y / (H - 1)) * 2 - 1
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(2)

        sampled = torch.nn.functional.grid_sample(
            images, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        colors = sampled.squeeze(-1).permute(0, 2, 1).contiguous()
        colors = colors * valid.unsqueeze(-1).to(colors.dtype)

        cam_centers = self.cameras.get_camera_center()
        view_vec = points_rep - cam_centers[:, None, :]
        view_dirs = view_vec / (view_vec.norm(dim=-1, keepdim=True) + eps)

        normals_rep = surface_normals.unsqueeze(0).expand(N, -1, -1)
        reflected_dirs = (
            view_dirs
            - 2.0 * (view_dirs * normals_rep).sum(dim=-1, keepdim=True) * normals_rep
        )
        reflected_dirs = F.normalize(reflected_dirs, dim=-1)

        sh_coeffs = fit_sh_coeffs_per_point(
            colors.float(),
            view_dirs.float(),
            valid.float(),
            max_degree=2,
            lambda_reg=1e-3,
        )

        opacities = torch.ones_like(points_world[:, 0])
        quats = torch.stack(
            [
                torch.tensor([1, 0, 0, 0]).cuda(),
            ]
            * points_world.shape[0]
        ).float()

        self.values = {
            "means": points_world,
            "harmonics": sh_coeffs,
            "rotations": quats,
            "scales": pc_scales * scale_constant,
            "opacities": opacities,
        }
        self.surface_normals = surface_normals
        current_environment_map = self._build_environment_map(
            reflected_dirs=reflected_dirs,
            colors=colors,
            valid=valid,
            view_dirs=view_dirs,
            normals=surface_normals,
        )
        current_environment_map = current_environment_map.permute(1, 2, 0)
        self.environment_map = self._fuse_environment_maps(
            previous_environment_map=previous_environment_map,
            current_environment_map=current_environment_map,
        )
        return self.values, self.surface_normals, self.environment_map

    def relight(self, rel_pose, min_valid_views: int = 3, lambda_reg: float = 1e-3):
        rel_pose = rel_pose.to(
            device=self.values["means"].device,
            dtype=self.values["means"].dtype,
        )
        values = self.values.copy()

        R = rel_pose[:3, :3]
        t = rel_pose[:3, 3]

        points0 = self.values["means"]
        points1 = (R @ points0.T).T + t[None, :]
        values["means"] = points1

        old_harmonics = self.values["harmonics"].float()

        env_map_hwc = self._ensure_env_hwc(self.environment_map).to(
            device=points1.device,
            dtype=points1.dtype,
        )

        # Rotate normals consistently with the rigid transform.
        normals0 = self.surface_normals.to(device=points1.device, dtype=points1.dtype)
        normals1 = F.normalize((R @ normals0.T).T, dim=-1, eps=1e-8)

        N = self.rig.poses_4x4.shape[0]
        points_rep = points1.unsqueeze(0).expand(N, -1, -1)
        image_size = torch.tensor([[self.H, self.W]], device=points1.device).expand(
            N, -1
        )
        points_screen = self.cameras.transform_points_screen(
            points_rep.float(), image_size=image_size
        )

        uv_px = points_screen[..., :2]
        z = points_screen[..., 2]
        x, y = uv_px[..., 0], uv_px[..., 1]
        in_bounds = (x >= 0) & (x <= (self.W - 1)) & (y >= 0) & (y <= (self.H - 1))
        in_front = z > 0
        valid_geom = in_bounds & in_front

        cam_centers = self.cameras.get_camera_center()
        view_vec = points_rep - cam_centers[:, None, :]
        view_dirs = view_vec / (view_vec.norm(dim=-1, keepdim=True) + 1e-8)

        normals_rep = normals1.unsqueeze(0).expand(N, -1, -1)
        reflected_dirs = (
            view_dirs
            - 2.0 * (view_dirs * normals_rep).sum(dim=-1, keepdim=True) * normals_rep
        )
        reflected_dirs = F.normalize(reflected_dirs, dim=-1)

        sampled_colors, env_valid = self._sample_environment_map(
            reflected_dirs, env_map_hwc
        )
        fit_valid = valid_geom & env_valid

        # Skip relighting if any required env sample is missing for this Gaussian.
        visible_count = valid_geom.sum(dim=0)
        has_missing_env = (valid_geom & (~env_valid)).any(dim=0)
        can_relight = (visible_count >= min_valid_views) & (~has_missing_env)

        new_harmonics = fit_sh_coeffs_per_point(
            sampled_colors.float(),
            view_dirs.float(),
            fit_valid.float(),
            max_degree=2,
            lambda_reg=lambda_reg,
        )
        values["harmonics"] = torch.where(
            can_relight[:, None, None], new_harmonics, old_harmonics
        )
        return values

    def transform(self, rel_pose):
        # Backward-compatible entrypoint used by refinement code.
        return self.relight(rel_pose)

    def transform_naive(self, rel_pose):
        rel_pose = rel_pose.to(self.values["means"].dtype)
        values = self.values.copy()
        try:
            values["harmonics"] = transform_shs(
                values["harmonics"].float(), rel_pose[:3, :3].float()
            )
        except Exception as e:
            values["harmonics"] = values["harmonics"].float()
        R = rel_pose[:3, :3]
        t = rel_pose[:3, 3]
        points0 = self.values["means"]
        points_centered = points0
        points1 = (R @ points_centered.T).T + t[None, :]
        values["means"] = points1
        return values

    def rasterize(self, rel_pose):
        if self.use_relight:
            values = self.relight(rel_pose)
        else:
            values = self.transform_naive(rel_pose)
        image, depth = batch_rasterize(
            points=values["means"].float(),
            quats=values["rotations"].float(),
            scales=values["scales"].float(),
            opacities=values["opacities"].float(),
            colors=values["harmonics"].float(),
            poses=torch.eye(4).unsqueeze(0).cuda(),
            camera_matrix=self.K,
            height=self.H,
            width=self.W,
        )
        image = torch.clamp(image, 0.0, 1.0)
        return image, depth


if __name__ == "__main__":
    from scipy.spatial.transform import Rotation
    from src.utilities import Visualizer

    K = torch.load("pts/K.pt")
    poses = torch.load("pts/poses_4x4.pt")
    poses_gt = torch.load("pts/poses_gt.pt")
    pc = torch.load("pts/pc_0000.pt")
    images = torch.load("pts/images_0000.pt")

    v = Visualizer()
    surface_lf_rig = SurfaceLFRig.build(
        K=K,
        poses_4x4=poses,
        image_size_hw=images.shape[2:4],
    )
    slf = SurfaceLF(surface_lf_rig, pc, images)
    image, depth = slf.rasterize()
    Image.fromarray((image.cpu().numpy() * 255).astype(np.uint8)).save(
        "test_render.png"
    )
