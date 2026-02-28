from time import time
from tqdm import tqdm
import torch
from pytorch3d.renderer.cameras import PerspectiveCameras
import torch.nn.functional as F
from PIL import Image
from sh_helpers import fit_sh_coeffs_per_point
from gsplat import rasterization
import numpy as np
from dataclasses import dataclass
from e3nn import o3
from src.utilities import Visualizer
from scipy.spatial.transform import Rotation as R

import torch
import torch.nn.functional as F
from pytorch3d.renderer import PerspectiveCameras


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
    render_mode="RGB",
    backgrounds=None,
):
    total_sh_degrees = 2
    rendered, alphas, info = rasterization(
        means=points.unsqueeze(0),
        quats=quats.unsqueeze(0),
        scales=scales.unsqueeze(0),
        opacities=opacities.unsqueeze(0),
        colors=colors.unsqueeze(0),
        viewmats=poses.unsqueeze(0),
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


class SurfaceLF:
    """
    Canonical frame = object frame at construction time.

    Inputs (kept as you requested):
      - LF: (N,C,H,W) tensor used for SH fitting
      - cam_poses: (N,4,4) camera poses in a common world frame (ASSUMED cam2world)
      - K: (3,3) intrinsics
      - current_object_pose: (4,4) object pose in that same world frame (object2world)
      - pc: (M,3) point cloud in world frame

    Internal contract (this is what removes confusion):
      - self.pc0 is in canonical object frame O_prev
      - self.cam2world_obj is cam2world, but "world" == O_prev
        cam2world_obj = inv(current_object_pose) @ cam_poses
      - PyTorch3D cameras are built from that same cam2world_obj
      - At render time, you can pass a right-hand delta in O_prev, and we apply it directly.
    """

    def __init__(
        self,
        LF,
        cam_poses,
        K,
        current_object_pose,
        pc,
        dtype=torch.float32,
        *,
        image_hw=None,
        points_scale=1e-3,
        eps=1e-8,
        # NEW (minimal but necessary): tell us what batch_rasterize expects
        raster_pose_convention: str = "world2cam",  # "world2cam" or "cam2world"
    ):
        if image_hw is None:
            if not (isinstance(LF, torch.Tensor) and LF.ndim == 4):
                raise ValueError(
                    "image_hw must be provided when LF is not an (N,C,H,W) tensor."
                )
            image_hw = (int(LF.shape[-2]), int(LF.shape[-1]))

        self.LF = LF.to(dtype=dtype)
        self.K = K.to(dtype=dtype)
        self.H, self.W = int(image_hw[0]), int(image_hw[1])

        if raster_pose_convention not in ("world2cam", "cam2world"):
            raise ValueError(
                'raster_pose_convention must be "world2cam" or "cam2world".'
            )
        self.raster_pose_convention = raster_pose_convention

        pc = pc.to(dtype=dtype)
        cam_poses = cam_poses.to(dtype=dtype)
        current_object_pose = current_object_pose.to(dtype=dtype)

        device = pc.device
        dtype = pc.dtype

        # --- IMPORTANT CHANGE #1: do NOT "ignore rotation" implicitly.
        # keep the full pose. (Your old code said "ignore rotation" but did not actually do it.) :contentReference[oaicite:2]{index=2}
        current_object_pose = current_object_pose.to(device=device, dtype=dtype).clone()

        # --- Canonicalize point cloud: world -> object (O_prev)
        points_world = pc.to(device=device, dtype=dtype)
        object_world_inv = torch.linalg.inv(current_object_pose)
        points_h = torch.cat(
            [
                points_world,
                torch.ones(points_world.shape[0], 1, device=device, dtype=dtype),
            ],
            dim=1,
        )
        points_object = (object_world_inv @ points_h.T).T[:, :3]
        self.pc0 = points_object.contiguous()

        # --- IMPORTANT CHANGE #2: store ONE camera convention consistently.
        # cam_poses assumed cam2world in the *original* world frame.
        # Convert them into the canonical object frame O_prev:
        # cam2world_obj = inv(T_WO_prev) @ T_WC  == inv(object_pose) @ cam_pose
        cam_poses = cam_poses.to(device=device, dtype=dtype)
        self.cam2world_obj = (object_world_inv @ cam_poses).contiguous()  # (N,4,4)

        # Build PyTorch3D cameras from the SAME cam2world_obj (no mismatch now). :contentReference[oaicite:3]{index=3}
        self.cameras = self._build_pytorch3d_cameras(
            K=self.K.to(device=device, dtype=dtype),
            cam2world=self.cam2world_obj,
            image_hw=(self.H, self.W),
        )

        # Keep around for backward-compat if you referenced this name elsewhere
        # (previously self.cam_poses was inverted; now it's cam2world in O_prev)
        self.cam_poses = self.cam2world_obj

        # For local-delta handling: in canonical O_prev, the "original object axes" are identity.
        self.object_pose_orig = torch.eye(4, device=device, dtype=dtype)

        # Keep calculate() exactly as you had it.
        self.calculate(
            points_object=self.pc0, images=LF, eps=eps, points_scale=points_scale
        )

    @staticmethod
    def _build_pytorch3d_cameras(K, cam2world, image_hw):
        device = cam2world.device
        dtype = cam2world.dtype
        H, W = image_hw
        N = cam2world.shape[0]

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        R_pose = cam2world[:, :3, :3]
        t_pose = cam2world[:, :3, 3]

        # Convert cam2world to world2cam for PyTorch3D's (R,T) convention, then apply your axis flips.
        R = R_pose.transpose(1, 2)
        T = -(R @ t_pose.unsqueeze(-1)).squeeze(-1)

        R = R.clone()
        T = T.clone()
        R[:, 0, :] *= -1
        R[:, 1, :] *= -1
        T[:, 0] *= -1
        T[:, 1] *= -1

        focal_length = torch.stack([fx.expand(N), fy.expand(N)], dim=-1)
        principal_point = torch.stack([cx.expand(N), cy.expand(N)], dim=-1)
        image_size = torch.tensor([[H, W]], device=device, dtype=dtype).expand(N, -1)
        return PerspectiveCameras(
            focal_length=focal_length,
            principal_point=principal_point,
            R=R,
            T=T,
            in_ndc=False,
            image_size=image_size,
            device=device,
        )

    # ---- calculate() LEFT IN (unchanged from your paste) :contentReference[oaicite:4]{index=4}
    def calculate(self, points_object, images, eps=1e-8, points_scale=1e-3):
        if not (isinstance(images, torch.Tensor) and images.ndim == 4):
            raise ValueError("`images` must be a torch tensor of shape (N,C,H,W).")

        device = points_object.device
        dtype = points_object.dtype
        images = images.to(device=device)
        N, _, H, W = images.shape

        points_rep = points_object.unsqueeze(0).expand(N, -1, -1)
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

        sampled = F.grid_sample(
            images, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        colors = sampled.squeeze(-1).permute(0, 2, 1).contiguous()
        colors = colors * valid.unsqueeze(-1).to(colors.dtype)

        cam_centers = self.cameras.get_camera_center()
        view_vec = points_rep - cam_centers[:, None, :]
        view_dirs = view_vec / (view_vec.norm(dim=-1, keepdim=True) + eps)

        self.harmonics0 = fit_sh_coeffs_per_point(
            colors.float(),
            view_dirs.float(),
            valid.float(),
            max_degree=2,
            lambda_reg=1e-3,
        ).to(dtype=dtype)

        self.scales0 = torch.full_like(points_object, float(points_scale))
        self.opacities0 = torch.ones(points_object.shape[0], device=device, dtype=dtype)
        self.quats0 = torch.zeros(
            (points_object.shape[0], 4), device=device, dtype=dtype
        )
        self.quats0[:, 0] = 1.0

        return {
            "means": points_object,
            "harmonics": self.harmonics0,
            "rotations": self.quats0,
            "scales": self.scales0,
            "opacities": self.opacities0,
        }

    def render(self, pose, view_idx=None, pose_is_local_delta=False):
        """
        pose: (4,4) delta pose.

        New clear behavior:
          - pose is assumed to be a right-hand delta in the canonical object frame O_prev.
          - pose_is_local_delta is kept for API compatibility, but in O_prev it should not be needed.
        """
        pose = pose.to(device=self.pc0.device, dtype=self.pc0.dtype)
        R_delta = pose[:3, :3]
        t_delta = pose[:3, 3]

        # In canonical object frame, just apply directly.
        R_eff, t_eff = R_delta, t_delta

        pc1 = (R_eff @ self.pc0.T).T + t_eff[None, :]

        harmonics1 = transform_shs(self.harmonics0.float(), R_eff.float()).to(
            self.pc0.dtype
        )

        if view_idx is None:
            view_idx = int(self.cam2world_obj.shape[0] // 2)

        cam2world = self.cam2world_obj[view_idx : view_idx + 1]  # (1,4,4)

        # batch_rasterize pose convention handling (minimal, but required to stop guessing)
        if self.raster_pose_convention == "cam2world":
            pose_for_raster = cam2world
        else:
            pose_for_raster = torch.linalg.inv(cam2world)

        image, depth = batch_rasterize(
            points=pc1.float(),
            quats=self.quats0.float(),
            scales=self.scales0.float(),
            opacities=self.opacities0.float(),
            colors=harmonics1.float(),
            poses=pose_for_raster.float(),
            camera_matrix=self.K.to(dtype=torch.float32, device=self.pc0.device),
            height=self.H,
            width=self.W,
        )
        image = torch.clamp(image, 0.0, 1.0)
        return image, depth


if __name__ == "__main__":
    from scipy.spatial.transform import Rotation as R

    frame = torch.load("frame_0000.pt")
    LF = (
        frame["LF"]
        .reshape(-1, frame["LF"].shape[2], frame["LF"].shape[3], 3)
        .permute(0, 3, 1, 2)
        .contiguous()
    )
    poses = frame["cam_poses"].reshape(-1, 4, 4)
    surface_lf = SurfaceLF(
        LF=LF,
        cam_poses=poses,
        K=frame["camera_matrix"],
        current_object_pose=frame["object_pose"],
        pc=frame["pc"],
        image_hw=(frame["LF"].shape[2], frame["LF"].shape[3]),
    )
    pose_rel = torch.eye(4)
    pose_rel[:3, :3] = torch.tensor(
        R.from_euler("z", 45, degrees=True).as_matrix()
    ).cuda()
    image, depth = surface_lf.render(pose_rel, pose_is_local_delta=True)
    image = (image.cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(image).save("rendered.png")
