from time import time
from tqdm import tqdm
import torch
from pytorch3d.renderer.cameras import PerspectiveCameras
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


class SurfaceLF:
    """
    Canonical frame = object frame:
      - object is centered at origin (pc shifted by centroid)
      - cameras are expressed relative to the object pose (translation only; rotation ignored)
      - cameras are stored inverted (so "camera pose at origin" -> inv(object_pose_no_rot))
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
    ):
        """
        Args:
          LF: whatever you want to keep (images/light-field tensor etc.)
          cam_poses: (N,4,4) camera poses in a common world frame (assumed cam2world)
          K: (3,3)
          current_object_pose: (4,4) object pose in that same world frame
          pc: (M,3) point cloud (world frame)
          LF: (N,C,H,W) image tensor used for SH fitting
          image_hw: (H, W), or inferred from LF
        """
        if image_hw is None:
            if not (isinstance(LF, torch.Tensor) and LF.ndim == 4):
                raise ValueError(
                    "image_hw must be provided when LF is not an (N,C,H,W) tensor."
                )
            image_hw = (int(LF.shape[-2]), int(LF.shape[-1]))

        self.LF = LF.to(dtype=dtype)
        self.K = K.to(dtype=dtype)
        self.H, self.W = int(image_hw[0]), int(image_hw[1])
        pc = pc.to(dtype=dtype)
        cam_poses = cam_poses.to(dtype=dtype)
        current_object_pose = current_object_pose.to(dtype=dtype)

        device = pc.device
        dtype = pc.dtype

        # 1) ignore object rotation, keep only translation
        object_pose_no_rot = current_object_pose.to(device=device, dtype=dtype).clone()
        object_pose_no_rot[:3, :3] = torch.eye(3, device=device, dtype=dtype)

        # 2) move point cloud from world frame to object frame, then center it.
        points_world = pc.to(device=device, dtype=dtype)
        object_world_inv = torch.linalg.inv(object_pose_no_rot)
        points_h = torch.cat(
            [
                points_world,
                torch.ones(points_world.shape[0], 1, device=device, dtype=dtype),
            ],
            dim=1,
        )
        points_object = (object_world_inv @ points_h.T).T[:, :3]

        self.pc_centroid = points_object.mean(dim=0)
        self.pc0 = (
            points_object - self.pc_centroid
        )  # canonical means used by Surface LF

        # 3) cameras: make them relative to object (translation-only object pose),
        #    then invert as you requested.
        #    If cam_poses are cam2world:
        #      cam_in_object = inv(object_pose_no_rot) @ cam_pose
        #    Then "invert cameras":
        #      stored = inv(cam_in_object)
        cam_poses = cam_poses.to(device=device, dtype=dtype)
        cam_in_object = torch.linalg.inv(object_pose_no_rot) @ cam_poses
        self.cam_poses = torch.linalg.inv(cam_in_object)
        self.cameras = self._build_pytorch3d_cameras(
            K=self.K.to(device=device, dtype=dtype),
            cam2world=cam_in_object,
            image_hw=(self.H, self.W),
        )

        # 4) always estimate appearance / gaussian params from LF + points.
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

    def render(self, pose):
        """
        pose: (4,4) relative pose you want to apply in canonical object frame

        Applies:
          pc1 = (R @ pc0.T).T + t
          harmonics rotated by R (same rotation as geometry)
        Returns:
          image, depth
        """
        pose = pose.to(device=self.pc0.device, dtype=self.pc0.dtype)
        R = pose[:3, :3]
        t = pose[:3, 3]

        pc1 = (R @ self.pc0.T).T + t[None, :]

        harmonics1 = transform_shs(self.harmonics0.float(), R.float()).to(
            self.pc0.dtype
        )

        image, depth = batch_rasterize(
            points=pc1.float(),
            quats=self.quats0.float(),
            scales=self.scales0.float(),
            opacities=self.opacities0.float(),
            colors=harmonics1.float(),
            poses=self.cam_poses.float(),
            camera_matrix=self.K.to(dtype=torch.float32, device=self.pc0.device),
            height=self.H,
            width=self.W,
        )
        image = torch.clamp(image, 0.0, 1.0)
        return image, depth


if __name__ == "__main__":
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
    image, depth = surface_lf.render(torch.eye(4))
    print(image.shape, depth.shape)
