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
    def __init__(self, rig: SurfaceLFRig, pc, images, current_pose):
        self.rig = rig
        self.pose = current_pose
        self.calculate(pc, images)

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

    def calculate(self, points_world, images, eps=1e-8, points_scale=1e-3):
        device = self.device
        N, _, H, W = images.shape

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

        sh_coeffs = fit_sh_coeffs_per_point(
            colors.float(),
            view_dirs.float(),
            valid.float(),
            max_degree=2,
            lambda_reg=1e-3,
        )

        opacities = torch.ones_like(points_world[:, 0])
        scales = torch.ones_like(points_world) * points_scale
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
            "scales": scales,
            "opacities": opacities,
        }
        return self.values

    def transform(self, rel_pose, inplace=True):
        rel_pose = rel_pose.to(self.values["means"].dtype)
        points_pose_new = rel_pose @ self.pose.to(rel_pose.dtype)
        pose_transform = points_pose_new @ torch.linalg.inv(
            self.pose.to(rel_pose.dtype)
        )

        values = self.values.copy()
        values["harmonics"] = transform_shs(
            values["harmonics"].float(), rel_pose[:3, :3].float()
        )
        points_world = self.values["means"]
        points_world = (
            (pose_transform[:3, :3] @ points_world.T) + pose_transform[:3, 3:4]
        ).T
        values["means"] = points_world
        if inplace:
            self.values = values
            self.pose = points_pose_new
        return values

    def rasterize(self, values=None):
        if values is None:
            values = self.values
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

    poses_rel_gt = [torch.eye(4).cuda()]
    for i in range(1, poses_gt.shape[0]):
        pose_rel_gt = poses_gt[i] @ torch.linalg.inv(poses_gt[i - 1])
        poses_rel_gt.append(pose_rel_gt)

    surface_lf_rig = SurfaceLFRig.build(
        K=K,
        poses_4x4=poses,
        image_size_hw=images.shape[2:4],
    )
    integrated_poses = []
    for i, pose_rel in enumerate(poses_rel_gt):
        if i == 0:
            surface_lf = SurfaceLF(
                rig=surface_lf_rig,
                pc=pc,
                images=images,
                current_pose=poses_gt[0].cuda(),
            )
            pc = surface_lf.values["means"]
            integrated_poses.append(poses_gt[0].cuda())
        else:
            values = surface_lf.transform(
                pose_rel,
                inplace=True,
            )
            pc = values["means"]
            integrated_poses.append(pose_rel @ integrated_poses[-1])
        v.add_point_cloud(f"pc_{i:04d}", pc.cpu().numpy())
        v.add_frame(f"pose_{i:04d}", surface_lf.pose.cpu().numpy())
        v.add_frame(f"pose_gt_{i:04d}", poses_gt[i].cpu().numpy())
        v.add_frame(f"pose_integrated{i:04d}", poses_gt[i].cpu().numpy())
    v.run()
