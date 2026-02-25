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
    total_sh_degrees = int((colors.shape[1] // 3) ** 0.5) - 1
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
    def __init__(self, rig: SurfaceLFRig, pc, images):
        self.rig = rig
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

    def rasterize(self):
        image, depth = batch_rasterize(
            points=self.values["means"].float(),
            quats=self.values["rotations"].float(),
            scales=self.values["scales"].float(),
            opacities=self.values["opacities"].float(),
            colors=self.values["harmonics"].float(),
            poses=torch.eye(4).unsqueeze(0).cuda(),
            camera_matrix=self.K,
            height=self.H,
            width=self.W,
        )
        # depth[image.sum(axis=-1) == 0] = 0
        # depth = (1 / (depth + 1e-8)).cpu().numpy()
        # depth = np.clip(depth, 0, np.percentile(depth, 75))
        # depth_img = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        Image.fromarray((image * 255).cpu().numpy().astype(np.uint8)).save(
            "rendered.png"
        )


if __name__ == "__main__":
    K = torch.load("K.pt")
    poses = torch.load("poses_4x4.pt")
    pc = torch.load("pc.pt")
    images = torch.load("images.pt")
    surface_lf_rig = SurfaceLFRig.build(
        K=K,
        poses_4x4=poses,
        image_size_hw=images.shape[2:4],
    )
    surface_lf = SurfaceLF(rig=surface_lf_rig, pc=pc, images=images)
    times = []
    for i in tqdm(range(1000)):
        start = time()
        surface_lf.rasterize()
        times.append(time() - start)
    print(f"Average fps: {1.0 / (sum(times) / len(times))}")
