import torch
from pytorch3d.renderer.cameras import PerspectiveCameras
import torch.nn.functional as F
from PIL import Image


class SurfaceLF:
    def __init__(
        self,
        K: torch.Tensor,
        poses_4x4: torch.Tensor,
        image_size_hw: tuple[int, int],
        pose_is_cam2world: bool = True,
    ):
        self.K = K
        self.poses_4x4 = poses_4x4
        self.image_size_hw = image_size_hw
        self.pose_is_cam2world = pose_is_cam2world
        self.device = poses_4x4.device
        dtype = poses_4x4.dtype
        N = poses_4x4.shape[0]
        H, W = image_size_hw

        K = K.to(device=self.device, dtype=dtype)
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        R_pose = poses_4x4[:, :3, :3]  # [N,3,3]
        t_pose = poses_4x4[:, :3, 3]  # [N,3]

        if pose_is_cam2world:
            R = R_pose.transpose(1, 2)  # [N,3,3]
            T = -(R @ t_pose.unsqueeze(-1)).squeeze(-1)  # [N,3]
        else:
            R = R_pose
            T = t_pose

        R[:, 0, :] *= -1
        R[:, 1, :] *= -1
        T[:, 0] *= -1
        T[:, 1] *= -1

        focal_length = torch.stack([fx.expand(N), fy.expand(N)], dim=-1)
        principal_point = torch.stack([cx.expand(N), cy.expand(N)], dim=-1)
        image_size = torch.tensor([[H, W]], device=self.device, dtype=dtype).expand(
            N, -1
        )
        self.cameras = PerspectiveCameras(
            focal_length=focal_length,
            principal_point=principal_point,
            R=R,
            T=T,
            in_ndc=False,
            image_size=image_size,
            device=self.device,
        )

    def get_points_directions(
        self,
        points_world: torch.Tensor,
        images: torch.Tensor,
        eps: float = 1e-8,
    ):
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

        sampled = F.grid_sample(
            images, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        colors = sampled.squeeze(-1).permute(0, 2, 1).contiguous()
        colors = colors * valid.unsqueeze(-1).to(colors.dtype)

        cam_centers = self.cameras.get_camera_center()
        view_vec = points_rep - cam_centers[:, None, :]
        view_dirs = view_vec / (view_vec.norm(dim=-1, keepdim=True) + eps)

        return colors, view_dirs, valid


if __name__ == "__main__":
    pass
