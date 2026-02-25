from surface_lf import SurfaceLF, SurfaceLFRig
import torch
from PIL import Image
import numpy as np


def loss(image_from, depth_from, image_to, depth_to, pose_coarse, pose_refined, i):
    depth_loss = (depth_from - depth_to) ** 2
    image_loss = (image_from - image_to) ** 2

    depth_loss_image = (depth_loss - depth_loss.min()) / (
        depth_loss.max() - depth_loss.min() + 1e-8
    )
    image_loss_image = (image_loss - image_loss.min()) / (
        image_loss.max() - image_loss.min() + 1e-8
    )
    Image.fromarray((image_loss_image.cpu().numpy() * 255).astype(np.uint8)).save(
        f"losses/image_loss_{i:04d}.png"
    )
    Image.fromarray((depth_loss_image.cpu().numpy() * 255).astype(np.uint8)).save(
        f"losses/depth_loss{i:04d}.png"
    )
    Image.fromarray((image_to.cpu().numpy() * 255).astype(np.uint8)).save(
        f"losses/target_image_{i:04d}.png"
    )
    Image.fromarray((image_from.cpu().numpy() * 255).astype(np.uint8)).save(
        f"losses/rendered_image_{i:04d}.png"
    )


def refine_pose(surface_lf_prev, image, depth, pose_coarse, i):
    surf_values = surface_lf_prev.transform(pose_coarse)
    surf_image, surf_depth = surface_lf_prev.rasterize()
    loss(surf_image, surf_depth, image, depth, pose_coarse, pose_coarse, i)


if __name__ == "__main__":
    K = torch.load("K.pt")
    poses = torch.load("poses_4x4.pt")
    surface_lf_rig = SurfaceLFRig.build(
        K=K,
        poses_4x4=poses,
        image_size_hw=(720, 1280),
    )
    poses = [
        torch.load(f"coarse_pose_{i:04d}.pt", weights_only=True) for i in range(20)
    ]
    poses_rel = [torch.eye(4).cuda()]
    for i in range(20):
        pose_coarse = torch.load(f"coarse_pose_{i:04d}.pt", weights_only=True)
        surface_lf = SurfaceLF(
            rig=surface_lf_rig,
            pc=torch.load(f"pc_{i:04d}.pt"),
            images=torch.load(f"images_{i:04d}.pt"),
        )
        # values = surface_lf.transform(pose_coarse)
        image, depth = surface_lf.rasterize()
        if i > 0:
            pose_coarse = refine_pose(surface_lf_prev, image, depth, pose_coarse, i)
        surface_lf_prev = surface_lf
