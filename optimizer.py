from surface_lf import SurfaceLF, SurfaceLFRig
import torch
from PIL import Image
import numpy as np
import os
import torch.nn.functional as F
from loss import loss


def refine_pose(surface_lf_prev, image, depth, pose_coarse, i, mask_prev, mask):
    surf_values = surface_lf_prev.transform(pose_coarse)
    surf_image, surf_depth = surface_lf_prev.rasterize(surf_values)
    loss(
        surf_image,
        surf_depth,
        mask_prev,
        image,
        depth,
        mask,
        pose_coarse,
        pose_coarse,
        i=i,
    )


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
    for i in range(1, 20):
        pose_rel = poses[i] @ torch.linalg.inv(poses[i - 1])
        poses_rel.append(pose_rel)
    for i in range(20):
        surface_lf = SurfaceLF(
            rig=surface_lf_rig,
            pc=torch.load(f"pc_{i:04d}.pt"),
            images=torch.load(f"images_{i:04d}.pt"),
        )
        mask = torch.load(f"mask_{i:04d}.pt")
        # values = surface_lf.transform(pose_coarse)
        image, depth = surface_lf.rasterize()
        if i > 0:
            pose_coarse = refine_pose(
                surface_lf_prev, image, depth, poses_rel[i], i, mask_prev, mask
            )
        surface_lf_prev = surface_lf
        mask_prev = mask
