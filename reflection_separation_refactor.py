import torch
import numpy as np
from PIL import Image
from src.dataset import LFDataset
from src.utilities import backproject_depth_to_pointcloud
from surface_lf import SurfaceLF, SurfaceLFRig
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
import os
from utils import srgb_to_linear, linear_to_srgb


def build_surface_lf(frame, s_size, t_size):
    mask = frame["masks"][s_size // 2, t_size // 2]
    depth = frame["depth"]
    camera_matrix = frame["camera_matrix"]

    pc, pc_scales = backproject_depth_to_pointcloud(
        pixel_indices=None,
        depths=depth,
        camera_matrix=camera_matrix,
        return_scales=True,
    )
    pc = pc[(mask > 0).reshape(-1)]
    pc_scales = pc_scales[(mask > 0).reshape(-1)]

    surface_lf_rig = SurfaceLFRig.build(
        K=frame["camera_matrix"],
        poses_4x4=frame["camera_poses_rel"].reshape(-1, 4, 4),
        image_size_hw=(
            frame["LF"].shape[2],
            frame["LF"].shape[3],
        ),
    )

    surface_lf = SurfaceLF(
        surface_lf_rig,
        pc,
        frame["LF"]
        .reshape(-1, frame["LF"].shape[2], frame["LF"].shape[3], 3)
        .permute(0, 3, 1, 2),
        pc_scales,
        previous_environment_map=None,
    )
    return surface_lf


if __name__ == "__main__":
    sequence_name = "bleach0"

    MIDDLE_REFLECTIVITY = 0.7
    ALPHA = 1 - MIDDLE_REFLECTIVITY

    vis_reflective_folder = f"vis_reflective_{MIDDLE_REFLECTIVITY}"
    vis_diffuse_folder = f"vis_diffuse_{MIDDLE_REFLECTIVITY}"
    os.makedirs(vis_reflective_folder, exist_ok=True)
    os.makedirs(vis_diffuse_folder, exist_ok=True)

    path_diffuse = f"/home/ngoncharov/cvpr2026/ycbv-eoat-lf/dataset_simple_box_reflective_full_0.0/{sequence_name}"
    path_reflective = f"/home/ngoncharov/cvpr2026/ycbv-eoat-lf/dataset_simple_box_reflective_full_1.0/{sequence_name}"
    path_middle = f"/home/ngoncharov/cvpr2026/ycbv-eoat-lf/dataset_simple_box_reflective_full_{MIDDLE_REFLECTIVITY}/{sequence_name}"

    dataset = LFDataset(path_middle)
    for i in range(len(dataset)):

        lf = dataset[i]["LF"][
            dataset.metadata["n_views"][0] // 2,
            dataset.metadata["n_views"][1] // 2,
        ]
        mask = dataset[i]["masks"][
            dataset.metadata["n_views"][0] // 2,
            dataset.metadata["n_views"][1] // 2,
        ]
        depth = dataset[i]["depth"]

        lf[mask == 0] = 0

        surface_lf = build_surface_lf(
            dataset[i],
            *dataset.metadata["n_views"],
        )
        surface_normals = surface_lf.surface_normals

        colors_middle = surface_lf.colors.permute(1, 0, 2)
        object_mask = torch.clone(mask)

        normal_map = torch.zeros(
            (*mask.shape, 3),
            device=surface_normals.device,
        )
        normal_map[mask > 0] = surface_normals.float()

        depth_map = torch.clone(depth)
        depth_map[mask == 0] = 0
        color_map = torch.zeros(
            (*mask.shape, colors_middle.shape[1], colors_middle.shape[2]),
            device=colors_middle.device,
        )
        color_map[mask > 0] = colors_middle

        color_map = srgb_to_linear(color_map)

        diffuse, reflective = separate_reflection(
            color_map_obs=color_map,
            alpha=ALPHA,
            mask=object_mask,
            normal_map=normal_map,
            depth_map=depth_map,
        )

        diffuse = linear_to_srgb(diffuse)
        reflective = linear_to_srgb(reflective)
        reflective = torch.clamp(reflective, 0.0, 1.0)

        diffuse_vis = (diffuse.cpu().numpy() * 255).astype(np.uint8)
        reflective_vis = (
            reflective[:, :, reflective.shape[2] // 2, :].cpu().numpy() * 255
        ).astype(np.uint8)

        Image.fromarray(diffuse_vis).save(
            f"{vis_diffuse_folder}/diffuse_estimate_{str(i).zfill(4)}.png"
        )
        Image.fromarray(reflective_vis).save(
            f"{vis_reflective_folder}/reflective_estimate_{str(i).zfill(4)}.png"
        )
