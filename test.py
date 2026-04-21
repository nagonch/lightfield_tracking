import torch
import numpy as np
from PIL import Image
from src.dataset import LFDataset
from src.utilities import backproject_depth_to_pointcloud
from surface_lf import SurfaceLF, SurfaceLFRig


def srgb_to_linear(srgb: torch.Tensor) -> torch.Tensor:
    """
    srgb: float tensor in [0,1]
    returns linear float tensor in [0,1]
    """
    cutoff = 0.04045
    below = srgb <= cutoff
    linear = torch.empty_like(srgb)
    linear[below] = srgb[below] / 12.92
    linear[~below] = ((srgb[~below] + 0.055) / 1.055) ** 2.4
    return linear


def linear_to_srgb(linear: torch.Tensor) -> torch.Tensor:
    """
    linear: float tensor in [0,1]
    returns srgb float tensor in [0,1]
    """
    cutoff = 0.0031308
    below = linear <= cutoff
    srgb = torch.empty_like(linear)
    srgb[below] = linear[below] * 12.92
    srgb[~below] = 1.055 * (linear[~below] ** (1.0 / 2.4)) - 0.055
    return srgb


def build_surface_lf_first_frame(dataset: LFDataset):
    frame = dataset[0]
    s_size, t_size = dataset.metadata["n_views"]

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
    image, depth_rendered, target_mask = surface_lf.rasterize(torch.eye(4).cuda())
    return surface_lf, image, depth_rendered, target_mask


if __name__ == "__main__":
    sequence_name = "bleach0"

    MIDDLE_REFLECTIVITY = 0.7

    path_diffuse = f"/home/ngoncharov/cvpr2026/ycbv-eoat-lf/dataset_simple_box_reflective_full_0.0/{sequence_name}"
    path_reflective = f"/home/ngoncharov/cvpr2026/ycbv-eoat-lf/dataset_simple_box_reflective_full_1.0/{sequence_name}"
    path_middle = f"/home/ngoncharov/cvpr2026/ycbv-eoat-lf/dataset_simple_box_reflective_full_{MIDDLE_REFLECTIVITY}/{sequence_name}"

    dataset_diffuse = LFDataset(path_diffuse)
    dataset_reflective = LFDataset(path_reflective)
    dataset_middle = LFDataset(path_middle)

    lf_diffuse_0 = dataset_diffuse[0]["LF"][
        dataset_diffuse.metadata["n_views"][0] // 2,
        dataset_diffuse.metadata["n_views"][1] // 2,
    ]
    lf_reflective_0 = dataset_reflective[0]["LF"][
        dataset_diffuse.metadata["n_views"][0] // 2,
        dataset_diffuse.metadata["n_views"][1] // 2,
    ]
    lf_middle_0 = dataset_middle[0]["LF"][
        dataset_diffuse.metadata["n_views"][0] // 2,
        dataset_diffuse.metadata["n_views"][1] // 2,
    ]
    mask = dataset_diffuse[0]["masks"][
        dataset_diffuse.metadata["n_views"][0] // 2,
        dataset_diffuse.metadata["n_views"][1] // 2,
    ]

    lf_diffuse_0[mask == 0] = 0
    lf_reflective_0[mask == 0] = 0
    lf_middle_0[mask == 0] = 0

    surface_lf_diffuse, image_diffuse, depth_diffuse, mask_diffuse = (
        build_surface_lf_first_frame(dataset_diffuse)
    )
    surface_lf_reflective, image_reflective, depth_reflective, mask_reflective = (
        build_surface_lf_first_frame(dataset_reflective)
    )
    surface_lf_middle, image_middle, depth_middle, mask_middle = (
        build_surface_lf_first_frame(dataset_middle)
    )

    colors_middle = surface_lf_middle.colors.permute(1, 0, 2)
    object_mask = torch.clone(mask)
    depth_map = torch.clone(depth_diffuse)
    depth_map[mask == 0] = 0
    color_map = torch.zeros(
        (*mask.shape, colors_middle.shape[1], colors_middle.shape[2]),
        device=colors_middle.device,
    )
    color_map[mask > 0] = colors_middle

    colors_diffuse = surface_lf_diffuse.colors.permute(1, 0, 2)
    color_map_diffuse = torch.zeros(
        (*mask.shape, colors_diffuse.shape[1], colors_diffuse.shape[2]),
        device=colors_diffuse.device,
    )
    color_map_diffuse[mask > 0] = colors_diffuse

    colors_reflective = surface_lf_reflective.colors.permute(1, 0, 2)
    color_map_reflective = torch.zeros(
        (*mask.shape, colors_reflective.shape[1], colors_reflective.shape[2]),
        device=colors_reflective.device,
    )
    color_map_reflective[mask > 0] = colors_reflective

    print(color_map_diffuse.shape)
    print(color_map_reflective.shape)
    print(color_map.shape)
