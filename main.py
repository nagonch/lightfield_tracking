from src.dataset import LFDataset
from time import time
from PIL import Image
import os
from segmentor import Segmentor
from depth_anything_3.api import DepthAnything3
import torch
from da3_functions import da3_run_from_tensors
from src.utilities import backproject_depth_to_pointcloud, Visualizer
import numpy as np
from src.disparity import get_LF_disparity
from depth_estimator import DepthEstimator
from surface_lf import SurfaceLF


def main():
    dataset = LFDataset(
        "/home/ngoncharov/cvpr2026/datasets/LiFT_dataset/jug_motion_prod"
    )
    s_size, t_size = dataset.metadata["n_views"]
    segmentor = Segmentor(prompt="shiny metal jug.")
    depth_estimator = DepthEstimator(infer_gs=False)

    v = Visualizer()
    for i, frame in enumerate(dataset):
        if i == 0:
            surface_lf = SurfaceLF(
                K=torch.clone(frame["camera_matrix"]),
                poses_4x4=torch.clone(frame["camera_poses_rel"].reshape(-1, 4, 4)),
                image_size_hw=frame["LF"].shape[2:4],
            )
        img_central = frame["LF"][s_size // 2, t_size // 2]

        camera_matrix = frame["camera_matrix"]
        depth = depth_estimator(frame)
        mask = segmentor(img_central)
        if i % 2 == 0:
            depth = depth[depth.shape[0] // 2]
            color = img_central[mask > 0].reshape(-1, 3)
            depth = depth * (mask > 0)
            pc = backproject_depth_to_pointcloud(
                pixel_indices=None,
                depths=depth,
                camera_matrix=camera_matrix,
            )
            pc = pc[(mask > 0).reshape(-1)]
            surface_lf.get_points_directions(
                points_world=pc,
                images=frame["LF"]
                .reshape(-1, *frame["LF"].shape[2:])
                .permute(0, 3, 1, 2),
            )

            depth_gt = frame["depth"]
            depth_gt = depth_gt * (mask > 0)
            pc_gt = backproject_depth_to_pointcloud(
                pixel_indices=None,
                depths=depth_gt,
                camera_matrix=camera_matrix,
            )
            pc_gt = pc_gt[(mask > 0).reshape(-1)]

            v.add_point_cloud(
                f"testpc_{i}", points=pc.cpu().numpy(), colors=color.cpu().numpy()
            )
            v.add_point_cloud(
                f"testpc_{i}_gt", points=pc_gt.cpu().numpy(), colors=color.cpu().numpy()
            )
    v.run()


if __name__ == "__main__":
    main()
