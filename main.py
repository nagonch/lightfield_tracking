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
from tracking import Tracker


def rebase_poses(gt_poses, est_poses):
    pose_est_0 = est_poses[0]
    pose_gt_0 = gt_poses[0]
    est_to_gt = np.linalg.inv(pose_est_0) @ pose_gt_0
    est_poses = [p @ est_to_gt for p in est_poses]
    est_poses = np.stack(est_poses, axis=0)

    return est_poses


def main():
    dataset = LFDataset(
        "/home/ngoncharov/cvpr2026/datasets/LiFT_dataset/box_motion_prod"
    )
    s_size, t_size = dataset.metadata["n_views"]
    segmentor = Segmentor(prompt="shiny metal jug.")
    depth_estimator = DepthEstimator(infer_gs=False)
    gt_poses = []
    est_poses = []
    v = Visualizer()
    for i, frame in enumerate(dataset):
        img_central = frame["LF"][s_size // 2, t_size // 2]

        camera_matrix = frame["camera_matrix"]
        depth = depth_estimator(frame)
        mask = segmentor(img_central)
        depth = depth[depth.shape[0] // 2]
        color = img_central[mask > 0].reshape(-1, 3)
        depth = depth * (mask > 0)
        pc = backproject_depth_to_pointcloud(
            pixel_indices=None,
            depths=depth,
            camera_matrix=camera_matrix,
        )
        pc = pc[(mask > 0).reshape(-1)]
        if i == 0:
            surface_lf = SurfaceLF(
                K=torch.clone(frame["camera_matrix"]),
                poses_4x4=torch.clone(frame["camera_poses_rel"].reshape(-1, 4, 4)),
                image_size_hw=frame["LF"].shape[2:4],
            )
            tracker = Tracker(pc, color)
            est_poses.append(np.eye(4))
        else:
            pose = tracker.track(pc, color)
            est_poses.append(pose)
        gt_poses.append(frame["object_pose"].cpu().numpy())
        torch.save(
            {"pc": pc, "color": color, "gt_pose": frame["object_pose"]},
            f"frame_{i:04d}.pt",
        )
        v.add_point_cloud(f"pc_{i:04d}", pc.cpu().numpy(), color.cpu().numpy())
    est_poses = np.stack(est_poses, axis=0)
    gt_poses = np.stack(gt_poses, axis=0)
    est_poses = rebase_poses(gt_poses, est_poses)
    for i, (pose, gt_pose) in enumerate(zip(est_poses, gt_poses)):
        v.add_frame(f"frame_{i:04d}", pose)
        v.add_frame(f"frame_{i:04d}_gt", gt_pose)
    v.run()


if __name__ == "__main__":
    main()
