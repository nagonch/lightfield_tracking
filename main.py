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
from surface_lf import SurfaceLF, SurfaceLFRig
from tracking import Tracker, pose_errors
from PIL import Image
from optimizer import refine_pose, refine_pose_nuclear_rotation_multistart


def rebase_poses(gt_poses, est_poses):
    pose_est_0 = est_poses[0]
    pose_gt_0 = gt_poses[0]
    est_to_gt = np.linalg.inv(pose_est_0) @ pose_gt_0
    est_poses = [p @ est_to_gt for p in est_poses]
    est_poses = np.stack(est_poses, axis=0)

    return est_poses


def main():
    dataset = LFDataset("/home/ngoncharov/cvpr2026/datasets/LiFT_dataset/jug_tilt_prod")
    s_size, t_size = dataset.metadata["n_views"]
    segmentor = Segmentor(prompt="shiny metal jug.")
    depth_estimator = DepthEstimator(infer_gs=False)
    gt_poses = []
    est_poses_coarse = []
    est_poses = []
    poses_gt = [frame["object_pose"] for frame in dataset]
    poses_gt = torch.stack(poses_gt, dim=0)
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
        LF = frame["LF"]
        LF_perm = LF.reshape(-1, LF.shape[2], LF.shape[3], 3)
        LF_perm = LF_perm.permute(0, 3, 1, 2)
        if i == 0:
            surface_lf_rig = SurfaceLFRig.build(
                K=torch.clone(frame["camera_matrix"]),
                poses_4x4=torch.clone(frame["camera_poses_rel"].reshape(-1, 4, 4)),
                image_size_hw=frame["LF"].shape[2:4],
            )
            tracker = Tracker(pc, color, pose0=frame["object_pose"])
            pose = torch.tensor(frame["object_pose"], dtype=torch.float32).cuda()
            est_poses.append(pose.cpu().numpy())
            est_poses_coarse.append(pose.cpu().numpy())
        else:
            pose = torch.tensor(tracker.track(pc, color), dtype=torch.float32).cuda()
            pose_rel = pose @ torch.linalg.inv(
                torch.tensor(est_poses[-1], dtype=torch.float32).cuda()
            )
            est_poses_coarse.append(pose.cpu().numpy())
        surface_lf = SurfaceLF(
            rig=surface_lf_rig, pc=pc, images=LF_perm, current_pose=pose
        )
        image, depth = surface_lf.rasterize()
        if i > 0:
            pose_refined, final_loss = refine_pose(
                surface_lf_prev, image, depth, pose_rel, i, mask_prev, mask
            )
            pose_refined, final_loss = refine_pose_nuclear_rotation_multistart(
                surface_lf_prev,
                image,
                depth,
                pose_coarse=pose_refined,
                pose_init=pose_refined,
                i=i,
                mask_prev=mask_prev,
                mask=mask,
                init_loss=final_loss,
                num_samples=96,
                max_angle_deg=25.0,
                topk=5,
                refine_topk=3,
                adam_iters=25,
                lr_rotation=2e-2,
                lr_translation=0.0,
            )
            est_poses.append(pose_refined.cpu().numpy() @ est_poses[-1])

        gt_poses.append(frame["object_pose"].cpu().numpy())
        surface_lf_prev = surface_lf
        mask_prev = mask

        v.add_point_cloud(f"pc_{i:04d}", pc.cpu().numpy(), color.cpu().numpy())
    est_poses = np.stack(est_poses, axis=0)
    est_poses_coarse = np.stack(est_poses_coarse, axis=0)
    gt_poses = np.stack(gt_poses, axis=0)
    est_poses_coarse = rebase_poses(gt_poses, est_poses_coarse)
    est_poses = rebase_poses(gt_poses, est_poses)
    print("Pose errors (coarse):", pose_errors(est_poses_coarse, gt_poses))
    print("Pose errors (refined):", pose_errors(est_poses, gt_poses))

    for i, (pose, gt_pose) in enumerate(zip(est_poses, gt_poses)):
        v.add_frame(f"frame_{i:04d}", pose)
        v.add_frame(f"frame_{i:04d}_gt", gt_pose)
    v.run()


if __name__ == "__main__":
    main()
