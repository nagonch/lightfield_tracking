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
from tracking import Tracker
from PIL import Image


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
    segmentor = Segmentor(prompt="white and blue box.")
    depth_estimator = DepthEstimator(infer_gs=False)
    gt_poses = []
    est_poses = []
    poses_gt = [frame["object_pose"] for frame in dataset]
    poses_gt = torch.stack(poses_gt, dim=0)
    torch.save(poses_gt, "pts/poses_gt.pt")
    raise
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
            est_poses.append(pose)
        else:
            pose = torch.tensor(tracker.track(pc, color), dtype=torch.float32).cuda()
            est_poses.append(pose)
        # torch.save(pose, f"pts/coarse_pose_{i:04d}.pt")
        # torch.save(frame["object_pose"], f"pts/pose_gt{i:04d}.pt")
        # torch.save(mask, f"pts/mask_{i:04d}.pt")
        # torch.save(torch.clone(frame["camera_matrix"]), "pts/K.pt")
        # torch.save(
        #     torch.clone(frame["camera_poses_rel"].reshape(-1, 4, 4)), "pts/poses_4x4.pt"
        # )
        # torch.save(pc, f"pts/pc_{i:04d}.pt")
        # torch.save(LF_perm, f"pts/images_{i:04d}.pt")
        # Image.fromarray((mask.cpu().numpy() * 255).astype(np.uint8)).save(
        #     f"pts/mask_{i:04d}.png"
        # )
        # Image.fromarray(
        #     (
        #         LF_perm.permute(0, 2, 3, 1)[LF_perm.shape[0] // 2].cpu().numpy() * 255
        #     ).astype(np.uint8)
        # ).save(f"pts/image_{i:04d}.png")
        surface_lf = SurfaceLF(
            rig=surface_lf_rig, pc=pc, images=LF_perm, current_pose=pose
        )
        gt_poses.append(frame["object_pose"].cpu().numpy())
        surface_lf_prev = surface_lf
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
