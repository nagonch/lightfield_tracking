import numpy as np
import open3d as o3d
from src.utilities import Visualizer
import torch
from icp import rebase_poses, pose_errors


def rel_pose_coarse(pc, pc_prev, color, color_prev, pose_prev, pose_rel_prev=None):
    translation_guess = np.mean(pc, axis=0) - np.mean(pc_prev, axis=0)
    pc_prev += translation_guess
    pose_rel = np.eye(4)
    pose_rel[:3, 3] = translation_guess
    pose = pose_rel @ pose_prev
    pose[:3, 3] = np.mean(pc, axis=0)
    return pose, pose_rel


def get_pose(pc):
    pose = np.eye(4)
    pose[:3, 3] = np.mean(pc, axis=0)
    return pose


if __name__ == "__main__":
    gt_poses = []
    est_poses = []
    v = Visualizer()
    pose_rel_prev = None
    for i in range(40):
        print(i)
        frame = torch.load(f"pcs_bottle/frame_{str(i).zfill(4)}.pt")
        gt_poses.append(frame["pose"].cpu().numpy())
        if i == 0:
            pc_prev = frame["pc"].cpu().numpy()
            color_prev = frame["color"].cpu().numpy()
            est_poses.append(frame["pose"].cpu().numpy())
        else:
            pc = frame["pc"].cpu().numpy()
            color = frame["color"].cpu().numpy()
            # pose_prev = est_poses[-1]
            # pose_new_world, pose_rel_prev = rel_pose_coarse(
            #     pc,
            #     pc_prev,
            #     color,
            #     color_prev,
            #     pose_prev,
            #     pose_rel_prev=pose_rel_prev,
            # )
            est_poses.append(get_pose(pc))
            pc_prev = pc
            color_prev = color
            v.add_point_cloud(f"pc_{i}", pc, color, point_size=1e-3)
    gt_poses = np.stack(gt_poses, axis=0)
    est_poses = np.stack(est_poses, axis=0)
    est_poses = rebase_poses(gt_poses, est_poses)
    print(pose_errors(gt_poses, est_poses))
    for i, (pose_est, pose_gt) in enumerate(zip(est_poses, gt_poses)):
        v.add_frame(f"{i}_est", pose_est, frames_scale=0.01)
        v.add_frame(f"{i}_gt", pose_gt, frames_scale=0.01, origin_color=(255, 255, 255))
    v.run()
