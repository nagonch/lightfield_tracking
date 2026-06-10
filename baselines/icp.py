import os
import numpy as np
import open3d as o3d
from dataloaders import EOAT, YCBV_LF, LIFT
from utils import rotation_angle_deg
from eval import Evaluator
from tqdm import tqdm


def pose_errors(gt_poses, est_poses):
    assert gt_poses.shape == est_poses.shape
    assert gt_poses.shape[-2:] == (4, 4)

    N = gt_poses.shape[0]

    R_gt = gt_poses[:, :3, :3]
    t_gt = gt_poses[:, :3, 3]
    R_est = est_poses[:, :3, :3]
    t_est = est_poses[:, :3, 3]
    R_err_abs = R_est @ np.transpose(R_gt, (0, 2, 1))  # (N, 3, 3)
    rot_err_abs = rotation_angle_deg(R_err_abs)  # (N,)
    trans_err_abs = np.linalg.norm(t_est - t_gt, axis=1)  # (N,)
    ate_rmse = np.sqrt((trans_err_abs**2).mean())

    rel_rot_errs = []
    rel_trans_errs = []
    for i in range(N - 1):
        T_gt_rel = np.linalg.inv(gt_poses[i]) @ gt_poses[i + 1]
        T_est_rel = np.linalg.inv(est_poses[i]) @ est_poses[i + 1]

        R_gt_rel = T_gt_rel[:3, :3]
        R_est_rel = T_est_rel[:3, :3]
        t_gt_rel = T_gt_rel[:3, 3]
        t_est_rel = T_est_rel[:3, 3]

        R_err_rel = R_est_rel @ R_gt_rel.transpose(-1, -2)
        rel_rot_errs.append(rotation_angle_deg(R_err_rel))
        rel_trans_errs.append(np.linalg.norm(t_est_rel - t_gt_rel))

    rel_rot_errs = np.array(rel_rot_errs)
    rel_trans_errs = np.array(rel_trans_errs)
    return {
        "mean_abs_rot_deg": rot_err_abs.mean().item(),
        "mean_abs_trans": trans_err_abs.mean().item(),
        "mean_rel_rot_deg": rel_rot_errs.mean().item(),
        "mean_rel_trans": rel_trans_errs.mean().item(),
        "ate_rmse": ate_rmse.item(),
    }


class ColoredICPTracker:
    def __init__(self, dataset, voxel_size=0.005, vis=False):
        self.dataset = dataset
        self.gt_poses = np.stack([sample["pose"] for sample in dataset])
        self.voxel_size = voxel_size
        self.model = o3d.geometry.PointCloud()
        self.criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=50
        )
        self.vis = vis

    def get_o3d_pc(self, rgb, depth, mask, camera_matrix):
        inds = np.stack(np.where(mask)).T
        rgbs = rgb[mask].astype(np.float64) / 255.0
        depths = depth[mask].astype(np.float64)

        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

        x = (inds[:, 1] - cx) * depths / fx
        y = (inds[:, 0] - cy) * depths / fy
        z = depths

        pts = np.stack([x, y, z], axis=-1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(rgbs)

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.voxel_size * 2, max_nn=30
            )
        )
        return pcd.voxel_down_sample(self.voxel_size)

    def register(self, idx, source, target, initial_guess):
        estimation = o3d.pipelines.registration.TransformationEstimationForColoredICP(
            lambda_geometric=0.968
        )
        try:
            result = o3d.pipelines.registration.registration_colored_icp(
                source,
                target,
                max_correspondence_distance=self.voxel_size * 20,
                init=initial_guess,
                estimation_method=estimation,
                criteria=self.criteria,
            ).transformation
        except Exception as e:
            print(f"ICP registration failed: {e}")
            result = initial_guess
        return result

    def run(self):
        poses_est = []
        camera_matrix = self.dataset.camera_intrinsics
        if self.vis:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Live ICP Tracking", width=1280, height=720)

        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            pcd_cur = self.get_o3d_pc(
                sample["rgb"], sample["depth"], sample["mask"], camera_matrix
            )

            if idx == 0:
                current_pose = np.eye(4)
                self.model = pcd_cur
                if self.vis:
                    vis.add_geometry(self.model)
            else:
                if idx == 1:
                    initial_guess_rel = np.eye(4)
                else:
                    initial_guess_rel = poses_est[-1] @ np.linalg.inv(poses_est[-2])
                pose_rel = self.register(idx, self.model, pcd_cur, initial_guess_rel)
                if self.vis:
                    vis.clear_geometries()
                    vis.add_geometry(self.model)
                    vis.add_geometry(pcd_cur)
                self.model = pcd_cur
                current_pose = pose_rel @ poses_est[-1]
            if self.vis:
                vis.poll_events()
                vis.update_renderer()

            poses_est.append(current_pose)
            print(f"Processed frame {idx+1}/{len(self.dataset)}", end="\r")
        if self.vis:
            print("\nTracking complete. Keeping window open...")
            vis.run()
            vis.destroy_window()

        # FOR EVALUATION
        gt_poses = np.copy(self.gt_poses)
        est_poses = np.stack(poses_est, axis=0)
        pose_est_0 = est_poses[0]
        pose_gt_0 = gt_poses[0]
        est_to_gt = np.linalg.inv(pose_est_0) @ pose_gt_0
        est_poses = [p @ est_to_gt for p in est_poses]
        est_poses = np.stack(est_poses, axis=0)

        return est_poses


if __name__ == "__main__":
    pass
