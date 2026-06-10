from dataloaders import YCBV_LF, LIFT, EOAT
import numpy as np
from utils import rotation_angle_deg, add_err, adds_err
from sklearn.metrics import auc
import os
from utils import project_frame_to_image
from PIL import Image
from tqdm import tqdm
import json


class Evaluator:
    def __init__(self, exp_name, dataset, est_poses_npy, gt_poses_npy=None):
        self.exp_name = exp_name
        self.save_folder = f"eval/{exp_name}"
        os.makedirs(self.save_folder, exist_ok=True)
        self.dataset = dataset
        self.est_poses = est_poses_npy
        if gt_poses_npy is not None:
            self.gt_poses = gt_poses_npy
        else:
            self.gt_poses = [frame["pose"] for frame in dataset]

    def add_metrics(self, threshold_max=0.1):
        thresholds_space = np.linspace(0, threshold_max, 100)
        gt_pc = self.dataset.gt_mesh.vertices.copy()
        adds_vals = []
        add_vals = []
        i = 0
        for gt_pose, est_pose in tqdm(
            zip(self.gt_poses, self.est_poses),
            total=len(self.gt_poses),
            desc="Computing ADD/ADDS metrics",
        ):
            add_val = add_err(est_pose, gt_pose, gt_pc)
            adds_val = adds_err(est_pose, gt_pose, gt_pc)
            add_vals.append(add_val)
            adds_vals.append(adds_val)
        adds_vals = np.array(adds_vals)
        add_vals = np.array(add_vals)
        adds_accuracies = [(adds_vals < t).mean() for t in thresholds_space]
        add_accuracies = [(add_vals < t).mean() for t in thresholds_space]
        adds_auc = auc(np.linspace(0, 1, 100), adds_accuracies)
        add_auc = auc(np.linspace(0, 1, 100), add_accuracies)
        return {"adds": adds_auc, "add": add_auc}

    def pose_errors(self):
        gt_poses = np.array(self.gt_poses)
        est_poses = np.array(self.est_poses)
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

    def visualize_tracking(self):
        os.makedirs(f"{self.save_folder}/vis", exist_ok=True)
        frames = [frame["rgb"] for frame in self.dataset]
        camera_matrix = self.dataset.camera_intrinsics
        for i, (frame, gt_pose, est_pose) in enumerate(
            zip(frames, self.gt_poses, self.est_poses)
        ):
            img_vis = project_frame_to_image(est_pose, camera_matrix, frame)
            img_vis_gt = project_frame_to_image(gt_pose, camera_matrix, frame)
            Image.fromarray(img_vis).save(
                f"{self.save_folder}/vis/{str(i).zfill(4)}_pred.png"
            )
            Image.fromarray(img_vis_gt).save(
                f"{self.save_folder}/vis/{str(i).zfill(4)}_gt.png"
            )

    def eval(self):
        # add_metrics = self.add_metrics()
        add_metrics = {}
        pose_errors = self.pose_errors()
        metrics = {**add_metrics, **pose_errors}
        with open(f"{self.save_folder}/metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        self.visualize_tracking()
        return metrics


if __name__ == "__main__":
    dataset = YCBV_LF(
        "/home/nagonch/repos/MonoGS/datasets/ycbv_lf/bleach_hard_00_03_chaitanya"
    )
    exp_name = "test"
    gt_poses_npy = "/home/nagonch/repos/MonoGS/results/ycbv_lf_bleach_hard_00_03_chaitanya/2026-02-14-11-31-27/est.npy"
    est_poses_npy = "/home/nagonch/repos/MonoGS/results/ycbv_lf_bleach_hard_00_03_chaitanya/2026-02-14-11-31-27/gt.npy"

    gt_poses = np.load(gt_poses_npy)
    est_poses = np.load(est_poses_npy)

    evaluator = Evaluator(exp_name, dataset, est_poses, gt_poses)
    print(evaluator.eval())
