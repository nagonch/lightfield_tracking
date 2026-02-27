import numpy as np
import open3d as o3d
from src.utilities import Visualizer


def rebase_poses(gt_poses, est_poses):
    pose_est_0 = est_poses[0]
    pose_gt_0 = gt_poses[0]
    est_to_gt = np.linalg.inv(pose_est_0) @ pose_gt_0
    est_poses = [p @ est_to_gt for p in est_poses]
    est_poses = np.stack(est_poses, axis=0)

    return est_poses


def rotation_angle_deg(R_err):
    trace = np.trace(R_err, axis1=-2, axis2=-1)
    cos_theta = np.clip((trace - 1) / 2, -1.0, 1.0)
    return np.arccos(cos_theta) * (180.0 / np.pi)


def pose_errors(gt_poses, est_poses):
    assert (
        gt_poses.shape == est_poses.shape
    ), f"GT poses shape {gt_poses.shape} does not match estimated poses shape {est_poses.shape}"
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


def icp_track(pc_curr, pc_prev, color_curr, color_prev, pose_prev, pose_rel_prev=None):
    voxel_size = 2e-3

    def process_pointcloud(points, colors):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 2.5,
                max_nn=30,
            )
        )
        pcd.orient_normals_consistent_tangent_plane(k=20)

        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 15.0,
                max_nn=200,
            ),
        )
        return pcd, fpfh

    def transform_to_origin(points, pose):
        pose_inv = np.linalg.inv(pose)
        points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        points_transformed = (pose_inv @ points_h.T).T[:, :3]
        return points_transformed

    def rotation_angle_deg_single(R):
        trace = np.trace(R)
        cos_theta = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))

    # --- move both to same frame (pose_prev-origin) ---
    pc_prev_origin = transform_to_origin(pc_prev, pose_prev)
    pc_curr_origin = transform_to_origin(pc_curr, pose_prev)

    pcd_prev, fpfh_prev = process_pointcloud(pc_prev_origin, color_prev)
    pcd_curr, fpfh_curr = process_pointcloud(pc_curr_origin, color_curr)

    # --- constant motion prior in this origin frame ---
    has_motion_prior = pose_rel_prev is not None
    T_motion_init = pose_rel_prev if has_motion_prior else np.eye(4)

    # --- optional: keep a global init as backup (RANSAC) ---
    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd_prev,
        pcd_curr,
        fpfh_prev,
        fpfh_curr,
        mutual_filter=True,
        max_correspondence_distance=voxel_size * 50,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            False
        ),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                voxel_size * 50
            ),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 0.999),
    )
    T_ransac_init = ransac_result.transformation

    # --- choose init: prefer motion prior; else ransac ---
    T_init = T_motion_init if has_motion_prior else T_ransac_init

    # --- refine with colored ICP ---
    icp_result = o3d.pipelines.registration.registration_colored_icp(
        source=pcd_prev,
        target=pcd_curr,
        max_correspondence_distance=voxel_size * 5.0,
        init=T_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=120,
        ),
    )

    T_rel_icp = icp_result.transformation

    # --- quality / sanity gating ---
    # Tune these based on your sequence.
    min_fitness = 0.20
    max_inlier_rmse = voxel_size * 2.5
    max_trans_jump = 0.05  # 5 cm per frame
    max_rot_jump_deg = 20.0  # 20 deg per frame

    t_jump = float(np.linalg.norm(T_rel_icp[:3, 3]))
    rot_jump_deg = rotation_angle_deg_single(T_rel_icp[:3, :3])

    icp_good = (
        (icp_result.fitness >= min_fitness)
        and (icp_result.inlier_rmse <= max_inlier_rmse)
        and (t_jump <= max_trans_jump)
        and (rot_jump_deg <= max_rot_jump_deg)
    )

    if icp_good:
        T_rel = T_rel_icp
    else:
        # fallback 1: if we had a motion prior, trust it
        if has_motion_prior:
            T_rel = T_motion_init
        else:
            # fallback 2: try ICP from ransac init (if we started from something else)
            icp_ransac = o3d.pipelines.registration.registration_colored_icp(
                source=pcd_prev,
                target=pcd_curr,
                max_correspondence_distance=voxel_size * 5.0,
                init=T_ransac_init,
                estimation_method=o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6,
                    relative_rmse=1e-6,
                    max_iteration=120,
                ),
            )
            T_rel = icp_ransac.transformation

    # --- lift back to world ---
    T_world = pose_prev @ T_rel
    return T_world, T_rel


import torch

if __name__ == "__main__":
    gt_poses = []
    est_poses = []
    v = Visualizer()
    pose_rel_prev = None
    for i in range(20):
        print(i)
        frame = torch.load(f"frame_{str(i).zfill(4)}.pt")
        gt_poses.append(frame["pose"].cpu().numpy())
        if i == 0:
            pc_prev = frame["pc"].cpu().numpy()
            color_prev = frame["color"].cpu().numpy()
            est_poses.append(frame["pose"].cpu().numpy())
        else:
            pc = frame["pc"].cpu().numpy()
            color = frame["color"].cpu().numpy()
            pose_prev = est_poses[-1]
            pose_new_world, pose_rel_prev = icp_track(
                pc,
                pc_prev,
                color,
                color_prev,
                pose_prev,
                pose_rel_prev=pose_rel_prev,
            )
            est_poses.append(pose_new_world)
            pc_prev = pc
            color_prev = color
    gt_poses = np.stack(gt_poses, axis=0)
    est_poses = np.stack(est_poses, axis=0)
    est_poses = rebase_poses(gt_poses, est_poses)
    print(pose_errors(gt_poses, est_poses))
    for i, (pose_est, pose_gt) in enumerate(zip(est_poses, gt_poses)):
        v.add_frame(f"{i}_est", pose_est)
        v.add_frame(f"{i}_gt", pose_gt)
    v.run()
