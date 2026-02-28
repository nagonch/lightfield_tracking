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


def icp_track(
    pc_curr,
    pc_prev,
    color_curr,
    color_prev,
    pose_prev,
    pose_rel_prev=None,
):
    # ---- helpers ----
    def transform_to_origin(
        points_xyz: np.ndarray, pose_world: np.ndarray
    ) -> np.ndarray:
        pose_world_inv = np.linalg.inv(pose_world)
        points_h = np.concatenate(
            [points_xyz, np.ones((points_xyz.shape[0], 1))], axis=1
        )
        return (pose_world_inv @ points_h.T).T[:, :3]

    def rotation_angle_deg_single(R: np.ndarray) -> float:
        trace_val = float(np.trace(R))
        cos_theta = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))

    def preprocess(points_xyz: np.ndarray, colors_rgb: np.ndarray, voxel_size: float):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb)

        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

        # stronger normals than your original (you had radius ~ 2.5*voxel)
        normal_radius = voxel_size * 6.0
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=50)
        )
        # optional but usually helps stability
        pcd.orient_normals_consistent_tangent_plane(k=30)
        return pcd

    def compute_fpfh(pcd: o3d.geometry.PointCloud, voxel_size: float):
        feat_radius = voxel_size * 15.0
        return o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=feat_radius, max_nn=200),
        )

    def run_multiscale_icp(
        source_base: o3d.geometry.PointCloud,
        target_base: o3d.geometry.PointCloud,
        initial_transform: np.ndarray,
        voxel_scales: list[float],
    ):
        T = initial_transform.copy()

        for voxel_size in voxel_scales:
            src = source_base.voxel_down_sample(voxel_size)
            tgt = target_base.voxel_down_sample(voxel_size)

            # normals at this scale
            normal_radius = voxel_size * 6.0
            for pcd in (src, tgt):
                pcd.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(
                        radius=normal_radius, max_nn=50
                    )
                )
                pcd.orient_normals_consistent_tangent_plane(k=30)

            max_corr = voxel_size * 4.0

            # 1) colored ICP (good for small residuals + texture)
            try:
                colored = o3d.pipelines.registration.registration_colored_icp(
                    source=src,
                    target=tgt,
                    max_correspondence_distance=max_corr,
                    init=T,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                        relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=60
                    ),
                )
            except RuntimeError:
                return initial_transform.copy(), 0.0, float("inf")
            T = colored.transformation

            # 2) point-to-plane ICP refinement (this is the rotation driver)
            try:
                p2l = o3d.pipelines.registration.registration_icp(
                    source=src,
                    target=tgt,
                    max_correspondence_distance=max_corr,
                    init=T,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                        max_iteration=40
                    ),
                )
            except RuntimeError:
                return initial_transform.copy(), 0.0, float("inf")
            T = p2l.transformation

        # score on finest scale using point-to-plane ICP result fields (fitness/rmse)
        # (we re-run one quick eval ICP to get comparable metrics)
        voxel_finest = voxel_scales[-1]
        src_f = source_base.voxel_down_sample(voxel_finest)
        tgt_f = target_base.voxel_down_sample(voxel_finest)
        normal_radius = voxel_finest * 6.0
        for pcd in (src_f, tgt_f):
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=50)
            )
        try:
            eval_icp = o3d.pipelines.registration.registration_icp(
                source=src_f,
                target=tgt_f,
                max_correspondence_distance=voxel_finest * 4.0,
                init=T,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                    max_iteration=1
                ),
            )
        except RuntimeError:
            return initial_transform.copy(), 0.0, float("inf")
        return T, float(eval_icp.fitness), float(eval_icp.inlier_rmse)

    # ---- put both clouds into the same frame: pose_prev-origin ----
    pc_prev_origin = transform_to_origin(pc_prev, pose_prev)
    pc_curr_origin = transform_to_origin(pc_curr, pose_prev)

    # ---- multi-scale schedule (coarse -> fine) ----
    # tune these; start coarser if your motion is bigger
    voxel_finest = 0.002  # 2mm (your original)
    voxel_scales = [voxel_finest * 4.0, voxel_finest * 2.0, voxel_finest]

    # preprocess at finest (we downsample inside the pyramid)
    pcd_prev_base = preprocess(pc_prev_origin, color_prev, voxel_finest)
    pcd_curr_base = preprocess(pc_curr_origin, color_curr, voxel_finest)

    # ---- build candidate initializations ----
    has_motion_prior = pose_rel_prev is not None
    T_motion_init = pose_rel_prev if has_motion_prior else np.eye(4)

    # feature init (RANSAC) at coarse scale to help rotation when texture/geometry is ambiguous
    pcd_prev_r = pcd_prev_base.voxel_down_sample(voxel_finest * 4.0)
    pcd_curr_r = pcd_curr_base.voxel_down_sample(voxel_finest * 4.0)
    for pcd in (pcd_prev_r, pcd_curr_r):
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=(voxel_finest * 4.0) * 6.0, max_nn=50
            )
        )

    fpfh_prev = compute_fpfh(pcd_prev_r, voxel_finest * 4.0)
    fpfh_curr = compute_fpfh(pcd_curr_r, voxel_finest * 4.0)

    ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd_prev_r,
        pcd_curr_r,
        fpfh_prev,
        fpfh_curr,
        mutual_filter=True,
        max_correspondence_distance=(voxel_finest * 4.0) * 5.0,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            False
        ),
        ransac_n=4,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                (voxel_finest * 4.0) * 5.0
            ),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    T_ransac_init = ransac_result.transformation

    # Try both: motion prior + ransac, then keep better
    candidates = []
    candidates.append(("motion", T_motion_init))
    candidates.append(("ransac", T_ransac_init))

    best = None
    for name, T_init in candidates:
        T_refined, fitness, rmse = run_multiscale_icp(
            source_base=pcd_prev_base,
            target_base=pcd_curr_base,
            initial_transform=T_init,
            voxel_scales=voxel_scales,
        )
        # score: prefer high fitness, then low rmse
        score = (fitness, -rmse)
        if (best is None) or (score > best["score"]):
            best = {
                "name": name,
                "T": T_refined,
                "fitness": fitness,
                "rmse": rmse,
                "score": score,
            }

    T_rel_icp = best["T"]

    # ---- sanity gating (loosen rotation gate) ----
    min_fitness = 0.20
    max_inlier_rmse = voxel_finest * 3.0

    max_trans_jump = 0.10  # 10 cm per frame (was 5 cm)
    max_rot_jump_deg = 60.0  # was 20 deg

    t_jump = float(np.linalg.norm(T_rel_icp[:3, 3]))
    rot_jump_deg = rotation_angle_deg_single(T_rel_icp[:3, :3])

    icp_good = (
        (best["fitness"] >= min_fitness)
        and (best["rmse"] <= max_inlier_rmse)
        and (t_jump <= max_trans_jump)
        and (rot_jump_deg <= max_rot_jump_deg)
    )

    if not icp_good:
        # fallback: trust motion prior if available, else keep ICP anyway (you can choose)
        if has_motion_prior:
            T_rel = T_motion_init
        else:
            T_rel = T_rel_icp
    else:
        T_rel = T_rel_icp

    # ---- lift back to world ----
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
        frame = torch.load(f"box_pts/frame_{str(i).zfill(4)}.pt")
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
