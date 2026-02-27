import numpy as np
import torch
import open3d as o3d
import copy


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


class Tracker:
    def __init__(
        self,
        points_torch: torch.Tensor,
        colors_torch: torch.Tensor,
        pose0: torch.Tensor | None = None,
        debug=False,
    ):
        self.voxel_size = 2e-3
        self.normals_neighbours = 30
        self.debug = debug
        # process_pointcloud
        self.normals_radius_mult = 2.5
        self.orient_normals_consistent_k = 20
        self.fpfh_radius_mult = 15.0
        self.fpfh_max_nn = 200

        # get_coarse_pose (RANSAC)
        self.ransac_distance_mult = 50
        self.ransac_mutual_filter = True
        self.ransac_point_to_point_with_scaling = False
        self.ransac_n = 4
        self.ransac_edge_length_checker = 0.8
        self.ransac_max_iterations = 200_000
        self.ransac_confidence = 0.999

        # refine_pose_colored_icp
        self.icp_max_corr_mult = 5.0
        self.icp_relative_fitness = 1e-6
        self.icp_relative_rmse = 1e-6
        self.icp_max_iteration = 120

        # visualize_ransac_correspondences
        self.vis_corr_fraction = 1e-2
        self.vis_rng_seed = 0
        self.vis_src_color = [1.0, 0.2, 0.2]
        self.vis_tgt_color = [0.2, 1.0, 0.2]
        self.vis_tgt_offset = [0.5, 0.0, 0.0]
        self.vis_line_color = [0.1, 0.1, 1.0]
        self.vis_frame_size = 0.2
        self.vis_point_size = 0.5

        # centroid_correction
        self.centroid_alpha = 0.8

        # ----------------------------
        # Original initialization logic
        # ----------------------------
        self.pcd_cur, self.fpfh_cur = self.process_pointcloud(
            points_torch.cpu().numpy(), colors_torch.cpu().numpy()
        )

        self.pose_prev = pose0.cpu().numpy() if pose0 is not None else np.eye(4)
        self.pose_rel_prev = np.eye(4)

    def get_normals(self):
        return np.array(self.pcd_cur.normals)

    def _robust_centroid(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        points_xyz = np.asarray(pcd.points)
        if points_xyz.size == 0:
            return np.zeros(3, dtype=np.float64)

        trim = float(getattr(self, "centroid_trim", 0.10))
        trim = np.clip(trim, 0.0, 0.49)

        if trim <= 0.0 or points_xyz.shape[0] < 20:
            # fall back to median for tiny clouds
            return np.median(points_xyz, axis=0)

        lo = np.quantile(points_xyz, trim, axis=0)
        hi = np.quantile(points_xyz, 1.0 - trim, axis=0)
        mask = np.all((points_xyz >= lo) & (points_xyz <= hi), axis=1)
        pts = points_xyz[mask] if np.any(mask) else points_xyz
        return np.mean(pts, axis=0)

    def visualize_pc(self, pcd):
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    def process_pointcloud(self, points, colors):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.voxel_size * self.normals_radius_mult,
                max_nn=self.normals_neighbours,
            )
        )
        # pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))
        pcd.orient_normals_consistent_tangent_plane(k=self.orient_normals_consistent_k)
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.voxel_size * self.fpfh_radius_mult, max_nn=self.fpfh_max_nn
            ),
        )
        if self.debug:
            self.visualize_pc(pcd)
        return pcd, fpfh

    def get_coarse_pose(
        self,
        source_pcd,
        target_pcd,
        source_fpfh,
        target_fpfh,
    ):
        distance_threshold = self.voxel_size * self.ransac_distance_mult
        source_pretransformed = o3d.geometry.PointCloud(source_pcd)

        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_pretransformed,
            target_pcd,
            source_fpfh,
            target_fpfh,
            mutual_filter=self.ransac_mutual_filter,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
                self.ransac_point_to_point_with_scaling
            ),
            ransac_n=self.ransac_n,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    self.ransac_edge_length_checker
                ),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold
                ),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                self.ransac_max_iterations, self.ransac_confidence
            ),
        )
        refined_transformation = result.transformation
        return refined_transformation, result

    def refine_pose_colored_icp(
        self,
        source_pcd: o3d.geometry.PointCloud,
        target_pcd: o3d.geometry.PointCloud,
        initial_transform_guess: np.ndarray,
    ):
        max_corr_dist = self.voxel_size * self.icp_max_corr_mult

        icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=self.icp_relative_fitness,
            relative_rmse=self.icp_relative_rmse,
            max_iteration=self.icp_max_iteration,
        )

        icp_result = o3d.pipelines.registration.registration_colored_icp(
            source=source_pcd,
            target=target_pcd,
            max_correspondence_distance=max_corr_dist,
            init=initial_transform_guess,
            estimation_method=o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            criteria=icp_criteria,
        )

        refined_transform = icp_result.transformation
        return refined_transform, icp_result

    def visualize_ransac_correspondences(
        self, source_pcd, target_pcd, ransac_result, title="RANSAC correspondences"
    ):
        corr = np.asarray(ransac_result.correspondence_set)
        N = int(len(corr) * self.vis_corr_fraction)
        if len(corr) > N:
            rng = np.random.default_rng(self.vis_rng_seed)
            corr = corr[rng.choice(len(corr), size=N, replace=False)]

        src = copy.deepcopy(source_pcd).paint_uniform_color(self.vis_src_color)
        tgt = copy.deepcopy(target_pcd).paint_uniform_color(self.vis_tgt_color)
        tgt.points = o3d.utility.Vector3dVector(
            np.asarray(tgt.points) + np.array(self.vis_tgt_offset)
        )

        line_set = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
            source_pcd, tgt, corr
        )

        line_set.paint_uniform_color(self.vis_line_color)

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=self.vis_frame_size
        )
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title)

        vis.add_geometry(src)
        vis.add_geometry(tgt)
        vis.add_geometry(line_set)
        vis.add_geometry(frame)

        render_options = vis.get_render_option()
        render_options.point_size = self.vis_point_size

        vis.run()
        vis.destroy_window()

    def centroid_correction(self, rel_transform, target_pcd):
        rel_transform = np.copy(rel_transform)
        R = rel_transform[:3, :3]
        t_icp = rel_transform[:3, 3]

        c_prev = self._robust_centroid(self.pcd_cur)
        c_cur = self._robust_centroid(target_pcd)

        t_centroid = c_cur - R @ c_prev

        alpha = self.centroid_alpha
        rel_transform[:3, 3] = (1 - alpha) * t_icp + alpha * t_centroid
        return rel_transform

    def track(self, points_torch: torch.Tensor, colors_torch: torch.Tensor):
        pcd, fpfh = self.process_pointcloud(
            points_torch.cpu().numpy(), colors_torch.cpu().numpy()
        )

        rel_transform, ransac_result = self.get_coarse_pose(
            source_pcd=self.pcd_cur,  # prev
            target_pcd=pcd,  # cur
            source_fpfh=self.fpfh_cur,
            target_fpfh=fpfh,
        )
        if self.debug:
            self.visualize_ransac_correspondences(self.pcd_cur, pcd, ransac_result)

        rel_transform, icp_result = self.refine_pose_colored_icp(
            source_pcd=self.pcd_cur,  # prev
            target_pcd=pcd,  # cur
            initial_transform_guess=rel_transform,
        )
        rel_transform = self.centroid_correction(rel_transform, pcd)

        new_pose = rel_transform @ self.pose_prev

        self.pose_prev = new_pose
        self.pose_rel_prev = rel_transform
        self.pcd_cur = pcd
        self.fpfh_cur = fpfh

        return rel_transform


def rebase_poses(gt_poses, est_poses):
    pose_est_0 = est_poses[0]
    pose_gt_0 = gt_poses[0]
    est_to_gt = np.linalg.inv(pose_est_0) @ pose_gt_0
    est_poses = [p @ est_to_gt for p in est_poses]
    est_poses = np.stack(est_poses, axis=0)

    return est_poses


def _make_trajectory_lineset(
    positions_xyz: np.ndarray, rgb=(1.0, 0.0, 0.0)
) -> o3d.geometry.LineSet:
    """
    positions_xyz: (N, 3)
    """
    positions_xyz = np.asarray(positions_xyz, dtype=np.float64)
    num_points = positions_xyz.shape[0]
    if num_points < 2:
        raise ValueError("Need at least 2 poses to draw a trajectory.")

    line_indices = np.stack(
        [np.arange(num_points - 1), np.arange(1, num_points)], axis=1
    ).astype(np.int32)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(positions_xyz)
    line_set.lines = o3d.utility.Vector2iVector(line_indices)
    line_set.colors = o3d.utility.Vector3dVector(
        np.tile(np.array(rgb, dtype=np.float64), (num_points - 1, 1))
    )
    return line_set


def _add_pose_frames(
    geometries: list,
    poses_4x4: np.ndarray,
    frame_size: float = 0.05,
    every_k: int = 1,
):
    """
    poses_4x4: (N, 4, 4) or (N, 3, 4) (will be promoted to 4x4)
    """
    poses_4x4 = np.asarray(poses_4x4)
    if poses_4x4.shape[-2:] == (3, 4):
        bottom_row = np.tile(
            np.array([0, 0, 0, 1], dtype=poses_4x4.dtype), (poses_4x4.shape[0], 1, 1)
        )
        poses_4x4 = np.concatenate([poses_4x4, bottom_row], axis=1)

    for pose_index in range(0, poses_4x4.shape[0], every_k):
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        coord_frame.transform(poses_4x4[pose_index])
        geometries.append(coord_frame)


def visualize_gt_vs_est(
    gt_poses_4x4: np.ndarray,
    est_poses_4x4: np.ndarray,
    first_frame_points: np.ndarray | None = None,
    first_frame_colors: np.ndarray | None = None,
    gt_color=(0.0, 0.8, 0.0),
    est_color=(0.9, 0.1, 0.1),
    show_frames=True,
    frame_size=0.05,
    frame_stride=2,
):
    """
    Expects poses as (N, 4, 4). Uses translation part to draw trajectories.
    """
    gt_poses_4x4 = np.asarray(gt_poses_4x4)
    est_poses_4x4 = np.asarray(est_poses_4x4)

    gt_positions = gt_poses_4x4[:, :3, 3]
    est_positions = est_poses_4x4[:, :3, 3]

    geometries: list[o3d.geometry.Geometry] = []

    # Trajectory lines
    geometries.append(_make_trajectory_lineset(gt_positions, rgb=gt_color))
    geometries.append(_make_trajectory_lineset(est_positions, rgb=est_color))

    # Pose frames (optional)
    if show_frames:
        _add_pose_frames(
            geometries, gt_poses_4x4, frame_size=frame_size, every_k=frame_stride
        )
        _add_pose_frames(
            geometries, est_poses_4x4, frame_size=frame_size * 0.9, every_k=frame_stride
        )

    # Optional: point cloud context
    if first_frame_points is not None:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.asarray(first_frame_points))
        if first_frame_colors is not None:
            colors = np.asarray(first_frame_colors)
            if colors.dtype != np.float64:
                colors = colors.astype(np.float64)
            if colors.max() > 1.0:
                colors = colors / 255.0
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(point_cloud)

    # Global frame at origin for reference
    geometries.append(
        o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size * 1.5)
    )

    o3d.visualization.draw_geometries(
        geometries,
        window_name="GT (green) vs Estimated (red)",
    )


if __name__ == "__main__":
    result_poses = [
        np.eye(4),
    ]
    gt_poses = []
    for i in range(20):
        print(i)
        frame = torch.load(f"frame_{i:04d}.pt")
        points, colors = frame["pc"], frame["color"]
        gt_pose = frame["gt_pose"].cpu().numpy()
        if i == 0:
            tracker = Tracker(
                points,
                colors,
                debug=False,
                pose0=frame["gt_pose"],
            )
            normals = tracker.get_normals()
            first_frame_points = points.cpu().numpy()
            first_frame_colors = colors.cpu().numpy()
            gt_poses.append(gt_pose)
        else:
            pose_rel = tracker.track(points, colors)
            result_poses.append(pose_rel @ result_poses[-1])
            gt_poses.append(gt_pose)
    result_poses = np.stack(result_poses)
    gt_poses = np.stack(gt_poses)
    result_poses = rebase_poses(gt_poses, result_poses)
    print(pose_errors(gt_poses, result_poses))
