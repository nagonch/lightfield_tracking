import numpy as np
import torch
import open3d as o3d
import copy


class Open3DColoredICPTracker:
    def __init__(
        self,
        points_torch: torch.Tensor,
        colors_torch: torch.Tensor,
        voxel_size=1e-3,
        normals_neighbours=30,
        debug=False,
    ):
        self.voxel_size = voxel_size
        self.normals_neighbours = normals_neighbours
        self.debug = debug
        self.pcd_cur, self.fpfh_cur = self.process_pointcloud(
            points_torch.cpu().numpy(), colors_torch.cpu().numpy()
        )

        self.pose_prev = np.eye(4)
        self.pose_rel_prev = np.eye(4)

    def get_normals(self):
        return np.array(self.pcd_cur.normals)

    def visualize_pc(self, pcd):
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    def process_pointcloud(self, points, colors):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.voxel_size * 2.5, max_nn=self.normals_neighbours
            )
        )
        pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.voxel_size * 5.0, max_nn=100
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
        initial_transform_guess,
    ):
        distance_threshold = self.voxel_size * 20
        source_pretransformed = o3d.geometry.PointCloud(source_pcd)
        source_pretransformed.transform(initial_transform_guess)

        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_pretransformed,
            target_pcd,
            source_fpfh,
            target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
                False
            ),
            ransac_n=4,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold
                ),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                100000, 0.999
            ),
        )
        refined_transformation = result.transformation @ initial_transform_guess
        return refined_transformation, result

    def refine_pose_colored_icp(
        self,
        source_pcd: o3d.geometry.PointCloud,
        target_pcd: o3d.geometry.PointCloud,
        initial_transform_guess: np.ndarray,
    ):
        max_corr_dist = self.voxel_size * 5.0

        icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=50,
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
        N = int(len(corr) * 1e-2)
        if len(corr) > N:
            rng = np.random.default_rng(0)
            corr = corr[rng.choice(len(corr), size=N, replace=False)]

        src = copy.deepcopy(source_pcd).paint_uniform_color([1.0, 0.2, 0.2])
        tgt = copy.deepcopy(target_pcd).paint_uniform_color([0.2, 1.0, 0.2])
        tgt.points = o3d.utility.Vector3dVector(
            np.asarray(tgt.points) + np.array([0.5, 0.0, 0.0])
        )

        line_set = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
            source_pcd, tgt, corr
        )

        line_set.paint_uniform_color([0.1, 0.1, 1.0])

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=title)

        vis.add_geometry(src)
        vis.add_geometry(tgt)
        vis.add_geometry(line_set)
        vis.add_geometry(frame)

        render_options = vis.get_render_option()
        render_options.point_size = 0.5

        vis.run()
        vis.destroy_window()

    def track(self, points_torch: torch.Tensor, colors_torch: torch.Tensor):
        pcd, fpfh = self.process_pointcloud(
            points_torch.cpu().numpy(), colors_torch.cpu().numpy()
        )
        rel_transform = self.pose_rel_prev

        rel_transform, ransac_result = self.get_coarse_pose(
            source_pcd=self.pcd_cur,  # prev
            target_pcd=pcd,  # cur
            source_fpfh=self.fpfh_cur,
            target_fpfh=fpfh,
            initial_transform_guess=self.pose_rel_prev,
        )
        if self.debug:
            self.visualize_ransac_correspondences(self.pcd_cur, pcd, ransac_result)

        rel_transform, icp_result = self.refine_pose_colored_icp(
            source_pcd=self.pcd_cur,  # prev
            target_pcd=pcd,  # cur
            initial_transform_guess=rel_transform,
        )
        new_pose = rel_transform @ self.pose_prev

        self.pose_prev = new_pose
        self.pose_rel_prev = rel_transform
        self.pcd_cur = pcd
        self.fpfh_cur = fpfh

        return new_pose


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
    result_poses = []
    gt_poses = []
    for i in range(20):
        frame = torch.load(f"frame_{i:04d}.pt")
        points, colors = frame["pc"], frame["color"]
        gt_pose = frame["gt_pose"].cpu().numpy()
        if i == 0:
            tracker = Open3DColoredICPTracker(
                points,
                colors,
                debug=False,
            )
            normals = tracker.get_normals()
            first_frame_points = points.cpu().numpy()
            first_frame_colors = colors.cpu().numpy()
        else:
            pose = tracker.track(points, colors)
            result_poses.append(pose)
            gt_poses.append(gt_pose)
    result_poses = np.stack(result_poses)
    gt_poses = np.stack(gt_poses)
    result_poses = rebase_poses(gt_poses, result_poses)
    visualize_gt_vs_est(
        gt_poses_4x4=gt_poses,
        est_poses_4x4=result_poses,
        first_frame_points=first_frame_points,
        first_frame_colors=first_frame_colors,
        gt_color=(0.0, 0.8, 0.0),
        est_color=(0.9, 0.1, 0.1),
        show_frames=True,
        frame_size=0.05,
        frame_stride=1,
    )
