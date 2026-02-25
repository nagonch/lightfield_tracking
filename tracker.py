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
            source_pcd=pcd,
            target_pcd=self.pcd_cur,
            source_fpfh=fpfh,
            target_fpfh=self.fpfh_cur,
            initial_transform_guess=self.pose_rel_prev,
        )
        if self.debug:
            self.visualize_ransac_correspondences(self.pcd_cur, pcd, ransac_result)
        new_pose = rel_transform @ self.pose_prev

        self.pose_prev = new_pose
        self.pose_rel_prev = rel_transform
        self.pcd_cur = pcd
        self.fpfh_cur = fpfh

        return new_pose, None


if __name__ == "__main__":
    for i in range(5):
        frame = torch.load(f"frame_{i:04d}.pt")
        points, colors = frame["pc"], frame["color"]
        if i == 0:
            tracker = Open3DColoredICPTracker(
                points,
                colors,
                debug=True,
            )
        else:
            pose, normals = tracker.track(points, colors)
            print(pose)
