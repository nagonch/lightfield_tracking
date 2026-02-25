import numpy as np
import torch
import open3d as o3d


class Open3DColoredICPTracker:
    def __init__(
        self,
        points_torch: torch.Tensor,
        colors_torch: torch.Tensor,
        voxel_size=1e-3,
        normals_neighbours=30,
    ):
        self.voxel_size = voxel_size
        self.normals_neighbours = normals_neighbours

        self.points_cur = points_torch.cpu().numpy()
        self.colors_cur = colors_torch.cpu().numpy()
        self.pcd_cur = self.process_pointcloud(self.points_cur, self.colors_cur)
        raise

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
        self.visualize_pc(pcd)
        return pcd

    def track(self, points_torch: torch.Tensor, colors_torch: torch.Tensor):
        initial_guess = self.pose_rel_prev
        relative_transform = np.copy(initial_guess)  # todo: change
        new_pose = relative_transform @ self.pose_prev

        self.pose_prev = new_pose
        self.pose_rel_prev = relative_transform
        self.points_cur = points_torch.cpu().numpy()
        self.colors_cur = colors_torch.cpu().numpy()

        return new_pose, None


if __name__ == "__main__":
    for i in range(5):
        frame = torch.load(f"frame_{i:04d}.pt")
        points, colors = frame["pc"], frame["color"]
        if i == 0:
            tracker = Open3DColoredICPTracker(
                points,
                colors,
            )
        else:
            pose, normals = tracker.track(points, colors)
            print(pose)
