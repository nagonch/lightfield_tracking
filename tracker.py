import numpy as np
import torch
import open3d as o3d


class Open3DColoredICPTracker:
    def __init__(
        self,
        points_torch: torch.Tensor,
        colors_torch: torch.Tensor,
    ):
        self.points_cur = points_torch.cpu().numpy()
        self.colors_cur = colors_torch.cpu().numpy()
        self.pose_prev = np.eye(4)
        self.pose_rel_prev = np.eye(4)

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
                points_torch=points,
                colors_torch=colors,
                debug_folder="icp_debug",
                point_size=2,
            )
        else:
            pose, normals = tracker.track(points_torch=points, colors_torch=colors)
            print(pose)
