from src.dataset import LFDataset
from src.utilities import Visualizer, backproject_depth_to_pointcloud
import numpy as np
import argparse

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset directory",
    )
    args = args.parse_args()
    data_path = args.dataset_path

    dataset = LFDataset(data_path)
    visualizer = Visualizer()
    for i, frame in enumerate(dataset):
        camera_matrix = frame["camera_matrix"]
        baseline = frame["baseline"]
        depth = frame["depth"]
        s_size, t_size = frame["LF"].shape[0], frame["LF"].shape[1]
        points = backproject_depth_to_pointcloud(None, depth, camera_matrix)
        colors = frame["LF"][s_size // 2, t_size // 2].cpu().numpy().reshape(-1, 3)
        if "masks" in frame and frame.get("masks", None) is not None:
            mask = frame["masks"][s_size // 2, t_size // 2].cpu().numpy().reshape(-1)
            points = points.reshape(-1, 3)[mask.reshape(-1) > 0]
            colors = colors.reshape(-1, 3)[mask.reshape(-1) > 0]

        object_to_base_pose = frame["object_pose"].cpu().numpy()
        object_to_cam_pose = (
            np.linalg.inv(frame["camera_poses"][s_size // 2, t_size // 2].cpu().numpy())
            @ object_to_base_pose
        )
        visualizer.add_point_cloud(
            f"points_{i}", points.cpu().numpy(), colors=colors, point_size=1e-3
        )
        visualizer.add_frame(name=f"obj_{i}", frame_T=object_to_cam_pose)
    visualizer.run()
