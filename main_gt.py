from icp_simple import (
    apply_transform_to_points,
    apply_transform_to_points,
    get_coarsest_pose,
    run_explorative_icp_with_centering,
    pose_errors,
)
from main import rebase_poses
from src.utilities import backproject_depth_to_pointcloud
from src.dataset import LFDataset
from src.utilities import Visualizer
from tqdm import tqdm
import numpy as np
import os

if __name__ == "__main__":
    for REFLECTIVITY in ["0.0", "0.5", "0.7", "1.0"]:
        RESULTS_FOLDER = f"ours_icp_colorless_{REFLECTIVITY}"
        os.makedirs(RESULTS_FOLDER, exist_ok=True)
        for sequence_name in os.listdir(
            f"/home/ngoncharov/cvpr2026/ycbv-eoat-lf/dataset_simple_box_reflective_full_{REFLECTIVITY}"
        ):
            print(f"running {sequence_name}")
            RESULTS_FOLDER = f"test"
            path = f"/home/ngoncharov/cvpr2026/ycbv-eoat-lf/dataset_simple_box_reflective_full_{REFLECTIVITY}/{sequence_name}"
            dataset = LFDataset(path)
            s_size, t_size = dataset.metadata["n_views"]

            gt_poses = []
            est_poses = []
            # v = Visualizer()
            pose_rel_prev = None
            for i, frame in tqdm(enumerate(dataset)):
                frame["pose"] = frame["object_pose"]
                mask = frame["masks"][s_size // 2, t_size // 2]
                img_central = frame["LF"][s_size // 2, t_size // 2]
                camera_matrix = frame["camera_matrix"]
                depth = frame["depth"]
                pc = backproject_depth_to_pointcloud(
                    pixel_indices=None,
                    depths=depth,
                    camera_matrix=camera_matrix,
                )
                pc = pc[(mask > 0).reshape(-1)].cpu().numpy()
                color = img_central[mask > 0].reshape(-1, 3).cpu().numpy()
                gt_poses.append(frame["pose"].cpu().numpy())
                if i == 0:
                    pc_prev = pc
                    color_prev = color
                    est_poses.append(frame["pose"].cpu().numpy())
                else:
                    coarsest_pose, pc_prev_trans = get_coarsest_pose(
                        pc_prev, pc, est_poses[-1]
                    )
                    registration_result = run_explorative_icp_with_centering(
                        source_points_xyz=pc_prev_trans,
                        target_points_xyz=pc,
                        source_colors_rgb=color_prev,
                        target_colors_rgb=color,
                        max_correspondence_distance=0.01,
                    )

                    coarse_pose = registration_result["transform_source_to_target"]
                    pc_refined = apply_transform_to_points(pc_prev_trans, coarse_pose)
                    coarse_pose = coarse_pose @ coarsest_pose

                    est_poses.append(coarse_pose)
                    # v.add_point_cloud(
                    #     f"pc_coarse_{i}", pc_prev_trans, color_prev, point_size=1e-3
                    # )
                    # v.add_point_cloud(
                    #     f"pc_aligned_{i}", pc_refined, color_prev, point_size=1e-3
                    # )
                    # v.add_point_cloud(f"pc_{i}", pc, color, point_size=1e-3)

                    pc_prev = pc
                    color_prev = color
            gt_poses = np.stack(gt_poses, axis=0)
            est_poses = np.stack(est_poses, axis=0)
            est_poses = rebase_poses(gt_poses, est_poses)
            np.save(os.path.join(RESULTS_FOLDER, f"{sequence_name}.npy"), est_poses)
            print(pose_errors(gt_poses, est_poses))
            # for i, (pose_est, pose_gt) in enumerate(zip(est_poses, gt_poses)):
            #     v.add_frame(f"{i}_est", pose_est, frames_scale=0.01)
            #     v.add_frame(f"{i}_gt", pose_gt, frames_scale=0.01, origin_color=(255, 255, 255))
            # v.run()
