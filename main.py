from src.dataset import LFDataset
from time import time
from PIL import Image
import os
from segmentor import Segmentor
from depth_anything_3.api import DepthAnything3
import torch
from da3_functions import da3_run_from_tensors
from src.utilities import backproject_depth_to_pointcloud, Visualizer
import numpy as np
from src.disparity import get_LF_disparity
from depth_estimator import DepthEstimator
from surface_lf import SurfaceLF, SurfaceLFRig
from tracking import Tracker, pose_errors
from PIL import Image
from optimizer import refine_pose
from loss import loss
from loss import loss
from utils import compose_pose
from icp import icp_track


def rebase_poses(gt_poses, est_poses):
    pose_est_0 = est_poses[0]
    pose_gt_0 = gt_poses[0]
    est_to_gt = np.linalg.inv(pose_est_0) @ pose_gt_0
    est_poses = [p @ est_to_gt for p in est_poses]
    est_poses = np.stack(est_poses, axis=0)

    return est_poses


def main():
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    dataset_name = "LiFT_dataset"
    sequences = [
        x
        for x in os.listdir(f"/home/ngoncharov/cvpr2026/datasets/{dataset_name}/")
        if not (x.startswith("car_") or x.endswith(".sh") or x == "prod_ref")
    ]
    segmentor = Segmentor(prompt=None)
    depth_estimator = DepthEstimator(infer_gs=False)
    for sequence in sequences:
        # v = Visualizer()
        print(f"Estimating on {sequence}")
        os.makedirs(f"{results_dir}/{dataset_name}", exist_ok=True)
        if os.path.exists(f"{results_dir}/{dataset_name}/{sequence}.npy"):
            print(f"Skipping {sequence}, exists")
            continue
        dataset = LFDataset(
            f"/home/ngoncharov/cvpr2026/datasets/{dataset_name}/{sequence}"
        )
        s_size, t_size = dataset.metadata["n_views"]
        with open(
            f"/home/ngoncharov/cvpr2026/datasets/{dataset_name}/{sequence}/gdino_prompt.txt",
            "r",
            encoding="utf-8",
        ) as text_file:
            prompt = text_file.read()
        segmentor.prompt = prompt
        gt_poses = []
        est_poses = []
        for i, frame in enumerate(dataset):
            if i == 0:
                surface_lf_rig = SurfaceLFRig.build(
                    K=dataset[0]["camera_matrix"],
                    poses_4x4=dataset[0]["camera_poses_rel"].reshape(-1, 4, 4),
                    image_size_hw=(
                        dataset[0]["LF"].shape[2],
                        dataset[0]["LF"].shape[3],
                    ),
                )
            img_central = frame["LF"][s_size // 2, t_size // 2]

            camera_matrix = frame["camera_matrix"]
            depth = depth_estimator(frame)
            mask = segmentor(img_central)
            result_img = Image.fromarray(
                (img_central * mask[..., None] * 255).cpu().numpy().astype(np.uint8)
            )
            result_img.save(f"test_masks/{sequence}_{i}.png")
            depth = depth[depth.shape[0] // 2]
            color = img_central[mask > 0].reshape(-1, 3)
            depth = depth * (mask > 0)
            pc = backproject_depth_to_pointcloud(
                pixel_indices=None,
                depths=depth,
                camera_matrix=camera_matrix,
            )
            pc = pc[(mask > 0).reshape(-1)]
            surface_lf = SurfaceLF(
                surface_lf_rig,
                pc,
                frame["LF"]
                .reshape(-1, frame["LF"].shape[2], frame["LF"].shape[3], 3)
                .permute(0, 3, 1, 2),
            )
            image, depth = surface_lf.rasterize(torch.eye(4).cuda())
            if i == 0:
                pose = torch.tensor(frame["object_pose"], dtype=torch.float32).cuda()
                est_poses.append(pose.cpu().numpy())
                pose_rel_lhs = None
                pose_rel_rhs = None
            else:
                pose_est, pose_rel_rhs = icp_track(
                    pc.cpu().numpy(),
                    pc_prev.cpu().numpy(),
                    color.cpu().numpy(),
                    color_prev.cpu().numpy(),
                    est_poses[-1],
                    pose_rel_prev=pose_rel_prev_rhs,
                )
                pose_rel_lhs = pose_est @ np.linalg.inv(est_poses[-1])
                pose_rel_lhs_refined, loss_prev = refine_pose(
                    surface_lf_prev=surface_lf_prev,
                    pose_coarse=torch.tensor(pose_rel_lhs).cuda().float(),
                    image=image,
                    depth=depth,
                    pivot_world=torch.tensor(est_poses[-1][:3, 3]).float().cuda(),
                    mask=mask,
                    mask_prev=mask_prev,
                    num_iterations=100,
                    loss_fn=loss,
                    compose_pose_fn=compose_pose,
                )
                pose_est = pose_rel_lhs_refined.cpu().numpy() @ est_poses[-1]
                # v.add_point_cloud(
                #     f"{i}_pc", pc.cpu().numpy(), color.cpu().numpy(), point_size=1e-3
                # )
                # v.add_frame(f"{i}_est", pose_est)
                # v.add_frame(f"{i}_gt", frame["object_pose"].cpu().numpy())
                est_poses.append(pose_est)
            gt_poses.append(frame["object_pose"].cpu().numpy())

            pc_prev = pc
            color_prev = color
            pose_rel_prev_rhs = pose_rel_rhs
            pose_rel_prev_lhs = pose_rel_lhs
            mask_prev = mask
            surface_lf_prev = surface_lf

        est_poses = np.stack(est_poses, axis=0)
        gt_poses = np.stack(gt_poses, axis=0)
        est_poses = rebase_poses(gt_poses, est_poses)
        print("Pose errors (refined):", pose_errors(est_poses, gt_poses))
        np.save(f"{results_dir}/{dataset_name}/{sequence}.npy", est_poses)
        # v.run()


if __name__ == "__main__":
    main()
