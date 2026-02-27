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
from optimizer import refine_pose, refine_pose_nuclear_rotation_multistart
from loss import loss
from loss import loss
from utils import compose_pose


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
        est_poses_coarse = []
        est_poses = []
        poses_gt = []
        for i, frame in enumerate(dataset):
            print(i)
            img_central = frame["LF"][s_size // 2, t_size // 2]

            camera_matrix = frame["camera_matrix"]
            depth = depth_estimator(frame)
            mask = segmentor(img_central)
            depth = depth[depth.shape[0] // 2]
            color = img_central[mask > 0].reshape(-1, 3)
            depth = depth * (mask > 0)
            pc = backproject_depth_to_pointcloud(
                pixel_indices=None,
                depths=depth,
                camera_matrix=camera_matrix,
            )
            pc = pc[(mask > 0).reshape(-1)]
            LF = frame["LF"]
            LF_perm = LF.reshape(-1, LF.shape[2], LF.shape[3], 3)
            LF_perm = LF_perm.permute(0, 3, 1, 2)
            poses_gt.append(frame["object_pose"])
            if i == 0:
                print(img_central.shape, mask.shape)
                Image.fromarray(
                    ((mask.float()[..., None] * img_central) * 255)
                    .cpu()
                    .numpy()
                    .astype(np.uint8),
                ).save("img_debug.png")
                surface_lf_rig = SurfaceLFRig.build(
                    K=torch.clone(frame["camera_matrix"]),
                    poses_4x4=torch.clone(frame["camera_poses_rel"].reshape(-1, 4, 4)),
                    image_size_hw=frame["LF"].shape[2:4],
                )
                tracker = Tracker(pc, color, pose0=frame["object_pose"])
                pose = torch.tensor(frame["object_pose"], dtype=torch.float32).cuda()
                est_poses.append(pose.cpu().numpy())
                est_poses_coarse.append(pose.cpu().numpy())
            else:
                pose = torch.tensor(
                    tracker.track(pc, color), dtype=torch.float32
                ).cuda()
                pose_rel = pose @ torch.linalg.inv(
                    torch.tensor(est_poses[-1], dtype=torch.float32).cuda()
                )
                est_poses_coarse.append(pose.cpu().numpy())
            surface_lf = SurfaceLF(
                rig=surface_lf_rig, pc=pc, images=LF_perm, current_pose=pose
            )
            image, depth = surface_lf.rasterize()
            if i > 0:
                pose_refined, final_loss = refine_pose(
                    surface_lf_prev=surface_lf_prev,
                    pose_coarse=pose_rel,
                    image=image,
                    depth=depth,
                    mask=mask,
                    mask_prev=mask_prev,
                    loss_fn=loss,
                    compose_pose_fn=compose_pose,
                )
                pose_refined, final_loss = refine_pose_nuclear_rotation_multistart(
                    surface_lf_prev,
                    image,
                    depth,
                    pose_coarse=pose_refined,
                    pose_init=pose_refined,
                    i=i,
                    mask_prev=mask_prev,
                    mask=mask,
                    init_loss=final_loss,
                    num_samples=96,
                    max_angle_deg=25.0,
                    topk=5,
                    refine_topk=3,
                    adam_iters=25,
                    lr_rotation=2e-2,
                    lr_translation=0.0,
                )
                est_poses.append(pose_refined.cpu().numpy() @ est_poses[-1])

            gt_poses.append(frame["object_pose"].cpu().numpy())
            surface_lf_prev = surface_lf
            mask_prev = mask

            # v.add_point_cloud(f"pc_{i:04d}", pc.cpu().numpy(), color.cpu().numpy())
        est_poses = np.stack(est_poses, axis=0)
        est_poses_coarse = np.stack(est_poses_coarse, axis=0)
        gt_poses = np.stack(gt_poses, axis=0)
        est_poses_coarse = rebase_poses(gt_poses, est_poses_coarse)
        est_poses = rebase_poses(gt_poses, est_poses)
        print("Pose errors (coarse):", pose_errors(est_poses_coarse, gt_poses))
        print("Pose errors (refined):", pose_errors(est_poses, gt_poses))
        np.save(f"{results_dir}/{dataset_name}/{sequence}.npy", est_poses)


if __name__ == "__main__":
    main()
