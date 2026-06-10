from depth_estimator import DepthEstimator
from icp import (
    apply_transform_to_points,
    apply_transform_to_points,
    get_coarsest_pose,
    run_explorative_icp_with_centering,
    pose_errors,
    rebase_poses,
)
from segmentor import Segmentor
from src.utilities import backproject_depth_to_pointcloud
from src.dataset import LFDataset
from tqdm import tqdm
import numpy as np
import os
from surface_lf import SurfaceLF, SurfaceLFRig
import torch
from loss import refine_pose
from src.slf_refinement_viewer import SurfaceLFRefinementViewer
from PIL import Image
from time import sleep

if __name__ == "__main__":
    EXP_NAME = "results_ours"
    USE_GT_DEPTH = True
    USE_GT_MASK = True
    USE_GT_ENV_MAP = False
    USE_RELIGHT = True
    USE_NAIVE_RELIGHT = False
    USE_ENV_MAP = USE_RELIGHT or USE_NAIVE_RELIGHT
    USE_ICP = True
    SEGMENTATION_PROMPT = "cube."
    ENABLE_REFINEMENT_VIEWER = False
    MASK_LOSS = False

    REFINEMENT_VIEWER_UPDATE_EVERY = 10

    env_gt_map = None
    if USE_GT_ENV_MAP:
        env_gt_map = torch.from_numpy(
            np.asarray(Image.open("env_gt.jpg").convert("RGB"), dtype=np.float32)
            / 255.0
        )
    segmentor = Segmentor(prompt=SEGMENTATION_PROMPT)
    for REFLECTIVITY in [
        "0.0",
        "0.5",
        "0.7",
        "1.0",
    ]:
        RESULTS_FOLDER = f"{EXP_NAME}_{REFLECTIVITY}/ycbv_lf"
        os.makedirs(RESULTS_FOLDER, exist_ok=True)
        if not USE_GT_DEPTH:
            depth_estimator = DepthEstimator(infer_gs=False)
        for sequence_name in os.listdir(
            f"/home/ngoncharov/cvpr2026/ycbv-eoat-lf/dataset_simple_box_reflective_full_{REFLECTIVITY}"
        ):
            print(f"running {sequence_name}")
            path = f"/home/ngoncharov/cvpr2026/ycbv-eoat-lf/dataset_simple_box_reflective_full_{REFLECTIVITY}/{sequence_name}"
            dataset = LFDataset(path)
            s_size, t_size = dataset.metadata["n_views"]
            refinement_viewer = SurfaceLFRefinementViewer(
                enabled=ENABLE_REFINEMENT_VIEWER,
                update_every=REFINEMENT_VIEWER_UPDATE_EVERY,
            )

            try:
                gt_poses = []
                est_poses_coarse = []
                est_poses = []
                pose_rel_prev = None
                env_map_prev = None
                for i, frame in tqdm(enumerate(dataset)):
                    frame["pose"] = frame["object_pose"]
                    img_central = frame["LF"][s_size // 2, t_size // 2]
                    if USE_GT_MASK:
                        mask = frame["masks"][s_size // 2, t_size // 2]
                    else:
                        mask = segmentor.segment(img_central.cpu().numpy())
                    camera_matrix = frame["camera_matrix"]
                    if USE_GT_DEPTH:
                        depth = frame["depth"]
                    else:
                        depth, depth_conf_mask = depth_estimator(frame, mask)
                        depth = depth[depth.shape[0] // 2]
                    pc, pc_scales = backproject_depth_to_pointcloud(
                        pixel_indices=None,
                        depths=depth,
                        camera_matrix=camera_matrix,
                        return_scales=True,
                    )
                    pc = pc[(mask > 0).reshape(-1)].cpu().numpy()
                    pc_scales = pc_scales[(mask > 0).reshape(-1)].cpu().numpy()
                    color = img_central[mask > 0].reshape(-1, 3).cpu().numpy()
                    gt_poses.append(frame["pose"].cpu().numpy())
                    if i == 0:
                        pc_prev = pc
                        color_prev = color
                        est_poses_coarse.append(frame["pose"].cpu().numpy())
                        est_poses.append(frame["pose"].cpu().numpy())
                        surface_lf_rig = SurfaceLFRig.build(
                            K=dataset[0]["camera_matrix"],
                            poses_4x4=dataset[0]["camera_poses_rel"].reshape(-1, 4, 4),
                            image_size_hw=(
                                dataset[0]["LF"].shape[2],
                                dataset[0]["LF"].shape[3],
                            ),
                        )
                        surface_lf = SurfaceLF(
                            surface_lf_rig,
                            torch.tensor(pc).cuda(),
                            frame["LF"]
                            .reshape(-1, frame["LF"].shape[2], frame["LF"].shape[3], 3)
                            .permute(0, 3, 1, 2),
                            torch.tensor(pc_scales).cuda(),
                            previous_environment_map=None,
                            separation_alpha=1.0 - float(REFLECTIVITY),
                            use_naive_relight=USE_NAIVE_RELIGHT,
                            use_environment_map=USE_ENV_MAP,
                            use_relight=USE_RELIGHT,
                            # use_relight=False,
                        )
                        if USE_GT_ENV_MAP and surface_lf.environment_map is not None:
                            surface_lf.environment_map = env_gt_map.to(
                                device=surface_lf.environment_map.device,
                                dtype=surface_lf.environment_map.dtype,
                            )
                        # reflection_separation(
                        #     surface_lf, alpha=(1 - float(REFLECTIVITY))
                        # )
                        image, depth, target_mask = surface_lf.rasterize(
                            torch.eye(4).cuda()
                        )
                    else:
                        surface_lf = SurfaceLF(
                            surface_lf_rig,
                            torch.tensor(pc).cuda(),
                            frame["LF"]
                            .reshape(-1, frame["LF"].shape[2], frame["LF"].shape[3], 3)
                            .permute(0, 3, 1, 2),
                            torch.tensor(pc_scales).cuda(),
                            previous_environment_map=None,
                            separation_alpha=1.0 - float(REFLECTIVITY),
                            use_naive_relight=USE_NAIVE_RELIGHT,
                            use_environment_map=USE_ENV_MAP,
                            use_relight=USE_RELIGHT,
                        )
                        if USE_GT_ENV_MAP and surface_lf.environment_map is not None:
                            surface_lf.environment_map = env_gt_map.to(
                                device=surface_lf.environment_map.device,
                                dtype=surface_lf.environment_map.dtype,
                            )
                        image, depth, target_mask = surface_lf.rasterize(
                            torch.eye(4).cuda()
                        )
                        if USE_ICP:
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

                            coarse_pose = registration_result[
                                "transform_source_to_target"
                            ]
                            pc_refined = apply_transform_to_points(
                                pc_prev_trans, coarse_pose
                            )
                            coarse_pose = coarse_pose @ coarsest_pose
                            est_poses_coarse.append(coarse_pose)

                            pose_rel_lhs = coarse_pose @ np.linalg.inv(est_poses[-1])
                            pose_rel_rhs = np.linalg.inv(est_poses[-1]) @ coarse_pose
                        else:
                            pose_rel_rhs = np.eye(4)

                        pose_rel_rhs_refined, pose_rel_lhs_refined, pose_history_rhs = (
                            refine_pose(
                                surface_lf_prev=surface_lf_prev,
                                surface_lf=surface_lf,
                                pose_coarse_rhs=torch.tensor(pose_rel_rhs)
                                .cuda()
                                .float(),
                                target_mask=target_mask.cuda() if MASK_LOSS else None,
                                image=image.cuda(),
                                depth=depth.cuda(),
                                pivot_world=torch.tensor(est_poses[-1]).float().cuda(),
                                convergence_plot_filename=None,
                                refinement_viewer=refinement_viewer,
                            )
                        )
                        pose_refined = pose_rel_lhs_refined @ est_poses[-1]
                        if not USE_ICP:
                            est_poses_coarse.append(pose_refined)
                        est_poses.append(pose_refined)
                        pc_prev = pc
                        color_prev = color
                    if surface_lf.environment_map is not None:
                        env_map_prev = surface_lf.environment_map.detach()
                        if USE_GT_ENV_MAP:
                            env_map_prev = env_gt_map
                        env_map_np = (
                            torch.clamp(env_map_prev, 0.0, 1.0).cpu().numpy() * 255.0
                        ).astype(np.uint8)
                        Image.fromarray(env_map_np).save(os.path.join("env_map.png"))
                    else:
                        env_map_prev = None
                    surface_lf_prev = surface_lf
                    mask_prev = mask
                    gt_poses_np = np.stack(gt_poses, axis=0)
                    est_poses_np = np.stack(est_poses, axis=0)
                    est_poses_coarse_np = np.stack(est_poses_coarse, axis=0)
                    est_poses_coarse_np = rebase_poses(gt_poses_np, est_poses_coarse_np)
                    est_poses_np = rebase_poses(gt_poses_np, est_poses_np)
                    np.save(
                        os.path.join(RESULTS_FOLDER, f"{sequence_name}.npy"), est_poses
                    )
                    print(pose_errors(gt_poses_np, est_poses_coarse_np))
                    print(pose_errors(gt_poses_np, est_poses_np))
            finally:
                refinement_viewer.close()
