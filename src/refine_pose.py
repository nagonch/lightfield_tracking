import torch
import torch.nn as nn
import torch.optim as optim
import viser
from pathlib import Path
import time
from .gaussian_splatting.gsplat_viewer import GsplatViewer
from .gaussian_splatting.train_tools import (
    get_optimizers_and_params,
    viewer_render_fn as _viewer_render_fn,
    batch_rasterize,
    viewer_tick,
    mask_aware_loss_img,
)
from .spherical_harmonics import transform_shs
from dataclasses import dataclass
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import numpy as np
from pytorch3d.transforms import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    se3_log_map,
    se3_exp_map,
)
from torch.optim.lr_scheduler import (
    StepLR,
)
from matplotlib import pyplot as plt
from torchvision.transforms import Resize


def resize(rendered, gt, depth, alphas, mask, coeff):
    B, H, W, C = rendered.shape
    size = (int(H * coeff), int(W * coeff))

    rendered = rendered.permute(0, 3, 1, 2)
    gt = gt.permute(0, 3, 1, 2)
    alphas = alphas[..., 0]
    mask_resized = F.interpolate(mask.unsqueeze(1).float(), size=size, mode="nearest")[
        :, 0
    ].long()
    resize_transform = Resize(size, antialias=True)
    rendered_resized = resize_transform(rendered)
    gt_resized = resize_transform(gt)
    depth_resized = resize_transform(depth.unsqueeze(1))[:, 0][..., None]
    alphas_resized = resize_transform(alphas.unsqueeze(1))[:, 0][..., None]
    rendered_resized = rendered_resized.permute(0, 2, 3, 1)
    gt_resized = gt_resized.permute(0, 2, 3, 1)
    return rendered_resized, gt_resized, depth_resized, alphas_resized, mask_resized


@dataclass
class Config:
    lr: float = 5e-3
    translation_lr_ratio: float = 0.1
    num_epochs: int = 500
    warmup_epochs: int = 10
    total_sh_degrees: int = 3
    max_grad_norm: float = 1
    enable_viewer: bool = True
    sh_lr_fraction: float = 0.1
    opacity_lr_fraction: float = 0.5
    verbose: bool = True
    batch_ratio: float = 0.3


def get_splat_poses(means, quats):
    rot_mats = quaternion_to_matrix(quats)
    poses = torch.eye(4).float().cuda().unsqueeze(0).repeat(means.shape[0], 1, 1)
    poses[:, :3, :3] = rot_mats
    poses[:, :3, 3] = means
    return poses


def transform_splats(means, colors, quats, rel_trans_lhs):
    splat_poses = get_splat_poses(means, quats)
    splat_poses = rel_trans_lhs @ splat_poses
    means = splat_poses[:, :3, 3]
    quats = matrix_to_quaternion(splat_poses[:, :3, :3])
    colors = transform_shs(colors, rel_trans_lhs[:3, :3])
    norm = quats.norm(dim=-1, keepdim=True)
    quats = quats / (norm + 1e-8)

    return means, colors, quats


def pose_to_SE3_vector(pose_T):
    pose_T[-1, :3] = pose_T[:3, -1]
    pose_T[:3, -1] = 0.0
    pose_vector = se3_log_map(pose_T[None])
    return pose_vector[0]


def SE3_vector_to_pose(pose_vector):
    rel_trans = se3_exp_map(pose_vector[None])[0]
    rel_trans[:3, -1] = rel_trans[-1, :3]
    rel_trans[-1, :3] = 0.0
    return rel_trans


def schedule_resolutions(total_epochs, resolution_list):
    epochs_per_stage = total_epochs // len(resolution_list)
    remainder_epochs = total_epochs % len(resolution_list)

    resolution_schedule = []
    for resolution in resolution_list:
        resolution_schedule += [resolution] * epochs_per_stage

    if remainder_epochs > 0:
        resolution_schedule += [resolution_list[-1]] * remainder_epochs

    return resolution_schedule


def add_frame(scene, name, frame_T, frames_scale=0.05):
    if not isinstance(frame_T, np.ndarray):
        frame_T = np.array(frame_T.cpu())
    position = frame_T[:3, 3]
    rotation = frame_T[:3, :3]
    wxyz = R.from_matrix(rotation).as_quat(scalar_first=True)
    scene.add_frame(
        name=name,
        position=position,
        wxyz=wxyz,
        axes_length=frames_scale * 2,
        origin_radius=frames_scale / 5,
        axes_radius=frames_scale / 10,
    )
    return scene


def refine_splat_camera_poses(
    splats_left,
    splats_right,
    train_images_left,
    train_poses_left,
    train_masks_left,
    train_images_right,
    train_poses_right,
    train_masks_right,
    transform,  # ob0_T_ob1
    transform_gt,
    object_pose,
    init_noise_norm=0.01,
    camera_matrix=torch.eye(3).cuda().float(),
    config: Config = Config(),
    lr=None,
    n_epochs=None,
    verbose=None,
    enable_viewer=None,
    ssim_weight=0,
    ctf_sizes=[1],
    device="cuda",
):
    if lr is not None:
        config.lr = lr
    if n_epochs is not None:
        config.num_epochs = n_epochs
    if enable_viewer is not None:
        config.enable_viewer = enable_viewer
    if verbose is not None:
        config.verbose = verbose
    transform_1_to_0 = torch.linalg.inv(transform)
    transform_right_hand = (
        torch.linalg.inv(object_pose) @ transform_1_to_0 @ object_pose
    )
    pose_vector = pose_to_SE3_vector(transform_right_hand)
    rel_translation = nn.Parameter(
        pose_vector[:3]
        + torch.normal(0, init_noise_norm * torch.norm(pose_vector[:3]), size=(3,)).to(
            device
        )
    )
    rel_rotation = nn.Parameter(
        pose_vector[3:]
        + torch.normal(0, init_noise_norm * torch.norm(pose_vector[3:]), size=(3,)).to(
            device
        )
    )
    transform_1_to_0_gt = torch.linalg.inv(transform_gt)
    transform_right_hand_gt = (
        torch.linalg.inv(object_pose) @ transform_1_to_0_gt @ object_pose
    )
    pose_vector_gt = pose_to_SE3_vector(transform_right_hand_gt)
    colors_right = splats_right["colors"]
    quats_right = splats_right["quats"]
    scales_right = splats_right["scales"]
    opacities_right = splats_right["opacities"]
    optimizers = [
        optim.AdamW(
            [
                {"params": params, "lr": lr, "name": name},
            ],
            betas=(0.9, 0.99),
            weight_decay=0.0,
        )
        for params, lr, name in [
            (
                rel_translation,
                config.lr * config.translation_lr_ratio,
                "rel_translation",
            ),
            (rel_rotation, config.lr, "rel_rotation"),
        ]
    ]
    optimizers += [
        optim.AdamW(
            [
                {"params": params, "lr": lr, "name": name},
            ],
            betas=(0.9, 0.99),
            weight_decay=0.0,
        )
        for params, lr, name in [
            (colors_right[:, :1], config.lr * config.sh_lr_fraction, "sh0"),
            (colors_right[:, 1:4], config.lr * config.sh_lr_fraction, "sh1"),
            (colors_right[:, 4:], config.lr * config.sh_lr_fraction, "sh2"),
            (scales_right, config.lr * config.sh_lr_fraction, "scales"),
            (
                opacities_right,
                config.lr * config.opacity_lr_fraction,
                "opacities",
            ),
            (quats_right, config.lr * config.sh_lr_fraction, "quats"),
        ]
    ]
    params, optimizers_dict = get_optimizers_and_params(optimizers)
    step_scheduler = {
        k: StepLR(opt, 20, gamma=0.1) for k, opt in optimizers_dict.items()
    }
    loss_fn = nn.L1Loss()
    rel_trans = SE3_vector_to_pose(
        torch.cat([params["rel_translation"], params["rel_rotation"]], dim=-1)
    )
    rel_trans_lhs = object_pose @ rel_trans @ torch.linalg.inv(object_pose)
    means_right, colors_right, quats_right = transform_splats(
        splats_right["means"],
        torch.cat((params["sh0"], params["sh1"], params["sh2"]), dim=1),
        params["quats"],
        rel_trans_lhs,
    )
    scales_right = params["scales"]
    opacities_right = params["opacities"]
    means = torch.cat([splats_left["means"], means_right], dim=0)
    colors = torch.cat([splats_left["colors"], colors_right], dim=0)
    quats = torch.cat([splats_left["quats"], quats_right], dim=0)
    scales = torch.cat([splats_left["scales"], scales_right], dim=0)
    opacities = torch.logit(
        torch.cat([splats_left["opacities"], opacities_right], dim=0)
    )
    resolution_schedule = schedule_resolutions(config.num_epochs, ctf_sizes)
    losses_log = []
    poses_log = []
    translation_log_x = []
    translation_log_y = []
    translation_log_z = []
    rel_rotation_log_x = []
    rel_rotation_log_y = []
    rel_rotation_log_z = []

    @torch.no_grad()
    def viewer_render_fn(camera_state, render_tab_state):
        return _viewer_render_fn(
            camera_state,
            render_tab_state,
            means,
            quats,
            scales,
            torch.sigmoid(opacities),
            colors=colors.unsqueeze(0),
            data_sh_degree=config.total_sh_degrees,
        )

    if config.enable_viewer:
        server = viser.ViserServer(verbose=False)
        viewer = GsplatViewer(
            server=server,
            render_fn=viewer_render_fn,
            output_dir=Path("/gs_output"),
            mode="training",
        )
    viewer.state = "paused"
    for epoch in tqdm(range(config.num_epochs), desc="Refining Epochs"):
        # for optimizer in optimizers:
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] = (
        #             (1 / resolution_schedule[epoch])
        #             * ctf_sizes[-1]
        #             * config.lr
        #             * (
        #                 config.translation_lr_ratio
        #                 if param_group["name"] == "rel_translation"
        #                 else 1.0
        #             )
        #         )
        rel_trans = SE3_vector_to_pose(
            torch.cat([params["rel_translation"], params["rel_rotation"]], dim=-1)
        )
        rel_trans_lhs = object_pose @ rel_trans @ torch.linalg.inv(object_pose)
        poses_log.append(rel_trans_lhs)
        means_right, colors_right, quats_right = transform_splats(
            splats_right["means"],
            torch.cat((params["sh0"], params["sh1"], params["sh2"]), dim=1),
            params["quats"],
            rel_trans_lhs,
        )
        if config.enable_viewer:
            while viewer.state == "paused":
                time.sleep(0.01)
            viewer.lock.acquire()
            tic = time.time()
        scales_right = params["scales"]
        opacities_right = params["opacities"]
        means = torch.cat([splats_left["means"], means_right], dim=0)
        colors = torch.cat([splats_left["colors"], colors_right], dim=0)
        quats = torch.cat([splats_left["quats"], quats_right], dim=0)
        scales = torch.cat([splats_left["scales"], scales_right], dim=0)
        opacities = torch.logit(
            torch.cat([splats_left["opacities"], opacities_right], dim=0)
        )
        camera_poses_right = rel_trans_lhs @ train_poses_left
        camera_poses = torch.cat([train_poses_left, camera_poses_right], dim=0)
        images = torch.cat([train_images_left, train_images_right], dim=0)
        masks = torch.cat([train_masks_left, train_masks_right], dim=0)
        images[masks == 1] = (images[masks == 1] - images[masks == 1].mean()) / (
            images[masks == 1].std() + 1e-8
        )
        images[masks == 0] = 0.0
        rendered, alphas, _ = batch_rasterize(
            means,
            quats,
            scales,
            torch.sigmoid(opacities),
            colors[:, :1],
            colors[:, 1:4],
            colors[:, 4:],
            camera_poses,
            camera_matrix,
            images,
            config,
            render_mode="RGB+D",
            backgrounds=torch.zeros((1, camera_poses.shape[0], colors.shape[-1])).to(
                images.device
            ),
            enable_sh=False,
        )
        depth = rendered[:, :, :, -1]
        rendered = rendered[:, :, :, :3]
        rendered, images, depth, alphas, masks = resize(
            rendered,
            images,
            depth,
            alphas,
            masks,
            resolution_schedule[epoch],
        )
        weights = torch.zeros_like(depth)
        weights[masks == 1] = torch.clamp(
            depth[masks == 1] / torch.median(depth[masks == 1]), 0.25, 4.0
        )
        rendered[masks == 1] = (rendered[masks == 1] - rendered[masks == 1].mean()) / (
            rendered[masks == 1].std() + 1e-8
        )
        if config.enable_viewer:
            if epoch % 50 == 0 and config.verbose:
                add_frame(server.scene, "object_pose", object_pose)
                img_width = images.shape[2]
                img_height = images.shape[1]
                fov = np.arctan2(img_width / 2, camera_matrix[0, 0].item()) * 2
                camera_poses_left = camera_poses[: camera_poses.shape[0] // 2]
                camera_poses_right = camera_poses[camera_poses.shape[0] // 2 :]
                imgs_left = rendered[rendered.shape[0] // 2]
                imgs_right = rendered[rendered.shape[0] // 2]
                camera_poses_left = camera_poses_left[camera_poses_left.shape[0] // 2]
                camera_poses_right = camera_poses_right[
                    camera_poses_right.shape[0] // 2
                ]
                imgs_left = imgs_left[imgs_left.shape[0] // 2]
                imgs_right = imgs_right[imgs_right.shape[0] // 2]
                server.scene.add_camera_frustum(
                    name=f"cam_pose_left",
                    aspect=img_width / img_height,
                    fov=fov.item(),
                    scale=1e-3,
                    line_width=0.5,
                    # image=imgs_left.detach().cpu().numpy(),
                    wxyz=R.from_matrix(
                        camera_poses_left[:3, :3].detach().cpu().numpy()
                    ).as_quat(scalar_first=True),
                    position=camera_poses_left[:3, 3].detach().cpu().numpy(),
                )
                server.scene.add_camera_frustum(
                    name=f"cam_pose_right",
                    aspect=img_width / img_height,
                    fov=fov.item(),
                    scale=1e-3,
                    line_width=0.5,
                    # image=imgs_right.detach().cpu().numpy(),
                    wxyz=R.from_matrix(
                        camera_poses_right[:3, :3].detach().cpu().numpy()
                    ).as_quat(scalar_first=True),
                    position=camera_poses_right[:3, 3].detach().cpu().numpy(),
                )
            viewer_tick(viewer, images, epoch, tic)
        masks = torch.ones_like(masks)
        loss = mask_aware_loss_img(loss_fn, rendered, alphas, images, masks)
        # weights = weights[..., 0]
        # loss *= weights
        loss = loss.mean()
        loss.backward()
        losses_log.append(loss.item())
        translation_log_x.append(rel_translation.detach().cpu().numpy()[0])
        translation_log_y.append(rel_translation.detach().cpu().numpy()[1])
        translation_log_z.append(rel_translation.detach().cpu().numpy()[2])
        rel_rotation_log_x.append(rel_rotation.detach().cpu().numpy()[0])
        rel_rotation_log_y.append(rel_rotation.detach().cpu().numpy()[1])
        rel_rotation_log_z.append(rel_rotation.detach().cpu().numpy()[2])
        for optimizer in optimizers_dict.values():
            optimizer.step()
            optimizer.zero_grad()
            clip_grad_norm_(
                parameters=[
                    p
                    for g in optimizer.param_groups
                    for p in g["params"]
                    if p.requires_grad
                ],
                max_norm=config.max_grad_norm,
            )
        for k in optimizers_dict.keys():
            step_scheduler[k].step(loss.detach())
        if epoch % 50 == 0 and config.verbose:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    losses_log = torch.stack([torch.tensor(l) for l in losses_log])
    min_loss_idx = torch.argmin(losses_log)
    best_pose = poses_log[min_loss_idx]
    del optimizer
    del optimizers
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    if config.enable_viewer:
        server.scene.reset()
        server.stop()
    result_pose_left = torch.linalg.inv(best_pose)
    return result_pose_left.detach()


if __name__ == "__main__":
    pass
