from surface_lf import SurfaceLF, SurfaceLFRig
import torch
from PIL import Image
import numpy as np
import os
import torch.nn.functional as F
from loss import loss
from utils import compose_pose, matrix_to_axis_angle, plot_refinement_summary


def refine_pose(
    surface_lf_prev,
    image,
    depth,
    pose_coarse,
    i,
    mask_prev,
    mask,
    num_iterations: int = 50,
    lr_translation: float = 1e-2,
    lr_rotation: float = 5e-3,
    *,
    verbose: bool = True,
    plot: bool = True,
    plot_every: int = 0,  # 0 => only final plot for the frame
    save_plots_dir: str | None = "opt_output",
):
    device = image.device

    # --- BEFORE ---
    values_before = surface_lf_prev.transform(pose_coarse)
    surf_image_before, surf_depth_before = surface_lf_prev.rasterize(values_before)
    loss_before, breakdown_before = loss(
        surf_image_before,
        surf_depth_before,
        mask_prev,
        image,
        depth,
        mask,
        pose_coarse,
        pose_coarse,
    )

    # --- params from coarse pose ---
    R_init = pose_coarse[:3, :3]
    t_init = pose_coarse[:3, 3]

    rotation_init = matrix_to_axis_angle(R_init).detach()
    translation_param = t_init.clone().detach().requires_grad_(True)
    rotation_param = rotation_init.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam(
        [
            {"params": translation_param, "lr": lr_translation},
            {"params": rotation_param, "lr": lr_rotation},
        ]
    )

    best_loss = float("inf")
    best_pose = pose_coarse.clone()

    loss_history: list[float] = []
    t_norm_history: list[float] = []
    r_norm_history: list[float] = []

    for iteration in range(num_iterations):
        optimizer.zero_grad()

        pose_delta = compose_pose(rotation_param, translation_param)
        pose_current = pose_delta @ pose_coarse

        surf_values = surface_lf_prev.transform(pose_current)
        surf_image, surf_depth = surface_lf_prev.rasterize(surf_values)

        loss_value, breakdown = loss(
            surf_image,
            surf_depth,
            mask_prev,
            image,
            depth,
            mask,
            pose_coarse,
            pose_current,
            visualize=False,
            i=0,
        )

        loss_value.backward()
        torch.nn.utils.clip_grad_norm_([rotation_param, translation_param], 10.0)
        optimizer.step()

        lv = float(loss_value.detach().cpu().item())
        loss_history.append(lv)
        t_norm_history.append(float(translation_param.detach().norm().cpu().item()))
        r_norm_history.append(float(rotation_param.detach().norm().cpu().item()))

        if verbose and (
            iteration == 0
            or (iteration + 1) % 10 == 0
            or iteration == num_iterations - 1
        ):
            # If breakdown is a dict of tensors, you can pretty print it here.
            print(
                f"[Frame {i:04d} | it {iteration+1:03d}] loss={lv:.6f}  ||t||={t_norm_history[-1]:.4f}  ||r||={r_norm_history[-1]:.4f}"
            )

        if lv < best_loss:
            best_loss = lv
            best_pose = pose_current.detach().clone()

        if plot and plot_every > 0 and ((iteration + 1) % plot_every == 0):
            # cheap mid-iteration plot: just loss curve so far
            import matplotlib.pyplot as plt

            plt.figure()
            plt.plot(loss_history)
            plt.title(f"Frame {i}: loss up to it {iteration+1}")
            plt.xlabel("iteration")
            plt.ylabel("loss")
            if save_plots_dir is not None:
                os.makedirs(save_plots_dir, exist_ok=True)
                plt.savefig(
                    os.path.join(
                        save_plots_dir, f"frame_{i:04d}_it_{iteration+1:03d}_loss.png"
                    ),
                    dpi=150,
                )
                plt.close()
            else:
                plt.show()

    # --- AFTER (best) ---
    values_after = surface_lf_prev.transform(best_pose)
    surf_image_after, surf_depth_after = surface_lf_prev.rasterize(values_after)
    final_loss, breakdown_after = loss(
        surf_image_after,
        surf_depth_after,
        mask_prev,
        image,
        depth,
        mask,
        pose_coarse,
        best_pose,
    )

    if verbose:
        print(
            f"[Frame {i:04d}] loss_before={float(loss_before):.6f}  final_loss={float(final_loss):.6f}  best_iter_loss={best_loss:.6f}"
        )

    if plot:
        plot_refinement_summary(
            frame_index=i,
            loss_history=loss_history,
            t_norm_history=t_norm_history,
            r_norm_history=r_norm_history,
            image_obs=image,
            depth_obs=depth,
            mask_obs=mask,
            image_render_before=surf_image_before,
            depth_render_before=surf_depth_before,
            image_render_after=surf_image_after,
            depth_render_after=surf_depth_after,
            save_dir=save_plots_dir,
        )

    return best_pose


if __name__ == "__main__":
    K = torch.load("pts/K.pt")
    poses = torch.load("pts/poses_4x4.pt")
    surface_lf_rig = SurfaceLFRig.build(
        K=K,
        poses_4x4=poses,
        image_size_hw=(720, 1280),
    )
    poses = [
        torch.load(f"pts/coarse_pose_{i:04d}.pt", weights_only=True) for i in range(20)
    ]
    poses_gt = [
        torch.load(f"pts/pose_gt{i:04d}.pt", weights_only=True) for i in range(20)
    ]
    poses_rel = [torch.eye(4).cuda()]
    poses_refined = [torch.eye(4).cuda()]
    for i in range(1, 20):
        pose_rel = poses[i] @ torch.linalg.inv(poses[i - 1])
        poses_rel.append(pose_rel)
    for i in range(20):
        surface_lf = SurfaceLF(
            rig=surface_lf_rig,
            pc=torch.load(f"pts/pc_{i:04d}.pt"),
            images=torch.load(f"pts/images_{i:04d}.pt"),
        )
        mask = torch.load(f"pts/mask_{i:04d}.pt")
        image, depth = surface_lf.rasterize()
        if i > 0:
            pose_refined = refine_pose(
                surface_lf_prev, image, depth, poses_rel[i], i, mask_prev, mask
            )
            poses_refined.append(pose_refined)

        surface_lf_prev = surface_lf
        mask_prev = mask
    poses_gt = torch.stack(poses_gt, dim=0).cpu().numpy()
    poses_coarse = torch.stack(poses, dim=0).cpu().numpy()
    poses_refined = torch.stack(poses_refined, dim=0).cpu().numpy()
    from tracking import rebase_poses, pose_errors

    result_poses_before = rebase_poses(poses_gt, poses_coarse)
    result_poses_after = rebase_poses(poses_gt, poses_refined)
    print(pose_errors(poses_gt, result_poses_before))
    print(pose_errors(poses_gt, result_poses_after))
