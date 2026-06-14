"""ReLiFT-6DoF — main tracking loop.

Algorithm per frame:
  1. Load LF → SurfaceLF  (reflection separation: diffuse colour + env map)
  2. Get diffuse middle view for LoFTR
  3. Coarse pose: LoFTR on diffuse images (α-weighted) + ICP fallback
  4. Refine pose: photometric loss on canonical model re-lit with current env map
  5. Fuse current frame into canonical model
  6. Save estimated poses

Assumptions (to relax later):
  - Depth from GT or synth (USE_DEPTH_SOURCE controls which)
  - Mask from GT dataset (USE_GT_MASK=True)
  - Reflectivity known per-split (REFLECTIVITY parameter)
"""

import os
from collections.abc import Callable
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image

from src.dataset import LFDataset
from src.utilities import backproject_depth_to_pointcloud
from surface_lf import SurfaceLF, SurfaceLFRig
from canonical_model import CanonicalModel
from diffuse_view import diffuse_midview_uint8, rasterize_diffuse
from coarse_pose import mixed_coarse_pose
from loftr_wrapper import LoftrRunner
from loss import rotation_6d_to_matrix, matrix_to_rotation_6d, simple_loss
from icp import rebase_poses
from src.slf_refinement_viewer import SurfaceLFRefinementViewer

# ── configuration ─────────────────────────────────────────────────────────────

DATASET_ROOT = "/home/ngoncharov/SpecTrack_dataset"
EXP_NAME = "results_relift"
USE_GT_MASK = True
ENABLE_VIS = True

DEPTH_SOURCES = ["gt", "synth"]
SPLIT_PREFIXES = ["cube"]
REFLECTIVITIES = ["0.7", "1.0"]


# ── helpers ───────────────────────────────────────────────────────────────────


def _build_surface_lf(
    frame: dict,
    s_size: int,
    t_size: int,
    mask: torch.Tensor,
    depth: torch.Tensor,
    alpha: float,
    previous_env_map: torch.Tensor | None = None,
) -> SurfaceLF:
    camera_matrix = frame["camera_matrix"]
    pc, pc_scales = backproject_depth_to_pointcloud(
        pixel_indices=None,
        depths=depth,
        camera_matrix=camera_matrix,
        return_scales=True,
    )
    pc = pc[(mask > 0).reshape(-1)]
    pc_scales = pc_scales[(mask > 0).reshape(-1)]

    rig = SurfaceLFRig.build(
        K=frame["camera_matrix"],
        poses_4x4=frame["camera_poses_rel"].reshape(-1, 4, 4),
        image_size_hw=(frame["LF"].shape[2], frame["LF"].shape[3]),
    )
    images = (
        frame["LF"]
        .reshape(-1, frame["LF"].shape[2], frame["LF"].shape[3], 3)
        .permute(0, 3, 1, 2)
    )
    use_env = alpha < 0.99
    return SurfaceLF(
        rig,
        pc.cuda(),
        images.cuda(),
        pc_scales.cuda(),
        previous_environment_map=previous_env_map,
        separation_alpha=alpha,
        use_environment_map=use_env,
        use_relight=use_env,
        use_naive_relight=False,
    )


def _pc_and_color(surface_lf: SurfaceLF) -> tuple[np.ndarray, np.ndarray]:
    pts = surface_lf.values["means"].cpu().numpy()
    if surface_lf.diffuse_color_per_point is not None:
        col = surface_lf.diffuse_color_per_point.cpu().numpy()
    else:
        valid = surface_lf.valid.float()
        count = valid.sum(dim=0).clamp(min=1).unsqueeze(-1)
        col = (
            ((surface_lf.colors * valid.unsqueeze(-1)).sum(dim=0) / count).cpu().numpy()
        )
    return pts, col


def _canonical_env_for_slf(canonical: CanonicalModel | None) -> torch.Tensor | None:
    """Return env map in CHW format for passing to next SurfaceLF."""
    if canonical is None or canonical.environment_map is None:
        return None
    env = canonical.environment_map
    if env.ndim == 3 and env.shape[-1] == 3:
        return env.permute(2, 0, 1).contiguous()
    return env


def _reflected_image(
    raw_lf_mid: np.ndarray,
    diffuse_u8: np.ndarray,
    amplify: float = 3.0,
) -> np.ndarray:
    """Specular component = raw − diffuse, amplified for visibility."""
    raw_f = raw_lf_mid.astype(np.float32)
    dif_f = diffuse_u8.astype(np.float32) / 255.0
    reflected = np.clip((raw_f - dif_f) * amplify, 0.0, 1.0)
    return (reflected * 255).astype(np.uint8)


# ── pose refinement with canonical model ─────────────────────────────────────


def refine_pose_canonical(
    canonical: CanonicalModel,
    diffuse_target: torch.Tensor,
    depth_target: torch.Tensor,
    pose_init_abs: torch.Tensor,
    num_iterations: int = 300,
    lr_rot: float = 1e-3,
    lr_trans: float = 1e-3,
    on_step: Callable | None = None,
) -> tuple[np.ndarray, float]:
    """Gradient-based refinement against canonical model re-lit rendering.

    pose_init_abs : [4, 4] absolute object pose in camera frame (initial guess).
    diffuse_target : [H, W, 3] float diffuse image of current frame.
    depth_target   : [H, W] float depth of current frame.
    Returns refined [4, 4] absolute pose as numpy float64.
    """
    device = diffuse_target.device

    pose_t = pose_init_abs.to(device=device, dtype=torch.float32)
    rot_6d = matrix_to_rotation_6d(pose_t[:3, :3]).clone().detach().requires_grad_(True)
    trans = pose_t[:3, 3].clone().detach().requires_grad_(True)

    optimizer = torch.optim.AdamW(
        [{"params": [rot_6d], "lr": lr_rot}, {"params": [trans], "lr": lr_trans}],
    )

    best_loss = float("inf")
    best_pose_np = pose_t.cpu().numpy()

    for step in range(num_iterations):
        optimizer.zero_grad()

        R = rotation_6d_to_matrix(rot_6d)
        pose_cand = torch.eye(4, device=device, dtype=torch.float32)
        pose_cand[:3, :3] = R
        pose_cand[:3, 3] = trans

        image_r, depth_r, _ = canonical.rasterize(pose_cand)

        loss = simple_loss(
            image_r,
            diffuse_target,
            depth_r,
            depth_target,
            aggregate=True,
        )
        val = loss.item()
        if val < best_loss:
            best_loss = val
            best_pose_np = pose_cand.detach().cpu().numpy()

        if on_step is not None and step % 25 == 0:
            on_step(
                step,
                val,
                image_r.detach().cpu().numpy(),
                diffuse_target.cpu().numpy(),
            )

        loss.backward()
        optimizer.step()

    return best_pose_np, best_loss


# ── per-sequence tracker ──────────────────────────────────────────────────────


def track_sequence(
    path: str,
    results_folder: str,
    sequence_name: str,
    alpha: float,
    loftr: LoftrRunner,
    rng: np.random.Generator,
    depth_source: str = "gt",
    vis=None,
):
    dataset = LFDataset(path, depth_source=depth_source)
    s_size, t_size = dataset.metadata["n_views"]
    if vis is not None:
        try:
            vis.total_frames = len(dataset)
        except TypeError:
            vis.total_frames = 100

    gt_poses: list[np.ndarray] = []
    est_poses: list[np.ndarray] = []

    diffuse_prev_u8 = None
    depth_prev_np = None
    mask_prev_np = None
    pc_prev = None
    color_prev = None
    canonical: CanonicalModel | None = None

    for i, frame in tqdm(enumerate(dataset), desc=f"  {sequence_name}", leave=False):
        frame["pose"] = frame["object_pose"]
        gt_poses.append(frame["pose"].cpu().numpy())

        # ── depth & mask ──────────────────────────────────────────────────────
        depth = frame["depth"]

        if USE_GT_MASK:
            mask = frame["masks"][s_size // 2, t_size // 2]
        else:
            raise NotImplementedError("Predicted mask not yet wired")

        # ── SurfaceLF (reflection separation + env map) ───────────────────────
        prev_env_chw = _canonical_env_for_slf(canonical)
        surface_lf = _build_surface_lf(
            frame=frame,
            s_size=s_size,
            t_size=t_size,
            mask=mask,
            depth=depth,
            alpha=alpha,
            previous_env_map=prev_env_chw,
        )

        # ── diffuse + reflected middle views ──────────────────────────────────
        diffuse_curr_u8 = diffuse_midview_uint8(surface_lf)
        raw_mid_np = (
            frame["LF"][s_size // 2, t_size // 2].cpu().numpy()
        )  # [H,W,3] float [0,1]
        reflected_u8 = _reflected_image(raw_mid_np, diffuse_curr_u8)

        depth_curr_np = depth.cpu().numpy()
        mask_curr_np = (mask > 0).cpu().numpy()
        pc_curr, color_curr = _pc_and_color(surface_lf)

        # Per-frame env map (HWC float) for vis — from SurfaceLF if available
        slf_env_np: np.ndarray | None = None
        if (
            hasattr(surface_lf, "environment_map")
            and surface_lf.environment_map is not None
        ):
            _env = surface_lf.environment_map
            if _env.ndim == 3 and _env.shape[0] == 3:  # CHW → HWC
                slf_env_np = _env.permute(1, 2, 0).cpu().numpy()
            else:
                slf_env_np = _env.cpu().numpy()

        # ── frame 0: initialise ───────────────────────────────────────────────
        if i == 0:
            est_poses.append(frame["pose"].cpu().numpy())
            canonical = CanonicalModel.from_surface_lf(
                surface_lf=surface_lf,
                pose0=frame["pose"],
                alpha=alpha,
            )
            if vis is not None:
                pts0 = canonical.points_obj.cpu().numpy()
                cols0 = canonical.diffuse_colors.cpu().numpy()
                env0 = (
                    canonical.environment_map.cpu().numpy()
                    if canonical.environment_map is not None
                    else None
                )
                vis.update_canonical(pts0, cols0, env0, 0)
                vis.add_estimated_frame(
                    frame_idx=0,
                    pose_abs=est_poses[0],
                    pts_obj=pts0,
                    colors_obj=cols0,
                    pts_cam=pc_curr,
                    colors_cam=color_curr,
                    img_rendered=None,
                    img_diffuse=diffuse_curr_u8,
                    img_reflected=reflected_u8,
                    env_hwc=slf_env_np,
                )

        else:
            K_np = frame["camera_matrix"].cpu().numpy().astype(np.float64)

            # ── coarse pose ───────────────────────────────────────────────────
            coarse_abs_np = mixed_coarse_pose(
                alpha=alpha,
                diffuse_prev=diffuse_prev_u8,
                diffuse_curr=diffuse_curr_u8,
                depth_prev=depth_prev_np,
                depth_curr=depth_curr_np,
                mask_prev=mask_prev_np,
                mask_curr=mask_curr_np,
                K=K_np,
                loftr=loftr,
                pc_prev=pc_prev,
                pc_curr=pc_curr,
                color_prev=color_prev,
                color_curr=color_curr,
                abs_pose_prev=est_poses[-1],
                rng=rng,
            )

            # ── pose refinement via canonical relighting ──────────────────────
            diffuse_target = rasterize_diffuse(surface_lf).cuda()
            depth_target = depth.cuda()
            pose_init = torch.tensor(coarse_abs_np, dtype=torch.float32).cuda()

            if vis is not None:
                vis.reset_refinement(i)

            refined_abs_np, best_loss = refine_pose_canonical(
                canonical=canonical,
                diffuse_target=diffuse_target,
                depth_target=depth_target,
                pose_init_abs=pose_init,
                on_step=vis.on_refine_step if vis is not None else None,
            )
            est_poses.append(refined_abs_np)

            if vis is not None:
                vis.finalize_refinement(best_loss)

            # ── fuse current frame into canonical model ───────────────────────
            canonical.fuse_frame(
                surface_lf=surface_lf,
                pose_t=torch.tensor(refined_abs_np, dtype=torch.float32).cuda(),
            )

            if vis is not None:
                pts_np = canonical.points_obj.cpu().numpy()
                cols_np = canonical.diffuse_colors.cpu().numpy()
                env_np = (
                    canonical.environment_map.cpu().numpy()
                    if canonical.environment_map is not None
                    else None
                )
                vis.update_canonical(pts_np, cols_np, env_np, i)
                with torch.no_grad():
                    img_r, _, _ = canonical.rasterize(
                        torch.tensor(refined_abs_np, dtype=torch.float32).cuda()
                    )
                vis.add_estimated_frame(
                    frame_idx=i,
                    pose_abs=refined_abs_np,
                    pts_obj=pts_np,
                    colors_obj=cols_np,
                    pts_cam=pc_curr,
                    colors_cam=color_curr,
                    img_rendered=img_r.cpu().numpy(),
                    img_diffuse=diffuse_curr_u8,
                    img_reflected=reflected_u8,
                    env_hwc=slf_env_np,
                )

        # ── save env map ──────────────────────────────────────────────────────
        if canonical.environment_map is not None:
            env_np = canonical.environment_map.cpu().numpy()
            env_np = np.clip(env_np * 255, 0, 255).astype(np.uint8)
            Image.fromarray(env_np).save(
                os.path.join(results_folder, f"{sequence_name}_env_map.png")
            )

        # bookkeeping
        diffuse_prev_u8 = diffuse_curr_u8
        depth_prev_np = depth_curr_np
        mask_prev_np = mask_curr_np
        pc_prev = pc_curr
        color_prev = color_curr

    # ── save results ──────────────────────────────────────────────────────────
    gt_np = np.stack(gt_poses)
    est_np = np.stack(est_poses)
    est_rebased = rebase_poses(gt_np, est_np)
    out_path = os.path.join(results_folder, f"{sequence_name}.npy")
    np.save(out_path, est_rebased)
    tqdm.write(f"  {sequence_name}: {est_rebased.shape} → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    vis = None
    if ENABLE_VIS:
        from vis import ReLiFTVis

        vis = ReLiFTVis(port=8080)

    loftr = LoftrRunner()
    rng = np.random.default_rng(seed=42)

    for depth_source in DEPTH_SOURCES:
        for split_prefix in SPLIT_PREFIXES:
            for REFLECTIVITY in REFLECTIVITIES:
                alpha = 1.0 - float(REFLECTIVITY)
                split_dir = f"{DATASET_ROOT}/{split_prefix}_{REFLECTIVITY}"
                results_folder = (
                    f"{EXP_NAME}/{depth_source}/{split_prefix}_{REFLECTIVITY}"
                )
                os.makedirs(results_folder, exist_ok=True)

                if not os.path.isdir(split_dir):
                    tqdm.write(f"Split not found, skipping: {split_dir}")
                    continue

                for sequence_name in sorted(os.listdir(split_dir)):
                    seq_path = os.path.join(split_dir, sequence_name)
                    if not os.path.isdir(seq_path):
                        continue

                    out_path = os.path.join(results_folder, f"{sequence_name}.npy")
                    if os.path.exists(out_path):
                        tqdm.write(f"  {sequence_name}: already done, skipping")
                        continue

                    tqdm.write(
                        f"\n[{depth_source}] [{split_prefix}_{REFLECTIVITY}] {sequence_name}"
                    )

                    if vis is not None:
                        vis.new_sequence(
                            f"{depth_source}/{split_prefix}_{REFLECTIVITY}/{sequence_name}"
                        )

                    try:
                        track_sequence(
                            path=seq_path,
                            results_folder=results_folder,
                            sequence_name=sequence_name,
                            alpha=alpha,
                            loftr=loftr,
                            rng=rng,
                            depth_source=depth_source,
                            vis=vis,
                        )
                    except Exception as e:
                        import traceback

                        tqdm.write(f"  FAILED: {e}")
                        traceback.print_exc()
