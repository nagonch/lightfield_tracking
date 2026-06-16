"""Diagnose the per-iteration rotation-error curve of photometric refine.

Pure-diffuse case (reflectivity 0.0 → alpha 1.0, no env map).  Runs the real
pipeline for the first few frames of a sequence, captures the per-iteration
diagnostics from refine_pose_photometric, and prints the rotation-error curve
per frame with pyramid-level boundaries marked, so we can see where it jumps.

Usage:  python diag_diffuse.py [sequence] [n_frames]
"""

import os
import sys

import numpy as np

from loftr_wrapper import LoftrRunner
from main import DATASET_ROOT, _build_pc
from src.dataset import LFDataset
from src.photometric import RefineConfig, refine_pose_photometric
from src.pose import track_pose
from src.reflection import frame_diffuse

SPLIT = "cube_0.0"
SEQ = sys.argv[1] if len(sys.argv) > 1 else "cracker_box_yalehand0"
N_FRAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 6
ALPHA = 1.0  # reflectivity 0.0
# START_GT=1 → start refinement *at the GT pose* to measure pure loss bias:
# how far does minimising the photometric loss drag rotation away from truth?
START_GT = os.environ.get("START_GT", "0") == "1"


def _print_curve(diag: list[dict]) -> None:
    """Compact rotation curve: one row per iter, '*' at level starts, ↑ on rises."""
    prev_rot = None
    prev_level = -1
    rises = 0
    for d in diag:
        mark = ""
        if d["level"] != prev_level:
            mark = f"  ── L{d['level']} (scale {d['scale']}) ──"
            prev_level = d["level"]
        arrow = ""
        if prev_rot is not None and d["rot_deg"] > prev_rot + 1e-4:
            arrow = " ↑"
            rises += 1
        prev_rot = d["rot_deg"]
        if d["step"] % 5 == 0 or arrow or mark:
            print(
                f"    L{d['level']} it{d['step']:3d}  lr={d['lr']:.2e}  "
                f"loss={d['loss']:.5f}  rot={d['rot_deg']:6.3f}°  "
                f"t={d['trans_mm']:6.3f}mm{arrow}{mark}"
            )
    n = len(diag)
    print(f"    [rises: {rises}/{n} iters had rotation error increase]")


def main() -> None:
    loftr = LoftrRunner()
    rng = np.random.default_rng(seed=42)
    import dataclasses as _dc

    cfg = _dc.replace(
        RefineConfig(),
        lr_rot=float(os.environ.get("LR_ROT", RefineConfig().lr_rot)),
        lambda_depth=float(os.environ.get("LAMBDA_DEPTH", RefineConfig().lambda_depth)),
    )
    print(f"START_GT={START_GT}  CONFIG: {cfg}\n")

    seq_path = f"{DATASET_ROOT}/{SPLIT}/{SEQ}"
    cache_dir = f"cache/diffuse/gt/{SPLIT}/{SEQ}"
    dataset = LFDataset(seq_path, depth_source="gt")
    s_size, t_size = dataset.metadata["n_views"]

    est_poses: list[np.ndarray] = []
    gt_poses: list[np.ndarray] = []
    prev = None
    prev_env = None

    for i, frame in enumerate(dataset):
        if i >= N_FRAMES:
            break
        gt = frame["object_pose"].cpu().numpy()
        gt_poses.append(gt)
        depth = frame["depth"]
        mask = frame["masks"][s_size // 2, t_size // 2]
        view, prev_env, slf = frame_diffuse(
            frame=frame, mask=mask, depth=depth, alpha=ALPHA,
            s_size=s_size, t_size=t_size,
            cache_path=os.path.join(cache_dir, f"diffuse_{i:04d}.png"),
            iterations=300, verbose=False, previous_environment_map=prev_env,
        )
        depth_np = depth.cpu().numpy()
        mask_np = (mask > 0).cpu().numpy()
        K_np = frame["camera_matrix"].cpu().numpy().astype(np.float64)
        pc, color = _build_pc(depth_np, mask_np, view, K_np)

        if i == 0:
            est_poses.append(gt)
        else:
            coarse = track_pose(
                abs_pose_prev=est_poses[-1], diffuse_prev=prev[0], diffuse_curr=view,
                depth_prev=prev[1], depth_curr=depth_np, mask_prev=prev[2],
                mask_curr=mask_np, K=K_np, loftr=loftr, alpha=ALPHA,
                pc_prev=prev[3], pc_curr=pc, color_prev=prev[4], color_curr=color, rng=rng,
            )
            start_pose = gt.astype(np.float64) if START_GT else coarse
            diag: list[dict] = []
            refined, _ = refine_pose_photometric(
                slf_prev=prev[6], slf_curr=slf, env_map_prev=None, env_map_curr=None,
                abs_pose_prev=est_poses[-1], pose_coarse=start_pose, alpha=ALPHA,
                cfg=cfg, gt_pose_curr=gt, diag=diag,
            )
            # in START_GT mode keep the trajectory on GT so each frame's bias is
            # measured independently from the previous-frame GT pose
            est_poses.append(gt.astype(np.float64) if START_GT else refined)
            cr = diag[0]["rot_deg"] if diag else float("nan")
            rr = diag[-1]["rot_deg"] if diag else float("nan")
            tag = "GT-start drift" if START_GT else "coarse → refined"
            print(f"\n=== frame {i}: {tag}: {cr:.3f}° → {rr:.3f}° ===")
            _print_curve(diag)

        prev = (view, depth_np, mask_np, pc, color, prev_env, slf)


if __name__ == "__main__":
    main()
