"""Decisive photometric-refinement strategy comparison.

For each consecutive frame pair of a sequence we compute the LoFTR coarse pose
once, then run several refinement *strategies* from that SAME coarse init and
record final rotation / translation error vs GT.  Aggregated across frames +
sequences, this tells us which parameterisation actually beats LoFTR.

Strategies:
  loftr        : no refine (reference to beat)
  cur_frozen   : current design — centroid-pivot rot, translation re-anchored to
                 LoFTR (lr_trans=0), report rotation + LoFTR translation
  joint        : honest joint 6DoF — pose=[dR@R0 | t0+dt], optimise dR & dt,
                 report exactly what is optimised, depth term anchors Z
  joint_rotonly: joint parameterisation but lr_trans=0 (rot only, consistent report)
  centroid     : translation pinned so prev centroid -> curr centroid (observed,
                 drift-free), rotation photometric, report consistent pose

Usage: python exp_strategy.py [split] [refl] [seq1,seq2,...] [n_frames]
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

from loftr_wrapper import LoftrRunner
from main import DATASET_ROOT, _build_pc
from src.dataset import LFDataset
from src.photometric import build_photometric_context, photometric_forward
from src.pose import track_pose
from src.reflection import frame_diffuse
from src.rotation import matrix_to_rotation_6d, rotation_6d_to_matrix

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "cube_0.0"
REFL = sys.argv[2] if len(sys.argv) > 2 else "0.0"
SEQS = (sys.argv[3].split(",") if len(sys.argv) > 3 else ["bleach0"])
N_FRAMES = int(sys.argv[4]) if len(sys.argv) > 4 else 19
ALPHA = 1.0 - float(REFL)
N_ITERS = 80
DEVICE = "cuda"


def _rot_err(Ra, Rb):
    c = np.clip((np.trace(Ra @ Rb.T) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def _trans_err(ta, tb):
    return float(np.linalg.norm(ta - tb))


def _optimise(ctx, slf_prev, slf_curr, pose_coarse, *, pivot, report, lr_rot, lr_trans,
              n_iters=N_ITERS):
    """Generic refinement. Returns refined 4x4 pose (np.float64).

    pivot  : 'centroid' (rotate about observed curr centroid) | 'origin'
    report : 'consistent' (report the composed pose) | 'loftr' (keep LoFTR t)
    """
    R0 = torch.from_numpy(pose_coarse[:3, :3]).float().to(DEVICE)
    t0 = torch.from_numpy(pose_coarse[:3, 3]).float().to(DEVICE)
    center = slf_curr.points.mean(0).float().to(DEVICE).detach()

    rot6 = matrix_to_rotation_6d(torch.eye(3, device=DEVICE)).clone().requires_grad_(True)
    dt = torch.zeros(3, device=DEVICE, requires_grad=True)
    opt = torch.optim.Adam([
        {"params": [rot6], "lr": lr_rot},
        {"params": [dt], "lr": lr_trans},
    ])

    def compose():
        dR = rotation_6d_to_matrix(rot6)
        pose = torch.eye(4, device=DEVICE)
        pose[:3, :3] = dR @ R0
        if pivot == "centroid":
            pose[:3, 3] = dR @ (t0 - center) + center + dt
        else:
            pose[:3, 3] = t0 + dt
        return pose

    def to_report(pose_opt):
        if report == "consistent":
            return pose_opt
        pose = pose_opt.clone()
        pose[:3, 3] = t0 + dt
        return pose

    best = pose_coarse.copy()
    for it in range(n_iters):
        opt.zero_grad()
        pose = compose()
        loss, _, _ = photometric_forward(ctx, pose)
        loss.backward()
        opt.step()
        with torch.no_grad():
            best = to_report(compose()).detach().cpu().numpy().astype(np.float64)
    return best


def _centroid_anchored(ctx, slf_prev, slf_curr, pose_coarse, abs_pose_prev, *,
                       lr_rot, n_iters=N_ITERS):
    """T_rel maps prev observed centroid -> curr observed centroid; rot photometric."""
    c_prev = slf_prev.points.mean(0).float().to(DEVICE).detach()
    c_curr = slf_curr.points.mean(0).float().to(DEVICE).detach()
    R0 = torch.from_numpy(pose_coarse[:3, :3]).float().to(DEVICE)
    t0 = torch.from_numpy(pose_coarse[:3, 3]).float().to(DEVICE)
    inv_prev = torch.linalg.inv(
        torch.from_numpy(abs_pose_prev).float().to(DEVICE))
    rot6 = matrix_to_rotation_6d(torch.eye(3, device=DEVICE)).clone().requires_grad_(True)
    opt = torch.optim.Adam([rot6], lr=lr_rot)

    best = pose_coarse.copy()
    for it in range(n_iters):
        opt.zero_grad()
        dR = rotation_6d_to_matrix(rot6)
        # relative transform: rotate prev points about their centroid, place at curr centroid
        Rrel = dR  # incremental on top of the coarse relative rotation
        # build absolute pose: rotation dR@R0, translation pins curr centroid
        Rabs = dR @ R0
        # object centroid in object frame: X̄ = R0^-1 (c_curr_coarse - t0)... approximate by
        # requiring observed centroid mapping. Use: t = c_curr - Rabs @ Xbar where
        # Xbar = R0^T (c_prev_in_obj). Simpler: pin so the *source* centroid lands on c_curr.
        pose = torch.eye(4, device=DEVICE)
        pose[:3, :3] = Rabs
        # source centroid after T_rel = pose @ inv_prev applied to prev points ≈ c_curr
        T_rel_R = Rabs @ inv_prev[:3, :3]
        pose[:3, 3] = c_curr - T_rel_R @ c_prev
        loss, _, _ = photometric_forward(ctx, pose)
        loss.backward()
        opt.step()
        with torch.no_grad():
            dR = rotation_6d_to_matrix(rot6)
            Rabs = dR @ R0
            pose = torch.eye(4, device=DEVICE)
            pose[:3, :3] = Rabs
            T_rel_R = Rabs @ inv_prev[:3, :3]
            pose[:3, 3] = c_curr - T_rel_R @ c_prev
            best = pose.detach().cpu().numpy().astype(np.float64)
    return best


STRATEGIES = {
    "cur_frozen":    dict(pivot="centroid", report="loftr",      lr_rot=5e-3, lr_trans=0.0,  ld=0.1),
    "joint_rotonly": dict(pivot="origin",   report="consistent", lr_rot=5e-3, lr_trans=0.0,  ld=0.1),
    "joint":         dict(pivot="origin",   report="consistent", lr_rot=5e-3, lr_trans=2e-4, ld=0.1),
    "joint_hidepth": dict(pivot="origin",   report="consistent", lr_rot=5e-3, lr_trans=2e-4, ld=2.0),
    "cpiv_consist":  dict(pivot="centroid", report="consistent", lr_rot=5e-3, lr_trans=2e-4, ld=2.0),
}


def main():
    loftr = LoftrRunner()
    rng = np.random.default_rng(seed=42)
    agg = {k: {"rot": [], "trans": []} for k in ["loftr", *STRATEGIES]}

    for seq in SEQS:
        seq_path = f"{DATASET_ROOT}/{SPLIT}/{seq}"
        cache_dir = f"cache/diffuse/gt/{SPLIT}/{seq}"
        if not os.path.isdir(seq_path):
            print(f"skip missing {seq_path}")
            continue
        ds = LFDataset(seq_path, depth_source="gt")
        s_size, t_size = ds.metadata["n_views"]
        prev = None
        prev_env = None
        est_prev = None
        for i, frame in enumerate(ds):
            if i >= N_FRAMES:
                break
            gt = frame["object_pose"].cpu().numpy().astype(np.float64)
            depth = frame["depth"]
            mask = frame["masks"][s_size // 2, t_size // 2]
            view, prev_env, slf, _a, _ = frame_diffuse(
                frame=frame, mask=mask, depth=depth, alpha=ALPHA,
                s_size=s_size, t_size=t_size,
                cache_path=os.path.join(cache_dir, f"diffuse_{i:04d}.png"),
                iterations=300, verbose=False, previous_environment_map=prev_env)
            depth_np = depth.cpu().numpy()
            mask_np = (mask > 0).cpu().numpy()
            K_np = frame["camera_matrix"].cpu().numpy().astype(np.float64)
            pc, color = _build_pc(depth_np, mask_np, view, K_np)

            if i == 0:
                est_prev = gt
            else:
                # NOTE: anchor each frame at GT-prev so strategies are compared on
                # the SAME per-frame coarse error, isolated from trajectory drift.
                coarse = track_pose(
                    abs_pose_prev=prev_gt, diffuse_prev=prev[0], diffuse_curr=view,
                    depth_prev=prev[1], depth_curr=depth_np, mask_prev=prev[2],
                    mask_curr=mask_np, K=K_np, loftr=loftr, alpha=ALPHA,
                    pc_prev=prev[3], pc_curr=pc, color_prev=prev[4], color_curr=color, rng=rng)
                agg["loftr"]["rot"].append(_rot_err(coarse[:3, :3], gt[:3, :3]))
                agg["loftr"]["trans"].append(_trans_err(coarse[:3, 3], gt[:3, 3]))
                for name, s in STRATEGIES.items():
                    ctx, _ = build_photometric_context(
                        slf_prev=prev[6], slf_curr=slf, env_map_prev=None, env_map_curr=None,
                        abs_pose_prev=prev_gt, pose_coarse=coarse, alpha=ALPHA,
                        lambda_depth=s["ld"], scale=1.0, device=DEVICE)
                    ref = _optimise(ctx, prev[6], slf, coarse, pivot=s["pivot"],
                                    report=s["report"], lr_rot=s["lr_rot"],
                                    lr_trans=s["lr_trans"])
                    agg[name]["rot"].append(_rot_err(ref[:3, :3], gt[:3, :3]))
                    agg[name]["trans"].append(_trans_err(ref[:3, 3], gt[:3, 3]))
                est_prev = coarse
            prev = (view, depth_np, mask_np, pc, color, prev_env, slf)
            prev_gt = gt

    print(f"\n=== {SPLIT}/{REFL}  seqs={SEQS}  frames/seq<={N_FRAMES}  iters={N_ITERS} ===")
    base_r = np.mean(agg["loftr"]["rot"])
    base_t = np.mean(agg["loftr"]["trans"]) * 1000
    for name in ["loftr", *STRATEGIES]:
        r = np.array(agg[name]["rot"])
        t = np.array(agg[name]["trans"]) * 1000
        if name == "loftr":
            print(f"  {name:28s} rot {r.mean():5.2f}°            trans {t.mean():6.2f}mm")
        else:
            lr = np.array(agg["loftr"]["rot"])
            lt = np.array(agg["loftr"]["trans"]) * 1000
            win_r = (r < lr).mean() * 100
            win_t = (t < lt).mean() * 100
            print(f"  {name:28s} rot {r.mean():5.2f}° ({r.mean()-base_r:+.2f}, win {win_r:3.0f}%)"
                  f"  trans {t.mean():6.2f}mm ({t.mean()-base_t:+.2f}, win {win_t:3.0f}%)")


if __name__ == "__main__":
    main()
