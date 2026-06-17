"""Does the RELIT photometric loss bottom out at GT for a reflective object?

Builds the accumulated env map causally (like the tracker), then for a frame pair
sweeps the candidate current pose about GT (rotation, one cam axis at a time) and
prints where the relit loss is minimised. If the min is far from 0°, the env-map /
normal / alpha error has displaced the loss basin → relight can't refine reliably.

Usage: python diag_relight.py [split] [seq] [frame]
"""
import os, sys
import numpy as np
import torch

from src.dataset import LFDataset
from src.photometric import build_photometric_context, photometric_forward
from src.reflection import frame_diffuse

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "cube_1.0"
SEQ = sys.argv[2] if len(sys.argv) > 2 else "bleach0"
FRAME = int(sys.argv[3]) if len(sys.argv) > 3 else 10
ROOT = "/home/ngoncharov/SpecTrack_dataset"
ALPHA = 1.0 - float(SPLIT.split("_")[1])


def _rodrigues(axis, ang):
    a = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def main():
    ds = LFDataset(f"{ROOT}/{SPLIT}/{SEQ}", depth_source="gt")
    s, t = ds.metadata["n_views"]
    cache = f"cache/diffuse/gt/{SPLIT}/{SEQ}"
    prev_env = None
    slf_prev = slf_curr = None
    gt_prev = gt_curr = None
    for i in range(FRAME + 1):
        fr = ds[i]
        m = fr["masks"][s // 2, t // 2]
        view, prev_env, slf, alpha_i, _ = frame_diffuse(
            frame=fr, mask=m, depth=fr["depth"], alpha=None, s_size=s, t_size=t,
            cache_path=os.path.join(cache, f"diffuse_{i:04d}.png"), iterations=300,
            verbose=False, previous_environment_map=prev_env)
        if i == FRAME - 1:
            slf_prev, gt_prev, env_prev = slf, fr["object_pose"].cpu().numpy().astype(np.float64), prev_env
        if i == FRAME:
            slf_curr, gt_curr, env_curr = slf, fr["object_pose"].cpu().numpy().astype(np.float64), prev_env
    print(f"{SPLIT}/{SEQ} f{FRAME}: estimated alpha={alpha_i:.3f}  env filled={float((env_curr.sum(-1)>1e-6).float().mean()):.2f}")

    center = slf_curr.points.mean(0).cpu().numpy().astype(np.float64)
    for mode, env_p, env_c, a, tag in [
        ("diffuse", None, None, 1.0, "DIFFUSE"),
        ("relit", env_prev, env_curr, alpha_i, "RELIT (est alpha)"),
        ("relit", env_prev, env_curr, max(alpha_i*0.3, 0.05), "RELIT (low alpha)"),
    ]:
        ctx, _ = build_photometric_context(
            slf_prev=slf_prev, slf_curr=slf_curr, env_map_prev=env_p, env_map_curr=env_c,
            abs_pose_prev=gt_prev, pose_coarse=gt_curr, alpha=a, lambda_depth=0.0,
            scale=1.0, mode=mode)
        line = []
        for ax_i, axname in enumerate("XYZ"):
            axis = np.eye(3)[ax_i]
            best_x, best_l = 0.0, 1e9
            l0 = None
            for x in np.linspace(-12, 12, 49):
                Ra = _rodrigues(axis, np.radians(x))
                P = gt_curr.copy()
                P[:3, :3] = Ra @ gt_curr[:3, :3]
                P[:3, 3] = Ra @ (gt_curr[:3, 3] - center) + center
                with torch.no_grad():
                    l = photometric_forward(ctx, torch.from_numpy(P).float().cuda())[0].item()
                if abs(x) < 0.3:
                    l0 = l
                if l < best_l:
                    best_l, best_x = l, x
            line.append(f"{axname}:min@{best_x:+5.1f}° (l0={l0:.4f} lmin={best_l:.4f})")
        print(f"  {tag:20s} " + "  ".join(line))


if __name__ == "__main__":
    main()
