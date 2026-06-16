"""Convergence test for photometric refinement.

For each consecutive frame pair we compute the LoFTR coarse pose exactly like
``main.py`` (using the *GT* previous pose as anchor, so the coarse error
reflects only the single-step LoFTR error), then run photometric refinement and
compare both against the GT current pose.

Goal: refinement should drive the angular — and especially the translation —
error *below* the LoFTR initialization.  Tweak the hyperparameters in the CONFIG
block and re-run.

Run:  python tune_photometric.py
"""

from __future__ import annotations

import numpy as np
import torch

from loftr_wrapper import LoftrRunner
from main import _build_pc
from src.dataset import LFDataset
from src.photometric import refine_pose_photometric
from src.pose import track_pose
from src.reflection import central_view
from analyze_photometric import _perturb_rot, _perturb_trans

# ── configuration ────────────────────────────────────────────────────────────
DATASET_ROOT = "/home/ngoncharov/SpecTrack_dataset"
SPLIT = "cube_0.0"
SEQUENCE = "cracker_box_yalehand0"
DEPTH_SOURCE = "gt"
FRAME_STRIDE = 3          # evaluate every Nth frame pair (speed)
MAX_FRAMES = 30           # cap number of evaluated pairs
ALPHA = 1.0               # cube_0.0 → pure diffuse

# init mode for the candidate pose the optimizer starts from:
#   "loftr"   → real LoFTR coarse (near-perfect single-step → hard to beat)
#   "perturb" → GT + random offset (simulates accumulated drift; refine's regime)
INIT_MODE = "perturb"
PERTURB_ROT_DEG = 6.0
PERTURB_TRANS_MM = 15.0

REFINE_KW = dict(
    num_iters=120,
    lr_rot=5e-3,
    lr_trans=1e-3,
    lambda_depth=0.1,
    lambda_mask=0.05,
    lambda_rot=0.0,
    lambda_trans=0.0,
    scales=(0.25, 0.5, 1.0),
)


# ── pose error metrics ───────────────────────────────────────────────────────


def _rot_err_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    R = Ra @ Rb.T
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def _trans_err(ta: np.ndarray, tb: np.ndarray) -> float:
    return float(np.linalg.norm(ta - tb))


def _errs(pose: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    return _rot_err_deg(pose[:3, :3], gt[:3, :3]), _trans_err(pose[:3, 3], gt[:3, 3])


# ── main ─────────────────────────────────────────────────────────────────────


def _load(dataset, idx, s_size, t_size):
    f = dataset[idx]
    depth = f["depth"].cpu().numpy()
    mask = (f["masks"][s_size // 2, t_size // 2] > 0).cpu().numpy()
    K = f["camera_matrix"].cpu().numpy().astype(np.float64)
    view = central_view(f, s_size, t_size)
    pc, color = _build_pc(depth, mask, view, K)
    gt = f["object_pose"].cpu().numpy().astype(np.float64)
    return dict(depth=depth, mask=mask, K=K, view=view, pc=pc, color=color, gt=gt)


def _perturbed_init(gt: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """GT + random rotation/translation offset (simulates accumulated drift)."""
    center = gt[:3, 3].copy()
    raxis = rng.normal(size=3)
    taxis = rng.normal(size=3)
    p = _perturb_rot(gt, raxis, np.radians(PERTURB_ROT_DEG), center)
    p = _perturb_trans(p, taxis, PERTURB_TRANS_MM / 1000.0)
    return p


def main():
    loftr = LoftrRunner() if INIT_MODE == "loftr" else None
    rng = np.random.default_rng(0)
    seq_path = f"{DATASET_ROOT}/{SPLIT}/{SEQUENCE}"
    dataset = LFDataset(seq_path, depth_source=DEPTH_SOURCE)
    s_size, t_size = dataset.metadata["n_views"]

    idxs = list(range(1, len(dataset), FRAME_STRIDE))[:MAX_FRAMES]
    print(f"{SPLIT}/{SEQUENCE}: evaluating {len(idxs)} frame pairs  (init={INIT_MODE})")
    print(f"refine kwargs: {REFINE_KW}\n")

    rows = []
    for idx in idxs:
        prev = _load(dataset, idx - 1, s_size, t_size)
        curr = _load(dataset, idx, s_size, t_size)

        if INIT_MODE == "loftr":
            coarse = track_pose(
                abs_pose_prev=prev["gt"],  # GT anchor → isolate single-step error
                diffuse_prev=prev["view"],
                diffuse_curr=curr["view"],
                depth_prev=prev["depth"],
                depth_curr=curr["depth"],
                mask_prev=prev["mask"],
                mask_curr=curr["mask"],
                K=curr["K"],
                loftr=loftr,
                alpha=ALPHA,
                pc_prev=prev["pc"],
                pc_curr=curr["pc"],
                color_prev=prev["color"],
                color_curr=curr["color"],
                rng=rng,
            )
        else:
            coarse = _perturbed_init(curr["gt"], rng)

        refined, _ = refine_pose_photometric(
            points_prev=prev["pc"],
            diffuse_prev=prev["color"],
            env_map_prev=None,
            points_curr=curr["pc"],
            diffuse_curr=curr["color"],
            env_map_curr=None,
            K=curr["K"],
            abs_pose_prev=prev["gt"],
            pose_coarse=coarse,
            alpha=ALPHA,
            depth_curr=curr["depth"],
            mask_curr=curr["mask"],
            **REFINE_KW,
        )

        cr, ct = _errs(coarse, curr["gt"])
        rr, rt = _errs(refined, curr["gt"])
        rows.append((idx, cr, ct, rr, rt))
        flag = "  rot↑" if rr > cr + 1e-6 else ""
        flag += "  trans↑" if rt > ct + 1e-6 else ""
        print(
            f"  f{idx:3d}  init: rot={cr:6.2f}° trans={ct * 1000:6.2f}mm   "
            f"refined: rot={rr:6.2f}° trans={rt * 1000:6.2f}mm{flag}"
        )

    a = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
    print("\n── mean over pairs ──")
    print(f"  init    : rot={a[:, 0].mean():6.2f}°  trans={a[:, 1].mean() * 1000:6.2f}mm")
    print(f"  refined : rot={a[:, 2].mean():6.2f}°  trans={a[:, 3].mean() * 1000:6.2f}mm")
    print(
        f"  median  init rot={np.median(a[:, 0]):.2f}° refined rot={np.median(a[:, 2]):.2f}°  "
        f"| init trans={np.median(a[:, 1]) * 1000:.2f}mm refined trans={np.median(a[:, 3]) * 1000:.2f}mm"
    )
    win_r = (a[:, 2] < a[:, 0]).mean() * 100
    win_t = (a[:, 3] < a[:, 1]).mean() * 100
    print(f"  refined better: rot {win_r:.0f}% of pairs, trans {win_t:.0f}% of pairs")


if __name__ == "__main__":
    main()
