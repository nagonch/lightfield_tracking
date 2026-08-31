"""Materialize the captured EPI cross sequences in the real-LiFT dataset layout.

The pipeline (main_lift.py -> track_sequence) consumes S x T light-field grids in
the LiFT on-disk format. The EPI cross's 17 views (middle row + column of a 9x9)
are representable as a 1 x 17 grid: sorted view names put the center cam_4_4 at
flat index 8 = 17//2, exactly where the pipeline's central-view indexing looks,
and each view's true orientation/position lives in camera_poses/ (the pipeline
never assumes a rectangular baseline layout beyond central indexing; the LF
plane-sweep uses per-view relative translations).

Per sequence, writes to <dst>/epi_<seq>_prod/:
    camera_matrix.txt     center-view pinhole K (undistortion target for ALL views)
    metadata.json         n_views [1,17], neighbour spacing 10 mm
    gdino_prompt.txt      segmentor prompt (marks the sequence as lift-format)
    camera_poses/00XX.txt view->world (world = center camera frame) = inv(extrinsic)
    LF_%04d/00XX.png      the 17 views, each undistorted from its own (K_i, D_i)
                          onto the common center K
    LF_%04d/masks/        the Cutie center mask (undistorted), replicated per view
                          (track_sequence only reads the central one)
    depth/%04d.png        simulated-RealSense stereo depth, undistorted (nearest)
    object_poses/%04d.txt poses_object (already in the center-camera frame)

Self-calibration (refine_epi_extrinsics.py): the shipped calibration leaves a few
side cameras (cam_1_4, cam_4_3, cam_7_4, cam_8_4) 3-8 px mis-registered against
the centre view. --corrections <view_corrections.json> folds the fitted per-view
delta pose (R_d, t_d), expressed in the rectified/centre-oriented frame of the
uncorrected copy, into the rectification (R = inv(R_i @ R_d)) and the camera
centre (c_k' = -R_d^T t_d). --drop excludes views (ablation); the centre view is
then re-located by the pipeline via len(views)//2, so drops must stay symmetric
(equal count on either side of index 8 in sorted order).

Run inside the lift6dof container:
    docker exec -w "$PWD" lift6dof python convert_epi_to_lift.py
    docker exec -w "$PWD" lift6dof python convert_epi_to_lift.py \
        --corrections ~/cvpr2026/datasets/EPI_LF_dataset/view_corrections.json \
        --dst-root ~/cvpr2026/datasets/EPI_LF_dataset_cal
"""

import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image

SRC_ROOT = "/home/ngoncharov/SpecTrack_dataset/captured/epi"
DST_ROOT = "/home/ngoncharov/cvpr2026/datasets/EPI_LF_dataset"
PROMPTS = {"diffuse": "yellow mustard bottle",
           "reflective": "object wrapped in aluminum foil"}
SIZE = (960, 1280)  # (w, h)


def convert(seq: str, dst_root: str = DST_ROOT, corrections: dict | None = None,
            drop: tuple[int, ...] = ()):
    src = os.path.join(SRC_ROOT, seq)
    dst = os.path.join(dst_root, f"epi_{seq}_prod")
    os.makedirs(dst, exist_ok=True)

    views_all = sorted(f[:-4] for f in os.listdir(os.path.join(src, "images", "000000")))
    assert len(views_all) == 17 and views_all[8] == "cam_4_4", views_all
    keep = [k for k in range(17) if k not in drop]
    assert 8 in keep and keep.index(8) == len(keep) // 2, "drops must be symmetric about the centre"
    views = [views_all[k] for k in keep]
    K_c = np.loadtxt(os.path.join(src, "intrinsics.txt"))

    # Per-view undistort + RECTIFY onto the common center K and center orientation.
    # Folding each view's small rotation (0.2-0.7 deg) into the resampling matters:
    # at f~990 it shifts pixels by ~3-9 px — same order as the ~12 px neighbour
    # disparity — which would alias straight into the plane-sweep depth (the LF
    # estimator models view offsets as pure global shifts). After rectification
    # every camera shares the center orientation and camera_poses are pure
    # translations, making that model exact.
    maps, cam_poses = [], []
    for k, v in zip(keep, views):
        K_i = np.loadtxt(os.path.join(src, "intrinsics", f"{v}.txt"))
        D_i = np.loadtxt(os.path.join(src, "distortion", f"{v}.txt"))
        T_i = np.loadtxt(os.path.join(src, "extrinsics", f"{v}.txt"))  # X_view = T @ X_c
        R_i, t_i = T_i[:3, :3], T_i[:3, 3]
        # Self-calibration delta in the rectified frame: X_rect = R_d X_c + t_d.
        # A centre-oriented ray d maps to rectified-image ray R_d d, so the
        # corrected rectification samples the old map at K_c R_d K_c^-1 u, i.e.
        # R_new^-1 = R_i @ R_d; the camera centre becomes -R_d^T t_d.
        if corrections is not None and str(k) in corrections and k != 8:
            R_d = np.asarray(corrections[str(k)]["R"], dtype=np.float64)
            t_d = np.asarray(corrections[str(k)]["t"], dtype=np.float64)
        else:
            R_d, t_d = np.eye(3), -(-R_i.T @ t_i)
        # initUndistortRectifyMap looks up X = R^-1 K_new^-1 u, then distorts with
        # (K_i, D_i); we need X = R_i @ R_d @ (K_c^-1 u), so R = (R_i R_d)^-1.
        maps.append(cv2.initUndistortRectifyMap(
            K_i, D_i, np.linalg.inv(R_i @ R_d), K_c, SIZE, cv2.CV_32FC1))
        pose = np.eye(4)  # center orientation; camera position in the center frame
        pose[:3, 3] = -R_d.T @ t_d
        cam_poses.append(pose)

    np.savetxt(os.path.join(dst, "camera_matrix.txt"), K_c)
    with open(os.path.join(dst, "metadata.json"), "w") as f:
        json.dump({"n_views": [1, len(views)], "x_spacing": 0.01, "y_spacing": 0.01}, f)
    with open(os.path.join(dst, "views.json"), "w") as f:
        json.dump({"views": views, "source_index": keep,
                   "corrections": corrections is not None, "dropped": list(drop)}, f)
    with open(os.path.join(dst, "gdino_prompt.txt"), "w") as f:
        f.write(PROMPTS[seq] + "\n")
    os.makedirs(os.path.join(dst, "camera_poses"), exist_ok=True)
    for k, P in enumerate(cam_poses):
        np.savetxt(os.path.join(dst, "camera_poses", f"{k:04d}.txt"), P)

    os.makedirs(os.path.join(dst, "depth"), exist_ok=True)
    os.makedirs(os.path.join(dst, "object_poses"), exist_ok=True)
    frames = sorted(os.listdir(os.path.join(src, "images")))
    cmx, cmy = maps[keep.index(8)]
    for i, fr in enumerate(frames):
        fdir = os.path.join(dst, f"LF_{i:04d}")
        mdir = os.path.join(fdir, "masks")
        os.makedirs(mdir, exist_ok=True)
        for k, v in enumerate(views):
            img = cv2.imread(os.path.join(src, "images", fr, f"{v}.png"))
            mx, my = maps[k]
            cv2.imwrite(os.path.join(fdir, f"{k:04d}.png"),
                        cv2.remap(img, mx, my, cv2.INTER_LINEAR))
        mask = np.asarray(Image.open(os.path.join(src, "masks", f"{fr}.png")))
        mask_u = cv2.remap(mask, cmx, cmy, cv2.INTER_NEAREST)
        for k in range(len(views)):
            Image.fromarray(mask_u).save(os.path.join(mdir, f"{k:04d}.png"))

        depth = np.asarray(Image.open(os.path.join(src, "depth", f"{fr}.png")))
        Image.fromarray(cv2.remap(depth, cmx, cmy, cv2.INTER_NEAREST)).save(
            os.path.join(dst, "depth", f"{i:04d}.png"))
        pose = np.loadtxt(os.path.join(src, "poses_object", f"{fr}.txt"))
        np.savetxt(os.path.join(dst, "object_poses", f"{i:04d}.txt"), pose)
        if i % 10 == 0:
            print(f"  [{seq}] frame {i + 1}/{len(frames)}", flush=True)
    print(f"[{seq}] -> {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dst-root", default=DST_ROOT)
    ap.add_argument("--corrections", default=None,
                    help="view_corrections.json from refine_epi_extrinsics.py")
    ap.add_argument("--drop", default="", help="comma-separated source view indices to exclude")
    ap.add_argument("--seqs", default="diffuse,reflective")
    args = ap.parse_args()
    corr = None
    if args.corrections:
        with open(args.corrections) as f:
            corr = json.load(f)
    drop = tuple(int(d) for d in args.drop.split(",") if d)
    for seq in args.seqs.split(","):
        convert(seq, args.dst_root, corr, drop)
    print("done")
