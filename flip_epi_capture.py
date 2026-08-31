"""One-shot, in-place orientation fix for the captured EPI cross sequences.

The cross-shaped EPI module was mounted rotated 90 deg CCW, so every captured image
is rotated 90 deg CCW (optical table on the right, robot pointing left).  This script
rotates the data back, 90 deg CLOCKWISE, making the scene upright and consistent with
the RealSense capture:

  * images: rotated 90 deg CW (1280x960 -> 960x1280), pixel map (u,v) -> (H-1-v, u)
  * view names: the old horizontal arm becomes the vertical arm and vice versa:
        cam_4_j -> cam_j_4          (old right = new down, index preserved)
        cam_i_4 -> cam_4_(8-i)      (old up    = new right, index reversed)
    verified against the calibrated extrinsics (e.g. old cam_0_4 sits 40 mm up ->
    new cam_4_8, 40 mm right).
  * intrinsics (per-view + top-level):  fx' = fy, fy' = fx, cx' = H-1-cy, cy' = cx
  * distortion (per-view + top-level):  radial k1,k2,k3 invariant; tangential
    (p1', p2') = (p2, -p1)   [camera coords rotate as x' = -y, y' = x]
  * extrinsics (X_view = T @ X_center): T' = R T R^T with R = Rz(90 deg CW frame fix)
  * poses/*.txt (flange pose in the old center-camera frame): T' = R @ T
  * writes baseline.txt (neighbouring-view baseline from calibration_results_2d.yml)

A marker file orientation_fixed.txt makes the script refuse to run twice.
Run inside the lift6dof container:
    docker exec -w "$PWD" lift6dof python flip_epi_capture.py
"""

import os

import numpy as np
from PIL import Image

ROOT = "/home/ngoncharov/SpecTrack_dataset/captured/epi"
H_OLD = 960  # pre-rotation image height
BASELINE_M = 0.010  # size_baseline_{hori,vert}_expected: 10 mm in calibration_results_2d.yml

# camera-frame change: X_new = R @ X_old  (new x = -old y, new y = old x)
R_FIX = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
R_FIX4 = np.eye(4)
R_FIX4[:3, :3] = R_FIX


def new_name(name: str) -> str:
    i, j = int(name[4]), int(name[6])
    if i == 4 and j == 4:
        ni, nj = 4, 4
    elif i == 4:
        ni, nj = j, 4
    else:
        ni, nj = 4, 8 - i
    return f"cam_{ni}_{nj}"


def rot_K(K):
    K2 = np.eye(3)
    K2[0, 0], K2[1, 1] = K[1, 1], K[0, 0]
    K2[0, 2] = H_OLD - 1 - K[1, 2]
    K2[1, 2] = K[0, 2]
    return K2


def rot_D(D):
    k1, k2, p1, p2, k3 = D
    return np.array([k1, k2, p2, -p1, k3])


def rename_dir_files(d, transform):
    """Rename cam_*.* files via the arm swap (through temp names), applying transform."""
    fnames = [f for f in os.listdir(d) if f.startswith("cam_")]
    for f in fnames:
        os.rename(os.path.join(d, f), os.path.join(d, "tmp_" + f))
    for f in fnames:
        stem, ext = os.path.splitext(f)
        src = os.path.join(d, "tmp_" + f)
        dst = os.path.join(d, new_name(stem) + ext)
        transform(src, dst)
        if os.path.exists(src):
            os.remove(src)


def fix_sequence(seq_dir: str):
    marker = os.path.join(seq_dir, "orientation_fixed.txt")
    if os.path.exists(marker):
        print(f"[{os.path.basename(seq_dir)}] already fixed, skipping")
        return

    # images: rotate 90 deg CW + arm-swap rename, per frame directory
    img_root = os.path.join(seq_dir, "images")
    frames = sorted(os.listdir(img_root))
    for k, fr in enumerate(frames):
        def rot_image(src, dst):
            Image.open(src).transpose(Image.ROTATE_270).save(dst)

        rename_dir_files(os.path.join(img_root, fr), rot_image)
        if k % 10 == 0:
            print(f"  images {fr} ({k + 1}/{len(frames)})", flush=True)

    # per-view calibration files
    def conv_K(src, dst):
        np.savetxt(dst, rot_K(np.loadtxt(src)))

    def conv_D(src, dst):
        np.savetxt(dst, rot_D(np.loadtxt(src)))

    def conv_E(src, dst):
        T = np.loadtxt(src)
        np.savetxt(dst, R_FIX4 @ T @ R_FIX4.T,
                   header="X_view = T @ X_center, meters (orientation-fixed)")

    rename_dir_files(os.path.join(seq_dir, "intrinsics"), conv_K)
    rename_dir_files(os.path.join(seq_dir, "distortion"), conv_D)
    rename_dir_files(os.path.join(seq_dir, "extrinsics"), conv_E)

    # top-level center-view intrinsics/distortion
    np.savetxt(os.path.join(seq_dir, "intrinsics.txt"),
               rot_K(np.loadtxt(os.path.join(seq_dir, "intrinsics.txt"))))
    np.savetxt(os.path.join(seq_dir, "distortion.txt"),
               rot_D(np.loadtxt(os.path.join(seq_dir, "distortion.txt"))))

    # flange poses in the center-camera frame
    pose_dir = os.path.join(seq_dir, "poses")
    for f in sorted(os.listdir(pose_dir)):
        p = os.path.join(pose_dir, f)
        np.savetxt(p, R_FIX4 @ np.loadtxt(p))

    np.savetxt(os.path.join(seq_dir, "baseline.txt"), [BASELINE_M],
               header="baseline between neighbouring views, meters "
                      "(from calibration_results_2d.yml)")

    with open(marker, "w") as fh:
        fh.write("Images/calibration/poses rotated 90 deg CW by flip_epi_capture.py; "
                 "view arms swapped: cam_4_j -> cam_j_4, cam_i_4 -> cam_4_(8-i).\n")
    print(f"[{os.path.basename(seq_dir)}] done")


if __name__ == "__main__":
    for seq in ["diffuse", "reflective"]:
        fix_sequence(os.path.join(ROOT, seq))
