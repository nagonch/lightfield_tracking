"""Simulated RealSense depth for the EPI cross capture, via two-view block matching.

View pair: the center view cam_4_4 (reference) and cam_4_8, which sits 40 mm to its
right — the closest match to a RealSense D435's 50 mm stereo baseline the cross
provides, and the widest clean horizontal pair.  Pipeline per frame:

  1. cv2.stereoRectify with the calibrated per-view K/D/extrinsics,
  2. StereoSGBM (full HH mode, internal left-right check via disp12MaxDiff,
     uniqueness + speckle filtering, median post-filter) on the rectified pair,
  3. resample the disparity back onto the *distorted* center-view pixel grid
     (cv2.undistortPoints with R1/P1 maps every center pixel into the rectified
     image), convert to metric Z in the center-camera frame,
  4. save depth/XXXXXX.png, uint16 millimeters, 960x1280, aligned to the raw center
     RGB — the same convention as the RealSense capture.

Two grades (--grade):
  * d435 (default): realistic RealSense D435 simulation.  The rectified pair is
    downscaled 2x before matching (480x640, f_rect ~ 495 -> f*B ~ 20 px*m, almost
    exactly the D435's 848x480 / f~425 / B=50mm ~ 21 px*m angular resolution), matched
    with the cheaper 5-path SGBM, and the disparity is upsampled back to the full
    rectified grid with nearest-neighbour — reproducing the blocky look of RealSense
    depth aligned to a higher-resolution color stream.
  * good: full-resolution MODE_HH matching — the best this stereo pair can do
    (written to depth_hq/ for reference; ~2x the D435's angular resolution).

Run inside the lift6dof container:
    docker exec -w "$PWD" lift6dof python epi_stereo_depth.py [--grade good]
"""

import argparse
import os

import cv2
import numpy as np
from PIL import Image

ROOT = "/home/ngoncharov/SpecTrack_dataset/captured/epi"
LEFT, RIGHT = "cam_4_4", "cam_4_8"
MIN_DISP, NUM_DISP = 8, 112  # Z range ~0.33..5 m at f~990, B~40 mm


def load_calib(seq_dir):
    K1 = np.loadtxt(os.path.join(seq_dir, "intrinsics", f"{LEFT}.txt"))
    D1 = np.loadtxt(os.path.join(seq_dir, "distortion", f"{LEFT}.txt"))
    K2 = np.loadtxt(os.path.join(seq_dir, "intrinsics", f"{RIGHT}.txt"))
    D2 = np.loadtxt(os.path.join(seq_dir, "distortion", f"{RIGHT}.txt"))
    T21 = np.loadtxt(os.path.join(seq_dir, "extrinsics", f"{RIGHT}.txt"))  # X_r = T @ X_c
    return K1, D1, K2, D2, T21[:3, :3], T21[:3, 3]


def make_matcher(grade):
    if grade == "d435":  # half-res matching: halve the disparity search range too
        return cv2.StereoSGBM_create(
            minDisparity=MIN_DISP // 2, numDisparities=64, blockSize=9,
            P1=8 * 81, P2=32 * 81,
            disp12MaxDiff=1, uniquenessRatio=10,
            speckleWindowSize=100, speckleRange=2,
            preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_SGBM)
    return cv2.StereoSGBM_create(
        minDisparity=MIN_DISP, numDisparities=NUM_DISP, blockSize=7,
        P1=8 * 49, P2=32 * 49,
        disp12MaxDiff=1, uniquenessRatio=8,
        speckleWindowSize=200, speckleRange=2,
        preFilterCap=31, mode=cv2.STEREO_SGBM_MODE_HH)


def run_sequence(seq_dir, grade):
    K1, D1, K2, D2, R21, t21 = load_calib(seq_dir)
    size = (960, 1280)  # (w, h) after the orientation fix
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, D1, K2, D2, size, R21, t21, flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)
    m1x, m1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_32FC1)
    m2x, m2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_32FC1)
    f_rect, B = P2[0, 0], -P2[0, 3] / P2[0, 0]

    # map every (distorted) center pixel into rectified-left coordinates
    W, H = size
    uv = np.stack(np.meshgrid(np.arange(W, dtype=np.float64),
                              np.arange(H, dtype=np.float64)), -1).reshape(-1, 1, 2)
    rect_uv = cv2.undistortPoints(uv, K1, D1, R=R1, P=P1).reshape(H, W, 2).astype(np.float32)

    matcher = make_matcher(grade)
    out_dir = os.path.join(seq_dir, "depth" if grade == "d435" else "depth_hq")
    os.makedirs(out_dir, exist_ok=True)
    frames = sorted(os.listdir(os.path.join(seq_dir, "images")))
    print(f"[{os.path.basename(seq_dir)}] grade={grade} f_rect={f_rect:.1f} "
          f"B={B * 1000:.1f} mm -> {out_dir}")

    for fr in frames:
        gl = cv2.imread(os.path.join(seq_dir, "images", fr, f"{LEFT}.png"),
                        cv2.IMREAD_GRAYSCALE)
        gr = cv2.imread(os.path.join(seq_dir, "images", fr, f"{RIGHT}.png"),
                        cv2.IMREAD_GRAYSCALE)
        rl = cv2.remap(gl, m1x, m1y, cv2.INTER_LINEAR)
        rr = cv2.remap(gr, m2x, m2y, cv2.INTER_LINEAR)
        if grade == "d435":
            # match at half resolution (D435 angular resolution), then bring the
            # disparity back to the full rectified grid nearest-neighbour (blocky,
            # like RealSense depth aligned to the higher-res color image)
            hl = cv2.resize(rl, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            hr = cv2.resize(rr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            dh = matcher.compute(hl, hr).astype(np.float32) / 16.0
            dh[dh < MIN_DISP // 2] = np.nan
            dh = cv2.medianBlur(dh, 5)
            disp = 2.0 * cv2.resize(dh, (rl.shape[1], rl.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
        else:
            disp = matcher.compute(rl, rr).astype(np.float32) / 16.0
            disp[disp < MIN_DISP] = np.nan
            disp = cv2.medianBlur(disp, 5)

        # nearest-neighbour lookup of disparity at each center pixel's rectified coords
        d = cv2.remap(disp, rect_uv[..., 0], rect_uv[..., 1], cv2.INTER_NEAREST)
        Zr = f_rect * B / d  # depth along rectified-left z
        # rectified-frame 3D point for each center pixel, then z in the center frame
        xr = (rect_uv[..., 0] - P1[0, 2]) / P1[0, 0] * Zr
        yr = (rect_uv[..., 1] - P1[1, 2]) / P1[1, 1] * Zr
        z_c = R1[0, 2] * xr + R1[1, 2] * yr + R1[2, 2] * Zr  # (R1^T X_rect).z
        z_c[~np.isfinite(z_c) | (z_c <= 0) | (z_c > 5.0)] = 0.0

        depth_mm = np.clip(z_c * 1000.0, 0, 65535).astype(np.uint16)
        Image.fromarray(depth_mm).save(os.path.join(out_dir, f"{fr}.png"))
        valid = (depth_mm > 0).mean()
        print(f"  {fr}: {valid * 100:.0f}% valid, median "
              f"{np.median(depth_mm[depth_mm > 0]) if valid else 0:.0f} mm", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grade", choices=["d435", "good"], default="d435")
    args = ap.parse_args()
    for seq in ["diffuse", "reflective"]:
        run_sequence(os.path.join(ROOT, seq), args.grade)
    print("done")
