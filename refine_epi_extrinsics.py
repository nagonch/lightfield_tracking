#!/usr/bin/env python3
"""Self-calibrate the EPI side cameras against the centre view.

Motivation: with the shipped calibration (calibration_results_2d.yml) 14 of the 16
side views register to the centre view to ~0.3 px after rectification, but a few
cameras (cam_4_3, cam_8_4, cam_1_4) sit 3-5 mm off the nominal 10 mm grid and
their views land 3-8 px off the plane-sweep prediction. Those views inflate the
surface-light-field cross-view variance (alpha estimator reads a diffuse object
as a mirror), bias the LF depth and hence the coarse LoFTR/Kabsch poses.

Method (on the LiFT-format copy written by convert_epi_to_lift.py, i.e. views
already undistorted + rectified to the centre orientation):
  * LoFTR-match the centre view to every side view on several frames
    (object AND background, any pixel with LF depth),
  * back-project the centre pixels with the LF plane-sweep depth -> 3D points X_c,
  * solvePnPRansac + LM refine for the side camera pose (R_d, t_d):
        u_k = pi(K_c, R_d X_c + t_d),   initialised at (I, -c_k) (the calibration).
  * Output per view the correction (R_d, t_d) and its size (rotation deg, camera
    centre shift mm) plus the reprojection residual before/after.

convert_epi_to_lift.py --corrections <json> folds (R_d, t_d) into the
rectification (R = inv(R_i @ R_d)) and camera_poses (c_k' = -R_d^T t_d).
Fit both captures separately: the rig is identical, so the two fits must agree.

Run inside lift6dof:
    docker exec -w "$PWD" lift6dof python refine_epi_extrinsics.py
"""

import argparse
import json
import os
import sys

os.environ.setdefault("CONFIG_OVERRIDES", "config_lift.yaml")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from loftr_baseline import _resize_for_loftr
from loftr_wrapper import LoftrRunner
from src.dataset import LFDataset, set_lift_options

DST_ROOT = "/home/ngoncharov/cvpr2026/datasets/EPI_LF_dataset"


def collect(ds, loftr, frames, K, max_per_view=30000):
    """Per side view: 3D centre-frame points (LF depth) and matched side pixels."""
    S, T = ds.metadata["n_views"]
    c = (S // 2) * T + T // 2
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    data = {k: ([], []) for k in range(S * T) if k != c}
    for i in frames:
        fr = ds[i]
        LF = fr["LF"].reshape(S * T, *fr["LF"].shape[2:]).cpu().numpy().astype(np.float32)
        depth = fr["depth"].cpu().numpy()
        H, W = depth.shape
        sc, s = _resize_for_loftr(LF[c], 1280)
        for k in data:
            so, _ = _resize_for_loftr(LF[k], 1280)
            corr = loftr.predict(sc[None], so[None])[0]
            uv_c = corr[:, :2] / s
            uv_k = corr[:, 2:4] / s
            us = np.round(uv_c[:, 0]).astype(int)
            vs = np.round(uv_c[:, 1]).astype(int)
            ok = (us >= 0) & (us < W) & (vs >= 0) & (vs < H)
            z = np.zeros(len(us))
            z[ok] = depth[vs[ok], us[ok]]
            ok &= (z > 0.3) & (z < 3.0)
            X = np.stack([(uv_c[ok, 0] - cx) / fx * z[ok], (uv_c[ok, 1] - cy) / fy * z[ok], z[ok]], 1)
            data[k][0].append(X)
            data[k][1].append(uv_k[ok])
    out = {}
    rng = np.random.default_rng(0)
    for k, (Xs, Us) in data.items():
        X, U = np.concatenate(Xs), np.concatenate(Us)
        if len(X) > max_per_view:
            sel = rng.choice(len(X), max_per_view, replace=False)
            X, U = X[sel], U[sel]
        out[k] = (X, U)
    return out, c


def fit_view(X, U, K, c_k, reproj_thr=2.0):
    """PnP for the side camera: u = pi(K, R X + t), init at (I, -c_k)."""
    rvec0 = np.zeros((3, 1))
    tvec0 = (-c_k).reshape(3, 1).astype(np.float64)
    X64 = np.ascontiguousarray(X, dtype=np.float64)
    U64 = np.ascontiguousarray(U, dtype=np.float64)

    def resid(rvec, tvec):
        proj = cv2.projectPoints(X64, rvec, tvec, K, None)[0][:, 0, :]
        return np.linalg.norm(proj - U64, axis=1)

    before = resid(rvec0, tvec0)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(
        X64, U64, K, None, rvec0.copy(), tvec0.copy(), useExtrinsicGuess=True,
        iterationsCount=300, reprojectionError=reproj_thr, confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok or inl is None or len(inl) < 50:
        raise RuntimeError("PnP failed")
    inl = inl[:, 0]
    rvec, tvec = cv2.solvePnPRefineLM(X64[inl], U64[inl], K, None, rvec, tvec)
    after = resid(rvec, tvec)
    Rd = cv2.Rodrigues(rvec)[0]
    td = tvec[:, 0]
    return Rd, td, before, after, len(inl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default="epi_diffuse_prod,epi_reflective_prod")
    ap.add_argument("--frames", default="0,4,8,12,16,20,24,28,32,36,40,44")
    ap.add_argument("--out", default=os.path.join(DST_ROOT, "view_corrections.json"))
    args = ap.parse_args()
    frames = [int(f) for f in args.frames.split(",")]
    set_lift_options(view_stride=1)
    loftr = LoftrRunner()

    fits = {}
    for seq in args.seqs.split(","):
        ds = LFDataset(os.path.join(DST_ROOT, seq), depth_source="lf")
        K = ds.camera_matrix.numpy().astype(np.float64)
        rel = ds[0]["camera_poses_rel"].reshape(-1, 4, 4).cpu().numpy()
        data, c = collect(ds, loftr, frames, K)
        fits[seq] = {}
        print(f"\n== {seq}: PnP self-calibration of side views (frames {frames}) ==")
        print(f" view | n     | inl   | resid px median before -> after | rot corr deg | centre shift dx dy dz (mm) | new centre (mm)")
        for k in sorted(data):
            X, U = data[k]
            c_k = rel[k, :3, 3]
            Rd, td, before, after, ninl = fit_view(X, U, K, c_k)
            c_new = -Rd.T @ td
            dc = (c_new - c_k) * 1000
            rot = np.rad2deg(np.linalg.norm(R.from_matrix(Rd).as_rotvec()))
            fits[seq][k] = dict(R=Rd.tolist(), t=td.tolist(), c_old=c_k.tolist(), c_new=c_new.tolist(),
                                rot_deg=float(rot), resid_before=float(np.median(before)),
                                resid_after=float(np.median(after)), n=int(len(X)), inliers=int(ninl))
            print(f" {k:4d} | {len(X):5d} | {ninl:5d} | {np.median(before):6.2f} -> {np.median(after):5.2f}            | "
                  f"{rot:7.3f}      | {dc[0]:6.2f} {dc[1]:6.2f} {dc[2]:6.2f}       | {np.round(c_new * 1000, 2)}", flush=True)

    seqs = list(fits)
    if len(seqs) == 2:
        print("\n== consistency between the two captures (same rig): centre shift disagreement (mm), rotation disagreement (deg) ==")
        for k in sorted(fits[seqs[0]]):
            a, b = fits[seqs[0]][k], fits[seqs[1]][k]
            dc = (np.array(a["c_new"]) - np.array(b["c_new"])) * 1000
            dR = np.rad2deg(np.linalg.norm(R.from_matrix(np.array(a["R"]) @ np.array(b["R"]).T).as_rotvec()))
            print(f" view {k:2d}: |dc| {np.linalg.norm(dc):5.2f} mm ({dc[0]:+.2f} {dc[1]:+.2f} {dc[2]:+.2f})   dR {dR:.3f} deg")
    # Averaged correction (rotation: average rotvec; translation: mean) -> one file for the rig.
    merged = {}
    for k in sorted(fits[seqs[0]]):
        Rs = [np.array(fits[s][k]["R"]) for s in seqs]
        ts = [np.array(fits[s][k]["t"]) for s in seqs]
        rv = np.mean([R.from_matrix(m).as_rotvec() for m in Rs], 0)
        merged[str(k)] = dict(R=R.from_rotvec(rv).as_matrix().tolist(), t=np.mean(ts, 0).tolist(),
                              per_seq={s: fits[s][k] for s in seqs})
    with open(args.out, "w") as f:
        json.dump(merged, f, indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
