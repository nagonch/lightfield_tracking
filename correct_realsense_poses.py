"""Fit the constant flange->object transform and write corrected object poses.

The dataset's original pose files (now poses_arm_endeffector/) are raw robot flange
poses in the camera frame; the bottle is mounted at an unknown rigid offset from the
flange.  Since that offset is constant, every frame's segmented depth cloud, brought
into the flange frame via inv(cam_T_ee[i]), must land on the same static bottle.  So:

  1. unproject the masked depth pixels of every frame and map them to the flange frame,
  2. fuse them into one aggregate cloud (its crispness is itself a validation of the
     arm poses -- inconsistent poses would smear it),
  3. run one robust point-to-plane ICP of the mesh against the aggregate -> ee_T_obj,
  4. refine ee_T_obj depth-free by silhouette: minimize the symmetric chamfer between
     the projected mesh silhouette and the RGB-derived segment masks over all frames
     (the RealSense depth is severely noisy, so the ICP result is only the init and
     the mask term gets the final say),
  5. write poses_object/XXXXXX.txt = cam_T_ee[i] @ ee_T_obj, plus ee_T_obj.txt.

The reflective sequence is the foil-wrapped bottle: foil depth is noisy, the foil adds
thickness, and the bare-bottle mesh does not match its silhouette, so its own fit is
only a cross-check and the diffuse-sequence ee_T_obj is what gets written (same
physical mount).  Override with --per-seq.

Run inside the lift6dof container:
    docker exec -w "$PWD" lift6dof python correct_realsense_poses.py
"""

import argparse
import os

import numpy as np
import open3d as o3d
import trimesh
from PIL import Image

ROOT = "/home/ngoncharov/SpecTrack_dataset/captured/realsense"
MESH_PATH = "/home/ngoncharov/SpecTrack_dataset/captured/mustard_bottle_mesh/textured_simple.obj"
DEPTH_SCALE = 1000.0
INIT_OFFSET = np.array([0.002, 0.020, 0.019])  # earlier translation-only estimate


def load_arm_poses(seq_dir):
    d = os.path.join(seq_dir, "poses_arm_endeffector")
    return np.stack([np.loadtxt(os.path.join(d, f)) for f in sorted(os.listdir(d))])


def load_distortion(seq_dir):
    p = os.path.join(seq_dir, "distortion.txt")
    return np.loadtxt(p) if os.path.exists(p) else None


def masked_cloud(seq_dir, idx, K, D=None, stride=2, max_depth=3.5, erode=3):
    """Object depth points (camera frame) for one frame, mask eroded to drop edge noise."""
    import cv2

    depth = np.asarray(Image.open(os.path.join(seq_dir, "depth", f"{idx:06d}.png")))
    mask = np.asarray(Image.open(os.path.join(seq_dir, "masks", f"{idx:06d}.png"))) > 127
    mask = cv2.erode(mask.astype(np.uint8), np.ones((erode, erode), np.uint8)).astype(bool)
    d = depth[::stride, ::stride].astype(np.float32) / DEPTH_SCALE
    m = mask[::stride, ::stride] & (d > 0) & (d < max_depth)
    H, W = d.shape
    u, v = np.meshgrid(np.arange(W) * stride, np.arange(H) * stride)
    z = d[m]
    uv = np.stack([u[m], v[m]], -1).astype(np.float64)
    if D is not None:  # distorted pixel grid -> normalized rays
        xy = cv2.undistortPoints(uv.reshape(-1, 1, 2), K, D).reshape(-1, 2)
    else:
        xy = (uv - K[:2, 2]) / np.array([K[0, 0], K[1, 1]])
    return np.concatenate([xy * z[:, None], z[:, None]], -1)


def aggregate_flange_cloud(seq_dir, K, cam_T_ee, D=None):
    pts = []
    for i in range(len(cam_T_ee)):
        p = masked_cloud(seq_dir, i, K, D=D)
        if len(p) == 0:
            continue
        ee_T_cam = np.linalg.inv(cam_T_ee[i])
        pts.append(p @ ee_T_cam[:3, :3].T + ee_T_cam[:3, 3])
    return np.concatenate(pts)


def fit_ee_T_obj(agg_pts, mesh, init):
    """Robust ICP: mesh sample -> aggregate cloud, returns ee_T_obj."""
    src = o3d.geometry.PointCloud()  # mesh points, to be moved by ee_T_obj
    src.points = o3d.utility.Vector3dVector(mesh.sample(30000))
    tgt = o3d.geometry.PointCloud()
    tgt.points = o3d.utility.Vector3dVector(agg_pts)
    tgt = tgt.voxel_down_sample(0.002)
    tgt, _ = tgt.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

    T = init.copy()
    for max_dist in (0.03, 0.015, 0.008):
        reg = o3d.pipelines.registration.registration_icp(
            src, tgt, max_dist, T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(
                o3d.pipelines.registration.TukeyLoss(k=0.01)),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
        T = reg.transformation.copy()
    return T, reg, tgt


def eval_fit(tgt, mesh, T):
    """Median/mean distance of the (cleaned) observed cloud to the posed mesh."""
    posed = o3d.geometry.PointCloud()
    posed.points = o3d.utility.Vector3dVector(
        mesh.sample(60000) @ T[:3, :3].T + T[:3, 3])
    d = np.asarray(tgt.compute_point_cloud_distance(posed)) * 1000
    return np.median(d), d.mean(), (d < 10).mean()


def silhouette_refine(seq_dir, K, cam_T_ee, mesh, T_init, down=4, step=1, maxiter=6000,
                      D=None, side="obj"):
    """Depth-free refinement against the segment masks.

    Loss per frame (masks downsampled by `down`): symmetric chamfer between the
    rasterized projected-mesh silhouette and the mask, via distance transforms.
    Optimized over 6 params (rotvec, t) with Powell.

    side="obj": refines the flange-side transform, pose = cam_T_ee @ (T_init @ dT);
                returns the full T_init @ dT.
    side="cam": refines a constant camera-side correction with the object transform
                FIXED at T_init, pose = dT @ cam_T_ee @ T_init; returns dT alone.
                This models a hand-eye (cam_T_base) calibration error, which shows up
                as a frame-constant shift in the camera frame and cannot be absorbed
                by any flange-side constant.
    """
    import cv2
    from scipy.optimize import minimize
    from scipy.spatial.transform import Rotation as R

    cam_T_ee = cam_T_ee[::step]
    frame_ids = list(range(0, len(cam_T_ee) * step, step))
    pts = mesh.sample(12000)
    Kd = K.copy()
    Kd[:2] /= down
    masks, dts, bnds = [], [], []
    for i in frame_ids:
        m = np.asarray(Image.open(os.path.join(seq_dir, "masks", f"{i:06d}.png")))
        m = (m[::down, ::down] > 127).astype(np.uint8)
        masks.append(m)
        b = m - cv2.erode(m, np.ones((3, 3), np.uint8))
        bnds.append(b)
        dts.append(cv2.distanceTransform(1 - b, cv2.DIST_L2, 3))  # dist to mask BOUNDARY
    Hd, Wd = masks[0].shape
    k5 = np.ones((5, 5), np.uint8)
    k3 = np.ones((3, 3), np.uint8)

    def loss(p):
        dT = np.eye(4)
        dT[:3, :3] = R.from_rotvec(p[:3]).as_matrix()
        dT[:3, 3] = p[3:]
        T = T_init @ dT
        total = 0.0
        for i in range(len(cam_T_ee)):
            P = dT @ cam_T_ee[i] @ T_init if side == "cam" else cam_T_ee[i] @ T
            cam = pts @ P[:3, :3].T + P[:3, 3]
            if D is not None:  # project into the distorted pixel grid, like the masks
                uv, _ = cv2.projectPoints(cam, np.zeros(3), np.zeros(3), Kd, D)
                uv = uv.reshape(-1, 2).round().astype(int)
            else:
                uv = cam @ Kd.T
                uv = (uv[:, :2] / uv[:, 2:]).round().astype(int)
            ok = (uv[:, 0] >= 0) & (uv[:, 0] < Wd) & (uv[:, 1] >= 0) & (uv[:, 1] < Hd)
            proj = np.zeros((Hd, Wd), np.uint8)
            proj[uv[ok, 1], uv[ok, 0]] = 1
            proj = cv2.morphologyEx(proj, cv2.MORPH_CLOSE, k5)
            if not proj.any():
                total += 50.0
                continue
            # symmetric chamfer between the two silhouette BOUNDARIES (interior
            # pixels would dilute the size/rotation signal)
            bnd = proj - cv2.erode(proj, k3)
            dt_proj = cv2.distanceTransform(1 - bnd, cv2.DIST_L2, 3)
            total += dts[i][bnd > 0].mean() + dt_proj[bnds[i] > 0].mean()
        return total / len(cam_T_ee)

    res = minimize(loss, np.zeros(6), method="Powell",
                   options=dict(xtol=1e-4, ftol=1e-4, maxiter=maxiter))
    dT = np.eye(4)
    dT[:3, :3] = R.from_rotvec(res.x[:3]).as_matrix()
    dT[:3, 3] = res.x[3:]
    out = dT if side == "cam" else T_init @ dT
    return out, loss(np.zeros(6)), res.fun


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT,
                    help="capture root with diffuse/ and reflective/ (realsense or epi)")
    ap.add_argument("--per-seq", action="store_true",
                    help="write each sequence's own fit instead of the diffuse one")
    ap.add_argument("--init", default=None,
                    help="path to an ee_T_obj.txt to initialize from (e.g. the "
                         "realsense fit for the epi capture — same physical mount)")
    ap.add_argument("--no-icp", action="store_true",
                    help="skip the depth ICP stage (use with --init when the depth "
                         "is too unreliable even for initialization)")
    ap.add_argument("--fit-cam", action="store_true",
                    help="keep ee_T_obj FIXED at --init and instead fit a constant "
                         "camera-side correction dT_cam (models a hand-eye "
                         "calibration error of this rig); implies --no-icp")
    args = ap.parse_args()
    if args.fit_cam:
        assert args.init, "--fit-cam requires --init (the known ee_T_obj)"
        args.no_icp = True

    mesh = trimesh.load(MESH_PATH, process=False)
    if args.init:
        init = np.loadtxt(args.init)
    else:
        init = np.eye(4)
        init[:3, 3] = INIT_OFFSET

    fits = {}
    for seq in ["diffuse", "reflective"]:
        seq_dir = os.path.join(args.root, seq)
        K = np.loadtxt(os.path.join(seq_dir, "intrinsics.txt"))
        D = load_distortion(seq_dir)
        cam_T_ee = load_arm_poses(seq_dir)
        if args.no_icp:
            T_icp = init.copy()
        else:
            agg = aggregate_flange_cloud(seq_dir, K, cam_T_ee, D=D)
            T_icp, reg, tgt = fit_ee_T_obj(agg, mesh, init)
            med, mean, frac = eval_fit(tgt, mesh, T_icp)
            print(f"[{seq}] {len(agg)} masked pts -> {len(tgt.points)} cleaned | "
                  f"icp fitness {reg.fitness:.3f}, rmse {reg.inlier_rmse * 1000:.1f} mm | "
                  f"cloud->mesh median {med:.1f} mm, mean {mean:.1f} mm, "
                  f"{frac * 100:.0f}% < 10 mm")
        if args.fit_cam:
            dT1, l0, _ = silhouette_refine(seq_dir, K, cam_T_ee, mesh, T_icp,
                                           down=4, D=D, side="cam")
            ee_adj = np.einsum("ij,njk->nik", dT1, cam_T_ee)
            dT2, _, l1 = silhouette_refine(seq_dir, K, ee_adj, mesh, T_icp,
                                           down=2, step=2, maxiter=2500, D=D,
                                           side="cam")
            dT_cam, T = dT2 @ dT1, T_icp
            cang = np.rad2deg(np.arccos(np.clip(
                (np.trace(dT_cam[:3, :3]) - 1) / 2, -1, 1)))
            fits[seq] = dict(T=T, dT_cam=dT_cam, cam_T_ee=cam_T_ee, seq_dir=seq_dir)
            print(f"[{seq}] cam-side fit: boundary chamfer {l0:.2f} px (down4, at init) "
                  f"-> {l1:.2f} px (down2) | dT_cam: t = "
                  f"{np.round(dT_cam[:3, 3] * 1000, 1)} mm, rot = {cang:.2f} deg")
            continue
        T1, l0, _ = silhouette_refine(seq_dir, K, cam_T_ee, mesh, T_icp, down=4, D=D)
        T, _, l1 = silhouette_refine(seq_dir, K, cam_T_ee, mesh, T1,
                                     down=2, step=2, maxiter=2500, D=D)
        dsil = np.linalg.inv(T_icp) @ T
        ang = np.rad2deg(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))
        fits[seq] = dict(T=T, dT_cam=np.eye(4), cam_T_ee=cam_T_ee, seq_dir=seq_dir)
        print(f"[{seq}] silhouette refine: boundary chamfer {l0:.2f} px (down4, at icp) "
              f"-> {l1:.2f} px (down2), moved "
              f"{np.linalg.norm(dsil[:3, 3]) * 1000:.1f} mm from icp")
        print(f"[{seq}] final ee_T_obj: t = {np.round(T[:3, 3] * 1000, 1)} mm, "
              f"rot = {ang:.2f} deg")

    which = "T" if not args.fit_cam else "dT_cam"
    dT = np.linalg.inv(fits["diffuse"][which]) @ fits["reflective"][which]
    dang = np.rad2deg(np.arccos(np.clip((np.trace(dT[:3, :3]) - 1) / 2, -1, 1)))
    print(f"diffuse-vs-reflective fit disagreement: "
          f"{np.linalg.norm(dT[:3, 3]) * 1000:.1f} mm, {dang:.2f} deg")

    for seq, f in fits.items():
        ref = f if args.per_seq else fits["diffuse"]
        T, dC = ref["T"], ref["dT_cam"]
        out = os.path.join(f["seq_dir"], "poses_object")
        os.makedirs(out, exist_ok=True)
        for i, cam_T_ee in enumerate(f["cam_T_ee"]):
            np.savetxt(os.path.join(out, f"{i:06d}.txt"), dC @ cam_T_ee @ T)
        np.savetxt(os.path.join(f["seq_dir"], "ee_T_obj.txt"), T)
        if args.fit_cam:
            np.savetxt(os.path.join(f["seq_dir"], "cam_correction.txt"), dC,
                       header="constant camera-side correction dT_cam (hand-eye "
                              "residual): cam_T_obj = dT_cam @ cam_T_ee @ ee_T_obj")
        print(f"[{seq}] wrote {len(f['cam_T_ee'])} poses_object + ee_T_obj.txt "
              f"({'own fit' if args.per_seq else 'diffuse fit'})")


if __name__ == "__main__":
    main()
