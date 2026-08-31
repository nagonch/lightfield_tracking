"""Per-frame refinement of the captured GT object poses against the segment masks.

poses_object/ is a RIGID model (flange pose @ constant ee_T_obj [@ camera-side fix]),
but the bottle is not perfectly rigidly mounted, so the projected mesh contour is
visibly offset from the object mask in some frames. This fits a small per-frame
correction delta_i (6-DoF, in the OBJECT frame — hence camera-independent) that
minimizes the symmetric boundary chamfer between the rasterized mesh silhouette and
the mask, jointly over every camera that observed that frame (realsense + epi replay
the same waypoint trajectory: arm poses agree to 0.1 mm), with a prior pulling
delta_i toward zero so the silhouette-weak directions (depth along the ray, spin
about the bottle's long axis) stay bounded. The known mesh scale is what pins depth.

    pose_c[i] = poses_object_c[i] @ delta_i        for every camera c

Reflective sequences: the foil-wrapped object is bulkier than the bare mesh, so a
free fit would drag the pose closer to inflate the silhouette. Mount flex depends on
the arm configuration, and both sequences replay the same trajectory, so the
DIFFUSE per-frame deltas are transferred to reflective; the script measures whether
reflective's own masks agree (chamfer before/after) as the validation.

Validation printed for diffuse: chamfer before/after per camera, delta magnitudes,
single-camera fits vs the joint fit (consistency), and leave-one-camera-out
(corrections fitted on one camera scored on the other).

Writes <seq>/poses_object_refined/XXXXXX.txt and <seq>/gt_refine_deltas.npy (N,4,4)
for all four sequences, plus eval/results_captured/gt_refine_report.json.

Run inside the lift6dof container (the dataset is read-only for the host user):
    docker exec -w "$PWD" lift6dof python refine_gt_poses.py
"""

import json
import os

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

ROOT = "/home/ngoncharov/SpecTrack_dataset/captured"
MESH_PATH = os.path.join(ROOT, "mustard_bottle_mesh", "textured_simple.obj")
REPORT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "eval", "results_captured", "gt_refine_report.json")
CAPTURES = ["realsense", "epi"]
DOWN = 2
# prior: penalty of 1 (px-equivalent) at 3 deg / 15 mm deviation from the rigid model
PRIOR_ROT_DEG, PRIOR_T_MM = 3.0, 15.0


class CamSeq:
    """Masks + intrinsics + rigid poses of one (capture, sequence)."""

    def __init__(self, capture, seq):
        self.dir = os.path.join(ROOT, capture, seq)
        self.K = np.loadtxt(os.path.join(self.dir, "intrinsics.txt"))
        dpath = os.path.join(self.dir, "distortion.txt")
        self.D = np.loadtxt(dpath) if os.path.exists(dpath) else None
        self.Kd = self.K.copy()
        self.Kd[:2] /= DOWN
        stems = sorted(f[:-4] for f in os.listdir(os.path.join(self.dir, "masks")))
        self.stems = stems
        self.poses = np.stack([
            np.loadtxt(os.path.join(self.dir, "poses_object", s + ".txt")) for s in stems])
        self.bnd, self.dt = [], []
        for s in stems:
            m = np.asarray(Image.open(os.path.join(self.dir, "masks", s + ".png")))
            m = (m[::DOWN, ::DOWN] > 127).astype(np.uint8)
            b = m - cv2.erode(m, np.ones((3, 3), np.uint8))
            self.bnd.append(b)
            self.dt.append(cv2.distanceTransform(1 - b, cv2.DIST_L2, 3))
        self.H, self.W = self.bnd[0].shape

    def chamfer(self, i, T, pts):
        """Symmetric boundary chamfer (px at 1/DOWN res) of the mesh at pose T."""
        cam = pts @ T[:3, :3].T + T[:3, 3]
        if self.D is not None:
            uv, _ = cv2.projectPoints(cam, np.zeros(3), np.zeros(3), self.Kd, self.D)
            uv = uv.reshape(-1, 2)
        else:
            uv = (cam[:, :2] / cam[:, 2:]) @ self.Kd[:2, :2].T + self.Kd[:2, 2]
        uv = uv.round().astype(int)
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < self.W) & (uv[:, 1] >= 0) & (uv[:, 1] < self.H)
        proj = np.zeros((self.H, self.W), np.uint8)
        proj[uv[ok, 1], uv[ok, 0]] = 1
        proj = cv2.morphologyEx(proj, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        if not proj.any():
            return 50.0
        b = proj - cv2.erode(proj, np.ones((3, 3), np.uint8))
        dtp = cv2.distanceTransform(1 - b, cv2.DIST_L2, 3)
        return float(self.dt[i][b > 0].mean() + dtp[self.bnd[i] > 0].mean())


def to_T(p):
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(p[:3]).as_matrix()
    T[:3, 3] = p[3:]
    return T


def fit_frame(cams, i, pts):
    """Joint per-frame delta (object frame) over the given cameras + rigid prior."""
    def loss(p):
        d = to_T(p)
        c = sum(cs.chamfer(i, cs.poses[i] @ d, pts) for cs in cams)
        prior = (np.linalg.norm(p[:3]) / np.deg2rad(PRIOR_ROT_DEG)) ** 2 + \
                (np.linalg.norm(p[3:]) / (PRIOR_T_MM / 1000)) ** 2
        return c + prior

    res = minimize(loss, np.zeros(6), method="Powell",
                   options=dict(xtol=1e-4, ftol=1e-4, maxiter=3000))
    return to_T(res.x)


def mag(d):
    ang = np.rad2deg(np.linalg.norm(R.from_matrix(d[:3, :3]).as_rotvec()))
    return ang, np.linalg.norm(d[:3, 3]) * 1000


def score(cams, deltas, pts):
    """Mean chamfer per camera with the given per-frame deltas (None = rigid)."""
    out = {}
    for cs in cams:
        vals = [cs.chamfer(i, cs.poses[i] @ (deltas[i] if deltas is not None else np.eye(4)),
                           pts) for i in range(len(cs.poses))]
        out[os.path.relpath(cs.dir, ROOT)] = float(np.mean(vals))
    return out


def main():
    mesh = trimesh.load(MESH_PATH, process=False)
    pts = mesh.sample(12000)
    report = {}

    # ---- diffuse: joint fit + validations ---------------------------------
    cams = [CamSeq(c, "diffuse") for c in CAPTURES]
    n = len(cams[0].poses)
    print("[diffuse] rigid chamfer:", score(cams, None, pts), flush=True)

    joint = np.stack([fit_frame(cams, i, pts) for i in range(n)])
    only = {cs: np.stack([fit_frame([cs], i, pts) for i in range(n)]) for cs in cams}
    print("[diffuse] joint-fit chamfer:", score(cams, joint, pts))
    mags = np.array([mag(d) for d in joint])
    print(f"[diffuse] joint deltas: rot median {np.median(mags[:, 0]):.2f} deg "
          f"(max {mags[:, 0].max():.2f}), trans median {np.median(mags[:, 1]):.1f} mm "
          f"(max {mags[:, 1].max():.1f})")

    # consistency between the two single-camera fits (how much is real vs ambiguous)
    dif = np.array([mag(np.linalg.inv(only[cams[0]][i]) @ only[cams[1]][i]) for i in range(n)])
    print(f"[diffuse] realsense-only vs epi-only deltas disagree by median "
          f"{np.median(dif[:, 0]):.2f} deg / {np.median(dif[:, 1]):.1f} mm")
    # leave-one-camera-out: corrections from one camera, scored on the other
    loo = {}
    for a, b in ((cams[0], cams[1]), (cams[1], cams[0])):
        rigid = score([b], None, pts)
        xfer = score([b], only[a], pts)
        k = os.path.relpath(b.dir, ROOT)
        loo[k] = dict(rigid=rigid[k], from_other_camera=xfer[k])
        print(f"[diffuse] LOO -> {k}: chamfer rigid {rigid[k]:.2f} -> "
              f"{xfer[k]:.2f} using deltas fitted on the other camera only")
    report["diffuse"] = dict(
        rigid=score(cams, None, pts), joint=score(cams, joint, pts),
        delta_rot_deg_median=float(np.median(mags[:, 0])),
        delta_rot_deg_max=float(mags[:, 0].max()),
        delta_t_mm_median=float(np.median(mags[:, 1])),
        delta_t_mm_max=float(mags[:, 1].max()),
        single_cam_disagreement_median=dict(rot_deg=float(np.median(dif[:, 0])),
                                            t_mm=float(np.median(dif[:, 1]))),
        leave_one_camera_out=loo,
        per_frame=[dict(rot_deg=float(a), t_mm=float(t)) for a, t in mags],
    )

    # ---- reflective: transfer the diffuse deltas, validate on its own masks --
    rcams = [CamSeq(c, "reflective") for c in CAPTURES]
    rigid_r, xfer_r = score(rcams, None, pts), score(rcams, joint, pts)
    print("[reflective] chamfer rigid:", rigid_r)
    print("[reflective] chamfer with transferred diffuse deltas:", xfer_r)
    report["reflective"] = dict(rigid=rigid_r, transferred_diffuse_deltas=xfer_r)

    # ---- write refined poses --------------------------------------------
    for cs_list in (cams, rcams):
        for cs in cs_list:
            out = os.path.join(cs.dir, "poses_object_refined")
            os.makedirs(out, exist_ok=True)
            for i, s in enumerate(cs.stems):
                np.savetxt(os.path.join(out, s + ".txt"), cs.poses[i] @ joint[i])
            np.save(os.path.join(cs.dir, "gt_refine_deltas.npy"), joint)
            print(f"wrote {out}")
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print("report ->", REPORT)


if __name__ == "__main__":
    main()
