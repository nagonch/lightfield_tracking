"""Decompose the per-frame GT refinement deltas into a constant (rigid-model
residual) part and a per-frame (mount-flex candidate) part, and test which part
actually generalizes: across cameras (leave-one-camera-out) and across executions
(diffuse -> reflective). Run after refine_gt_poses.py (inside lift6dof).

Reports symmetric boundary chamfer (px at half res) for each variant.
"""

import json
import os
import sys

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from refine_gt_poses import CAPTURES, MESH_PATH, ROOT, CamSeq, fit_frame, score  # noqa

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_captured",
                   "gt_refine_decomposition.json")


def mean_delta(deltas):
    """Constant part: mean rotvec + mean translation (object frame)."""
    rv = R.from_matrix(deltas[:, :3, :3]).as_rotvec().mean(0)
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(rv).as_matrix()
    T[:3, 3] = deltas[:, :3, 3].mean(0)
    return T


def residual(deltas, const):
    return np.einsum("ij,njk->nik", np.linalg.inv(const), deltas)


def main():
    mesh = trimesh.load(MESH_PATH, process=False)
    pts = mesh.sample(12000)
    cams = {c: CamSeq(c, "diffuse") for c in CAPTURES}
    rcams = {c: CamSeq(c, "reflective") for c in CAPTURES}
    n = len(cams["realsense"].poses)
    joint = np.load(os.path.join(ROOT, "realsense", "diffuse", "gt_refine_deltas.npy"))
    const_j = mean_delta(joint)
    resid_j = residual(joint, const_j)
    rep = {"joint_constant_mm": (const_j[:3, 3] * 1000).round(2).tolist(),
           "joint_constant_deg": float(np.rad2deg(np.linalg.norm(
               R.from_matrix(const_j[:3, :3]).as_rotvec())))}

    def sc(cs, deltas):
        return score([cs], deltas, pts)[os.path.relpath(cs.dir, ROOT)]

    rep["diffuse"] = {}
    rep["reflective"] = {}
    for c in CAPTURES:
        cs, rs = cams[c], rcams[c]
        const_only = np.repeat(const_j[None], n, 0)
        rep["diffuse"][c] = dict(rigid=sc(cs, None), joint_full=sc(cs, joint),
                                 joint_constant_only=sc(cs, const_only),
                                 joint_perframe_only=sc(cs, resid_j))
        rep["reflective"][c] = dict(rigid=sc(rs, None), joint_full=sc(rs, joint),
                                    joint_constant_only=sc(rs, const_only),
                                    joint_perframe_only=sc(rs, resid_j))
        print(c, "diffuse:", {k: round(v, 2) for k, v in rep["diffuse"][c].items()})
        print(c, "reflective:", {k: round(v, 2) for k, v in rep["reflective"][c].items()})

    # single-camera fits -> constant vs per-frame parts, cross-camera transfer
    only = {c: np.stack([fit_frame([cams[c]], i, pts) for i in range(n)]) for c in CAPTURES}
    np.save(OUT.replace(".json", "_single_cam_deltas.npy"),
            np.stack([only[c] for c in CAPTURES]))
    rep["loo"] = {}
    for a, b in ((CAPTURES[0], CAPTURES[1]), (CAPTURES[1], CAPTURES[0])):
        ca = mean_delta(only[a])
        ra = residual(only[a], ca)
        target = cams[b]
        rep["loo"][f"{a}->{b}"] = dict(
            rigid=sc(target, None),
            full=sc(target, only[a]),
            constant_only=sc(target, np.repeat(ca[None], n, 0)),
            perframe_only=sc(target, ra),
            perframe_plus_own_constant=sc(target, np.einsum(
                "ij,njk->nik", mean_delta(only[b]), ra)),
        )
        print(f"LOO {a}->{b}:", {k: round(v, 2) for k, v in rep["loo"][f"{a}->{b}"].items()})
    # do the per-frame residuals of the two cameras even correlate?
    r0 = residual(only[CAPTURES[0]], mean_delta(only[CAPTURES[0]]))[:, :3, 3]
    r1 = residual(only[CAPTURES[1]], mean_delta(only[CAPTURES[1]]))[:, :3, 3]
    corr = [float(np.corrcoef(r0[:, k], r1[:, k])[0, 1]) for k in range(3)]
    rep["perframe_translation_corr_xyz"] = corr
    print("per-frame residual translation correlation between cameras (x,y,z):",
          np.round(corr, 2))
    with open(OUT, "w") as f:
        json.dump(rep, f, indent=2)
    print("->", OUT)


if __name__ == "__main__":
    main()
