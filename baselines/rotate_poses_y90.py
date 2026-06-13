#!/usr/bin/env python3
"""Rotate each pose around its own Z axis by +90 degrees (post-multiply by R_z(+90))."""

import os
import shutil
import numpy as np
from scipy.spatial.transform import Rotation as R

SRC = os.path.join(os.path.dirname(__file__), "results_fp")
DST = os.path.join(os.path.dirname(__file__), "results_fp_rotated")

# R_z(+90 deg) in homogeneous coords
R_z90 = R.from_euler("x", 90, degrees=True).as_matrix()
R_z90 = np.pad(R_z90, ((0, 1), (0, 1)), mode="constant", constant_values=0)
R_z90[3, 3] = 1

count = 0
for root, dirs, files in os.walk(SRC):
    rel = os.path.relpath(root, SRC)
    dst_dir = os.path.join(DST, rel)
    os.makedirs(dst_dir, exist_ok=True)
    for fname in files:
        src_path = os.path.join(root, fname)
        dst_path = os.path.join(dst_dir, fname)
        if fname.endswith(".npy"):
            poses = np.load(src_path, allow_pickle=True)
            # poses: (N, 4, 4) — post-multiply to rotate in local frame
            rotated = poses @ R_z90
            np.save(dst_path, rotated)
            count += 1
        else:
            shutil.copy2(src_path, dst_path)

print(f"Done. Rotated {count} .npy files -> {DST}")
