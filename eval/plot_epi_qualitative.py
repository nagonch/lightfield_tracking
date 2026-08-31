#!/usr/bin/env python3
"""Qualitative keyframe grid for the EPI captured sequences: mesh contours per method.

For each sequence (diffuse / reflective), a grid with one row per method (GT first)
and one column per keyframe. Every panel is the undistorted center view cropped
around the GT object, with the mustard mesh silhouette contour rendered at that
method's estimated pose (solid, colored) and the GT contour as a thin green
reference. The contour is the exact rasterized mesh silhouette (all faces filled),
so it reflects the full 6-DoF pose, not just the position.

Outputs: eval/plots/epi_qual_<seq>.{pdf,png}

Run inside the lift6dof container:
    docker exec -w "$PWD" lift6dof python eval/plot_epi_qualitative.py
"""

import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_eval_captured import (  # noqa: E402
    MESH_PATH, RESULTS_ROOT, load_sequence_meta, rebase_to_gt,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")
METHODS = [  # (results dir key, display name)
    ("ours", "Ours"),
    ("bsdf", "BundleSDF"),
    ("fp", "FoundationPose"),
    ("loftr", "LoFTR"),
    ("icp", "ICP"),
    ("pnp", "PnP"),
]
KEYFRAMES = [0, 9, 18, 27, 36, 46]
CROP_W, CROP_H = 400, 500  # full-res pixels around the GT object
GT_COLOR = (40, 230, 40)
EST_COLOR = (255, 60, 60)


def silhouette(mesh, T, K, H, W):
    """Exact mesh silhouette mask (uint8 0/255) at pose T under pinhole K."""
    cam = mesh.vertices @ T[:3, :3].T + T[:3, 3]
    if (cam[:, 2] <= 1e-6).mean() > 0.5:  # mostly behind the camera: lost track
        return None
    z = np.clip(cam[:, 2], 1e-6, None)
    uv = (cam[:, :2] / z[:, None]) @ K[:2, :2].T + K[:2, 2]
    tris = uv[mesh.faces].astype(np.int32)  # [F, 3, 2]
    # drop triangles that are wildly off-screen (numerical blow-ups)
    ok = np.all(np.abs(tris.reshape(len(tris), -1)) < 20000, axis=1)
    mask = np.zeros((H, W), np.uint8)
    if ok.any():
        cv2.fillPoly(mask, list(tris[ok]), 255)
    return mask


def draw_contour(img, mask, color, thickness):
    if mask is None or not mask.any():
        return
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, cnts, -1, color, thickness, cv2.LINE_AA)


def build_figure(tag: str, gt_dir: str = "poses_object", suffix: str = ""):
    K, img_paths, gt, maps, gt0_rigid = load_sequence_meta(tag, gt_dir)
    mesh = trimesh.load(MESH_PATH, process=False)
    est = {}
    for key, _ in METHODS:
        p = os.path.join(RESULTS_ROOT, f"results_{key}", f"{tag}.npy")
        if os.path.exists(p):
            est[key] = rebase_to_gt(np.load(p).astype(np.float64), gt0_rigid, gt[0])
    rows = [("gt", "GT")] + [(k, n) for k, n in METHODS if k in est]

    fig, axes = plt.subplots(
        len(rows), len(KEYFRAMES),
        figsize=(1.55 * len(KEYFRAMES), 1.95 * len(rows)),
        squeeze=False,
    )
    for j, fr in enumerate(KEYFRAMES):
        img0 = np.asarray(Image.open(img_paths[fr]).convert("RGB"))
        if maps is not None:
            img0 = cv2.remap(img0, maps[0], maps[1], cv2.INTER_LINEAR)
        H, W = img0.shape[:2]
        gt_mask = silhouette(mesh, gt[fr], K, H, W)
        # crop centred on the GT object
        ys, xs = np.where(gt_mask > 0)
        cx, cy = int(xs.mean()), int(ys.mean())
        l = int(np.clip(cx - CROP_W // 2, 0, W - CROP_W))
        t = int(np.clip(cy - CROP_H // 2, 0, H - CROP_H))

        for i, (key, name) in enumerate(rows):
            img = img0.copy()
            if key == "gt":
                draw_contour(img, gt_mask, GT_COLOR, 4)
            else:
                draw_contour(img, gt_mask, GT_COLOR, 2)
                m = silhouette(mesh, est[key][fr], K, H, W)
                draw_contour(img, m, EST_COLOR, 4)
                lost = m is None or not m[t:t + CROP_H, l:l + CROP_W].any()
            ax = axes[i, j]
            ax.imshow(img[t:t + CROP_H, l:l + CROP_W])
            if key != "gt" and lost:
                ax.text(0.5, 0.06, "lost", transform=ax.transAxes, ha="center",
                        color="white", fontsize=8,
                        bbox=dict(facecolor="black", alpha=0.6, pad=2, lw=0))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(f"frame {fr}", fontsize=9, pad=3)
            if j == 0:
                ax.set_ylabel(name, fontsize=9, rotation=90, labelpad=4)

    fig.subplots_adjust(left=0.04, right=0.995, top=0.96, bottom=0.005,
                        wspace=0.03, hspace=0.04)
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"epi_qual_{tag.split('_', 1)[1]}{suffix}.{ext}"),
                    dpi=200)
    plt.close(fig)
    print(f"[{tag}] rows={[n for _, n in rows]} -> {OUT_DIR}/epi_qual_*.{{pdf,png}}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", default="poses_object",
                    help="poses_object (rigid) or poses_object_refined (per-frame)")
    args = ap.parse_args()
    suffix = "" if args.gt_dir == "poses_object" else "_refined"
    for tag in ("epi_diffuse", "epi_reflective"):
        build_figure(tag, args.gt_dir, suffix)
