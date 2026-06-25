#!/usr/bin/env python3
"""Qualitative tracking grid: rows = {GT, methods}, columns = time.

For every cell we draw, on the central RGB view:
  * the silhouette (projected mesh contour) of where that row's pose places
    the object, and
  * that pose's coordinate frame, sitting on top of the contour.

The top row is the ground-truth pose (its contour hugs the real object in the
RGB); each method row shows where *that* method thinks the object is, so drift
reads as the contour + frame sliding off the real object.

Chosen sequence: objects_1.0 / sugar_box1 with *synthetic / estimated* depth —
the one where our method beats every baseline (PnP, ICP, FoundationPose,
BundleSDF) by the largest margin on ADD-S AUC and ATE.

Run inside the lift6dof container:
    python eval/plot_tracking_grid.py
"""

import argparse
import os
import sys

import cv2
import numpy as np
import trimesh
from PIL import Image, ImageDraw

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as R  # noqa: E402

# ── what to plot ──────────────────────────────────────────────────────────────
DATASET_ROOT = "/home/ngoncharov/SpecTrack_dataset"
BASELINES_ROOT = os.path.join(os.path.dirname(R.__file__), "..", "baselines")

# ── BASELINE comparison (GT row + 4 trackers, our estimated/synth depth) ──────
BASELINE_SEQS = [
    ("objects_1.0", "sugar_box1"),
    ("objects_1.0", "sugar_box_yalehand0"),
    ("objects_0.7", "sugar_box1"),
    ("objects_1.0", "bleach_hard_00_03_chaitanya"),
    ("cube_1.0", "sugar_box1"),
]
# row = (folder, label, depth); "__gt__" → ground-truth pose (depth irrelevant)
BASELINE_ROWS = [
    ("__gt__", "GT", None),
    ("results_loftr", "LoFTR", "synth"),
    ("results_fp", "FoundationPose", "synth"),
    ("results_bsdf", "BundleSDF", "synth"),
    ("ablation_full_gt_mask", "Ours", "synth"),
]

# ── ABLATION of our method (objects @ reflectivity 1.0, our estimated LF depth) ─
# Modules ablated on our LF (light-field) depth; the last row swaps LF → raw
# sensor depth. ADD-S drops monotonically: 0.861 → 0.829 → 0.805 → 0.280.
ABLATION_SEQS = [
    ("objects_1.0", "sugar_box1"),          # #1 by ADD-S evolution (1.4→1.7→2.0→10.0 cm)
    ("objects_1.0", "cracker_box_yalehand0"),
    ("objects_1.0", "cracker_box_reorient"),
]
ABLATION_ROWS = [
    ("__gt__", "GT", None),
    ("ablation_full_gt_mask", "Full model", "lf"),
    ("ablation_no_refine_gt_mask", "− photometric\nrefinement", "lf"),
    ("ablation_no_separation_gt_mask", "− reflection\nseparation", "lf"),
    ("ablation_synth_depth_gt_mask",
     "− light field depth\n(instead: synth depth)", "synth"),
]

SKIP_SEQS = {"tomato_soup_can_yalehand0"}   # excluded from eval (bad data)


def all_sequences():
    """Every (split, seq) across cube_*/objects_* reflectivities, sorted."""
    out = []
    for split in sorted(os.listdir(DATASET_ROOT)):
        if not (split.startswith("cube_") or split.startswith("objects_")):
            continue
        sd = os.path.join(DATASET_ROOT, split)
        if not os.path.isdir(sd):
            continue
        for seq in sorted(os.listdir(sd)):
            if os.path.isdir(os.path.join(sd, seq)) and seq not in SKIP_SEQS \
                    and seq != "models":
                out.append((split, seq))
    return out


# mode → (rows, sequences, filename-stem, metric-key, badge-label, title)
# title=None → per-figure title built from the sequence; sequences=None → all
MODES = {
    "baseline": (BASELINE_ROWS, BASELINE_SEQS, "tracking_grid", "add_auc", "ADD",
                 "Tracking along time"),
    "ablation": (ABLATION_ROWS, ABLATION_SEQS, "ablation_grid", "add_auc", "ADD",
                 "Qualitative ablation"),
    "appendix": (BASELINE_ROWS, None, "appendix_grid", "add_auc", "ADD", None),
}

# first / 25% / 85% / 100% of the sequence
FRAC = [0.0, 0.25, 0.85, 1.0]

CROP_SIDE = 280           # square crop (px) in the 640×480 frame
CELL_PX = 340             # rendered cell resolution
AXIS_LEN = 0.06           # metres
PAD = 0.30                # extra margin factor when auto-sizing the crop
CONTOUR_COLOR = (255, 220, 0)   # silhouette outline (gold), RGB
CONTOUR_W = 3

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")


def rgb_path(seq_dir, frame_id):
    return os.path.join(seq_dir, f"LF_{frame_id}", "0012.png")


def mask_path(seq_dir, frame_id):
    return os.path.join(seq_dir, f"LF_{frame_id}", "masks", f"{R.CENTRAL_VIEW:04d}.png")


def object_center(seq_dir, frame_id):
    """Bounding-box center (cx, cy) of the object mask, plus its max dimension."""
    m = np.array(Image.open(mask_path(seq_dir, frame_id)).convert("L"))
    ys, xs = np.where(m > 10)
    cx = 0.5 * (xs.min() + xs.max())
    cy = 0.5 * (ys.min() + ys.max())
    return cx, cy, max(xs.max() - xs.min(), ys.max() - ys.min())


def load_mesh(split, seq):
    name = R.get_object_name(DATASET_ROOT, split, seq)
    p = os.path.join(DATASET_ROOT, "object_meshes", name, "textured_simple.obj")  # noqa: E501
    m = trimesh.load(p, force="mesh", process=False)
    return np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int32)


def draw_contour(arr, K, T, verts, faces):
    """Rasterise the mesh silhouette at pose T and stroke its outline onto arr (RGB)."""
    Vc = (T[:3, :3] @ verts.T).T + T[:3, 3]
    z = Vc[:, 2]
    u = K[0, 0] * Vc[:, 0] / z + K[0, 2]
    v = K[1, 1] * Vc[:, 1] / z + K[1, 2]
    pix = np.stack([u, v], axis=1)

    good = (z[faces] > 1e-6).all(axis=1)
    tris = pix[faces[good]].astype(np.int32)        # (k, 3, 2)
    if len(tris) == 0:
        return

    H, W = arr.shape[:2]
    sil = np.zeros((H, W), np.uint8)
    cv2.fillPoly(sil, tris, 255)
    contours, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # black halo then colour, so the outline reads on both the box and the floor
    cv2.polylines(arr, contours, True, (0, 0, 0), CONTOUR_W + 2, cv2.LINE_AA)
    cv2.polylines(arr, contours, True, CONTOUR_COLOR, CONTOUR_W, cv2.LINE_AA)


def draw_frame_halo(draw, K, T):
    """Coordinate frame with a black halo so the bright axes pop on any background."""
    origin = R._project(T[:3, 3], K)
    if origin is None:
        return
    tips = []
    for name, col in R._EST_COLORS.items():
        offset = {"x": [AXIS_LEN, 0, 0], "y": [0, AXIS_LEN, 0],
                  "z": [0, 0, AXIS_LEN]}[name]
        tip = R._project(T[:3, :3] @ np.array(offset) + T[:3, 3], K)
        if tip is not None:
            tips.append((tip, col))
    for tip, _ in tips:                                  # halo pass
        draw.line([origin, tip], fill=(0, 0, 0), width=8)
    for tip, col in tips:                                # colour pass
        draw.line([origin, tip], fill=col, width=4)
        r = 5
        draw.ellipse([tip[0] - r, tip[1] - r, tip[0] + r, tip[1] + r], fill=col)
    r = 4
    draw.ellipse([origin[0] - r, origin[1] - r, origin[0] + r, origin[1] + r],
                 fill=(25, 25, 25))


def render_cell(rgb, K, T, verts, faces, center, side):
    arr = np.array(rgb)                                  # RGB uint8
    draw_contour(arr, K, T, verts, faces)
    img = Image.fromarray(arr)
    draw_frame_halo(ImageDraw.Draw(img), K, T)

    W, H = img.size
    half = side / 2
    cx = float(np.clip(center[0], half, W - half))
    cy = float(np.clip(center[1], half, H - half))
    box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
    return img.crop(box).resize((CELL_PX, CELL_PX), Image.LANCZOS)


def auc_metric(poses, gt_poses, model_pts, key):
    """ADD / ADD-S AUC for a full estimated track (matches eval/run_eval.py)."""
    n = min(len(poses), len(gt_poses))
    return R.eval_sequence(poses[:n], gt_poses[:n], model_pts)[key]


def main(split, seq, rows, stem_prefix, metric_key, metric_label, title):
    seq_dir = os.path.join(DATASET_ROOT, split, seq)
    K = R.load_camera_matrix(seq_dir)
    gt_poses = R.load_gt_poses(seq_dir)
    model_pts = R.load_mesh_pts(DATASET_ROOT, split, seq)
    pose_files = sorted(os.listdir(os.path.join(seq_dir, "object_poses")))
    frame_ids = [pf[5:-4] for pf in pose_files]
    verts, faces = load_mesh(split, seq)

    # pose source + score per row (GT row uses GT poses, no score). Rows whose
    # .npy is missing are skipped so the batch never crashes on a gap.
    present_rows, pose_by_row, score_by_row = [], {}, {}
    for folder, label, depth in rows:
        if folder == "__gt__":
            pose_by_row[label] = gt_poses
            score_by_row[label] = None
            present_rows.append((folder, label, depth))
            continue
        p = os.path.join(BASELINES_ROOT, folder, depth, split, f"{seq}.npy")
        if not os.path.exists(p):
            print(f"  skip row {label}: missing {p}")
            continue
        poses = np.load(p)
        pose_by_row[label] = poses
        score_by_row[label] = auc_metric(poses, gt_poses, model_pts, metric_key)
        present_rows.append((folder, label, depth))
    rows = present_rows

    # sample frames within the length common to GT and every present track
    N = min(len(pose_by_row[lab]) for _, lab, _ in rows)
    idxs = [int(round(f * (N - 1))) for f in FRAC]
    centers = [object_center(seq_dir, frame_ids[i])[:2] for i in idxs]
    max_dim = max(object_center(seq_dir, frame_ids[i])[2] for i in idxs)
    side = max(CROP_SIDE, int(max_dim * (1 + PAD)))

    if title is None:                      # per-figure title from the sequence
        cls, _, refl = split.partition("_")
        title = f"{seq}    ({cls}, ρ = {refl})"

    rgbs = {i: Image.open(rgb_path(seq_dir, frame_ids[i])).convert("RGB") for i in idxs}

    nrow, ncol = len(rows), len(idxs)
    fig, axes = plt.subplots(
        nrow, ncol,
        figsize=(ncol * 1.7, nrow * 1.7),
        gridspec_kw={"wspace": 0.0, "hspace": 0.0},
    )

    for r, (folder, label, depth) in enumerate(rows):
        poses = pose_by_row[label]
        for c, i in enumerate(idxs):
            cell = render_cell(rgbs[i], K, poses[i], verts, faces, centers[c], side)
            ax = axes[r, c]
            ax.imshow(cell)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("white")
                s.set_linewidth(1.2)
            if c == 0:
                ax.set_ylabel(label, fontsize=9.5, fontweight="bold", labelpad=5,
                              linespacing=0.9)
                if score_by_row[label] is not None:
                    ax.text(
                        0.035, 0.965, f"{metric_label} {score_by_row[label]:.3f}",
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=9.5, color="white", fontweight="bold",
                        bbox=dict(facecolor="black", alpha=0.55, pad=2,
                                  edgecolor="none"),
                    )

    fig.subplots_adjust(left=0.215, right=0.995, top=0.945, bottom=0.005,
                        wspace=0.0, hspace=0.0)
    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.985)

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = f"{stem_prefix}_{split}_{seq}"
    for ext in ("png", "pdf"):
        out = os.path.join(OUT_DIR, f"{stem}.{ext}")
        fig.savefig(out, dpi=300)
        print("Saved →", out)
    plt.close(fig)
    print(f"Sequence: {split}/{seq}  N={N}  frames={idxs}  crop={side}px")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(MODES), default="baseline")
    ap.add_argument("--split", default=None, help="render a single split/seq instead of the batch")
    ap.add_argument("--seq", default=None)
    args = ap.parse_args()

    rows, seqs, stem_prefix, metric_key, metric_label, title = MODES[args.mode]
    if args.split and args.seq:
        seqs = [(args.split, args.seq)]
    elif seqs is None:
        seqs = all_sequences()

    for sp, sq in seqs:
        main(sp, sq, rows, stem_prefix, metric_key, metric_label, title)
