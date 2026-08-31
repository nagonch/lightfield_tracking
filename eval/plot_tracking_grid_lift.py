#!/usr/bin/env python3
"""Qualitative tracking grid on the real LiFT captures: rows = {GT, methods},
columns = time — the real-data twin of eval/plot_tracking_grid.py, with the
same cell rendering (gold silhouette outline + haloed coordinate frame,
object-centred square crops, ADD badge per row).

The real dataset has no meshes, so the silhouette comes from the same
pseudo-model used by the metrics (eval/run_eval_lift.py): the frame-0 masked
depth back-projected into the object frame, rasterised as a point splat at the
row's pose and stroked like the mesh contour.

Run inside the lift6dof container:
    python eval/plot_tracking_grid_lift.py [--seqs shiny_box_tilt]
"""

import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as R  # noqa: E402
import run_eval_lift as RL  # noqa: E402

DATASET_ROOT = "/home/ngoncharov/cvpr2026/datasets/LiFT_dataset"
RESULTS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                            "baselines_real")

# row = (folder, label, depth); "__gt__" → ground-truth pose. Mirrors the
# synthetic BASELINE_ROWS: the baselines run on sensor depth, ours on LF depth.
ROWS = [
    ("__gt__", "GT", None),
    ("results_loftr", "LoFTR", "gt"),
    ("results_fp", "FoundationPose", "gt"),
    ("results_bsdf", "BundleSDF", "gt"),
    ("tuned_v3", "Ours", "lf"),
]

# (sequence, display name) pairs — one 4-column time block per object.
SEQS = [("teabox_tilt_prod", "Tea tin"), ("jug_tilt_prod", "Steel jug")]
SEGMASK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots",
                           "segmasks")

# identical look parameters to plot_tracking_grid.py
FRAC = [0.0, 0.25, 0.85, 1.0]
CROP_SIDE = 280
CELL_PX = 340
AXIS_LEN = 0.06
PAD = 0.30
CONTOUR_COLOR = (255, 220, 0)
CONTOUR_W = 3
SPLAT_R = 4               # px radius per pseudo-model point at 1280x720
CLOSE_K = 9               # morphological closing kernel for the point splat

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")


def rgb_path(seq_dir, fid, cv_idx):
    return os.path.join(seq_dir, f"LF_{fid}", f"{cv_idx:04d}.png")


def object_center(seq_dir, fid, cv_idx):
    m = np.array(
        Image.open(os.path.join(seq_dir, f"LF_{fid}", "masks", f"{cv_idx:04d}.png"))
        .convert("L")
    )
    ys, xs = np.where(m > 10)
    cx = 0.5 * (xs.min() + xs.max())
    cy = 0.5 * (ys.min() + ys.max())
    return cx, cy, max(xs.max() - xs.min(), ys.max() - ys.min())


def draw_contour(arr, K, T, pts_obj):
    """Rasterise the pseudo-model silhouette at pose T and stroke its outline."""
    Pc = pts_obj @ T[:3, :3].T + T[:3, 3]
    z = Pc[:, 2]
    ok = z > 1e-6
    if not ok.any():
        return
    u = (K[0, 0] * Pc[ok, 0] / z[ok] + K[0, 2]).astype(np.int32)
    v = (K[1, 1] * Pc[ok, 1] / z[ok] + K[1, 2]).astype(np.int32)

    H, W = arr.shape[:2]
    sil = np.zeros((H, W), np.uint8)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    for x, y in zip(u[inb], v[inb]):
        cv2.circle(sil, (int(x), int(y)), SPLAT_R, 255, -1)
    sil = cv2.morphologyEx(
        sil, cv2.MORPH_CLOSE, np.ones((CLOSE_K, CLOSE_K), np.uint8)
    )
    contours, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return
    # The pseudo-model carries a little sensor bleed past the mask edge, whose
    # projections form small satellite blobs — stroke only the object outline.
    contours = [max(contours, key=cv2.contourArea)]
    cv2.polylines(arr, contours, True, (0, 0, 0), CONTOUR_W + 2, cv2.LINE_AA)
    cv2.polylines(arr, contours, True, CONTOUR_COLOR, CONTOUR_W, cv2.LINE_AA)


def draw_frame_halo(draw, K, T):
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
    for tip, _ in tips:
        draw.line([origin, tip], fill=(0, 0, 0), width=8)
    for tip, col in tips:
        draw.line([origin, tip], fill=col, width=4)
        r = 5
        draw.ellipse([tip[0] - r, tip[1] - r, tip[0] + r, tip[1] + r], fill=col)
    r = 4
    draw.ellipse([origin[0] - r, origin[1] - r, origin[0] + r, origin[1] + r],
                 fill=(25, 25, 25))


def render_cell(rgb, K, T, pts_obj, center, side):
    arr = np.array(rgb)
    draw_contour(arr, K, T, pts_obj)
    img = Image.fromarray(arr)
    draw_frame_halo(ImageDraw.Draw(img), K, T)
    W, H = img.size
    half = side / 2
    cx = float(np.clip(center[0], half, W - half))
    cy = float(np.clip(center[1], half, H - half))
    box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
    return img.crop(box).resize((CELL_PX, CELL_PX), Image.LANCZOS)


def pseudo_model_pts_from_mask(seq_dir, gt0, mask):
    """RL.pseudo_model_pts with a caller-supplied frame-0 mask (the displayed
    silhouette uses the predicted mask; the ADD badges keep the ground-truth
    pseudo-model so they match the tables)."""
    fid0 = RL.frame_ids(seq_dir)[0]
    K = np.loadtxt(os.path.join(seq_dir, "camera_matrix.txt"))
    depth = np.array(
        Image.open(os.path.join(seq_dir, "depth", f"{fid0}.png"))
    ).astype(np.float64) / 1000.0
    valid = mask & (depth > RL.DEPTH_ZNEAR) & (depth < RL.DEPTH_ZFAR)
    med = np.median(depth[valid])
    ys, xs = np.where(valid & (np.abs(depth - med) <= RL.MODEL_DEPTH_BAND))
    d = depth[ys, xs]
    pts_cam = np.stack(
        [(xs - K[0, 2]) * d / K[0, 0], (ys - K[1, 2]) * d / K[1, 1], d], axis=1
    )
    inv_gt0 = np.linalg.inv(gt0)
    pts_obj = pts_cam @ inv_gt0[:3, :3].T + inv_gt0[:3, 3]
    if len(pts_obj) > RL.MODEL_SAMPLE_PTS:
        rng = np.random.default_rng(0)
        pts_obj = pts_obj[rng.choice(len(pts_obj), RL.MODEL_SAMPLE_PTS,
                                     replace=False)]
    return pts_obj


def load_block(seq):
    """Everything one object's 4-column time block needs."""
    seq_dir = os.path.join(DATASET_ROOT, seq)
    cv_idx = RL.central_view_index(seq_dir)
    K = np.loadtxt(os.path.join(seq_dir, "camera_matrix.txt"))
    gt_poses = RL.load_gt_poses(seq_dir)
    metric_pts = RL.pseudo_model_pts(seq_dir, gt_poses[0])
    segmask_p = os.path.join(SEGMASK_DIR, f"{seq}.png")
    if os.path.exists(segmask_p):
        segmask = np.array(Image.open(segmask_p)) > 127
        display_pts = pseudo_model_pts_from_mask(seq_dir, gt_poses[0], segmask)
        print(f"  {seq}: silhouette from predicted mask ({segmask_p})")
    else:
        display_pts = metric_pts
        print(f"  {seq}: no segmask cache, silhouette from GT mask")
    fids = RL.frame_ids(seq_dir)

    present, pose_by_row, score_by_row = [], {}, {}
    for folder, label, depth in ROWS:
        if folder == "__gt__":
            pose_by_row[label] = gt_poses
            score_by_row[label] = None
            present.append((folder, label, depth))
            continue
        p = os.path.join(RESULTS_ROOT, folder, depth, f"{seq}.npy")
        if not os.path.exists(p):
            print(f"  skip row {label}: missing {p}")
            continue
        poses = np.load(p)
        n = min(len(poses), len(gt_poses))
        pose_by_row[label] = poses
        score_by_row[label] = R.eval_sequence(
            poses[:n], gt_poses[:n], metric_pts
        )["mean_abs_rot_deg"]
        present.append((folder, label, depth))

    N = min(len(pose_by_row[lab]) for _, lab, _ in present)
    idxs = [int(round(f * (N - 1))) for f in FRAC]
    centers = [object_center(seq_dir, fids[i], cv_idx)[:2] for i in idxs]
    max_dim = max(object_center(seq_dir, fids[i], cv_idx)[2] for i in idxs)
    side = max(CROP_SIDE, int(max_dim * (1 + PAD)))
    rgbs = {i: Image.open(rgb_path(seq_dir, fids[i], cv_idx)).convert("RGB")
            for i in idxs}
    return dict(K=K, rows=present, pose_by_row=pose_by_row,
                score_by_row=score_by_row, display_pts=display_pts,
                idxs=idxs, centers=centers, side=side, rgbs=rgbs)


def main(seq_pairs):
    blocks = [(name, load_block(seq)) for seq, name in seq_pairs]
    rows = blocks[0][1]["rows"]
    nrow, ncol = len(rows), sum(len(b["idxs"]) for _, b in blocks)

    fig, axes = plt.subplots(
        nrow, ncol,
        figsize=(ncol * 1.7, nrow * 1.7),
        gridspec_kw={"wspace": 0.0, "hspace": 0.0},
    )
    c0 = 0
    for name, b in blocks:
        for r, (folder, label, depth) in enumerate(rows):
            poses = b["pose_by_row"][label]
            for c, i in enumerate(b["idxs"]):
                cell = render_cell(b["rgbs"][i], b["K"], poses[i],
                                   b["display_pts"], b["centers"][c], b["side"])
                ax = axes[r, c0 + c]
                ax.imshow(cell)
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_color("white")
                    s.set_linewidth(1.2)
                if c0 + c == 0:
                    ax.set_ylabel(label, fontsize=9.5, fontweight="bold",
                                  labelpad=5, linespacing=0.9)
                if c == 0 and b["score_by_row"][label] is not None:
                    ax.text(
                        0.035, 0.965,
                        f"ARE {b['score_by_row'][label]:.1f}\N{DEGREE SIGN}",
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=9.5, color="white", fontweight="bold",
                        bbox=dict(facecolor="black", alpha=0.55, pad=2,
                                  edgecolor="none"),
                    )
                if r == 0:
                    ax.set_title("")
        # block header centred over its 4 columns
        mid_ax = axes[0, c0 + len(b["idxs"]) // 2]
        c0 += len(b["idxs"])
    # block headers via figure text (axes positions known after layout)
    fig.subplots_adjust(left=0.11, right=0.995, top=0.90, bottom=0.005,
                        wspace=0.0, hspace=0.0)
    c0 = 0
    for name, b in blocks:
        n = len(b["idxs"])
        x0 = axes[0, c0].get_position().x0
        x1 = axes[0, c0 + n - 1].get_position().x1
        fig.text(0.5 * (x0 + x1), 0.925, name, ha="center", va="bottom",
                 fontsize=13, fontweight="bold")
        c0 += n
    fig.suptitle("Tracking along time", fontsize=16, fontweight="bold", y=0.985)

    os.makedirs(OUT_DIR, exist_ok=True)
    stem = "tracking_grid_real_" + "_".join(s for s, _ in seq_pairs)
    for ext in ("png", "pdf"):
        out = os.path.join(OUT_DIR, f"{stem}.{ext}")
        fig.savefig(out, dpi=300)
        print("Saved →", out)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default=None,
                    help="comma-separated sequence substrings (block order)")
    args = ap.parse_args()
    pairs = SEQS
    if args.seqs:
        keys = [k for k in args.seqs.split(",") if k]
        pairs = [
            (d, d.replace("_prod", "").replace("_", " "))
            for k in keys
            for d in sorted(os.listdir(DATASET_ROOT))
            if d.endswith("_prod") and k in d
        ]
    main(pairs)
