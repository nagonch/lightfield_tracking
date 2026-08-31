#!/usr/bin/env python3
"""Combined qualitative tracking grid: synthetic sugar box (left block) and
real polished box (right block), rows = {GT, methods}, columns = time.

Same cell rendering as eval/plot_tracking_grid.py for the synthetic block
(gold mesh-silhouette outline + haloed coordinate frame, object-centred square
crops, ADD badge per row and block). The real objects have no ground-truth
meshes, so the real block draws no silhouettes at all: each cell shows only
coordinate frames — the row's estimated frame in full colour, over a faint
ground-truth frame in the method rows.

Run inside the lift6dof container:
    python eval/plot_tracking_grid_combined.py
"""

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as R  # noqa: E402
import run_eval_lift as RL  # noqa: E402
import plot_tracking_grid as PG  # noqa: E402
import plot_tracking_grid_lift as PL  # noqa: E402

SPECTRACK = "/home/ngoncharov/SpecTrack_dataset"
LIFT = "/home/ngoncharov/cvpr2026/datasets/LiFT_dataset"
SYNTH_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                          "baselines")
REAL_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                         "baselines_real")

ROWS = ["GT", "LoFTR", "FoundationPose", "BundleSDF", "Ours"]
SYNTH_FOLDERS = {"LoFTR": "results_loftr", "FoundationPose": "results_fp",
                 "BundleSDF": "results_bsdf", "Ours": "ablation_full_gt_mask"}
REAL_FOLDERS = {"LoFTR": ("results_loftr", "gt"),
                "FoundationPose": ("results_fp", "gt"),
                "BundleSDF": ("results_bsdf", "gt"),
                "Ours": ("tuned_v3", "lf")}

FRAC = PG.FRAC
OUT_DIR = PG.OUT_DIR


def synth_block(split="objects_1.0", seq="sugar_box_yalehand0"):
    seq_dir = os.path.join(SPECTRACK, split, seq)
    K = R.load_camera_matrix(seq_dir)
    gt_poses = R.load_gt_poses(seq_dir)
    model_pts = R.load_mesh_pts(SPECTRACK, split, seq)
    verts, faces = PG.load_mesh(split, seq)
    pose_files = sorted(os.listdir(os.path.join(seq_dir, "object_poses")))
    fids = [pf[5:-4] for pf in pose_files]

    poses, scores = {"GT": gt_poses}, {"GT": None}
    for lab in ROWS[1:]:
        p = os.path.join(SYNTH_BASE, SYNTH_FOLDERS[lab], "synth", split,
                         f"{seq}.npy")
        arr = np.load(p)
        n = min(len(arr), len(gt_poses))
        poses[lab] = arr
        scores[lab] = R.eval_sequence(arr[:n], gt_poses[:n], model_pts)["add_auc"]

    N = min(len(v) for v in poses.values())
    idxs = [int(round(f * (N - 1))) for f in FRAC]
    centers = [PG.object_center(seq_dir, fids[i])[:2] for i in idxs]
    max_dim = max(PG.object_center(seq_dir, fids[i])[2] for i in idxs)
    side = max(PG.CROP_SIDE, int(max_dim * (1 + PG.PAD)))
    rgbs = {i: Image.open(PG.rgb_path(seq_dir, fids[i])).convert("RGB")
            for i in idxs}

    def cell(lab, c):
        i = idxs[c]
        return PG.render_cell(rgbs[i], K, poses[lab][i], verts, faces,
                              centers[c], side)

    return dict(cell=cell, scores=scores, n=len(idxs))


def draw_frame_faint(img, K, T, alpha=140):
    """Ghosted ground-truth axes: same geometry as PL.draw_frame_halo, drawn
    semi-transparent, without the black halo, so the row's estimated frame
    stays visually dominant."""
    origin = R._project(T[:3, 3], K)
    if origin is None:
        return img
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    for name, col in R._EST_COLORS.items():
        offset = {"x": [PL.AXIS_LEN, 0, 0], "y": [0, PL.AXIS_LEN, 0],
                  "z": [0, 0, PL.AXIS_LEN]}[name]
        tip = R._project(T[:3, :3] @ np.array(offset) + T[:3, 3], K)
        if tip is None:
            continue
        d.line([origin, tip], fill=col + (alpha,), width=4)
        r = 4
        d.ellipse([tip[0] - r, tip[1] - r, tip[0] + r, tip[1] + r],
                  fill=col + (alpha,))
    r = 3
    d.ellipse([origin[0] - r, origin[1] - r, origin[0] + r, origin[1] + r],
              fill=(25, 25, 25, alpha))
    return Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")


def render_axes_cell(rgb, K, T, center, side, T_gt=None):
    """Real-block cell: no silhouettes (there is no ground-truth mesh to
    project) — only coordinate frames. The row's pose is drawn solid; method
    rows additionally get the faint ground-truth frame underneath."""
    img = rgb.copy()
    if T_gt is not None:
        img = draw_frame_faint(img, K, T_gt)
    PL.draw_frame_halo(ImageDraw.Draw(img), K, T)
    W, H = img.size
    half = side / 2
    cx = float(np.clip(center[0], half, W - half))
    cy = float(np.clip(center[1], half, H - half))
    box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
    return img.crop(box).resize((PL.CELL_PX, PL.CELL_PX), Image.LANCZOS)


def real_block(seq="shiny_box_tilt_prod"):
    seq_dir = os.path.join(LIFT, seq)
    cv_idx = RL.central_view_index(seq_dir)
    K = np.loadtxt(os.path.join(seq_dir, "camera_matrix.txt"))
    gt_poses = RL.load_gt_poses(seq_dir)
    model_pts = RL.pseudo_model_pts(seq_dir, gt_poses[0])  # ADD badges only
    fids = RL.frame_ids(seq_dir)

    poses, scores = {"GT": gt_poses}, {"GT": None}
    for lab in ROWS[1:]:
        folder, depth = REAL_FOLDERS[lab]
        arr = np.load(os.path.join(REAL_BASE, folder, depth, f"{seq}.npy"))
        n = min(len(arr), len(gt_poses))
        poses[lab] = arr
        scores[lab] = R.eval_sequence(arr[:n], gt_poses[:n], model_pts)["add_auc"]

    N = min(len(v) for v in poses.values())
    idxs = [int(round(f * (N - 1))) for f in FRAC]
    centers = [PL.object_center(seq_dir, fids[i], cv_idx)[:2] for i in idxs]
    max_dim = max(PL.object_center(seq_dir, fids[i], cv_idx)[2] for i in idxs)
    side = max(PL.CROP_SIDE, int(max_dim * (1 + PL.PAD)))
    rgbs = {i: Image.open(PL.rgb_path(seq_dir, fids[i], cv_idx)).convert("RGB")
            for i in idxs}

    def cell(lab, c):
        i = idxs[c]
        gt_faint = None if lab == "GT" else gt_poses[i]
        return render_axes_cell(rgbs[i], K, poses[lab][i], centers[c], side,
                                T_gt=gt_faint)

    return dict(cell=cell, scores=scores, n=len(idxs))


def main():
    blocks = [("Synthetic sugar box (r = 1.0)", synth_block()),
              ("Real polished box", real_block())]
    nrow = len(ROWS)
    ncol = sum(b["n"] for _, b in blocks)

    fig, axes = plt.subplots(
        nrow, ncol,
        figsize=(ncol * 1.7, nrow * 1.7),
        gridspec_kw={"wspace": 0.0, "hspace": 0.0},
    )
    c0 = 0
    for name, b in blocks:
        for r, lab in enumerate(ROWS):
            for c in range(b["n"]):
                ax = axes[r, c0 + c]
                ax.imshow(b["cell"](lab, c))
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_color("white")
                    s.set_linewidth(1.2)
                if c0 + c == 0:
                    ax.set_ylabel(lab, fontsize=9.5, fontweight="bold",
                                  labelpad=5, linespacing=0.9)
                if c == 0 and b["scores"][lab] is not None:
                    ax.text(
                        0.035, 0.965, f"ADD {b['scores'][lab]:.3f}",
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=9.5, color="white", fontweight="bold",
                        bbox=dict(facecolor="black", alpha=0.55, pad=2,
                                  edgecolor="none"),
                    )
        c0 += b["n"]
    fig.subplots_adjust(left=0.11, right=0.995, top=0.945, bottom=0.005,
                        wspace=0.0, hspace=0.0)
    c0 = 0
    for name, b in blocks:
        x0 = axes[0, c0].get_position().x0
        x1 = axes[0, c0 + b["n"] - 1].get_position().x1
        fig.text(0.5 * (x0 + x1), 0.952, name, ha="center", va="bottom",
                 fontsize=13, fontweight="bold")
        c0 += b["n"]

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        out = os.path.join(OUT_DIR, f"tracking_grid_combined.{ext}")
        fig.savefig(out, dpi=300)
        print("Saved →", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
