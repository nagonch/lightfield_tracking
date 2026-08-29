#!/usr/bin/env python3
"""Side-by-side method-comparison GIFs for the real LiFT dataset.

For every sequence, tiles one panel per method (central RGB view + dashed GT
axes + solid estimated axes, method name and per-frame rotation error in the
corner) into a grid and writes one GIF, so all trackers can be compared
back-to-back on the same frames.

Usage (inside the lift6dof container):
    python eval/make_lift_comparison_gifs.py                      # default 6 methods
    python eval/make_lift_comparison_gifs.py --scale 0.4 --cols 3
    python eval/make_lift_comparison_gifs.py \
        --method "Ours=baselines_real/ablation_full_gt_mask/gt" \
        --method "BundleSDF=baselines_real/results_bsdf/gt"

Each --method is "Label=<dir with <seq>.npy>"; with none given, the default
baseline set + ours is used. Output: eval/gifs_lift/<seq>.gif
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_eval import _EST_COLORS, _GT_COLORS, _draw_axes, rotation_angle_deg
from run_eval_lift import (
    DEFAULT_DATASET_ROOT,
    central_view_index,
    frame_ids,
    load_gt_poses,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_METHODS = [
    ("ICP", "baselines_real/results_icp/gt"),
    ("LoFTR", "baselines_real/results_loftr/gt"),
    ("PnP", "baselines_real/results_pnp/gt"),
    ("FoundationPose", "baselines_real/results_fp/gt"),
    ("BundleSDF", "baselines_real/results_bsdf/gt"),
    ("Ours", "baselines_real/tuned_v3/lf"),
]


def _font(size: int):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def make_sequence_gif(
    seq: str,
    methods: list,
    dataset_root: str,
    out_path: str,
    scale: float,
    cols: int,
    duration_ms: int,
    axis_len: float,
):
    seq_dir = os.path.join(dataset_root, seq)
    gt_poses = load_gt_poses(seq_dir)
    K = np.loadtxt(os.path.join(seq_dir, "camera_matrix.txt"))
    cv = central_view_index(seq_dir)
    fids = frame_ids(seq_dir)

    est = {}
    for label, path in methods:
        npy = os.path.join(path, f"{seq}.npy")
        if os.path.exists(npy):
            est[label] = np.load(npy)
    if not est:
        print(f"  [skip] {seq}: no method has results")
        return

    n = min(len(gt_poses), *(len(p) for p in est.values()))

    # Crop every panel to the union of GT-mask bboxes (padded) so the object
    # and axes dominate the frame instead of the mostly-static background.
    x0, y0, x1, y1 = 1e9, 1e9, 0, 0
    for i in range(n):
        m = np.array(
            Image.open(os.path.join(seq_dir, f"LF_{fids[i]}", "masks", f"{cv:04d}.png"))
        )
        ys, xs = np.where(m > 0)
        if len(xs):
            x0, y0 = min(x0, xs.min()), min(y0, ys.min())
            x1, y1 = max(x1, xs.max()), max(y1, ys.max())
    W_full, H_full = Image.open(
        os.path.join(seq_dir, f"LF_{fids[0]}", f"{cv:04d}.png")
    ).size
    pad = 0.35 * max(x1 - x0, y1 - y0)
    crop = (
        int(max(0, x0 - pad)),
        int(max(0, y0 - pad)),
        int(min(W_full, x1 + pad)),
        int(min(H_full, y1 + pad)),
    )

    rows = int(np.ceil(len(methods) / cols))
    font = _font(30)
    frames_out = []
    for i in range(n):
        base = (
            Image.open(os.path.join(seq_dir, f"LF_{fids[i]}", f"{cv:04d}.png"))
            .convert("RGB")
        )
        panels = []
        for label, _ in methods:
            panel = base.copy()
            draw = ImageDraw.Draw(panel)
            _draw_axes(draw, K, gt_poses[i], _GT_COLORS, axis_len, width=3, dashed=True)
            caption = label
            if label in est:
                p = est[label][i]
                _draw_axes(draw, K, p, _EST_COLORS, axis_len, width=4, dashed=False)
                rot = rotation_angle_deg(
                    (p[:3, :3] @ gt_poses[i][:3, :3].T)[None]
                )[0]
                caption = f"{label}  {rot:.1f}\N{DEGREE SIGN}"
            panel = panel.crop(crop)
            draw = ImageDraw.Draw(panel)
            tw = draw.textlength(caption, font=font)
            draw.rectangle([0, 0, tw + 20, 42], fill=(0, 0, 0))
            draw.text((10, 4), caption, fill=(255, 255, 255), font=font)
            panels.append(panel)

        W, H = panels[0].size
        grid = Image.new("RGB", (W * cols, H * rows), (20, 20, 20))
        for j, panel in enumerate(panels):
            grid.paste(panel, ((j % cols) * W, (j // cols) * H))
        if scale != 1.0:
            grid = grid.resize((int(grid.width * scale), int(grid.height * scale)))
        frames_out.append(grid)

    frames_out[0].save(
        out_path,
        save_all=True,
        append_images=frames_out[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"  [done] {out_path}  ({n} frames, {len(est)} methods)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--method",
        action="append",
        default=[],
        metavar="LABEL=DIR",
        help="repeatable; DIR holds <seq>.npy (relative to repo root ok). "
        "Default: icp/loftr/pnp/fp/bsdf baselines + ours (full, GT mask).",
    )
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--out-dir", default=os.path.join(REPO, "eval", "gifs_lift"))
    ap.add_argument("--seqs", default=None, help="comma-separated substrings")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--duration", type=int, default=400, help="ms per frame")
    ap.add_argument("--axis-len", type=float, default=0.05)
    args = ap.parse_args()

    if args.method:
        methods = []
        for m in args.method:
            label, _, path = m.partition("=")
            if not path:
                raise SystemExit(f"--method expects LABEL=DIR, got {m}")
            methods.append((label, path))
    else:
        methods = list(DEFAULT_METHODS)
    methods = [
        (l, p if os.path.isabs(p) else os.path.join(REPO, p)) for l, p in methods
    ]

    seqs = [
        d
        for d in sorted(os.listdir(args.dataset_root))
        if d.endswith("_prod")
        and not d.startswith("car_")
        and os.path.isdir(os.path.join(args.dataset_root, d))
    ]
    if args.seqs:
        keys = [k for k in args.seqs.split(",") if k]
        seqs = [s for s in seqs if any(k in s for k in keys)]

    os.makedirs(args.out_dir, exist_ok=True)
    for seq in seqs:
        make_sequence_gif(
            seq,
            methods,
            args.dataset_root,
            os.path.join(args.out_dir, f"{seq}.gif"),
            args.scale,
            args.cols,
            args.duration,
            args.axis_len,
        )


if __name__ == "__main__":
    main()
